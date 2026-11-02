"""Session 10 review of PR #15 (commit f2de5e6): regressions for the confirmed findings.

Synthetic data only. Windows code paths are exercised on every platform by selecting the
Windows branch (``db._is_windows``) and feeding security descriptors in SDDL; the real
Windows calls (``GetNamedSecurityInfoW``, ``icacls``, ``whoami``) run only on Windows
(``test_reviewer_auth.py::test_sqlite_directory_created_owner_only`` and the PowerShell
controls below). Tests marked "reviewer" keep the independent reviewer's oracle.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import src.state.db as db
from src.core.exceptions import SysRevError
from src.state.acl import owner_only_problems, parse_sddl

USER = "S-1-5-21-1111111111-2222222222-3333333333-1001"
OTHER = "S-1-5-21-1111111111-2222222222-3333333333-1002"
# What Windows produced on the reviewer's machine (Python 3.12.10 mkdir(0o700) gives
# SYSTEM, Administrators and OWNER RIGHTS; icacls then grants the current user): the four
# explicit entries the count-based check rejected.
OBSERVED = (
    f"O:{USER}G:{USER[:-4]}513"
    f"D:P(A;OICI;FA;;;{USER})(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)(A;OICI;FA;;;OW)"
)


# =============================================================== P2: structured ACL check
@pytest.mark.parametrize(
    ("dacl", "owner", "expected"),
    [
        # accepted: current user (+ privileged principals of the threat model)
        (f"D:P(A;OICI;FA;;;{USER})", USER, []),
        (f"D:PAI(A;OICI;FA;;;{USER})(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)", USER, []),
        ("D:P(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)(A;OICI;FA;;;OW)", USER, []),  # via OWNER RIGHTS
        (f"D:P(A;OICI;0x1f01ff;;;{USER})", "BA", []),  # elevated: owner Administrators
        (f"D:P(A;OICI;FA;;;{USER})(D;OICI;FA;;;WD)", USER, []),  # a deny entry only restricts
        ("D:P(A;OICI;FA;;;LA)", "LA", None),  # domain-relative alias of another account
    ],
)
def test_owner_only_accepts_the_defined_principals(
    dacl: str, owner: str, expected: list[str] | None
) -> None:
    problems = owner_only_problems(f"O:{owner}{dacl}", USER)
    if expected is None:
        assert problems  # LA is RID 500 of the domain, not the current user (RID 1001)
    else:
        assert problems == expected


def test_the_observed_windows_acl_is_owner_only() -> None:
    assert owner_only_problems(OBSERVED, USER) == []


@pytest.mark.parametrize(
    ("dacl", "owner", "fragment"),
    [
        ("D:P(A;OICI;FA;;;WD)", USER, "S-1-1-0"),  # reviewer: Everyone full control
        (f"D:P(A;OICI;FA;;;{USER})(A;OICI;FA;;;WD)", USER, "S-1-1-0"),
        (f"D:P(A;OICI;FA;;;{USER})(A;OICI;0x1301bf;;;AU)", USER, "S-1-5-11"),  # Auth. Users
        (f"D:P(A;OICI;FA;;;{USER})(A;OICI;FR;;;BU)", USER, "S-1-5-32-545"),  # Users: read
        (f"D:P(A;OICI;FA;;;{USER})(A;OICI;FR;;;BG)", USER, "S-1-5-32-546"),  # Guests
        (f"D:P(A;OICI;FA;;;{USER})(A;OICI;FA;;;{OTHER})", USER, OTHER),  # another account
        (f"D:P(A;OICI;FA;;;{USER})(A;OICI;FA;;;XY)", USER, "?XY"),  # unknown alias
        (f"D:P(A;OICI;FA;;;{USER})(OA;;CR;guid;;WD)", USER, "'OA'"),  # object ACE
        (f"D:AI(A;OICIID;FA;;;{USER})", USER, "inherited"),  # not protected
        ("D:P(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)(A;OICI;FA;;;OW)", OTHER, "owner"),  # OW -> other
        ("D:P(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)", USER, "no full control"),  # user locked out
        (f"D:P(A;OICI;FR;;;{USER})", USER, "no full control"),
        ("D:NO_ACCESS_CONTROL", USER, "no DACL"),  # null DACL: everyone
        ("", USER, "no DACL"),
    ],
)
def test_owner_only_rejects_other_access(dacl: str, owner: str, fragment: str) -> None:
    problems = owner_only_problems(f"O:{owner}{dacl}", USER)
    assert problems and any(fragment in p for p in problems), problems


def test_the_decision_uses_sids_not_names_or_entry_counts() -> None:
    sd = parse_sddl(OBSERVED, USER)
    assert len(sd.aces) == 4 and sd.dacl_protected and sd.owner == USER
    assert {a.sid for a in sd.aces} == {USER, "S-1-5-18", "S-1-5-32-544", "S-1-3-4"}
    # a single entry is not enough by itself
    assert owner_only_problems("O:WDD:P(A;OICI;FA;;;WD)", USER)


def test_preflight_does_not_call_everyone_full_control_owner_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reviewer: an ACL granting Everyone full control was reported ``owner_only=True``."""
    from validation.session10 import preflight

    monkeypatch.setattr(db, "_is_windows", lambda: True)
    monkeypatch.setattr(
        db,
        "windows_directory_problems",
        lambda d: owner_only_problems("O:WDD:P(A;OICI;FA;;;WD)", USER),
    )
    result = preflight.directory_protection(tmp_path)
    assert not result["owner_only"] and result["problems"], result
    monkeypatch.setattr(
        db, "windows_directory_problems", lambda d: owner_only_problems(OBSERVED, USER)
    )
    assert preflight.directory_protection(tmp_path) == {"owner_only": True, "problems": []}


# =============================================================== P1: fail closed on retry
class _Windows:
    """The Windows branch of prepare_sqlite_path with scripted ACL results."""

    def __init__(
        self, monkeypatch: pytest.MonkeyPatch, *, apply_ok: bool = True, sddl: str = OBSERVED
    ) -> None:
        self.applied: list[Path] = []
        self.verified: list[Path] = []
        self.apply_ok = apply_ok
        self.sddl = sddl
        monkeypatch.setattr(db, "_is_windows", lambda: True)
        monkeypatch.setattr(db, "restrict_directory_to_current_user", self.apply)
        monkeypatch.setattr(db, "windows_directory_problems", self.verify)

    def apply(self, d: Path) -> None:
        self.applied.append(d)
        if not self.apply_ok:
            raise SysRevError("synthetic permission failure")

    def verify(self, d: Path) -> list[str]:
        self.verified.append(d)
        return owner_only_problems(self.sddl, USER)


def test_failed_acl_setup_must_not_be_bypassed_by_restarting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reviewer: the second call returned without any protection."""
    win = _Windows(monkeypatch, apply_ok=False)
    directory = tmp_path / "new database folder" / "db"
    url = f"sqlite:///{directory}/sysrev.db"
    with pytest.raises(SysRevError):
        db.prepare_sqlite_path(url)
    with pytest.raises(SysRevError):  # the same normal startup again
        db.prepare_sqlite_path(url)
    assert win.applied == [directory, directory]
    # the directories this initialisation created were removed; the existing parent stays
    assert not (tmp_path / "new database folder").exists() and tmp_path.is_dir()


def test_failed_acl_then_normal_cli_restart_must_not_create_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reviewer: the reviewer CLI created the database on its second invocation."""
    from src.api_server.reviewer_cli import main

    _Windows(monkeypatch, apply_ok=False)
    database = tmp_path / "new protected database" / "sysrev.db"
    url = f"sqlite:///{database}"
    with pytest.raises(SysRevError):
        main(["list"], database_url=url, out=lambda _: None)
    assert not database.exists()
    with pytest.raises(SysRevError):
        main(["list"], database_url=url, out=lambda _: None)
    assert not database.exists(), "the retry created the database after ACL setup failed"


def test_applied_but_unverified_protection_also_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """icacls succeeded but the result is not owner-only (e.g. an explicit Everyone entry)."""
    win = _Windows(monkeypatch, sddl=f"O:{USER}D:P(A;OICI;FA;;;{USER})(A;OICI;FA;;;WD)")
    directory = tmp_path / "a" / "b"
    for _ in range(2):
        with pytest.raises(SysRevError, match="not restricted to the current user"):
            db.prepare_sqlite_path(f"sqlite:///{directory}/sysrev.db")
        assert not (tmp_path / "a").exists()
    assert win.verified == [directory, directory]


def test_a_leftover_empty_directory_is_protected_and_verified_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If a failed initialisation could not even remove its directory, the next start
    treats the empty directory as uninitialised: protect, verify, only then continue."""
    leftover = tmp_path / "db"
    leftover.mkdir()
    win = _Windows(monkeypatch, apply_ok=False)
    with pytest.raises(SysRevError):
        db.prepare_sqlite_path(f"sqlite:///{leftover}/sysrev.db")
    assert leftover.is_dir()  # not created by this call: never removed
    win.apply_ok = True
    db.prepare_sqlite_path(f"sqlite:///{leftover}/sysrev.db")
    assert win.applied == [leftover, leftover] and win.verified == [leftover]


def test_successful_initialisation_creates_only_after_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    win = _Windows(monkeypatch)
    directory = tmp_path / "Veri klasörü ş" / "db"
    db.prepare_sqlite_path(f"sqlite:///{directory}/sysrev.db")
    assert directory.is_dir() and win.applied == [directory] and win.verified == [directory]


def test_an_unrelated_existing_directory_is_never_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-empty existing folder without the database is verified, not modified; if it
    is shared, startup is refused and its contents stay."""
    folder = tmp_path / "Documents"
    folder.mkdir()
    (folder / "notes.txt").write_text("user's own file", encoding="utf-8")
    win = _Windows(monkeypatch, sddl=f"O:{USER}D:AI(A;OICIID;FA;;;{USER})(A;OICIID;FR;;;BU)")
    with pytest.raises(SysRevError, match="not changed automatically"):
        db.prepare_sqlite_path(f"sqlite:///{folder}/sysrev.db")
    assert win.applied == [] and (folder / "notes.txt").read_text(encoding="utf-8")
    assert not (folder / "sysrev.db").exists()
    # an owner-only existing folder is accepted as it is
    win.sddl = OBSERVED
    db.prepare_sqlite_path(f"sqlite:///{folder}/sysrev.db")
    assert win.applied == []


def test_an_existing_database_is_opened_as_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "sysrev.db").write_bytes(b"")
    win = _Windows(monkeypatch, sddl="O:WDD:P(A;OICI;FA;;;WD)")
    db.prepare_sqlite_path(f"sqlite:///{tmp_path}/sysrev.db")  # legacy: reported by preflight
    assert win.applied == [] and win.verified == []


def test_posix_behaviour_is_unchanged(tmp_path: Path) -> None:
    import os
    import stat

    if os.name == "nt":
        pytest.skip("POSIX modes")
    target = tmp_path / "x" / "db"
    db.prepare_sqlite_path(f"sqlite:///{target}/sysrev.db")
    assert stat.S_IMODE(target.stat().st_mode) == 0o700
    assert db.directory_protection_problems(target) == []
    target.chmod(0o755)
    assert db.directory_protection_problems(target)


# =============================================================== P2: statistics checker
def _stats_dir(tmp_path: Path, analyses: list[dict[str, object]]) -> tuple[Path, str]:
    import hashlib
    import json

    d = tmp_path / "stats"
    d.mkdir()
    body = json.dumps({"engine_version": "synthetic", "analyses": analyses})
    (d / "stats_output.json").write_text(body, encoding="utf-8")
    return d, hashlib.sha256(body.encode()).hexdigest()


def _run_checker(
    monkeypatch: pytest.MonkeyPatch, stats_dir: Path, out: Path, reference: dict[str, object]
) -> tuple[int, dict[str, object]]:
    """R is replaced by ``reference`` to isolate the integrity/verdict logic; these tests
    make no numerical claim (the live R comparisons are in test_session10_tooling.py)."""
    import json
    import subprocess

    from validation.session10 import check_pilot_stats as checker

    def fake_r(cmd: list[str], **kw: object) -> subprocess.CompletedProcess[str]:
        Path(cmd[-1]).write_text(json.dumps(reference), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(checker.subprocess, "run", fake_r)
    code = checker.main([str(stats_dir), "--out", str(out), "--rscript", "synthetic-r"])
    return code, json.loads((out / "pilot_stats_check.json").read_text(encoding="utf-8"))


def _matching_reference(analysis: dict[str, object]) -> dict[str, object]:
    """A stubbed 'R' result equal to the engine's own numbers, so every comparison passes
    and only the manifest binding decides."""
    from validation.session10 import check_pilot_stats as checker

    ref: dict[str, object] = {"analysis_id": analysis["analysis_id"]}
    models = [
        analysis.get("common_effect"),
        analysis.get("random_effects"),
        *(analysis.get("sensitivity") or []),
    ]  # type: ignore[list-item]
    for p in models:
        if p is None:
            continue
        key = checker.reference_key(analysis, p)  # type: ignore[arg-type]
        assert key is not None
        ref[key[0]] = p
    return {"software": {"R": "stub: numeric calculation is not under test"}, "analyses": [ref]}


def _rr_analysis() -> dict[str, object]:
    from tests.unit.test_session9_review_followup import REVIEW_TABLES, _ratio

    return _ratio(REVIEW_TABLES, "RR").model_dump(mode="json")  # type: ignore[no-any-return]


@pytest.mark.parametrize(
    ("make", "fragment"),
    [
        # reviewer: the digest only in an unrelated note
        (lambda d: {"stats_output_sha256": "0" * 64, "unrelated_note": d}, "does not match"),
        # reviewer (stronger): both bindings wrong, the digest elsewhere
        (
            lambda d: {
                "stats_output_sha256": "0" * 64,
                "files": {"stats_output.json": "0" * 64},
                "unrelated_note": d,
            },
            "does not match",
        ),
        # contradictory bindings
        (
            lambda d: {"stats_output_sha256": d, "files": {"stats_output.json": "f" * 64}},
            'files["stats_output.json"]',
        ),
        (
            lambda d: {"stats_output_sha256": "e" * 64, "files": {"stats_output.json": d}},
            "stats_output_sha256 = e",
        ),
        # missing / malformed bindings
        (lambda d: {"files": {"stats_output.json": d}}, "stats_output_sha256 is missing"),
        (lambda d: {"stats_output_sha256": d}, 'files["stats_output.json"] is missing'),
        (
            lambda d: {"stats_output_sha256": d.upper(), "files": {"stats_output.json": d}},
            "not a SHA-256",
        ),
        (lambda d: {"stats_output_sha256": d, "files": {"other.json": d}}, "is missing"),
        (lambda d: [d], "not a JSON object"),
    ],
)
def test_wrong_manifest_is_rejected_even_when_all_numeric_comparisons_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make: object, fragment: str
) -> None:
    import json

    a = _rr_analysis()
    d, digest = _stats_dir(tmp_path, [a])
    (d / "manifest.json").write_text(json.dumps(make(digest)), encoding="utf-8")  # type: ignore[operator]
    code, result = _run_checker(monkeypatch, d, tmp_path / "out", _matching_reference(a))
    assert result["comparisons"] and not result["failures"] and not result["not_compared"]  # type: ignore[truthy-bool]
    assert code == 1 and result["verdict"] == "failed" and not result["listed_in_manifest"]
    assert any(fragment in p for p in result["manifest_problems"]), result["manifest_problems"]  # type: ignore[attr-defined]


def test_an_unreadable_manifest_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    a = _rr_analysis()
    d, _ = _stats_dir(tmp_path, [a])
    (d / "manifest.json").write_text("{not json", encoding="utf-8")
    code, result = _run_checker(monkeypatch, d, tmp_path / "out", _matching_reference(a))
    assert code == 1 and result["verdict"] == "failed"


def test_a_real_stage5_artifact_is_the_positive_control(
    session: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The manifest written by run_stage5 in a guarded workflow binds its output."""
    import json

    from tests.unit.test_session9_review_followup import _workflow

    _workflow(session, "common_effect", tmp_path)  # type: ignore[arg-type]
    d = tmp_path / "stats"
    stats = json.loads((d / "stats_output.json").read_text(encoding="utf-8"))
    ref = {
        "software": {"R": "stub"},
        "analyses": [
            _matching_reference(a)["analyses"][0]  # type: ignore[index]
            for a in stats["analyses"]
            if a["status"] == "pooled"
        ],
    }
    code, result = _run_checker(monkeypatch, d, tmp_path / "out", ref)
    assert result["listed_in_manifest"] and result["manifest_problems"] == []
    assert (code, result["verdict"]) == (0, "validated"), result


def test_checker_must_not_succeed_without_any_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reviewer: an unsupported HR output (0 comparisons, 1 not compared) exited 0."""
    import json

    from src.core.schemas.stats import AnalysisSpec
    from src.stats_engine.engine import run_analysis
    from tests.stats_helpers import OUTCOMES, inputs

    a = run_analysis(
        AnalysisSpec(analysis_id="hr", outcome_id="survival", effect_measure="HR"),
        OUTCOMES["survival"],
        inputs([]),
    )
    assert a.status == "unsupported"
    d, digest = _stats_dir(tmp_path, [a.model_dump(mode="json")])
    manifest = {"stats_output_sha256": digest, "files": {"stats_output.json": digest}}
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    code, result = _run_checker(monkeypatch, d, tmp_path / "out", {"software": {}, "analyses": []})
    assert result["comparisons"] == 0 and result["not_compared"] == 1
    assert (code, result["verdict"]) == (3, "inconclusive")
    assert result["validated_analyses"] == [] and result["not_validated_analyses"] == ["hr"]


def test_a_partly_compared_output_is_inconclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One analysis reproduced, another not compared: completed, but not validated."""
    import json

    from src.core.schemas.stats import AnalysisSpec
    from src.stats_engine.engine import run_analysis
    from tests.stats_helpers import OUTCOMES, inputs

    rr = _rr_analysis()
    hr = run_analysis(
        AnalysisSpec(analysis_id="hr", outcome_id="survival", effect_measure="HR"),
        OUTCOMES["survival"],
        inputs([]),
    ).model_dump(mode="json")
    d, digest = _stats_dir(tmp_path, [rr, hr])
    manifest = {"stats_output_sha256": digest, "files": {"stats_output.json": digest}}
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    code, result = _run_checker(monkeypatch, d, tmp_path / "out", _matching_reference(rr))
    assert (code, result["verdict"]) == (3, "inconclusive")
    assert result["validated_analyses"] == [rr["analysis_id"]]


def test_an_r_failure_is_an_error_not_a_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    import subprocess

    from validation.session10 import check_pilot_stats as checker

    a = _rr_analysis()
    d, digest = _stats_dir(tmp_path, [a])
    manifest = {"stats_output_sha256": digest, "files": {"stats_output.json": digest}}
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        checker.subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "no metafor"),
    )
    out = tmp_path / "out"
    assert checker.main([str(d), "--out", str(out), "--rscript", "x"]) == 2
    result = json.loads((out / "pilot_stats_check.json").read_text(encoding="utf-8"))
    assert result["verdict"] == "error"


# =============================================================== preflight process tree
def test_preflight_stops_the_launchers_children_and_nothing_else(tmp_path: Path) -> None:
    """``python -m src.ui`` starts Streamlit as a child; stopping only the launcher left the
    UI listening on Windows. Here a launcher starts a child HTTP server the same way
    (``subprocess.call``); the whole tree must stop, an unrelated process must not."""
    import subprocess
    import sys
    import time

    from validation.session10 import preflight

    port = preflight.free_port()
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    launcher = (
        "import subprocess, sys; subprocess.call([sys.executable, '-m', 'http.server', "
        f"'{port}', '--bind', '127.0.0.1'])"
    )
    try:
        with (tmp_path / "log.txt").open("ab") as fh:
            proc = preflight.popen_tree(
                [sys.executable, "-c", launcher], dict(__import__("os").environ), tmp_path, fh
            )
            deadline = time.monotonic() + 30
            while not preflight.port_open(port) and time.monotonic() < deadline:
                time.sleep(0.2)
            assert preflight.port_open(port), "child server did not start"
            preflight.stop_tree(proc, hard=False)
        deadline = time.monotonic() + 15
        while preflight.port_open(port) and time.monotonic() < deadline:
            time.sleep(0.2)
        assert not preflight.port_open(port), "the launcher's child outlived the stop"
        assert proc.poll() is not None
        assert unrelated.poll() is None, "an unrelated process was stopped"
    finally:
        unrelated.kill()
        unrelated.wait(timeout=10)


# =============================================================== preflight network
class _Resp:
    def __init__(self, status: int, text: str = "") -> None:
        self.status_code = status
        self.text = text


@pytest.mark.parametrize(
    ("name", "statuses", "expected", "tries"),
    [
        ("europe_pmc", [200], "pass", 1),
        ("europe_pmc", [400, 200], "warn", 2),  # what the Windows review saw: transient
        ("europe_pmc", [400, 400], "FAIL", 2),
        ("openai_models_no_key", [401], "pass", 1),  # reachable without a key, nothing billed
        ("openai_models_no_key", [200], "FAIL", 2),  # a 200 without a key is unexpected
    ],
)
def test_network_check_records_every_attempt(
    monkeypatch: pytest.MonkeyPatch, name: str, statuses: list[int], expected: str, tries: int
) -> None:
    import httpx

    from validation.session10 import preflight

    answers = iter(statuses + [statuses[-1]] * 3)
    monkeypatch.setattr(
        httpx, "get", lambda url, **kw: _Resp(next(answers), '{"errMsg": "synthetic"}')
    )
    monkeypatch.setattr(preflight.time, "sleep", lambda s: None)
    result = preflight.network_check(name, "https://example.invalid/x", {})
    assert result["_status"] == expected and len(result["attempts"]) == tries
    unexpected = [
        a for a in result["attempts"] if a["http_status"] != statuses[-1] or expected == "FAIL"
    ]
    assert all("body_start" in a for a in unexpected if a["http_status"] not in (200, 401))


# =============================================================== heterogeneity population
@pytest.mark.parametrize("primary", ["random_effects", "common_effect"])
def test_heterogeneity_names_its_population_when_it_differs(
    session: object, tmp_path: Path, primary: str
) -> None:
    """With the Mantel-Haenszel primary model (k = 3, incl. a double-all-event table) Q/I²
    come from the 2 inverse-variance studies: the paragraph must say so. Numbers, facts and
    the evidence fingerprint are unchanged; with the IV primary the text is unchanged."""
    from src.manuscript.document import render
    from src.manuscript.evidence import build_evidence_package
    from src.manuscript.service import ManuscriptService
    from src.reporting.service import ReportingService
    from tests.manuscript_helpers import SYSTEM
    from tests.unit.test_session9_review_followup import _workflow

    repo, _, _ = _workflow(session, primary, tmp_path)  # type: ignore[arg-type]
    package = build_evidence_package(repo, "mc1")
    het = package.facts["stats.comp.heterogeneity"]
    assert het.text == "Q = 0.68 (df = 1; p = 0.411); I² = 0%"  # fact unchanged
    svc = ManuscriptService(repo, "mc1")
    version = svc.create_template_draft(SYSTEM)
    text = render(version.sections, package).markdown
    labelled = (
        "Heterogeneity (computed with inverse-variance weights from the 2 studies with a "
        "study-level estimate, not from the population of the primary model): Q = 0.68"
    )
    if primary == "common_effect":
        assert labelled in text and "3 studies contributed" in text
    else:
        assert "Heterogeneity: Q = 0.68" in text and labelled not in text
    report = svc.audit()
    assert [f for f in report.findings if "Heterogeneity" in f.excerpt] == []
    ds = next(r for r in ReportingService(repo, "mc1").grade() if r["analysis_id"] == "comp")
    population = ds["decision_support"]["domains"]["inconsistency"]["population"]
    if primary == "common_effect":
        assert (
            population.startswith("the 2 studies with a study-level estimate") and "3" in population
        )
    else:
        assert population == "the studies of the rated (primary) estimate"
