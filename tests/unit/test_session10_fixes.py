"""Session 10 regressions: defects confirmed while preparing the real-world Windows pilot.

Synthetic data only. These tests are not evidence about any real review."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from tests.unit.test_session9_review_followup import _workflow

ROOT = Path(__file__).resolve().parents[2]


# ======================================================================= sensitivity sentence
# Before Session 10 the results paragraph selected every fact whose key began with
# ``stats.<analysis>.sensitivity.``, which includes the sensitivity model's own population
# counts (``.k``, ``.participants``, ``.events``). With the Mantel-Haenszel primary model of
# the three-trial fixture the draft read "Sensitivity analysis: 2; 400; 70; 1.35 (...)".


def _sensitivity_sentence(text: str) -> str:
    m = re.search(r"Sensitivity analysis: [^\n]*?\.(?=\s|$)(?!\d)", text)
    assert m is not None, "no sensitivity sentence"
    return m.group(0)


@pytest.mark.parametrize("primary", ["random_effects", "common_effect"])
def test_sensitivity_sentence_selects_estimates_and_labels_counts(
    session: Session, tmp_path: Path, primary: str
) -> None:
    from src.manuscript.document import render
    from src.manuscript.evidence import build_evidence_package
    from src.manuscript.service import ManuscriptService
    from tests.manuscript_helpers import SYSTEM

    repo, _, _ = _workflow(session, primary, tmp_path)
    package = build_evidence_package(repo, "mc1")
    f = package.facts
    est = f["stats.comp.sensitivity.1"]
    assert est.role == "sensitivity_estimate"
    assert est.text == "1.35 (95% CI 0.88 to 2.06); common-effect model (inverse variance)"
    svc = ManuscriptService(repo, "mc1")
    version = svc.create_template_draft(SYSTEM)
    results = next(s for s in version.sections if s.section_id == "results").source
    sentence = _sensitivity_sentence(render(version.sections, package).markdown)
    # the unlabelled count list of the defect never appears
    assert "Sensitivity analysis: 2; 400; 70" not in sentence
    assert not re.search(r"Sensitivity analysis: \d+;", sentence)
    if primary == "common_effect":
        # the inverse-variance sensitivity model uses 2 of the 3 studies (Session 9 review):
        # its own population is stated, with units, next to its estimate
        assert sentence == (
            "Sensitivity analysis: 1.35 (95% CI 0.88 to 2.06); common-effect model "
            "(inverse variance) (population of this model: 2 studies, 400 participants, "
            "70 events)."
        )
        assert (f["stats.comp.sensitivity.1.k"].role, f["stats.comp.sensitivity.1.k"].text) == (
            "count",
            "2",
        )
        assert "{{stats.comp.sensitivity.1.k}} studies" in results
    else:
        # same population as the primary model: no model-specific counts exist or appear
        assert "stats.comp.sensitivity.1.k" not in f
        assert sentence == (
            "Sensitivity analysis: 1.35 (95% CI 0.88 to 2.06); common-effect model "
            "(inverse variance)."
        )
    # the validated estimate is unchanged
    assert est.ci == pytest.approx((0.88, 2.06), abs=0.005)
    # the Auditor recognises the sentence as template text and finds nothing wrong with it
    report = svc.audit()
    near = [x for x in report.findings if "Sensitivity analysis" in x.excerpt]
    assert near == [], [(x.scope, x.status, x.message) for x in near]


def test_sensitivity_clauses_ignore_non_estimate_facts_with_the_prefix() -> None:
    """Selection is by role: a count or any other fact under the prefix is never listed as an
    estimate; counts appear only with their unit, attached to their own model's estimate."""
    from types import SimpleNamespace

    from src.manuscript.drafting import _sensitivity_clauses
    from src.manuscript.evidence import Fact

    def fact(key: str, role: str, text: str) -> Fact:
        return Fact(key=key, text=text, description="d", source="s", category="statistics",
                    role=role)  # fmt: skip

    facts = {
        x.key: x
        for x in (
            fact("stats.a.sensitivity.2", "sensitivity_estimate", "0.5; model B"),
            fact("stats.a.sensitivity.1.k", "count", "2"),
            fact("stats.a.sensitivity.1.events", "count", "70"),
            fact("stats.a.sensitivity.1", "sensitivity_estimate", "1.35; model A"),
            fact("stats.a.sensitivity.10", "sensitivity_estimate", "0.9; model J"),
            fact("stats.a.sensitivity.1.note", "text", "free text"),
            fact("stats.ab.sensitivity.1", "sensitivity_estimate", "other analysis"),
        )
    }
    clauses = _sensitivity_clauses(SimpleNamespace(facts=facts), "stats.a")  # type: ignore[arg-type]
    assert clauses == [
        "{{stats.a.sensitivity.1}} (population of this model: "
        "{{stats.a.sensitivity.1.k}} studies, {{stats.a.sensitivity.1.events}} events)",
        "{{stats.a.sensitivity.2}}",
        "{{stats.a.sensitivity.10}}",
    ]


# ======================================================================= text encoding
# On Windows, ``open()``/``Path.read_text()`` without an encoding use the ANSI code page
# (e.g. cp1252 or cp1254), not UTF-8. Every file this project writes is UTF-8. Two tests
# failed on Windows until ``PYTHONUTF8=1`` was set; the fix is an explicit encoding at every
# text I/O call, which this test enforces for src, tests and validation.

TEXT_IO = {"read_text", "write_text", "open"}


def _locale_dependent_calls(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name not in TEXT_IO:
            continue
        kwargs = {k.arg for k in node.keywords}
        if "encoding" in kwargs:
            continue
        if name == "read_text" and node.args:  # Path.read_text(encoding) positional
            continue
        if name == "write_text" and len(node.args) >= 2:  # write_text(data, encoding)
            continue
        if name == "open":
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and (func.value.id in {"pdfplumber", "webbrowser", "Image"})
            ):
                continue
            # Path.open(mode), open(path, mode), gzip.open(path, mode): binary if any
            # candidate mode string (positional or ``mode=``) contains "b"
            modes = [
                a.value
                for a in [*node.args[:2], *(k.value for k in node.keywords if k.arg == "mode")]
                if isinstance(a, ast.Constant)
                and isinstance(a.value, str)
                and re.fullmatch(r"[rwxabt+]+", a.value)
            ]
            if any("b" in m for m in modes):
                continue
        shown = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name
        found.append(f"{shown}:{node.lineno}: {name}()")
    return found


def test_no_text_io_depends_on_the_locale_encoding() -> None:
    files = [p for d in ("src", "tests", "validation") for p in (ROOT / d).rglob("*.py")]
    assert len(files) > 50
    offending = [x for p in sorted(files) for x in _locale_dependent_calls(p)]
    assert offending == [], "text I/O without an explicit encoding:\n" + "\n".join(offending)


def test_encoding_scan_detects_the_defect(tmp_path: Path) -> None:
    """The scan itself must catch the pattern that failed on Windows."""
    sample = tmp_path / "sample.py"
    sample.write_text(
        "from pathlib import Path\n"
        "Path('x').read_text()\n"
        "Path('x').write_text('é')\n"
        "open('x')\n"
        "Path('x').open('w')\n"
        "Path('x').read_text('utf-8')\n"
        "Path('x').write_bytes(b'')\n"
        "open('x', 'rb')\n",
        encoding="utf-8",
    )
    found = _locale_dependent_calls(sample)
    assert [x.rsplit(": ", 1)[1] for x in found] == [
        "read_text()", "write_text()", "open()", "open()",
    ]  # fmt: skip


# ======================================================================= owner-only database
# ``mkdir(mode=0o700)`` has no effect on Windows: the directory that holds the reviewer
# credential hashes inherited its parent's ACL. The Unix assertion of
# ``test_sqlite_directory_created_owner_only`` failed there and the protection it describes
# did not exist. Session 10 applies an explicit owner-only ACL on Windows.


def windows_acl_entries(directory: Path) -> list[str]:
    """Access-control entries of ``directory`` as printed by ``icacls`` (Windows only)."""
    import subprocess

    from src.state.db import _windows_tool

    # run from the parent with the bare name, so that the echoed path is ASCII whatever the
    # console code page (only the SID-independent flags are inspected)
    name = directory.name
    assert name.isascii()
    out = subprocess.run(
        [_windows_tool("icacls.exe"), name],
        cwd=directory.parent,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    ).stdout
    lines = [x.strip() for x in out.splitlines()]
    first = lines[0]
    assert first.lower().startswith(name.lower()), out
    entries = [first[len(name) :].strip()]
    for x in lines[1:]:
        if not x or x.lower().startswith("successfully"):
            break
        entries.append(x)
    return [e for e in entries if e]


class _Runner:
    def __init__(self, icacls_returncode: int = 0) -> None:
        self.calls: list[list[str]] = []
        self.icacls_returncode = icacls_returncode

    def __call__(self, cmd: list[str], **kw: object):  # type: ignore[no-untyped-def]
        import subprocess

        self.calls.append(cmd)
        assert kw.get("encoding") == "utf-8" and kw.get("capture_output") is True
        if cmd[0].lower().endswith("whoami.exe"):
            return subprocess.CompletedProcess(
                cmd, 0, '"desktop\\\\dr ışık","S-1-5-21-111-222-333-1001"\n', ""
            )
        return subprocess.CompletedProcess(cmd, self.icacls_returncode, "", "Access is denied.")


def test_windows_acl_command_grants_only_the_current_user(tmp_path: Path) -> None:
    from src.state.db import restrict_directory_to_current_user

    run = _Runner()
    restrict_directory_to_current_user(tmp_path / "Veri klasörü ş", run=run)
    whoami, icacls = run.calls
    assert whoami[0].lower().endswith(("system32\\whoami.exe", "system32/whoami.exe"))
    assert icacls[0].lower().endswith(("system32\\icacls.exe", "system32/icacls.exe"))
    # one argument per item: a path with spaces/non-ASCII needs no quoting
    assert icacls[1:] == [
        str(tmp_path / "Veri klasörü ş"),
        "/inheritance:r",
        "/grant:r",
        "*S-1-5-21-111-222-333-1001:(OI)(CI)F",
    ]


def test_windows_acl_failure_fails_closed(tmp_path: Path) -> None:
    from src.core.exceptions import SysRevError
    from src.state.db import restrict_directory_to_current_user

    with pytest.raises(SysRevError, match="could not restrict"):
        restrict_directory_to_current_user(tmp_path, run=_Runner(icacls_returncode=5))


def test_prepare_sqlite_path_applies_the_acl_only_to_new_directories_on_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os
    import types

    import src.state.db as db

    calls: list[Path] = []
    monkeypatch.setattr(db, "os", types.SimpleNamespace(name="nt", environ=os.environ))
    monkeypatch.setattr(db, "restrict_directory_to_current_user", calls.append)
    new = tmp_path / "yeni dizin ğ" / "db"
    db.prepare_sqlite_path(f"sqlite:///{new}/sysrev.db")
    assert calls == [new] and new.is_dir()
    db.prepare_sqlite_path(f"sqlite:///{new}/sysrev.db")  # existing: left unchanged
    db.prepare_sqlite_path("sqlite:///:memory:")
    assert calls == [new]


def test_a_stored_draft_with_the_old_sentence_is_routed_to_human_review(
    session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A version created before Session 10 keeps its text; its values still verify, and the
    old unlabelled sentence is no longer template text, so a new audit asks a human about it
    (non-blocking) instead of passing it silently."""
    import src.manuscript.drafting as drafting
    from src.core.schemas.audit import CheckStatus
    from src.manuscript.service import ManuscriptService
    from tests.manuscript_helpers import SYSTEM

    def old(p: object, k: str) -> list[str]:  # the pre-Session-10 selection by key prefix
        return [f"{{{{{x}}}}}" for x in p.facts if x.startswith(f"{k}.sensitivity.")]  # type: ignore[attr-defined]

    repo, _, _ = _workflow(session, "common_effect", tmp_path)
    svc = ManuscriptService(repo, "mc1")
    with monkeypatch.context() as m:
        m.setattr(drafting, "_sensitivity_clauses", old)
        legacy = svc.create_template_draft(SYSTEM)
    assert "Sensitivity analysis: 2; 400; 70; 1.35" in legacy.rendered_markdown
    report = svc.audit()
    near = [x for x in report.findings if "Sensitivity analysis: 2; 400; 70" in x.excerpt]
    assert [(x.scope, x.status) for x in near] == [("narrative", CheckStatus.REQUIRES_HUMAN_REVIEW)]
    assert not any(x.blocking for x in near)
    # the stored version itself is unchanged by the upgrade
    assert repo.latest_manuscript_version("mc1").content_sha256 == legacy.content_sha256  # type: ignore[union-attr]
