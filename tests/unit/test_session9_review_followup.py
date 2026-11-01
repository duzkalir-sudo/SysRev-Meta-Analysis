"""Session 9 review of PR #14 (commit 3a4f6ab): regressions for the five remaining findings.

P1a  an upgrade must keep valid projects created by the previous release valid;
P1b  Gate 3 / Stage 5 -> 6 checking and recording are one protected transaction;
P1c  (UI, see tests/unit/test_operations_ui.py) decisions bound to the displayed PDF;
P2a  scale-specific domains of confidence limits (log allows an upper limit of 100 %);
P2b  the Mantel-Haenszel risk ratio uses every table that enters its sums.

All data are SYNTHETIC.
"""

from __future__ import annotations

import gzip
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select, text, update
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from src.api_server.app import create_app
from src.core.enums import ActorType, ArmRole, DataStatus, ExtractionDomain, Phase
from src.core.schemas.master import ExtractionItem
from src.core.schemas.stats import AnalysisSpec
from src.manuscript.service import completion_problems
from src.state.db import (
    GateDecisionORM,
    StateTransitionORM,
    StatsOutputORM,
    init_db,
    make_engine,
    make_session_factory,
)
from src.state.machine import PipelineStateMachine
from src.state.repository import Repository
from src.stats_engine import effect_sizes as es
from src.stats_engine.engine import run_analysis
from src.stats_engine.report import LEGACY_WEIGHTS_LABEL, analysis_markdown
from tests.manuscript_helpers import comparative_project
from tests.stats_helpers import OUTCOMES, YEARS_5, inputs, km_arm, stats_arm

LEGACY = Path(__file__).resolve().parents[1] / "fixtures" / "legacy"
OLD_COMMIT = "df1a79a72d26dd5b41d0cde341472c47e5f5bc6c"


# =========================================================================== P1a
def _open_legacy(db: Path) -> tuple[Any, dict[str, Any]]:
    engine = make_engine(f"sqlite:///{db}")
    init_db(engine)  # adds the Session 9 column; nothing else is rewritten
    with engine.connect() as conn:
        facts = json.loads(conn.execute(text("SELECT payload FROM legacy_facts")).scalar_one())
    return engine, facts


def _assert_still_valid(db: Path) -> None:
    engine, facts = _open_legacy(db)
    with make_session_factory(engine)() as s:
        repo = Repository(s)
        for pid in ("ms1", "mc1"):
            project = repo.get_project(pid)
            assert project.phase == facts[pid]["phase"]
            assert project.stats_output_sha256 == facts[pid]["stats_output_sha256"]
            assert repo.stats_output_problems(pid) == []
            assert repo.load_stage5_inputs(pid).sha256() == facts[pid]["input_sha256"]
            output = repo.current_stats_output(pid)
            assert output is not None and output.engine_version == "1.0.0"
            assert output.sha256() == facts[pid]["stats_output_sha256"]  # nothing rewritten
            values = [
                v for a in repo.load_extracted_arms(pid) for v in a.extracted_values().values()
            ]
            assert values and {v.format_status for v in values} == {"ok"}
        # the signed-off project can still be completed: audit, acknowledgements and sign-off
        # of the previous release remain valid for the unchanged evidence
        assert completion_problems(repo, "ms1") == []
        # engine 1.0.0 weights are labelled as what they are (inverse variance)
        comp = repo.current_stats_output("mc1")
        assert comp is not None
        assert f"Weight CE ({LEGACY_WEIGHTS_LABEL}) %" in analysis_markdown(comp.analyses[0])
    engine.dispose()


@pytest.fixture
def legacy_db(tmp_path: Path) -> Path:
    """The database created by commit df1a79a (tests/fixtures/legacy/make_legacy_db.py)."""
    db = tmp_path / "legacy.sqlite"
    with gzip.open(LEGACY / "df1a79a_projects.sqlite.gz", "rb") as src, db.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    return db


def test_p1a_projects_created_by_the_previous_release_stay_valid(legacy_db: Path) -> None:
    _assert_still_valid(legacy_db)


def test_p1a_tampering_with_a_legacy_artifact_is_still_rejected(legacy_db: Path) -> None:
    engine, _ = _open_legacy(legacy_db)
    with engine.begin() as conn:  # a raw edit: a pooled estimate becomes a single study
        content = conn.execute(
            select(StatsOutputORM.content).where(StatsOutputORM.project_id == "mc1")
        ).scalar_one()
        conn.execute(
            update(StatsOutputORM)
            .where(StatsOutputORM.project_id == "mc1")
            .values(content=content.replace('"pooled"', '"single_study"', 1))
        )
    with make_session_factory(engine)() as s:
        problems = Repository(s).stats_output_problems("mc1")
        assert any("does not match its SHA-256" in p for p in problems)
    engine.dispose()


def test_p1a_a_1_0_0_output_cannot_claim_1_1_0_content(legacy_db: Path) -> None:
    from src.core.schemas.stats import StatsOutput

    engine, _ = _open_legacy(legacy_db)
    with make_session_factory(engine)() as s:
        output = Repository(s).current_stats_output("mc1")
        assert output is not None
        data = output.model_dump(mode="json")
        data["analyses"][0]["studies"][0]["weight_sensitivity_pct"] = {"x": 100.0}
        with pytest.raises(ValueError, match=r"engine 1\.0\.0 outputs have no sensitivity"):
            StatsOutput.model_validate(data)
    engine.dispose()


def _old_source_available() -> Path | None:
    root = Path(__file__).resolve().parents[2]
    try:
        subprocess.run(["git", "cat-file", "-e", f"{OLD_COMMIT}^{{commit}}"], cwd=root,
                       check=True, capture_output=True)  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return None
    return root


def test_p1a_live_cross_version_database(tmp_path: Path) -> None:
    """Create the database with the *old* source (a git worktree of df1a79a), then open it
    with the current source. Skipped where the commit is not in the clone (shallow CI
    checkouts use the committed fixture above, built by the same script)."""
    root = _old_source_available()
    if root is None:
        pytest.skip(f"commit {OLD_COMMIT[:7]} is not available in this clone")
    old = tmp_path / "old_src"
    subprocess.run(["git", "worktree", "add", "--detach", str(old), OLD_COMMIT], cwd=root,
                   check=True, capture_output=True)  # fmt: skip
    try:
        db = tmp_path / "created_by_old_source.sqlite"
        subprocess.run(
            [sys.executable, "-I", str(LEGACY / "make_legacy_db.py"), str(db)],
            cwd=old, check=True, capture_output=True, timeout=600,
        )  # fmt: skip
        _assert_still_valid(db)
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(old)], cwd=root,
                       capture_output=True)  # fmt: skip


# =========================================================================== P1b
GATE3 = "/projects/mc1/gates/gate_3"


@pytest.fixture
def gate3_db(tmp_path: Path) -> tuple[str, str, str]:
    """A file-based database (separate connections) with a project ready for Gate 3."""
    url = f"sqlite:///{tmp_path / 'gate3.db'}"
    engine = make_engine(url)
    init_db(engine)
    with make_session_factory(engine)() as s:
        repo, rid, token = comparative_project(s)
        repo.get_project("mc1").phase = Phase.GATE3_EXTRACTION_APPROVAL.value
        s.commit()
    engine.dispose()
    return url, rid, token


def _pending_item() -> ExtractionItem:
    return ExtractionItem(
        record_id="t1", domain=ExtractionDomain.STUDY_CHARACTERISTICS, field_name="setting",
        value_kind="text", data_status=DataStatus.NOT_REPORTED, proposed_by=ActorType.AGENT,
        proposer_identity="llm:test",
    )  # fmt: skip


def _other_session_writes_pending_item(url: str) -> str:
    """A second connection saves a valid pending item through the normal repository method;
    a short busy timeout makes a held write lock visible as a refusal."""
    other = create_engine(url, connect_args={"timeout": 0.3})
    try:
        with Session(other) as s2:
            Repository(s2).save_extraction_item("mc1", _pending_item())
            s2.commit()
        return "committed"
    except OperationalError as exc:
        return "locked" if "locked" in str(exc) else f"failed: {exc}"
    finally:
        other.dispose()


def _history(url: str) -> tuple[int, int, int, str]:
    engine = make_engine(url)
    with make_session_factory(engine)() as s:
        counts = tuple(
            int(s.scalar(select(func.count()).select_from(t)) or 0)
            for t in (GateDecisionORM, StateTransitionORM)
        )
        basis = len(Repository(s).governance_records("mc1", "gate3_basis"))
        phase = Repository(s).current_phase("mc1")
    engine.dispose()
    return counts[0], counts[1], basis, phase


def test_p1b_write_between_guard_check_and_approval_cannot_happen(
    gate3_db: tuple[str, str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    url, _, token = gate3_db
    outcomes: list[str] = []
    original = PipelineStateMachine.submit_gate_decision

    def interleaved(self: PipelineStateMachine, decision: Any, context: Any) -> Any:
        transition = original(self, decision, context)  # the guards have passed
        outcomes.append(_other_session_writes_pending_item(url))
        return transition

    monkeypatch.setattr(PipelineStateMachine, "submit_gate_decision", interleaved)
    with TestClient(create_app(url, data_dir=tmp_path)) as c:
        r = c.post(GATE3, headers={"Authorization": f"Bearer {token}"},
                   json={"verdict": "approved"})  # fmt: skip
    assert r.status_code == 200, r.text
    assert outcomes == ["locked"]  # the competing write waited for the lock and was refused
    engine = make_engine(url)
    with make_session_factory(engine)() as s:
        repo = Repository(s)
        assert repo.current_phase("mc1") == Phase.STAGE5_META_ANALYSIS.value
        assert repo.load_extraction_items("mc1") == []  # no pending item slipped in
        assert repo.build_guard_context("mc1").gate3_basis_problems == ()
    engine.dispose()
    # after the approval the same write is refused as a change to Gate 3-approved data
    with pytest.raises(Exception, match="approved at Gate 3"):
        _other_session_writes_pending_item(url)


def test_p1b_pending_item_saved_before_the_request_refuses_gate3(
    gate3_db: tuple[str, str, str], tmp_path: Path
) -> None:
    url, _, token = gate3_db
    assert _other_session_writes_pending_item(url) == "committed"
    before = _history(url)
    with TestClient(create_app(url, data_dir=tmp_path)) as c:
        r = c.post(GATE3, headers={"Authorization": f"Bearer {token}"},
                   json={"verdict": "approved"})  # fmt: skip
    assert r.status_code == 409 and "await human review" in json.dumps(r.json())
    assert _history(url) == before  # no approval, no transition, no basis recorded
    assert before[3] == Phase.GATE3_EXTRACTION_APPROVAL.value


def test_p1b_successful_gate3_records_decision_transition_and_basis_together(
    gate3_db: tuple[str, str, str], tmp_path: Path
) -> None:
    url, _, token = gate3_db
    decisions, transitions, basis, _ = _history(url)
    with TestClient(create_app(url, data_dir=tmp_path)) as c:
        r = c.post(GATE3, headers={"Authorization": f"Bearer {token}"},
                   json={"verdict": "approved"})  # fmt: skip
    assert r.status_code == 200
    assert _history(url) == (
        decisions + 1,
        transitions + 1,
        basis + 1,
        Phase.STAGE5_META_ANALYSIS.value,
    )


def test_p1b_stage5_to_6_is_checked_and_recorded_under_the_lock(
    gate3_db: tuple[str, str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    url, rid, token = gate3_db
    with TestClient(create_app(url, data_dir=tmp_path)) as c:
        auth = {"Authorization": f"Bearer {token}"}
        assert c.post(GATE3, headers=auth, json={"verdict": "approved"}).status_code == 200
        engine = make_engine(url)
        with make_session_factory(engine)() as s:  # statistics for the approved data
            repo = Repository(s)
            from src.stats_engine.engine import run_plan
            from tests.conftest import NOW

            plan = repo.latest_analysis_plan("mc1")
            assert plan is not None
            repo.record_stats_output(
                "mc1", run_plan(plan, repo.load_stage5_inputs("mc1"), project_id="mc1",
                                generated_at=NOW),
            )  # fmt: skip
            s.commit()
        engine.dispose()
        outcomes: list[str] = []
        original = PipelineStateMachine.transition

        def interleaved(self: PipelineStateMachine, target: Phase, **kw: Any) -> Any:
            t = original(self, target, **kw)
            if target == Phase.STAGE6_MANUSCRIPT:  # a Stage 5 input changes after the check
                other = create_engine(url, connect_args={"timeout": 0.3})
                try:
                    with Session(other) as s2:
                        r2 = Repository(s2)
                        from src.extraction import record_unavailability

                        record_unavailability(
                            r2, "mc1", "t3", reviewer=r2.require_active_reviewer(rid),
                            reason_code="other", reason="late", outcome_id="complication",
                        )  # fmt: skip
                        s2.commit()
                    outcomes.append("committed")
                except OperationalError:
                    outcomes.append("locked")
                finally:
                    other.dispose()
            return t

        monkeypatch.setattr(PipelineStateMachine, "transition", interleaved)
        r = c.post("/projects/mc1/advance", headers=auth,
                   json={"target": Phase.STAGE6_MANUSCRIPT.value})  # fmt: skip
    assert r.status_code == 200, r.text
    assert outcomes == ["locked"]
    engine = make_engine(url)
    with make_session_factory(engine)() as s:
        repo = Repository(s)
        assert repo.current_phase("mc1") == Phase.STAGE6_MANUSCRIPT.value
        assert repo.load_unavailability("mc1") == []
        assert repo.stats_output_problems("mc1") == []  # the verified statistics stay valid
    engine.dispose()


# =========================================================================== P2a
def _km(arms: list[Any], scale: str) -> Any:
    spec = AnalysisSpec(analysis_id="k", outcome_id="control_km",
                        effect_measure="KM_PROBABILITY", timepoints=[YEARS_5],
                        reported_ci_scale=scale)  # fmt: skip
    return run_analysis(spec, OUTCOMES["control_km"], inputs(arms))


def test_p2a_log_scale_accepts_an_upper_limit_of_100_percent() -> None:
    # both intervals are exactly symmetric on the log scale
    a = km_arm("A", "80.0", "64.0", "100.0")
    b = km_arm("B", "50.0", "40.0", "62.5")
    res = _km([a, b], "log")
    assert res.status == "pooled" and res.k == 2 and not res.excluded
    study = next(s for s in res.studies if s.study_id == "A")
    z = 1.959963984540054
    assert study.vi == pytest.approx(((0.0 - math.log(0.64)) / (2 * z)) ** 2, rel=1e-12)
    assert study.data["ci_upper"] == 1.0  # the printed limit is unchanged, not clipped


@pytest.mark.parametrize(
    ("scale", "lower_zero_ok", "upper_one_ok"),
    [("plain", True, True), ("log", False, True), ("logit", False, False),
     ("log_log", False, False)],
)  # fmt: skip
def test_p2a_boundary_limits_per_scale(scale: str, lower_zero_ok: bool, upper_one_ok: bool) -> None:
    assert es.scale_domain_contains(0.0, scale) is lower_zero_ok  # type: ignore[arg-type]
    assert es.scale_domain_contains(1.0, scale) is upper_one_ok  # type: ignore[arg-type]
    control = km_arm("ctl", "50.0", "42.0", "58.0")  # symmetric on the identity scale only
    zero = km_arm("zero", "6.0", "0.0", "12.0")  # symmetric on the identity scale
    one = km_arm("one", "80.0", "64.0", "100.0")  # symmetric on the log scale
    res = _km([control, zero, one], scale)
    codes = {e.study_id: e.reason_code for e in res.excluded}
    assert ("zero" in codes) is not (scale == "plain")
    if not lower_zero_ok:
        assert codes["zero"] == "ci_bound_not_transformable"
    if upper_one_ok:
        # a limit inside the domain is assessed on its own merits (symmetry), never as
        # "not transformable"
        assert codes.get("one") != "ci_bound_not_transformable"
        assert ("one" in codes) is (scale != "log")
    else:
        assert codes["one"] == "ci_bound_not_transformable"


# =========================================================================== P2b
def _two(rid: str, a: str, n1: str, c: str, n2: str) -> list[Any]:
    return [
        stats_arm(rid, "complication", role=ArmRole.INTERVENTION, label="X", events=a,
                  n_analyzed=n1),
        stats_arm(rid, "complication", role=ArmRole.CONTROL, label="P", events=c, n_analyzed=n2),
    ]  # fmt: skip


REVIEW_TABLES = [("s1", "30", "100", "20", "100"), ("s2", "10", "100", "10", "100"),
                 ("s3", "100", "100", "100", "100")]  # fmt: skip


def _ratio(tables: list[tuple[str, ...]], measure: str, **kw: Any) -> Any:
    arms = [arm for t in tables for arm in _two(*t)]
    spec = AnalysisSpec(analysis_id="r", outcome_id="complication", effect_measure=measure, **kw)
    return run_analysis(spec, OUTCOMES["complication"], inputs(arms))


def test_p2b_mh_risk_ratio_uses_every_table_in_its_sums() -> None:
    a = _ratio(REVIEW_TABLES, "RR")
    mh = a.common_effect
    assert mh is not None and mh.method.startswith("Mantel-Haenszel")
    assert mh.estimate_natural == pytest.approx(70 / 65, rel=1e-12)  # complete raw counts
    assert mh.k == 3
    # inverse-variance models and heterogeneity use the two studies with a log RR
    assert a.random_effects is not None and a.random_effects.k == 2
    assert a.heterogeneity is not None and a.heterogeneity.df == 1
    s3 = next(s for s in a.studies if s.study_id == "s3")
    assert s3.yi is None and s3.vi is None and s3.estimate_natural is None
    assert s3.weight_random_pct is None and not s3.weight_sensitivity_pct
    assert s3.weight_common_pct == pytest.approx(100 * 50 / 65)  # c·n1/N, metafor 76.92 %
    assert sum(s.weight_common_pct or 0 for s in a.studies) == pytest.approx(100.0)
    assert a.total_participants == 600 and a.total_events == 270
    ledger = {r.record_id: r.status for r in a.reports}
    assert ledger == {"s1": "included", "s2": "included", "s3": "included"}
    assert any("Mantel-Haenszel risk ratio (k = 3)" in w for w in a.warnings)
    md = analysis_markdown(a)
    assert "Synthetic s3" in md and "Mantel-Haenszel only" not in md  # table row, no estimate


def test_p2b_controls_or_inverse_variance_and_double_zero() -> None:
    # OR: a double-all-event table contributes nothing to the MH OR sums; it stays excluded
    o = _ratio(REVIEW_TABLES, "OR")
    assert o.common_effect is not None and o.common_effect.k == 2
    assert {e.study_id: e.reason_code for e in o.excluded} == {"s3": "non_informative"}
    # inverse-variance common effect: no MH model, so no MH-only study
    iv = _ratio(REVIEW_TABLES, "RR", common_effect_method="inverse_variance")
    assert iv.common_effect is not None and iv.common_effect.k == 2
    assert {e.study_id: e.reason_code for e in iv.excluded} == {"s3": "non_informative"}
    # a double-zero table contributes nothing to any MH sum: excluded, estimate unchanged
    dz = _ratio([*REVIEW_TABLES, ("s4", "0", "50", "0", "50")], "RR")
    assert dz.common_effect is not None
    assert dz.common_effect.estimate_natural == pytest.approx(70 / 65, rel=1e-12)
    assert {e.study_id: e.reason_code for e in dz.excluded} == {"s4": "non_informative"}
    # fewer than two studies with a log RR: the MH-only table is not pooled, with a reason
    one = _ratio([REVIEW_TABLES[0], REVIEW_TABLES[2]], "RR")
    assert one.status == "single_study"
    assert {e.study_id: e.reason_code for e in one.excluded} == {"s3": "non_informative"}
    assert "fewer than two studies" in one.excluded[0].reason


def test_p2b_forest_plot_marks_the_mh_only_study(tmp_path: Path) -> None:
    from src.stats_engine.report import forest_plot

    a = _ratio(REVIEW_TABLES, "RR", primary_model="common_effect")
    svg = forest_plot(a, tmp_path / "r.svg")
    assert svg is not None
    content = svg.read_text()
    assert "no study-level estimate" in content and "(Mantel-Haenszel only)" in content


def test_p2b_reference_case_is_in_the_live_r_validation() -> None:
    cases = json.loads(
        (
            Path(__file__).resolve().parents[2] / "validation" / "r" / "synthetic_cases.json"
        ).read_text()
    )
    assert cases["two_by_two"]["mh_double_all_event"]["ai"] == [30, 10, 100]
    ref = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "stats"
                      / "metafor_reference.json").read_text())  # fmt: skip
    case = next(c for c in ref["cases"] if c["id"] == "mh_double_all_event")
    assert math.exp(case["expected"]["RR"]["mh"]["estimate"]) == pytest.approx(70 / 65)
    assert case["expected"]["RR"]["mh"]["weights_all"][2] == pytest.approx(100 * 50 / 65)
    assert case["expected"]["OR"]["mh"]["weights_all"][2] == 0
