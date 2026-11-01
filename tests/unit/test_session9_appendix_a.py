"""The ten Appendix A tests of the Session 9 audit report, with the oracles revised where the
independent review required it. All ten failed on main df1a79a; all pass after remediation.

Revisions (see the PR description):
* H1 (first test): the title allows "update the arm or block Gate 3". The implementation
  invalidates the stale arm, blocks Gate 3 until the arms are rebuilt, and the rebuilt arm
  carries the value; both halves are asserted.
* M1: on the identity ("plain") scale a confidence limit of 0 % is an ordinary value, so all
  three studies are retained (the audit oracle wrongly required excluding s3).
* M4: the validation-bypassing ``model_copy`` scenario is replaced by a real rerun in
  tests/unit/test_session9_remediation.py; here the same contract is checked through it.
All data are SYNTHETIC.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from src.core.enums import Phase
from src.core.exceptions import SysRevError
from src.core.schemas.extraction import parse_reported_number
from src.core.schemas.stats import AnalysisSpec
from src.extraction import (
    ReviewError,
    build_arms,
    correction_state_sha256,
    record_unavailability,
    reject_item,
    revise_reviewed_item,
)
from src.fulltext import register_upload
from src.stats_engine.engine import run_analysis
from src.stats_engine.stage5 import run_stage5
from tests.conftest import NOW
from tests.manuscript_helpers import comparative_project
from tests.stats_helpers import OUTCOMES, YEARS_5, inputs, km_arm
from tests.unit.test_master_table import (
    OTHER_PDF,
    PID,
    Project,
    crude,
    km,
)
from tests.unit.test_master_table import project as master_project  # noqa: F401


@pytest.fixture
def project(master_project: Project) -> Project:  # noqa: F811
    return master_project


def _arms(repo, rid="r1"):  # type: ignore[no-untyped-def]
    return [a for a in repo.load_extracted_arms(PID) if a.record_id == rid]


# H1
def test_h1_value_accepted_after_build_reaches_the_arm_or_blocks_gate3(project: Project) -> None:
    repo = project.repo
    (est,) = project.submit(km("reported_estimate", "62.1", 1, 1, 3))
    project.accept(est)
    build_arms(repo, PID, "r1", reviewer=project.reviewer)
    (risk,) = project.submit(km("n_at_risk", "40", 2, 1, 3))
    project.accept(risk)
    # either the arm already carries the value, or Gate 3 is blocked until it is rebuilt
    current = _arms(repo)
    if not (current and current[0].n_at_risk is not None):
        problems = repo.build_guard_context(PID).extraction_integrity_problems
        assert any("r1 / obliteration" in p for p in problems)
        build_arms(repo, PID, "r1", reviewer=project.reviewer)
    (arm,) = _arms(repo)
    assert arm.n_at_risk is not None and arm.n_at_risk.value == 40


def test_h1_correcting_a_rejected_value_invalidates_the_arm(project: Project) -> None:
    repo = project.repo
    est, risk = project.submit(
        km("reported_estimate", "62.1", 1, 1, 3), km("n_at_risk", "22", 2, 1, 3)
    )
    project.accept(est)
    reject_item(repo, PID, risk.item_id, reviewer=project.reviewer, note="wrong column")
    build_arms(repo, PID, "r1", reviewer=project.reviewer)
    repo.get_project(PID).phase = Phase.STAGE4_EXTRACTION_ROB.value
    repo.session.flush()
    shown = correction_state_sha256(repo, PID, repo.get_extraction_item(PID, risk.item_id))
    _, invalidated = revise_reviewed_item(
        repo, PID, risk.item_id, km("n_at_risk", "40", 2, 1, 3), project.doc("r1"),
        reviewer=project.reviewer, note="correct column", expected_state_sha256=shown,
    )  # fmt: skip
    assert invalidated


# H2
def test_h2_no_new_approved_arms_after_gate3(project: Project) -> None:
    repo = project.repo
    repo.get_project(PID).phase = Phase.STAGE5_META_ANALYSIS.value
    repo.session.flush()
    # the first step after Gate 3 is already refused, so no arm can be built
    with pytest.raises((ReviewError, SysRevError)):
        project.submit(crude("events", "45"), crude("n_analyzed", "57"))
    with pytest.raises((ReviewError, SysRevError)):
        build_arms(repo, PID, "r1", reviewer=project.reviewer)


# H3
def test_h3_pdf_replacement_after_stage5_is_refused(project: Project) -> None:
    repo = project.repo
    repo.get_project(PID).phase = Phase.STAGE6_MANUSCRIPT.value
    repo.session.flush()
    linked = repo.get_fulltext_item(PID, "r1")
    assert linked is not None
    with pytest.raises(SysRevError):
        register_upload(repo, project.store, PID, "r1", OTHER_PDF, original_filename="other.pdf",
                        reviewer=project.reviewer, replaces_sha256=linked.pdf_sha256)  # fmt: skip


def test_h3_pending_item_verified_against_old_pdf_cannot_be_accepted(project: Project) -> None:
    repo = project.repo
    (risk,) = project.submit(km("n_at_risk", "40", 2, 1, 3))
    linked = repo.get_fulltext_item(PID, "r1")
    assert linked is not None
    register_upload(repo, project.store, PID, "r1", OTHER_PDF, original_filename="other.pdf",
                    reviewer=project.reviewer, replaces_sha256=linked.pdf_sha256)  # fmt: skip
    with pytest.raises(ReviewError):
        project.accept(risk)


# M1 / M2
def _km_spec(scale: str) -> AnalysisSpec:
    return AnalysisSpec(analysis_id="k", outcome_id="control_km",
                        effect_measure="KM_PROBABILITY", timepoints=[YEARS_5],
                        reported_ci_scale=scale)  # fmt: skip


def test_m1_identity_scale_ci_at_zero_retains_all_three_studies() -> None:
    arms = [km_arm("s1", "40.0", "30.0", "50.0"), km_arm("s2", "50.0", "42.0", "58.0"),
            km_arm("s3", "6.0", "0.0", "12.0")]  # fmt: skip
    res = run_analysis(_km_spec("plain"), OUTCOMES["control_km"], inputs(arms))
    assert res.status == "pooled" and res.k == 3
    assert {r.record_id: r.status for r in res.reports}["s3"] == "included"


def test_m2_rounding_tolerance_is_per_printed_number() -> None:
    arms = [km_arm("s1", "62", "56.3", "67.8"), km_arm("s2", "71.0", "65.4", "75.9")]
    res = run_analysis(_km_spec("log_log"), OUTCOMES["control_km"], inputs(arms))
    assert {r.record_id: r for r in res.reports}["s1"].status == "included"


# M3 / M4
def test_m3_common_effect_weights_belong_to_the_mantel_haenszel_model(session: Session) -> None:
    repo, _, _ = comparative_project(session)
    a = repo.current_stats_output("mc1").analyses[0]  # type: ignore[union-attr]
    assert a.common_effect is not None
    assert a.common_effect.method.startswith("Mantel-Haenszel")
    mh = [s.data["events_control"] * s.data["n_intervention"]
          / (s.data["n_intervention"] + s.data["n_control"]) for s in a.studies]  # fmt: skip
    expected = [100 * w / sum(mh) for w in mh]
    assert [s.weight_common_pct for s in a.studies] == pytest.approx(expected, abs=1e-9)


def test_m4_rerun_removes_plots_of_the_previous_run(session: Session, tmp_path: Path) -> None:
    repo, rid, _ = comparative_project(session)
    repo.get_project("mc1").phase = Phase.STAGE5_META_ANALYSIS.value
    run_stage5(repo, "mc1", output_dir=tmp_path, generated_at=NOW)
    assert (tmp_path / "plots" / "comp.svg").exists()
    reviewer = repo.require_active_reviewer(rid)
    for t in ("t1", "t2", "t3"):
        record_unavailability(repo, "mc1", t, reviewer=reviewer, reason_code="other",
                              reason="synthetic", outcome_id="complication")  # fmt: skip
    rerun = run_stage5(repo, "mc1", output_dir=tmp_path, generated_at=NOW)
    assert rerun.output.analyses[0].status == "no_data"
    assert not (tmp_path / "plots" / "comp.svg").exists()


# M5
def test_m5_leading_zero_group_is_not_a_thousands_separator() -> None:
    assert parse_reported_number("0,500") == 0.5
