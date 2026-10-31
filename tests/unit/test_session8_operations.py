"""Session 8: human operations that were API/Python-only, now bound to the displayed state.

Corrections of reviewed values (with source evidence, history and arm invalidation),
report-to-study links (history, unlinking), double-counting decisions and analysis-plan
drafting/approval. All data SYNTHETIC; every decision is made by an enrolled reviewer.
"""

from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from src.core.enums import Phase, ReviewStatus
from src.core.exceptions import SysRevError
from src.core.schemas.stats import AnalysisSpec
from src.extraction import (
    ReviewError,
    StaleStateError,
    build_arms,
    correction_state_sha256,
    double_counting_state_sha256,
    link_reports_to_study,
    links_state_sha256,
    resolve_double_counting,
    revise_reviewed_item,
    unlink_reports,
)
from src.state.db import ExtractedArmORM
from src.state.repository import Repository
from src.stats_engine import plan_service
from tests.manuscript_helpers import comparative_project
from tests.unit.test_master_table import PID, Project, km
from tests.unit.test_master_table import project as master_project  # noqa: F401 (fixture)


@pytest.fixture
def project(master_project: Project) -> Project:  # noqa: F811
    """The Session 2 master-table project (three included reports with linked PDFs)."""
    return master_project


def _accepted_arm(project: Project) -> tuple[list, str]:  # type: ignore[type-arg]
    items = project.submit(km("reported_estimate", "62.1", 1, 1, 3), km("n_at_risk", "40", 2, 1, 3))
    for item in items:
        project.accept(item)
    (arm,) = build_arms(project.repo, PID, "r1", reviewer=project.reviewer)
    return items, arm.arm_id


def _stage4(repo: Repository) -> None:
    repo.get_project(PID).phase = Phase.STAGE4_EXTRACTION_ROB.value
    repo.session.flush()


# --------------------------------------------------------------------------- corrections
def test_correcting_an_accepted_value_invalidates_its_arm_and_keeps_history(
    project: Project,
) -> None:
    repo = project.repo
    items, old_arm = _accepted_arm(project)
    _stage4(repo)
    estimate = items[0]
    assert repo.build_guard_context(PID).included_reports_without_extraction == 2
    shown = correction_state_sha256(repo, PID, repo.get_extraction_item(PID, estimate.item_id))
    new, invalidated = revise_reviewed_item(
        repo,
        PID,
        estimate.item_id,
        km("reported_estimate", "81.4", 1, 2, 3),
        project.doc("r1"),
        reviewer=project.reviewer,
        note="the 3-year column was misread; 81.4 is printed for 3 years in this synthetic PDF",
        expected_state_sha256=shown,
    )
    assert invalidated == [old_arm]
    assert new.version == 2 and new.supersedes == estimate.item_id and new.value == 81.4
    assert new.review_status == ReviewStatus.ACCEPTED and new.verification is not None
    old = repo.get_extraction_item(PID, estimate.item_id)
    assert old.review_status == ReviewStatus.SUPERSEDED
    history = [e.action for e in repo.list_review_events(PID, estimate.item_id)]
    assert history == ["proposed", "accepted", "arms_built", "corrected", "arms_invalidated"]
    before = repo.list_review_events(PID, estimate.item_id)[3].before
    assert before is not None and before["reported_text"] == "62.1"  # earlier content kept
    # the invalidated arm stays stored (write-once) but no reader uses it
    assert repo.session.get(ExtractedArmORM, old_arm) is not None
    assert old_arm not in {a.arm_id for a in repo.load_extracted_arms(PID)}
    ctx = repo.build_guard_context(PID)
    assert ctx.included_reports_without_extraction == 3  # r1 must be rebuilt
    (rebuilt,) = build_arms(repo, PID, "r1", reviewer=project.reviewer)
    assert rebuilt.arm_id != old_arm and rebuilt.reported_estimate is not None
    assert rebuilt.reported_estimate.value == 81.4
    assert repo.build_guard_context(PID).included_reports_without_extraction == 2


def test_correction_from_a_stale_screen_or_late_phase_is_refused(project: Project) -> None:
    repo = project.repo
    items, _ = _accepted_arm(project)
    _stage4(repo)
    with pytest.raises(StaleStateError, match="has changed"):
        revise_reviewed_item(
            repo,
            PID,
            items[0].item_id,
            km("reported_estimate", "81.4", 1, 2, 3),
            project.doc("r1"),
            reviewer=project.reviewer,
            note="x",
            expected_state_sha256="0" * 64,
        )
    assert repo.get_extraction_item(PID, items[0].item_id).review_status == ReviewStatus.ACCEPTED
    with pytest.raises(ReviewError, match="cannot be stored"):  # number not in its quote
        revise_reviewed_item(
            repo,
            PID,
            items[0].item_id,
            km("reported_estimate", "99.9", 1, 2, 3),
            project.doc("r1"),
            reviewer=project.reviewer,
            note="x",
            expected_state_sha256=correction_state_sha256(
                repo, PID, repo.get_extraction_item(PID, items[0].item_id)
            ),
        )
    repo.get_project(PID).phase = Phase.STAGE5_META_ANALYSIS.value
    with pytest.raises(ReviewError, match="Stage 4 or at Gate 3"):
        revise_reviewed_item(
            repo,
            PID,
            items[0].item_id,
            km("reported_estimate", "81.4", 1, 2, 3),
            project.doc("r1"),
            reviewer=project.reviewer,
            note="x",
            expected_state_sha256=correction_state_sha256(
                repo, PID, repo.get_extraction_item(PID, items[0].item_id)
            ),
        )


# --------------------------------------------------------------------------- study links
def test_links_are_bound_to_the_displayed_state_and_keep_history(project: Project) -> None:
    repo, me = project.repo, project.reviewer
    shown = links_state_sha256(repo, PID)
    link_reports_to_study(
        repo,
        PID,
        ["r1", "r2"],
        study_id="S1",
        primary_record_id="r1",
        reviewer=me,
        note="same cohort",
        expected_state_sha256=shown,
    )
    with pytest.raises(StaleStateError, match="has changed"):  # the screen shown before
        link_reports_to_study(
            repo,
            PID,
            ["r3"],
            study_id="S1",
            primary_record_id="r3",
            reviewer=me,
            note="x",
            expected_state_sha256=shown,
        )
    current = links_state_sha256(repo, PID)
    link_reports_to_study(
        repo,
        PID,
        ["r2"],
        study_id="S2",
        primary_record_id="r2",
        reviewer=me,
        note="r2 is a different cohort after all",
        expected_state_sha256=current,
    )
    unlink_reports(
        repo,
        PID,
        ["r1"],
        reviewer=me,
        note="r1 is a single report",
        expected_state_sha256=links_state_sha256(repo, PID),
    )
    assert repo.study_ids(PID, {"r1", "r2"}) == {"r1": "r1", "r2": "S2"}
    history = [(h["record_id"], h["action"]) for h in repo.governance_records(PID, "study_link")]
    assert history == [("r1", "linked"), ("r2", "linked"), ("r2", "relinked"), ("r1", "unlinked")]


def test_link_changes_in_stage5_make_statistics_stale_and_rerunnable(session: Session) -> None:
    repo, rid, _ = comparative_project(session)
    repo.get_project("mc1").phase = Phase.STAGE5_META_ANALYSIS.value  # recomputation possible
    assert repo.stats_output_problems("mc1") == []
    reviewer = repo.require_active_reviewer(rid)
    link_reports_to_study(
        repo,
        "mc1",
        ["t1", "t2"],
        study_id="S",
        primary_record_id="t1",
        reviewer=reviewer,
        note="synthetic: one trial, two reports",
        expected_state_sha256=links_state_sha256(repo, "mc1"),
    )
    assert any("re-run Stage 5" in p for p in repo.stats_output_problems("mc1"))


# --------------------------------------------------------------------------- double counting
def test_double_counting_decision_requires_the_displayed_conflicts(project: Project) -> None:
    repo, me = project.repo, project.reviewer
    for rid in ("r1", "r2"):
        items = project.submit(km("reported_estimate", "62.1", 1, 1, 3), record_id=rid)
        project.accept(items[0])
        build_arms(repo, PID, rid, reviewer=me)
    shown = double_counting_state_sha256(repo, PID)
    link_reports_to_study(
        repo,
        PID,
        ["r1", "r2"],
        study_id="S1",
        primary_record_id="r1",
        reviewer=me,
        note="same cohort",
        expected_state_sha256=links_state_sha256(repo, PID),
    )
    assert len(repo.double_counting_conflicts(PID)) == 1
    with pytest.raises(StaleStateError):  # decided on the screen before the link existed
        resolve_double_counting(
            repo,
            PID,
            study_id="S1",
            outcome_id="obliteration",
            reviewer=me,
            justification="keep r1",
            keep_record_id="r1",
            drop_record_ids=["r2"],
            expected_state_sha256=shown,
        )
    resolve_double_counting(
        repo,
        PID,
        study_id="S1",
        outcome_id="obliteration",
        reviewer=me,
        justification="r2 is a follow-up of the same patients",
        keep_record_id="r1",
        drop_record_ids=["r2"],
        expected_state_sha256=double_counting_state_sha256(repo, PID),
    )
    assert repo.double_counting_conflicts(PID) == []


# --------------------------------------------------------------------------- analysis plans
def test_plan_drafts_revisions_and_bound_approval(session: Session) -> None:
    repo, rid, _ = comparative_project(session)
    reviewer = repo.require_active_reviewer(rid)
    repo.get_project("mc1").phase = Phase.STAGE5_META_ANALYSIS.value
    state = plan_service.plan_state(repo, "mc1")
    spec = AnalysisSpec(analysis_id="comp2", outcome_id="complication", effect_measure="OR")
    draft = plan_service.save_draft(
        repo,
        "mc1",
        [spec],
        rationale="OR as sensitivity",
        reviewer=reviewer,
        expected_latest_sha256=state["latest_sha256"],
        expected_protocol_sha256=plan_service.protocol_basis(repo, "mc1")[0],
    )
    with pytest.raises(StaleStateError):  # the earlier screen did not show this draft
        plan_service.save_draft(
            repo,
            "mc1",
            [spec],
            rationale=None,
            reviewer=reviewer,
            expected_latest_sha256=state["latest_sha256"],
            expected_protocol_sha256=plan_service.protocol_basis(repo, "mc1")[0],
        )
    revised = plan_service.save_draft(
        repo,
        "mc1",
        [spec.model_copy(update={"random_effects_ci": "wald"})],
        rationale="Wald",
        reviewer=reviewer,
        expected_latest_sha256=draft.sha256(),
        expected_protocol_sha256=plan_service.protocol_basis(repo, "mc1")[0],
    )
    assert revised.version == draft.version + 1 and revised.status == "draft"
    with pytest.raises(StaleStateError):  # approving the older draft that was displayed
        plan_service.approve_draft(
            repo,
            "mc1",
            draft.plan_id,
            reviewer=reviewer,
            expected_sha256=draft.sha256(),
            expected_latest_sha256=draft.sha256(),
            expected_protocol_sha256=plan_service.protocol_basis(repo, "mc1")[0],
        )
    approved = plan_service.approve_draft(
        repo,
        "mc1",
        revised.plan_id,
        reviewer=reviewer,
        expected_sha256=revised.sha256(),
        expected_latest_sha256=revised.sha256(),
        expected_protocol_sha256=plan_service.protocol_basis(repo, "mc1")[0],
    )
    assert approved.status == "approved" and approved.approved_reviewer_id == rid
    assert any("plan" in p for p in repo.stats_output_problems("mc1"))  # stats now stale
    history = [(p["version"], p["status"]) for p in plan_service.plan_state(repo, "mc1")["plans"]]
    assert history[-3:] == [(2, "draft"), (3, "draft"), (3, "approved")]


def test_plan_approval_after_stage5_is_refused(session: Session) -> None:
    repo, rid, _ = comparative_project(session)  # in Stage 6
    reviewer = repo.require_active_reviewer(rid)
    state = plan_service.plan_state(repo, "mc1")
    spec = AnalysisSpec(analysis_id="comp2", outcome_id="complication", effect_measure="OR")
    draft = plan_service.save_draft(
        repo,
        "mc1",
        [spec],
        rationale=None,
        reviewer=reviewer,
        expected_latest_sha256=state["latest_sha256"],
        expected_protocol_sha256=plan_service.protocol_basis(repo, "mc1")[0],
    )
    with pytest.raises(SysRevError, match="no route back"):
        plan_service.approve_draft(
            repo,
            "mc1",
            draft.plan_id,
            reviewer=reviewer,
            expected_sha256=draft.sha256(),
            expected_latest_sha256=draft.sha256(),
            expected_protocol_sha256=plan_service.protocol_basis(repo, "mc1")[0],
        )
    assert repo.stats_output_problems("mc1") == []  # nothing changed
