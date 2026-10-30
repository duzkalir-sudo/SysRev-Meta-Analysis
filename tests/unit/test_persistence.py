from __future__ import annotations

import pytest
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.core.enums import (
    ActorType,
    CheckStatus,
    FullTextStatus,
    GateId,
    GateVerdict,
    OverlapResolution,
    Phase,
    ReviewerRole,
    ScreeningStage,
    ScreeningVerdict,
)
from src.core.exceptions import SysRevError
from src.core.schemas import (
    AuditReport,
    CitationCheck,
    CohortOverlapWarning,
    FullTextItem,
    GateDecision,
    OverlapSignal,
    PRISMACounts,
    ScreeningConflict,
    ScreeningDecision,
)
from src.state.db import (
    AppendOnlyViolationError,
    ExtractedValueORM,
    SourceEvidenceORM,
    StateTransitionORM,
)
from src.state.repository import NotFoundError, Repository
from tests.conftest import NOW, READY, make_arm, make_protocol, make_record


@pytest.fixture
def repo(session: Session) -> Repository:
    r = Repository(session)
    r.create_project("p1", "Synthetic review")
    return r


def test_project_not_found(repo: Repository) -> None:
    with pytest.raises(NotFoundError):
        repo.get_project("missing")


def test_protocol_round_trip_and_versioning(repo: Repository) -> None:
    v1 = make_protocol()
    repo.save_protocol("p1", v1)
    v2 = make_protocol(version=2, review_title="Revised title")
    repo.save_protocol("p1", v2)
    latest = repo.latest_protocol("p1")
    assert latest == v2


def test_records_round_trip_with_duplicates(repo: Repository) -> None:
    rep = make_record("a", "Trial A", doi="10.5555/a")
    dup = make_record("b", "Trial A", doi="10.5555/a", duplicate_of="a")
    repo.save_records("p1", [dup, rep])  # order should not matter
    assert repo.load_records("p1") == [rep, dup]
    assert repo.load_records("p1", unique_only=True) == [rep]


def test_screening_decision_unique_per_reviewer_stage(repo: Repository) -> None:
    repo.save_records("p1", [make_record("a", "Trial A")])
    d = ScreeningDecision(
        record_id="a",
        reviewer=ReviewerRole.REVIEWER_1,
        reviewer_identity="agent",
        stage=ScreeningStage.TITLE_ABSTRACT,
        verdict=ScreeningVerdict.INCLUDE,
        rationale="ok",
    )
    repo.save_screening_decision("p1", d)
    assert repo.load_screening_decisions("p1") == [d]
    with pytest.raises(IntegrityError):
        repo.save_screening_decision("p1", d.model_copy(update={"decision_id": "other"}))


def test_foreign_keys_enforced(repo: Repository) -> None:
    d = ScreeningDecision(
        record_id="ghost",
        reviewer=ReviewerRole.REVIEWER_1,
        reviewer_identity="agent",
        stage=ScreeningStage.TITLE_ABSTRACT,
        verdict=ScreeningVerdict.INCLUDE,
        rationale="ok",
    )
    with pytest.raises(IntegrityError):
        repo.save_screening_decision("p1", d)


def test_extracted_arm_round_trip_preserves_evidence(repo: Repository, session: Session) -> None:
    repo.save_records("p1", [make_record("rec1", "Trial")])
    arm = make_arm("rec1")
    repo.save_extracted_arm("p1", arm)
    loaded = repo.load_extracted_arms("p1")
    assert loaded == [arm]
    # The four values share one SourceEvidence row.
    assert len(session.scalars(select(SourceEvidenceORM)).all()) == 1
    assert len(session.scalars(select(ExtractedValueORM)).all()) == 4


def test_extracted_arm_cannot_be_overwritten(repo: Repository) -> None:
    repo.save_records("p1", [make_record("rec1", "Trial")])
    arm = make_arm("rec1")
    repo.save_extracted_arm("p1", arm)
    with pytest.raises(SysRevError, match="already exists"):
        repo.save_extracted_arm("p1", arm)


def test_value_without_evidence_rejected_by_database(repo: Repository, session: Session) -> None:
    repo.save_records("p1", [make_record("rec1", "Trial")])
    arm = make_arm("rec1")
    repo.save_extracted_arm("p1", arm)
    session.commit()
    with pytest.raises(IntegrityError):
        session.execute(update(ExtractedValueORM).values(evidence_id=None))


def test_approve_extracted_arm(repo: Repository) -> None:
    repo.save_records("p1", [make_record("rec1", "Trial")])
    arm = make_arm("rec1")
    repo.save_extracted_arm("p1", arm)
    approved = repo.approve_extracted_arm(arm.arm_id, "Dr Reviewer")
    assert approved.approved_by_human and approved.approved_at is not None
    assert repo.load_extracted_arms("p1")[0].approved_by == "Dr Reviewer"
    with pytest.raises(NotFoundError):
        repo.approve_extracted_arm("missing", "x")


def test_append_only_tables(repo: Repository, session: Session) -> None:
    machine = repo.load_state_machine("p1")
    t = machine.transition(
        Phase.GATE1_SEARCH_STRATEGY, actor="orch", actor_type=ActorType.SYSTEM, context=READY
    )
    repo.record_transition(t)
    session.commit()
    row = session.scalars(select(StateTransitionORM)).one()
    row.actor = "tampered"
    with pytest.raises(AppendOnlyViolationError):
        session.flush()
    session.rollback()
    row = session.scalars(select(StateTransitionORM)).one()
    session.delete(row)
    with pytest.raises(AppendOnlyViolationError):
        session.flush()


def test_transition_persistence_and_reload(repo: Repository) -> None:
    m = repo.load_state_machine("p1")
    repo.record_transition(
        m.transition(
            Phase.GATE1_SEARCH_STRATEGY, actor="orch", actor_type=ActorType.SYSTEM, context=READY
        )
    )
    decision = GateDecision(gate=GateId.GATE_1, verdict=GateVerdict.APPROVED, reviewer_name="Dr R")
    repo.record_transition(m.submit_gate_decision(decision, READY))
    reloaded = repo.load_state_machine("p1")
    assert reloaded.phase == Phase.STAGE2_SEARCH_SCREENING
    assert reloaded.history == m.history
    assert reloaded.history[1].gate_decision == decision


def test_stale_transition_rejected(repo: Repository) -> None:
    m1 = repo.load_state_machine("p1")
    m2 = repo.load_state_machine("p1")
    repo.record_transition(
        m1.transition(
            Phase.GATE1_SEARCH_STRATEGY, actor="a", actor_type=ActorType.SYSTEM, context=READY
        )
    )
    m1.submit_gate_decision(
        GateDecision(gate=GateId.GATE_1, verdict=GateVerdict.APPROVED, reviewer_name="R"), READY
    )
    # m2 still believes the project is in Stage 1.
    stale = m2.transition(
        Phase.GATE1_SEARCH_STRATEGY, actor="b", actor_type=ActorType.SYSTEM, context=READY
    )
    with pytest.raises(SysRevError, match="stale"):
        repo.record_transition(stale)


def test_guard_context_reflects_database(repo: Repository) -> None:
    ctx = repo.build_guard_context("p1")
    assert ctx.protocol_blockers == ("protocol not drafted",)
    assert ctx.records_identified == 0

    repo.save_protocol("p1", make_protocol())
    repo.save_records("p1", [make_record("a", "Trial A"), make_record("b", "Trial B")])
    repo.save_conflict(
        "p1",
        ScreeningConflict(
            record_id="a",
            stage=ScreeningStage.TITLE_ABSTRACT,
            reviewer_1_verdict=ScreeningVerdict.INCLUDE,
            reviewer_2_verdict=ScreeningVerdict.EXCLUDE,
        ),
    )
    warning = CohortOverlapWarning(
        record_id_a="a",
        record_id_b="b",
        overlap_score=0.8,
        severity="high",
        signals=[OverlapSignal(kind="coauthor_overlap", score=0.8, detail="x")],
    )
    repo.save_overlap_warning("p1", warning)
    repo.save_fulltext_item("p1", FullTextItem(record_id="a", title="Trial A"))
    repo.save_records("p1", [make_record("rec1", "Trial")])
    repo.save_extracted_arm("p1", make_arm("rec1"))

    ctx = repo.build_guard_context("p1")
    assert ctx.protocol_blockers == ()
    assert ctx.records_identified == 3
    assert ctx.unresolved_screening_conflicts == 1
    assert ctx.unresolved_cohort_warnings == 1
    assert ctx.fulltext_items_blocking == 1
    assert ctx.extraction_arms == 1 and ctx.unapproved_extraction_arms == 1
    assert ctx.extracted_values_without_evidence == 0
    assert ctx.audit_passed is None

    # Resolve everything.
    conflict = repo.load_conflicts("p1")[0]
    repo.save_conflict(
        "p1",
        conflict.model_copy(
            update={
                "resolved_verdict": ScreeningVerdict.INCLUDE,
                "resolved_by": "Dr R",
                "resolved_at": NOW,
            }
        ),
    )
    repo.save_overlap_warning(
        "p1",
        warning.model_copy(
            update={"resolution": OverlapResolution.DISTINCT_COHORTS, "resolved_by": "Dr R"}
        ),
    )
    repo.save_fulltext_item(
        "p1",
        FullTextItem(
            record_id="a",
            title="Trial A",
            status=FullTextStatus.UNOBTAINABLE,
            unobtainable_reason="authors did not respond after two requests",
        ),
    )
    ctx = repo.build_guard_context("p1")
    assert ctx.unresolved_screening_conflicts == 0
    assert ctx.unresolved_cohort_warnings == 0
    assert ctx.fulltext_items_blocking == 0
    assert repo.missing_fulltext_manifest("p1") == []


def test_prisma_and_audit_persistence(repo: Repository) -> None:
    counts = PRISMACounts(
        identified_by_database={"pubmed": 3}, records_screened=3, records_excluded=3
    )
    repo.save_prisma("p1", counts)
    assert repo.load_prisma("p1") == counts

    report = AuditReport(
        project_id="p1",
        manuscript_version=1,
        stats_output_sha256="d" * 64,
        citation_checks=[
            CitationCheck(
                citation_text="ref",
                manuscript_location="ref 1",
                pmid="1",
                matched_record_id="a",
                status=CheckStatus.VERIFIED,
            )
        ],
    )
    repo.save_audit_report(report)
    loaded = repo.latest_audit_report("p1")
    assert loaded == report and loaded is not None and loaded.passed
    assert repo.build_guard_context("p1").audit_passed is True
