"""Fail-closed guards derived from persisted records (regression tests for the PR #1 review).

Each test builds real database state with the repository and shows that a Stage 2-4
transition is refused when the underlying work is absent, incomplete or inconsistent,
including states that the previous implementation would have let through.
"""

from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from src.core.enums import (
    FullTextStatus,
    Phase,
    ReviewerRole,
    ScreeningStage,
    ScreeningVerdict,
    SourceDatabase,
)
from src.core.exceptions import SysRevError
from src.core.schemas import FullTextItem, PRISMACounts, ScreeningDecision
from src.screening.deduplication import deduplicate
from src.screening.prisma import PRISMATracker
from src.state.machine import PipelineStateMachine
from src.state.repository import NotFoundError, Repository
from tests.conftest import approved_rob, make_arm, make_protocol, make_record

INC, EXC, UNC = ScreeningVerdict.INCLUDE, ScreeningVerdict.EXCLUDE, ScreeningVerdict.UNCERTAIN
TA, FT = ScreeningStage.TITLE_ABSTRACT, ScreeningStage.FULL_TEXT
R1, R2, ADJ = ReviewerRole.REVIEWER_1, ReviewerRole.REVIEWER_2, ReviewerRole.ADJUDICATOR
PID = "p1"


@pytest.fixture
def repo(session: Session) -> Repository:
    r = Repository(session)
    r.create_project(PID, "Synthetic review")
    r.save_protocol(PID, make_protocol())
    HUMAN["id"], _ = r.enroll_reviewer("Dr Human")
    return r


HUMAN: dict[str, str] = {}


def decide(
    repo: Repository,
    rid: str,
    reviewer: ReviewerRole,
    verdict: ScreeningVerdict,
    stage: ScreeningStage = TA,
) -> None:
    repo.save_screening_decision(
        PID,
        ScreeningDecision(
            record_id=rid,
            reviewer=reviewer,
            reviewer_identity="Dr Human" if reviewer == ADJ else f"agent-{reviewer.value}",
            is_human=reviewer == ADJ,
            reviewer_id=HUMAN["id"] if reviewer == ADJ else None,
            stage=stage,
            verdict=verdict,
            exclusion_reason="not eligible" if verdict == EXC else None,
            rationale="r",
        ),
    )


def violations(repo: Repository, phase: Phase, target: Phase) -> list[str]:
    return PipelineStateMachine(PID, phase).check(target, repo.build_guard_context(PID))


def save_prisma(repo: Repository, **kw: object) -> None:
    repo.save_prisma(PID, PRISMACounts(**kw))  # type: ignore[arg-type]


S2_TO_G15 = (Phase.STAGE2_SEARCH_SCREENING, Phase.GATE1_5_SCREENING_CONFLICTS)
G15_TO_S3 = (Phase.GATE1_5_SCREENING_CONFLICTS, Phase.STAGE3_FULLTEXT_RETRIEVAL)
S3_TO_G2 = (Phase.STAGE3_FULLTEXT_RETRIEVAL, Phase.GATE2_MISSING_FULLTEXT)
G2_TO_S4 = (Phase.GATE2_MISSING_FULLTEXT, Phase.STAGE4_EXTRACTION_ROB)
S4_TO_G3 = (Phase.STAGE4_EXTRACTION_ROB, Phase.GATE3_EXTRACTION_APPROVAL)
G3_TO_S5 = (Phase.GATE3_EXTRACTION_APPROVAL, Phase.STAGE5_META_ANALYSIS)
ALL_STEPS = [S2_TO_G15, G15_TO_S3, S3_TO_G2, G2_TO_S4, S4_TO_G3, G3_TO_S5]


@pytest.mark.parametrize("step", ALL_STEPS)
def test_fresh_project_is_blocked_at_every_stage_2_to_4_step(
    repo: Repository, step: tuple[Phase, Phase]
) -> None:
    assert violations(repo, *step)


def test_fresh_project_context_reports_nothing_as_done(repo: Repository) -> None:
    ctx = repo.build_guard_context(PID)
    assert ctx.records_identified == 0
    assert ctx.prisma_recorded is False
    assert ctx.ta_included == 0
    assert ctx.ft_included == 0


# --------------------------------------------------------------------------- screening


def test_records_without_any_screening_decisions_block_stage2(repo: Repository) -> None:
    repo.save_records(PID, [make_record("a", "T1"), make_record("b", "T2")])
    save_prisma(repo, identified_by_database={"pubmed": 2}, records_awaiting_screening=2)
    v = violations(repo, *S2_TO_G15)
    assert any("2 unique record(s) lack title/abstract decisions" in x for x in v)


def test_single_reviewer_is_not_complete_screening(repo: Repository) -> None:
    repo.save_records(PID, [make_record("a", "T1")])
    decide(repo, "a", R1, INC)
    ctx = repo.build_guard_context(PID)
    assert ctx.ta_missing_decisions == 1
    assert ctx.ta_included == 0


def test_disagreement_without_a_conflict_row_still_blocks(repo: Repository) -> None:
    """Previously only *saved* conflict rows were counted; an unsaved disagreement slipped by."""
    repo.save_records(PID, [make_record("a", "T1")])
    decide(repo, "a", R1, INC)
    decide(repo, "a", R2, EXC)
    save_prisma(repo, identified_by_database={"pubmed": 1}, records_awaiting_screening=1)
    assert repo.load_conflicts(PID) == []
    v = violations(repo, *G15_TO_S3)
    assert any("1 unresolved title/abstract screening conflict" in x for x in v)


def test_double_uncertain_requires_human(repo: Repository) -> None:
    repo.save_records(PID, [make_record("a", "T1")])
    decide(repo, "a", R1, UNC)
    decide(repo, "a", R2, UNC)
    assert repo.build_guard_context(PID).ta_unresolved_conflicts == 1
    decide(repo, "a", ADJ, INC)
    ctx = repo.build_guard_context(PID)
    assert (ctx.ta_unresolved_conflicts, ctx.ta_included) == (0, 1)


def test_duplicates_are_not_required_to_be_screened(repo: Repository) -> None:
    result = deduplicate(
        [make_record("a", "T", doi="10.5555/x"), make_record("b", "T2", doi="10.5555/x")]
    )
    repo.save_records(PID, result.all_records)
    rep = result.unique_records[0].record_id
    decide(repo, rep, R1, INC)
    decide(repo, rep, R2, INC)
    ctx = repo.build_guard_context(PID)
    assert (ctx.records_identified, ctx.records_unique, ctx.ta_missing_decisions) == (2, 1, 0)


# --------------------------------------------------------------------------- PRISMA


def _screened_project(repo: Repository) -> None:
    repo.save_records(PID, [make_record("a", "T1"), make_record("b", "T2")])
    for rid, v in (("a", INC), ("b", EXC)):
        decide(repo, rid, R1, v)
        decide(repo, rid, R2, v)


def test_missing_prisma_blocks_screening_exit(repo: Repository) -> None:
    _screened_project(repo)
    v = violations(repo, *G15_TO_S3)
    assert v == ["PRISMA counts have not been recorded for this project"]


def test_prisma_snapshot_inconsistent_with_records_blocks(repo: Repository) -> None:
    _screened_project(repo)
    # Internally consistent snapshot that does not match the database (claims 3 records).
    save_prisma(
        repo, identified_by_database={"pubmed": 3}, records_screened=3, records_excluded=2,
        reports_sought=1, reports_awaiting_retrieval=1,
    )  # fmt: skip
    v = violations(repo, *G15_TO_S3)
    assert any("total_identified is 3 but persisted records imply 2" in x for x in v)
    assert any("records_screened is 3 but persisted records imply 2" in x for x in v)


def test_consistent_prisma_from_tracker_passes_screening_exit(repo: Repository) -> None:
    _screened_project(repo)
    t = PRISMATracker()
    t.record_identified(SourceDatabase.PUBMED, 2)
    t.register_screening_pool(["a", "b"])
    t.record_title_abstract_decision("a", include=True)
    t.record_title_abstract_decision("b", include=False)
    repo.save_prisma(PID, t.snapshot())
    assert violations(repo, *G15_TO_S3) == []


def test_unverifiable_removals_are_rejected(repo: Repository) -> None:
    _screened_project(repo)
    save_prisma(
        repo, identified_by_database={"pubmed": 3}, marked_ineligible_by_automation=1,
        records_screened=2, records_excluded=1, reports_sought=1, reports_awaiting_retrieval=1,
    )  # fmt: skip
    v = violations(repo, *G15_TO_S3)
    assert any("cannot yet be verified" in x for x in v)


# --------------------------------------------------------------------------- full-text manifest


def _through_screening(repo: Repository) -> None:
    _screened_project(repo)
    t = PRISMATracker()
    t.record_identified(SourceDatabase.PUBMED, 2)
    t.record_title_abstract_decision("a", include=True)
    t.record_title_abstract_decision("b", include=False)
    repo.save_prisma(PID, t.snapshot())


def test_absent_manifest_is_not_a_complete_manifest(repo: Repository) -> None:
    """Previously zero manifest rows meant zero blocking rows, so Gate 2 opened."""
    _through_screening(repo)
    assert repo.missing_fulltext_manifest(PID) == []
    for step in (S3_TO_G2, G2_TO_S4):
        assert any("no full-text manifest entry" in x for x in violations(repo, *step))


def test_manifest_entry_unlocks_gate2_only_when_resolved(repo: Repository) -> None:
    _through_screening(repo)
    repo.save_fulltext_item(
        PID, FullTextItem(record_id="a", title="T1", status=FullTextStatus.MISSING)
    )
    assert violations(repo, *S3_TO_G2) == []
    assert any("still missing" in x for x in violations(repo, *G2_TO_S4))
    upload = FullTextItem(record_id="a", title="T1", status=FullTextStatus.USER_UPLOADED,
                          pdf_filename="a.pdf", pdf_sha256="a" * 64)  # fmt: skip
    with pytest.raises(SysRevError, match="authenticated"):
        repo.save_fulltext_item(PID, upload)  # an upload must name an enrolled reviewer
    repo.save_fulltext_item(
        PID, upload.model_copy(update={"recorded_by": "Dr Human",
                                       "recorded_by_reviewer_id": HUMAN["id"]}),
    )  # fmt: skip
    assert violations(repo, *G2_TO_S4) == []


# ------------------------------------------------------------------- eligibility + extraction


def _through_retrieval(repo: Repository) -> None:
    _through_screening(repo)
    repo.save_fulltext_item(
        PID, FullTextItem(record_id="a", title="T1", status=FullTextStatus.OA_DOWNLOADED,
                          pdf_filename="a.pdf", pdf_sha256="a" * 64),
    )  # fmt: skip


def test_extraction_without_eligibility_decision_is_blocked(repo: Repository) -> None:
    _through_retrieval(repo)
    repo.save_extracted_arm(PID, make_arm("a"))
    v = violations(repo, *S4_TO_G3)
    assert any("lack full-text eligibility decisions" in x for x in v)
    assert any("not included at full-text" in x for x in v)


def test_included_report_without_extraction_is_blocked(repo: Repository) -> None:
    _through_retrieval(repo)
    decide(repo, "a", R1, INC, FT)
    decide(repo, "a", R2, INC, FT)
    v = violations(repo, *S4_TO_G3)
    assert any("have no extracted data" in x for x in v)
    assert any("no extracted study arms" in x for x in v)


def test_fulltext_disagreement_blocks(repo: Repository) -> None:
    _through_retrieval(repo)
    decide(repo, "a", R1, INC, FT)
    decide(repo, "a", R2, EXC, FT)
    assert any("full-text eligibility conflict" in x for x in violations(repo, *S4_TO_G3))


def test_complete_stage4_and_prisma_reaches_stage5(repo: Repository) -> None:
    _through_retrieval(repo)
    decide(repo, "a", R1, INC, FT)
    decide(repo, "a", R2, INC, FT)
    arm = make_arm("a")
    repo.save_extracted_arm(PID, arm)
    assert any("await confirmation" in x for x in violations(repo, *S4_TO_G3))
    decide(repo, "a", ADJ, INC, FT)  # a human confirms automated full-text eligibility
    assert violations(repo, *S4_TO_G3) == []
    v = violations(repo, *G3_TO_S5)
    assert any("not approved by a human" in x for x in v)
    assert any("reports_assessed is 0 but persisted records imply 1" in x for x in v)
    repo.approve_extracted_arm(arm.arm_id, reviewer_id=HUMAN["id"])
    repo.save_rob(PID, approved_rob("a", HUMAN["id"], "Dr Human"))
    t = PRISMATracker()
    t.record_identified(SourceDatabase.PUBMED, 2)
    t.record_title_abstract_decision("a", include=True)
    t.record_title_abstract_decision("b", include=False)
    t.record_retrieval("a", retrieved=True)
    t.record_fulltext_decision("a", include=True)
    repo.save_prisma(PID, t.snapshot())
    assert violations(repo, *G3_TO_S5) == []


# --------------------------------------------------------------------------- duplicate candidates


def test_unresolved_duplicate_candidate_blocks_and_resolution_merges(repo: Repository) -> None:
    recs = [
        make_record("a", "Alpha", doi="10.5555/x", pmid="1"),
        make_record("b", "Beta", doi="10.5555/x", pmid="2"),
    ]
    result = deduplicate(recs)
    repo.save_records(PID, result.all_records)
    for pd in result.possible_duplicates:
        repo.save_duplicate_candidate(PID, pd.to_candidate())
    for rid in ("a", "b"):
        decide(repo, rid, R1, INC)
        decide(repo, rid, R2, INC)
    t = PRISMATracker()
    t.record_identified(SourceDatabase.PUBMED, 2)
    t.record_title_abstract_decision("a", include=True)
    t.record_title_abstract_decision("b", include=True)
    repo.save_prisma(PID, t.snapshot())
    assert any("1 unresolved possible-duplicate" in x for x in violations(repo, *G15_TO_S3))

    [cand] = repo.load_duplicate_candidates(PID)
    resolved = repo.resolve_duplicate_candidate(
        PID, cand.candidate_id, resolution="duplicate_keep_a", reviewer_id=HUMAN["id"]
    )
    assert resolved.is_resolved
    by_id = {r.record_id: r for r in repo.load_records(PID)}
    assert by_id["b"].duplicate_of == "a"
    # The stored PRISMA snapshot no longer matches the records: it must be rebuilt.
    v = violations(repo, *G15_TO_S3)
    assert any("duplicates_removed is 0 but persisted records imply 1" in x for x in v)

    t2 = PRISMATracker()
    t2.record_identified(SourceDatabase.PUBMED, 2)
    t2.record_duplicates_removed(1)
    t2.record_title_abstract_decision("a", include=True)
    repo.save_prisma(PID, t2.snapshot())
    assert violations(repo, *G15_TO_S3) == []


def test_resolve_distinct_keeps_records_separate(repo: Repository) -> None:
    result = deduplicate([make_record("a", "A", doi="10.5555/x", pmid="1"),
                          make_record("b", "B", doi="10.5555/x", pmid="2")])  # fmt: skip
    repo.save_records(PID, result.all_records)
    cand = result.possible_duplicates[0].to_candidate()
    repo.save_duplicate_candidate(PID, cand)
    repo.resolve_duplicate_candidate(
        PID, cand.candidate_id, resolution="distinct", reviewer_id=HUMAN["id"]
    )
    assert all(r.duplicate_of is None for r in repo.load_records(PID))
    assert repo.build_guard_context(PID).unresolved_duplicate_candidates == 0
    with pytest.raises(NotFoundError):
        repo.resolve_duplicate_candidate(
            PID, "nope", resolution="distinct", reviewer_id=HUMAN["id"]
        )


def test_resolution_repoints_existing_duplicates_of_dropped_cluster(repo: Repository) -> None:
    recs = [
        make_record("a", "A", doi="10.5555/x", pmid="1"),
        make_record("b", "B", doi="10.5555/x", pmid="2"),
        make_record("c", "C", pmid="2"),  # duplicate of b by PMID
    ]
    result = deduplicate(recs)
    assert result.duplicate_of == {"c": "b"}
    repo.save_records(PID, result.all_records)
    cand = next(
        p for p in result.possible_duplicates if {p.record_id_a, p.record_id_b} == {"a", "b"}
    )
    c = cand.to_candidate()
    repo.save_duplicate_candidate(PID, c)
    repo.resolve_duplicate_candidate(
        PID, c.candidate_id, resolution="duplicate_keep_a", reviewer_id=HUMAN["id"]
    )
    by_id = {r.record_id: r for r in repo.load_records(PID)}
    assert (by_id["b"].duplicate_of, by_id["c"].duplicate_of) == ("a", "a")
