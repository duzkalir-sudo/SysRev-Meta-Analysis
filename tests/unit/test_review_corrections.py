"""Regression tests for the PR #5 review corrections (synthetic data only).

1. Human resolutions (conflicts, duplicates, cohort overlap) must be authenticated and
   backed by real decisions; fabricated resolutions cannot release guards.
2. XML-only full text is "retrieved" but not extraction-ready.
3. One cohort cannot contribute the same patients twice to an outcome.
4. PDF downloads refuse non-public destinations, including via redirects.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx
from sqlalchemy.orm import Session

from src.core.enums import (
    ActorType,
    FullTextStatus,
    OverlapResolution,
    OverlapSeverity,
    Phase,
    ReviewerRole,
    ScreeningStage,
    ScreeningVerdict,
)
from src.core.exceptions import SysRevError
from src.core.schemas import (
    CohortOverlapWarning,
    DuplicateCandidate,
    FullTextItem,
    OverlapSignal,
    ScreeningConflict,
    ScreeningDecision,
)
from src.extraction import (
    ProposedValue,
    ReviewError,
    record_unavailability,
    resolve_double_counting,
    submit_proposals,
)
from src.fulltext import (
    DocumentStore,
    PDFDownloader,
    acknowledge_xml_only,
    build_manifest,
    register_upload,
    report_flow,
)
from src.fulltext.download import is_public_address
from src.pdf_extractor import parse_pdf
from src.screening.agreement import screening_status
from src.screening.llm_screening import adjudicate
from src.state.db import ScreeningConflictORM
from src.state.machine import PipelineStateMachine
from src.state.repository import Repository
from tests.conftest import NOW, make_arm, make_record
from tests.fakes import approved_protocol, enroll, include_at_title_abstract
from tests.fixtures.pdf_factory import SYNTHETIC_DOI, SYNTHETIC_TITLE, synthetic_article

PID = "p1"
TA, FT = ScreeningStage.TITLE_ABSTRACT, ScreeningStage.FULL_TEXT
INC, EXC = ScreeningVerdict.INCLUDE, ScreeningVerdict.EXCLUDE
R1, R2 = ReviewerRole.REVIEWER_1, ReviewerRole.REVIEWER_2


@pytest.fixture
def repo(session: Session) -> Repository:
    r = Repository(session)
    r.create_project(PID, "Synthetic review")
    r.save_protocol(PID, approved_protocol())
    r.save_records(
        PID,
        [
            make_record("r1", SYNTHETIC_TITLE, doi=SYNTHETIC_DOI),
            make_record("r2", "Second synthetic report"),
            make_record("r3", "Third synthetic report"),
        ],
    )
    return r


def automated(
    repo: Repository,
    rid: str,
    role: ReviewerRole,
    verdict: ScreeningVerdict,
    stage: ScreeningStage = TA,
) -> None:
    repo.save_screening_decision(
        PID,
        ScreeningDecision(
            record_id=rid,
            reviewer=role,
            reviewer_identity=f"llm-{role}",
            stage=stage,
            verdict=verdict,
            rationale="synthetic",
            exclusion_reason="E1: Animal studies" if verdict == EXC else None,
        ),
    )


def fabricated(reviewer_id: str | None = None, name: str = "Dr Human") -> ScreeningConflict:
    return ScreeningConflict(
        record_id="r1",
        stage=TA,
        reviewer_1_verdict=INC,
        reviewer_2_verdict=EXC,
        resolved_verdict=INC,
        resolved_by=name,
        resolved_by_reviewer_id=reviewer_id,
        resolved_at=NOW,
    )


# --------------------------------------------------------------------------- 1. resolutions


def test_fabricated_conflict_resolutions_are_refused(repo: Repository) -> None:
    automated(repo, "r1", R1, INC)
    automated(repo, "r1", R2, EXC)
    human = enroll(repo)
    with pytest.raises(SysRevError, match="authenticated"):
        repo.save_conflict(PID, fabricated())  # no reviewer at all
    with pytest.raises(SysRevError, match="not enrolled"):
        repo.save_conflict(PID, fabricated("0" * 32))
    with pytest.raises(SysRevError, match="not backed by a matching human adjudication"):
        repo.save_conflict(PID, fabricated(human.id))  # enrolled, but never adjudicated
    with pytest.raises(SysRevError, match="does not match"):
        repo.save_conflict(PID, fabricated(human.id, name="Dr Someone Else"))
    wrong_source = ScreeningConflict(
        record_id="r1", stage=TA, reviewer_1_verdict=EXC, reviewer_2_verdict=INC
    )
    with pytest.raises(SysRevError, match="does not match the stored reviewer_1"):
        repo.save_conflict(PID, wrong_source)
    no_source = ScreeningConflict(
        record_id="r2", stage=TA, reviewer_1_verdict=INC, reviewer_2_verdict=EXC
    )
    with pytest.raises(SysRevError, match="does not match"):
        repo.save_conflict(PID, no_source)
    assert repo.build_guard_context(PID).ta_unresolved_conflicts == 1


def test_injected_resolution_cannot_release_guards(repo: Repository, session: Session) -> None:
    """Even a row written past the repository (e.g. a tampered DB) decides nothing."""
    automated(repo, "r1", R1, INC)
    automated(repo, "r1", R2, EXC)
    for rid in ("r2", "r3"):  # r2/r3 have no decisions at all
        bogus = ScreeningConflict(
            record_id=rid,
            stage=TA,
            reviewer_1_verdict=INC,
            reviewer_2_verdict=EXC,
            resolved_verdict=INC,
            resolved_by="Dr Fake",
            resolved_at=NOW,
        )
        session.add(
            ScreeningConflictORM(
                id=bogus.conflict_id,
                project_id=PID,
                record_id=rid,
                stage=TA.value,
                resolved_verdict=INC.value,
                payload=bogus.model_dump(mode="json"),
            )
        )
    injected = fabricated("0" * 32)
    session.add(
        ScreeningConflictORM(
            id=injected.conflict_id,
            project_id=PID,
            record_id="r1",
            stage=TA.value,
            resolved_verdict=INC.value,
            payload=injected.model_dump(mode="json"),
        )
    )
    session.flush()
    status = screening_status(
        {"r1", "r2", "r3"}, repo.load_screening_decisions(PID), repo.load_conflicts(PID), TA
    )
    assert status.final == {} and status.unresolved == ("r1",)
    assert status.missing == ("r2", "r3")
    ctx = repo.build_guard_context(PID)
    assert (ctx.ta_unresolved_conflicts, ctx.ta_missing_decisions, ctx.ta_included) == (1, 2, 0)
    assert PipelineStateMachine(PID, Phase.GATE1_5_SCREENING_CONFLICTS).check(
        Phase.STAGE3_FULLTEXT_RETRIEVAL, ctx
    )


def test_authenticated_adjudication_resolves_and_is_final(repo: Repository) -> None:
    automated(repo, "r1", R1, INC)
    automated(repo, "r1", R2, EXC)
    repo.save_conflict(
        PID,
        ScreeningConflict(record_id="r1", stage=TA, reviewer_1_verdict=INC, reviewer_2_verdict=EXC),
    )
    human = enroll(repo)
    adjudicate(repo, PID, "r1", TA, verdict=INC, rationale="eligible", reviewer=human)
    conflict = repo.find_conflict(PID, "r1", TA)
    assert conflict is not None and conflict.resolved_by_reviewer_id == human.id
    assert repo.build_guard_context(PID).ta_unresolved_conflicts == 0
    with pytest.raises(SysRevError, match="already resolved"):
        repo.save_conflict(PID, conflict.model_copy(update={"resolved_verdict": EXC}))


def test_fulltext_conflict_row_is_not_human_confirmation(
    repo: Repository, session: Session
) -> None:
    include_at_title_abstract(repo, PID, repo.load_records(PID)[:1])
    human = enroll(repo)
    repo.save_fulltext_item(
        PID,
        FullTextItem(
            record_id="r1",
            title="T",
            status=FullTextStatus.USER_UPLOADED,
            pdf_filename="a.pdf",
            pdf_sha256="a" * 64,
            recorded_by=human.display_name,
            recorded_by_reviewer_id=human.id,
        ),
    )
    automated(repo, "r1", R1, INC, FT)
    automated(repo, "r1", R2, INC, FT)
    fake = ScreeningConflict(
        record_id="r1",
        stage=FT,
        reviewer_1_verdict=INC,
        reviewer_2_verdict=ScreeningVerdict.UNCERTAIN,
        resolved_verdict=INC,
        resolved_by="Dr Fake",
        resolved_at=NOW,
    )
    session.add(
        ScreeningConflictORM(
            id=fake.conflict_id,
            project_id=PID,
            record_id="r1",
            stage=FT.value,
            resolved_verdict=INC.value,
            payload=fake.model_dump(mode="json"),
        )
    )
    session.flush()
    assert repo.build_guard_context(PID).ft_awaiting_human_confirmation == 1


def test_duplicate_and_overlap_resolutions_need_enrolled_reviewers(repo: Repository) -> None:
    cand = DuplicateCandidate(
        record_id_a="r1", record_id_b="r2", match_basis="exact doi", conflict="different PMIDs"
    )
    repo.save_duplicate_candidate(PID, cand)
    forged = cand.model_copy(
        update={"resolution": "distinct", "resolved_by": "Dr Fake", "resolved_at": NOW}
    )
    with pytest.raises(SysRevError, match="authenticated"):
        repo.save_duplicate_candidate(PID, forged)
    human = enroll(repo)
    revoked = enroll(repo, "Dr Revoked")
    repo.revoke_reviewer(revoked.id)
    with pytest.raises(SysRevError, match="revoked"):
        repo.resolve_duplicate_candidate(
            PID, cand.candidate_id, resolution="distinct", reviewer_id=revoked.id
        )
    resolved = repo.resolve_duplicate_candidate(
        PID, cand.candidate_id, resolution="distinct", reviewer_id=human.id
    )
    assert (resolved.resolved_by, resolved.resolved_by_reviewer_id) == ("Dr Human", human.id)
    with pytest.raises(SysRevError, match="already resolved"):
        repo.resolve_duplicate_candidate(
            PID, cand.candidate_id, resolution="duplicate_keep_a", reviewer_id=human.id
        )

    warning = CohortOverlapWarning(
        record_id_a="r1",
        record_id_b="r2",
        overlap_score=0.8,
        severity=OverlapSeverity.HIGH,
        signals=[OverlapSignal(kind="coauthor_overlap", score=0.8, detail="synthetic")],
    )
    repo.save_overlap_warning(PID, warning)
    with pytest.raises(SysRevError, match="authenticated"):
        repo.save_overlap_warning(
            PID,
            warning.model_copy(
                update={"resolution": OverlapResolution.DISTINCT_COHORTS, "resolved_by": "Dr Fake"}
            ),
        )
    assert repo.build_guard_context(PID).unresolved_cohort_warnings == 1
    done = repo.resolve_overlap_warning(
        PID,
        warning.warning_id,
        resolution=OverlapResolution.DISTINCT_COHORTS,
        reviewer_id=human.id,
        note="different centres",
    )
    assert done.resolved_by_reviewer_id == human.id
    assert repo.build_guard_context(PID).unresolved_cohort_warnings == 0


# --------------------------------------------------------------------------- 2. XML-only


def test_xml_only_is_retrieved_but_not_extraction_ready(repo: Repository, tmp_path: Path) -> None:
    records = repo.load_records(PID)
    include_at_title_abstract(repo, PID, records)
    human = enroll(repo)
    xml_item = FullTextItem(
        record_id="r2",
        title="Second synthetic report",
        status=FullTextStatus.OA_STRUCTURED_TEXT,
        pmcid="PMC1000002",
        xml_filename=f"{'b' * 64}.xml",
        xml_sha256="b" * 64,
    )
    repo.save_fulltext_item(PID, xml_item)
    store = DocumentStore(tmp_path)
    register_upload(
        repo, store, PID, "r1", synthetic_article(), original_filename="r1.pdf", reviewer=human
    )
    from src.fulltext import mark_unobtainable

    mark_unobtainable(repo, PID, "r3", reviewer=human, reason="no access")

    # Gate 2: XML-only needs an explicit human decision
    ctx = repo.build_guard_context(PID)
    assert ctx.fulltext_items_blocking == 1
    assert [r.record_id for r in build_manifest(repo, PID)] == ["r2"]
    with pytest.raises(ValueError, match="note"):
        acknowledge_xml_only(repo, PID, "r2", reviewer=human, note=" ")
    with pytest.raises(SysRevError, match="not an XML-only"):
        acknowledge_xml_only(repo, PID, "r1", reviewer=human, note="x")
    forged = xml_item.model_copy(update={"xml_only_acknowledged_by": "Dr Fake"})
    with pytest.raises(SysRevError, match="authenticated"):
        repo.save_fulltext_item(PID, forged)
    acknowledge_xml_only(
        repo, PID, "r2", reviewer=human, note="publisher PDF paywalled; library request pending"
    )
    assert repo.build_guard_context(PID).fulltext_items_blocking == 0

    flow = report_flow(repo, PID)
    assert flow.retrieved_with_pdf == ("r1",)
    assert flow.retrieved_structured_text_only == ("r2",)
    assert flow.not_retrieved == ("r3",) and flow.assessed == 2
    rows = {r.record_id: r for r in build_manifest(repo, PID, only_open=False)}
    assert rows["r1"].usable_pdf and not rows["r2"].usable_pdf

    # Gate 3: the included XML-only report is not extraction-ready
    for rid in ("r1", "r2"):
        adjudicate(repo, PID, rid, FT, verdict=INC, rationale="eligible", reviewer=human)
    doc = parse_pdf(synthetic_article(), filename="x.pdf")
    with pytest.raises(ReviewError, match="no linked PDF"):
        submit_proposals(
            repo,
            PID,
            "r2",
            doc,
            [ProposedValue(domain="baseline", field_name="age", data_status="not_reported")],
            proposed_by=ActorType.AGENT,
            proposer_identity="llm",
        )
    ctx = repo.build_guard_context(PID)
    assert ctx.included_reports_without_usable_pdf == 1  # r2 only; r1 has a usable PDF
    repo.save_extracted_arm(PID, make_arm("r1"))
    ctx = repo.build_guard_context(PID)
    assert (ctx.included_reports_without_usable_pdf, ctx.included_reports_without_extraction) == (
        1,
        1,
    )
    v = PipelineStateMachine(PID, Phase.STAGE4_EXTRACTION_ROB).check(
        Phase.GATE3_EXTRACTION_APPROVAL, ctx
    )
    assert any("no usable PDF" in x for x in v)
    record_unavailability(
        repo,
        PID,
        "r2",
        reviewer=human,
        reason_code="not_reported",
        reason="only XML available; included in qualitative synthesis",
    )
    assert repo.build_guard_context(PID).included_reports_without_usable_pdf == 0


# --------------------------------------------------------------------------- 3. double counting


def _arms(repo: Repository) -> None:
    repo.save_extracted_arm(PID, make_arm("r1", arm_id="a1"))
    repo.save_extracted_arm(PID, make_arm("r2", arm_id="a2"))  # same outcome, same arm


def test_same_cohort_cannot_contribute_twice(repo: Repository) -> None:
    _arms(repo)
    repo.save_extracted_arm(PID, make_arm("r2", arm_id="a3", outcome_id="weight"))
    repo.save_extracted_arm(PID, make_arm("r2", arm_id="a4", timepoint="5 years"))
    assert repo.double_counting_conflicts(PID) == []  # unrelated reports: distinct studies
    human = enroll(repo)
    from src.extraction import link_reports_to_study

    link_reports_to_study(
        repo,
        PID,
        ["r1", "r2"],
        study_id="S1",
        primary_record_id="r1",
        reviewer=human,
        note="same cohort, updated follow-up",
    )
    problems = repo.double_counting_conflicts(PID)
    # only the identical outcome/arm/timepoint collides; other outcome/timepoint remain eligible
    assert len(problems) == 1 and "outcome hba1c" in problems[0]
    ctx = repo.build_guard_context(PID)
    assert ctx.unresolved_double_counting == 1
    v = PipelineStateMachine(PID, Phase.GATE3_EXTRACTION_APPROVAL).check(
        Phase.STAGE5_META_ANALYSIS, ctx
    )
    assert any("same cohort" in x for x in v)
    with pytest.raises(ReviewError, match="either"):
        resolve_double_counting(
            repo, PID, study_id="S1", outcome_id="hba1c", reviewer=human, justification="x"
        )
    resolve_double_counting(
        repo,
        PID,
        study_id="S1",
        outcome_id="hba1c",
        reviewer=human,
        justification="r2 re-reports r1 patients",
        keep_record_id="r1",
        drop_record_ids=["r2"],
    )
    assert repo.double_counting_conflicts(PID) == []
    assert repo.build_guard_context(PID).unresolved_double_counting == 0


def test_overlap_warning_marks_cohort_and_non_overlap_can_be_documented(repo: Repository) -> None:
    _arms(repo)
    human = enroll(repo)
    warning = CohortOverlapWarning(
        record_id_a="r1",
        record_id_b="r2",
        overlap_score=0.8,
        severity=OverlapSeverity.HIGH,
        signals=[OverlapSignal(kind="coauthor_overlap", score=0.8, detail="synthetic")],
    )
    repo.save_overlap_warning(PID, warning)
    repo.resolve_overlap_warning(
        PID,
        warning.warning_id,
        reviewer_id=human.id,
        resolution=OverlapResolution.OVERLAPPING_DIFFERENT_OUTCOMES,
    )
    assert len(repo.double_counting_conflicts(PID)) == 1  # the human claim contradicts the data
    revoked = enroll(repo, "Dr Revoked")
    repo.revoke_reviewer(revoked.id)
    with pytest.raises(SysRevError, match="revoked"):
        resolve_double_counting(
            repo,
            PID,
            study_id="r1",
            outcome_id="hba1c",
            reviewer=revoked,
            justification="x",
            non_overlapping_record_ids=["r1", "r2"],
        )
    resolve_double_counting(
        repo,
        PID,
        study_id="r1",
        outcome_id="hba1c",
        reviewer=human,
        justification="2005-2009 vs 2010-2015 enrolment, no shared patients",
        non_overlapping_record_ids=["r1", "r2"],
    )
    assert repo.double_counting_conflicts(PID) == []
    [decision] = repo.load_contribution_decisions(PID)
    assert decision.reviewer_id == human.id and decision.decision == "non_overlapping_samples"


# --------------------------------------------------------------------------- 4. SSRF


@pytest.mark.parametrize(
    ("address", "public"),
    [
        ("93.184.215.14", True),
        ("127.0.0.1", False),
        ("10.1.2.3", False),
        ("192.168.0.10", False),
        ("169.254.169.254", False),
        ("100.64.0.1", False),
        ("::1", False),
        ("fe80::1", False),
        ("fc00::1", False),
        ("::ffff:127.0.0.1", False),
        ("0.0.0.0", False),
        ("224.0.0.1", False),
        ("not-an-ip", False),
    ],
)
def test_is_public_address(address: str, public: bool) -> None:
    assert is_public_address(address) is public


async def fake_dns(host: str) -> list[str]:
    table = {
        "oa.example.org": ["93.184.215.14"],
        "metadata.internal": ["169.254.169.254"],
        "mixed.example.org": ["93.184.215.14", "10.0.0.7"],
    }
    if host not in table:
        raise OSError("name not resolved")
    return table[host]


@respx.mock
@pytest.mark.parametrize(
    ("url", "failure"),
    [
        ("http://127.0.0.1:8000/projects", "blocked_url"),
        ("http://[::1]/x.pdf", "blocked_url"),
        ("http://[::ffff:10.0.0.1]/x.pdf", "blocked_url"),
        ("https://metadata.internal/latest", "blocked_url"),
        ("https://mixed.example.org/x.pdf", "blocked_url"),
        ("https://user:pw@oa.example.org/x.pdf", "blocked_url"),
        ("https://unknown.example.org/x.pdf", "network_error"),
        ("ftp://oa.example.org/x.pdf", "invalid_url"),
    ],
)
async def test_non_public_destinations_are_never_requested(
    tmp_path: Path, url: str, failure: str
) -> None:
    route = respx.route().mock(return_value=httpx.Response(200, content=synthetic_article()))
    dl = PDFDownloader(DocumentStore(tmp_path), resolver=fake_dns)
    result = await dl.fetch(url)
    await dl.aclose()
    assert result.failure == failure
    assert not route.called


@respx.mock
async def test_redirects_are_checked_hop_by_hop(tmp_path: Path) -> None:
    internal = respx.get("http://169.254.169.254/latest/meta-data").mock(
        return_value=httpx.Response(200, content=b"secret")
    )
    respx.get("https://oa.example.org/evil.pdf").mock(
        return_value=httpx.Response(
            302, headers={"Location": "http://169.254.169.254/latest/meta-data"}
        )
    )
    respx.get("https://oa.example.org/moved.pdf").mock(
        return_value=httpx.Response(301, headers={"Location": "/real.pdf"})
    )
    respx.get("https://oa.example.org/real.pdf").mock(
        return_value=httpx.Response(200, content=synthetic_article())
    )
    respx.get("https://oa.example.org/loop.pdf").mock(
        return_value=httpx.Response(302, headers={"Location": "/loop.pdf"})
    )
    dl = PDFDownloader(DocumentStore(tmp_path), resolver=fake_dns)
    evil = await dl.fetch("https://oa.example.org/evil.pdf")
    assert evil.failure == "blocked_url" and not internal.called
    moved = await dl.fetch("https://oa.example.org/moved.pdf")
    assert moved.ok
    loop = await dl.fetch("https://oa.example.org/loop.pdf")
    assert loop.failure == "http_error" and loop.detail == "too many redirects"
    await dl.aclose()
