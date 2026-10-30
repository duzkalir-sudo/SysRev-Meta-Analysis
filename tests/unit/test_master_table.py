"""Session 2: source-grounded extraction and the human-reviewed Master Extraction Table.

Every number below comes from the SYNTHETIC fixture PDF and describes no real study.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from sqlalchemy.orm import Session

from src.core.enums import (
    ActorType,
    ArmRole,
    ComparisonType,
    DataStatus,
    EstimateType,
    EvidenceCheckStatus,
    EvidenceLocationType,
    ExtractionDomain,
    Phase,
    ReviewStatus,
    ScreeningStage,
    ScreeningVerdict,
    TimepointType,
)
from src.core.exceptions import SysRevError
from src.core.schemas import FollowUpTimepoint, OutcomeDefinition
from src.core.schemas.master import ExtractionItem, ExtractionReviewEvent
from src.extraction import (
    LLMExtractor,
    ProposedValue,
    ReviewError,
    accept_item,
    analysis_compatibility,
    build_arms,
    correct_item,
    double_counting_risks,
    export_csv,
    export_json,
    export_rows,
    link_reports_to_study,
    record_unavailability,
    reject_item,
    reverify_items,
    submit_proposals,
)
from src.fulltext import DocumentStore, confirm_pdf_match, register_upload
from src.pdf_extractor import parse_pdf
from src.pdf_extractor.models import ParsedDocument
from src.screening.llm_screening import adjudicate
from src.state.db import ReviewerORM
from src.state.machine import PipelineStateMachine
from src.state.repository import Repository
from tests.conftest import approved_rob, make_record
from tests.fakes import (
    ScriptedProvider,
    approved_protocol,
    config,
    enroll,
    include_at_title_abstract,
)
from tests.fixtures.pdf_factory import SYNTHETIC_DOI, SYNTHETIC_TITLE, synthetic_article

PID = "p1"
PDF = synthetic_article()
OTHER_PDF = synthetic_article(title="Another synthetic report", doi="10.5555/other.0002")
OUTCOME = "obliteration"
TABLE = EvidenceLocationType.TABLE
KM = EstimateType.KAPLAN_MEIER


def fixed(years: float) -> FollowUpTimepoint:
    return FollowUpTimepoint(kind=TimepointType.FIXED, value=years, unit="years")


def arm_value(field: str, text: str, quote: str, page: int = 2, **kw: Any) -> ProposedValue:
    data: dict[str, Any] = {
        "domain": ExtractionDomain.OUTCOME_DATA,
        "field_name": field,
        "outcome_id": OUTCOME,
        "arm_label": "SRS cohort",
        "arm_role": ArmRole.SINGLE_GROUP,
        "data_status": "reported",
        "reported_text": text,
        "page": page,
        "quote": quote,
    } | kw
    return ProposedValue(**data)


def km(field: str, text: str, row: int, col: int, years: int) -> ProposedValue:
    quote = {1: "Obliteration (KM, %) 62.1 81.4", 2: "Number at risk 40 22"}[row]
    extra = {"estimate_scale": "percent"} if field == "reported_estimate" else {}
    return arm_value(
        field,
        text,
        quote,
        location_type=TABLE,
        location_id=f"p2.t1.r{row}c{col}",
        timepoint=f"{years} years",
        follow_up=fixed(years),
        estimate_type=KM,
        **extra,
    )


CRUDE_QUOTE = "Complete obliteration was achieved in 45 of 57 patients (78.9%)."


def crude(field: str, text: str) -> ProposedValue:
    return arm_value(
        field,
        text,
        CRUDE_QUOTE,
        timepoint="last follow-up",
        follow_up=FollowUpTimepoint(kind=TimepointType.LAST_FOLLOW_UP),
        location_id="p2.b2",
    )


class Project:
    def __init__(self, repo: Repository, store: DocumentStore, reviewer: ReviewerORM) -> None:
        self.repo, self.store, self.reviewer = repo, store, reviewer

    def doc(self, record_id: str) -> ParsedDocument:
        item = self.repo.get_fulltext_item(PID, record_id)
        assert item is not None and item.pdf_filename and item.pdf_sha256
        data = self.store.read_verified(item.pdf_filename, item.pdf_sha256)
        return parse_pdf(data, filename=item.pdf_filename, expected_sha256=item.pdf_sha256)

    def submit(self, *proposals: ProposedValue, record_id: str = "r1") -> list[ExtractionItem]:
        return submit_proposals(
            self.repo,
            PID,
            record_id,
            self.doc(record_id),
            proposals,
            proposed_by=ActorType.AGENT,
            proposer_identity="llm:test",
        )

    def accept(self, item: ExtractionItem, note: str | None = None) -> ExtractionItem:
        return accept_item(self.repo, PID, item.item_id, reviewer=self.reviewer, note=note)


@pytest.fixture
def project(session: Session, tmp_path: Path) -> Project:
    repo = Repository(session)
    repo.create_project(PID, "Synthetic review")
    repo.save_protocol(PID, approved_protocol())
    recs = [
        make_record("r1", SYNTHETIC_TITLE, doi=SYNTHETIC_DOI),
        make_record("r2", "Synthetic follow-up report of the same cohort"),
        make_record("r3", "Another synthetic report", doi="10.5555/other.0002"),
    ]
    repo.save_records(PID, recs)
    include_at_title_abstract(repo, PID, recs)
    store = DocumentStore(tmp_path)
    reviewer = enroll(repo)
    for rid, pdf in (("r1", PDF), ("r2", PDF), ("r3", OTHER_PDF)):
        item = register_upload(
            repo, store, PID, rid, pdf, original_filename=f"{rid}.pdf", reviewer=reviewer
        )
        if item.match_unconfirmed:
            confirm_pdf_match(repo, PID, rid, reviewer=reviewer, note="checked manually")
        adjudicate(
            repo,
            PID,
            rid,
            ScreeningStage.FULL_TEXT,
            verdict=ScreeningVerdict.INCLUDE,
            rationale="eligible",
            reviewer=reviewer,
        )
    return Project(repo, store, reviewer)


# --------------------------------------------------------------------------- schema rules


def test_missing_data_is_never_zero_and_derived_is_refused() -> None:
    base: dict[str, Any] = {
        "record_id": "r1",
        "domain": ExtractionDomain.BASELINE,
        "field_name": "median_age",
        "proposed_by": ActorType.AGENT,
        "proposer_identity": "x",
    }
    missing = ExtractionItem(**base, data_status=DataStatus.NOT_REPORTED)
    assert missing.value is None
    with pytest.raises(ValidationError, match="never zero"):
        ExtractionItem(**base, data_status=DataStatus.NOT_REPORTED, value=0)
    with pytest.raises(ValidationError, match="not authorised"):
        ExtractionItem(**base, data_status=DataStatus.DERIVED)
    with pytest.raises(ValidationError, match="SourceEvidence"):
        ExtractionItem(**base, data_status=DataStatus.REPORTED, reported_text="54", value=54)


def test_unauthenticated_review_events_are_refused() -> None:
    with pytest.raises(ValidationError, match="authenticated human"):
        ExtractionReviewEvent(
            item_id="i", action="accepted", actor_type=ActorType.AGENT, actor="llm"
        )


# --------------------------------------------------------------------------- verification


def test_proposals_are_verified_against_the_document(project: Project) -> None:
    items = project.submit(
        km("reported_estimate", "62.1", 1, 1, 3),
        arm_value("events", "52", "Obliteration was achieved in 52 of 57 patients"),  # fabricated
        arm_value("events", "45", "45 of 57 patients", page=1),  # wrong page
        arm_value("events", "4", "45 of 57 patients"),  # number not a token of the quote
        arm_value("events", "62.1", "The actuarial obliteration rate was 62.1% at 3 years"),
        ProposedValue(
            domain=ExtractionDomain.BASELINE,
            field_name="median_age",
            data_status="not_reported",
            note="age not reported",
        ),
        arm_value("events", "45", CRUDE_QUOTE, data_status="ambiguous"),
    )
    status = [(i.data_status, i.verification.status if i.verification else None) for i in items]
    assert status[0] == (DataStatus.REPORTED, EvidenceCheckStatus.VERIFIED)
    assert status[1][1] == EvidenceCheckStatus.QUOTE_NOT_FOUND
    assert status[2][1] == EvidenceCheckStatus.WRONG_PAGE
    assert status[3][0] == DataStatus.AMBIGUOUS and items[3].value is None
    assert any("Kaplan-Meier/actuarial" in f for f in items[4].flags)
    assert items[5].data_status == DataStatus.NOT_REPORTED and items[5].value is None
    assert items[6].data_status == DataStatus.AMBIGUOUS and items[6].value is None
    ev = items[0].evidence
    assert ev is not None and ev.pdf_sha256 == project.doc("r1").pdf_sha256 and ev.page_number == 2
    with pytest.raises(ReviewError, match="did not verify"):
        project.accept(items[1])
    with pytest.raises(ReviewError, match="did not verify"):
        project.accept(items[2], note="looks right to me")  # a note cannot override grounding
    with pytest.raises(ReviewError, match="flagged items"):
        project.accept(items[4])  # verified quote, but estimator contradicts the field
    assert project.accept(items[5]).review_status == ReviewStatus.ACCEPTED  # confirmed missing
    events = project.repo.list_review_events(PID, items[0].item_id)
    assert [e.action for e in events] == ["proposed"]


def test_extraction_refuses_a_document_that_is_not_the_linked_pdf(project: Project) -> None:
    other = parse_pdf(OTHER_PDF, filename="r1.pdf")
    with pytest.raises(ReviewError, match="not the PDF linked"):
        submit_proposals(
            project.repo,
            PID,
            "r1",
            other,
            [km("reported_estimate", "62.1", 1, 1, 3)],
            proposed_by=ActorType.AGENT,
            proposer_identity="x",
        )


# --------------------------------------------------------------------------- review and arms


def test_multiple_timepoints_crude_vs_actuarial_and_gate3(project: Project) -> None:
    repo = project.repo
    items = project.submit(
        km("reported_estimate", "62.1", 1, 1, 3),
        km("n_at_risk", "40", 2, 1, 3),
        km("reported_estimate", "81.4", 1, 2, 5),
        km("n_at_risk", "22", 2, 2, 5),
        crude("events", "45"),
        crude("n_analyzed", "57"),
        ProposedValue(
            domain=ExtractionDomain.FOLLOW_UP,
            field_name="median_follow_up",
            data_status="reported",
            reported_text="48",
            unit="months",
            page=1,
            quote="Median follow-up was 48 months (IQR 30-72).",
        ),
        ProposedValue(
            domain=ExtractionDomain.STUDY_CHARACTERISTICS,
            field_name="institution",
            value_kind="text",
            data_status="reported",
            text_value="Synthetic Institute",
            page=1,
            quote="at the Synthetic Institute.",
        ),
    )
    assert all(i.verification and i.verification.acceptable for i in items)
    ctx = repo.build_guard_context(PID)
    assert ctx.extraction_items_pending == len(items)
    with pytest.raises(ReviewError, match="pending"):
        build_arms(repo, PID, "r1", reviewer=project.reviewer)
    for item in items:
        project.accept(item)
    arms = build_arms(repo, PID, "r1", reviewer=project.reviewer)
    assert len(arms) == 3 and all(a.approved_by_human for a in arms)
    by_tp = {a.timepoint: a for a in arms}
    assert by_tp["3 years"].estimate_type == KM and by_tp["3 years"].n_at_risk is not None
    assert by_tp["5 years"].reported_estimate is not None
    assert by_tp["5 years"].reported_estimate.value == 81.4
    crude_arm = by_tp["last follow-up"]
    assert crude_arm.estimate_type is None and crude_arm.events is not None
    assert build_arms(repo, PID, "r1", reviewer=project.reviewer) == []  # idempotent

    prop = OutcomeDefinition(
        outcome_id=OUTCOME,
        name="Obliteration",
        kind="primary",
        data_type="dichotomous",
        comparison=ComparisonType.SINGLE_GROUP,
        preferred_effect_measure="PROPORTION",
        timepoints=[FollowUpTimepoint(kind=TimepointType.LAST_FOLLOW_UP)],
    )
    km_outcome = OutcomeDefinition(
        outcome_id=OUTCOME,
        name="Obliteration",
        kind="primary",
        data_type="time_to_event",
        comparison=ComparisonType.SINGLE_GROUP,
        preferred_effect_measure="KM_PROBABILITY",
        timepoints=[fixed(3), fixed(5)],
    )
    assert analysis_compatibility(crude_arm, prop)[0]
    assert not analysis_compatibility(by_tp["5 years"], prop)[0]
    assert analysis_compatibility(by_tp["5 years"], km_outcome)[0]
    assert not analysis_compatibility(crude_arm, km_outcome)[0]

    ctx = repo.build_guard_context(PID)
    assert ctx.extraction_items_pending == 0
    # r2 and r3 are included but have no data yet; nobody has approved risk of bias
    assert ctx.included_reports_without_extraction == 2
    assert ctx.included_studies_without_approved_rob == 3


def test_reject_correct_attest_and_history(project: Project) -> None:
    repo = project.repo
    fabricated, wrong_page, figure = project.submit(
        arm_value("events", "52", "Obliteration was achieved in 52 of 57 patients"),
        arm_value("events", "45", CRUDE_QUOTE, page=1),
        arm_value("events", "3", "curve", page=3),
    )
    with pytest.raises(ReviewError, match="reason"):
        reject_item(repo, PID, fabricated.item_id, reviewer=project.reviewer, note=" ")
    reject_item(
        repo,
        PID,
        fabricated.item_id,
        reviewer=project.reviewer,
        note="quote does not exist in the PDF",
    )
    fixed_item = correct_item(
        repo,
        PID,
        wrong_page.item_id,
        crude("events", "45"),
        project.doc("r1"),
        reviewer=project.reviewer,
        note="page corrected to 2",
    )
    assert fixed_item.version == 2 and fixed_item.supersedes == wrong_page.item_id
    assert fixed_item.review_status == ReviewStatus.ACCEPTED
    assert fixed_item.proposed_by == ActorType.HUMAN
    old = repo.get_extraction_item(PID, wrong_page.item_id)
    assert old.review_status == ReviewStatus.SUPERSEDED
    with pytest.raises(SysRevError, match="already superseded"):
        repo.save_extraction_item(PID, old.model_copy(update={"value": None}))
    # figure-only value on an image-only page: must be attested explicitly by the human
    figure_value = arm_value("events", "3", "3 haemorrhages (read from figure)", page=3)
    with pytest.raises(ReviewError, match="page_unreadable"):
        correct_item(
            repo,
            PID,
            figure.item_id,
            figure_value,
            project.doc("r1"),
            reviewer=project.reviewer,
            note="read from figure",
        )
    attested = correct_item(
        repo,
        PID,
        figure.item_id,
        figure_value,
        project.doc("r1"),
        reviewer=project.reviewer,
        note="read from figure",
        attest_unverifiable=True,
    )
    assert attested.verification is not None
    assert attested.verification.status == EvidenceCheckStatus.HUMAN_ATTESTED
    assert "page_unreadable" in attested.verification.detail
    actions = [e.action for e in repo.list_review_events(PID, wrong_page.item_id)]
    assert actions == ["proposed", "corrected"]
    assert repo.list_review_events(PID, fixed_item.item_id)[0].reviewer_id == project.reviewer.id


def test_revoked_reviewer_cannot_approve(project: Project) -> None:
    (item,) = project.submit(km("reported_estimate", "62.1", 1, 1, 3))
    rogue = enroll(project.repo, "Dr Rogue")
    project.repo.revoke_reviewer(rogue.id)
    with pytest.raises(SysRevError, match="revoked"):
        accept_item(project.repo, PID, item.item_id, reviewer=rogue)
    forged = item.model_copy(
        update={
            "review_status": ReviewStatus.ACCEPTED,
            "reviewed_by": "Dr Rogue",
            "reviewer_id": "0" * 32,
            "reviewed_at": item.proposed_at,
        }
    )
    with pytest.raises(SysRevError, match="not enrolled"):
        project.repo.save_extraction_item(PID, forged)


def test_reverification_detects_replaced_pdf(project: Project) -> None:
    project.submit(km("reported_estimate", "62.1", 1, 1, 3))
    assert reverify_items(project.repo, PID, "r1", project.doc("r1")) == []
    replaced = parse_pdf(OTHER_PDF, filename=project.doc("r1").pdf_filename)
    failures = reverify_items(project.repo, PID, "r1", replaced)
    assert failures and failures[0][1].status == EvidenceCheckStatus.HASH_MISMATCH


# --------------------------------------------------------------------------- studies, export


def test_overlapping_reports_qualitative_only_and_export(project: Project) -> None:
    repo = project.repo
    for rid in ("r1", "r2"):
        for item in project.submit(km("reported_estimate", "62.1", 1, 1, 3), record_id=rid):
            project.accept(item)
        build_arms(repo, PID, rid, reviewer=project.reviewer)
    assert double_counting_risks(repo, PID) == []  # unlinked: two separate studies
    link_reports_to_study(
        repo,
        PID,
        ["r1", "r2"],
        study_id="S1",
        primary_record_id="r1",
        reviewer=project.reviewer,
        note="same cohort, later follow-up",
    )
    risks = double_counting_risks(repo, PID)
    assert len(risks) == 1 and "study S1" in risks[0] and "['r1', 'r2']" in risks[0]

    assert repo.build_guard_context(PID).included_reports_without_extraction == 1  # r3
    record_unavailability(
        repo,
        PID,
        "r3",
        reviewer=project.reviewer,
        reason_code="not_reported",
        reason="outcome only described narratively",
    )
    ctx = repo.build_guard_context(PID)
    assert ctx.included_reports_without_extraction == 0
    assert ctx.included_studies_without_approved_rob == 2  # S1 and r3
    repo.save_rob(PID, approved_rob("r2", project.reviewer.id, "Dr Human"))  # covers study S1
    assert repo.build_guard_context(PID).included_studies_without_approved_rob == 1

    rows = export_rows(repo, PID)
    parsed = list(csv.DictReader(io.StringIO(export_csv(rows))))
    estimate = next(r for r in parsed if r["record_id"] == "r1")
    assert estimate["study_id"] == "S1" and estimate["page"] == "2"
    assert estimate["pdf_sha256"] == project.doc("r1").pdf_sha256
    assert estimate["verbatim_quote"] == "Obliteration (KM, %) 62.1 81.4"
    assert estimate["estimate_type"] == "kaplan_meier" and estimate["reported_text"] == "62.1"
    qual = next(r for r in parsed if r["record_id"] == "r3")
    assert qual["field_name"] == "unavailable_for_quantitative_synthesis"
    assert qual["value"] == "" and "not_reported" in qual["text_value"]
    assert len(json.loads(export_json(rows))) == len(rows)
    v = PipelineStateMachine(PID, Phase.GATE3_EXTRACTION_APPROVAL).check(
        Phase.STAGE5_META_ANALYSIS, repo.build_guard_context(PID)
    )
    assert any("risk-of-bias" in x for x in v)


# --------------------------------------------------------------------------- LLM extractor


def test_llm_extractor_sends_only_relevant_context(project: Project) -> None:
    reply = {"values": [km("reported_estimate", "62.1", 1, 1, 3).model_dump(mode="json")]}
    provider = ScriptedProvider("model-a", [reply])
    extractor = LLMExtractor(provider, config("extractor"))
    doc = project.doc("r1")
    proposals = extractor.propose(approved_protocol(), doc, max_chars=600)
    assert proposals[0].reported_text == "62.1"
    prompt = provider.requests[0].prompt
    assert "<<page 2 | p2.t1 | table>>" in prompt
    assert len(prompt) < len(doc.full_text) + 1500
    (item,) = submit_proposals(
        project.repo,
        PID,
        "r1",
        doc,
        proposals,
        proposed_by=ActorType.AGENT,
        proposer_identity=extractor.identity,
    )
    assert item.evidence is not None and item.evidence.extractor_identity.startswith("llm:")
    bad = LLMExtractor(ScriptedProvider("model-a", [{"oops": []}]), config("extractor"))
    with pytest.raises(ValueError, match="values"):
        bad.propose(approved_protocol(), doc)
