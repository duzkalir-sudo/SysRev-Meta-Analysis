"""Session 2: risk-of-bias tool selection, evidence-checked LLM drafts, human approval.

Judgements below are SYNTHETIC test inputs, not appraisals of any real study.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError
from sqlalchemy.orm import Session

from src.core.enums import ActorType, ReviewDesign, RoBTool, StudyDesign
from src.core.exceptions import SysRevError
from src.core.schemas import DomainJudgement, RiskOfBiasAssessment
from src.core.schemas.extraction import ROB2_DOMAINS, ROBINS_I_DOMAINS
from src.extraction import LLMRoBDrafter, approve_rob, select_rob_tool
from src.llm import LLMResponseError
from src.pdf_extractor import parse_pdf
from src.pdf_extractor.models import ParsedDocument
from src.state.repository import Repository
from tests.conftest import NOW, make_record
from tests.fakes import ScriptedProvider, config, enroll
from tests.fixtures.pdf_factory import synthetic_article

PID = "p1"
REAL_QUOTE = "We retrospectively reviewed 57 patients treated between 2005 and 2015"


@pytest.fixture(scope="module")
def doc() -> ParsedDocument:
    return parse_pdf(synthetic_article(), filename="a.pdf")


@pytest.fixture
def repo(session: Session) -> Repository:
    r = Repository(session)
    r.create_project(PID, "Synthetic review")
    r.save_records(PID, [make_record("r1", "Synthetic cohort")])
    return r


def reply(domains: tuple[str, ...], **override: dict[str, Any]) -> dict[str, Any]:
    default = "moderate" if domains == ROBINS_I_DOMAINS else "low"
    out = []
    for d in domains:
        entry: dict[str, Any] = {
            "domain": d,
            "proposed_judgement": default,
            "support": "synthetic support",
            "quotes": [{"page": 1, "quote": REAL_QUOTE}],
        }
        out.append(entry | override.get(d, {}))
    return {"domains": out}


def test_tool_selection_matches_design() -> None:
    assert select_rob_tool(StudyDesign.RCT, ReviewDesign.COMPARATIVE).recommended == RoBTool.ROB2
    nrsi = select_rob_tool(StudyDesign.RETROSPECTIVE_COHORT, ReviewDesign.COMPARATIVE)
    assert nrsi.recommended == RoBTool.ROBINS_I and RoBTool.NOS in nrsi.allowed
    single = select_rob_tool(StudyDesign.RETROSPECTIVE_COHORT, ReviewDesign.SINGLE_ARM)
    assert single.recommended is None and single.allowed == {RoBTool.NOS, RoBTool.EXTERNAL}
    series = select_rob_tool(StudyDesign.CASE_SERIES, ReviewDesign.SINGLE_ARM)
    assert series.allowed == {RoBTool.EXTERNAL} and "not designed" in series.rationale


def test_llm_draft_keeps_only_verified_evidence_and_no_overall(doc: ParsedDocument) -> None:
    fabricated = {"quotes": [{"page": 1, "quote": "Outcome assessors were blinded."}]}
    missing = {"quotes": [], "missing_information": True, "proposed_judgement": "no_information"}
    provider = ScriptedProvider(
        "model-a",
        [reply(ROBINS_I_DOMAINS, confounding=fabricated, missing_data=missing)],
    )
    draft = LLMRoBDrafter(provider, config("rob")).draft(
        "r1", doc, design=StudyDesign.RETROSPECTIVE_COHORT, review_design=ReviewDesign.COMPARATIVE
    )
    assert draft.tool == RoBTool.ROBINS_I and draft.overall is None
    assert not draft.approved_by_human and draft.assessed_by == ActorType.AGENT
    by = {d.domain: d for d in draft.domains}
    assert by["confounding"].evidence == [] and by["confounding"].missing_information
    assert any("quote_not_found" in a for a in by["confounding"].needs_human_attention)
    assert by["missing_data"].missing_information
    assert by["selection_of_participants"].evidence[0].page_number == 1
    assert "Do not give an overall judgement" in provider.requests[0].system


def test_llm_draft_rejects_invalid_replies_and_non_comparative_designs(doc: ParsedDocument) -> None:
    bad = ScriptedProvider("model-a", [reply(ROB2_DOMAINS[:3])])
    with pytest.raises(LLMResponseError, match="exactly"):
        LLMRoBDrafter(bad, config("rob")).draft(
            "r1", doc, design=StudyDesign.RCT, review_design=ReviewDesign.COMPARATIVE
        )
    invalid = ScriptedProvider(
        "model-a", [reply(ROB2_DOMAINS, randomization_process={"proposed_judgement": "fine"})]
    )
    with pytest.raises(LLMResponseError, match="invalid judgement"):
        LLMRoBDrafter(invalid, config("rob")).draft(
            "r1", doc, design=StudyDesign.RCT, review_design=ReviewDesign.COMPARATIVE
        )
    with pytest.raises(SysRevError, match="human must perform"):
        LLMRoBDrafter(ScriptedProvider("model-a", []), config("rob")).draft(
            "r1", doc, design=StudyDesign.CASE_SERIES, review_design=ReviewDesign.SINGLE_ARM
        )


def test_agents_cannot_certify_low_risk() -> None:
    domains = [DomainJudgement(domain=d, judgement="low", support="s") for d in ROB2_DOMAINS]
    with pytest.raises(ValidationError, match="cannot set an overall"):
        RiskOfBiasAssessment(
            record_id="r1",
            tool=RoBTool.ROB2,
            domains=domains,
            overall="low",
            assessor="llm",
            assessed_by=ActorType.AGENT,
        )
    with pytest.raises(ValidationError, match="supporting evidence or an explicit"):
        RiskOfBiasAssessment(
            record_id="r1",
            tool=RoBTool.ROB2,
            domains=domains,
            assessor="llm",
            assessed_by=ActorType.AGENT,
        )
    with pytest.raises(ValidationError, match="approval requires"):
        RiskOfBiasAssessment(
            record_id="r1",
            tool=RoBTool.ROB2,
            domains=domains,
            overall="low",
            assessor="llm",
            assessed_by=ActorType.AGENT,
            approved_by_human=True,
        )
    with pytest.raises(ValidationError, match="humans only"):
        RiskOfBiasAssessment(
            record_id="r1",
            tool=RoBTool.EXTERNAL,
            assessor="llm",
            assessed_by=ActorType.AGENT,
            tool_rationale="MINORS",
        )


def test_draft_versus_approved_status_and_gate(repo: Repository, doc: ParsedDocument) -> None:
    provider = ScriptedProvider("model-a", [reply(ROB2_DOMAINS)])
    draft = LLMRoBDrafter(provider, config("rob")).draft(
        "r1", doc, design=StudyDesign.RCT, review_design=ReviewDesign.COMPARATIVE
    )
    repo.save_rob(PID, draft)
    stored = repo.get_rob(PID, draft.assessment_id)
    assert not stored.approved_by_human and stored.overall is None
    forged = RiskOfBiasAssessment.model_validate(
        stored.model_dump()
        | {
            "approved_by_human": True,
            "overall": "low",
            "approved_by": "Dr X",
            "approved_reviewer_id": "0" * 32,
            "approved_at": NOW,
        }
    )
    with pytest.raises(SysRevError, match="not enrolled"):
        repo.save_rob(PID, forged)
    reviewer = enroll(repo)
    with pytest.raises(ValidationError, match="less severe"):
        approve_rob(
            repo,
            PID,
            draft.assessment_id,
            reviewer=reviewer,
            overall="low",
            domain_judgements={"missing_outcome_data": ("high", "40% lost")},
        )
    approved = approve_rob(
        repo,
        PID,
        draft.assessment_id,
        reviewer=reviewer,
        overall="some_concerns",
        domain_judgements={"measurement_of_the_outcome": ("some_concerns", "unblinded assessors")},
        note="reviewed against full text",
    )
    assert approved.approved_by_human and approved.approved_by == "Dr Human"
    assert approved.approved_reviewer_id == reviewer.id
    with pytest.raises(SysRevError, match="already approved"):
        approve_rob(repo, PID, draft.assessment_id, reviewer=reviewer, overall="high")
    revoked = enroll(repo, "Dr Gone")
    repo.revoke_reviewer(revoked.id)
    second = LLMRoBDrafter(ScriptedProvider("model-a", [reply(ROB2_DOMAINS)]), config("rob")).draft(
        "r1", doc, design=StudyDesign.RCT, review_design=ReviewDesign.COMPARATIVE
    )
    repo.save_rob(PID, second)
    with pytest.raises(SysRevError, match="revoked"):
        approve_rob(repo, PID, second.assessment_id, reviewer=revoked, overall="low")
