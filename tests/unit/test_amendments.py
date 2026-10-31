"""Session 8: protocol amendments after Gate 1. All data SYNTHETIC; no network calls.

The approved protocol is never unlocked: a proposal is stored separately, approval by an
enrolled reviewer activates version n+1 (version n stays stored, superseded), affected work
becomes pending re-review, and revised verdicts are new human decisions — earlier rows are
never rewritten.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.core.enums import (
    ActorType,
    ArmRole,
    DataStatus,
    ExtractionDomain,
    Phase,
    ReviewerRole,
    ScreeningStage,
    ScreeningVerdict,
)
from src.core.schemas import ScreeningDecision
from src.core.schemas.master import ExtractionItem
from src.core.schemas.stats import AnalysisSpec
from src.manuscript.service import ManuscriptService
from src.screening.agreement import screening_status
from src.search import amendments
from src.search import protocol_editor as editor
from src.search.amendments import AmendmentError
from src.search.prisma_sync import derive_prisma
from src.state.db import ReviewerORM, ScreeningDecisionORM
from src.state.machine import GUARDS
from src.state.repository import Repository
from src.stats_engine import plan_service
from tests.fakes import enroll
from tests.manuscript_helpers import comparative_project
from tests.search_helpers import MeSHAlwaysFound, approved_in_stage2, draft_data

PID = "a8"


@pytest.fixture
def repo(session: Session) -> Repository:
    r = Repository(session)
    r.create_project(PID, "Synthetic amendment review")
    return r


@pytest.fixture
def human(repo: Repository) -> ReviewerORM:
    return enroll(repo, "Dr Amend")


def _approved(repo: Repository, pid: str = PID) -> tuple[dict[str, Any], str]:
    protocol = repo.latest_protocol(pid)
    assert protocol is not None and protocol.status == "approved"
    return protocol.model_dump(mode="json"), editor.protocol_sha256(protocol)


def _screen(repo: Repository, record_id: str, verdict: ScreeningVerdict) -> None:
    for role in (ReviewerRole.REVIEWER_1, ReviewerRole.REVIEWER_2):
        repo.save_screening_decision(
            PID,
            ScreeningDecision(
                record_id=record_id,
                reviewer=role,
                reviewer_identity=f"llm-{role}",
                stage=ScreeningStage.TITLE_ABSTRACT,
                verdict=verdict,
                rationale="synthetic",
                exclusion_reason="E1: Animal studies"
                if verdict == ScreeningVerdict.EXCLUDE
                else None,
            ),
        )


def _records(repo: Repository) -> None:
    from tests.conftest import make_record

    repo.save_records(
        PID, [make_record("ra", "Synthetic trial A"), make_record("rb", "Synthetic trial B")]
    )
    _screen(repo, "ra", ScreeningVerdict.INCLUDE)
    _screen(repo, "rb", ScreeningVerdict.EXCLUDE)


# --------------------------------------------------------------------------- lifecycle
def test_before_gate1_the_protocol_is_drafted_not_amended(repo: Repository, human: Any) -> None:
    editor.save_draft(repo, PID, draft_data(), reviewer=human)
    with pytest.raises(AmendmentError, match="edited as a draft"):
        amendments.propose(
            repo, PID, draft_data(), rationale="x", reviewer=human, expected_base_sha256="0" * 64
        )


def test_proposal_does_not_change_the_approved_protocol_until_approved(
    repo: Repository, human: Any
) -> None:
    approved_in_stage2(repo, PID, human)
    data, sha = _approved(repo)
    with pytest.raises(AmendmentError, match="changed since it was displayed"):
        amendments.propose(
            repo,
            PID,
            data | {"exclusion_criteria": ["Animal studies", "Paediatric studies"]},
            rationale="x",
            reviewer=human,
            expected_base_sha256="0" * 64,
        )
    proposal = amendments.propose(
        repo,
        PID,
        data | {"exclusion_criteria": ["Animal studies", "Paediatric studies"]},
        rationale="the target population is adults only",
        reviewer=human,
        expected_base_sha256=sha,
    )
    assert proposal.material_changes == {"eligibility": ["exclusion_criteria"]}
    assert "title_abstract_screening" in proposal.impact[0].affected
    current = repo.latest_protocol(PID)
    assert current is not None and current.version == 1 and current.status == "approved"
    # an undecided amendment blocks every forward transition
    ctx = repo.build_guard_context(PID)
    edge = (Phase.STAGE2_SEARCH_SCREENING, Phase.GATE1_5_SCREENING_CONFLICTS)
    assert any("awaits a decision" in v for v in GUARDS[edge](ctx))
    with pytest.raises(AmendmentError, match="another amendment is open"):
        amendments.propose(repo, PID, data, rationale="x", reviewer=human, expected_base_sha256=sha)
    with pytest.raises(AmendmentError, match="has changed"):  # an older revision displayed
        amendments.decide(
            repo,
            PID,
            proposal.amendment_id,
            "approved",
            reviewer=human,
            comments="ok",
            expected_sha256="0" * 64,
        )
    decision = amendments.decide(
        repo,
        PID,
        proposal.amendment_id,
        "approved",
        reviewer=human,
        comments="agreed by the review team",
        expected_sha256=proposal.sha256,
    )
    assert decision.protocol_version == 2
    versions = [(p.version, p.status) for p in repo.list_protocols(PID)]
    assert versions == [(1, "superseded"), (2, "approved")]  # version 1 preserved
    events = [e.action for e, _ in repo.list_protocol_events(PID)]
    assert events[-1] == "amendment_approved"
    assert not repo.build_guard_context(PID).amendment_problems  # nothing was screened yet


def test_withdrawn_and_rejected_amendments_change_nothing(repo: Repository, human: Any) -> None:
    approved_in_stage2(repo, PID, human)
    data, sha = _approved(repo)
    for decision in ("withdrawn", "rejected"):
        p = amendments.propose(
            repo,
            PID,
            data | {"population": f"Adults ({decision})"},
            rationale="x",
            reviewer=human,
            expected_base_sha256=sha,
        )
        amendments.decide(
            repo,
            PID,
            p.amendment_id,
            decision,
            reviewer=human,  # type: ignore[arg-type]
            comments="not needed",
            expected_sha256=p.sha256,
        )
    assert [p.version for p in repo.list_protocols(PID)] == [1]
    assert len(amendments.proposals(repo, PID)) == 2  # history kept


def test_new_mesh_terms_must_be_verified_before_approval(repo: Repository, human: Any) -> None:
    approved_in_stage2(repo, PID, human)
    data, sha = _approved(repo)
    data["mesh_terms"] = [*data["mesh_terms"], {"descriptor": "Hypoglycemia"}]
    data["population"] = "Adults with condition Y and hypoglycaemia risk"
    p = amendments.propose(
        repo, PID, data, rationale="add a MeSH heading", reviewer=human, expected_base_sha256=sha
    )
    with pytest.raises(AmendmentError, match="cannot be approved"):
        amendments.decide(
            repo,
            PID,
            p.amendment_id,
            "approved",
            reviewer=human,
            comments="ok",
            expected_sha256=p.sha256,
        )
    revised = asyncio.run(
        amendments.verify_mesh(
            repo, PID, p.amendment_id, MeSHAlwaysFound(), reviewer=human, expected_sha256=p.sha256
        )
    )
    assert revised.revision == 2 and all(m.verified for m in revised.protocol.mesh_terms)
    amendments.decide(
        repo,
        PID,
        p.amendment_id,
        "approved",
        reviewer=human,
        comments="ok",
        expected_sha256=revised.sha256,
    )
    assert repo.latest_protocol(PID).version == 2  # type: ignore[union-attr]


# --------------------------------------------------------------------------- re-review
def test_eligibility_amendment_requires_rereview_without_rewriting_decisions(
    repo: Repository, human: Any
) -> None:
    approved_in_stage2(repo, PID, human)
    _records(repo)
    data, sha = _approved(repo)
    p = amendments.propose(
        repo,
        PID,
        data | {"exclusion_criteria": ["Paediatric studies"]},
        rationale="animal studies no longer excluded by design",
        reviewer=human,
        expected_base_sha256=sha,
    )
    amendments.decide(
        repo,
        PID,
        p.amendment_id,
        "approved",
        reviewer=human,
        comments="ok",
        expected_sha256=p.sha256,
    )
    rows = {o["subject"]: o for o in amendments.obligation_status(repo, PID)}
    assert set(rows) == {"ra", "rb"} and {o["state"] for o in rows.values()} == {"open"}
    ctx = repo.build_guard_context(PID)
    assert any("2 title abstract screening item(s)" in v for v in ctx.amendment_problems)
    stored = repo.session.scalar(select(func.count()).select_from(ScreeningDecisionORM))
    with pytest.raises(AmendmentError, match="unchanged"):
        amendments.resolve(
            repo,
            PID,
            rows["rb"]["obligation_id"],
            "revised",
            reviewer=human,
            note="x",
            verdict=ScreeningVerdict.EXCLUDE,
            exclusion_reason="E1",
        )
    amendments.resolve(
        repo,
        PID,
        rows["ra"]["obligation_id"],
        "confirmed",
        reviewer=human,
        note="still eligible under version 2",
    )
    amendments.resolve(
        repo,
        PID,
        rows["rb"]["obligation_id"],
        "revised",
        reviewer=human,
        note="the exclusion relied on criterion E1, which was removed",
        verdict=ScreeningVerdict.INCLUDE,
    )
    with pytest.raises(AmendmentError, match="has changed"):  # already resolved elsewhere
        amendments.resolve(
            repo, PID, rows["ra"]["obligation_id"], "confirmed", reviewer=human, note="again"
        )
    # the revision is the effective human decision; the stored rows are unchanged
    decisions = repo.load_screening_decisions(PID)
    status = screening_status({"ra", "rb"}, decisions, [], ScreeningStage.TITLE_ABSTRACT)
    assert status.final == {"ra": ScreeningVerdict.INCLUDE, "rb": ScreeningVerdict.INCLUDE}
    assert repo.session.scalar(select(func.count()).select_from(ScreeningDecisionORM)) == stored
    original = [d for d in decisions if d.record_id == "rb" and d.reviewer != "adjudicator"]
    assert {d.verdict for d in original} == {ScreeningVerdict.EXCLUDE}
    assert derive_prisma(repo, PID).records_excluded == 0
    assert not repo.build_guard_context(PID).amendment_problems


def test_phase_rules_are_explicit(session: Session) -> None:
    repo, rid, _ = comparative_project(session)  # Stage 6
    reviewer = repo.require_active_reviewer(rid)
    data, sha = _approved(repo, "mc1")
    with pytest.raises(AmendmentError, match="cannot be repeated after Gate 3"):
        amendments.propose(
            repo,
            "mc1",
            data | {"population": "Older adults"},
            rationale="x",
            reviewer=reviewer,
            expected_base_sha256=sha,
        )
    strategies = [
        *data["search_strategies"],
        {"database": "europe_pmc", "query": "drug x AND placebo"},
    ]
    with pytest.raises(AmendmentError, match="Stage 2 only"):
        amendments.propose(
            repo,
            "mc1",
            data
            | {"search_strategies": strategies, "selected_databases": ["pubmed", "europe_pmc"]},
            rationale="x",
            reviewer=reviewer,
            expected_base_sha256=sha,
        )
    # administrative amendments remain possible and invalidate the evidence fingerprint
    before = ManuscriptService(repo, "mc1").package().fingerprint_sha256
    p = amendments.propose(
        repo,
        "mc1",
        data | {"review_title": "Corrected synthetic title"},
        rationale="typo in the registered title",
        reviewer=reviewer,
        expected_base_sha256=sha,
    )
    assert [i.affected for i in p.impact] == [["reporting", "audit", "sign_off"]]
    amendments.decide(
        repo,
        "mc1",
        p.amendment_id,
        "approved",
        reviewer=reviewer,
        comments="administrative correction",
        expected_sha256=p.sha256,
    )
    assert ManuscriptService(repo, "mc1").package().fingerprint_sha256 != before
    assert repo.stats_output_problems("mc1") == []  # outcomes unchanged: statistics still valid


def test_outcome_amendment_requires_extraction_and_plan_rereview(session: Session) -> None:
    repo, rid, _ = comparative_project(session)
    reviewer = repo.require_active_reviewer(rid)
    repo.get_project("mc1").phase = Phase.STAGE4_EXTRACTION_ROB.value
    repo.save_extraction_item(
        "mc1",
        ExtractionItem(
            record_id="t1",
            domain=ExtractionDomain.OUTCOME_DATA,
            field_name="events",
            outcome_id="complication",
            arm_label="X",
            arm_role=ArmRole.INTERVENTION,
            data_status=DataStatus.NOT_REPORTED,
            proposed_by=ActorType.AGENT,
            proposer_identity="llm:synthetic",
        ),
    )
    data, sha = _approved(repo, "mc1")
    data["outcomes"][0] |= {
        "name": "Serious complication",
        "data_type": "continuous",
        "preferred_effect_measure": "MD",
    }
    p = amendments.propose(
        repo,
        "mc1",
        data,
        rationale="definition narrowed",
        reviewer=reviewer,
        expected_base_sha256=sha,
    )
    amendments.decide(
        repo,
        "mc1",
        p.amendment_id,
        "approved",
        reviewer=reviewer,
        comments="ok",
        expected_sha256=p.sha256,
    )
    rows = {o["area"]: o for o in amendments.obligation_status(repo, "mc1")}
    assert set(rows) == {"extraction", "analysis_plans"}
    assert rows["extraction"]["subject"] == "t1|complication"
    with pytest.raises(AmendmentError, match="contradicts the amended protocol"):
        amendments.resolve(
            repo,
            "mc1",
            rows["analysis_plans"]["obligation_id"],
            "confirmed",
            reviewer=reviewer,
            note="still fine",
        )
    state = plan_service.plan_state(repo, "mc1")
    draft = plan_service.save_draft(
        repo,
        "mc1",
        [AnalysisSpec(analysis_id="md", outcome_id="complication", effect_measure="MD")],
        rationale="re-specified after the amendment",
        reviewer=reviewer,
        expected_latest_sha256=state["latest_sha256"],
    )
    plan_service.approve_draft(
        repo,
        "mc1",
        draft.plan_id,
        reviewer=reviewer,
        expected_sha256=draft.sha256(),
        expected_latest_sha256=draft.sha256(),
    )
    rows = {o["area"]: o for o in amendments.obligation_status(repo, "mc1")}
    assert rows["analysis_plans"]["state"] == "resolved"  # resolved by the newer approval
    assert rows["extraction"]["state"] == "open"
    amendments.resolve(
        repo,
        "mc1",
        rows["extraction"]["obligation_id"],
        "confirmed",
        reviewer=reviewer,
        note="not reported under either definition",
    )
    assert not repo.build_guard_context("mc1").amendment_problems
    assert any("re-run Stage 5" in x or "plan" in x for x in repo.stats_output_problems("mc1"))
