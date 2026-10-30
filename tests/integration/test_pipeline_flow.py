"""End-to-end flow across Stages 1-6 with mocked literature APIs and a real SQLite database.

Stage 5 runs the real statistics engine; Stage 6 runs the deterministic manuscript drafter,
the independent Auditor and the human acknowledgement / sign-off workflow (Session 6).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from src.api_server.app import create_app
from src.apis.europepmc import EUROPEPMC_BASE, EuropePMCClient
from src.apis.openalex import OPENALEX_BASE, OpenAlexClient
from src.apis.pubmed import EUTILS_BASE, PubMedClient
from src.apis.rate_limit import AsyncRateLimiter
from src.core.enums import (
    ActorType,
    ArmRole,
    CheckStatus,
    EffectMeasure,
    FullTextStatus,
    GateId,
    GateVerdict,
    OverlapResolution,
    Phase,
    ReviewerRole,
    ScreeningStage,
    ScreeningVerdict,
    SourceDatabase,
    UploadMatchStatus,
)
from src.core.exceptions import GuardViolationError, StatisticsInputError
from src.core.schemas import (
    ExtractedValue,
    FullTextItem,
    GateDecision,
    ScreeningDecision,
)
from src.core.schemas.stats import AnalysisPlan, AnalysisSpec
from src.extraction import record_unavailability
from src.manuscript.service import Actor, ManuscriptService
from src.screening import (
    CohortOverlapDetector,
    Deduplicator,
    PRISMATracker,
    agreement_for_stage,
    detect_conflicts,
    final_verdicts,
)
from src.screening.llm_screening import adjudicate
from src.state.repository import Repository
from src.stats_engine.stage5 import run_stage5
from tests.conftest import (
    NOW,
    approved_by,
    approved_rob,
    make_arm,
    make_evidence,
    make_protocol,
)
from tests.manuscript_helpers import complete_manuscript
from tests.unit.test_api_clients import EPMC_RESULT, OPENALEX_WORK, esearch_json

FIXTURES = Path(__file__).parent.parent / "fixtures"
Sleep = Callable[[float], Awaitable[None]]
PID = "proj1"


REVIEWER: dict[str, str] = {}  # set once the human reviewer is enrolled


def gate(g: GateId, verdict: GateVerdict = GateVerdict.APPROVED) -> GateDecision:
    return GateDecision(
        gate=g,
        verdict=verdict,
        reviewer_name="Dr Human Reviewer",
        reviewer_id=REVIEWER["id"],
        comments=None if verdict == GateVerdict.APPROVED else "please revise",
    )


def advance(repo: Repository, target: Phase) -> None:
    m = repo.load_state_machine(PID)
    t = m.transition(
        target,
        actor="orchestrator",
        actor_type=ActorType.SYSTEM,
        context=repo.build_guard_context(PID),
    )
    repo.record_transition(t)


def decide(repo: Repository, decision: GateDecision) -> None:
    m = repo.load_state_machine(PID)
    repo.record_transition(m.submit_gate_decision(decision, repo.build_guard_context(PID)))


@respx.mock
async def test_full_pipeline(
    session: Session, no_wait_limiter: AsyncRateLimiter, no_sleep: Sleep, tmp_path: Path
) -> None:
    repo = Repository(session)
    repo.create_project(PID, "Synthetic review of drug X")
    REVIEWER["id"], _ = repo.enroll_reviewer("Dr Human Reviewer")

    # ---------------- Stage 1: protocol drafted -> Gate 1 -------------------------------
    advance(repo, Phase.GATE1_SEARCH_STRATEGY)
    with pytest.raises(GuardViolationError, match="protocol not drafted"):
        decide(repo, gate(GateId.GATE_1))
    # Human rejects -> back to Stage 1, protocol revised, approved.
    decide(repo, gate(GateId.GATE_1, GateVerdict.REJECTED))
    assert repo.get_project(PID).phase == Phase.STAGE1_PROTOCOL
    repo.save_protocol(PID, make_protocol())  # a draft is not enough (Session 5)
    advance(repo, Phase.GATE1_SEARCH_STRATEGY)
    with pytest.raises(GuardViolationError, match="not been approved by an authenticated"):
        decide(repo, gate(GateId.GATE_1))
    decide(repo, gate(GateId.GATE_1, GateVerdict.REJECTED))
    repo.save_protocol(PID, approved_by(REVIEWER["id"], version=2))
    advance(repo, Phase.GATE1_SEARCH_STRATEGY)
    decide(repo, gate(GateId.GATE_1))
    assert repo.get_project(PID).phase == Phase.STAGE2_SEARCH_SCREENING

    # ---------------- Stage 2: deterministic multi-database search -----------------------
    respx.get(f"{EUTILS_BASE}esearch.fcgi").mock(
        side_effect=[
            httpx.Response(200, json=esearch_json(2, [])),
            httpx.Response(200, json=esearch_json(2, ["90000001", "90000002"])),
        ]
    )
    respx.post(f"{EUTILS_BASE}efetch.fcgi").mock(
        return_value=httpx.Response(
            200, content=(FIXTURES / "pubmed_efetch_synthetic.xml").read_bytes()
        )
    )
    respx.get(f"{EUROPEPMC_BASE}search").mock(
        return_value=httpx.Response(
            200,
            json={"hitCount": 1, "nextCursorMark": "*", "resultList": {"result": [EPMC_RESULT]}},
        )
    )
    respx.get(f"{OPENALEX_BASE}works").mock(
        side_effect=[
            httpx.Response(
                200, json={"meta": {"count": 1, "next_cursor": "n"}, "results": [OPENALEX_WORK]}
            ),
            httpx.Response(200, json={"meta": {"count": 1, "next_cursor": None}, "results": []}),
        ]
    )
    kw = {"rate_limiter": no_wait_limiter, "sleep": no_sleep}
    async with (
        PubMedClient(email="t@example.org", **kw) as pubmed,  # type: ignore[arg-type]
        EuropePMCClient(**kw) as epmc,  # type: ignore[arg-type]
        OpenAlexClient(mailto="t@example.org", **kw) as openalex,  # type: ignore[arg-type]
    ):
        _, pm_records = await pubmed.search("drug x")
        _, epmc_records = await epmc.search("drug x")
        _, oa_records = await openalex.search(search="drug x")

    prisma = PRISMATracker()
    prisma.record_identified(SourceDatabase.PUBMED, len(pm_records))
    prisma.record_identified(SourceDatabase.EUROPE_PMC, len(epmc_records))
    prisma.record_identified(SourceDatabase.OPENALEX, len(oa_records))
    all_records = pm_records + epmc_records + oa_records
    assert len(all_records) == 4

    dedup = Deduplicator().deduplicate(all_records)
    # The same synthetic trial was returned by all three databases.
    assert dedup.duplicates_removed == 2
    prisma.record_duplicates_removed(dedup.duplicates_removed)
    repo.save_records(PID, dedup.all_records)
    unique = repo.load_records(PID, unique_only=True)
    assert len(unique) == 2

    # Dual independent screening (decisions would come from two LLM reviewer agents).
    main_id = next(r.record_id for r in unique if r.pmid == "90000001")
    follow_id = next(r.record_id for r in unique if r.pmid == "90000002")
    decisions = [
        ScreeningDecision(
            record_id=main_id,
            reviewer=ReviewerRole.REVIEWER_1,
            reviewer_identity="agent-1",
            stage=ScreeningStage.TITLE_ABSTRACT,
            verdict=ScreeningVerdict.INCLUDE,
            rationale="RCT in adults",
        ),
        ScreeningDecision(
            record_id=main_id,
            reviewer=ReviewerRole.REVIEWER_2,
            reviewer_identity="agent-2",
            stage=ScreeningStage.TITLE_ABSTRACT,
            verdict=ScreeningVerdict.INCLUDE,
            rationale="RCT in adults",
        ),
        ScreeningDecision(
            record_id=follow_id,
            reviewer=ReviewerRole.REVIEWER_1,
            reviewer_identity="agent-1",
            stage=ScreeningStage.TITLE_ABSTRACT,
            verdict=ScreeningVerdict.INCLUDE,
            rationale="follow-up of RCT",
        ),
        ScreeningDecision(
            record_id=follow_id,
            reviewer=ReviewerRole.REVIEWER_2,
            reviewer_identity="agent-2",
            stage=ScreeningStage.TITLE_ABSTRACT,
            verdict=ScreeningVerdict.EXCLUDE,
            exclusion_reason="no comparator",
            rationale="cohort only",
        ),
    ]
    for d in decisions:
        repo.save_screening_decision(PID, d)
    kappa = agreement_for_stage(decisions, ScreeningStage.TITLE_ABSTRACT)
    assert kappa.observed_agreement == pytest.approx(0.5)
    conflicts = detect_conflicts(decisions, ScreeningStage.TITLE_ABSTRACT)
    assert [c.record_id for c in conflicts] == [follow_id]
    for c in conflicts:
        repo.save_conflict(PID, c)

    warnings = CohortOverlapDetector().detect(unique)
    assert len(warnings) == 1  # same first author group, close publication years
    for w in warnings:
        repo.save_overlap_warning(PID, w)

    # PRISMA snapshot during screening: one record decided, one awaiting adjudication.
    prisma.register_screening_pool([r.record_id for r in unique])
    prisma.record_title_abstract_decision(main_id, include=True)
    with pytest.raises(GuardViolationError, match="PRISMA counts have not been recorded"):
        advance(repo, Phase.GATE1_5_SCREENING_CONFLICTS)
    repo.save_prisma(PID, prisma.snapshot())
    assert repo.load_prisma(PID).records_awaiting_screening == 1  # type: ignore[union-attr]
    advance(repo, Phase.GATE1_5_SCREENING_CONFLICTS)

    # ---------------- Gate 1.5: human must resolve conflicts + overlap warnings ---------
    with pytest.raises(GuardViolationError) as info:
        decide(repo, gate(GateId.GATE_1_5))
    assert any("screening conflict" in v for v in info.value.violations)
    assert any("cohort-overlap" in v for v in info.value.violations)

    human = repo.require_active_reviewer(REVIEWER["id"])
    for c in repo.load_conflicts(PID):
        adjudicate(repo, PID, c.record_id, c.stage, verdict=ScreeningVerdict.INCLUDE,
                   rationale="eligible on adjudication", reviewer=human)  # fmt: skip
    for w in repo.load_overlap_warnings(PID):
        repo.resolve_overlap_warning(
            PID, w.warning_id, resolution=OverlapResolution.OVERLAPPING_DIFFERENT_OUTCOMES,
            reviewer_id=REVIEWER["id"], note="same cohort, different follow-up",
        )  # fmt: skip

    verdicts = final_verdicts(
        repo.load_screening_decisions(PID), repo.load_conflicts(PID), ScreeningStage.TITLE_ABSTRACT
    )
    for rid, verdict in verdicts.items():
        prisma.record_title_abstract_decision(rid, include=verdict == ScreeningVerdict.INCLUDE)
    repo.save_prisma(PID, prisma.snapshot())
    decide(repo, gate(GateId.GATE_1_5))
    assert repo.get_project(PID).phase == Phase.STAGE3_FULLTEXT_RETRIEVAL

    # ---------------- Stage 3: OA retrieval + missing manifest -> Gate 2 ----------------
    by_id = {r.record_id: r for r in unique}
    repo.save_fulltext_item(
        PID,
        FullTextItem(
            record_id=main_id,
            title=by_id[main_id].title,
            pmid=by_id[main_id].pmid,
            status=FullTextStatus.OA_DOWNLOADED,
            pdf_filename="main.pdf",
            pdf_sha256="e" * 64,
            match_status=UploadMatchStatus.CONFIRMED,  # synthetic: DOI matched on page 1
        ),
    )
    repo.save_fulltext_item(
        PID,
        FullTextItem(
            record_id=follow_id,
            title=by_id[follow_id].title,
            pmid=by_id[follow_id].pmid,
            status=FullTextStatus.MISSING,
        ),
    )
    manifest = repo.missing_fulltext_manifest(PID)
    assert [m.pmid for m in manifest] == ["90000002"]
    advance(repo, Phase.GATE2_MISSING_FULLTEXT)
    with pytest.raises(GuardViolationError, match="still missing"):
        decide(repo, gate(GateId.GATE_2))
    repo.save_fulltext_item(
        PID,
        manifest[0].model_copy(
            update={
                "status": FullTextStatus.USER_UPLOADED,
                "pdf_filename": "followup.pdf",
                "pdf_sha256": "f" * 64,
                "recorded_by": "Dr Human Reviewer",
                "recorded_by_reviewer_id": REVIEWER["id"],
                "match_status": UploadMatchStatus.CONFIRMED,
            }
        ),
    )
    decide(repo, gate(GateId.GATE_2))
    prisma.record_retrieval(main_id, retrieved=True)
    prisma.record_retrieval(follow_id, retrieved=True)
    repo.save_prisma(PID, prisma.snapshot())

    # ---------------- Stage 4: grounded extraction -> Gate 3 ----------------------------
    with pytest.raises(GuardViolationError, match="lack full-text eligibility decisions"):
        advance(repo, Phase.GATE3_EXTRACTION_APPROVAL)
    for rid in (main_id, follow_id):  # dual full-text eligibility assessment
        for reviewer in (ReviewerRole.REVIEWER_1, ReviewerRole.REVIEWER_2):
            repo.save_screening_decision(
                PID,
                ScreeningDecision(
                    record_id=rid, reviewer=reviewer, reviewer_identity=f"agent-{reviewer.value}",
                    stage=ScreeningStage.FULL_TEXT, verdict=ScreeningVerdict.INCLUDE,
                    rationale="eligible on full text",
                ),
            )  # fmt: skip
    with pytest.raises(GuardViolationError, match="await confirmation by an authenticated"):
        advance(repo, Phase.GATE3_EXTRACTION_APPROVAL)
    for rid in (main_id, follow_id):  # Session 2: a human owns full-text eligibility
        repo.save_screening_decision(
            PID,
            ScreeningDecision(
                record_id=rid, reviewer=ReviewerRole.ADJUDICATOR, is_human=True,
                reviewer_identity="Dr Human Reviewer", reviewer_id=REVIEWER["id"],
                stage=ScreeningStage.FULL_TEXT, verdict=ScreeningVerdict.INCLUDE,
                rationale="confirmed eligible on full text",
            ),
        )  # fmt: skip
    with pytest.raises(GuardViolationError, match="no extracted"):
        advance(repo, Phase.GATE3_EXTRACTION_APPROVAL)
    arm = make_arm(main_id)
    repo.save_extracted_arm(PID, arm)
    ctrl_ev = make_evidence("Placebo (n=118): mean 13·0 (SD 3·4); 115 analysed")
    control_arm = make_arm(
        main_id, arm_label="Placebo", arm_role=ArmRole.CONTROL,
        n_randomized=ExtractedValue(value=118, reported_text="118", evidence=ctrl_ev),
        n_analyzed=ExtractedValue(value=115, reported_text="115", evidence=ctrl_ev),
        mean=ExtractedValue(value=13.0, reported_text="13·0", evidence=ctrl_ev),
        sd=ExtractedValue(value=3.4, reported_text="3·4", evidence=ctrl_ev),
    )  # fmt: skip
    repo.save_extracted_arm(PID, control_arm)
    with pytest.raises(GuardViolationError, match="1 included report"):
        advance(repo, Phase.GATE3_EXTRACTION_APPROVAL)
    follow_arm = make_arm(follow_id)
    repo.save_extracted_arm(PID, follow_arm)
    advance(repo, Phase.GATE3_EXTRACTION_APPROVAL)
    with pytest.raises(GuardViolationError, match="not approved by a human"):
        decide(repo, gate(GateId.GATE_3))
    repo.approve_extracted_arm(arm.arm_id, reviewer_id=REVIEWER["id"])
    repo.approve_extracted_arm(control_arm.arm_id, reviewer_id=REVIEWER["id"])
    repo.approve_extracted_arm(follow_arm.arm_id, reviewer_id=REVIEWER["id"])
    with pytest.raises(GuardViolationError, match="risk-of-bias"):
        decide(repo, gate(GateId.GATE_3))
    for rid in (main_id, follow_id):
        repo.save_rob(PID, approved_rob(rid, REVIEWER["id"]))
    with pytest.raises(GuardViolationError, match="PRISMA flow not final"):
        decide(repo, gate(GateId.GATE_3))
    prisma.record_fulltext_decision(main_id, include=True)
    prisma.record_fulltext_decision(follow_id, include=True)
    prisma.set_studies_included(1)  # two reports of one cohort (overlap warning resolved)
    repo.save_prisma(PID, prisma.snapshot())
    # Both reports of the same cohort carry arm data for the same outcome: blocked until a
    # human chooses one report, so the cohort's patients are not pooled twice.
    with pytest.raises(GuardViolationError, match="more than one report of the same cohort"):
        decide(repo, gate(GateId.GATE_3))
    record_unavailability(
        repo,
        PID,
        follow_id,
        reviewer=human,
        reason_code="overlapping_cohort",
        reason="follow-up report of the main cohort",
        outcome_id="hba1c",
    )
    decide(repo, gate(GateId.GATE_3))
    assert repo.get_project(PID).phase == Phase.STAGE5_META_ANALYSIS

    # ---------------- Stage 5-6: stats -> manuscript -> audit loop ----------------------
    with pytest.raises(GuardViolationError, match="stats_output"):
        advance(repo, Phase.STAGE6_MANUSCRIPT)
    project = repo.get_project(PID)
    project.stats_output_sha256 = "1" * 64  # a hash without a verified artifact is refused
    with pytest.raises(GuardViolationError, match="no stored artifact"):
        advance(repo, Phase.STAGE6_MANUSCRIPT)
    project.stats_output_sha256 = None
    with pytest.raises(StatisticsInputError, match="analysis plan"):
        run_stage5(repo, PID, output_dir=tmp_path / "stats")
    repo.save_analysis_plan(
        PID,
        AnalysisPlan(
            analyses=[AnalysisSpec(analysis_id="hba1c_md", outcome_id="hba1c",
                                   effect_measure=EffectMeasure.MD)],
            drafted_by="Dr Human Reviewer", status="approved", approved_by="Dr Human Reviewer",
            approved_reviewer_id=REVIEWER["id"], approved_at=NOW,
        ),
    )  # fmt: skip
    stage5 = run_stage5(repo, PID, output_dir=tmp_path / "stats", generated_at=NOW)
    md = stage5.output.analyses[0]
    # one study: its own MD is reported, nothing is "pooled" and no heterogeneity is computed
    assert (md.status, md.k, md.heterogeneity) == ("single_study", 1, None)
    assert md.studies[0].estimate_natural == pytest.approx(-0.5)
    # the follow-up report belongs to the included cohort: it is not a separate excluded
    # study, but the per-report ledger accounts for it with the human's reason
    assert md.excluded == []
    ledger = {r.record_id: r for r in md.reports}
    assert set(ledger) == {main_id, follow_id}
    assert ledger[main_id].status == "included"
    follow = ledger[follow_id]
    assert (follow.status, follow.reason_code) == ("excluded", "recorded_unavailable")
    assert follow.reason is not None and "not counted as a separate study" in follow.reason
    assert (tmp_path / "stats" / "stats_output.json").read_text() == (
        stage5.output.to_canonical_json()
    )
    advance(repo, Phase.STAGE6_MANUSCRIPT)
    with pytest.raises(GuardViolationError, match="no manuscript draft"):
        advance(repo, Phase.STAGE6_AUDIT)

    # ---------------- Stage 6: evidence-grounded draft -> independent audit -> sign-off ----
    svc = ManuscriptService(repo, PID)
    draft = svc.create_template_draft(Actor("orchestrator", ActorType.SYSTEM))
    # every number in the draft comes from the evidence package, e.g. the single-study MD
    assert "only one study contributed data" in draft.rendered_markdown
    advance(repo, Phase.STAGE6_AUDIT)
    first = svc.audit()
    assert first.complete and not first.passed  # [[AUTHOR INPUT]] markers block it
    assert any(
        f.status == CheckStatus.MISSING and "author input" in f.message for f in first.findings
    )
    with pytest.raises(GuardViolationError, match="not passed"):
        advance(repo, Phase.COMPLETED)
    advance(repo, Phase.STAGE6_MANUSCRIPT)  # audit failed -> redraft
    reviewer = Actor("Dr Human Reviewer", ActorType.HUMAN, REVIEWER["id"])
    complete_manuscript(svc, reviewer)
    # the earlier audit examined version 1 and can never authorise version 2
    assert any("examined version 1" in p for p in repo.build_guard_context(PID).audit_problems)
    advance(repo, Phase.STAGE6_AUDIT)
    report = svc.audit()
    assert report.passed, [f.message for f in report.findings if f.blocking]
    for f in report.findings:
        if f.needs_human:
            svc.acknowledge(
                report.report_id,
                f.finding_id,
                decision="confirmed_supported",
                note="checked against the source by the reviewer",
                reviewer_id=REVIEWER["id"],
                reviewer_name="Dr Human Reviewer",
            )
    with pytest.raises(GuardViolationError, match="sign-off"):
        advance(repo, Phase.COMPLETED)
    svc.approve(
        reviewer_id=REVIEWER["id"],
        reviewer_name="Dr Human Reviewer",
        statement="I have read the manuscript and the audit and approve version 2.",
    )
    advance(repo, Phase.COMPLETED)

    machine = repo.load_state_machine(PID)
    assert machine.is_terminal
    phases = [t.to_phase for t in machine.history]
    assert phases.count(Phase.STAGE6_AUDIT) == 2
    assert [v.version for v in repo.list_manuscript_versions(PID)] == [1, 2]
    assert len(repo.list_audit_reports(PID)) == 2  # the failed audit is preserved
    human_gate_steps = [t for t in machine.history if t.gate_decision is not None]
    assert {t.gate_decision.gate for t in human_gate_steps if t.gate_decision} == set(GateId)
    final = repo.load_prisma(PID)
    assert final is not None and final.is_final
    assert (final.records_screened, final.reports_included, final.studies_included) == (2, 2, 1)


def test_hitl_flow_over_http() -> None:
    """Gate decisions submitted through the FastAPI layer are authenticated, guarded, recorded."""
    app = create_app("sqlite:///:memory:")
    with TestClient(app) as client:
        pid = client.post("/projects", json={"title": "HTTP review"}).json()["project_id"]

        # System steps have no HTTP endpoint yet; drive them through the repository.
        # The reviewer is enrolled out of band (in production: the terminal-only CLI).
        with app.state.session_factory() as session:
            repo = Repository(session)
            _, token = repo.enroll_reviewer("Dr R")
            m = repo.load_state_machine(pid)
            repo.record_transition(
                m.transition(
                    Phase.GATE1_SEARCH_STRATEGY,
                    actor="orchestrator",
                    actor_type=ActorType.SYSTEM,
                    context=repo.build_guard_context(pid),
                )
            )
            session.commit()
        auth = {"Authorization": f"Bearer {token}"}
        approve = {"verdict": "approved"}

        view = client.get(f"/projects/{pid}").json()
        assert view["awaiting_gate"] == "gate_1"
        guards = client.get(f"/projects/{pid}/guards").json()
        assert guards["stage2_search_screening"] == ["protocol: protocol not drafted"]
        assert guards["stage1_protocol"] == []

        # An unauthenticated request naming a reviewer is refused outright.
        impersonation = client.post(
            f"/projects/{pid}/gates/gate_1", json={"verdict": "approved", "reviewer_name": "Dr R"}
        )
        assert impersonation.status_code == 401
        assert client.get(f"/projects/{pid}").json()["phase"] == "gate1_search_strategy"

        blocked = client.post(f"/projects/{pid}/gates/gate_1", json=approve, headers=auth)
        assert blocked.status_code == 409
        assert blocked.json()["detail"]["violations"] == ["protocol: protocol not drafted"]

        wrong_gate = client.post(f"/projects/{pid}/gates/gate_3", json=approve, headers=auth)
        assert wrong_gate.status_code == 409

        with app.state.session_factory() as session:
            repo = Repository(session)
            repo.save_protocol(pid, make_protocol())
            session.commit()
        draft_only = client.post(f"/projects/{pid}/gates/gate_1", json=approve, headers=auth)
        assert draft_only.status_code == 409
        with app.state.session_factory() as session:
            repo = Repository(session)
            reviewer_id = repo.list_reviewers()[0].id
            repo.save_protocol(pid, approved_by(reviewer_id, name="Dr R", version=2))
            session.commit()

        ok = client.post(f"/projects/{pid}/gates/gate_1", json=approve, headers=auth)
        assert ok.status_code == 200
        body = ok.json()
        assert body["phase"] == "stage2_search_screening"
        last = body["history"][-1]
        assert last["actor_type"] == "human"
        assert last["gate_decision"]["reviewer_name"] == "Dr R"
        assert last["gate_decision"]["reviewer_id"]
