"""Regressions for the five confirmed findings of the independent review of PR #13
(commit 4748b58). Each test reproduces the reported scenario and checks that it is now refused
with nothing stored, or (finding 4) that the review can continue correctly.

All data SYNTHETIC; no network call is made (no model or provider is contacted).
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.api_server.app import create_app
from src.core.enums import (
    ActorType,
    GateId,
    GateVerdict,
    Phase,
    ReviewStatus,
    ScreeningStage,
    ScreeningVerdict,
)
from src.core.exceptions import SysRevError
from src.core.schemas.pipeline import GateDecision
from src.core.schemas.stats import AnalysisSpec
from src.extraction import (
    ReviewError,
    StaleStateError,
    accept_item,
    arms_state_sha256,
    build_arms,
    correct_item,
    correction_state_sha256,
    link_reports_to_study,
    links_state_sha256,
    record_unavailability,
    resolve_double_counting,
    revise_reviewed_item,
    submit_proposals,
    unlink_reports,
    withdraw_arms,
)
from src.fulltext import confirm_pdf_match, register_upload
from src.fulltext.storage import DocumentStore
from src.pdf_extractor import parse_pdf
from src.screening.llm_screening import adjudicate
from src.search import amendments
from src.search import protocol_editor as editor
from src.search.prisma_sync import derive_prisma
from src.state.db import (
    ExtractedArmORM,
    ExtractionItemORM,
    ExtractionReviewEventORM,
    ReviewerORM,
    init_db,
    make_engine,
    make_session_factory,
)
from src.state.repository import Repository
from src.stats_engine import plan_service
from src.stats_engine.stage5 import run_stage5
from tests.conftest import approved_by, approved_rob, make_record
from tests.fakes import include_at_title_abstract
from tests.fixtures.pdf_factory import SYNTHETIC_DOI, SYNTHETIC_TITLE, synthetic_article
from tests.manuscript_helpers import comparative_project
from tests.unit.test_master_table import km

PID = "p13"
PDF = synthetic_article()
REVISED_PDF = PDF + b"\n% revised edition of the same synthetic article\n"  # valid, new hash
OTHER_PDF = synthetic_article(title="Another synthetic report", doi="10.5555/other.0002")


# =========================================================================== helpers
@dataclass
class Proj:
    repo: Repository
    store: DocumentStore
    reviewer: ReviewerORM

    def doc(self, record_id: str) -> Any:
        ft = self.repo.get_fulltext_item(PID, record_id)
        assert ft is not None and ft.pdf_filename and ft.pdf_sha256
        data = self.store.read_verified(ft.pdf_filename, ft.pdf_sha256)
        return parse_pdf(data, filename=ft.pdf_filename, expected_sha256=ft.pdf_sha256)

    def upload(self, record_id: str, data: bytes, name: str) -> None:
        item = register_upload(
            self.repo,
            self.store,
            PID,
            record_id,
            data,
            original_filename=name,
            reviewer=self.reviewer,
        )
        if item.match_unconfirmed:
            confirm_pdf_match(self.repo, PID, record_id, reviewer=self.reviewer, note="checked")


def build(session: Session, store_dir: Path, records: dict[str, bytes]) -> Proj:
    """Reports with linked PDFs, included at title/abstract and (by a human) at full text."""
    repo = Repository(session)
    repo.create_project(PID, "Synthetic review (PR #13 fixes)")
    rid, _ = repo.enroll_reviewer("Dr Fixer")
    reviewer = repo.require_active_reviewer(rid)
    repo.save_protocol(PID, approved_by(rid, name="Dr Fixer"))
    titles = {
        PDF: (SYNTHETIC_TITLE, SYNTHETIC_DOI),
        OTHER_PDF: ("Another synthetic report", "10.5555/other.0002"),
    }
    recs = [make_record(r, titles[d][0], doi=titles[d][1]) for r, d in records.items()]
    repo.save_records(PID, recs)
    include_at_title_abstract(repo, PID, recs)
    p = Proj(repo, DocumentStore(store_dir), reviewer)
    for record_id, data in records.items():
        p.upload(record_id, data, f"{record_id}.pdf")
        adjudicate(
            repo,
            PID,
            record_id,
            ScreeningStage.FULL_TEXT,
            verdict=ScreeningVerdict.INCLUDE,
            rationale="eligible",
            reviewer=reviewer,
        )
    return p


def accepted_estimate(p: Proj, record_id: str = "r1", arms: bool = True) -> Any:
    (item,) = submit_proposals(
        p.repo,
        PID,
        record_id,
        p.doc(record_id),
        [km("reported_estimate", "62.1", 1, 1, 3)],
        proposed_by=ActorType.AGENT,
        proposer_identity="llm:t",
    )
    accepted = accept_item(p.repo, PID, item.item_id, reviewer=p.reviewer, note=None)
    if arms:
        build_arms(p.repo, PID, record_id, reviewer=p.reviewer)
    return accepted


def counts(repo: Repository) -> tuple[int, int, int]:
    s = repo.session
    return (
        s.scalar(select(func.count()).select_from(ExtractionItemORM)) or 0,
        s.scalar(select(func.count()).select_from(ExtractionReviewEventORM)) or 0,
        len(repo.governance_records(PID, "arm_invalidation")),
    )


def stage(repo: Repository, phase: Phase, pid: str = PID) -> None:
    repo.get_project(pid).phase = phase.value
    repo.session.flush()


# =========================================================================== finding 1
def test_f1_pending_correction_is_refused_after_the_pdf_was_replaced(
    session: Session, tmp_path: Path
) -> None:
    p = build(session, tmp_path, {"r1": PDF})
    (item,) = submit_proposals(
        p.repo,
        PID,
        "r1",
        p.doc("r1"),
        [km("reported_estimate", "62.1", 1, 1, 3)],
        proposed_by=ActorType.AGENT,
        proposer_identity="llm:t",
    )
    shown = correction_state_sha256(p.repo, PID, item)  # the screen with the original PDF
    p.upload("r1", REVISED_PDF, "revised.pdf")  # replaced; same quoted evidence, confirmed
    before = counts(p.repo)
    with pytest.raises(StaleStateError, match="linked PDF"):
        correct_item(
            p.repo,
            PID,
            item.item_id,
            km("reported_estimate", "81.4", 1, 2, 3),
            p.doc("r1"),
            reviewer=p.reviewer,
            note="from the old screen",
            expected_state_sha256=shown,
        )
    assert counts(p.repo) == before
    assert p.repo.get_extraction_item(PID, item.item_id).review_status == ReviewStatus.PENDING


def test_f1_reviewed_correction_refused_after_replacement_or_reconfirmation(
    session: Session, tmp_path: Path
) -> None:
    p = build(session, tmp_path, {"r1": PDF})
    stage(p.repo, Phase.STAGE4_EXTRACTION_ROB)
    item = accepted_estimate(p)
    shown = correction_state_sha256(p.repo, PID, item)
    # (a) the identity confirmation changes (another reviewer re-checks the same PDF)
    other_id, _ = p.repo.enroll_reviewer("Dr Second Opinion")
    confirm_pdf_match(
        p.repo,
        PID,
        "r1",
        reviewer=p.repo.require_active_reviewer(other_id),
        note="re-checked against the journal site",
    )
    before = counts(p.repo)
    with pytest.raises(StaleStateError, match="has changed"):
        revise_reviewed_item(
            p.repo,
            PID,
            item.item_id,
            km("reported_estimate", "81.4", 1, 2, 3),
            p.doc("r1"),
            reviewer=p.reviewer,
            note="old screen",
            expected_state_sha256=shown,
        )
    assert counts(p.repo) == before
    # (b) the PDF is replaced
    shown = correction_state_sha256(p.repo, PID, item)
    p.upload("r1", REVISED_PDF, "revised.pdf")
    with pytest.raises(StaleStateError, match="has changed"):
        revise_reviewed_item(
            p.repo,
            PID,
            item.item_id,
            km("reported_estimate", "81.4", 1, 2, 3),
            p.doc("r1"),
            reviewer=p.reviewer,
            note="old screen",
            expected_state_sha256=shown,
        )
    assert counts(p.repo) == before
    assert p.repo.get_extraction_item(PID, item.item_id).review_status == ReviewStatus.ACCEPTED
    # control: from the current screen the correction is accepted and bound to the new PDF
    current = correction_state_sha256(p.repo, PID, item)
    new, invalidated = revise_reviewed_item(
        p.repo,
        PID,
        item.item_id,
        km("reported_estimate", "81.4", 1, 2, 3),
        p.doc("r1"),
        reviewer=p.reviewer,
        note="reviewed against the revised PDF",
        expected_state_sha256=current,
    )
    ft = p.repo.get_fulltext_item(PID, "r1")
    assert ft is not None and new.evidence is not None
    assert new.evidence.pdf_sha256 == ft.pdf_sha256 and invalidated


def test_f1_http_correction_from_an_old_screen_returns_409_and_stores_nothing() -> None:
    tmp = Path(tempfile.mkdtemp())
    with TestClient(create_app("sqlite:///:memory:", data_dir=tmp)) as c:
        with c.app.state.session_factory() as s:  # type: ignore[attr-defined]
            p = build(s, tmp / "fulltext", {"r1": PDF})
            stage(p.repo, Phase.STAGE4_EXTRACTION_ROB)
            accepted_estimate(p, arms=False)
            _, token = p.repo.enroll_reviewer("Dr HTTP")
            s.commit()
        shown = c.get(f"/projects/{PID}/extraction/items").json()[0]
        assert shown["source"]["pdf_sha256"] == shown["evidence"]["pdf_sha256"]
        with c.app.state.session_factory() as s:  # type: ignore[attr-defined]
            repo = Repository(s)
            Proj(
                repo,
                DocumentStore(tmp / "fulltext"),
                repo.require_active_reviewer(repo.list_reviewers()[0].id),
            ).upload("r1", REVISED_PDF, "revised.pdf")
            s.commit()
        body = {
            "correction": km("reported_estimate", "81.4", 1, 2, 3).model_dump(mode="json"),
            "note": "old screen",
            "expected_state_sha256": shown["state_sha256"],
        }
        r = c.post(
            f"/projects/{PID}/extraction/items/{shown['item_id']}/correct",
            json=body,
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 409 and "has changed" in r.json()["detail"]
        after = c.get(f"/projects/{PID}/extraction/items", params={"include_superseded": True})
        assert [(i["item_id"], i["review_status"]) for i in after.json()] == [
            (shown["item_id"], "accepted")
        ]
        hist = c.get(f"/projects/{PID}/extraction/items/{shown['item_id']}/history").json()
        assert [h["action"] for h in hist] == ["proposed", "accepted"]


# =========================================================================== finding 2
def test_f2_outcome_amendment_requires_rereview_of_every_report_with_arms(
    session: Session,
) -> None:
    repo, rid, _ = comparative_project(session)  # 6 approved arms, no extraction items
    reviewer = repo.require_active_reviewer(rid)
    stage(repo, Phase.STAGE4_EXTRACTION_ROB, "mc1")
    assert len(repo.load_extracted_arms("mc1")) == 6 and not repo.load_extraction_items("mc1")
    base = repo.latest_protocol("mc1")
    assert base is not None
    data = base.model_dump(mode="json")
    data["outcomes"][0] |= {
        "name": "Serious complication requiring re-operation",
        "definition": "Complication requiring re-operation",
    }
    p = amendments.propose(
        repo,
        "mc1",
        data,
        rationale="narrowed definition",
        reviewer=reviewer,
        expected_base_sha256=editor.protocol_sha256(base),
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
    rows = amendments.obligation_status(repo, "mc1")
    extraction = {o["subject"]: o for o in rows if o["area"] == "extraction"}
    assert set(extraction) == {"t1|complication", "t2|complication", "t3|complication"}
    plan_row = next(o for o in rows if o["area"] == "analysis_plans")
    amendments.resolve(
        repo,
        "mc1",
        plan_row["obligation_id"],
        "confirmed",
        reviewer=reviewer,
        note="the method still applies",
    )
    ctx = repo.build_guard_context("mc1")
    assert any("3 extraction item(s)" in x for x in ctx.amendment_problems)  # data unreviewed
    machine = repo.load_state_machine("mc1")
    blocked = machine.check(Phase.GATE3_EXTRACTION_APPROVAL, ctx)
    assert any("extraction" in v for v in blocked)
    for subject, row in extraction.items():
        amendments.resolve(
            repo,
            "mc1",
            row["obligation_id"],
            "confirmed",
            reviewer=reviewer,
            note=f"{subject}: only re-operations were counted",
        )
    assert not repo.build_guard_context("mc1").amendment_problems


# =========================================================================== finding 3
def _draft_and_amend(repo: Repository, rid: str) -> tuple[dict[str, Any], Any]:
    reviewer = repo.require_active_reviewer(rid)
    stage(repo, Phase.STAGE4_EXTRACTION_ROB, "mc1")
    state = plan_service.plan_state(repo, "mc1")
    plan_service.save_draft(
        repo,
        "mc1",
        [AnalysisSpec(analysis_id="or", outcome_id="complication", effect_measure="OR")],
        rationale="odds ratio",
        reviewer=reviewer,
        expected_latest_sha256=state["latest_sha256"],
        expected_protocol_sha256=state["protocol_sha256"],
    )
    shown = plan_service.plan_state(repo, "mc1")  # the screen the reviewer is looking at
    base = repo.latest_protocol("mc1")
    assert base is not None
    data = base.model_dump(mode="json")
    data["outcomes"][0]["name"] = "Serious complication requiring re-operation"
    p = amendments.propose(
        repo,
        "mc1",
        data,
        rationale="narrowed",
        reviewer=reviewer,
        expected_base_sha256=editor.protocol_sha256(base),
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
    return shown, reviewer


def _plan_obligation(repo: Repository) -> str:
    return next(
        o["state"]
        for o in amendments.obligation_status(repo, "mc1")
        if o["area"] == "analysis_plans"
    )


def test_f3_plan_approval_from_a_screen_of_the_old_protocol_is_refused(session: Session) -> None:
    repo, rid, _ = comparative_project(session)
    shown, reviewer = _draft_and_amend(repo, rid)
    draft = shown["plans"][-1]
    n = len(plan_service.list_plans(repo, "mc1"))
    with pytest.raises(StaleStateError, match="protocol"):
        plan_service.approve_draft(
            repo,
            "mc1",
            draft["plan_id"],
            reviewer=reviewer,
            expected_sha256=draft["sha256"],
            expected_latest_sha256=shown["latest_sha256"],
            expected_protocol_sha256=shown["protocol_sha256"],
        )
    with pytest.raises(StaleStateError, match="protocol"):
        plan_service.save_draft(
            repo,
            "mc1",
            [AnalysisSpec(analysis_id="rr", outcome_id="complication", effect_measure="RR")],
            rationale=None,
            reviewer=reviewer,
            expected_latest_sha256=shown["latest_sha256"],
            expected_protocol_sha256=shown["protocol_sha256"],
        )
    assert len(plan_service.list_plans(repo, "mc1")) == n  # nothing stored
    assert _plan_obligation(repo) == "open"  # and the re-review is not closed
    # control: after an explicit reload the reviewer approves against the amended protocol
    current = plan_service.plan_state(repo, "mc1")
    plan_service.approve_draft(
        repo,
        "mc1",
        draft["plan_id"],
        reviewer=reviewer,
        expected_sha256=draft["sha256"],
        expected_latest_sha256=current["latest_sha256"],
        expected_protocol_sha256=current["protocol_sha256"],
    )
    assert _plan_obligation(repo) == "resolved"


def test_f3_http_and_legacy_endpoint(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'f3.db'}"
    engine = make_engine(url)
    init_db(engine)
    with make_session_factory(engine)() as s:
        repo, rid, token = comparative_project(s)
        shown, _ = _draft_and_amend(repo, rid)
        s.commit()
    engine.dispose()
    with TestClient(create_app(url, data_dir=tmp_path)) as c:
        h = {"Authorization": f"Bearer {token}"}
        draft = shown["plans"][-1]
        path = f"/projects/mc1/stats/plan/drafts/{draft['plan_id']}/approve"
        stale = {
            "expected_sha256": draft["sha256"],
            "expected_latest_sha256": shown["latest_sha256"],
            "expected_protocol_sha256": shown["protocol_sha256"],
        }
        r = c.post(path, json=stale, headers=h)
        assert r.status_code == 409 and "has changed" in r.json()["detail"]
        assert (
            c.post(
                path,
                json={k: v for k, v in stale.items() if k != "expected_protocol_sha256"},
                headers=h,
            ).status_code
            == 422
        )
        # the legacy direct approval never closes the amendment's plan re-review
        legacy = c.post(
            "/projects/mc1/stats/plan",
            headers=h,
            json={
                "analyses": [
                    {"analysis_id": "rr2", "outcome_id": "complication", "effect_measure": "RR"}
                ]
            },
        )
        assert legacy.status_code == 201
        with c.app.state.session_factory() as s:  # type: ignore[attr-defined]
            assert _plan_obligation(Repository(s)) == "open"
        current = c.get("/projects/mc1/stats/plans").json()  # explicit reload
        assert current["protocol_sha256"] != shown["protocol_sha256"]
        latest = current["plans"][-1]
        assert latest["status"] == "approved"  # the legacy approval is the newest version


# =========================================================================== finding 4
def test_f4_excluded_report_arms_are_withdrawn_and_the_review_continues_to_stage5(
    session: Session, tmp_path: Path
) -> None:
    p = build(session, tmp_path, {"r1": PDF, "r3": OTHER_PDF})
    repo, me = p.repo, p.reviewer
    stage(repo, Phase.STAGE4_EXTRACTION_ROB)
    for rid in ("r1", "r3"):
        accepted_estimate(p, rid)
        repo.save_rob(PID, approved_rob(rid, me.id, name=me.display_name))
    repo.save_prisma(PID, derive_prisma(repo, PID))
    base = repo.latest_protocol(PID)
    assert base is not None
    data = base.model_dump(mode="json") | {
        "exclusion_criteria": ["Animal studies", "Series with fewer than 60 patients"]
    }
    a = amendments.propose(
        repo,
        PID,
        data,
        rationale="minimum series size",
        reviewer=me,
        expected_base_sha256=editor.protocol_sha256(base),
    )
    amendments.decide(
        repo,
        PID,
        a.amendment_id,
        "approved",
        reviewer=me,
        comments="agreed",
        expected_sha256=a.sha256,
    )
    for o in amendments.obligation_status(repo, PID):
        if o["area"] == "full_text_screening" and o["subject"] == "r1":
            amendments.resolve(
                repo,
                PID,
                o["obligation_id"],
                "revised",
                reviewer=me,
                note="57 patients",
                verdict=ScreeningVerdict.EXCLUDE,
                exclusion_reason="E2: fewer than 60 patients",
            )
        else:
            amendments.resolve(
                repo, PID, o["obligation_id"], "confirmed", reviewer=me, note="still eligible"
            )
    repo.save_prisma(PID, derive_prisma(repo, PID))
    ctx = repo.build_guard_context(PID)
    assert not ctx.amendment_problems and ctx.arms_for_non_included_reports == 1
    # the reported dead end: Gate 3 is blocked by the excluded report's arm
    machine = repo.load_state_machine(PID)
    assert any("not included" in v for v in machine.check(Phase.GATE3_EXTRACTION_APPROVAL, ctx))
    with pytest.raises(ReviewError, match="included at full text"):  # included data stay
        withdraw_arms(
            repo,
            PID,
            "r3",
            reviewer=me,
            reason="x",
            expected_state_sha256=arms_state_sha256(repo, PID, "r3"),
        )
    shown = arms_state_sha256(repo, PID, "r1")
    withdrawn = withdraw_arms(
        repo,
        PID,
        "r1",
        reviewer=me,
        expected_state_sha256=shown,
        reason="excluded on re-review after amendment",
    )
    assert withdrawn and all(repo.session.get(ExtractedArmORM, a) for a in withdrawn)  # kept
    record = repo.governance_records(PID, "arm_invalidation")[-1]
    assert record["kind"] == "withdrawal" and record["reviewer_id"] == me.id
    with pytest.raises(ReviewError, match="withdrawn"):  # not silently rebuilt
        build_arms(repo, PID, "r1", reviewer=me)
    # continuation: Stage 4 -> Gate 3 -> Stage 5 -> statistics
    m = repo.load_state_machine(PID)
    repo.record_transition(
        m.transition(
            Phase.GATE3_EXTRACTION_APPROVAL,
            actor=me.display_name,
            actor_type=ActorType.HUMAN,
            context=repo.build_guard_context(PID),
        )
    )
    m = repo.load_state_machine(PID)
    decision = GateDecision(
        gate=GateId.GATE_3,
        verdict=GateVerdict.APPROVED,
        reviewer_name=me.display_name,
        reviewer_id=me.id,
    )
    repo.record_transition(m.submit_gate_decision(decision, repo.build_guard_context(PID)))
    assert repo.get_project(PID).phase == Phase.STAGE5_META_ANALYSIS.value
    state = plan_service.plan_state(repo, PID)
    draft = plan_service.save_draft(
        repo,
        PID,
        [AnalysisSpec(analysis_id="md", outcome_id="hba1c", effect_measure="MD")],
        rationale=None,
        reviewer=me,
        expected_latest_sha256=state["latest_sha256"],
        expected_protocol_sha256=state["protocol_sha256"],
    )
    plan_service.approve_draft(
        repo,
        PID,
        draft.plan_id,
        reviewer=me,
        expected_sha256=draft.sha256(),
        expected_latest_sha256=draft.sha256(),
        expected_protocol_sha256=state["protocol_sha256"],
    )
    artifacts = run_stage5(repo, PID, output_dir=tmp_path / "stats")
    assert artifacts.sha256 and repo.stats_output_problems(PID) == []
    inputs = repo.load_stage5_inputs(PID)
    assert "r1" not in inputs.eligible_record_ids and all(a.record_id == "r3" for a in inputs.arms)


# =========================================================================== finding 5
def test_f5_late_changes_to_statistics_inputs_are_refused_before_writing(
    session: Session,
) -> None:
    repo, rid, _ = comparative_project(session, study_links=[["t1", "t2"]])  # Stage 6
    reviewer = repo.require_active_reviewer(rid)
    links_before = [x.model_dump(mode="json") for x in repo.load_study_links("mc1")]
    history_before = repo.governance_records("mc1", "study_link")
    assert repo.stats_output_problems("mc1") == []
    repo.save_records("mc1", [make_record("t9", "A late synthetic report")])
    attempts = [
        lambda: unlink_reports(
            repo,
            "mc1",
            ["t2"],
            reviewer=reviewer,
            note="late",
            expected_state_sha256=links_state_sha256(repo, "mc1"),
        ),
        lambda: link_reports_to_study(
            repo,
            "mc1",
            ["t3"],
            study_id="study-t1",
            primary_record_id="t3",
            reviewer=reviewer,
            note="late",
            expected_state_sha256=links_state_sha256(repo, "mc1"),
        ),
        lambda: resolve_double_counting(
            repo,
            "mc1",
            study_id="study-t1",
            outcome_id="complication",
            reviewer=reviewer,
            justification="late",
            keep_record_id="t1",
            drop_record_ids=["t2"],
        ),
        lambda: record_unavailability(
            repo, "mc1", "t3", reviewer=reviewer, reason_code="other", reason="late"
        ),
        lambda: adjudicate(
            repo,
            "mc1",
            "t9",
            ScreeningStage.FULL_TEXT,
            verdict=ScreeningVerdict.EXCLUDE,
            rationale="late",
            exclusion_reason="E1",
            reviewer=reviewer,
        ),
    ]
    for attempt in attempts:
        with pytest.raises(SysRevError, match="cannot be recomputed"):
            attempt()
    assert [x.model_dump(mode="json") for x in repo.load_study_links("mc1")] == links_before
    assert repo.governance_records("mc1", "study_link") == history_before
    assert repo.load_unavailability("mc1") == []
    assert repo.stats_output_problems("mc1") == []  # the verified statistics stay valid


def test_f5_http_unlink_in_stage6_returns_409(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'f5.db'}"
    engine = make_engine(url)
    init_db(engine)
    with make_session_factory(engine)() as s:
        _, _, token = comparative_project(s, study_links=[["t1", "t2"]])
        s.commit()
    engine.dispose()
    with TestClient(create_app(url, data_dir=tmp_path)) as c:
        view = c.get("/projects/mc1/studies").json()
        r = c.post(
            "/projects/mc1/studies/unlink",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "record_ids": ["t2"],
                "note": "late",
                "expected_state_sha256": view["state_sha256"],
            },
        )
        assert r.status_code == 409 and "cannot be recomputed" in r.json()["detail"]
        assert c.get("/projects/mc1/studies").json()["state_sha256"] == view["state_sha256"]
        assert c.get("/projects/mc1/stats/output").json()["problems"] == []


def test_f5_in_stage5_the_supported_route_is_change_then_rerun(
    session: Session, tmp_path: Path
) -> None:
    repo, rid, _ = comparative_project(session, study_links=[["t1", "t2"]])
    stage(repo, Phase.STAGE5_META_ANALYSIS, "mc1")
    unlink_reports(
        repo,
        "mc1",
        ["t2"],
        reviewer=repo.require_active_reviewer(rid),
        note="t2 is a separate trial",
        expected_state_sha256=links_state_sha256(repo, "mc1"),
    )
    assert repo.stats_output_problems("mc1")  # stale ...
    run_stage5(repo, "mc1", output_dir=tmp_path / "rerun")
    assert repo.stats_output_problems("mc1") == []  # ... and recomputed


@pytest.fixture
def session() -> Iterator[Session]:
    engine = make_engine()
    init_db(engine)
    with make_session_factory(engine)() as s:
        yield s
    engine.dispose()
