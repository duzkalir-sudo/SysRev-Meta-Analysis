"""Regressions for the four findings of the independent follow-up review of PR #13 (commit
e7dd405): inconsistent screen snapshots and late-stage mutations.

The interleavings are deterministic: another *real* SQLite session (file database, separate
connection) commits at exactly the point between two reads, by wrapping the first read.
Authentication, validation and the save paths are the real ones. All data SYNTHETIC; no
network call is made.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

import src.api_server.review_routes as review_routes
from src.api_server.app import create_app
from src.core.enums import ActorType, Phase, ScreeningStage, ScreeningVerdict
from src.core.schemas.stats import AnalysisSpec
from src.extraction import (
    accept_item,
    correction_state_sha256,
    links_state_sha256,
    resolve_double_counting,
    source_state,
    submit_proposals,
    unlink_reports,
)
from src.fulltext import confirm_pdf_match, register_upload
from src.fulltext.storage import DocumentStore
from src.pdf_extractor import parse_pdf
from src.screening.agreement import screening_status
from src.screening.llm_screening import adjudicate
from src.search import amendments
from src.search import protocol_editor as editor
from src.state.db import (
    ExtractionItemORM,
    ExtractionReviewEventORM,
    ScreeningConflictORM,
    ScreeningDecisionORM,
    init_db,
    make_engine,
    make_session_factory,
)
from src.state.repository import Repository
from src.stats_engine import plan_service
from src.stats_engine.engine import run_plan
from src.stats_engine.stage5 import run_stage5
from tests.conftest import NOW, approved_by, make_record
from tests.fakes import include_at_title_abstract
from tests.fixtures.pdf_factory import SYNTHETIC_DOI, SYNTHETIC_TITLE, synthetic_article
from tests.manuscript_helpers import comparative_project
from tests.unit.test_master_table import km

PDF_A = synthetic_article()
PDF_B = PDF_A + b"\n% revised edition (synthetic)\n"  # valid PDF, different SHA-256


@dataclass
class Db:
    url: str
    tmp: Path
    factory: sessionmaker[Session]

    def session(self) -> Session:
        return self.factory()


@pytest.fixture
def db(tmp_path: Path) -> Iterator[Db]:
    url = f"sqlite:///{tmp_path / 'followup.db'}"  # a file: every session has its own connection
    engine = make_engine(url)
    init_db(engine)
    yield Db(url, tmp_path, make_session_factory(engine))
    engine.dispose()


def once(action: Callable[[], None]) -> Callable[[], None]:
    done: list[bool] = []

    def run() -> None:
        if not done:
            done.append(True)
            action()

    return run


def counts(s: Session, *models: Any) -> tuple[int, ...]:
    return tuple(s.scalar(select(func.count()).select_from(m)) or 0 for m in models)


# =========================================================================== finding 1
def test_pdf_screen_hash_must_describe_the_source_in_the_same_response(
    db: Db, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = DocumentStore(db.tmp / "fulltext")  # the API app's store for this data dir
    with db.session() as s:
        repo = Repository(s)
        repo.create_project("p", "Synthetic correction review")
        rid, token = repo.enroll_reviewer("Dr Corrector")
        reviewer = repo.require_active_reviewer(rid)
        repo.save_protocol("p", approved_by(rid, name="Dr Corrector"))
        rec = make_record("r1", SYNTHETIC_TITLE, doi=SYNTHETIC_DOI)
        repo.save_records("p", [rec])
        include_at_title_abstract(repo, "p", [rec])
        linked = register_upload(
            repo, store, "p", "r1", PDF_A, original_filename="a.pdf", reviewer=reviewer
        )
        if linked.match_unconfirmed:
            confirm_pdf_match(repo, "p", "r1", reviewer=reviewer, note="checked")
        adjudicate(
            repo,
            "p",
            "r1",
            ScreeningStage.FULL_TEXT,
            verdict=ScreeningVerdict.INCLUDE,
            rationale="eligible",
            reviewer=reviewer,
        )
        ft = repo.get_fulltext_item("p", "r1")
        assert ft is not None and ft.pdf_filename and ft.pdf_sha256
        doc = parse_pdf(
            store.read_verified(ft.pdf_filename, ft.pdf_sha256), filename=ft.pdf_filename
        )
        (item,) = submit_proposals(
            repo,
            "p",
            "r1",
            doc,
            [km("reported_estimate", "62.1", 1, 1, 3)],
            proposed_by=ActorType.AGENT,
            proposer_identity="llm:t",
        )
        accept_item(repo, "p", item.item_id, reviewer=reviewer, note=None)
        repo.get_project("p").phase = Phase.STAGE4_EXTRACTION_ROB.value
        s.commit()
        pdf_a = ft.pdf_sha256

    def upload_b() -> None:  # another reviewer's real session replaces and confirms the PDF
        with db.session() as s2:
            r2 = Repository(s2)
            rv2 = r2.require_active_reviewer(rid)
            current = r2.get_fulltext_item("p", "r1")
            replaced = register_upload(
                r2,
                store,
                "p",
                "r1",
                PDF_B,
                original_filename="b.pdf",
                reviewer=rv2,
                replaces_sha256=current.pdf_sha256 if current else None,
            )
            if replaced.match_unconfirmed:
                confirm_pdf_match(r2, "p", "r1", reviewer=rv2, note="revised edition checked")
            s2.commit()

    interleave = once(upload_b)
    original = review_routes.source_state

    def read_then_interleave(repo: Repository, pid: str, record_id: str) -> dict[str, Any]:
        captured = original(repo, pid, record_id)  # the source the response will display
        interleave()  # ... and B becomes current before anything else is read
        return captured

    monkeypatch.setattr(review_routes, "source_state", read_then_interleave)
    headers = {"Authorization": f"Bearer {token}"}
    with TestClient(create_app(db.url, data_dir=db.tmp)) as c:
        shown = c.get("/projects/p/extraction/items").json()[0]
        monkeypatch.setattr(review_routes, "source_state", original)
        assert shown["source"]["pdf_sha256"] == pdf_a  # the screen displays PDF A ...
        with db.session() as s:  # ... and its hash describes exactly that displayed source
            repo = Repository(s)
            current = repo.get_fulltext_item("p", "r1")
            assert current is not None and current.pdf_sha256 != pdf_a  # B is current now
            displayed_item = repo.get_extraction_item("p", shown["item_id"])
            assert shown["state_sha256"] == correction_state_sha256(
                repo, "p", displayed_item, shown["source"]
            )
            before = counts(s, ExtractionItemORM, ExtractionReviewEventORM)
        body = {
            "correction": km("reported_estimate", "81.4", 1, 2, 3).model_dump(mode="json"),
            "note": "decided on the screen showing PDF A",
            "expected_state_sha256": shown["state_sha256"],
        }
        path = f"/projects/p/extraction/items/{shown['item_id']}/correct"
        refused = c.post(path, json=body, headers=headers)
        assert refused.status_code == 409 and "has changed" in refused.json()["detail"]
        with db.session() as s:  # nothing saved; B and the other reviewer's work intact
            assert counts(s, ExtractionItemORM, ExtractionReviewEventORM) == before
            repo = Repository(s)
            assert repo.get_extraction_item("p", shown["item_id"]).review_status == "accepted"
            ft_now = repo.get_fulltext_item("p", "r1")
            assert ft_now is not None and ft_now.pdf_sha256 == current.pdf_sha256
        # control: after a reload the response is consistent around B and is accepted
        reloaded = c.get("/projects/p/extraction/items").json()[0]
        assert reloaded["source"]["pdf_sha256"] == current.pdf_sha256
        ok = c.post(
            path, headers=headers, json=body | {"expected_state_sha256": reloaded["state_sha256"]}
        )
        assert ok.status_code == 200, ok.json()
        assert ok.json()["evidence"]["pdf_sha256"] == current.pdf_sha256


def test_source_state_hash_refuses_a_source_of_another_report(db: Db) -> None:
    with db.session() as s:
        repo, _, _ = comparative_project(s)
        from src.core.enums import ArmRole, DataStatus, ExtractionDomain
        from src.core.schemas.master import ExtractionItem

        item = ExtractionItem(
            record_id="t1",
            domain=ExtractionDomain.OUTCOME_DATA,
            field_name="events",
            outcome_id="complication",
            arm_label="X",
            arm_role=ArmRole.INTERVENTION,
            data_status=DataStatus.NOT_REPORTED,
            proposed_by=ActorType.AGENT,
            proposer_identity="llm:t",
        )
        with pytest.raises(ValueError, match="another report"):
            correction_state_sha256(repo, "mc1", item, source_state(repo, "mc1", "t2"))


# =========================================================================== finding 2
def _narrow(db: Db, rid: str) -> None:
    with db.session() as s2:
        r2 = Repository(s2)
        rv2 = r2.require_active_reviewer(rid)
        base = r2.latest_protocol("mc1")
        assert base is not None
        data = base.model_dump(mode="json")
        data["outcomes"][0]["name"] = "Serious complication requiring re-operation"
        a = amendments.propose(
            r2,
            "mc1",
            data,
            rationale="narrowed",
            reviewer=rv2,
            expected_base_sha256=editor.protocol_sha256(base),
        )
        amendments.decide(
            r2,
            "mc1",
            a.amendment_id,
            "approved",
            reviewer=rv2,
            comments="ok",
            expected_sha256=a.sha256,
        )
        s2.commit()


def test_plan_screen_hash_must_describe_the_outcomes_in_the_same_response(
    db: Db, monkeypatch: pytest.MonkeyPatch
) -> None:
    with db.session() as s:
        repo, rid, token = comparative_project(s)
        reviewer = repo.require_active_reviewer(rid)
        repo.get_project("mc1").phase = Phase.STAGE4_EXTRACTION_ROB.value
        st = plan_service.plan_state(repo, "mc1")
        plan_service.save_draft(
            repo,
            "mc1",
            [AnalysisSpec(analysis_id="or", outcome_id="complication", effect_measure="OR")],
            rationale="odds ratio",
            reviewer=reviewer,
            expected_latest_sha256=st["latest_sha256"],
            expected_protocol_sha256=st["protocol_sha256"],
        )
        s.commit()
        v1_sha = editor.protocol_sha256(repo.latest_protocol("mc1"))  # type: ignore[arg-type]

    interleave = once(lambda: _narrow(db, rid))
    original = Repository.latest_protocol

    def read_then_interleave(self: Repository, project_id: str) -> Any:
        captured = original(self, project_id)  # the protocol this response captures ...
        interleave()  # ... then the amendment is approved by another session
        return captured

    monkeypatch.setattr(Repository, "latest_protocol", read_then_interleave)
    headers = {"Authorization": f"Bearer {token}"}
    with TestClient(create_app(db.url, data_dir=db.tmp)) as c:
        shown = c.get("/projects/mc1/stats/plans").json()
        monkeypatch.setattr(Repository, "latest_protocol", original)
        # one captured protocol: old definitions, old version, old hash — never mixed
        assert shown["outcomes"][0]["name"] == "Any complication"
        assert (shown["protocol_version"], shown["protocol_sha256"]) == (1, v1_sha)

        def obligation() -> str:
            with db.session() as s:
                return str(
                    next(
                        o["state"]
                        for o in amendments.obligation_status(Repository(s), "mc1")
                        if o["area"] == "analysis_plans"
                    )
                )

        assert obligation() == "open"
        with db.session() as s:
            before = (
                len(plan_service.list_plans(Repository(s), "mc1")),
                len(Repository(s).governance_records("mc1", "plan_approval_basis")),
            )
        draft = shown["plans"][-1]
        path = f"/projects/mc1/stats/plan/drafts/{draft['plan_id']}/approve"
        stale = {
            "expected_sha256": draft["sha256"],
            "expected_latest_sha256": shown["latest_sha256"],
            "expected_protocol_sha256": shown["protocol_sha256"],
        }
        refused = c.post(path, json=stale, headers=headers)
        assert refused.status_code == 409 and "has changed" in refused.json()["detail"]
        with db.session() as s:  # nothing written; the re-review stays open
            after = (
                len(plan_service.list_plans(Repository(s), "mc1")),
                len(Repository(s).governance_records("mc1", "plan_approval_basis")),
            )
        assert after == before and obligation() == "open"
        # control: a consistent reload shows the amended definitions and is accepted
        current = c.get("/projects/mc1/stats/plans").json()
        assert current["outcomes"][0]["name"] == "Serious complication requiring re-operation"
        assert current["protocol_version"] == 2
        ok = c.post(
            path,
            headers=headers,
            json=stale
            | {
                "expected_latest_sha256": current["latest_sha256"],
                "expected_protocol_sha256": current["protocol_sha256"],
            },
        )
        assert ok.status_code == 201, ok.json()
        assert obligation() == "resolved"


# =========================================================================== finding 3
def test_late_http_title_abstract_exclusion_must_be_refused(db: Db) -> None:
    with db.session() as s:
        repo, _, token = comparative_project(s)  # Stage 6; TA = automated consensus include
        s.commit()
        decisions_before = [d.model_dump(mode="json") for d in repo.load_screening_decisions("mc1")]
        rows_before = counts(s, ScreeningDecisionORM, ScreeningConflictORM)
    headers = {"Authorization": f"Bearer {token}"}
    with TestClient(create_app(db.url, data_dir=db.tmp)) as c:
        r = c.post(
            "/projects/mc1/screening/title_abstract/records/t1/adjudication",
            headers=headers,
            json={
                "verdict": "exclude",
                "rationale": "late",
                "exclusion_reason": "E1: late reconsideration",
            },
        )
        assert r.status_code == 409 and "approved at Gate 3" in r.json()["detail"]  # Session 9
    with db.session() as s:
        repo = Repository(s)
        assert counts(s, ScreeningDecisionORM, ScreeningConflictORM) == rows_before
        assert [d.model_dump(mode="json") for d in repo.load_screening_decisions("mc1")] == (
            decisions_before
        )
        ta = screening_status(
            {"t1"}, repo.load_screening_decisions("mc1"), [], ScreeningStage.TITLE_ABSTRACT
        )
        assert ta.final == {"t1": ScreeningVerdict.INCLUDE}
        assert "t1" in repo.load_stage5_inputs("mc1").eligible_record_ids
        assert repo.stats_output_problems("mc1") == []


def test_title_abstract_decision_in_stage5_is_refused_after_gate3(db: Db) -> None:
    """Session 9 (H2) deliberately reverses the earlier "still supported in Stage 5" rule:
    an eligibility decision after Gate 3 changes the set of reports a human approved at
    Gate 3 (extraction, risk of bias and PRISMA would no longer have been reviewed for it),
    and the pipeline has no route back to Stage 4. It is refused before anything is written;
    the Stage 4 / Gate 3 control below shows the decision itself is still accepted there."""
    with db.session() as s:
        repo, _, token = comparative_project(s)
        repo.get_project("mc1").phase = Phase.STAGE5_META_ANALYSIS.value
        s.commit()
        rows_before = counts(s, ScreeningDecisionORM, ScreeningConflictORM)
    body = {"verdict": "include", "rationale": "confirmed by a human"}
    headers = {"Authorization": f"Bearer {token}"}
    path = "/projects/mc1/screening/title_abstract/records/t1/adjudication"
    with TestClient(create_app(db.url, data_dir=db.tmp)) as c:
        r = c.post(path, headers=headers, json=body)
        assert r.status_code == 409 and "approved at Gate 3" in r.json()["detail"]
    with db.session() as s:
        assert counts(s, ScreeningDecisionORM, ScreeningConflictORM) == rows_before
        Repository(s).get_project("mc1").phase = Phase.GATE3_EXTRACTION_APPROVAL.value
        s.commit()
    with TestClient(create_app(db.url, data_dir=db.tmp)) as c:
        assert c.post(path, headers=headers, json=body).status_code == 201


# =========================================================================== finding 4
def _stage5_with_link(db: Db) -> str:
    with db.session() as s:
        repo, rid, _ = comparative_project(s, study_links=[["t1", "t2"]])
        repo.get_project("mc1").phase = Phase.STAGE5_META_ANALYSIS.value
        # Session 9 (H2): Stage 5 -> 6 re-checks Gate 3, so the double counting the link
        # creates is resolved (a decision that stays possible in Stage 5)
        resolve_double_counting(
            repo, "mc1", study_id="study-t1", outcome_id="complication",
            reviewer=repo.require_active_reviewer(rid),
            justification="synthetic: distinct recruitment periods",
            non_overlapping_record_ids=["t1", "t2"],
        )  # fmt: skip
        approved = repo.latest_analysis_plan("mc1")
        assert approved is not None  # the decision changed the inputs: re-run Stage 5
        repo.record_stats_output(
            "mc1",
            run_plan(approved, repo.load_stage5_inputs("mc1"), project_id="mc1", generated_at=NOW),
        )
        s.commit()
    return rid


def test_cached_stage5_project_must_not_allow_unlink_after_stage6_transition(db: Db) -> None:
    rid = _stage5_with_link(db)
    session_a = db.session()
    repo_a = Repository(session_a)
    cached = repo_a.get_project("mc1")  # retained across the commit (expire_on_commit=False)
    assert cached.phase == Phase.STAGE5_META_ANALYSIS.value
    session_a.commit()
    with db.session() as sb:  # session B: a real state-machine transition to Stage 6
        rb = Repository(sb)
        reviewer_b = rb.require_active_reviewer(rid)
        machine = rb.load_state_machine("mc1")
        rb.record_transition(
            machine.transition(
                Phase.STAGE6_MANUSCRIPT,
                actor=reviewer_b.display_name,
                actor_type=ActorType.HUMAN,
                context=rb.build_guard_context("mc1"),
            )
        )
        sb.commit()
    links_before = [x.model_dump(mode="json") for x in repo_a.load_study_links("mc1")]
    history_before = repo_a.governance_records("mc1", "study_link")
    with pytest.raises(Exception, match="cannot be recomputed"):
        unlink_reports(
            repo_a,
            "mc1",
            ["t2"],
            reviewer=repo_a.require_active_reviewer(rid),
            note="from a session that still believes Stage 5",
            expected_state_sha256=links_state_sha256(repo_a, "mc1"),
        )
    session_a.rollback()
    session_a.close()
    with db.session() as fresh:
        repo = Repository(fresh)
        assert repo.get_project("mc1").phase == Phase.STAGE6_MANUSCRIPT.value
        assert [x.model_dump(mode="json") for x in repo.load_study_links("mc1")] == links_before
        assert repo.governance_records("mc1", "study_link") == history_before
        assert repo.stats_output_problems("mc1") == []


def test_cached_stage5_project_still_allows_unlink_while_in_stage5(db: Db) -> None:
    rid = _stage5_with_link(db)
    session_a = db.session()
    repo_a = Repository(session_a)
    cached = repo_a.get_project("mc1")
    session_a.commit()
    unlink_reports(
        repo_a,
        "mc1",
        ["t2"],
        reviewer=repo_a.require_active_reviewer(rid),
        note="t2 is a separate trial",
        expected_state_sha256=links_state_sha256(repo_a, "mc1"),
    )
    session_a.commit()
    assert cached.phase == Phase.STAGE5_META_ANALYSIS.value
    assert repo_a.stats_output_problems("mc1")  # stale, and the supported route works:
    run_stage5(repo_a, "mc1", output_dir=db.tmp / "rerun")
    session_a.commit()
    assert repo_a.stats_output_problems("mc1") == []
    session_a.close()
