"""Session 4: AI steps through the existing runners, usage ledger, budget and API reads.

Providers are MOCKED (respx HTTP mocks or scripted providers); nothing is live.
All clinical content is SYNTHETIC.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.api_server.app import create_app
from src.core.enums import (
    DataStatus,
    Phase,
    ReviewStatus,
    ScreeningStage,
    ScreeningVerdict,
)
from src.core.exceptions import SysRevError
from src.core.schemas.ai import AIVerification
from src.fulltext import DocumentStore, confirm_pdf_match, register_upload
from src.llm import LLMConfig, LLMRequest
from src.llm.adapters import ANTHROPIC_URL
from src.llm.catalogue import ModelCatalogue
from src.llm.credentials import CredentialResolver
from src.llm.tasks import AITask, ProjectAISettings, TaskModelSettings
from src.llm.usage import AuditedProvider, BudgetGuard, LLMCallRecord, RedundantCallError
from src.screening.llm_screening import adjudicate
from src.state.db import Base, LLMCallORM
from src.state.repository import Repository
from src.workflows import AIWorkflow, WorkflowStepError
from tests.conftest import make_record
from tests.fakes import ScriptedProvider, approved_protocol, enroll, include_at_title_abstract
from tests.fixtures.pdf_factory import SYNTHETIC_DOI, SYNTHETIC_TITLE, synthetic_article

PID = "p1"
KEY = "sk-ant-workflow-test-0123456789SECRET"
CAT = ModelCatalogue.load()


def CREDS() -> CredentialResolver:
    return CredentialResolver(use_keyring=False, use_dotenv=False)


INCLUDE = {"verdict": "include", "information_sufficient": True, "rationale": "fits"}


def settings(**kw: Any) -> ProjectAISettings:
    opus = TaskModelSettings(provider="anthropic", model_id="claude-opus-5-5", effort="low",
                             max_output_tokens=4096)  # fmt: skip
    sonnet = TaskModelSettings(provider="anthropic", model_id="claude-sonnet-5-5",
                               effort="medium", max_output_tokens=4096)  # fmt: skip
    return ProjectAISettings(
        tasks={AITask.SCREENING_REVIEWER_1: opus, AITask.SCREENING_REVIEWER_2: sonnet,
               AITask.EXTRACTION: opus, AITask.VERIFICATION: sonnet,
               AITask.RISK_OF_BIAS: opus}, **kw,
    )  # fmt: skip


def setup(session: Session, tmp_path: Path, phase: Phase, **kw: Any) -> tuple[Repository, Any]:
    repo = Repository(session)
    repo.create_project(PID, "Synthetic review")
    repo.save_protocol(PID, approved_protocol())
    repo.save_records(PID, [make_record("r1", SYNTHETIC_TITLE, doi=SYNTHETIC_DOI),
                            make_record("r2", "Second synthetic record")])  # fmt: skip
    repo.get_project(PID).phase = phase.value
    repo.save_ai_settings(PID, settings(**kw))
    return repo, DocumentStore(tmp_path)


def scripted_builder(responder: Any) -> Any:
    def build(cfg: LLMConfig) -> Any:
        return ScriptedProvider(cfg.model_id, responder), CAT.validate(cfg)

    return build


def a_reply(text: str, model: str, in_tok: int = 1000, out_tok: int = 200) -> httpx.Response:
    return httpx.Response(200, json={
        "model": model, "stop_reason": "end_turn", "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": in_tok, "output_tokens": out_tok}})  # fmt: skip


# --------------------------------------------------------------------------- screening


@respx.mock
def test_screening_with_real_adapters_records_automated_decisions_and_ledger(
    session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    repo, store = setup(session, tmp_path, Phase.STAGE2_SEARCH_SCREENING)

    def reply(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return a_reply(json.dumps(INCLUDE), body["model"])

    route = respx.post(ANTHROPIC_URL).mock(side_effect=reply)
    wf = AIWorkflow(repo, PID, catalogue=CAT, credentials=CREDS(),
                    store=store)  # fmt: skip
    summary = wf.run_title_abstract_screening()
    assert summary.decided == 4 and summary.halted is None and route.call_count == 4
    decisions = repo.load_screening_decisions(PID)
    assert all(not d.is_human for d in decisions)
    assert {d.automation.model_id for d in decisions if d.automation} == {
        "claude-opus-5-5", "claude-sonnet-5-5"}  # fmt: skip
    calls = repo.load_llm_calls(PID)
    assert len(calls) == 4 and {c.status for c in calls} == {"ok"}
    assert {c.task for c in calls} == {"screening_reviewer_1", "screening_reviewer_2"}
    opus = next(c for c in calls if c.requested_model == "claude-opus-5-5")
    assert opus.effort_requested == "low" and opus.effort_effective == "low"
    assert opus.cost_usd_estimate == pytest.approx((1000 * 4 + 200 * 20) / 1e6)
    totals = repo.llm_usage_totals(PID)
    assert totals["tokens"] == 4 * 1200 and totals["successful_calls"] == 4
    # resumable: a second run makes no calls
    assert wf.run_title_abstract_screening().decided == 0 and route.call_count == 4
    # the API key never reaches the database
    for table in Base.metadata.sorted_tables:
        for row in session.execute(select(table)).all():
            assert KEY not in json.dumps([str(v) for v in row], default=str)


@respx.mock
def test_invalid_key_halts_screening_after_one_call(
    session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    repo, store = setup(session, tmp_path, Phase.STAGE2_SEARCH_SCREENING)
    route = respx.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(
            401,
            json={"type": "error", "error": {"type": "authentication_error", "message": "invalid"}},
        )
    )
    wf = AIWorkflow(repo, PID, catalogue=CAT, credentials=CREDS(),
                    store=store)  # fmt: skip
    summary = wf.run_title_abstract_screening()
    assert summary.halted and "key was rejected" in summary.halted
    assert summary.decided == 0 and route.call_count == 1
    assert repo.load_screening_decisions(PID) == []
    [call] = repo.load_llm_calls(PID)
    assert call.status == "error" and call.error_type == "ProviderAccessError"


def test_malformed_screening_reply_becomes_uncertain_never_exclude(
    session: Session, tmp_path: Path
) -> None:
    repo, store = setup(session, tmp_path, Phase.STAGE2_SEARCH_SCREENING)
    builder = scripted_builder(lambda _r: "not json")
    wf = AIWorkflow(repo, PID, catalogue=CAT, credentials=CREDS(),
                    store=store, provider_builder=builder)  # fmt: skip
    wf.run_title_abstract_screening()
    assert {d.verdict for d in repo.load_screening_decisions(PID)} == {ScreeningVerdict.UNCERTAIN}
    assert len(repo.load_conflicts(PID)) == 2  # every uncertain record goes to a human


def test_budget_stop_threshold_blocks_new_calls(session: Session, tmp_path: Path) -> None:
    repo, store = setup(session, tmp_path, Phase.STAGE2_SEARCH_SCREENING, stop_tokens=10)
    repo.record_llm_call(LLMCallRecord(
        project_id=PID, task="extraction", label="x", provider="anthropic",
        requested_model="claude-opus-5-5", status="ok", call_key="k", request_sha256="s",
        input_tokens=50, output_tokens=5))  # fmt: skip
    built: list[ScriptedProvider] = []

    def build(cfg: LLMConfig) -> Any:
        built.append(ScriptedProvider(cfg.model_id, [json.dumps(INCLUDE)] * 10))
        return built[-1], CAT.validate(cfg)

    wf = AIWorkflow(repo, PID, catalogue=CAT, credentials=CREDS(),
                    store=store, provider_builder=build)  # fmt: skip
    summary = wf.run_title_abstract_screening()
    assert summary.halted and "stop threshold" in summary.halted
    assert len(built) == 2 and all(p.requests == [] for p in built)
    assert [c.status for c in repo.load_llm_calls(PID)][-1] == "blocked_budget"


def test_cost_stop_threshold_requires_priced_model(session: Session, tmp_path: Path) -> None:
    repo, store = setup(session, tmp_path, Phase.STAGE2_SEARCH_SCREENING, stop_usd=1.0)
    (tmp_path / "model_catalogue.json").write_text(json.dumps({"models": [{
        "provider": "openai", "model_id": "local-unpriced", "efforts": ["low"],
        "display_name": "Local verified, unpriced",
        "source": "https://example.invalid/model-docs (synthetic test entry)",
        "verified": True}]}), encoding="utf-8")  # fmt: skip
    cat = ModelCatalogue.load(tmp_path)
    unpriced = TaskModelSettings(provider="openai", model_id="local-unpriced")
    repo.save_ai_settings(PID, settings(stop_usd=1.0).model_copy(update={"tasks": {
        AITask.SCREENING_REVIEWER_1: unpriced,
        AITask.SCREENING_REVIEWER_2: unpriced.model_copy(update={"effort": "low"})}}))  # fmt: skip

    def build(cfg: LLMConfig) -> Any:
        return ScriptedProvider(cfg.model_id, []), cat.validate(cfg)

    wf = AIWorkflow(repo, PID, catalogue=cat, credentials=CREDS(), store=store,
                    provider_builder=build)  # fmt: skip
    with pytest.raises(SysRevError, match="no list price"):
        wf.run_title_abstract_screening()


def test_steps_refuse_wrong_phase_and_missing_settings(session: Session, tmp_path: Path) -> None:
    repo, store = setup(session, tmp_path, Phase.STAGE1_PROTOCOL)
    wf = AIWorkflow(repo, PID, catalogue=CAT, credentials=CREDS(),
                    store=store, provider_builder=scripted_builder([]))  # fmt: skip
    with pytest.raises(WorkflowStepError, match="stage2"):
        wf.run_title_abstract_screening()
    with pytest.raises(WorkflowStepError, match="stage4"):
        wf.run_extraction(["r1"])
    repo.create_project("p2", "no settings")
    with pytest.raises(WorkflowStepError, match="no AI model settings"):
        AIWorkflow(repo, "p2", catalogue=CAT, credentials=CREDS(),
                   store=store).settings()  # fmt: skip


def test_preflight_reports_missing_key_without_calling(
    session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    repo, store = setup(session, tmp_path, Phase.STAGE2_SEARCH_SCREENING)
    wf = AIWorkflow(repo, PID, catalogue=CAT, credentials=CREDS(),
                    store=store)  # fmt: skip
    [problem] = wf.preflight(AITask.EXTRACTION)
    assert "no API key for anthropic" in problem and "subscription" in problem
    est = wf.estimate(AITask.EXTRACTION, [4000, 8000])
    assert est["calls"] == 2 and est["input_tokens_estimate"] == 3000
    assert est["cost_usd_upper_bound"] == pytest.approx((3000 * 4 + 2 * 4096 * 20) / 1e6, abs=1e-4)


# --------------------------------------------------------------------------- extraction


def stage4(session: Session, tmp_path: Path) -> tuple[Repository, DocumentStore]:
    repo, store = setup(session, tmp_path, Phase.STAGE4_EXTRACTION_ROB)
    recs = repo.load_records(PID)
    include_at_title_abstract(repo, PID, recs[:1])
    reviewer = enroll(repo)
    item = register_upload(repo, store, PID, "r1", synthetic_article(),
                           original_filename="r1.pdf", reviewer=reviewer)  # fmt: skip
    if item.match_unconfirmed:
        confirm_pdf_match(repo, PID, "r1", reviewer=reviewer, note="checked")
    adjudicate(repo, PID, "r1", ScreeningStage.FULL_TEXT, verdict=ScreeningVerdict.INCLUDE,
               rationale="eligible", reviewer=reviewer)  # fmt: skip
    return repo, store


EXTRACTION_REPLY = {"values": [{
    "domain": "outcome_data", "field_name": "events", "outcome_id": "obliteration",
    "arm_label": "SRS cohort", "arm_role": "single_group", "timepoint": "last follow-up",
    "follow_up": {"kind": "last_follow_up"}, "data_status": "reported", "reported_text": "45",
    "page": 2, "location_id": "p2.b2",
    "quote": "Complete obliteration was achieved in 45 of 57 patients (78.9%)."}]}  # fmt: skip


def test_extraction_and_advisory_verification_never_approve(
    session: Session, tmp_path: Path
) -> None:
    repo, store = stage4(session, tmp_path)

    def respond(request: LLMRequest) -> Any:
        if "check one data item" in request.system:
            return {"verdict": "supports", "rationale": "quote states 45 of 57"}
        return EXTRACTION_REPLY

    wf = AIWorkflow(repo, PID, catalogue=CAT, credentials=CREDS(),
                    store=store, provider_builder=scripted_builder(respond))  # fmt: skip
    report = wf.run_extraction(["r1", "r2"])
    assert report.done == ["r1: 1 proposal(s), pending human review"]
    assert report.failed and report.failed[0].startswith("r2")  # no PDF → per-record failure
    [item] = repo.load_extraction_items(PID)
    assert item.review_status == ReviewStatus.PENDING and item.data_status == DataStatus.REPORTED
    assert item.verification is not None and item.verification.acceptable
    assert item.proposer_identity.startswith("llm:anthropic/claude-opus-5-5")
    # redundant re-run is skipped without a call
    again = wf.run_extraction(["r1"])
    assert again.skipped and not again.done

    v = wf.run_verification()
    assert v.done == [f"{item.item_id}: supports (advisory)"]
    [stored] = repo.load_ai_verifications(PID)
    assert isinstance(stored, AIVerification) and stored.is_human is False
    assert stored.call_id is not None and stored.model_id == "claude-sonnet-5-5"
    assert repo.get_extraction_item(PID, item.item_id).review_status == ReviewStatus.PENDING
    assert wf.run_verification().skipped  # not repeated


def test_verifier_with_same_model_as_extractor_is_flagged(session: Session, tmp_path: Path) -> None:
    repo, store = stage4(session, tmp_path)
    same = settings()
    same.tasks[AITask.VERIFICATION] = same.tasks[AITask.EXTRACTION]
    repo.save_ai_settings(PID, same)
    wf = AIWorkflow(repo, PID, catalogue=CAT, credentials=CREDS(),
                    store=store, provider_builder=scripted_builder([]))  # fmt: skip
    assert any("not an independent check" in w for w in wf.run_verification().warnings)


# --------------------------------------------------------------------------- audited provider


def test_audited_provider_dedupes_and_records_failures() -> None:
    records: list[LLMCallRecord] = []
    inner = ScriptedProvider("claude-opus-5-5", [RuntimeError(f"boom {KEY}"), "{}", "{}"])
    seen: set[str] = set()
    audited = AuditedProvider(
        inner, project_id=PID, task="extraction", label="extraction", effort="low",
        capability=CAT.get("anthropic", "claude-opus-5-5"), sink=records.append,
        budget=BudgetGuard(spent_usd=0, tokens=0, warn_tokens=0),
        already_answered=lambda key: key in seen,
    )  # fmt: skip
    req = LLMRequest(system="s", prompt="p")
    with pytest.raises(RuntimeError):
        audited.complete(req)
    assert records[0].status == "error" and KEY not in (records[0].error_message or "")
    audited.complete(req)
    assert records[1].status == "ok"
    seen.add(records[1].call_key)
    with pytest.raises(RedundantCallError):
        audited.complete(req)
    assert records[2].status == "skipped_duplicate" and len(inner.requests) == 2
    other_task = AuditedProvider(
        inner, project_id=PID, task="verification", label="v", effort="low", capability=None,
        sink=records.append,
    )  # fmt: skip
    assert other_task.complete(req) and records[3].call_key != records[1].call_key


def test_ledger_rows_are_append_only(session: Session) -> None:
    repo = Repository(session)
    repo.create_project(PID, "x")
    repo.record_llm_call(LLMCallRecord(
        project_id=PID, task="t", label="t", provider="openai", requested_model="m",
        status="ok", call_key="k", request_sha256="s"))  # fmt: skip
    row = session.scalars(select(LLMCallORM)).one()
    row.status = "error"
    with pytest.raises(Exception, match="append-only"):
        session.flush()


# --------------------------------------------------------------------------- API


def test_api_lists_projects_reads_usage_and_advance_requires_auth(tmp_path: Path) -> None:
    client = TestClient(create_app("sqlite:///:memory:", tmp_path))
    pid = client.post("/projects", json={"title": "UI project"}).json()["project_id"]
    assert [p["project_id"] for p in client.get("/projects").json()] == [pid]
    r = client.post(f"/projects/{pid}/advance", json={"target": "gate1_search_strategy"})
    assert r.status_code == 401
    r = client.post(f"/projects/{pid}/advance", json={"target": "gate1_search_strategy"},
                    headers={"Authorization": "Bearer rvw_bogus"})  # fmt: skip
    assert r.status_code == 401
    usage = client.get(f"/projects/{pid}/ai/usage").json()
    assert usage["totals"]["calls"] == 0 and "estimates" in usage["note"]
    assert client.get(f"/projects/{pid}/ai/settings").status_code == 404
    assert client.get(f"/projects/{pid}/screening/title_abstract/decisions").json() == []
    assert client.get(f"/projects/{pid}/extraction/ai-verifications").json() == []


def test_api_advance_is_guarded_and_attributed_to_the_human(tmp_path: Path) -> None:
    client = TestClient(create_app("sqlite:///:memory:", tmp_path))
    pid = client.post("/projects", json={"title": "P"}).json()["project_id"]
    factory = client.app.state.session_factory  # type: ignore[attr-defined]
    with factory() as s:
        _, token = Repository(s).enroll_reviewer("Dr Human")
        s.commit()
    auth = {"Authorization": f"Bearer {token}"}
    r = client.post(f"/projects/{pid}/advance", json={"target": "stage3_fulltext_retrieval"},
                    headers=auth)  # fmt: skip
    assert r.status_code == 409  # illegal transition from Stage 1
    r = client.post(f"/projects/{pid}/advance", json={"target": "gate1_search_strategy"},
                    headers=auth)  # fmt: skip
    assert r.status_code == 200 and r.json()["phase"] == "gate1_search_strategy"
    last = r.json()["history"][-1]
    assert last["actor"] == "Dr Human" and last["actor_type"] == "human"
    # a gate can only be left through an authenticated gate decision
    r = client.post(f"/projects/{pid}/advance", json={"target": "stage2_search_screening"},
                    headers=auth)  # fmt: skip
    assert r.status_code == 409 and "gate" in r.json()["detail"]
    with factory() as s:
        repo = Repository(s)
        repo.get_project(pid).phase = Phase.STAGE2_SEARCH_SCREENING.value
        s.commit()
    r = client.post(f"/projects/{pid}/advance", json={"target": "gate1_5_screening_conflicts"},
                    headers=auth)  # fmt: skip
    assert r.status_code == 409 and r.json()["detail"]["violations"]  # guards fail closed


def test_settings_validation_and_no_secret_fields() -> None:
    with pytest.raises(ValueError, match="warn_usd"):
        ProjectAISettings(warn_usd=5, stop_usd=1)
    with pytest.raises(ValueError, match="api_key"):
        TaskModelSettings(provider="anthropic", model_id="m", api_key="x")  # type: ignore[call-arg]
    assert "key" not in json.dumps(settings().model_dump(mode="json")).lower()


# --------------------------------------------------------------------------- review fixes (PR #8)


def _record(provider: str, **tokens: int) -> LLMCallRecord:
    return LLMCallRecord(project_id=PID, task="t", label="t", provider=provider,
                         requested_model="m", status="ok", call_key="k", request_sha256="s",
                         **tokens)  # fmt: skip


def test_openai_cached_tokens_are_not_double_counted(session: Session) -> None:
    rec = _record("openai", input_tokens=2000, output_tokens=800, cache_read_tokens=500,
                  reasoning_tokens=600)  # fmt: skip
    assert rec.total_tokens == 2800  # cached ⊂ input, reasoning ⊂ output
    assert (rec.input_tokens, rec.cache_read_tokens, rec.reasoning_tokens) == (2000, 500, 600)
    repo = Repository(session)
    repo.create_project(PID, "x")
    repo.record_llm_call(rec)
    assert repo.llm_usage_totals(PID)["tokens"] == 2800
    assert session.scalars(select(LLMCallORM.total_tokens)).one() == 2800
    guard = BudgetGuard(spent_usd=0, tokens=0, stop_tokens=3000)
    guard.add(rec)
    guard.check()  # 2800 < 3000: the old 3300 count would have paused the run


def test_anthropic_cache_categories_are_disjoint_and_summed() -> None:
    rec = _record("anthropic", input_tokens=1000, output_tokens=500, cache_read_tokens=2000,
                  cache_write_tokens=100)  # fmt: skip
    assert rec.total_tokens == 3600
    assert _record("scripted", input_tokens=10, output_tokens=5,
                   cache_read_tokens=7).total_tokens == 15  # fmt: skip


def test_malformed_extraction_reply_does_not_block_an_ordinary_retry(
    session: Session, tmp_path: Path
) -> None:
    repo, store = stage4(session, tmp_path)
    replies: list[Any] = ['{"values": [', EXTRACTION_REPLY]  # truncated JSON, then valid
    wf = AIWorkflow(repo, PID, catalogue=CAT, credentials=CREDS(), store=store,
                    provider_builder=scripted_builder(lambda _r: replies.pop(0)))  # fmt: skip
    first = wf.run_extraction(["r1"])
    assert first.failed and not first.done and repo.load_extraction_items(PID) == []
    [call] = repo.load_llm_calls(PID)
    assert call.status == "ok"  # the provider answered and billed: kept in the audit trail
    assert not repo.has_persisted_result(PID, call.call_key)

    second = wf.run_extraction(["r1"])  # no force needed
    assert second.done == ["r1: 1 proposal(s), pending human review"] and not second.skipped
    [item] = repo.load_extraction_items(PID)
    assert item.review_status == ReviewStatus.PENDING  # still needs the human
    calls = repo.load_llm_calls(PID)
    assert [c.status for c in calls] == ["ok", "ok"] and calls[0].call_key == calls[1].call_key
    assert repo.has_persisted_result(PID, calls[1].call_key)

    third = wf.run_extraction(["r1"])  # a validated result exists: no call, no duplicate
    assert third.skipped and not third.done
    assert len(repo.load_extraction_items(PID)) == 1 and len(repo.load_llm_calls(PID)) == 2


def test_identical_request_with_persisted_result_is_not_repeated(
    session: Session, tmp_path: Path
) -> None:
    repo, store = stage4(session, tmp_path)
    responses = [EXTRACTION_REPLY, EXTRACTION_REPLY]
    wf = AIWorkflow(repo, PID, catalogue=CAT, credentials=CREDS(), store=store,
                    provider_builder=scripted_builder(lambda _r: responses.pop(0)))  # fmt: skip
    assert wf.run_extraction(["r1"]).done
    again = wf.run_extraction(["r1"])  # validated result saved: no second paid call
    assert repo.has_persisted_result(PID, repo.load_llm_calls(PID)[0].call_key)
    assert again.skipped and len(responses) == 1


def test_empty_extraction_reply_is_not_a_completed_extraction(
    session: Session, tmp_path: Path
) -> None:
    repo, store = stage4(session, tmp_path)
    replies: list[Any] = [{"values": []}, EXTRACTION_REPLY]
    wf = AIWorkflow(repo, PID, catalogue=CAT, credentials=CREDS(), store=store,
                    provider_builder=scripted_builder(lambda _r: replies.pop(0)))  # fmt: skip
    first = wf.run_extraction(["r1"])
    assert not first.done and first.failed and "no values" in first.failed[0]
    assert repo.load_extraction_items(PID) == []
    [call] = repo.load_llm_calls(PID)
    assert call.status == "ok" and call.input_tokens is None  # billed reply stays logged
    assert not repo.has_persisted_result(PID, call.call_key)
    second = wf.run_extraction(["r1"])  # ordinary retry, no force
    assert second.done == ["r1: 1 proposal(s), pending human review"]
    [item] = repo.load_extraction_items(PID)
    assert item.review_status == ReviewStatus.PENDING
    assert len(repo.load_llm_calls(PID)) == 2
