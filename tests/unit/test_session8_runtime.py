"""Session 8 runtime safeguards for issues #3/#4 (mocked providers, SYNTHETIC data only).

Parse-once cache invalidation, independence of the two screening passes under caching,
catalogue provenance on every call, unknown-price reporting, bounded retries and a budget
pause that preserves state. No measurement of savings is claimed here: these tests check
correctness, not cost.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from sqlalchemy.orm import Session

from src.core.enums import Phase
from src.llm.adapters import ANTHROPIC_URL
from src.llm.provider import LLMConfig, LLMRequest, ProviderUnavailableError
from src.llm.tasks import AITask, build_api_provider
from src.llm.usage import AuditedProvider, BudgetGuard, LLMCallRecord, call_key
from src.pdf_extractor.cache import PARSER_ID, ParsedDocumentCache
from src.pdf_extractor.parser import HashMismatchError, sha256_bytes
from src.workflows import AIWorkflow
from tests.fakes import ScriptedProvider
from tests.fixtures.pdf_factory import synthetic_article
from tests.unit.test_ai_workflow import CAT, CREDS, INCLUDE, KEY, PID, a_reply, settings, setup

PDF = synthetic_article()
OTHER = synthetic_article(title="Another synthetic report", doi="10.5555/other.0003")


# --------------------------------------------------------------------------- parse once
def test_parse_cache_reuses_only_identical_bytes_filename_and_parser() -> None:
    cache = ParsedDocumentCache(maxsize=2)
    sha = sha256_bytes(PDF)
    first = cache.parse(PDF, filename="a.pdf", expected_sha256=sha)
    assert cache.parse(PDF, filename="a.pdf", expected_sha256=sha) is first
    assert (cache.hits, cache.misses) == (1, 1)
    renamed = cache.parse(PDF, filename="b.pdf", expected_sha256=sha)  # cited under b.pdf
    assert renamed is not first and renamed.pdf_filename == "b.pdf"
    replaced = cache.parse(OTHER, filename="a.pdf", expected_sha256=sha256_bytes(OTHER))
    assert replaced.pdf_sha256 != first.pdf_sha256  # a replaced PDF is never served stale
    with pytest.raises(HashMismatchError):  # the bytes are verified even when cached
        cache.parse(PDF, filename="a.pdf", expected_sha256=sha256_bytes(OTHER))
    assert cache.parse(PDF, filename="a.pdf", expected_sha256=sha) is not first  # evicted (LRU)
    assert "pdfplumber-" in PARSER_ID and "pypdf-" in PARSER_ID


# --------------------------------------------------------------------------- independence
def test_screening_passes_never_share_a_redundancy_key_even_with_caching() -> None:
    req = LLMRequest(system="s", cacheable_prefix="PROTOCOL\n", prompt="RECORD", temperature=None)
    k1 = call_key("screening_reviewer_1", "anthropic", "claude-opus-5-5", "low", req)
    k2 = call_key("screening_reviewer_2", "anthropic", "claude-opus-5-5", "low", req)
    assert k1 != k2
    # a changed protocol changes the prompt and therefore the key (no stale reuse)
    changed = LLMRequest(
        system="s", cacheable_prefix="PROTOCOL v2\n", prompt="RECORD", temperature=None
    )
    assert call_key("screening_reviewer_1", "anthropic", "claude-opus-5-5", "low", changed) != k1


@respx.mock
def test_every_call_records_the_catalogue_it_was_validated_against(
    session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    repo, store = setup(session, tmp_path, Phase.STAGE2_SEARCH_SCREENING)
    respx.post(ANTHROPIC_URL).mock(
        side_effect=lambda req: a_reply(json.dumps(INCLUDE), json.loads(req.content)["model"])
    )
    wf = AIWorkflow(repo, PID, catalogue=CAT, credentials=CREDS(), store=store)
    wf.run_title_abstract_screening()
    calls = repo.load_llm_calls(PID)
    assert calls and all(c.settings["catalogue_sha256"] == CAT.sha256 for c in calls)
    assert all(c.settings["catalogue_snapshot_id"] == "built-in" for c in calls)
    assert {c.settings["prompt_caching"] for c in calls} == {"not requested"}  # never implicit


@respx.mock
def test_opt_in_caching_is_per_task_and_recorded(
    session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    s = settings()
    s.tasks[AITask.SCREENING_REVIEWER_1] = s.tasks[AITask.SCREENING_REVIEWER_1].model_copy(
        update={"prompt_caching": True}
    )
    repo, store = setup(session, tmp_path, Phase.STAGE2_SEARCH_SCREENING)
    repo.save_ai_settings(PID, s)
    route = respx.post(ANTHROPIC_URL).mock(
        side_effect=lambda req: a_reply(json.dumps(INCLUDE), json.loads(req.content)["model"])
    )
    AIWorkflow(
        repo, PID, catalogue=CAT, credentials=CREDS(), store=store
    ).run_title_abstract_screening()
    bodies = [json.loads(c.request.content) for c in route.calls]
    cached = [b for b in bodies if isinstance(b["messages"][0]["content"], list)]
    plain = [b for b in bodies if isinstance(b["messages"][0]["content"], str)]
    assert {b["model"] for b in cached} == {"claude-opus-5-5"}  # pass 1 only (opted in)
    assert {b["model"] for b in plain} == {"claude-sonnet-5-5"}
    for b in cached:  # the same text the uncached pass reads
        blocks = b["messages"][0]["content"]
        assert blocks[0]["cache_control"] == {"type": "ephemeral"}
        assert "CRITERIA:" in blocks[0]["text"] and "<<<RECORD" in blocks[1]["text"]


# --------------------------------------------------------------------------- accounting
def test_unknown_prices_are_reported_as_unavailable_not_zero(session: Session) -> None:
    from src.state.repository import Repository

    repo = Repository(session)
    repo.create_project(PID, "synthetic")

    def rec(cost: float | None) -> LLMCallRecord:
        return LLMCallRecord(
            project_id=PID,
            task="extraction",
            label="x",
            provider="openai",
            requested_model="m",
            status="ok",
            call_key="k",
            request_sha256="s",
            input_tokens=10,
            output_tokens=5,
            cost_usd_estimate=cost,
        )

    repo.record_llm_call(rec(None))
    assert repo.llm_usage_totals(PID)["cost_status"].startswith("unavailable")
    repo.record_llm_call(rec(0.002))
    totals = repo.llm_usage_totals(PID)
    assert totals["cost_status"].startswith("partial") and totals["unpriced_successful_calls"] == 1


# --------------------------------------------------------------------------- retries, budget
@respx.mock
def test_retries_are_bounded_and_each_attempt_is_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    route = respx.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(
            529, json={"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}
        )
    )
    cfg = LLMConfig(
        provider="anthropic",
        model_id="claude-opus-5-5",
        label="x",
        temperature=None,
        max_output_tokens=4096,
    )
    provider, cap = build_api_provider(cfg, CAT, CREDS())
    provider._sleep = lambda _s: None  # type: ignore[attr-defined]
    records: list[LLMCallRecord] = []
    audited = AuditedProvider(
        provider,
        project_id=PID,
        task="extraction",
        label="x",
        effort=None,
        capability=cap,
        sink=records.append,
    )
    with pytest.raises(ProviderUnavailableError):
        audited.complete(
            LLMRequest(system="s", prompt="p", temperature=None, max_output_tokens=4096)
        )
    assert route.call_count == 3  # 1 + 2 bounded retries, then the item can be retried later
    assert records[-1].status == "error"


def test_budget_pause_records_the_block_and_saves_nothing(session: Session, tmp_path: Path) -> None:
    repo, store = setup(session, tmp_path, Phase.STAGE2_SEARCH_SCREENING, stop_usd=0.001)
    repo.record_llm_call(
        LLMCallRecord(
            project_id=PID,
            task="extraction",
            label="x",
            provider="anthropic",
            requested_model="claude-opus-5-5",
            status="ok",
            call_key="k",
            request_sha256="s",
            input_tokens=50,
            output_tokens=5,
            cost_usd_estimate=0.01,
        )
    )
    built: list[ScriptedProvider] = []

    def build(cfg: LLMConfig) -> Any:
        built.append(ScriptedProvider(cfg.model_id, [json.dumps(INCLUDE)] * 10))
        return built[-1], CAT.validate(cfg)

    wf = AIWorkflow(
        repo, PID, catalogue=CAT, credentials=CREDS(), store=store, provider_builder=build
    )
    summary = wf.run_title_abstract_screening()
    assert summary.halted and "stop threshold" in summary.halted
    assert all(p.requests == [] for p in built)  # paused before any call
    assert repo.load_screening_decisions(PID) == []  # nothing skipped, nothing recorded as done
    assert repo.load_llm_calls(PID)[-1].status == "blocked_budget"
    guard = BudgetGuard(spent_usd=0.01, tokens=0, stop_usd=0.001)
    with pytest.raises(Exception, match="stop threshold"):
        guard.check()
