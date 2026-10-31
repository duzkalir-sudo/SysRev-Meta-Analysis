"""Session 8 — Refresh Models (issue #9) against MOCKED provider list endpoints.

No test contacts a provider. Listing responses are shaped after the official API references
(Anthropic ``GET /v1/models``, OpenAI ``GET /v1/models``, retrieved 2026-10-31). Model IDs
such as ``claude-opus-9-test`` and ``gpt-9-test`` are invented for the tests.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy.orm import Session

from src.core.exceptions import SysRevError
from src.llm import LLMConfig, LLMRequest, UnsupportedConfigurationError
from src.llm.adapters import ANTHROPIC_URL
from src.llm.catalogue import ModelCatalogue
from src.llm.credentials import CredentialResolver
from src.llm.discovery import ANTHROPIC_MODELS_URL, OPENAI_MODELS_URL, ModelLister
from src.llm.model_registry import (
    ModelVerification,
    VerificationRefusedError,
    diff,
    model_status,
    refresh_snapshot,
    verification_snapshot,
)
from src.llm.tasks import AITask, ProjectAISettings, TaskModelSettings, build_api_provider
from src.state.repository import Repository
from src.workflows.models import refresh_catalogue

ANTHROPIC_KEY = "sk-ant-test-0123456789abcdefSECRETLIST"
OPENAI_KEY = "sk-proj-test0123456789abcdefSECRETLIST"
NOW = datetime(2026, 10, 31, 9, 0, tzinfo=UTC)
BASE = ModelCatalogue.load()


def creds(monkeypatch: pytest.MonkeyPatch, *, openai: bool = True) -> CredentialResolver:
    monkeypatch.setenv("ANTHROPIC_API_KEY", ANTHROPIC_KEY)
    if openai:
        monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    else:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return CredentialResolver(use_keyring=False, use_dotenv=False)


def a_model(
    mid: str,
    *,
    lifecycle: str = "active",
    efforts: dict[str, bool] | None = None,
    max_tokens: int | None = 128000,
    caps: bool = True,
    **kw: Any,
) -> dict[str, Any]:
    effort = {
        k: {"supported": v}
        for k, v in (
            efforts or {"low": True, "medium": True, "high": True, "xhigh": True, "max": True}
        ).items()
    }
    effort["supported"] = True  # type: ignore[assignment]
    return {
        "type": "model",
        "id": mid,
        "display_name": kw.pop("display_name", mid),
        "created_at": "2026-10-20T00:00:00Z",
        "lifecycle": lifecycle,
        "deprecated_at": None,
        "retires_at": kw.pop("retires_at", None),
        "max_tokens": max_tokens,
        "max_input_tokens": 1000000,
        "line": "opus",
        "capabilities": {
            "effort": effort,
            "structured_outputs": {"supported": True},
            "batch": {"supported": True},
            "thinking": {"supported": True, "types": {"adaptive": {"supported": True}}},
        }
        if caps
        else None,
    } | kw


def anthropic_page(*models: dict[str, Any], has_more: bool = False) -> dict[str, Any]:
    return {
        "data": list(models),
        "has_more": has_more,
        "first_id": models[0]["id"] if models else None,
        "last_id": models[-1]["id"] if models else None,
    }


def openai_list(*ids: str, shutdown: dict[str, str] | None = None) -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": i,
                "object": "model",
                "created": 1790000000,
                "owned_by": "openai",
                "shutdown_date": (shutdown or {}).get(i),
            }
            for i in ids
        ],
    }


def lister(monkeypatch: pytest.MonkeyPatch, **kw: Any) -> ModelLister:
    return ModelLister(creds(monkeypatch, **kw), client=httpx.Client(), clock=lambda: NOW)


def mock_both(anthropic: Any, openai: Any) -> None:
    respx.get(ANTHROPIC_MODELS_URL).mock(return_value=anthropic)
    respx.get(OPENAI_MODELS_URL).mock(return_value=openai)


KNOWN_A = [a_model(m.model_id) for m in BASE.models("anthropic")]
KNOWN_O = [m.model_id for m in BASE.models("openai")]


# --------------------------------------------------------------------------- discovery
@respx.mock
def test_refresh_lists_only_and_flags_new_models_unverified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_both(
        httpx.Response(200, json=anthropic_page(a_model("claude-opus-9-test"), *KNOWN_A)),
        httpx.Response(200, json=openai_list("gpt-9-test", *KNOWN_O)),
    )
    results = lister(monkeypatch).refresh()
    # only the two documented GET list calls: no inference request, nothing billed
    assert [(c.request.method, str(c.request.url).split("?")[0]) for c in respx.calls] == [
        ("GET", ANTHROPIC_MODELS_URL),
        ("GET", OPENAI_MODELS_URL),
    ]
    assert respx.calls[0].request.headers["anthropic-version"]
    assert all(r.ok for r in results)
    snap = refresh_snapshot(BASE, results, created_by="test")
    cat = snap.catalogue()
    for provider, mid in (("anthropic", "claude-opus-9-test"), ("openai", "gpt-9-test")):
        new = cat.get(provider, mid)
        assert not new.verified and "UNVERIFIED" in new.display_name
        assert new.pricing is None and new.efforts == ()
        with pytest.raises(UnsupportedConfigurationError, match="unverified"):
            cat.validate(LLMConfig(provider=provider, model_id=mid, label="x", temperature=None))
    # provider-reported metadata is kept for the reviewer, but never used to execute
    assert cat.get("anthropic", "claude-opus-9-test").provider_reported["effort_levels"] == [
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    ]
    # existing verified entries are unchanged in everything that matters for execution
    for m in BASE.models():
        kept = cat.get(m.provider, m.model_id)
        assert (kept.efforts, kept.pricing, kept.verified) == (m.efforts, m.pricing, m.verified)
    changes = diff(BASE, cat)
    assert changes["added"] == ["anthropic/claude-opus-9-test", "openai/gpt-9-test"]
    assert changes["removed"] == [] and changes["newly_executable"] == []
    # credentials never reach the stored snapshot
    dumped = json.dumps(snap.model_dump(mode="json"))
    assert ANTHROPIC_KEY not in dumped and OPENAI_KEY not in dumped


@respx.mock
def test_outage_and_missing_key_change_nothing_and_are_reported_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    respx.get(ANTHROPIC_MODELS_URL).mock(return_value=httpx.Response(529, text="overloaded"))
    results = lister(monkeypatch, openai=False).refresh()
    by = {r.provider: r for r in results}
    assert by["anthropic"].status == "unavailable" and "nothing changed" in by["anthropic"].error
    assert by["openai"].status == "no_credentials"
    assert len(respx.calls) == 1  # no OpenAI call without a key
    snap = refresh_snapshot(BASE, results, created_by="test")
    assert snap.sha256 == BASE.sha256  # catalogue unchanged
    rows = {
        r["model_id"]: r
        for r in model_status(snap.catalogue(), {d.provider: d for d in snap.discovery})
    }
    opus = rows["claude-opus-5-5"]
    # availability unknown because of the outage, but the model stays executable/compatible
    assert opus["account_availability"].startswith("unknown — last refresh failed")
    assert opus["executable"] and opus["adapter_compatibility"].startswith("anthropic.messages")


@respx.mock
def test_access_error_is_not_model_incompatibility(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_both(
        httpx.Response(401, json={"error": {"type": "authentication_error"}}),
        httpx.Response(403, json={"error": {"message": "no"}}),
    )
    results = lister(monkeypatch).refresh()
    assert {r.status for r in results} == {"access_error"}
    snap = refresh_snapshot(BASE, results, created_by="test")
    assert all(m.verified for m in snap.models) and snap.sha256 == BASE.sha256


@respx.mock
def test_deprecated_retired_and_unlisted_models(monkeypatch: pytest.MonkeyPatch) -> None:
    others = [a for a in KNOWN_A if a["id"] not in ("claude-opus-4-6", "claude-sonnet-4-6")]
    mock_both(
        httpx.Response(
            200,
            json=anthropic_page(
                a_model(
                    "claude-opus-4-6", lifecycle="deprecated", retires_at="2027-03-01T00:00:00Z"
                ),
                *others,
            ),
        ),  # sonnet-4-6 not listed
        httpx.Response(200, json=openai_list(*KNOWN_O, shutdown={"gpt-5.6-sol": "2026-10-01"})),
    )
    snap = refresh_snapshot(BASE, lister(monkeypatch).refresh(), created_by="test")
    cat = snap.catalogue()
    dep = cat.get("anthropic", "claude-opus-4-6")
    assert dep.lifecycle == "deprecated" and dep.retires_at == date(2027, 3, 1)
    assert not dep.execution_problems() and "deprecated" in dep.warnings()[0]
    retired = cat.get("openai", "gpt-5.6-sol")  # shutdown date in the past
    assert retired.lifecycle == "retired"
    with pytest.raises(UnsupportedConfigurationError, match=r"retired.*no other model"):
        cat.validate(
            LLMConfig(provider="openai", model_id="gpt-5.6-sol", label="x", temperature=None)
        )
    # a model that is no longer listed is kept (never silently removed) and flagged
    assert cat.get("anthropic", "claude-sonnet-4-6").verified
    rows = {r["model_id"]: r for r in model_status(cat, {d.provider: d for d in snap.discovery})}
    assert rows["claude-sonnet-4-6"]["account_availability"].startswith("NOT listed")
    changes = diff(BASE, cat)
    assert changes["removed"] == [] and "openai/gpt-5.6-sol" in changes["newly_blocked"]


@respx.mock
def test_incomplete_capability_metadata_and_pagination(monkeypatch: pytest.MonkeyPatch) -> None:
    route = respx.get(ANTHROPIC_MODELS_URL)
    route.side_effect = [
        httpx.Response(
            200,
            json=anthropic_page(
                a_model("claude-x-test", caps=False, max_tokens=None), has_more=True
            ),
        ),
        httpx.Response(200, json=anthropic_page(*KNOWN_A)),
    ]
    result = lister(monkeypatch).list_models("anthropic")
    assert result.ok and result.requests == 2
    assert route.calls[1].request.url.params["after_id"] == "claude-x-test"
    new = (
        refresh_snapshot(BASE, [result], created_by="t")
        .catalogue()
        .get("anthropic", "claude-x-test")
    )
    assert not new.verified and new.efforts == () and new.max_output_tokens is None
    reported = new.provider_reported or {}
    assert "effort_levels" not in reported and "thinking_adaptive" not in reported  # not invented


@respx.mock
def test_repeated_refresh_is_stable(monkeypatch: pytest.MonkeyPatch, session: Session) -> None:
    mock_both(
        httpx.Response(200, json=anthropic_page(a_model("claude-opus-9-test"), *KNOWN_A)),
        httpx.Response(200, json=openai_list(*KNOWN_O)),
    )
    repo = Repository(session)
    first, _ = refresh_catalogue(repo, BASE, lister(monkeypatch))
    second, changes = refresh_catalogue(repo, BASE, lister(monkeypatch))
    assert first.sha256 == second.sha256 and first.snapshot_id != second.snapshot_id
    assert changes["added"] == ["anthropic/claude-opus-9-test"]
    assert repo.active_catalogue(BASE).sha256 == BASE.sha256  # nothing active without a human
    rid, _ = repo.enroll_reviewer("Dr Model Steward")
    repo.activate_catalogue_snapshot(
        second.snapshot_id,
        base=BASE,
        expected_sha256=second.sha256,
        expected_active_sha256=BASE.sha256,
        reviewer_id=rid,
    )
    third, changes = refresh_catalogue(repo, BASE, lister(monkeypatch))
    assert changes["unchanged"] and third.sha256 == second.sha256


# --------------------------------------------------------------------------- verification
def _verification(**kw: Any) -> ModelVerification:
    data: dict[str, Any] = {
        "provider": "anthropic",
        "model_id": "claude-opus-9-test",
        "display_name": "Claude Opus 9 (test)",
        "efforts": ["low", "medium", "high"],
        "provider_default_effort": "high",
        "thinking": "adaptive",
        "max_output_tokens": 64000,
        "adapter": "anthropic.messages",
        "capability_source": "https://platform.claude.com/docs/en/about-claude/models/overview "
        "(test fixture)",
        "capabilities_verified_on": "2026-10-31",
    } | kw
    return ModelVerification.model_validate(data)


def _discovered() -> ModelCatalogue:
    from src.llm.discovery import DiscoveredModel, DiscoveryResult, anthropic_reported

    found = DiscoveredModel(
        "anthropic",
        "claude-opus-9-test",
        "Claude Opus 9",
        reported=anthropic_reported(
            a_model(
                "claude-opus-9-test",
                max_tokens=64000,
                efforts={"low": True, "medium": True, "high": True, "xhigh": False, "max": False},
            )
        ),
    )
    result = DiscoveryResult("anthropic", "ok", ANTHROPIC_MODELS_URL, NOW, (found,))
    return refresh_snapshot(BASE, [result], created_by="t").catalogue()


def test_verification_cross_checks_provider_reported_metadata() -> None:
    cat = _discovered()
    with pytest.raises(VerificationRefusedError, match="xhigh"):  # mismatched effort
        verification_snapshot(
            cat, _verification(efforts=["low", "high", "xhigh"]), created_by="Dr V", reviewer_id="r"
        )
    with pytest.raises(VerificationRefusedError, match="max_tokens 64000"):
        verification_snapshot(
            cat, _verification(max_output_tokens=128000), created_by="Dr V", reviewer_id="r"
        )
    with pytest.raises(ValidationError, match="https://"):
        _verification(capability_source="I think it supports high")
    with pytest.raises(ValidationError, match="own source"):
        _verification(pricing={"input": 1.0, "output": 2.0})  # price without its evidence
    with pytest.raises(ValidationError, match=r"openai\.responses"):
        _verification(provider="openai", adapter="anthropic.messages")


@respx.mock
def test_verified_new_model_runs_after_activation_without_code_changes(
    monkeypatch: pytest.MonkeyPatch, session: Session
) -> None:
    repo = Repository(session)
    rid, _ = repo.enroll_reviewer("Dr Model Steward")
    discovered = _discovered()
    from src.llm.model_registry import CatalogueSnapshot

    refresh = CatalogueSnapshot(
        kind="refresh", created_by="t", parent_sha256=BASE.sha256, models=discovered.models()
    )
    repo.save_catalogue_snapshot(refresh)
    repo.activate_catalogue_snapshot(
        refresh.snapshot_id,
        base=BASE,
        expected_sha256=refresh.sha256,
        expected_active_sha256=BASE.sha256,
        reviewer_id=rid,
    )
    active = repo.active_catalogue(BASE)
    cfg = LLMConfig(
        provider="anthropic",
        model_id="claude-opus-9-test",
        label="x",
        temperature=None,
        effort="high",
        max_output_tokens=4096,
    )
    resolver = creds(monkeypatch)
    with pytest.raises(UnsupportedConfigurationError, match="unverified"):
        build_api_provider(cfg, active, resolver)  # discovered but not verified: no call
    assert not respx.calls
    verified = verification_snapshot(
        active,
        _verification(
            pricing={"input": 3.0, "output": 15.0},
            pricing_verified_on="2026-10-31",
            pricing_source="https://platform.claude.com/docs/en/about-claude/pricing",
        ),
        created_by="Dr Model Steward",
        reviewer_id=rid,
    )
    repo.save_catalogue_snapshot(verified)
    with pytest.raises(UnsupportedConfigurationError):  # still inactive
        repo.active_catalogue(BASE).validate(cfg)
    repo.activate_catalogue_snapshot(
        verified.snapshot_id,
        base=BASE,
        expected_sha256=verified.sha256,
        expected_active_sha256=active.sha256,
        reviewer_id=rid,
    )
    route = respx.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "m",
                "type": "message",
                "model": "claude-opus-9-test",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "{}"}],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )
    )
    provider, cap = build_api_provider(cfg, repo.active_catalogue(BASE), resolver)
    provider.complete(LLMRequest(system="s", prompt="p", temperature=None, max_output_tokens=4096))
    body = json.loads(route.calls[0].request.content)
    assert body["model"] == "claude-opus-9-test" and body["output_config"] == {"effort": "high"}
    assert cap.pricing is not None and cap.pricing_verified_at == date(2026, 10, 31)


def test_activation_is_bound_to_the_previewed_state(session: Session) -> None:
    repo = Repository(session)
    rid, _ = repo.enroll_reviewer("Dr Model Steward")
    cat = _discovered()
    a = verification_snapshot(cat, _verification(), created_by="Dr Model Steward", reviewer_id=rid)
    from src.llm.model_registry import CatalogueSnapshot

    first = CatalogueSnapshot(
        kind="refresh", created_by="t", parent_sha256=BASE.sha256, models=cat.models()
    )
    repo.save_catalogue_snapshot(first)
    repo.save_catalogue_snapshot(a)  # built on `cat`, which is not active yet
    with pytest.raises(SysRevError, match="different catalogue"):
        repo.activate_catalogue_snapshot(
            a.snapshot_id,
            base=BASE,
            expected_sha256=a.sha256,
            expected_active_sha256=BASE.sha256,
            reviewer_id=rid,
        )
    with pytest.raises(SysRevError, match="changed since it was previewed"):
        repo.activate_catalogue_snapshot(
            first.snapshot_id,
            base=BASE,
            expected_sha256="0" * 64,
            expected_active_sha256=BASE.sha256,
            reviewer_id=rid,
        )
    repo.activate_catalogue_snapshot(
        first.snapshot_id,
        base=BASE,
        expected_sha256=first.sha256,
        expected_active_sha256=BASE.sha256,
        reviewer_id=rid,
    )
    with pytest.raises(SysRevError, match="active catalogue has changed"):  # stale screen
        repo.activate_catalogue_snapshot(
            a.snapshot_id,
            base=BASE,
            expected_sha256=a.sha256,
            expected_active_sha256=BASE.sha256,
            reviewer_id=rid,
        )
    repo.activate_catalogue_snapshot(
        a.snapshot_id,
        base=BASE,
        expected_sha256=a.sha256,
        expected_active_sha256=first.sha256,
        reviewer_id=rid,
    )
    assert repo.active_catalogue(BASE).get("anthropic", "claude-opus-9-test").verified


def test_project_selection_is_preserved_when_a_model_is_retired(
    session: Session, tmp_path: Path
) -> None:
    from src.core.enums import Phase
    from src.fulltext.storage import DocumentStore
    from src.llm.discovery import DiscoveredModel, DiscoveryResult
    from src.workflows import AIWorkflow

    repo = Repository(session)
    repo.create_project("p1", "synthetic")
    repo.get_project("p1").phase = Phase.STAGE2_SEARCH_SCREENING.value
    rid, _ = repo.enroll_reviewer("Dr Model Steward")
    chosen = TaskModelSettings(provider="openai", model_id="gpt-5.6-sol", effort="high")
    repo.save_ai_settings(
        "p1", ProjectAISettings(tasks={AITask.EXTRACTION: chosen}, catalogue_sha256=BASE.sha256)
    )
    gone = DiscoveryResult(
        "openai",
        "ok",
        OPENAI_MODELS_URL,
        NOW,
        (
            DiscoveredModel(
                "openai", "gpt-5.6-sol", lifecycle="retired", retires_at=date(2026, 10, 1)
            ),
        ),
    )
    snap = refresh_snapshot(BASE, [gone], created_by="t")
    repo.save_catalogue_snapshot(snap)
    repo.activate_catalogue_snapshot(
        snap.snapshot_id,
        base=BASE,
        expected_sha256=snap.sha256,
        expected_active_sha256=BASE.sha256,
        reviewer_id=rid,
    )
    active = repo.active_catalogue(BASE)
    wf = AIWorkflow(
        repo,
        "p1",
        catalogue=active,
        credentials=CredentialResolver(use_keyring=False, use_dotenv=False),
        store=DocumentStore(tmp_path),
    )
    problems = wf.preflight(AITask.EXTRACTION)
    assert problems and "retired" in problems[0] and "no other model" in problems[0]
    stored = repo.latest_ai_settings("p1")
    assert stored is not None and stored.tasks[AITask.EXTRACTION] == chosen  # never rewritten
    # the catalogue each historical state was validated against is reconstructable
    assert repo.get_catalogue_snapshot(snap.snapshot_id).catalogue().sha256 == active.sha256
    assert stored.catalogue_sha256 == BASE.sha256 != active.sha256


# --------------------------------------------------------------------------- API
def test_activation_endpoint_requires_a_reviewer_and_the_previewed_hashes(
    tmp_path: Path,
) -> None:
    from src.api_server.app import create_app

    url = f"sqlite:///{tmp_path / 'm.db'}"
    client = TestClient(create_app(url, tmp_path))
    with client:
        from src.state.db import make_engine, make_session_factory

        with make_session_factory(make_engine(url))() as s:
            repo = Repository(s)
            _, token = repo.enroll_reviewer("Dr Model Steward")
            snap = refresh_snapshot(client.app.state.base_catalogue, [], created_by="t")  # type: ignore[attr-defined]
            snap = snap.model_copy(update={"models": _discovered().models()})
            repo.save_catalogue_snapshot(snap)
            s.commit()
        view = client.get(f"/models/snapshots/{snap.snapshot_id}/preview").json()
        assert view["changes"]["added"] == ["anthropic/claude-opus-9-test"]
        body = {"expected_sha256": view["sha256"], "expected_active_sha256": "f" * 64}
        path = f"/models/snapshots/{snap.snapshot_id}/activate"
        assert client.post(path, json=body).status_code == 401
        auth = {"Authorization": f"Bearer {token}"}
        refused = client.post(path, json=body, headers=auth)
        assert refused.status_code == 409 and "has changed" in refused.json()["detail"]
        body["expected_active_sha256"] = view["active_sha256"]
        assert client.post(path, json=body, headers=auth).status_code == 200
        rows = client.get("/models/catalogue").json()["models"]
        new = next(r for r in rows if r["model_id"] == "claude-opus-9-test")
        assert not new["executable"] and new["capabilities"] == "UNVERIFIED"
        assert new["pricing"].startswith("unavailable")


# --------------------------------------------------------------------------- prompt caching
@respx.mock
def test_anthropic_prompt_caching_is_opt_in_and_sends_the_same_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = respx.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "m",
                "type": "message",
                "model": "claude-opus-5-5",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "{}"}],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "cache_read_input_tokens": 900,
                    "cache_creation_input_tokens": 0,
                },
            },
        )
    )
    req = LLMRequest(
        system="s",
        cacheable_prefix="PROTOCOL AND CRITERIA\n",
        prompt="RECORD 1",
        temperature=None,
        max_output_tokens=4096,
    )
    plain = LLMRequest(
        system="s",
        prompt="PROTOCOL AND CRITERIA\nRECORD 1",
        temperature=None,
        max_output_tokens=4096,
    )
    assert req.sha256 == plain.sha256  # identity of the request is unchanged by caching
    resolver = creds(monkeypatch)
    for caching in (False, True):
        cfg = LLMConfig(
            provider="anthropic",
            model_id="claude-opus-5-5",
            label="x",
            temperature=None,
            max_output_tokens=4096,
            prompt_caching=caching,
        )
        provider, _ = build_api_provider(cfg, BASE, resolver)
        resp = provider.complete(req)
        assert ("explicit" in resp.effective_settings["prompt_caching"]) is caching
    off, on = (json.loads(c.request.content)["messages"][0]["content"] for c in route.calls)
    assert off == "PROTOCOL AND CRITERIA\nRECORD 1"
    assert on[0]["cache_control"] == {"type": "ephemeral"} and "cache_control" not in on[1]
    assert on[0]["text"] + on[1]["text"] == off  # the model reads exactly the same text
    with pytest.raises(UnsupportedConfigurationError, match="automatically"):
        BASE.validate(
            LLMConfig(
                provider="openai",
                model_id="gpt-6-luna",
                label="x",
                temperature=None,
                prompt_caching=True,
            )
        )
    assert len(respx.calls) == 2 and all(str(c.request.url) == ANTHROPIC_URL for c in respx.calls)


def test_builtin_pricing_has_separate_verified_evidence() -> None:
    for m in BASE.models():
        assert m.verified and m.verified_at is not None
        assert m.pricing is not None and m.pricing_source and m.pricing_verified_at
        assert m.lifecycle is None  # not recorded until a provider refresh reports it
    sonnet = BASE.get("anthropic", "claude-sonnet-5-5").pricing
    assert sonnet is not None and sonnet.cache_read == pytest.approx(0.10)  # official page
    haiku = BASE.get("anthropic", "claude-haiku-5-5").pricing
    assert haiku is not None and (haiku.long_prompt_cache_read, haiku.long_prompt_cache_write) == (
        0.05,
        0.625,
    )
