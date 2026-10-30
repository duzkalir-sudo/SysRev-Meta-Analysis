"""Session 4: Anthropic Messages / OpenAI Responses adapters against MOCKED HTTP.

No test here contacts a provider: every response is a respx mock shaped after the
provider's documented format. These tests are not live validation.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
import pytest
import respx
from pydantic import SecretStr

from src.llm import (
    InsufficientFundsError,
    LLMConfig,
    LLMOutputError,
    LLMRequest,
    ProviderAccessError,
    ProviderRegistry,
    ProviderUnavailableError,
    TokenUsage,
    UnsupportedConfigurationError,
)
from src.llm.adapters import (
    ANTHROPIC_URL,
    ANTHROPIC_VERSION,
    OPENAI_URL,
    AnthropicMessagesProvider,
    OpenAIResponsesProvider,
)
from src.llm.catalogue import ModelCatalogue, estimate_cost
from src.llm.credentials import CredentialResolver, RedactingFilter, redact
from src.llm.tasks import AITask, TaskModelSettings, build_api_provider, register_api_providers

ANTHROPIC_KEY = "sk-ant-test-0123456789abcdefSECRET"
OPENAI_KEY = "sk-proj-test0123456789abcdefSECRET"
CAT = ModelCatalogue.load()
REQ = LLMRequest(system="sys", prompt='Reply {"ok": true}', temperature=None,
                 max_output_tokens=2048)  # fmt: skip


def anthropic(effort: str | None = "high", model: str = "claude-opus-5-5", **kw: Any) -> Any:
    cfg = LLMConfig(provider="anthropic", model_id=model, label="t", temperature=None,
                    max_output_tokens=2048, effort=effort)  # fmt: skip
    sleeps: list[float] = []
    p = AnthropicMessagesProvider(cfg, CAT.validate(cfg), SecretStr(ANTHROPIC_KEY),
                                  client=httpx.Client(), sleep=sleeps.append, **kw)  # fmt: skip
    return p, sleeps


def openai(effort: str | None = "high", **kw: Any) -> Any:
    cfg = LLMConfig(provider="openai", model_id="gpt-6-luna", label="t", temperature=None,
                    max_output_tokens=2048, effort=effort)  # fmt: skip
    sleeps: list[float] = []
    p = OpenAIResponsesProvider(cfg, CAT.validate(cfg), SecretStr(OPENAI_KEY),
                                client=httpx.Client(), sleep=sleeps.append, **kw)  # fmt: skip
    return p, sleeps


def a_ok(text: str = '{"ok": true}', model: str = "claude-opus-5-5", **kw: Any) -> dict[str, Any]:
    return {
        "id": "msg_1", "type": "message", "role": "assistant", "model": model,
        "content": [{"type": "thinking", "thinking": ""}, {"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1000, "output_tokens": 500, "cache_read_input_tokens": 0,
                  "cache_creation_input_tokens": 0},
    } | kw  # fmt: skip


def o_ok(text: str = '{"ok": true}', **kw: Any) -> dict[str, Any]:
    return {
        "id": "resp_1", "object": "response", "status": "completed",
        "model": "gpt-6-luna-2026-09-01", "reasoning": {"effort": "high"},
        "output": [{"type": "reasoning", "summary": []},
                   {"type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": text}]}],
        "usage": {"input_tokens": 2000,
                  "input_tokens_details": {"cached_tokens": 500, "cache_write_tokens": 300},
                  "output_tokens": 800, "output_tokens_details": {"reasoning_tokens": 600}},
    } | kw  # fmt: skip


# --------------------------------------------------------------------------- success


@respx.mock
def test_anthropic_request_shape_usage_and_settings() -> None:
    route = respx.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(200, json=a_ok(), headers={"request-id": "req_9"})
    )
    p, _ = anthropic("xhigh")
    resp = p.complete(REQ)
    sent = route.calls.last.request
    body = json.loads(sent.content)
    assert sent.headers["anthropic-version"] == ANTHROPIC_VERSION
    assert sent.headers["x-api-key"] == ANTHROPIC_KEY
    assert body["model"] == "claude-opus-5-5"
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"] == {"effort": "xhigh"}
    assert "temperature" not in body and "fallbacks" not in body
    assert resp.text == '{"ok": true}'
    assert resp.usage == TokenUsage(1000, 500, None, 0, 0)
    assert resp.effective_settings["effort_effective"] == "xhigh"
    assert resp.provider_request_id == "req_9" and resp.attempts == 1


@respx.mock
def test_anthropic_provider_default_effort_is_not_sent_and_recorded_as_default() -> None:
    route = respx.post(ANTHROPIC_URL).mock(return_value=httpx.Response(200, json=a_ok()))
    p, _ = anthropic(None)
    resp = p.complete(REQ)
    assert "output_config" not in json.loads(route.calls.last.request.content)
    assert resp.effective_settings["effort_sent"] is None
    assert "provider default" in resp.effective_settings["effort_effective"]
    assert "medium" in resp.effective_settings["effort_effective"]


@respx.mock
def test_openai_request_shape_usage_and_served_snapshot() -> None:
    route = respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=o_ok()))
    p, _ = openai("high")
    resp = p.complete(REQ)
    sent = route.calls.last.request
    body = json.loads(sent.content)
    assert sent.headers["authorization"] == f"Bearer {OPENAI_KEY}"
    assert body["reasoning"] == {"effort": "high"} and body["store"] is False
    assert body["instructions"] == "sys" and body["max_output_tokens"] == 2048
    assert "temperature" not in body
    assert resp.model_id == "gpt-6-luna-2026-09-01"  # a dated snapshot is recorded as served
    assert resp.usage == TokenUsage(2000, 800, 600, 500, 300)


# --------------------------------------------------------------------------- validation


def test_unsupported_effort_and_unknown_model_fail_before_any_call() -> None:
    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(ANTHROPIC_URL)
        bad_effort = LLMConfig(provider="anthropic", model_id="claude-opus-4-6", label="x",
                               temperature=None, effort="xhigh")  # fmt: skip
        with pytest.raises(UnsupportedConfigurationError, match="xhigh"):
            CAT.validate(bad_effort)
        unknown = bad_effort.model_copy(update={"model_id": "claude-imaginary-9", "effort": None})
        with pytest.raises(UnsupportedConfigurationError, match="not in the anthropic"):
            CAT.validate(unknown)
        with_temp = bad_effort.model_copy(update={"effort": None, "temperature": 0.0})
        with pytest.raises(UnsupportedConfigurationError, match="temperature"):
            CAT.validate(with_temp)
        too_long = bad_effort.model_copy(update={"effort": None, "max_output_tokens": 128001})
        with pytest.raises(UnsupportedConfigurationError, match="exceeds"):
            CAT.validate(too_long)
        assert not route.called


def test_task_settings_never_send_temperature() -> None:
    cfg = TaskModelSettings(provider="openai", model_id="gpt-6-luna", effort="none").to_llm_config(
        AITask.EXTRACTION
    )
    assert cfg.temperature is None and cfg.effort == "none" and cfg.label == "extraction"


@respx.mock
def test_provider_side_parameter_rejection_is_explicit_and_not_retried() -> None:
    route = respx.post(OPENAI_URL).mock(return_value=httpx.Response(400, json={"error": {
        "type": "invalid_request_error", "code": "unsupported_value",
        "message": "Unsupported value: 'reasoning.effort' does not support 'max'"}}))  # fmt: skip
    p, sleeps = openai("max")
    with pytest.raises(UnsupportedConfigurationError, match="does not support 'max'"):
        p.complete(REQ)
    assert route.call_count == 1 and sleeps == []


@respx.mock
def test_served_model_mismatch_is_rejected() -> None:
    respx.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(200, json=a_ok(model="claude-sonnet-5-5"))
    )
    p, _ = anthropic()
    with pytest.raises(UnsupportedConfigurationError, match="served model"):
        p.complete(REQ)


@respx.mock
def test_openai_effort_echo_mismatch_is_rejected() -> None:
    respx.post(OPENAI_URL).mock(
        return_value=httpx.Response(200, json=o_ok(reasoning={"effort": "medium"}))
    )
    p, _ = openai("high")
    with pytest.raises(UnsupportedConfigurationError, match="applied reasoning effort"):
        p.complete(REQ)


# --------------------------------------------------------------------------- credentials & funds


@respx.mock
def test_invalid_credentials_halt_without_retry() -> None:
    a = respx.post(ANTHROPIC_URL).mock(return_value=httpx.Response(401, json={
        "type": "error", "error": {"type": "authentication_error",
                                   "message": "invalid x-api-key"}}))  # fmt: skip
    o = respx.post(OPENAI_URL).mock(return_value=httpx.Response(401, json={
        "error": {"code": "invalid_api_key", "message": "Incorrect API key"}}))  # fmt: skip
    with pytest.raises(ProviderAccessError, match="key was rejected"):
        anthropic()[0].complete(REQ)
    with pytest.raises(ProviderAccessError, match="key was rejected"):
        openai()[0].complete(REQ)
    assert a.call_count == 1 and o.call_count == 1


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (400, {"type": "error", "error": {"type": "invalid_request_error",
               "message": "Your credit balance is too low to access the Anthropic API."}}),
        (402, {"type": "error", "error": {"type": "billing_error", "message": "billing"}}),
    ],
)  # fmt: skip
@respx.mock
def test_anthropic_insufficient_funds(status: int, body: dict[str, Any]) -> None:
    route = respx.post(ANTHROPIC_URL).mock(return_value=httpx.Response(status, json=body))
    with pytest.raises(InsufficientFundsError, match="development credits do not"):
        anthropic()[0].complete(REQ)
    assert route.call_count == 1


@respx.mock
def test_openai_insufficient_quota_is_not_a_retryable_rate_limit() -> None:
    route = respx.post(OPENAI_URL).mock(return_value=httpx.Response(429, json={"error": {
        "type": "insufficient_quota", "code": "insufficient_quota",
        "message": "You exceeded your current quota"}}))  # fmt: skip
    p, sleeps = openai()
    with pytest.raises(InsufficientFundsError, match="ChatGPT subscriptions do not apply"):
        p.complete(REQ)
    assert route.call_count == 1 and sleeps == []


def test_missing_key_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    creds = CredentialResolver(use_keyring=False, use_dotenv=False)
    cfg = TaskModelSettings(provider="openai", model_id="gpt-6-luna").to_llm_config(
        AITask.EXTRACTION
    )
    with pytest.raises(ProviderAccessError, match="no API key for openai"):
        build_api_provider(cfg, CAT, creds)


@respx.mock
def test_model_not_available_to_account() -> None:
    respx.post(ANTHROPIC_URL).mock(return_value=httpx.Response(404, json={"type": "error",
        "error": {"type": "not_found_error", "message": "model: claude-opus-5-5"}}))  # fmt: skip
    with pytest.raises(ProviderAccessError, match="not found or not available"):
        anthropic()[0].complete(REQ)


# --------------------------------------------------------------------------- failures & retries


@respx.mock
def test_overload_is_retried_with_bounded_backoff_and_attempts_recorded() -> None:
    route = respx.post(ANTHROPIC_URL).mock(side_effect=[
        httpx.Response(529, json={"type": "error", "error": {"type": "overloaded_error",
                                                             "message": "Overloaded"}}),
        httpx.Response(429, headers={"retry-after": "999"},
                       json={"type": "error", "error": {"type": "rate_limit_error",
                                                        "message": "slow down"}}),
        httpx.Response(200, json=a_ok()),
    ])  # fmt: skip
    p, sleeps = anthropic()
    resp = p.complete(REQ)
    assert route.call_count == 3 and resp.attempts == 3
    assert sleeps == [2.0, 30.0]  # exponential, and retry-after capped at 30 s


@respx.mock
def test_persistent_server_error_gives_up_after_bounded_retries() -> None:
    route = respx.post(OPENAI_URL).mock(
        return_value=httpx.Response(500, json={"error": {"message": "server error"}})
    )
    p, sleeps = openai(max_retries=2)
    with pytest.raises(ProviderUnavailableError, match="temporarily unavailable"):
        p.complete(REQ)
    assert route.call_count == 3 and len(sleeps) == 2


@respx.mock
def test_read_timeout_is_not_retried_because_it_may_have_been_billed() -> None:
    route = respx.post(ANTHROPIC_URL).mock(side_effect=httpx.ReadTimeout("slow"))
    p, sleeps = anthropic()
    with pytest.raises(ProviderUnavailableError, match="may already have processed"):
        p.complete(REQ)
    assert route.call_count == 1 and sleeps == []


@respx.mock
def test_connection_failure_is_retried() -> None:
    route = respx.post(ANTHROPIC_URL).mock(
        side_effect=[httpx.ConnectError("down"), httpx.Response(200, json=a_ok())]
    )
    p, _ = anthropic()
    assert p.complete(REQ).attempts == 2 and route.call_count == 2


# --------------------------------------------------------------------------- malformed output


@respx.mock
def test_refusal_truncation_and_empty_outputs_raise() -> None:
    route = respx.post(ANTHROPIC_URL)
    p, _ = anthropic()
    route.mock(return_value=httpx.Response(200, json=a_ok(
        stop_reason="refusal", stop_details={"type": "refusal", "category": "bio"})))  # fmt: skip
    with pytest.raises(LLMOutputError, match="declined"):
        p.complete(REQ)
    route.mock(return_value=httpx.Response(200, json=a_ok(stop_reason="max_tokens")))
    with pytest.raises(LLMOutputError, match="truncated"):
        p.complete(REQ)
    route.mock(return_value=httpx.Response(200, json=a_ok(content=[])))
    with pytest.raises(LLMOutputError, match="no text"):
        p.complete(REQ)
    route.mock(return_value=httpx.Response(200, text="<html>gateway</html>"))
    with pytest.raises(LLMOutputError, match="not JSON"):
        p.complete(REQ)


@respx.mock
def test_openai_incomplete_and_refusal() -> None:
    route = respx.post(OPENAI_URL)
    p, _ = openai()
    route.mock(return_value=httpx.Response(200, json=o_ok(
        status="incomplete", incomplete_details={"reason": "max_output_tokens"})))  # fmt: skip
    with pytest.raises(LLMOutputError, match="max_output_tokens"):
        p.complete(REQ)
    route.mock(return_value=httpx.Response(200, json=o_ok(output=[{"type": "message",
        "content": [{"type": "refusal", "refusal": "no"}]}])))  # fmt: skip
    with pytest.raises(LLMOutputError, match="declined"):
        p.complete(REQ)


# --------------------------------------------------------------------------- accounting


def test_cost_estimates_follow_each_providers_token_semantics() -> None:
    opus = CAT.get("anthropic", "claude-opus-5-5")
    # Anthropic: input excludes cache reads → 1000*4 + 2000*0.20 + 500*20 per 1M
    est = estimate_cost(opus, TokenUsage(1000, 500, None, 2000, 0))
    assert est.usd == pytest.approx((1000 * 4 + 2000 * 0.20 + 500 * 20) / 1e6)
    assert est.basis.startswith("estimate")
    haiku = CAT.get("anthropic", "claude-haiku-5-5")
    long_prompt = estimate_cost(haiku, TokenUsage(150_000, 1000, None, 0, 0))
    assert long_prompt.usd == pytest.approx((150_000 * 0.5 + 1000 * 2.5) / 1e6)
    assert "long-prompt" in long_prompt.basis
    luna = CAT.get("openai", "gpt-6-luna")
    # OpenAI: cached and cache-write tokens are part of input_tokens
    est = estimate_cost(luna, TokenUsage(2000, 800, 600, 500, 300))
    assert est.usd == pytest.approx((1200 * 0.1 + 500 * 0.01 + 300 * 0.125 + 800 * 0.5) / 1e6)
    sol61 = CAT.get("openai", "gpt-6.1-sol")
    at_threshold = estimate_cost(sol61, TokenUsage(272_000, 1000, None, 0, None))
    assert at_threshold.usd == pytest.approx((272_000 * 2 + 1000 * 10) / 1e6)  # not > 272K
    long_req = estimate_cost(sol61, TokenUsage(300_000, 1000, None, 100_000, None))
    assert long_req.usd == pytest.approx((200_000 * 4 + 100_000 * 0.2 + 1000 * 15) / 1e6)
    sol56 = CAT.get("openai", "gpt-5.6-sol")
    long56 = estimate_cost(sol56, TokenUsage(300_000, 1000, None, 100_000, None))
    # long-tier cache price is not documented for GPT-5.6 Sol → priced at 2x input (bound)
    assert long56.usd == pytest.approx((200_000 * 8 + 100_000 * 8 + 1000 * 30) / 1e6)
    assert "upper bound" in long56.basis
    assert estimate_cost(opus, None).usd is None


# --------------------------------------------------------------------------- redaction


@respx.mock
def test_secrets_are_redacted_from_errors_and_logs(caplog: pytest.LogCaptureFixture) -> None:
    respx.post(ANTHROPIC_URL).mock(return_value=httpx.Response(401, json={"type": "error",
        "error": {"type": "authentication_error",
                  "message": f"bad key {ANTHROPIC_KEY} and Bearer {OPENAI_KEY}"}}))  # fmt: skip
    p, _ = anthropic()
    with pytest.raises(ProviderAccessError) as info:
        p.complete(REQ)
    assert ANTHROPIC_KEY not in str(info.value) and OPENAI_KEY not in str(info.value)
    assert "[REDACTED]" in str(info.value)
    assert ANTHROPIC_KEY not in repr(p.__dict__.get("_key"))
    logger = logging.getLogger("sysrev.test")
    handler = logging.StreamHandler()
    handler.addFilter(RedactingFilter())
    record = logger.makeRecord("t", logging.INFO, "f", 1, "key=%s", (ANTHROPIC_KEY,), None)
    handler.filter(record)
    assert ANTHROPIC_KEY not in record.getMessage()
    assert redact("x-api-key: abcdefgh12345678") == "x-api-key: [REDACTED]"
    assert redact(f"token rvw_{'a' * 32}_{'B' * 43}") == "token [REDACTED]"


# --------------------------------------------------------------------------- registry


@respx.mock
def test_adapters_plug_into_the_existing_provider_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", ANTHROPIC_KEY)
    respx.post(ANTHROPIC_URL).mock(return_value=httpx.Response(200, json=a_ok()))
    registry = ProviderRegistry()
    register_api_providers(registry, CAT, CredentialResolver(use_keyring=False, use_dotenv=False))
    assert registry.names == ("anthropic", "openai")
    cfg = LLMConfig(provider="anthropic", model_id="claude-opus-5-5", label="x",
                    temperature=None, effort="low")  # fmt: skip
    provider = registry.build(cfg)
    assert provider.complete(REQ).provider == "anthropic"
    with pytest.raises(UnsupportedConfigurationError):
        registry.build(cfg.model_copy(update={"effort": "extreme"}))


# --------------------------------------------------------------------------- catalogue validation


def _entry(**kw: Any) -> dict[str, Any]:
    return {"provider": "openai", "model_id": "m-1", "display_name": "M", "efforts": ["low"],
            "provider_default_effort": "low", "source": "Official docs (2026-10-30)",
            "verified": True} | kw  # fmt: skip


def test_builtin_catalogue_is_internally_consistent() -> None:
    models = CAT.models()
    assert len({(m.provider, m.model_id) for m in models}) == len(models)
    for m in models:
        assert m.provider_default_effort is None or m.provider_default_effort in m.efforts
        assert m.verified != m.source.startswith("UNVERIFIED")
        if m.pricing is not None:
            assert m.pricing.input > 0 and m.pricing.output > 0
    openai = {m.model_id: m for m in CAT.models("openai")}
    # exact IDs from the official model pages; the routing alias "gpt-5.6" is not listed
    assert set(openai) == {"gpt-5.6-sol", "gpt-6.1-sol", "gpt-6-luna", "gpt-6-astra"}
    for m in openai.values():
        assert m.verified and m.pricing is not None and m.max_output_tokens == 128000
        assert m.pricing.long_prompt_threshold_tokens == 272000
        assert m.pricing.long_prompt_input == 2 * m.pricing.input
        assert m.pricing.long_prompt_output == pytest.approx(1.5 * m.pricing.output)
    assert openai["gpt-6.1-sol"].efforts == ("low", "medium", "high", "xhigh", "max")
    assert "none" in openai["gpt-6-luna"].efforts and "none" in openai["gpt-5.6-sol"].efforts
    assert openai["gpt-6-astra"].provider_default_effort is None  # not documented
    assert (openai["gpt-6-astra"].pricing.input, openai["gpt-6-astra"].pricing.output) == (10, 50)
    alias = LLMConfig(provider="openai", model_id="gpt-5.6", label="x", temperature=None)
    with pytest.raises(UnsupportedConfigurationError, match="not in the openai"):
        CAT.validate(alias)
    no_none = alias.model_copy(update={"model_id": "gpt-6.1-sol", "effort": "none"})
    with pytest.raises(UnsupportedConfigurationError, match="supported values: low"):
        CAT.validate(no_none)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"provider_default_effort": "high"}, "default effort"),
        ({"efforts": ["low", "low"]}, "duplicate effort"),
        ({"verified": False}, "UNVERIFIED"),
        ({"source": "UNVERIFIED guess"}, "UNVERIFIED"),
        ({"verified": False, "source": "UNVERIFIED: x"}, "must say so"),
        ({"pricing": {"input": 1, "output": 2, "long_prompt_threshold_tokens": 1000}},
         "long-prompt"),
        ({"provider": "mistral"}, "provider"),
    ],
)  # fmt: skip
def test_invalid_catalogue_entries_are_rejected(override: dict[str, Any], message: str) -> None:
    from pydantic import ValidationError

    from src.llm.catalogue import ModelCapability

    with pytest.raises(ValidationError, match=message):
        ModelCapability.model_validate(_entry(**override))


def test_local_override_adds_and_replaces_entries(tmp_path: Any) -> None:
    (tmp_path / "model_catalogue.json").write_text(json.dumps({"models": [
        _entry(model_id="gpt-local", efforts=["none", "low"], provider_default_effort="none",
               pricing={"input": 1.0, "output": 2.0}),
        _entry(model_id="gpt-6-luna", display_name="GPT-5.6", efforts=["low"],
               provider_default_effort="low"),
    ]}), encoding="utf-8")  # fmt: skip
    cat = ModelCatalogue.load(tmp_path)
    assert cat.get("openai", "gpt-local").pricing is not None
    assert cat.get("openai", "gpt-6-luna").efforts == ("low",)
    cfg = LLMConfig(provider="openai", model_id="gpt-6-luna", label="x", temperature=None,
                    effort="high")  # fmt: skip
    with pytest.raises(UnsupportedConfigurationError, match="supported values: low"):
        cat.validate(cfg)  # never adjusted to another effort
