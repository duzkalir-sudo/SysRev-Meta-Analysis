"""HTTP adapters for the Anthropic Messages API and the OpenAI Responses API.

Both adapters implement :class:`src.llm.provider.LLMProvider` with plain ``httpx`` calls
(the project's existing HTTP stack), so retries, error classification and usage
accounting are explicit, bounded and testable with mocked transports.

Rules shared by both adapters:

* the configuration is validated against the capability catalogue *before* a request;
* the model and effort are sent exactly as configured — no fallback model, no
  downgraded effort. A provider that serves a different model, or reports a different
  effort, makes the call fail (:class:`UnsupportedConfigurationError`);
* retries are bounded (default 2) and only for rate limits, overload, 5xx and connection
  failures that happened *before* the request was sent. A read timeout is not retried,
  because the provider may already have processed (and billed) the request;
* missing/invalid credentials, exhausted funds, permission and unsupported-parameter
  errors are never retried and halt the run;
* refusals, truncation at the output limit and empty outputs raise :class:`LLMOutputError`
  — nothing is recorded as a decision from them;
* error texts are redacted (no API key can reach a log or exception message).

Provider facts used here (verify when updating):

* Anthropic: ``POST /v1/messages`` with ``anthropic-version: 2026-10-21`` (sampling
  parameters are rejected on that version), adaptive thinking ``{"type": "adaptive"}``
  and ``output_config.effort``; usage fields ``input_tokens``, ``output_tokens``,
  ``cache_read_input_tokens``, ``cache_creation_input_tokens``; billing problems are
  reported as ``billing_error`` (402) or a "credit balance is too low" 400. Optional
  explicit prompt caching (Session 8, opt-in per task): a ``cache_control: {"type":
  "ephemeral"}`` breakpoint on a content block caches the prompt up to that block for 5
  minutes; 5-minute cache writes cost 1.25x and reads 0.1x of base input (0.05x on Opus 5.5
  and Sonnet 5.5, 0.025x on Fable 5.1); minimum cacheable length 512-4,096 tokens by model,
  shorter prompts are processed uncached without error; caching "has no effect on output
  token generation" — Anthropic prompt-caching and pricing documentation, retrieved
  2026-10-31.
* OpenAI: ``POST /v1/responses`` with ``instructions``, ``input``, ``max_output_tokens``,
  ``reasoning.effort`` and ``store: false``; usage fields ``input_tokens`` (with the
  breakdown ``input_tokens_details.cached_tokens`` / ``cache_write_tokens``),
  ``output_tokens`` (with ``output_tokens_details.reasoning_tokens``) and
  ``incomplete_details.reason`` — checked against the official "Create a response" API
  reference on 2026-10-30. Exhausted quota as a 429 with code ``insufficient_quota``
  follows OpenAI's documented error codes as known up to 2025 (not re-checked). Prompt
  caching is automatic on OpenAI ("enabled by default for supported OpenAI models", minimum
  1,024 tokens on GPT-5.6 and later; cache writes billed 1.25x for 30-minute lifetimes on
  GPT-5.6+), so this adapter configures nothing and records the reported cached and
  cache-write tokens (OpenAI prompt-caching guide, retrieved 2026-10-31). The
  OpenAI path is validated against mocked responses only until a live check is run.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import httpx
from pydantic import SecretStr

from src.llm.catalogue import ModelCapability
from src.llm.credentials import redact, register_secret
from src.llm.provider import (
    InsufficientFundsError,
    LLMConfig,
    LLMOutputError,
    LLMRequest,
    LLMResponse,
    ProviderAccessError,
    ProviderUnavailableError,
    TokenUsage,
    UnsupportedConfigurationError,
)

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2026-10-21"
OPENAI_URL = "https://api.openai.com/v1/responses"
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504, 529})


class _RetryError(Exception):
    def __init__(self, error: Exception, retry_after: float | None = None) -> None:
        self.error = error
        self.retry_after = retry_after


def _error_message(resp: httpx.Response) -> tuple[str, str, str]:
    """(type, code, message) from a provider error body; all redacted and bounded."""
    try:
        body = resp.json()
    except ValueError:
        return "", "", redact(resp.text[:300])
    err = body.get("error") if isinstance(body, dict) else None
    if not isinstance(err, dict):
        return "", "", redact(str(body)[:300])
    return (
        str(err.get("type") or ""),
        str(err.get("code") or ""),
        redact(str(err.get("message") or ""))[:300],
    )


def _retry_after(resp: httpx.Response) -> float | None:
    value = resp.headers.get("retry-after")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def _int(value: Any) -> int | None:
    return int(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


class _HTTPProvider:
    name = ""
    url = ""

    def __init__(
        self,
        config: LLMConfig,
        capability: ModelCapability,
        api_key: SecretStr,
        *,
        client: httpx.Client | None = None,
        max_retries: int = 2,
        timeout_seconds: float = 300.0,
        max_backoff_seconds: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if config.provider != self.name or capability.provider != self.name:
            raise UnsupportedConfigurationError(f"configuration is not for {self.name}")
        if capability.model_id != config.model_id:
            raise UnsupportedConfigurationError("capability entry names a different model")
        if not 0 <= max_retries <= 5:
            raise ValueError("max_retries must be between 0 and 5")
        self.config = config
        self.capability = capability
        self._key = api_key
        register_secret(api_key.get_secret_value())
        self._client = client or httpx.Client()
        self._max_retries = max_retries
        self._timeout = timeout_seconds
        self._max_backoff = max_backoff_seconds
        self._sleep = sleep

    @property
    def provider_name(self) -> str:
        return self.name

    @property
    def model_id(self) -> str:
        return self.config.model_id

    # -- to implement -------------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        raise NotImplementedError

    def _body(self, request: LLMRequest) -> dict[str, Any]:
        raise NotImplementedError

    def _classify(self, resp: httpx.Response) -> Exception:
        raise NotImplementedError

    def _parse(self, data: dict[str, Any], request_id: str | None, attempts: int) -> LLMResponse:
        raise NotImplementedError

    # -- shared -------------------------------------------------------------------
    def _check_request(self, request: LLMRequest) -> None:
        if request.temperature is not None and not self.capability.supports_temperature:
            raise UnsupportedConfigurationError(
                f"{self.name}/{self.model_id} does not accept a temperature in this "
                "integration; configure the task with temperature unset"
            )
        limit = self.capability.max_output_tokens
        if limit is not None and request.max_output_tokens > limit:
            raise UnsupportedConfigurationError(
                f"max_output_tokens {request.max_output_tokens} exceeds {limit}"
            )

    def _check_model(self, served: Any) -> str:
        served_id = str(served or "")
        if served_id != self.model_id and not served_id.startswith(self.model_id + "-"):
            raise UnsupportedConfigurationError(
                f"{self.name} served model {served_id!r} although {self.model_id!r} was "
                "requested; the result was discarded"
            )
        return served_id

    def complete(self, request: LLMRequest) -> LLMResponse:
        self._check_request(request)
        body = self._body(request)
        attempts = 0
        while True:
            attempts += 1
            try:
                return self._attempt(body, attempts)
            except _RetryError as retry:
                if attempts > self._max_retries:
                    raise retry.error from None
                delay = retry.retry_after if retry.retry_after is not None else 2.0**attempts
                self._sleep(min(max(delay, 0.0), self._max_backoff))

    def _attempt(self, body: dict[str, Any], attempts: int) -> LLMResponse:
        try:
            resp = self._client.post(
                self.url, json=body, headers=self._headers(), timeout=self._timeout
            )
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise _RetryError(
                ProviderUnavailableError(f"{self.name}: connection failed ({type(exc).__name__})")
            ) from None
        except httpx.TimeoutException as exc:
            raise ProviderUnavailableError(
                f"{self.name}: no response within {self._timeout:.0f}s ({type(exc).__name__}); "
                "not retried automatically because the provider may already have processed "
                "and billed the request"
            ) from None
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(
                f"{self.name}: transport error ({type(exc).__name__})"
            ) from None
        request_id = resp.headers.get("request-id") or resp.headers.get("x-request-id")
        if resp.status_code != 200:
            error = self._classify(resp)
            if resp.status_code in RETRYABLE_STATUS and isinstance(error, ProviderUnavailableError):
                raise _RetryError(error, _retry_after(resp))
            raise error
        try:
            data = resp.json()
        except ValueError:
            raise LLMOutputError(f"{self.name}: response body is not JSON") from None
        if not isinstance(data, dict):
            raise LLMOutputError(f"{self.name}: unexpected response body")
        return self._parse(data, request_id, attempts)

    def _effective_effort(self) -> str:
        if self.config.effort is not None:
            return self.config.effort
        default = self.capability.provider_default_effort
        return f"provider default (not sent; catalogue lists {default or 'unknown'})"


class AnthropicMessagesProvider(_HTTPProvider):
    name = "anthropic"
    url = ANTHROPIC_URL

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._key.get_secret_value(),
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

    def _body(self, request: LLMRequest) -> dict[str, Any]:
        content: str | list[dict[str, Any]] = request.full_prompt
        if self.config.prompt_caching and request.cacheable_prefix:
            # Explicit breakpoint after the record-invariant prefix (protocol, criteria): the
            # same bytes are sent, split into two text blocks. Prompts below the model's
            # minimum cacheable length are processed without caching (no error).
            content = [
                {"type": "text", "text": request.cacheable_prefix,
                 "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": request.prompt},
            ]  # fmt: skip
        body: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": request.max_output_tokens,
            "system": request.system,
            "messages": [{"role": "user", "content": content}],
        }
        if self.capability.thinking == "adaptive":
            body["thinking"] = {"type": "adaptive"}
        if self.config.effort is not None:
            body["output_config"] = {"effort": self.config.effort}
        return body

    def _classify(self, resp: httpx.Response) -> Exception:
        etype, _, msg = _error_message(resp)
        status = resp.status_code
        where = f"Anthropic API ({status} {etype or 'error'})"
        if status == 401 or etype == "authentication_error":
            return ProviderAccessError(f"{where}: the API key was rejected. {msg}")
        if status == 402 or etype == "billing_error" or "credit balance" in msg.lower():
            return InsufficientFundsError(
                f"{where}: the Anthropic API account cannot pay for this request (add credit "
                f"in the Anthropic Console; Claude subscription/development credits do not "
                f"apply). {msg}"
            )
        if status == 403 or etype == "permission_error":
            return ProviderAccessError(f"{where}: this key may not use this model. {msg}")
        if status == 404 or etype == "not_found_error":
            return ProviderAccessError(
                f"{where}: model {self.model_id!r} not found or not available to this "
                f"account. {msg}"
            )
        if status == 413:
            return ProviderUnavailableError(f"{where}: request too large for the model. {msg}")
        if status in RETRYABLE_STATUS:
            return ProviderUnavailableError(f"{where}: temporarily unavailable. {msg}")
        if status == 400:
            return UnsupportedConfigurationError(f"{where}: request rejected — {msg}")
        return ProviderUnavailableError(f"{where}: {msg}")

    def _parse(self, data: dict[str, Any], request_id: str | None, attempts: int) -> LLMResponse:
        served = self._check_model(data.get("model"))
        stop = data.get("stop_reason")
        if stop == "refusal":
            details = data.get("stop_details") or {}
            category = details.get("category") if isinstance(details, dict) else None
            raise LLMOutputError(f"Anthropic: the model declined the request ({category})")
        if stop == "max_tokens":
            raise LLMOutputError(
                f"Anthropic: output truncated at max_output_tokens={self.config.max_output_tokens}"
                " (thinking tokens count towards this limit); raise it for this task"
            )
        blocks = data.get("content") or []
        text = "".join(
            str(b.get("text", ""))
            for b in blocks
            if isinstance(b, dict) and b.get("type") == "text"
        )
        if not text.strip():
            raise LLMOutputError("Anthropic: response contains no text")
        raw = data.get("usage") or {}
        usage = TokenUsage(
            input_tokens=_int(raw.get("input_tokens")),
            output_tokens=_int(raw.get("output_tokens")),
            reasoning_tokens=None,  # included in output_tokens; not reported separately
            cache_read_tokens=_int(raw.get("cache_read_input_tokens")),
            cache_write_tokens=_int(raw.get("cache_creation_input_tokens")),
        )
        settings = {
            "api": "anthropic.messages",
            "anthropic_version": ANTHROPIC_VERSION,
            "requested_model": self.model_id,
            "served_model": served,
            "thinking": "adaptive" if self.capability.thinking == "adaptive" else "not sent",
            "effort_sent": self.config.effort,
            "effort_effective": self._effective_effort(),
            "max_output_tokens": self.config.max_output_tokens,
            "temperature": None,
            "stop_reason": stop,
            "prompt_caching": "explicit 5-minute breakpoint after the record-invariant prefix"
            if self.config.prompt_caching
            else "not requested",
        }
        return LLMResponse(
            text=text, provider=self.name, model_id=served, usage=usage,
            effective_settings=settings, provider_request_id=request_id, attempts=attempts,
        )  # fmt: skip


class OpenAIResponsesProvider(_HTTPProvider):
    name = "openai"
    url = OPENAI_URL

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._key.get_secret_value()}",
            "content-type": "application/json",
        }

    def _body(self, request: LLMRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model_id,
            "instructions": request.system,
            "input": request.full_prompt,
            "max_output_tokens": request.max_output_tokens,
            "store": False,
        }
        if self.config.effort is not None:
            body["reasoning"] = {"effort": self.config.effort}
        return body

    def _classify(self, resp: httpx.Response) -> Exception:
        etype, code, msg = _error_message(resp)
        status = resp.status_code
        where = f"OpenAI API ({status} {code or etype or 'error'})"
        if status == 401 or code == "invalid_api_key":
            return ProviderAccessError(f"{where}: the API key was rejected. {msg}")
        if code == "insufficient_quota" or etype == "insufficient_quota":
            return InsufficientFundsError(
                f"{where}: the OpenAI API account has no remaining quota/credit (check "
                f"billing on the OpenAI Platform; ChatGPT subscriptions do not apply). {msg}"
            )
        if status == 403:
            return ProviderAccessError(f"{where}: access denied for this key/model. {msg}")
        if status == 404 or code == "model_not_found":
            return ProviderAccessError(
                f"{where}: model {self.model_id!r} not found or not available to this "
                f"account. {msg}"
            )
        if code == "context_length_exceeded" or status == 413:
            return ProviderUnavailableError(f"{where}: input too long for the model. {msg}")
        if status in RETRYABLE_STATUS:
            return ProviderUnavailableError(f"{where}: temporarily unavailable. {msg}")
        if status == 400:
            return UnsupportedConfigurationError(f"{where}: request rejected — {msg}")
        return ProviderUnavailableError(f"{where}: {msg}")

    def _parse(self, data: dict[str, Any], request_id: str | None, attempts: int) -> LLMResponse:
        served = self._check_model(data.get("model"))
        status = data.get("status")
        if status == "incomplete":
            details = data.get("incomplete_details") or {}
            reason = details.get("reason") if isinstance(details, dict) else None
            raise LLMOutputError(
                f"OpenAI: response incomplete ({reason}); with max_output_tokens="
                f"{self.config.max_output_tokens} (reasoning tokens count towards it)"
            )
        if status not in ("completed", None):
            raise LLMOutputError(f"OpenAI: response status {status!r}")
        echoed = data.get("reasoning")
        echoed_effort = echoed.get("effort") if isinstance(echoed, dict) else None
        if self.config.effort is not None and echoed_effort not in (None, self.config.effort):
            raise UnsupportedConfigurationError(
                f"OpenAI applied reasoning effort {echoed_effort!r} although "
                f"{self.config.effort!r} was requested; the result was discarded"
            )
        parts: list[str] = []
        for item in data.get("output") or []:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            for c in item.get("content") or []:
                if isinstance(c, dict) and c.get("type") == "refusal":
                    raise LLMOutputError("OpenAI: the model declined the request (refusal)")
                if isinstance(c, dict) and c.get("type") == "output_text":
                    parts.append(str(c.get("text", "")))
        text = "".join(parts)
        if not text.strip():
            raise LLMOutputError("OpenAI: response contains no output text")
        raw = data.get("usage") or {}
        in_details = raw.get("input_tokens_details") or {}
        out_details = raw.get("output_tokens_details") or {}
        usage = TokenUsage(
            input_tokens=_int(raw.get("input_tokens")),
            output_tokens=_int(raw.get("output_tokens")),
            reasoning_tokens=_int(out_details.get("reasoning_tokens")),
            cache_read_tokens=_int(in_details.get("cached_tokens")),
            cache_write_tokens=_int(in_details.get("cache_write_tokens")),
        )
        settings = {
            "api": "openai.responses",
            "requested_model": self.model_id,
            "served_model": served,
            "effort_sent": self.config.effort,
            "effort_effective": echoed_effort or self._effective_effort(),
            "max_output_tokens": self.config.max_output_tokens,
            "temperature": None,
            "store": False,
            "status": status,
            "prompt_caching": "provider automatic (not configured by this software)",
        }
        return LLMResponse(
            text=text, provider=self.name, model_id=served, usage=usage,
            effective_settings=settings, provider_request_id=request_id, attempts=attempts,
        )  # fmt: skip
