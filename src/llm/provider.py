"""LLM provider interface, configuration and registry.

Task modules (screening, extraction, risk of bias) depend only on :class:`LLMProvider`;
which vendor and model run is decided by an :class:`LLMConfig` resolved through a
:class:`ProviderRegistry`. No provider is registered by default and none is hard-coded:
asking for an unregistered provider raises :class:`ProviderNotConfiguredError` rather
than silently falling back to anything. Unit tests register deterministic scripted
providers, so no credentials or network access are needed.

LLM output is always treated as a *proposal*: callers validate it against schemas and
against the source documents before anything is persisted.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pydantic import Field

from src.core.exceptions import SysRevError
from src.core.schemas.base import NonEmptyStr, StrictModel


class ProviderNotConfiguredError(SysRevError):
    """No usable LLM provider is configured for the requested task."""


class LLMResponseError(SysRevError, ValueError):
    """A provider response could not be parsed into the expected structure."""


class LLMRunHaltError(SysRevError):
    """A failure that makes every further call of the run pointless (or unsafe).

    Batch runners stop at the first one instead of recording a failure per record:
    missing/invalid credentials, exhausted funds, an unsupported model/effort
    combination, or a local budget pause. Nothing is recorded for the halted item.
    """


class ProviderAccessError(LLMRunHaltError):
    """Credentials missing or rejected, no permission, or the model is not available."""


class InsufficientFundsError(ProviderAccessError):
    """The provider reports exhausted credit/quota for the API account (not retried)."""


class UnsupportedConfigurationError(LLMRunHaltError, ProviderNotConfiguredError):
    """The requested provider/model/effort/parameter combination is not supported.

    Raised before any network call when the capability catalogue rules it out, and
    after one when the provider rejects a parameter. Never answered by silently
    switching to another model or effort.
    """


class BudgetPauseError(LLMRunHaltError):
    """The project's locally configured stop threshold was reached; no call was made."""


class ProviderUnavailableError(SysRevError):
    """Transient provider failure (rate limit, overload, server/network error) that
    persisted after the bounded retries. The item can be retried later."""


class LLMOutputError(LLMResponseError):
    """The provider answered without usable text: refusal, truncation at the output
    token limit, or an empty/incomplete response."""


class LLMConfig(StrictModel):
    """Which model performs a task. Kept separate from task logic and prompts."""

    provider: NonEmptyStr = Field(description='registered provider name, e.g. "anthropic"')
    model_id: NonEmptyStr
    label: NonEmptyStr = Field(description='role label, e.g. "llm-pass-1"')
    temperature: float | None = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=1024, ge=16, le=128000)
    effort: str | None = Field(
        default=None,
        description="exact provider effort value (e.g. 'high'); None = do not send one "
        "and let the provider apply its documented default",
    )
    prompt_caching: bool = Field(
        default=False,
        description="Anthropic only, explicit opt-in: mark the request's record-invariant "
        "prefix with a 5-minute cache_control breakpoint. Off by default (cache writes cost "
        "more than ordinary input; savings depend on reuse and are not assumed)",
    )


@dataclass(frozen=True, slots=True)
class LLMRequest:
    """One model request. ``cacheable_prefix`` is the record-invariant beginning of the user
    prompt (e.g. the protocol and criteria); the text sent is ``cacheable_prefix + prompt``
    whether or not caching is used, and the hash covers exactly that text, so enabling
    caching never changes what the model reads or the request's identity."""

    system: str
    prompt: str
    response_schema: dict[str, Any] = field(default_factory=dict)
    temperature: float | None = 0.0
    max_output_tokens: int = 1024
    cacheable_prefix: str = ""

    @property
    def full_prompt(self) -> str:
        return self.cacheable_prefix + self.prompt

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            {"system": self.system, "prompt": self.full_prompt, "schema": self.response_schema},
            sort_keys=True,
            ensure_ascii=False,
        )
        return sha256_text(payload)


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token counts exactly as reported by the provider (``None`` = not reported)."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    provider: str
    model_id: str
    usage: TokenUsage | None = None
    effective_settings: dict[str, Any] = field(default_factory=dict)
    provider_request_id: str | None = None
    attempts: int = 1

    @property
    def sha256(self) -> str:
        return sha256_text(self.text)


@runtime_checkable
class LLMProvider(Protocol):
    @property
    def provider_name(self) -> str: ...

    @property
    def model_id(self) -> str: ...

    def complete(self, request: LLMRequest) -> LLMResponse: ...


ProviderFactory = Callable[[LLMConfig], LLMProvider]


class ProviderRegistry:
    """Maps provider names to factories. Empty by default — nothing is implicit."""

    def __init__(self) -> None:
        self._factories: dict[str, ProviderFactory] = {}

    def register(self, name: str, factory: ProviderFactory) -> None:
        if not name.strip():
            raise ValueError("provider name must not be empty")
        self._factories[name] = factory

    def unregister(self, name: str) -> None:
        self._factories.pop(name, None)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

    def build(self, config: LLMConfig | None) -> LLMProvider:
        if config is None:
            raise ProviderNotConfiguredError(
                "no LLM configuration supplied; configure a provider and model explicitly"
            )
        factory = self._factories.get(config.provider)
        if factory is None:
            raise ProviderNotConfiguredError(
                f"LLM provider {config.provider!r} is not registered "
                f"(registered: {list(self.names) or 'none'})"
            )
        provider = factory(config)
        if provider.model_id != config.model_id:
            raise ProviderNotConfiguredError(
                f"provider built model {provider.model_id!r}, configuration asked for "
                f"{config.model_id!r}"
            )
        return provider


_default = ProviderRegistry()


def default_registry() -> ProviderRegistry:
    return _default


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a model reply that must be exactly one JSON object (code fences tolerated)."""
    stripped = text.strip()
    fenced = _FENCE_RE.match(stripped)
    if fenced:
        stripped = fenced.group(1)
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise LLMResponseError(f"response is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise LLMResponseError("response JSON must be an object")
    return data
