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


class LLMConfig(StrictModel):
    """Which model performs a task. Kept separate from task logic and prompts."""

    provider: NonEmptyStr = Field(description='registered provider name, e.g. "anthropic"')
    model_id: NonEmptyStr
    label: NonEmptyStr = Field(description='role label, e.g. "llm-pass-1"')
    temperature: float | None = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=1024, ge=16, le=32000)


@dataclass(frozen=True, slots=True)
class LLMRequest:
    system: str
    prompt: str
    response_schema: dict[str, Any] = field(default_factory=dict)
    temperature: float | None = 0.0
    max_output_tokens: int = 1024

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            {"system": self.system, "prompt": self.prompt, "schema": self.response_schema},
            sort_keys=True,
            ensure_ascii=False,
        )
        return sha256_text(payload)


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    provider: str
    model_id: str

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
