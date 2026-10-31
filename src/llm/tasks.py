"""Per-task model configuration and construction of real (API) providers.

Each AI task has its own provider, exact API model ID, effort and output limit, so the
two screening passes, extraction, the advisory verifier and risk-of-bias drafting can use
different models. Settings are stored per project (append-only versions) and contain no
credentials. ``effort=None`` means "provider default": no effort parameter is sent, and
the audit trail records that the provider's documented default applied.
"""

from __future__ import annotations

from enum import StrEnum

import httpx
from pydantic import Field, model_validator

from src.core.schemas.base import NonEmptyStr, StrictModel
from src.llm.adapters import AnthropicMessagesProvider, OpenAIResponsesProvider
from src.llm.catalogue import ModelCapability, ModelCatalogue
from src.llm.credentials import CredentialResolver
from src.llm.provider import (
    LLMConfig,
    LLMProvider,
    ProviderNotConfiguredError,
    ProviderRegistry,
)


class AITask(StrEnum):
    SCREENING_REVIEWER_1 = "screening_reviewer_1"
    SCREENING_REVIEWER_2 = "screening_reviewer_2"
    EXTRACTION = "extraction"
    VERIFICATION = "verification"
    RISK_OF_BIAS = "risk_of_bias"
    # Session 6 (absent from older saved settings, which therefore remain valid)
    MANUSCRIPT_DRAFTING = "manuscript_drafting"
    CLAIM_REVIEW = "claim_review"


TASK_LABELS: dict[AITask, str] = {
    AITask.SCREENING_REVIEWER_1: "Screening — automated pass 1",
    AITask.SCREENING_REVIEWER_2: "Screening — automated pass 2",
    AITask.EXTRACTION: "PDF data extraction (proposals)",
    AITask.VERIFICATION: "Advisory verification of extracted values",
    AITask.RISK_OF_BIAS: "Risk-of-bias drafts",
    AITask.MANUSCRIPT_DRAFTING: "Manuscript prose drafting (numbers/citations as markers only)",
    AITask.CLAIM_REVIEW: "Advisory claim-support review (never an approval)",
}


class TaskModelSettings(StrictModel):
    provider: NonEmptyStr
    model_id: NonEmptyStr
    effort: str | None = None
    max_output_tokens: int = Field(default=16000, ge=256, le=128000)
    prompt_caching: bool = Field(
        default=False,
        description="Anthropic explicit prompt caching of the record-invariant prefix "
        "(opt-in; absent from settings saved before Session 8, which therefore stay off)",
    )

    def to_llm_config(self, task: AITask) -> LLMConfig:
        return LLMConfig(
            provider=self.provider,
            model_id=self.model_id,
            label=task.value,
            temperature=None,
            max_output_tokens=self.max_output_tokens,
            effort=self.effort,
            prompt_caching=self.prompt_caching,
        )


class ProjectAISettings(StrictModel):
    """No credentials here — only which model runs each task and local usage limits."""

    tasks: dict[AITask, TaskModelSettings] = Field(default_factory=dict)
    warn_usd: float | None = Field(default=None, ge=0)
    stop_usd: float | None = Field(default=None, ge=0)
    warn_tokens: int | None = Field(default=None, ge=0)
    stop_tokens: int | None = Field(default=None, ge=0)
    note: str | None = None
    catalogue_sha256: str | None = Field(
        default=None,
        description="SHA-256 of the model catalogue these settings were checked against when "
        "saved (Session 8; None for older settings). Provenance only: a later catalogue never "
        "rewrites the stored selection",
    )

    @model_validator(mode="after")
    def _ordered_limits(self) -> ProjectAISettings:
        if None not in (self.warn_usd, self.stop_usd) and self.warn_usd > self.stop_usd:  # type: ignore[operator]
            raise ValueError("warn_usd must not exceed stop_usd")
        if None not in (self.warn_tokens, self.stop_tokens) and self.warn_tokens > self.stop_tokens:  # type: ignore[operator]
            raise ValueError("warn_tokens must not exceed stop_tokens")
        return self

    def config_for(self, task: AITask) -> LLMConfig:
        settings = self.tasks.get(task)
        if settings is None:
            raise ProviderNotConfiguredError(f"no model configured for task {task.value!r}")
        return settings.to_llm_config(task)


def build_api_provider(
    config: LLMConfig,
    catalogue: ModelCatalogue,
    credentials: CredentialResolver,
    *,
    client: httpx.Client | None = None,
    max_retries: int = 2,
) -> tuple[LLMProvider, ModelCapability]:
    """Validate ``config`` exactly as given, then build the matching adapter."""
    capability = catalogue.validate(config)
    key = credentials.get(config.provider)
    cls = AnthropicMessagesProvider if config.provider == "anthropic" else OpenAIResponsesProvider
    return cls(config, capability, key, client=client, max_retries=max_retries), capability


def register_api_providers(
    registry: ProviderRegistry,
    catalogue: ModelCatalogue,
    credentials: CredentialResolver,
    *,
    client: httpx.Client | None = None,
) -> None:
    """Register the ``anthropic`` and ``openai`` adapters in an existing registry."""
    for name in ("anthropic", "openai"):
        registry.register(
            name,
            lambda cfg: build_api_provider(cfg, catalogue, credentials, client=client)[0],
        )
