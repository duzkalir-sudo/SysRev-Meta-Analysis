"""Reproducible audit trail and usage accounting for every LLM call.

:class:`AuditedProvider` wraps any :class:`LLMProvider` and records one
:class:`LLMCallRecord` per call — successful, failed or blocked — with the task, the
requested and served model, the effort sent and the effective effort, token counts as
reported by the provider, a list-price cost *estimate*, retries, latency and hashes of
the prompt and reply. Prompts, replies and credentials are not stored in the ledger.

:class:`BudgetGuard` implements *local* warnings and a *local* stop threshold: before
each call it compares the project's recorded estimated spend/tokens with the configured
limits and refuses to start a new call once a stop limit is reached. This is not a hard
spending cap — estimates can differ from the invoice and a call already in flight is
always completed — so a provider-side limit (Anthropic Console / OpenAI Platform) is
still needed for a guaranteed ceiling.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Literal

from pydantic import Field

from src.core.exceptions import SysRevError
from src.core.schemas.base import AwareDatetime, StrictModel, new_id, utc_now
from src.llm.catalogue import ModelCapability, estimate_cost
from src.llm.credentials import redact
from src.llm.provider import (
    BudgetPauseError,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    TokenUsage,
    UnsupportedConfigurationError,
    sha256_text,
)

CallStatus = Literal["ok", "error", "blocked_budget", "skipped_duplicate"]


class RedundantCallError(SysRevError):
    """An identical request (same task, model, effort and prompt) already produced a
    validated result that was persisted."""


class LLMCallRecord(StrictModel):
    call_id: str = Field(default_factory=new_id)
    project_id: str
    task: str
    label: str
    provider: str
    requested_model: str
    served_model: str | None = None
    effort_requested: str | None = None
    effort_effective: str | None = None
    status: CallStatus
    error_type: str | None = None
    error_message: str | None = None
    call_key: str
    request_sha256: str
    response_sha256: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    cost_usd_estimate: float | None = None
    cost_basis: str | None = None
    provider_request_id: str | None = None
    attempts: int = 0
    latency_ms: int | None = None
    settings: dict[str, Any] = Field(default_factory=dict)
    created_at: AwareDatetime = Field(default_factory=utc_now)

    @property
    def total_tokens(self) -> int:
        """Tokens processed by the call, without double counting.

        The raw fields are kept exactly as reported; their meaning differs by provider:

        * OpenAI Responses: ``input_tokens`` already *include* the cached and
          cache-write tokens (``input_tokens_details`` is a breakdown of the input) and
          ``output_tokens`` include the reasoning tokens, so the total is input + output.
        * Anthropic Messages: ``input_tokens``, ``cache_read_input_tokens`` and
          ``cache_creation_input_tokens`` are disjoint, so all three are added to output.
        * Any other provider: input + output (cache fields are not assumed to be disjoint).
        """
        base = (self.input_tokens or 0) + (self.output_tokens or 0)
        if self.provider == "anthropic":
            return base + (self.cache_read_tokens or 0) + (self.cache_write_tokens or 0)
        return base


def call_key(task: str, provider: str, model: str, effort: str | None, request: LLMRequest) -> str:
    """Identity of a request for redundancy checks. The task is part of it, so two
    independent screening passes never share a key."""
    return sha256_text(f"{task}|{provider}|{model}|{effort or ''}|{request.sha256}")


class BudgetGuard:
    def __init__(
        self,
        *,
        spent_usd: float,
        tokens: int,
        unpriced_calls: int = 0,
        warn_usd: float | None = None,
        stop_usd: float | None = None,
        warn_tokens: int | None = None,
        stop_tokens: int | None = None,
    ) -> None:
        self.spent_usd = spent_usd
        self.tokens = tokens
        self.unpriced_calls = unpriced_calls
        self.warn_usd, self.stop_usd = warn_usd, stop_usd
        self.warn_tokens, self.stop_tokens = warn_tokens, stop_tokens

    def require_priced(self, capability: ModelCapability) -> None:
        if self.stop_usd is not None and capability.pricing is None:
            raise UnsupportedConfigurationError(
                f"a cost stop threshold is set but {capability.provider}/"
                f"{capability.model_id} has no list price in the catalogue, so its spend "
                "cannot be tracked; add pricing to the catalogue or use a token threshold"
            )

    def check(self) -> None:
        if self.stop_usd is not None and self.spent_usd >= self.stop_usd:
            raise BudgetPauseError(
                f"estimated project spend ${self.spent_usd:.4f} reached the local stop "
                f"threshold ${self.stop_usd:.2f}; raise the threshold to continue"
            )
        if self.stop_tokens is not None and self.tokens >= self.stop_tokens:
            raise BudgetPauseError(
                f"project token usage {self.tokens} reached the local stop threshold "
                f"{self.stop_tokens}; raise the threshold to continue"
            )

    def add(self, record: LLMCallRecord) -> None:
        self.tokens += record.total_tokens
        if record.cost_usd_estimate is not None:
            self.spent_usd += record.cost_usd_estimate
        elif record.status == "ok":
            self.unpriced_calls += 1

    def warnings(self) -> list[str]:
        out = []
        if self.warn_usd is not None and self.spent_usd >= self.warn_usd:
            out.append(f"estimated spend ${self.spent_usd:.4f} ≥ warning ${self.warn_usd:.2f}")
        if self.warn_tokens is not None and self.tokens >= self.warn_tokens:
            out.append(f"token usage {self.tokens} ≥ warning {self.warn_tokens}")
        if self.unpriced_calls:
            out.append(f"{self.unpriced_calls} call(s) without a price: spend is understated")
        return out


class AuditedProvider:
    """Records every call of ``inner`` through ``sink`` (the billing/usage audit trail);
    optionally enforces the budget guard.

    ``already_answered(call_key)`` must return True only when a *validated workflow
    result* of an identical request was persisted — not merely when the provider
    answered. A reply that later failed JSON/schema validation or evidence grounding
    therefore never blocks an ordinary retry.
    """

    def __init__(
        self,
        inner: LLMProvider,
        *,
        project_id: str,
        task: str,
        label: str,
        effort: str | None,
        capability: ModelCapability | None,
        sink: Callable[[LLMCallRecord], None],
        budget: BudgetGuard | None = None,
        already_answered: Callable[[str], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.inner = inner
        self.project_id = project_id
        self.task = task
        self.label = label
        self.effort = effort
        self.capability = capability
        self._sink = sink
        self.budget = budget
        self._already_answered = already_answered
        self._clock = clock
        self.records: list[LLMCallRecord] = []

    @property
    def provider_name(self) -> str:
        return self.inner.provider_name

    @property
    def model_id(self) -> str:
        return self.inner.model_id

    def _record(self, request: LLMRequest, key: str, **fields: Any) -> LLMCallRecord:
        record = LLMCallRecord(
            project_id=self.project_id, task=self.task, label=self.label,
            provider=self.provider_name, requested_model=self.model_id,
            effort_requested=self.effort, call_key=key, request_sha256=request.sha256,
            **fields,
        )  # fmt: skip
        self.records.append(record)
        self._sink(record)
        if self.budget is not None:
            self.budget.add(record)
        return record

    def complete(self, request: LLMRequest) -> LLMResponse:
        key = call_key(self.task, self.provider_name, self.model_id, self.effort, request)
        if self._already_answered is not None and self._already_answered(key):
            self._record(request, key, status="skipped_duplicate")
            raise RedundantCallError(
                f"an identical {self.task} request to {self.model_id} already produced a "
                "saved, validated result; not repeated (use 'force' to pay for a repeat)"
            )
        if self.budget is not None:
            try:
                self.budget.check()
            except BudgetPauseError as exc:
                self._record(request, key, status="blocked_budget", error_message=str(exc))
                raise
        start = self._clock()
        try:
            response = self.inner.complete(request)
        except Exception as exc:
            self._record(
                request, key, status="error", error_type=type(exc).__name__,
                error_message=redact(str(exc))[:500],
                latency_ms=int((self._clock() - start) * 1000),
            )  # fmt: skip
            raise
        usage = response.usage or TokenUsage()
        cost = estimate_cost(self.capability, response.usage)
        settings = dict(response.effective_settings)
        self._record(
            request, key, status="ok", served_model=response.model_id,
            effort_effective=settings.get("effort_effective"),
            response_sha256=response.sha256, input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens, reasoning_tokens=usage.reasoning_tokens,
            cache_read_tokens=usage.cache_read_tokens, cache_write_tokens=usage.cache_write_tokens,
            cost_usd_estimate=cost.usd, cost_basis=cost.basis,
            provider_request_id=response.provider_request_id, attempts=response.attempts,
            latency_ms=int((self._clock() - start) * 1000), settings=settings,
        )  # fmt: skip
        return response
