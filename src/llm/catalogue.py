"""Model capability catalogue: validates a task configuration before any API call.

Provider parameters are *not* interchangeable: an effort value is passed to the provider
exactly as written (Anthropic ``output_config.effort``, OpenAI ``reasoning.effort``) and
only if the catalogue lists it for that exact model. No effort level is assumed to be
equivalent across providers. A model that is not in the catalogue cannot be used — the
user extends the catalogue (local override file, or a reviewed and activated catalogue
snapshot, :mod:`src.llm.model_registry`) after checking the provider's current
documentation, rather than the software guessing.

Session 8 separates four facts about a model, each with its own evidence:

* **discovery / account availability** — whether the provider's official model-list
  endpoint returned the ID for this API key (:mod:`src.llm.discovery`). Being listed never
  makes a model executable;
* **adapter compatibility** — which adapter (Anthropic Messages / OpenAI Responses) the
  documented interface of the model matches (``adapter``; ``unsupported`` blocks it);
* **verified capabilities** — efforts, default effort, thinking mode and output limit,
  checked against official documentation (``verified``, ``source``, ``verified_at``);
* **verified pricing** — list prices with their own source and date (``pricing_source``,
  ``pricing_verified_at``); unknown prices are reported as unavailable, never as zero.

Only a verified entry whose lifecycle is not ``retired`` and whose adapter is supported can
run. Nothing is substituted: a retired or unverified model fails with an explicit error.

Prices are list prices used for *estimates* only; the provider's invoice is authoritative.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from importlib import resources
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.llm.provider import LLMConfig, TokenUsage, UnsupportedConfigurationError

SUPPORTED_PROVIDERS = ("anthropic", "openai")
OVERRIDE_FILENAME = "model_catalogue.json"
ADAPTER_OF_PROVIDER: dict[str, str] = {
    "anthropic": "anthropic.messages",
    "openai": "openai.responses",
}
Lifecycle = Literal["active", "deprecated", "retired"]


class Pricing(BaseModel):
    """USD per million tokens."""

    model_config = ConfigDict(extra="forbid")
    input: float = Field(ge=0)
    output: float = Field(ge=0)
    cache_read: float | None = Field(default=None, ge=0)
    cache_write: float | None = Field(default=None, ge=0)
    long_prompt_threshold_tokens: int | None = Field(default=None, gt=0)
    long_prompt_input: float | None = Field(default=None, ge=0)
    long_prompt_output: float | None = Field(default=None, ge=0)
    long_prompt_cache_read: float | None = Field(default=None, ge=0)
    long_prompt_cache_write: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _long_prompt_complete(self) -> Pricing:
        tier = (self.long_prompt_threshold_tokens, self.long_prompt_input, self.long_prompt_output)
        if any(v is not None for v in tier) and any(v is None for v in tier):
            raise ValueError(
                "long-prompt pricing needs threshold, input and output prices together"
            )
        return self


class ModelCapability(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    provider: Literal["anthropic", "openai"]
    model_id: str = Field(min_length=1)
    display_name: str
    efforts: tuple[str, ...] = ()
    provider_default_effort: str | None = None
    thinking: Literal["adaptive", "none"] = "none"
    supports_temperature: bool = False
    max_output_tokens: int | None = Field(default=None, gt=0)
    pricing: Pricing | None = None
    source: str = Field(min_length=1)
    verified: bool
    # ---- Session 8 (all optional, so earlier catalogue files and snapshots stay valid)
    verified_at: date | None = Field(
        default=None, description="date the capabilities were checked against the source"
    )
    adapter: Literal["anthropic.messages", "openai.responses", "unsupported"] | None = Field(
        default=None,
        description="documented interface; None = the provider's standard adapter (the only "
        "one a verified entry could use before Session 8)",
    )
    pricing_source: str | None = Field(
        default=None, description="where the list prices come from (None: see source)"
    )
    pricing_verified_at: date | None = None
    lifecycle: Lifecycle | None = Field(
        default=None, description="provider lifecycle stage if recorded (None: not recorded)"
    )
    retires_at: date | None = None
    lifecycle_source: str | None = None
    provider_reported: dict[str, object] | None = Field(
        default=None,
        description="metadata the provider's model-list endpoint reported at discovery; shown "
        "to reviewers, never used for execution",
    )

    @property
    def interface(self) -> str:
        """The adapter this entry is run with (``unsupported`` = none)."""
        return self.adapter or ADAPTER_OF_PROVIDER[self.provider]

    def execution_problems(self) -> list[str]:
        """Why this entry may not run (empty = executable). Never answered by substitution."""
        out: list[str] = []
        name = f"{self.provider}/{self.model_id}"
        if not self.verified:
            out.append(
                f"{name} is an unverified catalogue entry (discovered or entered without "
                "verified capabilities); it is not available for execution until its "
                "capabilities are verified against official documentation and the catalogue "
                "snapshot is activated"
            )
        if self.interface == "unsupported":
            out.append(f"{name} is not compatible with this software's {self.provider} adapter")
        elif self.interface != ADAPTER_OF_PROVIDER[self.provider]:
            out.append(f"{name}: adapter {self.interface!r} does not belong to {self.provider}")
        if self.lifecycle == "retired":
            out.append(
                f"{name} is retired by the provider ({self.lifecycle_source or 'recorded'}); "
                "no other model is substituted — choose a model explicitly"
            )
        return out

    def warnings(self) -> list[str]:
        out: list[str] = []
        if self.lifecycle == "deprecated":
            when = f" (retirement scheduled {self.retires_at})" if self.retires_at else ""
            out.append(f"{self.provider}/{self.model_id} is deprecated{when}")
        if self.pricing is None:
            out.append(f"{self.provider}/{self.model_id} has no verified list price: costs "
                       "are reported as unavailable")  # fmt: skip
        return out

    @model_validator(mode="after")
    def _consistent(self) -> ModelCapability:
        if len(set(self.efforts)) != len(self.efforts):
            raise ValueError(f"{self.model_id}: duplicate effort values")
        if (
            self.provider_default_effort is not None
            and self.efforts
            and (self.provider_default_effort not in self.efforts)
        ):
            raise ValueError(f"{self.model_id}: default effort is not a listed effort")
        unverified_label = self.source.startswith("UNVERIFIED")
        if self.verified == unverified_label:
            raise ValueError(
                f"{self.model_id}: an unverified entry's source must start with "
                "'UNVERIFIED', and a verified entry's source must not"
            )
        if not self.verified and "UNVERIFIED" not in self.display_name:
            raise ValueError(f"{self.model_id}: unverified entries must say so in their name")
        if self.pricing_source is not None and self.pricing_source.startswith("UNVERIFIED"):
            raise ValueError(f"{self.model_id}: a pricing source cannot be UNVERIFIED")
        if self.pricing_verified_at is not None and self.pricing is None:
            raise ValueError(f"{self.model_id}: a pricing verification date needs prices")
        return self


class ModelCatalogue:
    def __init__(self, models: list[ModelCapability], *, snapshot_id: str | None = None) -> None:
        self._models: dict[tuple[str, str], ModelCapability] = {
            (m.provider, m.model_id): m for m in models
        }
        self.snapshot_id = snapshot_id  # None: built-in catalogue (+ local override file)

    @property
    def sha256(self) -> str:
        """Content hash of the catalogue (recorded with settings and every call)."""
        return catalogue_sha256(self.models())

    @classmethod
    def load(cls, override_dir: Path | None = None) -> ModelCatalogue:
        """Built-in catalogue, with entries from ``override_dir/model_catalogue.json``
        replacing or adding to it."""
        raw = resources.files("src.llm").joinpath("model_catalogue.json").read_text("utf-8")
        models = [ModelCapability.model_validate(m) for m in json.loads(raw)["models"]]
        keys = [(m.provider, m.model_id) for m in models]
        if len(set(keys)) != len(keys):
            raise ValueError("the built-in model catalogue lists a model more than once")
        if override_dir is not None and (override_dir / OVERRIDE_FILENAME).is_file():
            extra = json.loads((override_dir / OVERRIDE_FILENAME).read_text("utf-8"))
            models += [ModelCapability.model_validate(m) for m in extra.get("models", [])]
        return cls(models)

    def models(self, provider: str | None = None) -> list[ModelCapability]:
        return [m for (p, _), m in sorted(self._models.items()) if provider in (None, p)]

    def get(self, provider: str, model_id: str) -> ModelCapability:
        found = self._models.get((provider, model_id))
        if found is None:
            raise UnsupportedConfigurationError(
                f"model {model_id!r} is not in the {provider} capability catalogue; check "
                "the provider's documentation and add it to the local catalogue override "
                f"({OVERRIDE_FILENAME}) — the software does not guess model parameters"
            )
        return found

    def validate(self, config: LLMConfig) -> ModelCapability:
        """Raise :class:`UnsupportedConfigurationError` unless the configuration is
        supported exactly as requested. Nothing is adjusted or downgraded."""
        if config.provider not in SUPPORTED_PROVIDERS:
            raise UnsupportedConfigurationError(f"unknown provider {config.provider!r}")
        cap = self.get(config.provider, config.model_id)
        problems = cap.execution_problems()
        if problems:
            raise UnsupportedConfigurationError("; ".join(problems))
        if config.prompt_caching and cap.provider != "anthropic":
            raise UnsupportedConfigurationError(
                "explicit prompt caching is only configurable for Anthropic; OpenAI caches "
                "prompts automatically (cached tokens are recorded either way)"
            )
        if config.effort is not None and config.effort not in cap.efforts:
            allowed = ", ".join(cap.efforts) or "none (provider default only)"
            raise UnsupportedConfigurationError(
                f"effort {config.effort!r} is not supported by {config.provider}/"
                f"{config.model_id}; supported values: {allowed}"
            )
        if config.temperature is not None and not cap.supports_temperature:
            raise UnsupportedConfigurationError(
                f"{config.provider}/{config.model_id} does not accept a temperature "
                "parameter in this integration; leave it unset"
            )
        if cap.max_output_tokens is not None and config.max_output_tokens > cap.max_output_tokens:
            raise UnsupportedConfigurationError(
                f"max_output_tokens {config.max_output_tokens} exceeds the "
                f"{cap.max_output_tokens} limit of {config.model_id}"
            )
        return cap


def catalogue_sha256(models: list[ModelCapability]) -> str:
    data = sorted(
        (m.model_dump(mode="json") for m in models), key=lambda d: (d["provider"], d["model_id"])
    )
    text = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CostEstimate:
    usd: float | None
    basis: str


def estimate_cost(cap: ModelCapability | None, usage: TokenUsage | None) -> CostEstimate:
    """List-price estimate from reported tokens. ``usd=None`` when not derivable.

    Token semantics differ by provider and are normalised here:

    * Anthropic ``input_tokens`` exclude cache reads/writes (reported separately);
    * OpenAI ``input_tokens`` *include* ``cached_tokens`` and ``cache_write_tokens``
      (``input_tokens_details`` is a breakdown of the input); reasoning tokens are part
      of ``output_tokens`` (billed as output) on both.

    A long-prompt tier applies to the whole request when the prompt exceeds the
    threshold. Where the catalogue gives no long-tier cache price, cache tokens in that
    tier are priced at the long-tier input rate (an upper bound, stated in the basis).
    """
    if usage is None or usage.input_tokens is None or usage.output_tokens is None:
        return CostEstimate(None, "token usage not reported by the provider")
    if cap is None or cap.pricing is None:
        return CostEstimate(None, "no list price in the catalogue for this model")
    p = cap.pricing
    cache_read = usage.cache_read_tokens or 0
    cache_write = usage.cache_write_tokens or 0
    uncached = usage.input_tokens
    if cap.provider == "openai":
        uncached = max(usage.input_tokens - cache_read - cache_write, 0)
    total_prompt = uncached + cache_read + cache_write
    notes = [f"list price ({(cap.pricing_source or cap.source).split(';')[0]})"]
    in_rate, out_rate = p.input, p.output
    read_rate, write_rate = p.cache_read, p.cache_write
    if p.long_prompt_threshold_tokens and total_prompt > p.long_prompt_threshold_tokens:
        assert p.long_prompt_input is not None
        assert p.long_prompt_output is not None
        in_rate, out_rate = p.long_prompt_input, p.long_prompt_output
        read_rate, write_rate = p.long_prompt_cache_read, p.long_prompt_cache_write
        notes.append("long-prompt tier")
    if (cache_read and read_rate is None) or (cache_write and write_rate is None):
        notes.append("cache tokens priced at the input rate (upper bound)")
    usd = (
        uncached * in_rate
        + cache_read * (read_rate if read_rate is not None else in_rate)
        + cache_write * (write_rate if write_rate is not None else in_rate)
        + usage.output_tokens * out_rate
    ) / 1_000_000
    return CostEstimate(round(usd, 9), "estimate: " + "; ".join(notes))


def rough_prompt_tokens(text: str) -> int:
    """Crude pre-run size estimate (~4 characters per token). Only for warnings."""
    return max(1, len(text) // 4)
