"""Versioned model-catalogue snapshots: refresh, human verification, preview and activation.

The catalogue that validates task configurations (:class:`ModelCatalogue`) changes only
through an explicit, authenticated **activation** of a stored snapshot:

1. **Refresh** (:func:`refresh_snapshot`) lists models through the official endpoints
   (:mod:`src.llm.discovery`) and stores a *candidate* snapshot: every existing entry is kept
   unchanged, newly listed IDs are added as ``verified=False`` entries (never executable), and
   provider-reported lifecycle data (deprecated / retired) is recorded. Nothing is removed: a
   model the provider no longer lists stays in the catalogue, flagged as not listed. A failed
   listing changes nothing for that provider and is reported separately from model
   incompatibility.
2. **Verification** (:func:`verification_snapshot`) records capabilities a reviewer checked
   against official documentation (source URL and date) and, separately, list prices with
   their own source and date. Provider-reported metadata is used as a cross-check: an effort
   level or output limit the provider reports as unsupported is refused.
3. **Preview** (:func:`diff`) shows what activation would change.
4. **Activation** (``Repository.activate_catalogue_snapshot``) by an enrolled reviewer, bound to
   the snapshot hash and to the catalogue the preview was computed against.

Project settings are never rewritten: a stored selection that the new catalogue no longer
supports (e.g. a retired model) fails its preflight with an explicit reason and is shown as
stored. Every model call records the catalogue hash it was validated against.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any, Literal

from pydantic import Field, model_validator

from src.core.schemas.base import AwareDatetime, NonEmptyStr, StrictModel, new_id, utc_now
from src.llm.catalogue import (
    ADAPTER_OF_PROVIDER,
    ModelCapability,
    ModelCatalogue,
    Pricing,
    catalogue_sha256,
)
from src.llm.discovery import DiscoveryResult

SnapshotKind = Literal["refresh", "verification"]


class ProviderDiscovery(StrictModel):
    provider: Literal["anthropic", "openai"]
    status: Literal["ok", "access_error", "unavailable", "no_credentials", "bad_response"]
    endpoint: NonEmptyStr
    fetched_at: AwareDatetime
    listed_ids: list[str] = Field(default_factory=list)
    error: str | None = None


class CatalogueSnapshot(StrictModel):
    """An immutable candidate catalogue. ``sha256`` covers the model entries exactly."""

    snapshot_id: str = Field(default_factory=new_id)
    kind: SnapshotKind
    created_at: AwareDatetime = Field(default_factory=utc_now)
    created_by: NonEmptyStr
    reviewer_id: str | None = None
    parent_sha256: str = Field(min_length=64, max_length=64)
    parent_snapshot_id: str | None = None
    models: list[ModelCapability]
    discovery: list[ProviderDiscovery] = Field(default_factory=list)
    note: str | None = None

    @property
    def sha256(self) -> str:
        return catalogue_sha256(self.models)

    def catalogue(self) -> ModelCatalogue:
        return ModelCatalogue(list(self.models), snapshot_id=self.snapshot_id)


class ModelVerification(StrictModel):
    """Capabilities (and, separately, prices) a reviewer verified for one exact model ID."""

    provider: Literal["anthropic", "openai"]
    model_id: NonEmptyStr
    display_name: NonEmptyStr
    efforts: list[NonEmptyStr] = Field(default_factory=list)
    provider_default_effort: str | None = None
    thinking: Literal["adaptive", "none"] = "none"
    max_output_tokens: int | None = Field(default=None, gt=0)
    adapter: Literal["anthropic.messages", "openai.responses"]
    capability_source: NonEmptyStr
    capabilities_verified_on: date
    pricing: Pricing | None = None
    pricing_source: str | None = None
    pricing_verified_on: date | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _evidence(self) -> ModelVerification:
        if self.adapter != ADAPTER_OF_PROVIDER[self.provider]:
            raise ValueError(
                f"{self.provider} models run only with {ADAPTER_OF_PROVIDER[self.provider]}"
            )
        for label, source in (
            ("capability", self.capability_source),
            ("pricing", self.pricing_source),
        ):
            if source is not None and (
                source.upper().startswith("UNVERIFIED") or "https://" not in source
            ):
                raise ValueError(
                    f"the {label} source must cite the official documentation (an https:// "
                    "reference), not an unverified note"
                )
        priced = (self.pricing, self.pricing_source, self.pricing_verified_on)
        if any(v is not None for v in priced) and any(v is None for v in priced):
            raise ValueError(
                "prices need their own source and verification date (or leave all three "
                "empty: costs are then reported as unavailable)"
            )
        today = datetime.now(UTC).date()
        for d in (self.capabilities_verified_on, self.pricing_verified_on):
            if d is not None and d > today:
                raise ValueError("a verification date cannot be in the future")
        if len(set(self.efforts)) != len(self.efforts):
            raise ValueError("duplicate effort values")
        if (
            self.provider_default_effort is not None
            and self.provider_default_effort not in self.efforts
        ):
            raise ValueError("the provider default effort must be one of the listed efforts")
        return self


# --------------------------------------------------------------------------- building
def _unverified_entry(found: Any, endpoint: str, on: date) -> ModelCapability:
    name = found.display_name or found.model_id
    return ModelCapability(
        provider=found.provider,
        model_id=found.model_id,
        display_name=f"{name} (UNVERIFIED — discovered)",
        efforts=(),
        provider_default_effort=None,
        thinking="none",
        max_output_tokens=None,
        pricing=None,
        source=(
            f"UNVERIFIED: listed by {endpoint} on {on.isoformat()}; capabilities, adapter "
            "compatibility and prices not verified (an ID may also be an alias, which is never "
            "mapped to another model)"
        ),
        verified=False,
        lifecycle=found.lifecycle,
        retires_at=found.retires_at,
        lifecycle_source=f"{endpoint} ({on.isoformat()})" if found.lifecycle else None,
        provider_reported=dict(found.reported) or None,
    )


def refresh_snapshot(
    active: ModelCatalogue, results: list[DiscoveryResult], *, created_by: str
) -> CatalogueSnapshot:
    """A candidate catalogue from ``active`` plus what the providers listed. Never removes or
    verifies anything; refreshing twice with the same listing yields the same entries."""
    models: dict[tuple[str, str], ModelCapability] = {
        (m.provider, m.model_id): m for m in active.models()
    }
    discovery: list[ProviderDiscovery] = []
    for result in results:
        discovery.append(
            ProviderDiscovery(
                provider=result.provider,  # type: ignore[arg-type]
                status=result.status,
                endpoint=result.endpoint,
                fetched_at=result.fetched_at,
                listed_ids=result.listed_ids,
                error=result.error,
            )
        )
        if not result.ok:
            continue  # outage or access problem: the catalogue is left as it is
        on = result.fetched_at.date()
        for found in result.models:
            key = (found.provider, found.model_id)
            existing = models.get(key)
            if existing is None:
                models[key] = _unverified_entry(found, result.endpoint, on)
                continue
            update: dict[str, Any] = {}
            if found.lifecycle is not None and (
                found.lifecycle != existing.lifecycle or found.retires_at != existing.retires_at
            ):
                update |= {
                    "lifecycle": found.lifecycle,
                    "retires_at": found.retires_at,
                    "lifecycle_source": f"{result.endpoint} ({on.isoformat()})",
                }
            if found.reported and found.reported != existing.provider_reported:
                update["provider_reported"] = dict(found.reported)
            if update:
                models[key] = ModelCapability.model_validate(existing.model_dump() | update)
    return CatalogueSnapshot(
        kind="refresh",
        created_by=created_by,
        parent_sha256=active.sha256,
        parent_snapshot_id=active.snapshot_id,
        models=sorted(models.values(), key=lambda m: (m.provider, m.model_id)),
        discovery=discovery,
    )


class VerificationRefusedError(ValueError):
    """The verification contradicts what the provider reported, or is incomplete."""


def verification_snapshot(
    active: ModelCatalogue,
    verification: ModelVerification,
    *,
    created_by: str,
    reviewer_id: str,
) -> CatalogueSnapshot:
    v = verification
    existing = next((m for m in active.models(v.provider) if m.model_id == v.model_id), None)
    reported = (existing.provider_reported if existing else None) or {}
    problems: list[str] = []
    levels = reported.get("effort_levels")
    if isinstance(levels, list) and reported.get("effort_supported") is not None:
        unsupported = [e for e in v.efforts if e not in levels]
        if unsupported:
            problems.append(
                f"effort level(s) {unsupported} are not supported according to the provider's "
                f"model list (reported: {levels or 'none'})"
            )
    reported_max = reported.get("max_tokens")
    if (
        isinstance(reported_max, int)
        and reported_max > 0
        and v.max_output_tokens is not None
        and v.max_output_tokens > reported_max
    ):
        problems.append(
            f"max_output_tokens {v.max_output_tokens} exceeds the provider-reported "
            f"max_tokens {reported_max}"
        )
    if v.thinking == "adaptive" and reported.get("thinking_adaptive") is False:
        problems.append("the provider reports that adaptive thinking is not accepted")
    if v.thinking == "adaptive" and v.provider != "anthropic":
        problems.append("adaptive thinking is an Anthropic Messages setting")
    if problems:
        raise VerificationRefusedError("; ".join(problems))
    entry = ModelCapability(
        provider=v.provider,
        model_id=v.model_id,
        display_name=v.display_name.replace("UNVERIFIED", "").strip(" —()") or v.model_id,
        efforts=tuple(v.efforts),
        provider_default_effort=v.provider_default_effort,
        thinking=v.thinking,
        supports_temperature=False,
        max_output_tokens=v.max_output_tokens,
        pricing=v.pricing,
        source=v.capability_source,
        verified=True,
        verified_at=v.capabilities_verified_on,
        adapter=v.adapter,
        pricing_source=v.pricing_source,
        pricing_verified_at=v.pricing_verified_on,
        lifecycle=existing.lifecycle if existing else None,
        retires_at=existing.retires_at if existing else None,
        lifecycle_source=existing.lifecycle_source if existing else None,
        provider_reported=existing.provider_reported if existing else None,
    )
    models = {(m.provider, m.model_id): m for m in active.models()}
    models[(v.provider, v.model_id)] = entry
    return CatalogueSnapshot(
        kind="verification",
        created_by=created_by,
        reviewer_id=reviewer_id,
        parent_sha256=active.sha256,
        parent_snapshot_id=active.snapshot_id,
        models=sorted(models.values(), key=lambda m: (m.provider, m.model_id)),
        note=v.note,
    )


# --------------------------------------------------------------------------- preview
SHOWN_FIELDS = (
    "verified", "verified_at", "source", "efforts", "provider_default_effort", "thinking",
    "max_output_tokens", "adapter", "pricing", "pricing_source", "pricing_verified_at",
    "lifecycle", "retires_at",
)  # fmt: skip


def diff(old: ModelCatalogue, new: ModelCatalogue) -> dict[str, Any]:
    """What activating ``new`` in place of ``old`` changes (nothing is ever removed)."""
    a = {(m.provider, m.model_id): m for m in old.models()}
    b = {(m.provider, m.model_id): m for m in new.models()}
    added = [f"{p}/{mid}" for (p, mid) in sorted(set(b) - set(a))]
    removed = [f"{p}/{mid}" for (p, mid) in sorted(set(a) - set(b))]
    changed: dict[str, dict[str, list[Any]]] = {}
    for key in sorted(set(a) & set(b)):
        x, y = a[key].model_dump(mode="json"), b[key].model_dump(mode="json")
        fields = {f: [x.get(f), y.get(f)] for f in SHOWN_FIELDS if x.get(f) != y.get(f)}
        if fields:
            changed[f"{key[0]}/{key[1]}"] = fields
    newly_executable = [
        k for k in added + list(changed)
        if not b[tuple(k.split("/", 1))].execution_problems()  # type: ignore[index]
        and (tuple(k.split("/", 1)) not in a or a[tuple(k.split("/", 1))].execution_problems())  # type: ignore[index]
    ]  # fmt: skip
    newly_blocked = [
        k for k in changed
        if b[tuple(k.split("/", 1))].execution_problems()  # type: ignore[index]
        and not a[tuple(k.split("/", 1))].execution_problems()  # type: ignore[index]
    ]  # fmt: skip
    return {
        "from_sha256": old.sha256,
        "to_sha256": new.sha256,
        "added": added,
        "removed": removed,
        "changed": changed,
        "newly_executable": newly_executable,
        "newly_blocked": newly_blocked,
        "unchanged": new.sha256 == old.sha256,
    }


# --------------------------------------------------------------------------- status view
def model_status(
    catalogue: ModelCatalogue, discovery: dict[str, ProviderDiscovery]
) -> list[dict[str, Any]]:
    """One row per model, keeping discovery, availability, compatibility, capability and
    pricing verification separate (as required by issue #9)."""
    rows = []
    for m in catalogue.models():
        d = discovery.get(m.provider)
        if d is None:
            listed = "not refreshed"
        elif d.status != "ok":
            listed = f"unknown — last refresh failed ({d.status}) {d.fetched_at:%Y-%m-%d}"
        elif m.model_id in d.listed_ids:
            listed = f"listed for this API key ({d.fetched_at:%Y-%m-%d})"
        else:
            listed = f"NOT listed at the last refresh ({d.fetched_at:%Y-%m-%d})"
        if not m.verified:
            adapter = "unknown (capabilities not verified)"
        elif m.interface == "unsupported":
            adapter = "incompatible with this software"
        else:
            adapter = f"{m.interface} (documented; verified entry)"
        problems = m.execution_problems()
        rows.append({
            "provider": m.provider,
            "model_id": m.model_id,
            "display_name": m.display_name,
            "account_availability": listed,
            "adapter_compatibility": adapter,
            "capabilities": (
                f"verified {m.verified_at or '(date not recorded)'} — {m.source}"
                if m.verified else "UNVERIFIED"
            ),
            "efforts": list(m.efforts),
            "default_effort": m.provider_default_effort,
            "max_output_tokens": m.max_output_tokens,
            "pricing": (
                "unavailable (no verified list price; costs shown as unavailable)"
                if m.pricing is None
                else f"verified {m.pricing_verified_at or '(date not recorded)'} — "
                f"{m.pricing_source or m.source}"
            ),
            "lifecycle": m.lifecycle or "not recorded",
            "retires_at": m.retires_at.isoformat() if m.retires_at else None,
            "executable": not problems,
            "problems": problems,
            "warnings": m.warnings(),
            "provider_reported": m.provider_reported or {},
        })  # fmt: skip
    return rows
