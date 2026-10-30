"""Shared Pydantic base classes and annotated field types."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from src.core.identifiers import normalize_doi, normalize_pmcid, normalize_pmid


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_id() -> str:
    return uuid.uuid4().hex


def _require_aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware (UTC recommended)")
    return value


def _doi(value: str | None) -> str | None:
    return normalize_doi(value)


def _pmid(value: str | None) -> str | None:
    return normalize_pmid(value)


def _pmcid(value: str | None) -> str | None:
    return normalize_pmcid(value)


AwareDatetime = Annotated[datetime, AfterValidator(_require_aware)]
DOI = Annotated[str | None, AfterValidator(_doi)]
PMID = Annotated[str | None, AfterValidator(_pmid)]
PMCID = Annotated[str | None, AfterValidator(_pmcid)]
NonEmptyStr = Annotated[str, Field(min_length=1)]
Identifier = Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$")]


class StrictModel(BaseModel):
    """Base for all data contracts: unknown fields are rejected, assignments re-validated."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
        validate_default=True,
    )


class FrozenModel(BaseModel):
    """Immutable contract (provenance records must never be edited in place)."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        frozen=True,
        validate_default=True,
    )
