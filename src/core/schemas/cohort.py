"""Overlapping-cohort warnings (prevent double-counting of participants)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from src.core.enums import OverlapResolution, OverlapSeverity
from src.core.schemas.base import AwareDatetime, Identifier, StrictModel, new_id, utc_now


class OverlapSignal(StrictModel):
    kind: Literal[
        "shared_registry_id",
        "affiliation_similarity",
        "coauthor_overlap",
        "recruitment_overlap",
        "publication_proximity",
    ]
    score: float = Field(ge=0.0, le=1.0)
    detail: str


class CohortOverlapWarning(StrictModel):
    """Heuristic flag that two reports may describe the same participants.

    The detector never excludes anything; a human must resolve every warning.
    The pair is stored in canonical order (``record_id_a < record_id_b``).
    """

    warning_id: Identifier = Field(default_factory=new_id)
    record_id_a: Identifier
    record_id_b: Identifier
    overlap_score: float = Field(ge=0.0, le=1.0)
    severity: OverlapSeverity
    signals: list[OverlapSignal] = Field(min_length=1)
    shared_authors: list[str] = Field(default_factory=list)
    shared_institutions: list[str] = Field(default_factory=list)
    shared_registry_ids: list[str] = Field(default_factory=list)
    recruitment_overlap_years: int | None = Field(default=None, ge=0)
    created_at: AwareDatetime = Field(default_factory=utc_now)
    resolution: OverlapResolution = OverlapResolution.UNRESOLVED
    resolved_by: str | None = None
    resolution_note: str | None = None

    @model_validator(mode="after")
    def _rules(self) -> CohortOverlapWarning:
        if self.record_id_a == self.record_id_b:
            raise ValueError("a record cannot overlap with itself")
        if self.record_id_a > self.record_id_b:
            raise ValueError("record ids must be in canonical order (a < b)")
        if self.resolution != OverlapResolution.UNRESOLVED and not self.resolved_by:
            raise ValueError("resolved warnings require resolved_by")
        return self
