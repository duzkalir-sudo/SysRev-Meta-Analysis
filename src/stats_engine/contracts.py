"""Interfaces for the Stage 5 engine. Not yet implemented: calls raise ``NotImplementedError``."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol

from src.core.schemas.extraction import ExtractedStudyArm

PoolingMethod = Literal["fixed_iv", "mantel_haenszel", "dersimonian_laird", "reml"]


class MetaAnalysisEngine(Protocol):
    def pool(
        self,
        arms: Sequence[ExtractedStudyArm],
        *,
        outcome_id: str,
        method: PoolingMethod,
        hartung_knapp: bool = False,
    ) -> dict[str, float]:
        """Return pooled estimate, CI, tau², I², Q and p-values for one outcome."""
        ...


class NotYetImplementedEngine:
    def pool(
        self,
        arms: Sequence[ExtractedStudyArm],
        *,
        outcome_id: str,
        method: PoolingMethod,
        hartung_knapp: bool = False,
    ) -> dict[str, float]:
        raise NotImplementedError("Stage 5 statistics engine is scheduled for a later session")
