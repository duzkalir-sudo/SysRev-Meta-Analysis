"""Deduplication, dual-review screening support, PRISMA tracking and cohort-overlap detection."""

from src.screening.agreement import (
    agreement_for_stage,
    cohens_kappa,
    detect_conflicts,
    final_verdicts,
)
from src.screening.cohort_overlap import CohortOverlapConfig, CohortOverlapDetector
from src.screening.deduplication import DeduplicationResult, Deduplicator, PossibleDuplicate
from src.screening.prisma import PRISMATracker

__all__ = [
    "CohortOverlapConfig",
    "CohortOverlapDetector",
    "DeduplicationResult",
    "Deduplicator",
    "PRISMATracker",
    "PossibleDuplicate",
    "agreement_for_stage",
    "cohens_kappa",
    "detect_conflicts",
    "final_verdicts",
]
