"""Shared fixtures. All bibliographic data in the test suite is SYNTHETIC.

DOIs use the 10.5555 test prefix; titles, authors and PMIDs are invented and
must never be read as real citations.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy.orm import Session

from src.apis.rate_limit import AsyncRateLimiter
from src.core.enums import (
    ActorType,
    ArmRole,
    EvidenceLocationType,
    SourceDatabase,
    StudyDesign,
)
from src.core.schemas import (
    Author,
    ExtractedStudyArm,
    ExtractedValue,
    MeSHTerm,
    OutcomeDefinition,
    PICOProtocol,
    RecruitmentPeriod,
    SearchStrategy,
    SourceEvidence,
    StudyRecord,
)
from src.state.db import init_db, make_engine, make_session_factory
from src.state.machine import GuardContext

NOW = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)

# A guard context in which every precondition is satisfied (every field must be explicit).
READY = GuardContext(
    protocol_blockers=(),
    records_identified=100,
    records_unique=80,
    unresolved_duplicate_candidates=0,
    ta_missing_decisions=0,
    ta_unresolved_conflicts=0,
    ta_included=12,
    unresolved_cohort_warnings=0,
    prisma_recorded=True,
    prisma_violations=(),
    prisma_awaiting_screening=0,
    prisma_pending_reports=0,
    fulltext_items_missing=0,
    fulltext_items_blocking=0,
    ft_missing_decisions=0,
    ft_unresolved_conflicts=0,
    ft_included=10,
    included_reports_without_extraction=0,
    arms_for_non_included_reports=0,
    extracted_values_without_evidence=0,
    extraction_arms=20,
    unapproved_extraction_arms=0,
    stats_output_present=True,
    manuscript_present=True,
    audit_passed=True,
)

# What the repository reports for a freshly created project: nothing recorded yet.
EMPTY = GuardContext(
    protocol_blockers=("protocol not drafted",),
    records_identified=0,
    records_unique=0,
    unresolved_duplicate_candidates=0,
    ta_missing_decisions=0,
    ta_unresolved_conflicts=0,
    ta_included=0,
    unresolved_cohort_warnings=0,
    prisma_recorded=False,
    prisma_violations=(),
    prisma_awaiting_screening=0,
    prisma_pending_reports=0,
    fulltext_items_missing=0,
    fulltext_items_blocking=0,
    ft_missing_decisions=0,
    ft_unresolved_conflicts=0,
    ft_included=0,
    included_reports_without_extraction=0,
    arms_for_non_included_reports=0,
    extracted_values_without_evidence=0,
    extraction_arms=0,
    unapproved_extraction_arms=0,
    stats_output_present=False,
    manuscript_present=False,
    audit_passed=None,
)


class FakeClock:
    """Deterministic monotonic clock; ``sleep`` advances time instantly."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def no_wait_limiter() -> AsyncRateLimiter:
    clock = FakeClock()
    return AsyncRateLimiter(1000, 1.0, clock=clock, sleep=clock.sleep)


@pytest.fixture
def no_sleep() -> Callable[[float], Awaitable[None]]:
    async def _sleep(_: float) -> None:
        return None

    return _sleep


@pytest.fixture
def session() -> Iterator[Session]:
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    factory = make_session_factory(engine)
    with factory() as s:
        yield s
    engine.dispose()


def make_author(last: str, fore: str, *affiliations: str) -> Author:
    return Author(last_name=last, fore_name=fore, affiliations=list(affiliations))


def make_record(record_id: str, title: str, **overrides: Any) -> StudyRecord:
    data: dict[str, Any] = {
        "record_id": record_id,
        "source_database": SourceDatabase.PUBMED,
        "title": title,
        "publication_year": 2020,
        "authors": [make_author("Smith", "Jane")],
    }
    data.update(overrides)
    return StudyRecord(**data)


def make_evidence(quote: str, **overrides: Any) -> SourceEvidence:
    data: dict[str, Any] = {
        "pdf_filename": "synthetic_trial.pdf",
        "pdf_sha256": "a" * 64,
        "page_number": 4,
        "location_type": EvidenceLocationType.TABLE,
        "location_identifier": "Table 2",
        "verbatim_quote": quote,
        "extracted_by": ActorType.AGENT,
        "extractor_identity": "extractor-agent-test",
        "extracted_at": NOW,
    }
    data.update(overrides)
    return SourceEvidence(**data)


def make_value(text: str, value: float, quote: str | None = None) -> ExtractedValue:
    return ExtractedValue(
        value=value, reported_text=text, evidence=make_evidence(quote or f"value {text}")
    )


def make_arm(record_id: str = "rec1", **overrides: Any) -> ExtractedStudyArm:
    quote = "Intervention (n=120): mean 12·5 (SD 3·1); 118 analysed"
    ev = make_evidence(quote)
    data: dict[str, Any] = {
        "record_id": record_id,
        "outcome_id": "hba1c",
        "arm_label": "Intervention",
        "arm_role": ArmRole.INTERVENTION,
        "n_randomized": ExtractedValue(value=120, reported_text="120", evidence=ev),
        "n_analyzed": ExtractedValue(value=118, reported_text="118", evidence=ev),
        "mean": ExtractedValue(value=12.5, reported_text="12·5", evidence=ev),
        "sd": ExtractedValue(value=3.1, reported_text="3·1", evidence=ev),
    }
    data.update(overrides)
    return ExtractedStudyArm(**data)


def make_protocol(**overrides: Any) -> PICOProtocol:
    data: dict[str, Any] = {
        "review_title": "Synthetic intervention X versus placebo for condition Y",
        "research_question": "Does X reduce outcome Z in adults with Y?",
        "population": "Adults with condition Y",
        "intervention": "Intervention X",
        "comparator": "Placebo",
        "outcomes": [
            OutcomeDefinition(
                outcome_id="hba1c",
                name="HbA1c change",
                kind="primary",
                data_type="continuous",
                preferred_effect_measure="MD",
            )
        ],
        "study_designs": [StudyDesign.RCT],
        "inclusion_criteria": ["Randomised controlled trials in adults"],
        "exclusion_criteria": ["Animal studies"],
        "mesh_terms": [MeSHTerm(descriptor="Diabetes Mellitus", verified=True, verified_at=NOW)],
        "search_strategies": [
            SearchStrategy(
                database=SourceDatabase.PUBMED,
                query='("diabetes mellitus"[MeSH Terms]) AND (randomized controlled trial[pt])',
                syntax_validated=True,
            )
        ],
    }
    data.update(overrides)
    return PICOProtocol(**data)


__all__ = [
    "EMPTY",
    "NOW",
    "READY",
    "FakeClock",
    "RecruitmentPeriod",
    "make_arm",
    "make_author",
    "make_evidence",
    "make_protocol",
    "make_record",
    "make_value",
]
