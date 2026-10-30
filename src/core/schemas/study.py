"""Stage 2 contracts: bibliographic records retrieved from literature databases."""

from __future__ import annotations

from pydantic import Field, model_validator

from src.core.enums import SourceDatabase
from src.core.identifiers import normalize_person_name, normalize_title
from src.core.schemas.base import (
    DOI,
    PMCID,
    PMID,
    AwareDatetime,
    FrozenModel,
    Identifier,
    NonEmptyStr,
    StrictModel,
    new_id,
    utc_now,
)


class Author(StrictModel):
    """An author as reported by the source API (never inferred or completed by an LLM)."""

    last_name: str | None = None
    fore_name: str | None = None
    initials: str | None = None
    collective_name: str | None = None
    full_name: str | None = None
    orcid: str | None = Field(default=None, pattern=r"^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$")
    affiliations: list[NonEmptyStr] = Field(default_factory=list)
    name_parsed_heuristically: bool = False

    @model_validator(mode="after")
    def _has_some_name(self) -> Author:
        if not (self.last_name or self.collective_name or self.full_name):
            raise ValueError("author requires last_name, collective_name or full_name")
        return self

    @property
    def match_key(self) -> str:
        """Normalised ``"surname i"`` key used for co-author network comparison."""
        if self.last_name:
            return normalize_person_name(self.last_name, self.fore_name or self.initials)
        if self.collective_name:
            return normalize_title(self.collective_name)
        return normalize_title(self.full_name)


class RecruitmentPeriod(StrictModel):
    """Participant recruitment window; populated from full text in Stage 4 or by a human."""

    start_year: int = Field(ge=1900, le=2100)
    end_year: int = Field(ge=1900, le=2100)

    @model_validator(mode="after")
    def _order(self) -> RecruitmentPeriod:
        if self.start_year > self.end_year:
            raise ValueError("start_year must not be after end_year")
        return self


class SearchProvenance(FrozenModel):
    """Exactly how a record was obtained, for PRISMA-S reproducibility."""

    database: SourceDatabase
    query: NonEmptyStr
    endpoint: NonEmptyStr
    executed_at: AwareDatetime
    total_hits_reported: int = Field(ge=0)
    source_record_id: NonEmptyStr


class StudyRecord(StrictModel):
    """One bibliographic record (a *report* in PRISMA 2020 terminology)."""

    record_id: Identifier = Field(default_factory=new_id)
    source_database: SourceDatabase
    provenance: SearchProvenance | None = None
    pmid: PMID = None
    pmcid: PMCID = None
    doi: DOI = None
    openalex_id: str | None = Field(default=None, pattern=r"^W\d+$")
    title: NonEmptyStr
    abstract: str | None = None
    authors: list[Author] = Field(default_factory=list)
    journal: str | None = None
    publication_year: int | None = Field(default=None, ge=1800, le=2100)
    publication_types: list[str] = Field(default_factory=list)
    mesh_headings: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    language: str | None = None
    registry_ids: list[str] = Field(default_factory=list)
    recruitment_period: RecruitmentPeriod | None = None
    duplicate_of: Identifier | None = None
    retrieved_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _not_self_duplicate(self) -> StudyRecord:
        if self.duplicate_of is not None and self.duplicate_of == self.record_id:
            raise ValueError("a record cannot be a duplicate of itself")
        return self

    @property
    def normalized_title(self) -> str:
        return normalize_title(self.title)

    @property
    def first_author_key(self) -> str:
        return self.authors[0].match_key if self.authors else ""

    @property
    def all_affiliations(self) -> list[str]:
        return [aff for author in self.authors for aff in author.affiliations]

    def completeness_score(self) -> int:
        """Deterministic metadata richness score used to pick cluster representatives."""
        return sum(
            (
                4 * bool(self.pmid),
                3 * bool(self.doi),
                2 * bool(self.abstract),
                1 * bool(self.pmcid),
                1 * bool(self.authors),
                1 * bool(self.journal),
                1 * bool(self.publication_year),
                1 * bool(self.mesh_headings),
            )
        )
