"""Stage 1 contracts: PICO protocol, MeSH terms and database search strategies."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import Field, field_validator, model_validator

from src.core.enums import ActorType, SourceDatabase, StudyDesign
from src.core.schemas.base import AwareDatetime, Identifier, NonEmptyStr, StrictModel, new_id


class MeSHTerm(StrictModel):
    """A Medical Subject Heading descriptor.

    ``verified`` may only be set after the descriptor was confirmed to exist in
    the NLM MeSH database through a deterministic lookup (``PubMedClient.mesh_descriptor_exists``);
    LLM-proposed headings start unverified.
    """

    descriptor: NonEmptyStr
    descriptor_ui: str | None = Field(default=None, pattern=r"^D\d{6,9}$")
    explode: bool = True
    major_topic_only: bool = False
    verified: bool = False
    verified_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def _verification_timestamp(self) -> MeSHTerm:
        if self.verified and self.verified_at is None:
            raise ValueError("verified MeSH terms require verified_at")
        return self


class SearchStrategy(StrictModel):
    """An executable, database-specific search string (PRISMA-S item 8)."""

    database: SourceDatabase
    query: NonEmptyStr
    date_from: date | None = None
    date_to: date | None = None
    languages: list[str] = Field(default_factory=list)
    filters: list[str] = Field(default_factory=list)
    generated_by: ActorType = ActorType.AGENT
    syntax_validated: bool = False

    @model_validator(mode="after")
    def _date_order(self) -> SearchStrategy:
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from must not be after date_to")
        return self

    @field_validator("query")
    @classmethod
    def _balanced_syntax(cls, value: str) -> str:
        if value.count("(") != value.count(")"):
            raise ValueError("unbalanced parentheses in search query")
        if value.count('"') % 2:
            raise ValueError("unbalanced double quotes in search query")
        return value


class OutcomeDefinition(StrictModel):
    outcome_id: Identifier
    name: NonEmptyStr
    kind: Literal["primary", "secondary"]
    data_type: Literal["dichotomous", "continuous", "time_to_event"]
    preferred_effect_measure: Literal["RR", "OR", "RD", "MD", "SMD", "HR"]
    timepoint: str | None = None

    @model_validator(mode="after")
    def _measure_matches_type(self) -> OutcomeDefinition:
        allowed = {
            "dichotomous": {"RR", "OR", "RD"},
            "continuous": {"MD", "SMD"},
            "time_to_event": {"HR"},
        }[self.data_type]
        if self.preferred_effect_measure not in allowed:
            raise ValueError(
                f"effect measure {self.preferred_effect_measure} is invalid for "
                f"{self.data_type} outcomes (allowed: {sorted(allowed)})"
            )
        return self


class PICOProtocol(StrictModel):
    """Review protocol (PRISMA 2020 items 5-7, 24; PROSPERO registration fields)."""

    protocol_id: Identifier = Field(default_factory=new_id)
    version: int = Field(default=1, ge=1)
    review_title: NonEmptyStr
    research_question: NonEmptyStr
    population: NonEmptyStr
    intervention: NonEmptyStr
    comparator: NonEmptyStr
    outcomes: list[OutcomeDefinition] = Field(min_length=1)
    study_designs: list[StudyDesign] = Field(min_length=1)
    inclusion_criteria: list[NonEmptyStr] = Field(min_length=1)
    exclusion_criteria: list[NonEmptyStr] = Field(min_length=1)
    mesh_terms: list[MeSHTerm] = Field(default_factory=list)
    search_strategies: list[SearchStrategy] = Field(default_factory=list)
    prospero_id: str | None = Field(default=None, pattern=r"^CRD\d{11}$")
    status: Literal["draft", "approved", "superseded"] = "draft"
    approved_by: str | None = None
    approved_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def _consistency(self) -> PICOProtocol:
        ids = [o.outcome_id for o in self.outcomes]
        if len(ids) != len(set(ids)):
            raise ValueError("outcome_id values must be unique")
        if not any(o.kind == "primary" for o in self.outcomes):
            raise ValueError("at least one primary outcome is required")
        if self.status == "approved":
            if not (self.approved_by and self.approved_at):
                raise ValueError("approved protocols require approved_by and approved_at")
            problems = self.approval_blockers()
            if problems:
                raise ValueError("protocol cannot be approved: " + "; ".join(problems))
        return self

    def approval_blockers(self) -> list[str]:
        """Reasons the protocol is not yet ready for HITL Gate 1 approval."""
        problems: list[str] = []
        if not self.search_strategies:
            problems.append("no search strategies defined")
        if any(not s.syntax_validated for s in self.search_strategies):
            problems.append("one or more search strategies are not syntax-validated")
        if any(not m.verified for m in self.mesh_terms):
            problems.append("one or more MeSH terms are not verified against NLM MeSH")
        return problems
