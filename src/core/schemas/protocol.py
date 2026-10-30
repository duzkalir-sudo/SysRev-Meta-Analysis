"""Stage 1 contracts: PICO protocol, MeSH terms and database search strategies."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Literal

from pydantic import Field, field_validator, model_validator

from src.core.enums import (
    ActorType,
    ClinicalOutcomeType,
    ComparisonType,
    EffectMeasure,
    ReviewDesign,
    SourceDatabase,
    StudyDesign,
    TimepointType,
)
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


# Fields that define what a strategy retrieves; any change makes it a different search.
STRATEGY_EXECUTABLE_FIELDS = (
    "database", "platform", "query", "date_from", "date_to", "languages", "filters",
)  # fmt: skip


class SearchStrategy(StrictModel):
    """An executable, database-specific search string (PRISMA-S item 8).

    ``syntax_validated`` is set only by the system: after a successful API preview of this
    exact strategy, or after an authenticated human attests that it ran in the database's
    own interface (subscription databases). The ``validation_*`` fields say which.
    """

    database: SourceDatabase
    platform: str | None = Field(
        default=None, description='interface/vendor, e.g. "Embase.com", "Ovid", "PubMed"'
    )
    query: NonEmptyStr
    date_from: date | None = None
    date_to: date | None = None
    languages: list[str] = Field(default_factory=list)
    filters: list[str] = Field(default_factory=list)
    generated_by: ActorType = ActorType.AGENT
    syntax_validated: bool = False
    validation_method: Literal["api_preview", "api_accepted", "human_attestation"] | None = None
    validated_by: str | None = None
    validated_at: AwareDatetime | None = None
    validation_note: str | None = None

    @model_validator(mode="after")
    def _date_order(self) -> SearchStrategy:
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from must not be after date_to")
        if self.validation_method is not None and not self.syntax_validated:
            raise ValueError("validation details given for a strategy that is not validated")
        return self

    @property
    def sha256(self) -> str:
        """Content hash of the executable definition (stable across validation status)."""
        data = self.model_dump(mode="json", include=set(STRATEGY_EXECUTABLE_FIELDS))
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @field_validator("query")
    @classmethod
    def _balanced_syntax(cls, value: str) -> str:
        if value.count("(") != value.count(")"):
            raise ValueError("unbalanced parentheses in search query")
        if value.count('"') % 2:
            raise ValueError("unbalanced double quotes in search query")
        return value


class FollowUpTimepoint(StrictModel):
    """When an outcome is assessed. Purely descriptive; no time arithmetic is performed."""

    kind: TimepointType
    value: float | None = Field(default=None, gt=0)
    window_min: float | None = Field(default=None, ge=0)
    window_max: float | None = Field(default=None, gt=0)
    unit: Literal["days", "weeks", "months", "years"] | None = None

    @model_validator(mode="after")
    def _shape(self) -> FollowUpTimepoint:
        if self.kind == TimepointType.FIXED:
            if self.value is None or self.unit is None:
                raise ValueError("a fixed timepoint needs value and unit (e.g. 5 years)")
            if self.window_min is not None or self.window_max is not None:
                raise ValueError("a fixed timepoint cannot have a window")
        elif self.kind == TimepointType.WINDOW:
            if self.window_min is None or self.window_max is None or self.unit is None:
                raise ValueError("a window timepoint needs window_min, window_max and unit")
            if self.window_min > self.window_max:
                raise ValueError("window_min must not exceed window_max")
            if self.value is not None:
                raise ValueError("a window timepoint cannot also have a fixed value")
        elif any(v is not None for v in (self.value, self.window_min, self.window_max)):
            raise ValueError("'last follow-up' timepoints carry no numeric time")
        return self

    @property
    def label(self) -> str:
        if self.kind == TimepointType.FIXED:
            return f"{self.value:g} {self.unit}"
        if self.kind == TimepointType.WINDOW:
            return f"{self.window_min:g}-{self.window_max:g} {self.unit}"
        return "last follow-up"


# Which summary measures are valid for each (comparison, data type) combination.
ALLOWED_MEASURES: dict[tuple[ComparisonType, str], frozenset[EffectMeasure]] = {
    (ComparisonType.COMPARATIVE, "dichotomous"): frozenset(
        {EffectMeasure.RR, EffectMeasure.OR, EffectMeasure.RD}
    ),
    (ComparisonType.COMPARATIVE, "continuous"): frozenset({EffectMeasure.MD, EffectMeasure.SMD}),
    (ComparisonType.COMPARATIVE, "time_to_event"): frozenset({EffectMeasure.HR}),
    (ComparisonType.COMPARATIVE, "count_rate"): frozenset({EffectMeasure.IRR}),
    (ComparisonType.SINGLE_GROUP, "dichotomous"): frozenset({EffectMeasure.PROPORTION}),
    (ComparisonType.SINGLE_GROUP, "continuous"): frozenset(
        {EffectMeasure.MEAN, EffectMeasure.MEDIAN}
    ),
    (ComparisonType.SINGLE_GROUP, "time_to_event"): frozenset(
        {
            EffectMeasure.KM_PROBABILITY,
            EffectMeasure.CUMULATIVE_INCIDENCE,
            EffectMeasure.MEDIAN_TIME_TO_EVENT,
        }
    ),
    (ComparisonType.SINGLE_GROUP, "count_rate"): frozenset({EffectMeasure.INCIDENCE_RATE}),
}
# Estimates defined *at* a time t (survival-type probabilities) need fixed timepoints.
TIME_POINT_ESTIMATES = frozenset({EffectMeasure.KM_PROBABILITY, EffectMeasure.CUMULATIVE_INCIDENCE})


class OutcomeDefinition(StrictModel):
    """A protocol outcome.

    ``data_type`` + ``comparison`` determine the admissible summary measures. A crude
    proportion (``dichotomous`` / ``PROPORTION``, events ÷ n, censoring ignored) is a
    different quantity from a Kaplan-Meier probability (``time_to_event`` /
    ``KM_PROBABILITY``) or a competing-risks cumulative incidence
    (``CUMULATIVE_INCIDENCE``); the schema keeps them apart so they cannot be pooled
    together by accident.
    """

    outcome_id: Identifier
    name: NonEmptyStr
    kind: Literal["primary", "secondary"]
    clinical_type: ClinicalOutcomeType = ClinicalOutcomeType.OTHER
    definition: str | None = Field(
        default=None, description='operational definition, e.g. "no growth >2 mm on MRI"'
    )
    grading_system: str | None = Field(default=None, description='e.g. "CTCAE v5.0", "BNI"')
    data_type: Literal["dichotomous", "continuous", "time_to_event", "count_rate"]
    comparison: ComparisonType = ComparisonType.COMPARATIVE
    preferred_effect_measure: EffectMeasure
    timepoints: list[FollowUpTimepoint] = Field(default_factory=list)

    @model_validator(mode="after")
    def _measure_matches_type(self) -> OutcomeDefinition:
        allowed = ALLOWED_MEASURES[(self.comparison, self.data_type)]
        if self.preferred_effect_measure not in allowed:
            raise ValueError(
                f"effect measure {self.preferred_effect_measure} is invalid for "
                f"{self.comparison} {self.data_type} outcomes "
                f"(allowed: {sorted(m.value for m in allowed)})"
            )
        needs_time = (
            self.comparison == ComparisonType.SINGLE_GROUP
            and self.preferred_effect_measure != EffectMeasure.MEDIAN_TIME_TO_EVENT
        )
        if needs_time and not self.timepoints:
            raise ValueError("single-group outcomes must specify at least one follow-up timepoint")
        if self.preferred_effect_measure in TIME_POINT_ESTIMATES and any(
            t.kind != TimepointType.FIXED for t in self.timepoints
        ):
            raise ValueError(
                f"{self.preferred_effect_measure} is an estimate at a time t: every timepoint "
                "must be fixed (e.g. 5 years), not a window or last follow-up"
            )
        return self

    @property
    def is_safety_outcome(self) -> bool:
        return self.clinical_type == ClinicalOutcomeType.ADVERSE_EVENT


# Case-control sampling fixes the outcome distribution by design, so proportions and
# incidence cannot be estimated from it; it is therefore excluded from single-arm reviews.
SINGLE_ARM_EXCLUDED_DESIGNS = frozenset({StudyDesign.CASE_CONTROL})
# Designs without a concurrent comparison group cannot inform a comparative review.
NON_COMPARATIVE_DESIGNS = frozenset({StudyDesign.SINGLE_ARM_TRIAL, StudyDesign.CASE_SERIES})


class PICOProtocol(StrictModel):
    """Review protocol (PRISMA 2020 items 5-7, 24; PROSPERO registration fields)."""

    protocol_id: Identifier = Field(default_factory=new_id)
    version: int = Field(default=1, ge=1)
    review_title: NonEmptyStr
    research_question: NonEmptyStr
    population: NonEmptyStr
    intervention: NonEmptyStr
    review_design: ReviewDesign = ReviewDesign.COMPARATIVE
    comparator: str | None = Field(
        default=None, description="required for comparative reviews; absent for single-arm"
    )
    outcomes: list[OutcomeDefinition] = Field(min_length=1)
    study_designs: list[StudyDesign] = Field(min_length=1)
    inclusion_criteria: list[NonEmptyStr] = Field(min_length=1)
    exclusion_criteria: list[NonEmptyStr] = Field(min_length=1)
    mesh_terms: list[MeSHTerm] = Field(default_factory=list)
    selected_databases: list[SourceDatabase] = Field(
        default_factory=list, description="databases chosen at project setup (Session 5)"
    )
    search_strategies: list[SearchStrategy] = Field(default_factory=list)
    prospero_id: str | None = Field(default=None, pattern=r"^CRD\d{11}$")
    status: Literal["draft", "approved", "superseded"] = "draft"
    approved_by: str | None = None
    approved_at: AwareDatetime | None = None
    approved_reviewer_id: Identifier | None = Field(
        default=None, description="enrolled reviewer who approved (authenticated approvals)"
    )

    @model_validator(mode="after")
    def _consistency(self) -> PICOProtocol:
        ids = [o.outcome_id for o in self.outcomes]
        if len(ids) != len(set(ids)):
            raise ValueError("outcome_id values must be unique")
        if not any(o.kind == "primary" for o in self.outcomes):
            raise ValueError("at least one primary outcome is required")
        self._design_consistency()
        if self.status == "approved":
            if not (self.approved_by and self.approved_at):
                raise ValueError("approved protocols require approved_by and approved_at")
            problems = self.approval_blockers()
            if problems:
                raise ValueError("protocol cannot be approved: " + "; ".join(problems))
        return self

    def _design_consistency(self) -> None:
        designs = set(self.study_designs)
        if self.review_design == ReviewDesign.COMPARATIVE:
            if not (self.comparator and self.comparator.strip()):
                raise ValueError("comparative reviews require a comparator")
            bad = sorted(d.value for d in designs & NON_COMPARATIVE_DESIGNS)
            if bad:
                raise ValueError(
                    f"designs without a comparison group in a comparative review: {bad}"
                )
        else:
            if self.comparator is not None:
                raise ValueError("single-arm reviews must not define a comparator")
            comparative = [o.outcome_id for o in self.outcomes
                           if o.comparison == ComparisonType.COMPARATIVE]  # fmt: skip
            if comparative:
                raise ValueError(
                    f"single-arm reviews cannot have comparative outcomes: {comparative}"
                )
            bad = sorted(d.value for d in designs & SINGLE_ARM_EXCLUDED_DESIGNS)
            if bad:
                raise ValueError(f"designs that cannot estimate single-group outcomes: {bad}")

    def approval_blockers(self) -> list[str]:
        """Reasons the protocol is not yet ready for HITL Gate 1 approval."""
        problems: list[str] = []
        if not self.search_strategies:
            problems.append("no search strategies defined")
        if any(not s.syntax_validated for s in self.search_strategies):
            problems.append("one or more search strategies are not syntax-validated")
        if any(not m.verified for m in self.mesh_terms):
            problems.append("one or more MeSH terms are not verified against NLM MeSH")
        hashes = [s.sha256 for s in self.search_strategies]
        if len(hashes) != len(set(hashes)):
            problems.append("two search strategies are identical")
        if self.selected_databases:
            chosen = set(self.selected_databases)
            with_strategy = {s.database for s in self.search_strategies}
            for db in sorted(chosen - with_strategy):
                problems.append(f"selected database {db} has no search strategy")
            for db in sorted(with_strategy - chosen):
                problems.append(f"search strategy for {db}, which is not a selected database")
        return problems

    def strategy(self, sha256: str) -> SearchStrategy | None:
        return next((s for s in self.search_strategies if s.sha256 == sha256), None)
