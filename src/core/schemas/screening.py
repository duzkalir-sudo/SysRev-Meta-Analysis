"""Stage 2 contracts: dual independent screening decisions and conflicts."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from src.core.enums import ReviewerRole, ScreeningStage, ScreeningVerdict
from src.core.schemas.base import (
    AwareDatetime,
    FrozenModel,
    Identifier,
    NonEmptyStr,
    StrictModel,
    new_id,
    utc_now,
)


class AutomatedReviewerProvenance(FrozenModel):
    """How an automated (LLM) screening decision was produced. Never describes a human."""

    provider: NonEmptyStr
    model_id: NonEmptyStr
    reviewer_label: NonEmptyStr = Field(description='e.g. "llm-pass-1"')
    prompt_version: NonEmptyStr
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    temperature: float | None = None
    raw_verdict: str | None = Field(
        default=None, description="verdict proposed by the model before the conservative policy"
    )
    policy_notes: list[str] = Field(default_factory=list)
    run_id: Identifier = Field(default_factory=new_id)


class ScreeningDecision(StrictModel):
    """A single reviewer's eligibility judgement for one record at one stage."""

    decision_id: Identifier = Field(default_factory=new_id)
    record_id: Identifier
    reviewer: ReviewerRole
    reviewer_identity: str = Field(min_length=1, description="model id or human name")
    is_human: bool = False
    stage: ScreeningStage
    verdict: ScreeningVerdict
    exclusion_reason: str | None = None
    rationale: str = Field(min_length=1)
    criteria_cited: list[str] = Field(default_factory=list)
    decided_at: AwareDatetime = Field(default_factory=utc_now)
    reviewer_id: Identifier | None = Field(
        default=None, description="enrolled reviewer id; required to persist a human decision"
    )
    automation: AutomatedReviewerProvenance | None = None

    @model_validator(mode="after")
    def _rules(self) -> ScreeningDecision:
        if self.verdict == ScreeningVerdict.EXCLUDE and not self.exclusion_reason:
            raise ValueError("exclusion decisions require an exclusion_reason")
        if self.verdict != ScreeningVerdict.EXCLUDE and self.exclusion_reason:
            raise ValueError("exclusion_reason is only allowed for exclude verdicts")
        if self.reviewer == ReviewerRole.ADJUDICATOR and not self.is_human:
            raise ValueError("adjudication decisions must be made by a human")
        if self.is_human and self.automation is not None:
            raise ValueError("a human decision cannot carry automated-reviewer provenance")
        if not self.is_human and self.reviewer_id is not None:
            raise ValueError("reviewer_id identifies enrolled humans only")
        # Humans must commit at full text; an automated pass may defer to a human instead
        # of guessing (conservative policy), which surfaces the record as a conflict.
        if (
            self.stage == ScreeningStage.FULL_TEXT
            and self.verdict == ScreeningVerdict.UNCERTAIN
            and self.is_human
        ):
            raise ValueError("human full-text decisions must be include or exclude")
        return self


class ScreeningConflict(StrictModel):
    """Disagreement between Reviewer 1 and Reviewer 2 awaiting human resolution (Gate 1.5)."""

    conflict_id: Identifier = Field(default_factory=new_id)
    record_id: Identifier
    stage: ScreeningStage
    reviewer_1_verdict: ScreeningVerdict
    reviewer_2_verdict: ScreeningVerdict
    resolved_verdict: ScreeningVerdict | None = None
    resolved_by: str | None = None
    resolved_at: AwareDatetime | None = None
    resolution_note: str | None = None

    @model_validator(mode="after")
    def _rules(self) -> ScreeningConflict:
        if self.reviewer_1_verdict == self.reviewer_2_verdict and (
            self.reviewer_1_verdict != ScreeningVerdict.UNCERTAIN
        ):
            raise ValueError("identical definitive verdicts are not a conflict")
        resolved_fields = (self.resolved_verdict, self.resolved_by, self.resolved_at)
        if any(f is not None for f in resolved_fields) and not all(
            f is not None for f in resolved_fields
        ):
            raise ValueError("resolved_verdict, resolved_by and resolved_at must be set together")
        if self.resolved_verdict == ScreeningVerdict.UNCERTAIN:
            raise ValueError("a resolution must be include or exclude")
        return self

    @property
    def is_resolved(self) -> bool:
        return self.resolved_verdict is not None


class AgreementStatistics(StrictModel):
    """Inter-rater agreement computed deterministically (Cohen, 1960)."""

    n_items: int = Field(ge=0)
    observed_agreement: float = Field(ge=0.0, le=1.0)
    expected_agreement: float = Field(ge=0.0, le=1.0)
    kappa: float | None = Field(
        default=None, ge=-1.0, le=1.0, description="None when undefined (p_e == 1)"
    )


class DuplicateCandidate(StrictModel):
    """A pair the deduplicator matched but refused to merge because evidence conflicts.

    Examples: a shared DOI with different PMIDs, or near-identical titles with different
    DOIs. A human must adjudicate every candidate before Gate 1.5 can be released.
    The pair is stored in canonical order (``record_id_a < record_id_b``).
    """

    candidate_id: Identifier = Field(default_factory=new_id)
    record_id_a: Identifier
    record_id_b: Identifier
    match_basis: str = Field(min_length=1, description='e.g. "exact doi (10.x/y)", "fuzzy title"')
    conflict: str = Field(min_length=1)
    title_similarity: float | None = Field(default=None, ge=0.0, le=100.0)
    resolution: Literal["unresolved", "distinct", "duplicate_keep_a", "duplicate_keep_b"] = (
        "unresolved"
    )
    resolved_by: str | None = None
    resolved_at: AwareDatetime | None = None
    resolution_note: str | None = None

    @model_validator(mode="after")
    def _rules(self) -> DuplicateCandidate:
        if self.record_id_a == self.record_id_b:
            raise ValueError("a record cannot be a duplicate candidate of itself")
        if self.record_id_a > self.record_id_b:
            raise ValueError("record ids must be in canonical order (a < b)")
        stamps = (self.resolved_by, self.resolved_at)
        if self.resolution != "unresolved":
            if any(v is None for v in stamps):
                raise ValueError("resolved candidates need resolved_by and resolved_at")
        elif any(v is not None for v in stamps):
            raise ValueError("resolved_by/resolved_at are only allowed once resolved")
        return self

    @property
    def is_resolved(self) -> bool:
        return self.resolution != "unresolved"
