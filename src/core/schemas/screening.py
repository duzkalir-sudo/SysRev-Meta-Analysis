"""Stage 2 contracts: dual independent screening decisions and conflicts."""

from __future__ import annotations

from pydantic import Field, model_validator

from src.core.enums import ReviewerRole, ScreeningStage, ScreeningVerdict
from src.core.schemas.base import AwareDatetime, Identifier, StrictModel, new_id, utc_now


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

    @model_validator(mode="after")
    def _rules(self) -> ScreeningDecision:
        if self.verdict == ScreeningVerdict.EXCLUDE and not self.exclusion_reason:
            raise ValueError("exclusion decisions require an exclusion_reason")
        if self.verdict != ScreeningVerdict.EXCLUDE and self.exclusion_reason:
            raise ValueError("exclusion_reason is only allowed for exclude verdicts")
        if self.reviewer == ReviewerRole.ADJUDICATOR and not self.is_human:
            raise ValueError("adjudication decisions must be made by a human")
        if self.stage == ScreeningStage.FULL_TEXT and self.verdict == ScreeningVerdict.UNCERTAIN:
            raise ValueError("full-text decisions must be include or exclude")
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
