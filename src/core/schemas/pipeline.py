"""State-machine contracts: HITL gate decisions, transitions and PRISMA flow counts."""

from __future__ import annotations

from typing import Any

from pydantic import Field, ModelWrapValidatorHandler, computed_field, model_validator

from src.core.enums import ActorType, GateId, GateVerdict, Phase, SourceDatabase
from src.core.schemas.base import (
    AwareDatetime,
    FrozenModel,
    Identifier,
    NonEmptyStr,
    StrictModel,
    new_id,
    utc_now,
)


class GateDecision(FrozenModel):
    """A human reviewer's approval or rejection at a HITL gate."""

    decision_id: Identifier = Field(default_factory=new_id)
    gate: GateId
    verdict: GateVerdict
    reviewer_name: NonEmptyStr
    reviewer_id: Identifier | None = Field(
        default=None,
        description="enrolled reviewer who authenticated this decision; required to persist it",
    )
    decided_at: AwareDatetime = Field(default_factory=utc_now)
    comments: str | None = None

    @model_validator(mode="after")
    def _rejections_need_reasons(self) -> GateDecision:
        if self.verdict == GateVerdict.REJECTED and not (self.comments and self.comments.strip()):
            raise ValueError("a rejection must include comments explaining what to change")
        return self


class StateTransition(FrozenModel):
    """Append-only audit-trail entry for one phase change."""

    transition_id: Identifier = Field(default_factory=new_id)
    project_id: Identifier
    from_phase: Phase
    to_phase: Phase
    actor_type: ActorType
    actor: NonEmptyStr
    reason: str | None = None
    gate_decision: GateDecision | None = None
    occurred_at: AwareDatetime = Field(default_factory=utc_now)


class PRISMACounts(StrictModel):
    """PRISMA 2020 flow-diagram numbers (Page MJ et al. BMJ 2021;372:n71).

    Arithmetic invariants are enforced on construction so an inconsistent flow
    diagram can never be rendered.
    """

    identified_by_database: dict[SourceDatabase, int] = Field(default_factory=dict)
    identified_from_registers: int = Field(default=0, ge=0)
    duplicates_removed: int = Field(default=0, ge=0)
    marked_ineligible_by_automation: int = Field(default=0, ge=0)
    removed_other_reasons: int = Field(default=0, ge=0)
    records_screened: int = Field(default=0, ge=0)
    records_excluded: int = Field(default=0, ge=0)
    reports_sought: int = Field(default=0, ge=0)
    reports_not_retrieved: int = Field(default=0, ge=0)
    reports_assessed: int = Field(default=0, ge=0)
    reports_excluded_by_reason: dict[str, int] = Field(default_factory=dict)
    studies_included: int = Field(default=0, ge=0)
    reports_included: int = Field(default=0, ge=0)
    # Work in progress (not PRISMA boxes): must be zero before a flow diagram is final.
    # ``records_screened`` counts only records with a *completed* title/abstract decision.
    records_awaiting_screening: int = Field(default=0, ge=0)
    reports_awaiting_retrieval: int = Field(default=0, ge=0)
    reports_awaiting_assessment: int = Field(default=0, ge=0)

    @model_validator(mode="wrap")
    @classmethod
    def _derived_total_is_not_input(
        cls, data: Any, handler: ModelWrapValidatorHandler[PRISMACounts]
    ) -> PRISMACounts:
        claimed = None
        if isinstance(data, dict) and "total_identified" in data:
            data = dict(data)
            claimed = data.pop("total_identified")
        counts = handler(data)
        if claimed is not None and claimed != counts.total_identified:
            raise ValueError("claimed total_identified contradicts per-source counts")
        return counts

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_identified(self) -> int:
        return sum(self.identified_by_database.values()) + self.identified_from_registers

    @property
    def is_final(self) -> bool:
        """True when no record awaits screening and no report awaits retrieval/assessment."""
        return (
            self.records_awaiting_screening == 0
            and self.reports_awaiting_retrieval == 0
            and self.reports_awaiting_assessment == 0
        )

    def invariant_violations(self) -> list[str]:
        v: list[str] = []
        if any(n < 0 for n in self.identified_by_database.values()):
            v.append("per-database counts must be non-negative")
        if any(n < 0 for n in self.reports_excluded_by_reason.values()):
            v.append("exclusion-reason counts must be non-negative")
        removed = (
            self.duplicates_removed
            + self.marked_ineligible_by_automation
            + self.removed_other_reasons
        )
        entering_screening = self.total_identified - removed
        if self.records_screened + self.records_awaiting_screening != entering_screening:
            v.append(
                f"records_screened ({self.records_screened}) + awaiting screening "
                f"({self.records_awaiting_screening}) != identified ({self.total_identified})"
                f" - removed before screening ({removed})"
            )
        if self.reports_sought != self.records_screened - self.records_excluded:
            v.append(
                f"reports_sought ({self.reports_sought}) != screened ({self.records_screened})"
                f" - excluded ({self.records_excluded})"
            )
        retrieved = (
            self.reports_sought - self.reports_not_retrieved - self.reports_awaiting_retrieval
        )
        if self.reports_assessed != retrieved:
            v.append(
                f"reports_assessed ({self.reports_assessed}) != sought ({self.reports_sought})"
                f" - not retrieved ({self.reports_not_retrieved})"
                f" - awaiting retrieval ({self.reports_awaiting_retrieval})"
            )
        excluded = sum(self.reports_excluded_by_reason.values())
        decided = self.reports_assessed - excluded - self.reports_awaiting_assessment
        if self.reports_included != decided:
            v.append(
                f"reports_included ({self.reports_included}) != assessed "
                f"({self.reports_assessed}) - excluded with reasons ({excluded})"
                f" - awaiting assessment ({self.reports_awaiting_assessment})"
            )
        if self.studies_included > self.reports_included:
            v.append("studies_included cannot exceed reports_included")
        return v

    @model_validator(mode="after")
    def _invariants(self) -> PRISMACounts:
        violations = self.invariant_violations()
        if violations:
            raise ValueError("PRISMA invariants violated: " + "; ".join(violations))
        return self


class ProjectState(StrictModel):
    project_id: Identifier = Field(default_factory=new_id)
    title: NonEmptyStr
    phase: Phase = Phase.STAGE1_PROTOCOL
    history: list[StateTransition] = Field(default_factory=list)
    created_at: AwareDatetime = Field(default_factory=utc_now)
