"""Deterministic 6-stage pipeline state machine with human-in-the-loop gates.

This module is the single source of truth for *what may happen next*. Any
orchestrator (LangGraph graph, CLI, FastAPI endpoint) must request transitions
through :class:`PipelineStateMachine`; it cannot set a phase directly.

Rules enforced here:

* Only transitions listed in :data:`TRANSITIONS` are legal.
* Leaving a gate phase requires a :class:`GateDecision` for *that* gate, made by
  a human actor. Approval moves forward; rejection (with comments) moves back.
* Each forward step has guards evaluated against a :class:`GuardContext`
  snapshot (e.g. no unresolved screening conflicts before leaving Gate 1.5).
* Every transition appends an immutable :class:`StateTransition` to the history.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from src.core.enums import ActorType, GateId, GateVerdict, Phase
from src.core.exceptions import GuardViolationError, StateTransitionError
from src.core.schemas.pipeline import GateDecision, StateTransition

GATE_OF_PHASE: dict[Phase, GateId] = {
    Phase.GATE1_SEARCH_STRATEGY: GateId.GATE_1,
    Phase.GATE1_5_SCREENING_CONFLICTS: GateId.GATE_1_5,
    Phase.GATE2_MISSING_FULLTEXT: GateId.GATE_2,
    Phase.GATE3_EXTRACTION_APPROVAL: GateId.GATE_3,
}

# gate phase -> (phase on approval, phase on rejection)
GATE_OUTCOMES: dict[Phase, tuple[Phase, Phase]] = {
    Phase.GATE1_SEARCH_STRATEGY: (Phase.STAGE2_SEARCH_SCREENING, Phase.STAGE1_PROTOCOL),
    Phase.GATE1_5_SCREENING_CONFLICTS: (
        Phase.STAGE3_FULLTEXT_RETRIEVAL,
        Phase.STAGE2_SEARCH_SCREENING,
    ),
    Phase.GATE2_MISSING_FULLTEXT: (Phase.STAGE4_EXTRACTION_ROB, Phase.STAGE3_FULLTEXT_RETRIEVAL),
    Phase.GATE3_EXTRACTION_APPROVAL: (Phase.STAGE5_META_ANALYSIS, Phase.STAGE4_EXTRACTION_ROB),
}

TRANSITIONS: dict[Phase, frozenset[Phase]] = {
    Phase.STAGE1_PROTOCOL: frozenset({Phase.GATE1_SEARCH_STRATEGY}),
    Phase.STAGE2_SEARCH_SCREENING: frozenset({Phase.GATE1_5_SCREENING_CONFLICTS}),
    Phase.STAGE3_FULLTEXT_RETRIEVAL: frozenset({Phase.GATE2_MISSING_FULLTEXT}),
    Phase.STAGE4_EXTRACTION_ROB: frozenset({Phase.GATE3_EXTRACTION_APPROVAL}),
    Phase.STAGE5_META_ANALYSIS: frozenset({Phase.STAGE6_MANUSCRIPT}),
    Phase.STAGE6_MANUSCRIPT: frozenset({Phase.STAGE6_AUDIT}),
    Phase.STAGE6_AUDIT: frozenset({Phase.COMPLETED, Phase.STAGE6_MANUSCRIPT}),
    Phase.COMPLETED: frozenset(),
    **{gate: frozenset(outcomes) for gate, outcomes in GATE_OUTCOMES.items()},
}


@dataclass(frozen=True, slots=True)
class GuardContext:
    """Snapshot of project facts that guards evaluate. Built by the persistence layer."""

    protocol_blockers: tuple[str, ...] = ("protocol not drafted",)
    records_identified: int = 0
    unresolved_screening_conflicts: int = 0
    unresolved_cohort_warnings: int = 0
    prisma_violations: tuple[str, ...] = ()
    prisma_pending_reports: int = 0
    fulltext_items_blocking: int = 0
    extracted_values_without_evidence: int = 0
    extraction_arms: int = 0
    unapproved_extraction_arms: int = 0
    stats_output_present: bool = False
    manuscript_present: bool = False
    audit_passed: bool | None = None


Guard = Callable[[GuardContext], list[str]]


def _protocol_ready(c: GuardContext) -> list[str]:
    return [f"protocol: {b}" for b in c.protocol_blockers]


def _screening_complete(c: GuardContext) -> list[str]:
    v: list[str] = []
    if c.records_identified == 0:
        v.append("no records identified by the search")
    if c.unresolved_screening_conflicts:
        v.append(f"{c.unresolved_screening_conflicts} unresolved screening conflict(s)")
    if c.unresolved_cohort_warnings:
        v.append(f"{c.unresolved_cohort_warnings} unresolved cohort-overlap warning(s)")
    v.extend(f"PRISMA: {p}" for p in c.prisma_violations)
    return v


def _fulltext_resolved(c: GuardContext) -> list[str]:
    if c.fulltext_items_blocking:
        return [
            f"{c.fulltext_items_blocking} report(s) still missing: upload the PDF or mark "
            "unobtainable with a reason"
        ]
    return []


def _extraction_grounded(c: GuardContext) -> list[str]:
    v: list[str] = []
    if c.extraction_arms == 0:
        v.append("no extracted study arms")
    if c.extracted_values_without_evidence:
        v.append(f"{c.extracted_values_without_evidence} extracted value(s) lack SourceEvidence")
    return v


def _extraction_approved(c: GuardContext) -> list[str]:
    v = _extraction_grounded(c)
    if c.unapproved_extraction_arms:
        v.append(f"{c.unapproved_extraction_arms} extraction arm(s) not approved by a human")
    if c.prisma_pending_reports:
        v.append(f"PRISMA flow not final: {c.prisma_pending_reports} report(s) awaiting decision")
    v.extend(f"PRISMA: {p}" for p in c.prisma_violations)
    return v


def _stats_present(c: GuardContext) -> list[str]:
    return [] if c.stats_output_present else ["stats_output.json has not been produced"]


def _manuscript_present(c: GuardContext) -> list[str]:
    return [] if c.manuscript_present else ["no manuscript draft to audit"]


def _audit_passed(c: GuardContext) -> list[str]:
    return [] if c.audit_passed is True else ["latest audit report has not passed"]


def _audit_failed(c: GuardContext) -> list[str]:
    return [] if c.audit_passed is False else ["no failed audit report to act on"]


GUARDS: dict[tuple[Phase, Phase], Guard] = {
    (Phase.GATE1_SEARCH_STRATEGY, Phase.STAGE2_SEARCH_SCREENING): _protocol_ready,
    (Phase.STAGE2_SEARCH_SCREENING, Phase.GATE1_5_SCREENING_CONFLICTS): (
        lambda c: [] if c.records_identified else ["no records identified by the search"]
    ),
    (Phase.GATE1_5_SCREENING_CONFLICTS, Phase.STAGE3_FULLTEXT_RETRIEVAL): _screening_complete,
    (Phase.GATE2_MISSING_FULLTEXT, Phase.STAGE4_EXTRACTION_ROB): _fulltext_resolved,
    (Phase.STAGE4_EXTRACTION_ROB, Phase.GATE3_EXTRACTION_APPROVAL): _extraction_grounded,
    (Phase.GATE3_EXTRACTION_APPROVAL, Phase.STAGE5_META_ANALYSIS): _extraction_approved,
    (Phase.STAGE5_META_ANALYSIS, Phase.STAGE6_MANUSCRIPT): _stats_present,
    (Phase.STAGE6_MANUSCRIPT, Phase.STAGE6_AUDIT): _manuscript_present,
    (Phase.STAGE6_AUDIT, Phase.COMPLETED): _audit_passed,
    (Phase.STAGE6_AUDIT, Phase.STAGE6_MANUSCRIPT): _audit_failed,
}


class PipelineStateMachine:
    """Holds the current phase of one project and validates every transition."""

    def __init__(
        self,
        project_id: str,
        phase: Phase = Phase.STAGE1_PROTOCOL,
        history: list[StateTransition] | None = None,
    ) -> None:
        self.project_id = project_id
        self._phase = phase
        self._history: list[StateTransition] = list(history or [])

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def history(self) -> tuple[StateTransition, ...]:
        return tuple(self._history)

    @property
    def is_paused_at_gate(self) -> bool:
        return self._phase in GATE_OF_PHASE

    @property
    def is_terminal(self) -> bool:
        return not TRANSITIONS[self._phase]

    def allowed_targets(self) -> frozenset[Phase]:
        return TRANSITIONS[self._phase]

    def check(self, target: Phase, context: GuardContext) -> list[str]:
        """Return guard violations for ``current -> target`` without transitioning."""
        guard = GUARDS.get((self._phase, target))
        return guard(context) if guard else []

    def transition(
        self,
        target: Phase,
        *,
        actor: str,
        actor_type: ActorType,
        context: GuardContext,
        gate_decision: GateDecision | None = None,
        reason: str | None = None,
    ) -> StateTransition:
        source = self._phase
        if target not in TRANSITIONS[source]:
            raise StateTransitionError(f"illegal transition {source} -> {target}")

        if source in GATE_OF_PHASE:
            self._validate_gate_exit(source, target, actor_type, gate_decision)
        elif gate_decision is not None:
            raise StateTransitionError(f"{source} is not a gate; gate decisions are not accepted")

        rejected = gate_decision is not None and gate_decision.verdict == GateVerdict.REJECTED
        if not rejected:
            violations = self.check(target, context)
            if violations:
                raise GuardViolationError(
                    f"cannot move {source} -> {target}: " + "; ".join(violations), violations
                )

        record = StateTransition(
            project_id=self.project_id,
            from_phase=source,
            to_phase=target,
            actor_type=actor_type,
            actor=actor,
            reason=reason or (gate_decision.comments if gate_decision else None),
            gate_decision=gate_decision,
        )
        self._history.append(record)
        self._phase = target
        return record

    def submit_gate_decision(
        self, decision: GateDecision, context: GuardContext
    ) -> StateTransition:
        """Convenience wrapper: route a human gate decision to the correct next phase."""
        if self._phase not in GATE_OF_PHASE:
            raise StateTransitionError(f"pipeline is not paused at a gate (phase={self._phase})")
        approve_to, reject_to = GATE_OUTCOMES[self._phase]
        target = approve_to if decision.verdict == GateVerdict.APPROVED else reject_to
        return self.transition(
            target,
            actor=decision.reviewer_name,
            actor_type=ActorType.HUMAN,
            context=context,
            gate_decision=decision,
        )

    @staticmethod
    def _validate_gate_exit(
        source: Phase,
        target: Phase,
        actor_type: ActorType,
        decision: GateDecision | None,
    ) -> None:
        if actor_type != ActorType.HUMAN:
            raise StateTransitionError(f"only a human can release HITL gate {source}")
        if decision is None:
            raise StateTransitionError(f"leaving {source} requires a GateDecision")
        expected_gate = GATE_OF_PHASE[source]
        if decision.gate != expected_gate:
            raise StateTransitionError(
                f"decision is for {decision.gate}, but the pipeline is at {expected_gate}"
            )
        approve_to, reject_to = GATE_OUTCOMES[source]
        expected_target = approve_to if decision.verdict == GateVerdict.APPROVED else reject_to
        if target != expected_target:
            raise StateTransitionError(
                f"a {decision.verdict} decision at {source} leads to {expected_target}, "
                f"not {target}"
            )
