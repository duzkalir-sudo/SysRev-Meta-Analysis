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


@dataclass(frozen=True, slots=True, kw_only=True)
class GuardContext:
    """Snapshot of persisted project facts that guards evaluate.

    Every field is required: there are deliberately no defaults, so a caller cannot obtain
    a passing guard by omitting information. Production code builds this exclusively with
    :meth:`Repository.build_guard_context`, which derives each value from stored records
    and decisions. Absent data is reported as missing work (fail closed), never as done.
    """

    protocol_blockers: tuple[str, ...]
    # Stage 2 — identification, deduplication, title/abstract screening
    records_identified: int
    records_unique: int
    unresolved_duplicate_candidates: int
    ta_missing_decisions: int
    ta_unresolved_conflicts: int
    ta_included: int
    unresolved_cohort_warnings: int
    # PRISMA snapshot, cross-checked against persisted records
    prisma_recorded: bool
    prisma_violations: tuple[str, ...]
    prisma_awaiting_screening: int
    prisma_pending_reports: int
    # Stage 3 — full-text retrieval manifest
    fulltext_items_missing: int
    fulltext_items_blocking: int
    # Stage 4 — full-text eligibility and extraction
    ft_missing_decisions: int
    ft_unresolved_conflicts: int
    ft_included: int
    ft_awaiting_human_confirmation: int
    included_reports_without_extraction: int
    arms_for_non_included_reports: int
    extracted_values_without_evidence: int
    extraction_arms: int
    unapproved_extraction_arms: int
    extraction_items_pending: int
    included_studies_without_approved_rob: int
    # Stages 5-6
    stats_output_present: bool
    manuscript_present: bool
    audit_passed: bool | None


Guard = Callable[[GuardContext], list[str]]


def _protocol_ready(c: GuardContext) -> list[str]:
    return [f"protocol: {b}" for b in c.protocol_blockers]


def _prisma_recorded(c: GuardContext) -> list[str]:
    if not c.prisma_recorded:
        return ["PRISMA counts have not been recorded for this project"]
    return [f"PRISMA: {p}" for p in c.prisma_violations]


def _dual_screening_done(c: GuardContext) -> list[str]:
    """Stage 2 → Gate 1.5: search executed, both reviewers screened every unique record."""
    v: list[str] = []
    if c.records_identified == 0:
        v.append("no records identified by the search")
    if c.ta_missing_decisions:
        v.append(
            f"{c.ta_missing_decisions} unique record(s) lack title/abstract decisions from "
            "both reviewers"
        )
    v.extend(_prisma_recorded(c))
    return v


def _screening_complete(c: GuardContext) -> list[str]:
    """Gate 1.5 → Stage 3: every conflict, duplicate and overlap question settled by a human."""
    v = _dual_screening_done(c)
    if c.ta_unresolved_conflicts:
        v.append(f"{c.ta_unresolved_conflicts} unresolved title/abstract screening conflict(s)")
    if c.unresolved_duplicate_candidates:
        v.append(f"{c.unresolved_duplicate_candidates} unresolved possible-duplicate pair(s)")
    if c.unresolved_cohort_warnings:
        v.append(f"{c.unresolved_cohort_warnings} unresolved cohort-overlap warning(s)")
    if c.prisma_recorded and c.prisma_awaiting_screening:
        v.append(f"PRISMA: {c.prisma_awaiting_screening} record(s) still awaiting screening")
    if c.records_identified and c.ta_included == 0 and not c.ta_missing_decisions:
        v.append("no records were included at title/abstract screening")
    return v


def _manifest_complete(c: GuardContext) -> list[str]:
    """Stage 3 → Gate 2: every title/abstract-included record has a full-text manifest entry."""
    v: list[str] = []
    if c.ta_included == 0:
        v.append("no records were included at title/abstract screening")
    if c.fulltext_items_missing:
        v.append(f"{c.fulltext_items_missing} included record(s) have no full-text manifest entry")
    return v


def _fulltext_resolved(c: GuardContext) -> list[str]:
    v = _manifest_complete(c)
    if c.fulltext_items_blocking:
        v.append(
            f"{c.fulltext_items_blocking} report(s) still missing: upload the PDF or mark "
            "unobtainable with a reason"
        )
    return v


def _eligibility_documented(c: GuardContext) -> list[str]:
    v: list[str] = []
    if c.ft_missing_decisions:
        v.append(
            f"{c.ft_missing_decisions} retrieved report(s) lack full-text eligibility decisions"
        )
    if c.ft_unresolved_conflicts:
        v.append(f"{c.ft_unresolved_conflicts} unresolved full-text eligibility conflict(s)")
    if c.ft_included == 0 and not c.ft_missing_decisions and not c.ft_unresolved_conflicts:
        v.append("no reports were included after full-text assessment")
    if c.ft_awaiting_human_confirmation:
        v.append(
            f"{c.ft_awaiting_human_confirmation} full-text eligibility outcome(s) rest only on "
            "automated reviewers and await confirmation by an authenticated human"
        )
    return v


def _extraction_grounded(c: GuardContext) -> list[str]:
    """Stage 4 → Gate 3: eligibility documented and every included report extracted."""
    v = _eligibility_documented(c)
    if c.extraction_arms == 0:
        v.append("no extracted study arms")
    if c.included_reports_without_extraction:
        v.append(
            f"{c.included_reports_without_extraction} included report(s) have no extracted data"
        )
    if c.arms_for_non_included_reports:
        v.append(
            f"{c.arms_for_non_included_reports} extracted arm(s) belong to reports not included "
            "at full-text assessment"
        )
    if c.extracted_values_without_evidence:
        v.append(f"{c.extracted_values_without_evidence} extracted value(s) lack SourceEvidence")
    return v


def _extraction_approved(c: GuardContext) -> list[str]:
    v = _extraction_grounded(c)
    if c.unapproved_extraction_arms:
        v.append(f"{c.unapproved_extraction_arms} extraction arm(s) not approved by a human")
    if c.extraction_items_pending:
        v.append(f"{c.extraction_items_pending} master-table extraction item(s) await human review")
    if c.included_studies_without_approved_rob:
        v.append(
            f"{c.included_studies_without_approved_rob} included study(ies) lack a "
            "human-approved risk-of-bias assessment"
        )
    v.extend(_prisma_recorded(c))
    pending = c.prisma_awaiting_screening + c.prisma_pending_reports
    if c.prisma_recorded and pending:
        v.append(f"PRISMA flow not final: {pending} record(s)/report(s) awaiting decision")
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
    (Phase.STAGE2_SEARCH_SCREENING, Phase.GATE1_5_SCREENING_CONFLICTS): _dual_screening_done,
    (Phase.GATE1_5_SCREENING_CONFLICTS, Phase.STAGE3_FULLTEXT_RETRIEVAL): _screening_complete,
    (Phase.STAGE3_FULLTEXT_RETRIEVAL, Phase.GATE2_MISSING_FULLTEXT): _manifest_complete,
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
