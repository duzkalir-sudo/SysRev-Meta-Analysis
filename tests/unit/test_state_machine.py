from __future__ import annotations

import dataclasses

import pytest

from src.core.enums import ActorType, GateId, GateVerdict, Phase
from src.core.exceptions import GuardViolationError, StateTransitionError
from src.core.schemas import GateDecision
from src.state.machine import (
    GATE_OF_PHASE,
    GATE_OUTCOMES,
    GUARDS,
    TRANSITIONS,
    GuardContext,
    PipelineStateMachine,
)
from tests.conftest import EMPTY, READY


def approve(gate: GateId) -> GateDecision:
    return GateDecision(gate=gate, verdict=GateVerdict.APPROVED, reviewer_name="Dr Reviewer")


def reject(gate: GateId) -> GateDecision:
    return GateDecision(
        gate=gate, verdict=GateVerdict.REJECTED, reviewer_name="Dr Reviewer", comments="revise"
    )


def system_step(m: PipelineStateMachine, target: Phase, ctx: GuardContext = READY) -> None:
    m.transition(target, actor="orchestrator", actor_type=ActorType.SYSTEM, context=ctx)


def test_transition_table_is_complete_and_consistent() -> None:
    assert set(TRANSITIONS) == set(Phase)
    for gate_phase, (ok, back) in GATE_OUTCOMES.items():
        assert TRANSITIONS[gate_phase] == {ok, back}
    for src, dst in GUARDS:
        assert dst in TRANSITIONS[src]
    # Every phase except COMPLETED is reachable from Stage 1.
    seen, frontier = {Phase.STAGE1_PROTOCOL}, [Phase.STAGE1_PROTOCOL]
    while frontier:
        for nxt in TRANSITIONS[frontier.pop()]:
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    assert seen == set(Phase)


def test_happy_path_through_all_six_stages() -> None:
    m = PipelineStateMachine("p1")
    system_step(m, Phase.GATE1_SEARCH_STRATEGY)
    assert m.is_paused_at_gate
    m.submit_gate_decision(approve(GateId.GATE_1), READY)
    system_step(m, Phase.GATE1_5_SCREENING_CONFLICTS)
    m.submit_gate_decision(approve(GateId.GATE_1_5), READY)
    system_step(m, Phase.GATE2_MISSING_FULLTEXT)
    m.submit_gate_decision(approve(GateId.GATE_2), READY)
    system_step(m, Phase.GATE3_EXTRACTION_APPROVAL)
    m.submit_gate_decision(approve(GateId.GATE_3), READY)
    assert m.phase == Phase.STAGE5_META_ANALYSIS
    system_step(m, Phase.STAGE6_MANUSCRIPT)
    system_step(m, Phase.STAGE6_AUDIT)
    system_step(m, Phase.COMPLETED)
    assert m.is_terminal
    assert len(m.history) == 11
    assert [t.to_phase for t in m.history][-1] == Phase.COMPLETED
    gate_transitions = [t for t in m.history if t.gate_decision is not None]
    assert len(gate_transitions) == 4
    assert all(t.actor_type == ActorType.HUMAN for t in gate_transitions)


def test_illegal_transition_rejected() -> None:
    m = PipelineStateMachine("p1")
    with pytest.raises(StateTransitionError, match="illegal"):
        system_step(m, Phase.STAGE5_META_ANALYSIS)
    assert m.phase == Phase.STAGE1_PROTOCOL
    assert m.history == ()


def test_completed_is_terminal() -> None:
    m = PipelineStateMachine("p1", Phase.COMPLETED)
    assert m.allowed_targets() == frozenset()
    with pytest.raises(StateTransitionError):
        system_step(m, Phase.STAGE1_PROTOCOL)


@pytest.mark.parametrize("gate_phase", list(GATE_OF_PHASE))
def test_gates_cannot_be_released_by_system_or_agent(gate_phase: Phase) -> None:
    m = PipelineStateMachine("p1", gate_phase)
    approve_to, _ = GATE_OUTCOMES[gate_phase]
    for actor_type in (ActorType.SYSTEM, ActorType.AGENT):
        with pytest.raises(StateTransitionError, match="only a human"):
            m.transition(
                approve_to,
                actor="llm",
                actor_type=actor_type,
                context=READY,
                gate_decision=approve(GATE_OF_PHASE[gate_phase]),
            )
    assert m.phase == gate_phase


def test_gate_requires_decision() -> None:
    m = PipelineStateMachine("p1", Phase.GATE1_SEARCH_STRATEGY)
    with pytest.raises(StateTransitionError, match="requires a GateDecision"):
        m.transition(
            Phase.STAGE2_SEARCH_SCREENING, actor="Dr R", actor_type=ActorType.HUMAN, context=READY
        )


def test_gate_decision_must_match_current_gate() -> None:
    m = PipelineStateMachine("p1", Phase.GATE1_SEARCH_STRATEGY)
    with pytest.raises(StateTransitionError, match="decision is for"):
        m.submit_gate_decision(approve(GateId.GATE_3), READY)


def test_gate_verdict_must_match_target() -> None:
    m = PipelineStateMachine("p1", Phase.GATE1_SEARCH_STRATEGY)
    with pytest.raises(StateTransitionError, match="leads to"):
        m.transition(
            Phase.STAGE1_PROTOCOL,
            actor="Dr R",
            actor_type=ActorType.HUMAN,
            context=READY,
            gate_decision=approve(GateId.GATE_1),
        )


def test_gate_decision_rejected_outside_gates() -> None:
    m = PipelineStateMachine("p1")
    with pytest.raises(StateTransitionError, match="not a gate"):
        m.transition(
            Phase.GATE1_SEARCH_STRATEGY,
            actor="x",
            actor_type=ActorType.HUMAN,
            context=READY,
            gate_decision=approve(GateId.GATE_1),
        )
    with pytest.raises(StateTransitionError, match="not paused"):
        m.submit_gate_decision(approve(GateId.GATE_1), READY)


@pytest.mark.parametrize("gate_phase", list(GATE_OF_PHASE))
def test_rejection_moves_back_and_skips_guards(gate_phase: Phase) -> None:
    m = PipelineStateMachine("p1", gate_phase)
    t = m.submit_gate_decision(reject(GATE_OF_PHASE[gate_phase]), EMPTY)
    assert m.phase == GATE_OUTCOMES[gate_phase][1]
    assert t.reason == "revise"


@pytest.mark.parametrize(
    ("phase", "target", "bad_context", "fragment"),
    [
        (Phase.GATE1_SEARCH_STRATEGY, Phase.STAGE2_SEARCH_SCREENING,
         {"protocol_blockers": ("MeSH unverified",)}, "MeSH unverified"),
        (Phase.STAGE2_SEARCH_SCREENING, Phase.GATE1_5_SCREENING_CONFLICTS,
         {"records_identified": 0}, "no records"),
        (Phase.GATE1_5_SCREENING_CONFLICTS, Phase.STAGE3_FULLTEXT_RETRIEVAL,
         {"ta_unresolved_conflicts": 3}, "3 unresolved title/abstract"),
        (Phase.GATE1_5_SCREENING_CONFLICTS, Phase.STAGE3_FULLTEXT_RETRIEVAL,
         {"unresolved_cohort_warnings": 1}, "cohort-overlap"),
        (Phase.GATE1_5_SCREENING_CONFLICTS, Phase.STAGE3_FULLTEXT_RETRIEVAL,
         {"prisma_violations": ("bad sum",)}, "PRISMA: bad sum"),
        (Phase.GATE2_MISSING_FULLTEXT, Phase.STAGE4_EXTRACTION_ROB,
         {"fulltext_items_blocking": 2}, "2 report"),
        (Phase.STAGE4_EXTRACTION_ROB, Phase.GATE3_EXTRACTION_APPROVAL,
         {"extracted_values_without_evidence": 1}, "SourceEvidence"),
        (Phase.STAGE4_EXTRACTION_ROB, Phase.GATE3_EXTRACTION_APPROVAL,
         {"extraction_arms": 0}, "no extracted"),
        (Phase.GATE3_EXTRACTION_APPROVAL, Phase.STAGE5_META_ANALYSIS,
         {"unapproved_extraction_arms": 4}, "not approved"),
        (Phase.STAGE5_META_ANALYSIS, Phase.STAGE6_MANUSCRIPT,
         {"stats_output_present": False}, "stats_output.json"),
        (Phase.STAGE6_MANUSCRIPT, Phase.STAGE6_AUDIT,
         {"manuscript_present": False}, "manuscript"),
        (Phase.STAGE6_AUDIT, Phase.COMPLETED, {"audit_passed": False}, "not passed"),
        (Phase.STAGE6_AUDIT, Phase.COMPLETED, {"audit_passed": None}, "not passed"),
        (Phase.STAGE6_AUDIT, Phase.COMPLETED, {"audit_problems": ("no human sign-off",)},
         "sign-off"),
    ],
)  # fmt: skip
def test_guards_block_unsafe_progress(
    phase: Phase, target: Phase, bad_context: dict[str, object], fragment: str
) -> None:
    m = PipelineStateMachine("p1", phase)
    ctx = dataclasses.replace(READY, **bad_context)  # type: ignore[arg-type]
    actor_type = ActorType.HUMAN if phase in GATE_OF_PHASE else ActorType.SYSTEM
    decision = approve(GATE_OF_PHASE[phase]) if phase in GATE_OF_PHASE else None
    with pytest.raises(GuardViolationError, match=fragment) as info:
        m.transition(target, actor="x", actor_type=actor_type, context=ctx, gate_decision=decision)
    assert info.value.violations
    assert m.phase == phase
    assert m.check(target, ctx)


def test_failed_audit_loops_back_to_drafting() -> None:
    m = PipelineStateMachine("p1", Phase.STAGE6_AUDIT)
    system_step(m, Phase.STAGE6_MANUSCRIPT, dataclasses.replace(READY, audit_passed=False))
    assert m.phase == Phase.STAGE6_MANUSCRIPT


def test_revision_after_passing_audit_is_allowed() -> None:
    """Returning to drafting never certifies anything (Session 6)."""
    m = PipelineStateMachine("p1", Phase.STAGE6_AUDIT)
    system_step(m, Phase.STAGE6_MANUSCRIPT, READY)
    assert m.phase == Phase.STAGE6_MANUSCRIPT


def test_history_is_immutable_view() -> None:
    m = PipelineStateMachine("p1")
    system_step(m, Phase.GATE1_SEARCH_STRATEGY)
    assert isinstance(m.history, tuple)
    with pytest.raises(Exception, match=r"frozen|immutable"):
        m.history[0].actor = "tampered"  # type: ignore[misc]


def test_stats_require_final_prisma_flow() -> None:
    m = PipelineStateMachine("p1", Phase.GATE3_EXTRACTION_APPROVAL)
    ctx = dataclasses.replace(READY, prisma_pending_reports=2)
    with pytest.raises(GuardViolationError, match="PRISMA flow not final"):
        m.submit_gate_decision(approve(GateId.GATE_3), ctx)


# --------------------------------------------------------------------------- fail-closed guards
# Regression tests for the PR #1 review: absent data must never read as completed work.


def test_guard_context_has_no_defaults() -> None:
    """Omitting any fact is a TypeError, so no caller can get a passing guard by omission."""
    with pytest.raises(TypeError):
        GuardContext()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        GuardContext(protocol_blockers=())  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("phase", "target"),
    [
        (Phase.STAGE2_SEARCH_SCREENING, Phase.GATE1_5_SCREENING_CONFLICTS),
        (Phase.GATE1_5_SCREENING_CONFLICTS, Phase.STAGE3_FULLTEXT_RETRIEVAL),
        (Phase.STAGE3_FULLTEXT_RETRIEVAL, Phase.GATE2_MISSING_FULLTEXT),
        (Phase.GATE2_MISSING_FULLTEXT, Phase.STAGE4_EXTRACTION_ROB),
        (Phase.STAGE4_EXTRACTION_ROB, Phase.GATE3_EXTRACTION_APPROVAL),
        (Phase.GATE3_EXTRACTION_APPROVAL, Phase.STAGE5_META_ANALYSIS),
    ],
)
def test_empty_project_cannot_pass_any_stage_2_to_4_step(phase: Phase, target: Phase) -> None:
    m = PipelineStateMachine("p1", phase)
    decision = approve(GATE_OF_PHASE[phase]) if phase in GATE_OF_PHASE else None
    actor_type = ActorType.HUMAN if decision else ActorType.SYSTEM
    with pytest.raises(GuardViolationError):
        m.transition(target, actor="x", actor_type=actor_type, context=EMPTY,
                     gate_decision=decision)  # fmt: skip
    assert m.phase == phase


@pytest.mark.parametrize(
    ("phase", "target", "bad_context", "fragment"),
    [
        (Phase.STAGE2_SEARCH_SCREENING, Phase.GATE1_5_SCREENING_CONFLICTS,
         {"prisma_recorded": False}, "PRISMA counts have not been recorded"),
        (Phase.STAGE2_SEARCH_SCREENING, Phase.GATE1_5_SCREENING_CONFLICTS,
         {"ta_missing_decisions": 4}, "lack title/abstract decisions"),
        (Phase.GATE1_5_SCREENING_CONFLICTS, Phase.STAGE3_FULLTEXT_RETRIEVAL,
         {"unresolved_duplicate_candidates": 2}, "possible-duplicate"),
        (Phase.GATE1_5_SCREENING_CONFLICTS, Phase.STAGE3_FULLTEXT_RETRIEVAL,
         {"prisma_awaiting_screening": 5}, "awaiting screening"),
        (Phase.GATE1_5_SCREENING_CONFLICTS, Phase.STAGE3_FULLTEXT_RETRIEVAL,
         {"ta_included": 0}, "no records were included"),
        (Phase.STAGE3_FULLTEXT_RETRIEVAL, Phase.GATE2_MISSING_FULLTEXT,
         {"fulltext_items_missing": 3}, "no full-text manifest entry"),
        (Phase.GATE2_MISSING_FULLTEXT, Phase.STAGE4_EXTRACTION_ROB,
         {"fulltext_items_missing": 1}, "no full-text manifest entry"),
        (Phase.STAGE4_EXTRACTION_ROB, Phase.GATE3_EXTRACTION_APPROVAL,
         {"ft_missing_decisions": 2}, "lack full-text eligibility decisions"),
        (Phase.STAGE4_EXTRACTION_ROB, Phase.GATE3_EXTRACTION_APPROVAL,
         {"ft_unresolved_conflicts": 1}, "full-text eligibility conflict"),
        (Phase.STAGE4_EXTRACTION_ROB, Phase.GATE3_EXTRACTION_APPROVAL,
         {"ft_included": 0}, "no reports were included after full-text"),
        (Phase.STAGE4_EXTRACTION_ROB, Phase.GATE3_EXTRACTION_APPROVAL,
         {"included_reports_without_extraction": 1}, "have no extracted data"),
        (Phase.STAGE4_EXTRACTION_ROB, Phase.GATE3_EXTRACTION_APPROVAL,
         {"arms_for_non_included_reports": 1}, "not included at full-text"),
        (Phase.GATE3_EXTRACTION_APPROVAL, Phase.STAGE5_META_ANALYSIS,
         {"prisma_recorded": False}, "PRISMA counts have not been recorded"),
        (Phase.GATE3_EXTRACTION_APPROVAL, Phase.STAGE5_META_ANALYSIS,
         {"prisma_awaiting_screening": 1}, "PRISMA flow not final"),
    ],
)  # fmt: skip
def test_fail_closed_guards(
    phase: Phase, target: Phase, bad_context: dict[str, object], fragment: str
) -> None:
    m = PipelineStateMachine("p1", phase)
    ctx = dataclasses.replace(READY, **bad_context)  # type: ignore[arg-type]
    decision = approve(GATE_OF_PHASE[phase]) if phase in GATE_OF_PHASE else None
    actor_type = ActorType.HUMAN if decision else ActorType.SYSTEM
    with pytest.raises(GuardViolationError, match=fragment):
        m.transition(target, actor="x", actor_type=actor_type, context=ctx,
                     gate_decision=decision)  # fmt: skip
