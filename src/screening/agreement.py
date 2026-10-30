"""Inter-rater agreement and dual-review conflict detection (deterministic)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Sequence

from src.core.enums import ReviewerRole, ScreeningStage, ScreeningVerdict
from src.core.schemas.screening import AgreementStatistics, ScreeningConflict, ScreeningDecision


def cohens_kappa(rater_a: Sequence[Hashable], rater_b: Sequence[Hashable]) -> AgreementStatistics:
    """Cohen's unweighted kappa for two raters over the same items.

    κ = (p_o − p_e) / (1 − p_e), where p_o is observed agreement and p_e the
    agreement expected by chance from each rater's marginal proportions
    (Cohen J. Educ Psychol Meas 1960;20:37-46). κ is undefined when p_e = 1
    (both raters used one identical category for every item); ``kappa`` is then
    ``None`` rather than an arbitrary number.
    """
    if len(rater_a) != len(rater_b):
        raise ValueError("raters must label the same number of items")
    n = len(rater_a)
    if n == 0:
        raise ValueError("at least one item is required")
    p_o = sum(a == b for a, b in zip(rater_a, rater_b, strict=True)) / n
    ca, cb = Counter(rater_a), Counter(rater_b)
    p_e = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    kappa = None if p_e >= 1.0 else (p_o - p_e) / (1.0 - p_e)
    if kappa is not None:
        kappa = max(-1.0, min(1.0, kappa))  # guard float rounding at the bounds
    return AgreementStatistics(
        n_items=n, observed_agreement=p_o, expected_agreement=p_e, kappa=kappa
    )


def _pair_decisions(
    decisions: Sequence[ScreeningDecision], stage: ScreeningStage
) -> dict[str, tuple[ScreeningDecision, ScreeningDecision]]:
    r1: dict[str, ScreeningDecision] = {}
    r2: dict[str, ScreeningDecision] = {}
    for d in decisions:
        if d.stage != stage:
            continue
        target = {ReviewerRole.REVIEWER_1: r1, ReviewerRole.REVIEWER_2: r2}.get(d.reviewer)
        if target is None:
            continue
        if d.record_id in target:
            raise ValueError(f"duplicate {d.reviewer} decision for record {d.record_id}")
        target[d.record_id] = d
    missing = set(r1) ^ set(r2)
    if missing:
        raise ValueError(f"records lacking a decision from both reviewers: {sorted(missing)}")
    return {rid: (r1[rid], r2[rid]) for rid in sorted(r1)}


def agreement_for_stage(
    decisions: Sequence[ScreeningDecision], stage: ScreeningStage
) -> AgreementStatistics:
    pairs = _pair_decisions(decisions, stage)
    return cohens_kappa(
        [p[0].verdict for p in pairs.values()], [p[1].verdict for p in pairs.values()]
    )


def detect_conflicts(
    decisions: Sequence[ScreeningDecision], stage: ScreeningStage
) -> list[ScreeningConflict]:
    """Records needing human adjudication: verdicts differ, or both reviewers were uncertain."""
    conflicts: list[ScreeningConflict] = []
    for rid, (d1, d2) in _pair_decisions(decisions, stage).items():
        if d1.verdict != d2.verdict or d1.verdict == ScreeningVerdict.UNCERTAIN:
            conflicts.append(
                ScreeningConflict(
                    record_id=rid,
                    stage=stage,
                    reviewer_1_verdict=d1.verdict,
                    reviewer_2_verdict=d2.verdict,
                )
            )
    return conflicts


def final_verdicts(
    decisions: Sequence[ScreeningDecision],
    conflicts: Sequence[ScreeningConflict],
    stage: ScreeningStage,
) -> dict[str, ScreeningVerdict]:
    """Consensus verdict per record. Raises if any conflict is still unresolved."""
    resolved = {c.record_id: c for c in conflicts if c.stage == stage}
    out: dict[str, ScreeningVerdict] = {}
    for rid, (d1, d2) in _pair_decisions(decisions, stage).items():
        needs_human = d1.verdict != d2.verdict or d1.verdict == ScreeningVerdict.UNCERTAIN
        if not needs_human:
            out[rid] = d1.verdict
            continue
        conflict = resolved.get(rid)
        if conflict is None or conflict.resolved_verdict is None:
            raise ValueError(f"record {rid} has an unresolved screening conflict")
        out[rid] = conflict.resolved_verdict
    return out
