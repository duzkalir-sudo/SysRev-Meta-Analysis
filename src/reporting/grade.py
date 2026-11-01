"""Outcome-level GRADE support: question type, applicability, decision support, input hashes.

What this module does **not** do: decide any domain judgement or the final certainty.
Statistical results (confidence intervals, I², study and participant counts) and approved
risk-of-bias assessments are displayed as *decision support* only. The certainty implied by
the human's domain judgements is computed mechanically (initial level − downgrades +
upgrades, bounded by very low and high; :class:`GradeAssessment.implied_certainty`), and a
final rating that departs from it must be explained by the reviewer.

Starting certainty is not universal (GRADE guidelines 18, Schünemann HJ, et al. J Clin
Epidemiol 2019;111:105-114): randomised trials start high; non-randomised studies of
interventions start high when risk of bias is assessed with ROBINS-I and low when it is
assessed with other tools. A single-group estimate (crude proportion, Kaplan-Meier or
cumulative incidence from case series or single-arm studies) is not an intervention effect;
it can be rated only with the GRADE approach for overall prognosis (Iorio A, et al. BMJ
2015;350:h870) when the reviewer documents why that framework fits, and records the initial
level and its justification. Otherwise its certainty stays *not assessed*, and is labelled so.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from src.core.enums import EffectMeasure, RoBTool
from src.core.schemas.extraction import RiskOfBiasAssessment
from src.core.schemas.protocol import OutcomeDefinition
from src.core.schemas.reporting import (
    APPROACH_LABEL,
    APPROACH_QUESTION,
    DOMAIN_LABEL,
    LEVEL_LABEL,
    UPGRADE_LABEL,
    GradeAssessment,
    QuestionType,
)
from src.core.schemas.stats import AnalysisResult, canonical_sha256

SINGLE_GROUP = frozenset(
    {EffectMeasure.PROPORTION, EffectMeasure.KM_PROBABILITY, EffectMeasure.CUMULATIVE_INCIDENCE}
)
COMPARATIVE = frozenset({EffectMeasure.RR, EffectMeasure.OR, EffectMeasure.MD, EffectMeasure.SMD})
ESTIMATED = frozenset({"pooled", "single_study"})
APPROACH_TOOLS: dict[str, frozenset[RoBTool]] = {
    "interventions_randomised": frozenset({RoBTool.ROB2}),
    "interventions_nrsi_robins_i": frozenset({RoBTool.ROBINS_I}),
    "interventions_nonrandomised_other": frozenset({RoBTool.NOS, RoBTool.EXTERNAL}),
}


def question_type(a: AnalysisResult) -> QuestionType:
    if a.effect_measure in SINGLE_GROUP:
        return "single_group_estimate"
    if a.effect_measure in COMPARATIVE:
        return "comparative_intervention_effect"
    # HR, RD, IRR, rates, means: the engine reports them as unsupported; treat by comparison.
    return (
        "comparative_intervention_effect"
        if a.effect_measure in {EffectMeasure.HR, EffectMeasure.RD, EffectMeasure.IRR}
        else "single_group_estimate"
    )


def contributing_records(a: AnalysisResult) -> list[str]:
    """Reports behind the estimate that GRADE rates: the primary model's population
    (Session 9 review; an auxiliary model can use other studies, e.g. the Mantel-Haenszel
    risk ratio's tables without a study-level estimate). Engine 1.0.0 outputs: every study."""
    return list(a.population().record_ids)


def latest_approved_rob(
    assessments: list[RiskOfBiasAssessment], record_ids: list[str]
) -> dict[str, RiskOfBiasAssessment]:
    """The most recently approved assessment per report (approval time, then id)."""
    out: dict[str, RiskOfBiasAssessment] = {}
    approved = [x for x in assessments if x.approved_by_human and x.record_id in record_ids]
    for x in sorted(approved, key=lambda x: (x.approved_at, x.assessment_id)):
        out[x.record_id] = x
    return out


def inputs_sha256(
    a: AnalysisResult,
    outcome: OutcomeDefinition | None,
    review_design: str,
    rob: list[RiskOfBiasAssessment],
    study_of: dict[str, str],
) -> str:
    """Everything an assessment for analysis ``a`` depends on. Any change (the analysis
    result, the outcome definition, the contributing studies' approved risk of bias or their
    report-to-study links) makes an earlier assessment stale."""
    rids = contributing_records(a)
    return canonical_sha256(
        {
            "analysis": a.model_dump(mode="json"),
            "outcome": outcome.model_dump(mode="json") if outcome else None,
            "review_design": review_design,
            # every approved assessment of a contributing report: any new approval counts
            "rob": sorted(
                (x.model_dump(mode="json") for x in rob if x.record_id in rids),
                key=lambda d: (d["record_id"], d["assessment_id"]),
            ),
            "study_links": {r: study_of.get(r, r) for r in rids},
        }
    )


def applicability_problems(
    approach: str | None,
    a: AnalysisResult,
    rob: dict[str, RiskOfBiasAssessment],
) -> list[str]:
    """Why ``approach`` cannot be applied to analysis ``a`` (empty list: applicable)."""
    problems: list[str] = []
    qt = question_type(a)
    if a.status not in ESTIMATED:
        problems.append(
            f"the analysis has no estimate (status {a.status}: {a.status_reason}); a GRADE "
            "rating of a statistical result needs one — record the outcome as not assessed, "
            "or describe the certainty of any synthesis without meta-analysis in that synthesis"
        )
    if approach is None:
        return problems
    if APPROACH_QUESTION[approach] != qt:
        problems.append(
            f"{APPROACH_LABEL[approach]} does not apply to a {qt.replace('_', ' ')} "
            f"({a.effect_measure.value}); a single-group estimate is not an intervention "
            "effect, and a comparative effect is not a prognosis"
        )
    tools = APPROACH_TOOLS.get(approach)
    if tools is not None:
        rids = contributing_records(a)
        missing = [r for r in rids if r not in rob]
        if missing:
            problems.append(
                "no approved risk-of-bias assessment for contributing report(s) "
                + ", ".join(missing)
            )
        other = sorted({rob[r].tool.value for r in rids if r in rob and rob[r].tool not in tools})
        if other:
            problems.append(
                f"{APPROACH_LABEL[approach]} requires contributing studies assessed with "
                f"{'/'.join(sorted(t.value for t in tools))}, but tool(s) {other} were used; "
                "randomised and non-randomised bodies of evidence are rated separately, each "
                "with its own starting level"
            )
    return problems


def decision_support(a: AnalysisResult, rob: dict[str, RiskOfBiasAssessment]) -> dict[str, Any]:
    """Information shown next to each domain. It never pre-fills a judgement."""
    rids = contributing_records(a)
    overall = Counter(
        (rob[r].overall or "not stated") if r in rob else "not assessed" for r in rids
    )
    r = a.primary
    est: dict[str, Any] = {}
    if r is not None:
        est = {
            "estimate": r.estimate_natural,
            "ci_lower": r.ci_lower_natural,
            "ci_upper": r.ci_upper_natural,
            "model": r.model,
            "method": r.method,
            "model_label": r.label,
        }
    elif a.status == "single_study" and a.studies:
        s = a.studies[0]
        est = {
            "estimate": s.estimate_natural,
            "ci_lower": s.ci_lower_natural,
            "ci_upper": s.ci_upper_natural,
            "model": "single study (not pooled)",
        }
    null = (
        1.0
        if a.effect_measure in {EffectMeasure.RR, EffectMeasure.OR}
        else (0.0 if a.effect_measure in {EffectMeasure.MD, EffectMeasure.SMD} else None)
    )
    crosses = None
    if null is not None and est.get("ci_lower") is not None and est.get("ci_upper") is not None:
        crosses = est["ci_lower"] <= null <= est["ci_upper"]
    h = a.heterogeneity
    pi = a.random_effects.prediction_interval_natural if a.random_effects else None
    # Counts shown with the estimate describe the primary model (Session 9 review); other
    # reported models with a different population stay visible, labelled as auxiliary.
    pop = a.population()
    auxiliary = [
        {
            "model": r.label,
            "role": "auxiliary (not the estimate rated here)",
            "estimate": r.estimate_natural,
            "ci_lower": r.ci_lower_natural,
            "ci_upper": r.ci_upper_natural,
            "k": aux.k,
            "participants": aux.total_participants,
            "events": aux.total_events,
        }
        for r, aux in a.auxiliary_populations()
    ]
    return {
        "analysis_id": a.analysis_id,
        "outcome": a.outcome_name,
        "measure": a.effect_measure.value,
        "question_type": question_type(a),
        "status": a.status,
        "k": pop.k,
        "participants": pop.total_participants,
        "events": pop.total_events,
        "population_model": a.primary.label if a.primary is not None else "all listed studies",
        "auxiliary_models": auxiliary,
        "estimate": est,
        "domains": {
            "risk_of_bias": {
                "overall_judgements": dict(sorted(overall.items())),
                "tools": sorted({rob[x].tool.value for x in rids if x in rob}),
            },
            "inconsistency": (
                {
                    "i2": h.i2,
                    "q": h.q,
                    "df": h.df,
                    "p_value": h.p_value,
                    "tau2": a.random_effects.tau2 if a.random_effects else None,
                    "prediction_interval": list(pi) if pi else None,
                }
                if h is not None
                else {
                    "note": "fewer than two studies: statistical heterogeneity cannot be "
                    "estimated; judge consistency from the study results"
                }
            ),
            "indirectness": {
                "note": "no automated information: compare the populations, interventions, "
                "comparators and outcome measurement of the contributing studies with the "
                "review question"
            },
            "imprecision": {
                "ci_includes_null": crosses,
                "null_value": null,
                "participants": pop.total_participants,
                "events": pop.total_events,
                "note": "no minimally important difference, decision threshold or optimal "
                "information size is recorded in this project, and none is assumed",
            },
            "publication_bias": {
                "k": pop.k,
                "note": "no test for small-study effects or funnel-plot asymmetry was "
                "performed by the statistics engine; the absence of a test, or a "
                "non-significant one, is not evidence that publication bias is absent",
            },
        },
        "warnings": list(a.warnings) + list(a.limitations),
        "disclaimer": "Decision support only: no value here determines a GRADE judgement.",
    }


def certainty_statement(g: GradeAssessment, outcome_name: str) -> str:
    """Journal text for an approved assessment (no digits; every word from human inputs)."""
    assert g.final_certainty is not None
    assert g.approach is not None
    down = [f"{DOMAIN_LABEL[d.domain]} ({d.concern.replace('_', ' ')})" for d in g.rated_down]
    up = [UPGRADE_LABEL[u.factor] for u in g.upgrades]
    reasons: list[str] = []
    if down:
        reasons.append("rated down for " + _join(down))
    if up:
        reasons.append("rated up for " + _join(up))
    if not reasons:
        reasons.append("not rated down or up")
    scope = (
        "; rated as an event-rate (prognosis) question with the GRADE approach for overall "
        "prognosis, not as an intervention effect"
        if g.approach == "overall_prognosis"
        else ""
    )
    return (
        f"The certainty of the evidence for {outcome_name} was {LEVEL_LABEL[g.final_certainty]} "
        f"(GRADE; {'; '.join(reasons)}{scope})"
    )


def reasons_text(g: GradeAssessment) -> str:
    parts = [f"{DOMAIN_LABEL[d.domain]}: {d.rationale.rstrip('.')}" for d in g.rated_down]
    parts += [f"{UPGRADE_LABEL[u.factor]}: {u.rationale.rstrip('.')}" for u in g.upgrades]
    if g.departure_rationale:
        parts.append(f"overall judgement: {g.departure_rationale.rstrip('.')}")
    return "; ".join(parts) or "no domain was rated down"


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]
