"""Builders for Stage 5 tests. All study data are SYNTHETIC (invented for testing)."""

from __future__ import annotations

from typing import Any

from src.core.enums import (
    ArmRole,
    ClinicalOutcomeType,
    ComparisonType,
    EffectMeasure,
    EstimateType,
    TimepointType,
)
from src.core.schemas import ExtractedStudyArm, ExtractedValue, OutcomeDefinition
from src.core.schemas.extraction import parse_reported_number
from src.core.schemas.protocol import FollowUpTimepoint
from src.core.schemas.stats import AnalysisPlan, AnalysisSpec
from src.stats_engine.inputs import AnalysisInputs
from tests.conftest import NOW, make_evidence

REVIEWER = "Dr Stats Reviewer"

YEARS_5 = FollowUpTimepoint(kind=TimepointType.FIXED, value=5, unit="years")
YEARS_2 = FollowUpTimepoint(kind=TimepointType.FIXED, value=2, unit="years")
MONTHS_24 = FollowUpTimepoint(kind=TimepointType.FIXED, value=24, unit="months")
LAST_FU = FollowUpTimepoint(kind=TimepointType.LAST_FOLLOW_UP)


def stats_arm(
    record_id: str,
    outcome_id: str,
    *,
    role: ArmRole = ArmRole.SINGLE_GROUP,
    label: str = "SRS cohort",
    follow_up: FollowUpTimepoint | None = None,
    approved: bool = True,
    estimate_type: EstimateType | None = None,
    estimate_scale: str | None = None,
    ci_level: float | None = None,
    units: dict[str, str] | None = None,
    **numbers: str,
) -> ExtractedStudyArm:
    """An arm whose numbers are all quoted verbatim in one synthetic evidence passage."""
    quote = f"{label}: " + "; ".join(f"{k} {v}" for k, v in numbers.items())
    ev = make_evidence(quote)
    fields: dict[str, Any] = {
        k: ExtractedValue(
            value=parse_reported_number(v), reported_text=v, unit=(units or {}).get(k), evidence=ev
        )
        for k, v in numbers.items()
    }
    return ExtractedStudyArm(
        record_id=record_id,
        outcome_id=outcome_id,
        arm_label=label,
        arm_role=role,
        follow_up=follow_up,
        estimate_type=estimate_type,
        estimate_scale=estimate_scale,
        ci_level=ci_level,
        approved_by_human=approved,
        approved_by=REVIEWER if approved else None,
        approved_at=NOW if approved else None,
        **fields,
    )


def km_arm(
    record_id: str,
    est: str,
    lo: str | None,
    hi: str | None,
    *,
    outcome_id: str = "control_km",
    follow_up: FollowUpTimepoint = YEARS_5,
    estimate_type: EstimateType = EstimateType.KAPLAN_MEIER,
) -> ExtractedStudyArm:
    nums = {"reported_estimate": est}
    if lo is not None and hi is not None:
        nums |= {"ci_lower": lo, "ci_upper": hi}
    return stats_arm(
        record_id,
        outcome_id,
        follow_up=follow_up,
        estimate_type=estimate_type,
        estimate_scale="percent",
        ci_level=95 if lo is not None else None,
        **nums,
    )


OUTCOMES = {
    "control_crude": OutcomeDefinition(
        outcome_id="control_crude",
        name="Local tumour control (crude)",
        kind="primary",
        clinical_type=ClinicalOutcomeType.TUMOR_CONTROL,
        data_type="dichotomous",
        comparison=ComparisonType.SINGLE_GROUP,
        preferred_effect_measure=EffectMeasure.PROPORTION,
        timepoints=[LAST_FU, MONTHS_24],
    ),
    "necrosis": OutcomeDefinition(
        outcome_id="necrosis",
        name="Radiation necrosis (crude)",
        kind="secondary",
        clinical_type=ClinicalOutcomeType.ADVERSE_EVENT,
        data_type="dichotomous",
        comparison=ComparisonType.SINGLE_GROUP,
        preferred_effect_measure=EffectMeasure.PROPORTION,
        timepoints=[LAST_FU],
    ),
    "control_km": OutcomeDefinition(
        outcome_id="control_km",
        name="Local control (Kaplan-Meier)",
        kind="primary",
        data_type="time_to_event",
        comparison=ComparisonType.SINGLE_GROUP,
        preferred_effect_measure=EffectMeasure.KM_PROBABILITY,
        timepoints=[YEARS_5, YEARS_2],
    ),
    "necrosis_cif": OutcomeDefinition(
        outcome_id="necrosis_cif",
        name="Radiation necrosis (cumulative incidence)",
        kind="secondary",
        data_type="time_to_event",
        comparison=ComparisonType.SINGLE_GROUP,
        preferred_effect_measure=EffectMeasure.CUMULATIVE_INCIDENCE,
        timepoints=[YEARS_2],
    ),
    "complication": OutcomeDefinition(
        outcome_id="complication",
        name="Any complication",
        kind="primary",
        data_type="dichotomous",
        preferred_effect_measure=EffectMeasure.RR,
    ),
    "score": OutcomeDefinition(
        outcome_id="score",
        name="Functional score",
        kind="secondary",
        data_type="continuous",
        preferred_effect_measure=EffectMeasure.MD,
    ),
    "survival": OutcomeDefinition(
        outcome_id="survival",
        name="Overall survival",
        kind="secondary",
        data_type="time_to_event",
        preferred_effect_measure=EffectMeasure.HR,
    ),
}


def inputs(arms: list[ExtractedStudyArm], **kw: Any) -> AnalysisInputs:
    rids = sorted({a.record_id for a in arms})
    data: dict[str, Any] = {
        "outcomes": OUTCOMES,
        "arms": arms,
        "study_of": {r: r for r in rids},
        "cohort_of": {r: r for r in rids},
        "labels": {r: f"Synthetic {r}" for r in rids},
    }
    data.update(kw)
    return AnalysisInputs(**data)


def plan(
    *specs: AnalysisSpec, approved: bool = True, reviewer_id: str = "rvw1", name: str = REVIEWER
) -> AnalysisPlan:
    if approved:
        return AnalysisPlan(
            analyses=list(specs),
            drafted_by=name,
            status="approved",
            approved_by=name,
            approved_reviewer_id=reviewer_id,
            approved_at=NOW,
        )
    return AnalysisPlan(analyses=list(specs), drafted_by=name)


# Synthetic SRS-style crude-proportion data with zero-event studies
NECROSIS = [
    ("s1", "0", "25"),
    ("s2", "3", "40"),
    ("s3", "1", "18"),
    ("s4", "0", "12"),
    ("s5", "5", "60"),
    ("s6", "2", "33"),
]


def necrosis_arms() -> list[ExtractedStudyArm]:
    return [
        stats_arm(r, "necrosis", follow_up=LAST_FU, events=x, n_analyzed=n) for r, x, n in NECROSIS
    ]
