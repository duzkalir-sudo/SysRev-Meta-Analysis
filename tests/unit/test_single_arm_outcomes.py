"""Single-arm / case-series reviews and clinical outcomes (regression tests for the PR #1 review).

Examples are modelled on stereotactic radiosurgery reviews but all data are synthetic.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError
from sqlalchemy.orm import Session

from src.core.enums import (
    ArmRole,
    ClinicalOutcomeType,
    ComparisonType,
    EffectMeasure,
    EstimateType,
    ReviewDesign,
    StudyDesign,
    TimepointType,
)
from src.core.schemas import ExtractedStudyArm, ExtractedValue, FollowUpTimepoint, OutcomeDefinition
from src.state.repository import Repository
from tests.conftest import make_evidence, make_protocol, make_record

FIVE_YEARS = FollowUpTimepoint(kind=TimepointType.FIXED, value=5, unit="years")
TEN_YEARS = FollowUpTimepoint(kind=TimepointType.FIXED, value=10, unit="years")
LAST_FU = FollowUpTimepoint(kind=TimepointType.LAST_FOLLOW_UP)
WINDOW = FollowUpTimepoint(kind=TimepointType.WINDOW, window_min=24, window_max=36, unit="months")


def outcome(**kw: Any) -> OutcomeDefinition:
    data: dict[str, Any] = {
        "outcome_id": "tc_km",
        "name": "Tumour control",
        "kind": "primary",
        "clinical_type": ClinicalOutcomeType.TUMOR_CONTROL,
        "definition": "no radiological growth >2 mm",
        "data_type": "time_to_event",
        "comparison": ComparisonType.SINGLE_GROUP,
        "preferred_effect_measure": EffectMeasure.KM_PROBABILITY,
        "timepoints": [FIVE_YEARS, TEN_YEARS],
    }
    data.update(kw)
    return OutcomeDefinition(**data)


SRS_OUTCOMES = [
    outcome(),
    outcome(outcome_id="tc_crude", kind="secondary", data_type="dichotomous",
            preferred_effect_measure=EffectMeasure.PROPORTION, timepoints=[LAST_FU]),
    outcome(outcome_id="avm_oblit", name="AVM obliteration", kind="secondary",
            clinical_type=ClinicalOutcomeType.AVM_OBLITERATION, data_type="dichotomous",
            preferred_effect_measure=EffectMeasure.PROPORTION, timepoints=[WINDOW]),
    outcome(outcome_id="pain", name="Pain relief (BNI I-IIIa)", kind="secondary",
            clinical_type=ClinicalOutcomeType.PAIN_RELIEF, grading_system="BNI pain scale",
            data_type="dichotomous", preferred_effect_measure=EffectMeasure.PROPORTION,
            timepoints=[FollowUpTimepoint(kind=TimepointType.FIXED, value=12, unit="months")]),
    outcome(outcome_id="rn", name="Symptomatic radiation necrosis", kind="secondary",
            clinical_type=ClinicalOutcomeType.ADVERSE_EVENT, grading_system="CTCAE v5.0",
            preferred_effect_measure=EffectMeasure.CUMULATIVE_INCIDENCE,
            timepoints=[FollowUpTimepoint(kind=TimepointType.FIXED, value=2, unit="years")]),
    outcome(outcome_id="ae_rate", name="Adverse events", kind="secondary",
            clinical_type=ClinicalOutcomeType.ADVERSE_EVENT, data_type="count_rate",
            preferred_effect_measure=EffectMeasure.INCIDENCE_RATE, timepoints=[LAST_FU]),
    outcome(outcome_id="ttp", name="Time to progression", kind="secondary",
            preferred_effect_measure=EffectMeasure.MEDIAN_TIME_TO_EVENT, timepoints=[]),
]  # fmt: skip


def single_arm_protocol(**kw: Any) -> Any:
    data: dict[str, Any] = {
        "review_design": ReviewDesign.SINGLE_ARM,
        "comparator": None,
        "intervention": "Stereotactic radiosurgery",
        "population": "Adults with a synthetic benign skull-base tumour",
        "outcomes": SRS_OUTCOMES,
        "study_designs": [StudyDesign.CASE_SERIES, StudyDesign.RETROSPECTIVE_COHORT,
                          StudyDesign.SINGLE_ARM_TRIAL],
    }  # fmt: skip
    data.update(kw)
    return make_protocol(**data)


# --------------------------------------------------------------------------- protocol


def test_single_arm_protocol_without_comparator() -> None:
    p = single_arm_protocol()
    assert p.comparator is None
    assert p.review_design == ReviewDesign.SINGLE_ARM
    assert [o.outcome_id for o in p.outcomes if o.is_safety_outcome] == ["rn", "ae_rate"]


def test_comparative_review_still_requires_comparator() -> None:
    with pytest.raises(ValidationError, match="require a comparator"):
        make_protocol(comparator=None)
    with pytest.raises(ValidationError, match="require a comparator"):
        make_protocol(comparator="   ")


def test_single_arm_review_rejects_comparator_and_comparative_outcomes() -> None:
    with pytest.raises(ValidationError, match="must not define a comparator"):
        single_arm_protocol(comparator="Observation")
    with pytest.raises(ValidationError, match="cannot have comparative outcomes"):
        single_arm_protocol(outcomes=[*SRS_OUTCOMES, make_protocol().outcomes[0]])


def test_design_restrictions() -> None:
    with pytest.raises(ValidationError, match="cannot estimate single-group outcomes"):
        single_arm_protocol(study_designs=[StudyDesign.CASE_CONTROL])
    with pytest.raises(ValidationError, match="without a comparison group"):
        make_protocol(study_designs=[StudyDesign.RCT, StudyDesign.CASE_SERIES])


def test_comparative_review_may_report_single_group_safety_outcomes() -> None:
    p = make_protocol(outcomes=[*make_protocol().outcomes, SRS_OUTCOMES[4]])
    assert len(p.outcomes) == 2


# --------------------------------------------------------------------------- outcomes


@pytest.mark.parametrize(
    ("comparison", "data_type", "measure"),
    [
        (ComparisonType.SINGLE_GROUP, "dichotomous", EffectMeasure.RR),
        (ComparisonType.SINGLE_GROUP, "dichotomous", EffectMeasure.KM_PROBABILITY),
        (ComparisonType.SINGLE_GROUP, "time_to_event", EffectMeasure.PROPORTION),
        (ComparisonType.COMPARATIVE, "dichotomous", EffectMeasure.PROPORTION),
        (ComparisonType.COMPARATIVE, "time_to_event", EffectMeasure.KM_PROBABILITY),
    ],
)
def test_crude_and_time_to_event_measures_are_not_interchangeable(
    comparison: ComparisonType, data_type: str, measure: EffectMeasure
) -> None:
    with pytest.raises(ValidationError, match="is invalid for"):
        outcome(comparison=comparison, data_type=data_type, preferred_effect_measure=measure,
                timepoints=[FIVE_YEARS])  # fmt: skip


def test_single_group_outcome_needs_timepoint() -> None:
    with pytest.raises(ValidationError, match="at least one follow-up timepoint"):
        outcome(data_type="dichotomous", preferred_effect_measure=EffectMeasure.PROPORTION,
                timepoints=[])  # fmt: skip


@pytest.mark.parametrize("tp", [LAST_FU, WINDOW])
def test_km_and_cumulative_incidence_need_fixed_timepoints(tp: FollowUpTimepoint) -> None:
    for measure in (EffectMeasure.KM_PROBABILITY, EffectMeasure.CUMULATIVE_INCIDENCE):
        with pytest.raises(ValidationError, match="estimate at a time t"):
            outcome(preferred_effect_measure=measure, timepoints=[FIVE_YEARS, tp])


def test_timepoint_shapes_and_labels() -> None:
    assert FIVE_YEARS.label == "5 years"
    assert WINDOW.label == "24-36 months"
    assert LAST_FU.label == "last follow-up"
    with pytest.raises(ValidationError, match="value and unit"):
        FollowUpTimepoint(kind=TimepointType.FIXED, value=5)
    with pytest.raises(ValidationError, match="cannot have a window"):
        FollowUpTimepoint(kind=TimepointType.FIXED, value=5, unit="years", window_max=6)
    with pytest.raises(ValidationError, match="window_min must not exceed"):
        FollowUpTimepoint(kind=TimepointType.WINDOW, window_min=36, window_max=24, unit="months")
    with pytest.raises(ValidationError, match="fixed value"):
        FollowUpTimepoint(kind=TimepointType.WINDOW, window_min=1, window_max=2, unit="years",
                          value=1)  # fmt: skip
    with pytest.raises(ValidationError, match="no numeric time"):
        FollowUpTimepoint(kind=TimepointType.LAST_FOLLOW_UP, value=3)


# --------------------------------------------------------------------------- arms

KM_QUOTE = (
    "Actuarial tumour control was 92·3% (95% CI 88·1-95·0) at 5 years in 214 patients "
    "(41 at risk); median follow-up 62 months. Crude control at last follow-up: 198/214."
)


def v(text: str, value: float, unit: str | None = None) -> ExtractedValue:
    evidence = make_evidence(KM_QUOTE, location_identifier="Results ¶2")
    return ExtractedValue(value=value, reported_text=text, unit=unit, evidence=evidence)


def km_arm(**kw: Any) -> ExtractedStudyArm:
    data: dict[str, Any] = {
        "record_id": "rec1",
        "outcome_id": "tc_km",
        "arm_label": "SRS cohort",
        "arm_role": ArmRole.SINGLE_GROUP,
        "n_analyzed": v("214", 214),
        "estimate_type": EstimateType.KAPLAN_MEIER,
        "estimate_scale": "percent",
        "reported_estimate": v("92·3", 92.3, "%"),
        "ci_lower": v("88·1", 88.1, "%"),
        "ci_upper": v("95·0", 95.0, "%"),
        "ci_level": 95,
        "follow_up": FIVE_YEARS,
        "n_at_risk": v("41", 41),
        "follow_up_median": v("62", 62, "months"),
    }
    data.update(kw)
    return ExtractedStudyArm(**data)


def test_km_arm_is_distinct_from_crude_counts() -> None:
    arm = km_arm()
    assert arm.events is None  # the KM probability is not a count
    crude = ExtractedStudyArm(
        record_id="rec1", outcome_id="tc_crude", arm_label="SRS cohort",
        arm_role=ArmRole.SINGLE_GROUP, events=v("198", 198), n_analyzed=v("214", 214),
        follow_up=LAST_FU,
    )  # fmt: skip
    assert crude.reported_estimate is None and crude.estimate_type is None


def test_km_estimate_requires_fixed_follow_up() -> None:
    with pytest.raises(ValidationError, match="fixed timepoint"):
        km_arm(follow_up=LAST_FU)
    with pytest.raises(ValidationError, match="follow_up timepoint"):
        km_arm(follow_up=None)


def test_crude_proportion_estimate_may_use_last_follow_up() -> None:
    arm = km_arm(estimate_type=EstimateType.CRUDE_PROPORTION, follow_up=LAST_FU,
                 ci_lower=None, ci_upper=None, ci_level=None, n_at_risk=None)  # fmt: skip
    assert arm.estimate_type == EstimateType.CRUDE_PROPORTION


@pytest.mark.parametrize(
    ("override", "fragment"),
    [
        ({"estimate_type": None}, "given together"),
        ({"estimate_scale": None}, "need estimate_scale"),
        ({"estimate_scale": "proportion"}, "outside 0-1"),
        ({"ci_level": None}, "ci_level is required"),
        ({"ci_upper": None}, "both bounds"),
        ({"ci_lower": v("95·0", 95.0)}, "within its confidence interval"),
        ({"n_at_risk": v("214", 214), "n_analyzed": v("41", 41)}, "n_at_risk cannot exceed"),
    ],
)
def test_reported_estimate_validation(override: dict[str, Any], fragment: str) -> None:
    with pytest.raises(ValidationError, match=fragment):
        km_arm(**override)


def test_rate_and_median_time_estimates_need_units_and_no_scale() -> None:
    base = {"ci_lower": None, "ci_upper": None, "ci_level": None, "n_at_risk": None}
    with pytest.raises(ValidationError, match="does not apply"):
        km_arm(estimate_type=EstimateType.INCIDENCE_RATE, follow_up=LAST_FU, **base)
    with pytest.raises(ValidationError, match="need a unit"):
        km_arm(estimate_type=EstimateType.MEDIAN_TIME_TO_EVENT, estimate_scale=None,
               reported_estimate=v("62", 62), follow_up=None, **base)  # fmt: skip
    ok = km_arm(estimate_type=EstimateType.MEDIAN_TIME_TO_EVENT, estimate_scale=None,
                reported_estimate=v("62", 62, "months"), follow_up=None, **base)  # fmt: skip
    assert ok.reported_estimate is not None


def test_single_arm_extraction_round_trips_through_database(session: Session) -> None:
    repo = Repository(session)
    repo.create_project("p1", "SRS review")
    repo.save_records("p1", [make_record("rec1", "Synthetic SRS case series")])
    arm = km_arm()
    repo.save_extracted_arm("p1", arm)
    [loaded] = repo.load_extracted_arms("p1")
    assert loaded == arm
    assert loaded.follow_up == FIVE_YEARS
    assert loaded.estimate_type == EstimateType.KAPLAN_MEIER
