"""Stage 5 engine: input integrity, estimator/timepoint/cohort safety, statuses, outputs.

All data are SYNTHETIC.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.core.enums import ArmRole, EstimateType
from src.core.exceptions import StatisticsInputError
from src.core.schemas.master import CohortContributionDecision, QuantitativeUnavailability
from src.core.schemas.stats import AnalysisSpec, ArmSelection, ReportAccount, StatsOutput
from src.stats_engine import effect_sizes as es
from src.stats_engine.engine import run_analysis, run_plan
from src.stats_engine.glmm import glmm_logit
from src.stats_engine.report import analysis_csv, analysis_markdown, forest_plot, report_markdown
from tests.conftest import NOW
from tests.stats_helpers import (
    LAST_FU,
    MONTHS_24,
    NECROSIS,
    OUTCOMES,
    YEARS_2,
    YEARS_5,
    inputs,
    km_arm,
    necrosis_arms,
    plan,
    stats_arm,
)

NEC = AnalysisSpec(
    analysis_id="nec", outcome_id="necrosis", effect_measure="PROPORTION", timepoints=[LAST_FU]
)
KM = AnalysisSpec(
    analysis_id="km5",
    outcome_id="control_km",
    effect_measure="KM_PROBABILITY",
    timepoints=[YEARS_5],
    reported_ci_scale="log_log",
)


def run(spec: AnalysisSpec, arms: list, **kw: object):  # type: ignore[no-untyped-def]
    return run_analysis(spec, OUTCOMES[spec.outcome_id], inputs(arms, **kw))


# --------------------------------------------------------------------------- proportions


def test_proportion_glmm_primary_with_zero_event_studies() -> None:
    res = run(NEC, necrosis_arms())
    assert res.status == "pooled" and res.k == 6
    g = glmm_logit([int(x) for _, x, _ in NECROSIS], [int(n) for _, _, n in NECROSIS])
    assert res.random_effects is not None and res.primary == res.random_effects
    assert res.random_effects.estimate == pytest.approx(g.mu, abs=1e-12)
    assert res.random_effects.estimate_natural == pytest.approx(es.expit(g.mu))
    assert "GLMM" in res.random_effects.method
    # inverse-variance logit model with continuity correction is the sensitivity analysis
    assert any("continuity" in s.method for s in res.sensitivity)
    zero = [s for s in res.studies if s.data["events"] == 0]
    assert len(zero) == 2 and all(s.continuity_corrected for s in zero)
    assert all(s.ci_lower_natural == 0.0 for s in zero)  # Clopper-Pearson lower bound
    assert res.total_events == 11 and res.total_participants == 188
    assert res.heterogeneity is not None and res.heterogeneity.df == 5


def test_proportion_logit_iv_primary_swaps_roles() -> None:
    spec = NEC.model_copy(update={"proportion_method": "logit_inverse_variance"})
    res = run(spec, necrosis_arms())
    assert res.random_effects is not None and "continuity" in res.random_effects.method
    assert any("GLMM" in s.method for s in res.sensitivity)


def test_single_study_is_not_pooled() -> None:
    res = run(NEC, necrosis_arms()[:1])
    assert res.status == "single_study" and res.k == 1
    assert res.primary is None and res.heterogeneity is None
    assert res.studies[0].estimate_natural == 0.0


def test_all_zero_events_is_descriptive_only() -> None:
    arms = [
        stats_arm(f"z{i}", "necrosis", follow_up=LAST_FU, events="0", n_analyzed=n)
        for i, n in enumerate(("10", "20", "15"))
    ]
    res = run(NEC, arms)
    assert res.status == "descriptive_only" and res.primary is None
    assert res.total_events == 0 and res.total_participants == 45


def test_all_event_studies_pool_with_glmm() -> None:
    data = [("a", "45", "45"), ("b", "30", "31"), ("c", "58", "60"), ("d", "22", "22")]
    spec = AnalysisSpec(
        analysis_id="ctl",
        outcome_id="control_crude",
        effect_measure="PROPORTION",
        timepoints=[LAST_FU],
    )
    arms = [
        stats_arm(r, "control_crude", follow_up=LAST_FU, events=x, n_analyzed=n) for r, x, n in data
    ]
    res = run(spec, arms)
    assert res.status == "pooled"
    assert res.primary is not None and 0.9 < res.primary.estimate_natural < 1.0


def test_unapproved_arm_aborts() -> None:
    arms = necrosis_arms()
    arms.append(
        stats_arm("s7", "necrosis", follow_up=LAST_FU, events="1", n_analyzed="9", approved=False)
    )
    with pytest.raises(StatisticsInputError, match="not human-approved"):
        run(NEC, arms)


def test_missing_denominator_not_substituted() -> None:
    no_n = stats_arm("s9", "necrosis", follow_up=LAST_FU, events="2", n_randomized="30")
    arms = [*necrosis_arms()[:2], no_n]
    res = run(NEC, arms)
    ex = {e.study_id: e.reason_code for e in res.excluded}
    assert ex == {"s9": "missing_data"} and res.k == 2


# --------------------------------------------------------------------------- estimators & time


def test_km_estimate_never_used_as_crude_proportion() -> None:
    km = stats_arm(
        "k1",
        "necrosis",
        follow_up=YEARS_2,
        estimate_type=EstimateType.KAPLAN_MEIER,
        estimate_scale="percent",
        reported_estimate="7.5",
    )
    arms = [*necrosis_arms()[:2], km]
    res = run(NEC, arms)
    assert {e.study_id: e.reason_code for e in res.excluded} == {"k1": "incompatible_estimator"}


def test_crude_counts_never_used_for_km_analysis() -> None:
    arms = [
        km_arm("a", "92.3", "87.7", "95.2"),
        km_arm("b", "85.0", "78.6", "89.6"),
        stats_arm("c", "control_km", follow_up=YEARS_5, events="40", n_analyzed="44"),
    ]
    res = run(KM, arms)
    assert {e.study_id: e.reason_code for e in res.excluded} == {"c": "incompatible_estimator"}
    assert res.status == "pooled" and res.k == 2


def test_timepoints_never_converted_or_mixed() -> None:
    spec = AnalysisSpec(
        analysis_id="c24",
        outcome_id="control_crude",
        effect_measure="PROPORTION",
        timepoints=[MONTHS_24],
    )
    two_years = YEARS_2  # same duration, different unit: not converted
    arms = [
        stats_arm("a", "control_crude", follow_up=MONTHS_24, events="40", n_analyzed="44"),
        stats_arm("b", "control_crude", follow_up=MONTHS_24, events="30", n_analyzed="35"),
        stats_arm("c", "control_crude", follow_up=two_years, events="20", n_analyzed="22"),
        stats_arm("d", "control_crude", events="20", n_analyzed="22"),
    ]
    res = run(spec, arms)
    codes = {e.study_id: e.reason_code for e in res.excluded}
    assert codes == {"c": "timepoint_mismatch", "d": "timepoint_missing"}
    assert res.k == 2


def test_plan_timepoints_must_be_prespecified_and_justified() -> None:
    with pytest.raises(ValidationError, match="timepoint_justification"):
        AnalysisSpec(
            analysis_id="x",
            outcome_id="control_km",
            effect_measure="KM_PROBABILITY",
            timepoints=[YEARS_5, YEARS_2],
            reported_ci_scale="log_log",
        )
    spec = AnalysisSpec(
        analysis_id="x", outcome_id="necrosis", effect_measure="PROPORTION", timepoints=[YEARS_5]
    )
    with pytest.raises(StatisticsInputError, match="not pre-specified"):
        run(spec, necrosis_arms())
    bad = AnalysisSpec(analysis_id="x", outcome_id="necrosis", effect_measure="RR")
    with pytest.raises(StatisticsInputError, match="not admissible"):
        run(bad, necrosis_arms())


def test_multi_timepoint_reports_need_arm_selection() -> None:
    spec = AnalysisSpec(
        analysis_id="both",
        outcome_id="control_crude",
        effect_measure="PROPORTION",
        timepoints=[LAST_FU, MONTHS_24],
        timepoint_justification="synthetic test",
    )
    arms = [
        stats_arm("a", "control_crude", follow_up=MONTHS_24, events="40", n_analyzed="44"),
        stats_arm(
            "a",
            "control_crude",
            label="SRS cohort (last FU)",
            follow_up=LAST_FU,
            events="38",
            n_analyzed="44",
        ),
        stats_arm("b", "control_crude", follow_up=LAST_FU, events="30", n_analyzed="35"),
    ]
    res = run(spec, arms)
    assert {e.study_id: e.reason_code for e in res.excluded} == {"a": "arm_ambiguous"}
    chosen = spec.model_copy(
        update={
            "arm_selections": [
                ArmSelection(record_id="a", role="single_group", arm_label="SRS cohort")
            ]
        }
    )
    res2 = run(chosen, arms)
    assert res2.k == 2 and not res2.excluded


# --------------------------------------------------------------------------- KM / CIF


def test_km_pooling_and_conservative_exclusions() -> None:
    arms = [
        km_arm("a", "92.3", "87.7", "95.2"),
        km_arm("b", "85.0", "78.6", "89.6"),
        km_arm("c", "88.6", "83.6", "92.1"),
        km_arm("plain", "90.0", "84.0", "96.0"),  # symmetric on the probability scale only
        km_arm("full", "100", "88.0", "100"),  # boundary estimate
        km_arm("noci", "91.0", None, None),
        km_arm("act", "89.0", "83.1", "93.0", estimate_type=EstimateType.ACTUARIAL_LIFE_TABLE),
    ]
    res = run(KM, arms)
    codes = {e.study_id: e.reason_code for e in res.excluded}
    assert codes == {
        "plain": "ci_inconsistent",
        "full": "boundary_estimate",
        "noci": "missing_data",
        "act": "incompatible_estimator",
    }
    assert res.status == "pooled" and res.k == 3
    assert res.scale == "log_log"
    assert res.primary is not None and 0.85 < res.primary.estimate_natural < 0.93
    lo, hi = res.primary.ci_lower_natural, res.primary.ci_upper_natural
    assert lo < res.primary.estimate_natural < hi
    # the study SE is recovered from the reported CI on the declared scale
    s = next(x for x in res.studies if x.study_id == "a")
    z = 1.959963984540054
    g = lambda p: math.log(-math.log(p))  # noqa: E731
    assert s.vi == pytest.approx(((g(0.877) - g(0.952)) / (2 * z)) ** 2, rel=1e-12)


def test_km_and_actuarial_combined_only_with_justification() -> None:
    with pytest.raises(ValidationError, match="reason"):
        KM.model_copy(update={"combine_km_and_actuarial": True}).model_validate(
            KM.model_dump() | {"combine_km_and_actuarial": True}
        )
    spec = AnalysisSpec.model_validate(
        KM.model_dump()
        | {"combine_km_and_actuarial": True, "combine_justification": "same estimand (synthetic)"}
    )
    arms = [
        km_arm("a", "92.3", "87.7", "95.2"),
        km_arm("act", "89.0", "83.1", "93.0", estimate_type=EstimateType.ACTUARIAL_LIFE_TABLE),
    ]
    res = run(spec, arms)
    assert res.k == 2 and not res.excluded


def test_km_requires_declared_ci_scale() -> None:
    with pytest.raises(ValidationError, match="reported_ci_scale"):
        AnalysisSpec(
            analysis_id="x",
            outcome_id="control_km",
            effect_measure="KM_PROBABILITY",
            timepoints=[YEARS_5],
        )


def test_cumulative_incidence_kept_apart_from_km() -> None:
    spec = AnalysisSpec(
        analysis_id="cif",
        outcome_id="necrosis_cif",
        effect_measure="CUMULATIVE_INCIDENCE",
        timepoints=[YEARS_2],
        reported_ci_scale="log_log",
    )
    arms = [
        km_arm(
            "a",
            "7.7",
            "5.5",
            "10.3",
            outcome_id="necrosis_cif",
            follow_up=YEARS_2,
            estimate_type=EstimateType.CUMULATIVE_INCIDENCE,
        ),
        km_arm(
            "b",
            "9.0",
            "4.0",
            "15.0",
            outcome_id="necrosis_cif",
            follow_up=YEARS_2,
            estimate_type=EstimateType.KAPLAN_MEIER,
        ),
    ]
    res = run(spec, arms)
    assert {e.study_id: e.reason_code for e in res.excluded} == {"b": "incompatible_estimator"}
    assert res.status == "single_study"


# --------------------------------------------------------------------------- cohorts


def test_overlapping_reports_excluded_unless_human_declared_distinct() -> None:
    later = stats_arm("s1b", "necrosis", follow_up=LAST_FU, events="1", n_analyzed="30")
    arms = [*necrosis_arms()[:3], later]
    cohort = {"s1": "cohortA", "s1b": "cohortA", "s2": "s2", "s3": "s3"}
    res = run(NEC, arms, cohort_of=cohort)
    ex = [e for e in res.excluded if e.reason_code == "overlapping_cohort"]
    assert len(ex) == 1 and ex[0].record_ids == ["s1", "s1b"] and res.k == 2
    decision = CohortContributionDecision(
        study_id="cohortA",
        outcome_id="necrosis",
        record_ids=["s1", "s1b"],
        justification="different recruitment years (synthetic)",
        decided_by="Dr Stats Reviewer",
        reviewer_id="rvw1",
    )
    res2 = run(NEC, arms, cohort_of=cohort, contributions=[decision])
    assert res2.k == 4 and not res2.excluded
    assert any("non-overlapping" in w for w in res2.warnings)


def test_recorded_unavailability_excludes_with_reason() -> None:
    u = QuantitativeUnavailability(
        record_id="s2",
        outcome_id="necrosis",
        reason_code="overlapping_cohort",
        reason="follow-up report",
        recorded_by="Dr Stats Reviewer",
        reviewer_id="rvw1",
    )
    res = run(NEC, necrosis_arms(), unavailability=[u])
    e = next(e for e in res.excluded if e.study_id == "s2")
    assert e.reason_code == "recorded_unavailable" and "follow-up report" in e.reason


# --------------------------------------------------------------------------- comparative


def two_arm(rid: str, outcome: str, i: dict[str, str], c: dict[str, str], **kw: object) -> list:  # type: ignore[type-arg]
    return [
        stats_arm(rid, outcome, role=ArmRole.INTERVENTION, label="SRS", **kw, **i),  # type: ignore[arg-type]
        stats_arm(rid, outcome, role=ArmRole.CONTROL, label="Observation", **kw, **c),
    ]  # type: ignore[arg-type]


def rr_arms() -> list:  # type: ignore[type-arg]
    rows = [
        ("t1", "2", "40", "6", "41"),
        ("t2", "0", "25", "3", "24"),
        ("t3", "0", "15", "0", "16"),
        ("t4", "5", "60", "9", "58"),
    ]
    out = []
    for rid, a, n1, c, n2 in rows:
        out += two_arm(
            rid, "complication", {"events": a, "n_analyzed": n1}, {"events": c, "n_analyzed": n2}
        )
    return out


def test_rr_mantel_haenszel_and_zero_cells() -> None:
    spec = AnalysisSpec(analysis_id="rr", outcome_id="complication", effect_measure="RR")
    res = run(spec, rr_arms())
    assert {e.study_id: e.reason_code for e in res.excluded} == {"t3": "non_informative"}
    assert res.k == 3 and res.common_effect is not None
    assert res.common_effect.method.startswith("Mantel-Haenszel")
    assert next(s for s in res.studies if s.study_id == "t2").continuity_corrected
    assert any(s.method == "inverse variance" for s in res.sensitivity)
    assert res.primary == res.random_effects
    assert res.random_effects is not None and res.random_effects.prediction_interval is not None


def test_or_inverse_variance_common_effect_option() -> None:
    spec = AnalysisSpec(
        analysis_id="or",
        outcome_id="complication",
        effect_measure="OR",
        common_effect_method="inverse_variance",
        primary_model="common_effect",
    )
    res = run(spec, rr_arms())
    assert res.primary == res.common_effect and res.common_effect is not None
    assert res.common_effect.method == "inverse variance"


def test_md_requires_sd_and_consistent_units() -> None:
    spec = AnalysisSpec(analysis_id="md", outcome_id="score", effect_measure="MD")
    arms = (
        two_arm(
            "m1",
            "score",
            {"mean": "10.2", "sd": "3.1", "n_analyzed": "30"},
            {"mean": "12.0", "sd": "3.5", "n_analyzed": "31"},
        )
        + two_arm(
            "m2",
            "score",
            {"mean": "9.8", "sd": "2.9", "n_analyzed": "25"},
            {"mean": "11.1", "sd": "3.0", "n_analyzed": "26"},
        )
        + two_arm(
            "m3",
            "score",
            {"mean": "9.0", "se": "0.6", "n_analyzed": "20"},
            {"mean": "10.0", "se": "0.7", "n_analyzed": "20"},
        )
    )
    res = run(spec, arms)
    assert {e.study_id: e.reason_code for e in res.excluded} == {"m3": "missing_data"}
    assert res.status == "pooled" and res.k == 2
    assert res.studies[0].yi == pytest.approx(10.2 - 12.0)
    mixed = arms[:2] + two_arm(
        "m4",
        "score",
        {"mean": "1.0", "sd": "0.3", "n_analyzed": "20"},
        {"mean": "1.2", "sd": "0.3", "n_analyzed": "20"},
        units={"mean": "cm", "sd": "cm"},
    )
    res2 = run(spec, mixed)
    assert res2.status == "not_estimable" and "units" in res2.status_reason


def test_smd_hedges_g() -> None:
    spec = AnalysisSpec(analysis_id="smd", outcome_id="score", effect_measure="SMD")
    arms = two_arm(
        "m1",
        "score",
        {"mean": "10.2", "sd": "3.1", "n_analyzed": "30"},
        {"mean": "12.0", "sd": "3.5", "n_analyzed": "31"},
    ) + two_arm(
        "m2",
        "score",
        {"mean": "9.8", "sd": "2.9", "n_analyzed": "25"},
        {"mean": "11.1", "sd": "3.0", "n_analyzed": "26"},
    )
    res = run(spec, arms)
    expected = es.standardised_mean_difference(10.2, 3.1, 30, 12.0, 3.5, 31)
    assert res.studies[0].yi == pytest.approx(expected.yi)
    assert res.warnings and "two studies" in " ".join(res.warnings)
    assert res.random_effects is not None and res.random_effects.prediction_interval is None


def test_hazard_ratio_is_unsupported_not_derived() -> None:
    spec = AnalysisSpec(analysis_id="os", outcome_id="survival", effect_measure="HR")
    res = run(spec, [])
    assert res.status == "unsupported" and "never derived" in res.status_reason


def test_comparative_missing_arm() -> None:
    spec = AnalysisSpec(analysis_id="rr", outcome_id="complication", effect_measure="RR")
    arms = [
        stats_arm(
            "x", "complication", role=ArmRole.INTERVENTION, label="SRS", events="1", n_analyzed="10"
        )
    ]
    res = run(spec, arms)
    assert res.status in ("descriptive_only", "no_data")
    assert [e.reason_code for e in res.excluded] == ["arm_missing"]


# --------------------------------------------------------------------------- output object


SPEC_RR = AnalysisSpec(analysis_id="rr", outcome_id="complication", effect_measure="RR")
PLAN = plan(NEC, SPEC_RR)


def _output(data: list | None = None) -> StatsOutput:  # type: ignore[type-arg]
    data = data if data is not None else necrosis_arms() + rr_arms()
    return run_plan(PLAN, inputs(data), project_id="p1", generated_at=NOW)


def test_output_is_deterministic_and_tamper_evident() -> None:
    data = necrosis_arms() + rr_arms()
    a, b = _output(data), _output(list(reversed(data)))
    assert a.results_sha256 == b.results_sha256 and a.input_sha256 == b.input_sha256
    payload = a.model_dump(mode="json")
    payload["analyses"][0]["random_effects"]["estimate"] += 1e-9
    with pytest.raises(ValidationError, match="results_sha256"):
        StatsOutput.model_validate(payload)
    roundtrip = StatsOutput.model_validate_json(a.to_canonical_json())
    assert roundtrip.to_canonical_json() == a.to_canonical_json()


def test_output_metadata_and_flatten() -> None:
    out = _output()
    assert out.software["scipy"] and out.software["numpy"] and out.engine_version
    flat = out.flatten()
    assert "nec.random_effects.estimate_natural" in flat and flat["rr.k"] == 3.0
    assert any("hazard ratios are never derived" in x for x in out.limitations)


def test_plan_must_be_approved_and_cover_known_outcomes() -> None:
    with pytest.raises(StatisticsInputError, match="not been approved"):
        run_plan(
            plan(NEC, approved=False), inputs(necrosis_arms()), project_id="p", generated_at=NOW
        )
    unknown = AnalysisSpec(analysis_id="u", outcome_id="nope", effect_measure="PROPORTION")
    with pytest.raises(StatisticsInputError, match="not in the protocol"):
        run_plan(plan(unknown), inputs(necrosis_arms()), project_id="p", generated_at=NOW)


def test_pooled_result_invariants() -> None:
    res = run(NEC, necrosis_arms())
    payload = res.model_dump(mode="json") | {"status": "single_study"}
    with pytest.raises(ValidationError):
        type(res).model_validate(payload)
    payload = res.model_dump(mode="json") | {"k": 2}
    with pytest.raises(ValidationError, match="k must equal"):
        type(res).model_validate(payload)


def test_reports_and_deterministic_forest_plot(tmp_path: Path) -> None:
    out = _output()
    md = report_markdown(out)
    assert "Excluded studies" in md and "non_informative" in md
    csv_text = analysis_csv(out.analyses[1])
    assert "excluded" in csv_text and "t3" in csv_text
    p1 = forest_plot(out.analyses[0], tmp_path / "a.svg")
    p2 = forest_plot(out.analyses[0], tmp_path / "b.svg")
    assert p1 is not None and p2 is not None
    assert p1.read_bytes() == p2.read_bytes()
    assert (
        forest_plot(
            run(AnalysisSpec(analysis_id="os", outcome_id="survival", effect_measure="HR"), []),
            tmp_path / "c.svg",
        )
        is None
    )
    assert forest_plot(out.analyses[1], tmp_path / "rr.svg") is not None


# --------------------------------------------------------------------------- failure paths


def test_mantel_haenszel_undefined_is_reported_not_patched() -> None:
    arms: list = []  # type: ignore[type-arg]
    for rid, n1, c, n2 in (
        ("u1", "20", "3", "21"),
        ("u2", "15", "2", "14"),
        ("u3", "30", "4", "31"),
    ):
        arms += two_arm(
            rid, "complication", {"events": "0", "n_analyzed": n1}, {"events": c, "n_analyzed": n2}
        )
    spec = AnalysisSpec(analysis_id="rr", outcome_id="complication", effect_measure="RR")
    res = run(spec, arms)
    assert res.status == "pooled" and res.common_effect is None
    assert any("Mantel-Haenszel estimate not estimable" in w for w in res.warnings)
    common = spec.model_copy(update={"primary_model": "common_effect"})
    res2 = run(common, arms)
    assert res2.status == "not_estimable" and res2.primary is None


def test_identical_estimates_hksj_not_estimable_with_reason() -> None:
    spec = AnalysisSpec(analysis_id="md", outcome_id="score", effect_measure="MD")
    arms = [
        *two_arm(
            "a",
            "score",
            {"mean": "10.0", "sd": "2.0", "n_analyzed": "20"},
            {"mean": "11.0", "sd": "2.0", "n_analyzed": "20"},
        ),
        *two_arm(
            "b",
            "score",
            {"mean": "12.0", "sd": "3.0", "n_analyzed": "30"},
            {"mean": "13.0", "sd": "3.0", "n_analyzed": "30"},
        ),
    ]
    res = run(spec, arms)
    assert res.status == "not_estimable" and "Hartung-Knapp" in res.status_reason
    adhoc = spec.model_copy(update={"random_effects_ci": "hksj_adhoc"})
    assert run(adhoc, arms).status == "pooled"


def test_smd_invalid_data_and_zero_denominator_excluded() -> None:
    spec = AnalysisSpec(analysis_id="smd", outcome_id="score", effect_measure="SMD")
    arms = two_arm(
        "z",
        "score",
        {"mean": "10.0", "sd": "0", "n_analyzed": "20"},
        {"mean": "11.0", "sd": "0", "n_analyzed": "20"},
    )
    res = run(spec, arms)
    assert [e.reason_code for e in res.excluded] == ["invalid_data"]
    rr = AnalysisSpec(analysis_id="rr", outcome_id="complication", effect_measure="RR")
    zero_n = two_arm(
        "n0",
        "complication",
        {"events": "0", "n_analyzed": "0"},
        {"events": "1", "n_analyzed": "10"},
    )
    assert [e.reason_code for e in run(rr, zero_n).excluded] == ["invalid_data"]
    nec0 = [stats_arm("x", "necrosis", follow_up=LAST_FU, events="0", n_analyzed="0")]
    assert [e.reason_code for e in run(NEC, nec0).excluded] == ["invalid_data"]


def test_km_ci_at_boundary_or_zero_width_excluded() -> None:
    arms = [
        km_arm("a", "92.3", "87.7", "95.2"),
        km_arm("b", "85.0", "78.6", "89.6"),
        km_arm("edge", "97.0", "80.0", "100"),
        km_arm("flat", "90.0", "90.0", "90.0"),
    ]
    codes = {e.study_id: e.reason_code for e in run(KM, arms).excluded}
    assert codes == {"edge": "ci_inconsistent", "flat": "invalid_data"}


# --------------------------------------------------------------------------- review fixes


def _half_arms() -> list:  # type: ignore[type-arg]
    return [stats_arm(r, "necrosis", follow_up=LAST_FU, events="5", n_analyzed="10")
            for r in ("h1", "h2")]  # fmt: skip


def test_identical_proportions_glmm_primary_survives_iv_failure() -> None:
    """5/10 and 5/10: the HKSJ SE of the IV sensitivity model is zero; the GLMM is fine."""
    res = run(NEC, _half_arms())
    assert res.status == "pooled" and res.k == 2
    assert [s.data for s in res.studies] == [{"events": 5, "n": 10}] * 2
    g = res.primary
    assert g is not None and g == res.random_effects and "GLMM" in g.method
    assert g.estimate == pytest.approx(0.0, abs=1e-6)
    assert g.estimate_natural == pytest.approx(0.5, abs=1e-6)
    assert g.tau2 == 0.0
    assert g.se == pytest.approx(math.sqrt(1 / 10 + 1 / 10), rel=1e-4)  # binomial ML, τ = 0
    # the failed sensitivity analysis is explained, not replaced by another interval
    assert not any("inverse variance" in r.method for r in res.sensitivity)
    failed = [w for w in res.warnings if w.startswith("Sensitivity analysis")]
    assert len(failed) == 1 and "Hartung-Knapp standard error is zero" in failed[0]
    assert res.heterogeneity is not None and res.heterogeneity.q == pytest.approx(0.0, abs=1e-12)
    assert [r.status for r in res.reports] == ["included", "included"]


def test_identical_proportions_iv_primary_reports_its_failure() -> None:
    spec = NEC.model_copy(update={"proportion_method": "logit_inverse_variance"})
    res = run(spec, _half_arms())
    assert res.status == "not_estimable" and res.k == 2
    assert "Hartung-Knapp standard error is zero" in res.status_reason
    assert res.primary is None and res.random_effects is None and res.common_effect is None
    assert [s.data for s in res.studies] == [{"events": 5, "n": 10}] * 2
    # estimable auxiliary models stay visible, explicitly labelled, never promoted to primary
    methods = [r.method for r in res.sensitivity]
    assert any("GLMM" in m for m in methods)
    assert any(m == "binomial maximum likelihood" for m in methods)
    assert res.heterogeneity is not None
    adhoc = spec.model_copy(update={"random_effects_ci": "hksj_adhoc"})
    assert run(adhoc, _half_arms()).status == "pooled"


def test_every_eligible_report_is_accounted_for() -> None:
    """20 eligible reports: 12 with the outcome, 8 without (never read as zero events)."""
    with_data = [(f"d{i:02d}", str(i % 4), str(20 + i)) for i in range(12)]
    arms = [stats_arm(r, "necrosis", follow_up=LAST_FU, events=x, n_analyzed=n)
            for r, x, n in with_data]  # fmt: skip
    no_data = [f"n{i:02d}" for i in range(8)]
    codes = {"n00": "not_reported", "n01": "not_reported", "n02": "figure_only",
             "n03": "insufficient_detail"}  # fmt: skip
    unavailable = [
        QuantitativeUnavailability(
            record_id=rid, outcome_id=None if rid == "n03" else "necrosis",
            reason_code=code, reason="synthetic reviewer note",  # type: ignore[arg-type]
            recorded_by="Dr Stats Reviewer", reviewer_id="rvw1",
        )
        for rid, code in codes.items()
    ]  # fmt: skip
    res = run(NEC, arms, eligible=no_data, unavailability=unavailable)
    assert res.k == 12 and len(res.reports) == 20
    status = {r.record_id: (r.status, r.reason_code) for r in res.reports}
    assert all(status[r] == ("included", None) for r, _, _ in with_data)
    assert [status[r][1] for r in no_data] == [
        *["confirmed_not_reported"] * 2,  # a reviewer confirmed the source does not report it
        *["recorded_unavailable"] * 2,
        *["outcome_data_not_extracted_or_unverified"] * 4,  # neutral: nothing extracted
    ]
    assert sorted(e.record_ids[0] for e in res.excluded) == no_data  # one entry per study
    assert res.total_participants == sum(int(n) for _, _, n in with_data)
    assert res.total_events == sum(int(x) for _, x, _ in with_data)
    # the 8 reports without data change nothing in the calculation
    alone = run(NEC, arms)
    assert alone.primary is not None and res.primary is not None
    assert res.primary.estimate == alone.primary.estimate
    assert res.heterogeneity == alone.heterogeneity


def test_reports_of_one_cohort_are_not_counted_as_separate_studies() -> None:
    arms = necrosis_arms()[:2]  # s1, s2
    cohorts = {"s1": "A", "s1b": "A", "s2": "s2", "c1": "C", "c2": "C"}
    res = run(NEC, arms, eligible=["s1b", "c1", "c2"], cohort_of=cohorts)
    assert res.k == 2 and len(res.reports) == 5
    # s1b: another report of contributing cohort A -> ledger only, not an excluded study
    s1b = next(r for r in res.reports if r.record_id == "s1b")
    assert s1b.reason_code == "outcome_data_not_extracted_or_unverified"
    assert s1b.reason is not None
    assert "not counted as a separate study" in s1b.reason
    # cohort C contributes nothing -> exactly one excluded study listing both reports
    assert [(e.record_ids, e.reason_code) for e in res.excluded] == [
        (["c1", "c2"], "outcome_data_not_extracted_or_unverified")
    ]
    # the secondary report and the missing outcome data stay visible in the human reports
    md, table = analysis_markdown(res), analysis_csv(res)
    for rid in ("s1b", "c1", "c2"):
        assert f"| {rid} |" in md
        assert any(row.startswith("report,") and f",{rid}," in row for row in table.splitlines())


def test_unsupported_measure_still_accounts_for_every_report() -> None:
    spec = AnalysisSpec(analysis_id="os", outcome_id="survival", effect_measure="HR")
    res = run(spec, [], eligible=["a", "b"])
    assert res.status == "unsupported"
    assert {r.reason_code for r in res.reports} == {"unsupported_measure"}
    assert len(res.reports) == 2


def test_arms_of_non_eligible_reports_abort() -> None:
    arms = necrosis_arms()[:2]
    data = inputs(arms)
    restricted = type(data)(**(vars(data) | {"eligible_record_ids": frozenset({"s1"})}))
    with pytest.raises(StatisticsInputError, match="not included at full text"):
        run_analysis(NEC, OUTCOMES["necrosis"], restricted)


def test_report_ledger_invariants() -> None:
    res = run(NEC, necrosis_arms()[:3], eligible=["x1"])
    payload = res.model_dump(mode="json")
    dup = payload | {"reports": [*payload["reports"], payload["reports"][0]]}
    with pytest.raises(ValidationError, match="appears once"):
        type(res).model_validate(dup)
    dropped = payload | {"reports": payload["reports"][1:]}
    with pytest.raises(ValidationError, match="must match the included studies"):
        type(res).model_validate(dropped)
    orphan = payload | {"reports": [r for r in payload["reports"] if r["record_id"] != "x1"]}
    with pytest.raises(ValidationError, match="must appear in the report ledger"):
        type(res).model_validate(orphan)
    entry = payload["reports"][0]
    with pytest.raises(ValidationError, match="names its unit"):
        ReportAccount.model_validate(entry | {"reason_code": "missing_data"})
    with pytest.raises(ValidationError, match="needs a reason"):
        ReportAccount.model_validate(entry | {"status": "excluded", "unit_id": None})


NON_REPORTING_CLAIMS = ("not reported", "not_reported", "did not report", "does not report",
                        "failed to report", "not report the")  # fmt: skip


def test_missing_extraction_is_never_described_as_not_reported() -> None:
    """An eligible report without approved outcome data may simply not have been extracted
    or verified yet: no output may claim that the paper failed to report the outcome."""
    res = run(NEC, necrosis_arms()[:2], eligible=["quiet"])
    entry = next(r for r in res.reports if r.record_id == "quiet")
    assert entry.status == "excluded"
    assert entry.reason_code == "outcome_data_not_extracted_or_unverified"
    excluded = next(e for e in res.excluded if e.record_ids == ["quiet"])
    assert excluded.reason_code == entry.reason_code
    texts = [entry.reason or "", excluded.reason]
    texts += [line for line in analysis_markdown(res).splitlines() if "| quiet |" in line]
    texts += [row for row in analysis_csv(res).splitlines() if ",quiet," in row]
    assert len(texts) == 5  # ledger, excluded study, Markdown row, CSV excluded + report rows
    for text in texts:
        assert not any(claim in text.lower() for claim in NON_REPORTING_CLAIMS), text
    assert res.k == 2 and res.total_participants == 65  # 25 + 40; nothing counted as zero


def test_only_a_reviewer_confirms_non_reporting() -> None:
    def unavailable(code: str) -> QuantitativeUnavailability:
        return QuantitativeUnavailability(
            record_id="quiet", outcome_id="necrosis", reason_code=code,  # type: ignore[arg-type]
            reason="checked the full text (synthetic)", recorded_by="Dr Stats Reviewer",
            reviewer_id="rvw1",
        )  # fmt: skip

    confirmed = run(NEC, necrosis_arms()[:2], eligible=["quiet"],
                    unavailability=[unavailable("not_reported")])  # fmt: skip
    entry = next(r for r in confirmed.reports if r.record_id == "quiet")
    assert entry.reason_code == "confirmed_not_reported"
    assert entry.reason is not None and "Dr Stats Reviewer" in entry.reason
    other = run(NEC, necrosis_arms()[:2], eligible=["quiet"],
                unavailability=[unavailable("figure_only")])  # fmt: skip
    assert next(r for r in other.reports if r.record_id == "quiet").reason_code == (
        "recorded_unavailable"
    )
    # identical numbers whatever the non-contributing report's status
    assert confirmed.primary == other.primary == run(NEC, necrosis_arms()[:2]).primary
