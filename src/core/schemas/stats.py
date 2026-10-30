"""Stage 5 contracts: the human-approved analysis plan and the versioned statistics output.

The **analysis plan** states, per analysis, the outcome, summary measure, follow-up
timepoint(s) and model settings. The statistics engine never chooses a timepoint, never
merges timepoints, estimators or overlapping cohorts on its own, and runs only from a plan
approved by an enrolled human reviewer.

The **statistics output** (``stats_output.json``) records every included and excluded
study with reasons, all model results at full binary64 precision, the methods and their
limitations, the software versions and the SHA-256 of the exact inputs and plan, so any
number in the manuscript can be traced and reproduced.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import Field, model_validator

from src.core.enums import ComparisonType, EffectMeasure
from src.core.schemas.base import (
    AwareDatetime,
    FrozenModel,
    Identifier,
    NonEmptyStr,
    StrictModel,
    new_id,
)
from src.core.schemas.protocol import ALLOWED_MEASURES, FollowUpTimepoint, OutcomeDefinition

STATS_SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"

# Measures the engine can pool after validation against metafor (Session 3).
POOLABLE_MEASURES = frozenset(
    {
        EffectMeasure.PROPORTION,
        EffectMeasure.KM_PROBABILITY,
        EffectMeasure.CUMULATIVE_INCIDENCE,
        EffectMeasure.RR,
        EffectMeasure.OR,
        EffectMeasure.MD,
        EffectMeasure.SMD,
    }
)

AnalysisStatus = Literal[
    "pooled",  # ≥ 2 studies combined
    "single_study",  # exactly one eligible study: reported, never "pooled"
    "descriptive_only",  # data exist but cannot be pooled defensibly (reasons given)
    "not_estimable",  # a model was attempted and has no valid estimate (reasons given)
    "unsupported",  # the requested measure/method is not implemented or not derivable
    "no_data",  # no eligible study contributes data
]
ProbabilityCIScale = Literal["log_log", "log", "logit", "plain"]


class ArmSelection(StrictModel):
    """A human choice of which arm of a multi-arm report enters an analysis."""

    record_id: Identifier
    role: Literal["intervention", "control", "single_group"]
    arm_label: NonEmptyStr


class AnalysisSpec(StrictModel):
    analysis_id: Identifier
    outcome_id: Identifier
    effect_measure: EffectMeasure
    timepoints: list[FollowUpTimepoint] = Field(
        default_factory=list,
        description="arms must report exactly one of these structured timepoints",
    )
    timepoint_justification: str | None = Field(
        default=None, description="required when more than one timepoint is pooled"
    )
    primary_model: Literal["random_effects", "common_effect"] = "random_effects"
    tau2_estimator: Literal["REML", "DL"] = "REML"
    random_effects_ci: Literal["hksj", "hksj_adhoc", "wald"] = "hksj"
    common_effect_method: Literal["auto", "inverse_variance", "mantel_haenszel"] = "auto"
    proportion_method: Literal["glmm", "logit_inverse_variance"] = "glmm"
    reported_ci_scale: ProbabilityCIScale | None = Field(
        default=None,
        description=(
            "Kaplan-Meier / cumulative-incidence analyses only: the scale on which the "
            "authors' confidence intervals are assumed symmetric. Each study's CI is checked "
            "against it (allowing for rounding); inconsistent studies are not pooled."
        ),
    )
    combine_km_and_actuarial: bool = False
    combine_justification: str | None = None
    arm_selections: list[ArmSelection] = Field(default_factory=list)
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    display_decimals: int = Field(default=2, ge=0, le=6)

    @model_validator(mode="after")
    def _rules(self) -> AnalysisSpec:
        if len(self.timepoints) > 1 and not (
            self.timepoint_justification and self.timepoint_justification.strip()
        ):
            raise ValueError("pooling several timepoints requires timepoint_justification")
        labels = [t.model_dump_json() for t in self.timepoints]
        if len(labels) != len(set(labels)):
            raise ValueError("duplicate timepoints")
        time_t = self.effect_measure in (
            EffectMeasure.KM_PROBABILITY,
            EffectMeasure.CUMULATIVE_INCIDENCE,
        )
        if time_t and self.reported_ci_scale is None:
            raise ValueError(
                f"{self.effect_measure} analyses must declare reported_ci_scale; the engine "
                "does not guess how the authors computed their confidence intervals"
            )
        if not time_t and self.reported_ci_scale is not None:
            raise ValueError("reported_ci_scale only applies to KM / cumulative-incidence")
        if self.combine_km_and_actuarial:
            if self.effect_measure != EffectMeasure.KM_PROBABILITY:
                raise ValueError("combine_km_and_actuarial only applies to KM_PROBABILITY")
            if not (self.combine_justification and self.combine_justification.strip()):
                raise ValueError("combining Kaplan-Meier and actuarial estimates needs a reason")
        if self.common_effect_method == "mantel_haenszel" and self.effect_measure not in (
            EffectMeasure.RR,
            EffectMeasure.OR,
        ):
            raise ValueError("Mantel-Haenszel applies to RR and OR only")
        keys = [(s.record_id, s.role) for s in self.arm_selections]
        if len(keys) != len(set(keys)):
            raise ValueError("one arm selection per record and role")
        return self

    def check_against_outcome(self, outcome: OutcomeDefinition) -> list[str]:
        """Problems that make this spec inconsistent with the approved protocol outcome."""
        problems: list[str] = []
        allowed = ALLOWED_MEASURES[(outcome.comparison, outcome.data_type)]
        if self.effect_measure not in allowed:
            problems.append(
                f"{self.effect_measure} is not admissible for {outcome.comparison} "
                f"{outcome.data_type} outcome {outcome.outcome_id}"
            )
        if outcome.timepoints:
            declared = {t.model_dump_json() for t in outcome.timepoints}
            for t in self.timepoints:
                if t.model_dump_json() not in declared:
                    problems.append(
                        f"timepoint {t.label} is not pre-specified for outcome {outcome.outcome_id}"
                    )
            if not self.timepoints:
                problems.append(
                    f"outcome {outcome.outcome_id} defines timepoints; the analysis must choose"
                )
        if outcome.comparison == ComparisonType.SINGLE_GROUP:
            bad = [s for s in self.arm_selections if s.role != "single_group"]
        else:
            bad = [s for s in self.arm_selections if s.role == "single_group"]
        if bad:
            problems.append("arm selections do not match the outcome's comparison type")
        return problems


class AnalysisPlan(StrictModel):
    plan_id: Identifier = Field(default_factory=new_id)
    version: int = Field(default=1, ge=1)
    analyses: list[AnalysisSpec] = Field(min_length=1)
    rationale: str | None = None
    drafted_by: NonEmptyStr
    status: Literal["draft", "approved"] = "draft"
    approved_by: str | None = None
    approved_reviewer_id: Identifier | None = None
    approved_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def _rules(self) -> AnalysisPlan:
        ids = [a.analysis_id for a in self.analyses]
        if len(ids) != len(set(ids)):
            raise ValueError("analysis_id values must be unique")
        stamps = (self.approved_by, self.approved_reviewer_id, self.approved_at)
        if self.status == "approved" and any(s is None for s in stamps):
            raise ValueError("approved plans need approved_by, approved_reviewer_id, approved_at")
        if self.status == "draft" and any(s is not None for s in stamps):
            raise ValueError("draft plans cannot carry approval stamps")
        return self

    def sha256(self) -> str:
        return canonical_sha256(self.model_dump(mode="json"))


# --------------------------------------------------------------------------- output


class StudyResult(FrozenModel):
    """One unit of analysis (a study, or a report a human declared non-overlapping)."""

    study_id: Identifier
    record_ids: list[Identifier]
    label: NonEmptyStr
    arm_ids: list[Identifier]
    data: dict[str, float] = Field(description="the extracted inputs actually used")
    yi: float | None = Field(description="estimate on the analysis scale")
    vi: float | None = Field(description="sampling variance on the analysis scale")
    ci_lower: float | None = None
    ci_upper: float | None = None
    estimate_natural: float | None = None
    ci_lower_natural: float | None = None
    ci_upper_natural: float | None = None
    ci_method: str
    weight_common_pct: float | None = None
    weight_random_pct: float | None = None
    continuity_corrected: bool = False
    notes: list[str] = Field(default_factory=list)


ExclusionCode = Literal[
    "recorded_unavailable",
    "outcome_not_reported",  # eligible report without approved data for this outcome
    "incompatible_estimator",
    "timepoint_mismatch",
    "timepoint_missing",
    "arm_ambiguous",
    "arm_missing",
    "missing_data",
    "overlapping_cohort",
    "non_informative",
    "boundary_estimate",
    "ci_inconsistent",
    "unit_mismatch",
    "invalid_data",
    "unsupported_measure",
    "analysis_failed",
]


class ExcludedStudy(FrozenModel):
    """A study/cohort that contributes nothing to the analysis (reports grouped by cohort)."""

    study_id: Identifier
    record_ids: list[Identifier]
    label: NonEmptyStr
    arm_ids: list[Identifier] = Field(default_factory=list)
    reason_code: ExclusionCode
    reason: NonEmptyStr


class ReportAccount(FrozenModel):
    """Fate of one full-text-eligible report in one analysis (complete per-report ledger).

    Reports of a study/cohort that already contributes are listed here with their reason
    but are not counted again as separate studies.
    """

    record_id: Identifier
    study_id: Identifier
    cohort_id: NonEmptyStr
    label: NonEmptyStr
    status: Literal["included", "excluded"]
    unit_id: Identifier | None = Field(default=None, description="analysis unit it enters")
    reason_code: ExclusionCode | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def _shape(self) -> ReportAccount:
        if self.status == "included":
            if self.unit_id is None or self.reason_code is not None:
                raise ValueError("an included report names its unit and carries no reason")
        elif self.reason_code is None or not (self.reason and self.reason.strip()):
            raise ValueError("an excluded report needs a reason code and reason")
        return self


class PooledResult(FrozenModel):
    model: Literal["common_effect", "random_effects"]
    method: NonEmptyStr
    scale: NonEmptyStr
    estimate: float
    se: float
    ci_lower: float
    ci_upper: float
    estimate_natural: float
    ci_lower_natural: float
    ci_upper_natural: float
    statistic: float
    p_value: float
    distribution: Literal["normal", "t"]
    df: int | None = None
    tau2: float | None = None
    tau2_ci: tuple[float, float] | None = None
    i2_model: float | None = None
    i2_ci: tuple[float, float] | None = None
    prediction_interval: tuple[float, float] | None = None
    prediction_interval_natural: tuple[float, float] | None = None
    prediction_df: int | None = None
    k: int = Field(ge=1)


class HeterogeneityResult(FrozenModel):
    basis: NonEmptyStr = Field(description="which estimates/weights Q was computed from")
    q: float
    df: int
    p_value: float
    i2: float
    h2: float
    s2_typical: float


class AnalysisResult(FrozenModel):
    analysis_id: Identifier
    outcome_id: Identifier
    outcome_name: NonEmptyStr
    effect_measure: EffectMeasure
    scale: NonEmptyStr
    timepoints: list[str]
    status: AnalysisStatus
    status_reason: NonEmptyStr
    k: int = Field(ge=0)
    studies: list[StudyResult] = Field(default_factory=list)
    excluded: list[ExcludedStudy] = Field(default_factory=list)
    reports: list[ReportAccount] = Field(
        default_factory=list, description="every full-text-eligible report and its fate"
    )
    primary: PooledResult | None = None
    common_effect: PooledResult | None = None
    random_effects: PooledResult | None = None
    sensitivity: list[PooledResult] = Field(default_factory=list)
    heterogeneity: HeterogeneityResult | None = None
    total_events: int | None = None
    total_participants: int | None = None
    settings: dict[str, Any]
    methods: list[str]
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistency(self) -> AnalysisResult:
        if self.k != len(self.studies):
            raise ValueError("k must equal the number of included studies")
        if self.status == "pooled":
            if self.k < 2 or self.primary is None:
                raise ValueError("a pooled analysis needs ≥ 2 studies and a primary result")
        elif any(r is not None for r in (self.primary, self.common_effect, self.random_effects)):
            raise ValueError(f"a {self.status} analysis cannot carry pooled results")
        if self.status == "single_study" and self.k != 1:
            raise ValueError("single_study status requires exactly one study")
        if self.heterogeneity is not None and self.k < 2:
            raise ValueError("heterogeneity statistics are meaningless for k < 2")
        ledger = [r.record_id for r in self.reports]
        if len(ledger) != len(set(ledger)):
            raise ValueError("each report appears once in the report ledger")
        in_studies = sorted(rid for s in self.studies for rid in s.record_ids)
        if sorted(r.record_id for r in self.reports if r.status == "included") != in_studies:
            raise ValueError("included reports in the ledger must match the included studies")
        if any(rid not in ledger for e in self.excluded for rid in e.record_ids):
            raise ValueError("every excluded report must appear in the report ledger")
        return self


class StatsOutput(FrozenModel):
    schema_version: Literal["1.0.0"] = STATS_SCHEMA_VERSION
    computed_by: Literal["src.stats_engine"] = "src.stats_engine"
    engine_version: NonEmptyStr
    output_id: Identifier = Field(default_factory=new_id)
    project_id: Identifier
    plan_id: Identifier
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    results_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generated_at: AwareDatetime
    software: dict[str, str]
    numeric_precision: NonEmptyStr
    analyses: list[AnalysisResult]
    limitations: list[str]

    @model_validator(mode="after")
    def _results_hash(self) -> StatsOutput:
        expected = results_fingerprint(self.analyses, self.plan_sha256, self.input_sha256)
        if self.results_sha256 != expected:
            raise ValueError("results_sha256 does not match the analyses (tampered or corrupt)")
        return self

    def to_canonical_json(self) -> str:
        return canonical_json(self.model_dump(mode="json"))

    def sha256(self) -> str:
        return hashlib.sha256(self.to_canonical_json().encode("utf-8")).hexdigest()

    def flatten(self) -> dict[str, float]:
        """``"<analysis_id>.<model>.<field>" -> value`` for the Stage 6 Auditor."""
        flat: dict[str, float] = {}
        for a in self.analyses:
            for name in ("common_effect", "random_effects", "primary"):
                r: PooledResult | None = getattr(a, name)
                if r is None:
                    continue
                for field in (
                    "estimate_natural",
                    "ci_lower_natural",
                    "ci_upper_natural",
                    "p_value",
                    "tau2",
                    "i2_model",
                ):
                    value = getattr(r, field)
                    if value is not None:
                        flat[f"{a.analysis_id}.{name}.{field}"] = float(value)
            if a.heterogeneity is not None:
                flat[f"{a.analysis_id}.heterogeneity.i2"] = a.heterogeneity.i2
                flat[f"{a.analysis_id}.heterogeneity.q"] = a.heterogeneity.q
            flat[f"{a.analysis_id}.k"] = float(a.k)
        return flat


def canonical_json(payload: Any) -> str:
    """Deterministic JSON: sorted keys, no NaN/Infinity, shortest round-trip floats."""
    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def results_fingerprint(analyses: list[AnalysisResult], plan_sha: str, input_sha: str) -> str:
    return canonical_sha256(
        {
            "plan_sha256": plan_sha,
            "input_sha256": input_sha,
            "analyses": [a.model_dump(mode="json") for a in analyses],
        }
    )
