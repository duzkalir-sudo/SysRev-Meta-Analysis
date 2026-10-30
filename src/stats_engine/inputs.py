"""Stage 5 input integrity: from human-approved arms to analysis units.

The engine accepts only human-approved arms. For each analysis it keeps, per study, the
arm(s) whose **outcome**, **estimator** and **structured follow-up timepoint** match the
approved analysis plan, and refuses (with a recorded reason) anything ambiguous:

* an arm that is not human-approved aborts the whole run (fail closed);
* a report recorded as unavailable for the outcome is excluded with the human's reason;
* estimators are never substituted (crude counts ≠ Kaplan-Meier ≠ cumulative incidence);
* follow-up timepoints must match one declared in the plan exactly — no unit conversion
  and no "nearest timepoint";
* several reports of one study/cohort contribute only if an authenticated human
  documented that they include non-overlapping participants for this outcome;
* multi-arm reports need an explicit human arm selection (no automatic splitting or
  combining of arms).

Nothing is computed or imputed here.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

from src.core.enums import ArmRole, ComparisonType, EffectMeasure, EstimateType
from src.core.exceptions import StatisticsInputError
from src.core.schemas.extraction import ExtractedStudyArm
from src.core.schemas.master import CohortContributionDecision, QuantitativeUnavailability
from src.core.schemas.protocol import FollowUpTimepoint, OutcomeDefinition
from src.core.schemas.stats import AnalysisSpec, ExcludedStudy, canonical_sha256

CRUDE = frozenset({None, EstimateType.CRUDE_PROPORTION})


@dataclass(frozen=True)
class AnalysisInputs:
    """Every persisted fact Stage 5 reads. Its hash is stored in ``stats_output.json``."""

    outcomes: dict[str, OutcomeDefinition]
    arms: list[ExtractedStudyArm]
    study_of: dict[str, str]  # record_id -> study_id
    cohort_of: dict[str, str]  # record_id -> cohort label (studies + human-resolved overlap)
    labels: dict[str, str]  # record_id -> "Author Year"
    unavailability: list[QuantitativeUnavailability] = field(default_factory=list)
    contributions: list[CohortContributionDecision] = field(default_factory=list)

    def require_approved(self) -> None:
        unapproved = sorted(a.arm_id for a in self.arms if not a.approved_by_human)
        if unapproved:
            raise StatisticsInputError(
                f"{len(unapproved)} extracted arm(s) are not human-approved: {unapproved[:5]}"
            )

    def sha256(self) -> str:
        return canonical_sha256(
            {
                "outcomes": {
                    k: v.model_dump(mode="json") for k, v in sorted(self.outcomes.items())
                },
                "arms": [
                    a.model_dump(mode="json") for a in sorted(self.arms, key=lambda a: a.arm_id)
                ],
                "study_of": dict(sorted(self.study_of.items())),
                "cohort_of": dict(sorted(self.cohort_of.items())),
                "labels": dict(sorted(self.labels.items())),
                "unavailability": sorted(
                    (u.model_dump(mode="json") for u in self.unavailability),
                    key=lambda d: str(d["unavailability_id"]),
                ),
                "contributions": sorted(
                    (c.model_dump(mode="json") for c in self.contributions),
                    key=lambda d: str(d["decision_id"]),
                ),
            }
        )


@dataclass(frozen=True)
class Unit:
    """One unit of analysis with its selected arm per role."""

    unit_id: str
    record_ids: tuple[str, ...]
    label: str
    arms: dict[str, ExtractedStudyArm]

    @property
    def arm_ids(self) -> list[str]:
        return [a.arm_id for _, a in sorted(self.arms.items())]


@dataclass
class Selection:
    units: list[Unit]
    excluded: list[ExcludedStudy]
    notes: list[str]


class _Exclusions:
    def __init__(self, inputs: AnalysisInputs) -> None:
        self.inputs = inputs
        self.items: list[ExcludedStudy] = []

    def add(self, rids: Iterable[str], code: str, reason: str, arms: Iterable[str] = ()) -> None:
        rid_list = sorted(set(rids))
        self.items.append(
            ExcludedStudy.model_validate(
                {
                    "study_id": self.inputs.study_of.get(rid_list[0], rid_list[0]),
                    "record_ids": rid_list,
                    "label": " + ".join(self.inputs.labels.get(r, r) for r in rid_list),
                    "arm_ids": sorted(arms),
                    "reason_code": code,
                    "reason": reason,
                }
            )
        )


def compatible_estimators(spec: AnalysisSpec) -> frozenset[EstimateType | None]:
    m = spec.effect_measure
    if m in (EffectMeasure.PROPORTION, EffectMeasure.RR, EffectMeasure.OR):
        return CRUDE
    if m == EffectMeasure.KM_PROBABILITY:
        kinds: set[EstimateType | None] = {EstimateType.KAPLAN_MEIER}
        if spec.combine_km_and_actuarial:
            kinds.add(EstimateType.ACTUARIAL_LIFE_TABLE)
        return frozenset(kinds)
    if m == EffectMeasure.CUMULATIVE_INCIDENCE:
        return frozenset({EstimateType.CUMULATIVE_INCIDENCE})
    return frozenset({None})  # MD / SMD: raw arm-level summaries only


def _timepoint_key(t: FollowUpTimepoint | None) -> str | None:
    return None if t is None else t.model_dump_json()


def _roles(outcome: OutcomeDefinition) -> tuple[str, ...]:
    if outcome.comparison == ComparisonType.SINGLE_GROUP:
        return (ArmRole.SINGLE_GROUP.value,)
    return (ArmRole.INTERVENTION.value, ArmRole.CONTROL.value)


def select_units(
    spec: AnalysisSpec, outcome: OutcomeDefinition, inputs: AnalysisInputs
) -> Selection:
    inputs.require_approved()
    problems = spec.check_against_outcome(outcome)
    if problems:
        raise StatisticsInputError(f"analysis {spec.analysis_id}: " + "; ".join(problems))
    excluded = _Exclusions(inputs)
    exclude = excluded.add
    notes: list[str] = []

    by_record: dict[str, list[ExtractedStudyArm]] = defaultdict(list)
    for arm in inputs.arms:
        if arm.outcome_id == spec.outcome_id:
            by_record[arm.record_id].append(arm)

    unavailable = {
        (u.record_id, u.outcome_id): u for u in inputs.unavailability
    }  # outcome None = all outcomes
    estimators = compatible_estimators(spec)
    wanted_times = {_timepoint_key(t) for t in spec.timepoints} or {None}
    roles = _roles(outcome)
    chosen: dict[str, dict[str, ExtractedStudyArm]] = {}

    for rid in sorted(by_record):
        arms = by_record[rid]
        arm_ids = [a.arm_id for a in arms]
        u = unavailable.get((rid, spec.outcome_id)) or unavailable.get((rid, None))
        if u is not None:
            exclude(
                [rid],
                "recorded_unavailable",
                f"recorded by {u.recorded_by} as unavailable ({u.reason_code}): {u.reason}",
                arm_ids,
            )
            continue
        usable = [a for a in arms if a.estimate_type in estimators]
        if not usable:
            kinds = sorted({str(a.estimate_type or "raw counts/summaries") for a in arms})
            exclude(
                [rid],
                "incompatible_estimator",
                f"reports {', '.join(kinds)}, not an estimate compatible with "
                f"{spec.effect_measure}; estimators are never converted",
                arm_ids,
            )
            continue
        timed = [a for a in usable if _timepoint_key(a.follow_up) in wanted_times]
        if not timed:
            reported = sorted({a.follow_up.label if a.follow_up else "none" for a in usable})
            wanted = ", ".join(t.label for t in spec.timepoints) or "no structured timepoint"
            if all(a.follow_up is None for a in usable):
                exclude(
                    [rid],
                    "timepoint_missing",
                    f"no structured follow-up timepoint recorded; the plan requires {wanted}",
                    arm_ids,
                )
            else:
                exclude(
                    [rid],
                    "timepoint_mismatch",
                    f"reports at {', '.join(reported)}; the plan requires {wanted} "
                    "(timepoints are never converted or approximated)",
                    arm_ids,
                )
            continue
        picked, problem = _pick_arms(rid, timed, roles, spec)
        if problem is not None:
            exclude([rid], problem[0], problem[1], arm_ids)
            continue
        chosen[rid] = picked

    units = _group_cohorts(spec, inputs, chosen, excluded, notes)
    _disambiguate_labels(units)
    return Selection(units=units, excluded=excluded.items, notes=notes)


def _pick_arms(
    rid: str, arms: list[ExtractedStudyArm], roles: tuple[str, ...], spec: AnalysisSpec
) -> tuple[dict[str, ExtractedStudyArm], tuple[str, str] | None]:
    picked: dict[str, ExtractedStudyArm] = {}
    selections: dict[tuple[str, str], str] = {
        (s.record_id, s.role): s.arm_label for s in spec.arm_selections
    }
    for role in roles:
        candidates = [a for a in arms if a.arm_role.value == role]
        label = selections.get((rid, role))
        if label is not None:
            candidates = [a for a in candidates if a.arm_label == label]
            if not candidates:
                return {}, ("arm_missing", f"the plan selects {role} arm {label!r}, not found")
        if not candidates:
            return {}, ("arm_missing", f"no approved {role} arm for this outcome and timepoint")
        if len(candidates) > 1:
            desc = ", ".join(
                sorted(
                    f"{a.arm_label} ({a.follow_up.label if a.follow_up else 'no time'})"
                    for a in candidates
                )
            )
            return {}, (
                "arm_ambiguous",
                f"{len(candidates)} candidate {role} arms ({desc}); multi-arm or multi-"
                "timepoint reports need an explicit arm selection in the analysis plan",
            )
        picked[role] = candidates[0]
    return picked, None


def _group_cohorts(
    spec: AnalysisSpec,
    inputs: AnalysisInputs,
    chosen: dict[str, dict[str, ExtractedStudyArm]],
    exclusions: _Exclusions,
    notes: list[str],
) -> list[Unit]:
    by_cohort: dict[str, list[str]] = defaultdict(list)
    for rid in chosen:
        by_cohort[inputs.cohort_of.get(rid, inputs.study_of.get(rid, rid))].append(rid)
    declared = [set(c.record_ids) for c in inputs.contributions if c.outcome_id == spec.outcome_id]
    units: list[Unit] = []
    for cohort, rids in sorted(by_cohort.items()):
        rids = sorted(rids)
        if len(rids) == 1:
            rid = rids[0]
            units.append(
                Unit(
                    inputs.study_of.get(rid, rid), (rid,), inputs.labels.get(rid, rid), chosen[rid]
                )
            )
            continue
        if any(set(rids) <= d for d in declared):
            notes.append(
                f"reports {rids} of study/cohort {cohort} enter as separate units: a reviewer "
                "documented non-overlapping participants for this outcome"
            )
            for rid in rids:
                units.append(Unit(rid, (rid,), inputs.labels.get(rid, rid), chosen[rid]))
            continue
        arm_ids = [a.arm_id for rid in rids for a in chosen[rid].values()]
        exclusions.add(
            rids,
            "overlapping_cohort",
            f"{len(rids)} reports of study/cohort {cohort} carry data for this analysis "
            "without a human decision on overlap; none is used (no double counting)",
            arm_ids,
        )
    return units


def _disambiguate_labels(units: list[Unit]) -> None:
    counts: dict[str, int] = defaultdict(int)
    for u in units:
        counts[u.label] += 1
    for i, u in enumerate(units):
        if counts[u.label] > 1:
            units[i] = Unit(u.unit_id, u.record_ids, f"{u.label} [{u.record_ids[0][:8]}]", u.arms)
