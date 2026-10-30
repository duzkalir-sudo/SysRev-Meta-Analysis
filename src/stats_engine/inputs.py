"""Stage 5 input integrity: from human-approved arms to analysis units.

The engine accepts only human-approved arms. For each analysis it keeps, per study, the
arm(s) whose **outcome**, **estimator** and **structured follow-up timepoint** match the
approved analysis plan, and refuses (with a recorded reason) anything ambiguous:

* every **full-text-eligible report** is accounted for in every analysis — included,
  recorded as unavailable, without data for the outcome (never read as zero events), or
  excluded with a reason — and reports of one study/cohort are never counted as separate
  studies;
* an arm that is not human-approved, or that belongs to a report outside the eligible
  set, aborts the whole run (fail closed);
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
from src.core.schemas.stats import (
    AnalysisSpec,
    ExcludedStudy,
    ExclusionCode,
    ReportAccount,
    canonical_sha256,
)

CRUDE = frozenset({None, EstimateType.CRUDE_PROPORTION})


@dataclass(frozen=True)
class AnalysisInputs:
    """Every persisted fact Stage 5 reads. Its hash is stored in ``stats_output.json``."""

    outcomes: dict[str, OutcomeDefinition]
    arms: list[ExtractedStudyArm]
    eligible_record_ids: frozenset[str]  # reports included at full-text assessment
    study_of: dict[str, str]  # record_id -> study_id
    cohort_of: dict[str, str]  # record_id -> cohort label (studies + human-resolved overlap)
    labels: dict[str, str]  # record_id -> "Author Year"
    unavailability: list[QuantitativeUnavailability] = field(default_factory=list)
    contributions: list[CohortContributionDecision] = field(default_factory=list)

    def validate(self) -> None:
        """Fail closed: only human-approved arms of full-text-eligible reports."""
        unapproved = sorted(a.arm_id for a in self.arms if not a.approved_by_human)
        if unapproved:
            raise StatisticsInputError(
                f"{len(unapproved)} extracted arm(s) are not human-approved: {unapproved[:5]}"
            )
        stray = sorted({a.record_id for a in self.arms} - self.eligible_record_ids)
        if stray:
            raise StatisticsInputError(
                f"arms exist for reports not included at full text: {stray[:5]}"
            )

    def study(self, record_id: str) -> str:
        return self.study_of.get(record_id, record_id)

    def cohort(self, record_id: str) -> str:
        return self.cohort_of.get(record_id, self.study(record_id))

    def label(self, record_id: str) -> str:
        return self.labels.get(record_id, record_id)

    def sha256(self) -> str:
        return canonical_sha256(
            {
                "outcomes": {
                    k: v.model_dump(mode="json") for k, v in sorted(self.outcomes.items())
                },
                "arms": [
                    a.model_dump(mode="json") for a in sorted(self.arms, key=lambda a: a.arm_id)
                ],
                "eligible_record_ids": sorted(self.eligible_record_ids),
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


@dataclass(frozen=True)
class RecordReason:
    code: ExclusionCode
    reason: str
    arm_ids: tuple[str, ...] = ()


@dataclass
class Selection:
    units: list[Unit]
    excluded: list[ExcludedStudy]  # cohort-level exclusions decided during selection
    record_reasons: dict[str, RecordReason]  # per-report reasons (cohort grouping later)
    notes: list[str]


def _excluded_study(
    inputs: AnalysisInputs, rids: Iterable[str], code: ExclusionCode, reason: str,
    arms: Iterable[str] = (),
) -> ExcludedStudy:  # fmt: skip
    rid_list = sorted(set(rids))
    return ExcludedStudy(
        study_id=inputs.study(rid_list[0]),
        record_ids=rid_list,
        label=" + ".join(inputs.label(r) for r in rid_list),
        arm_ids=sorted(arms),
        reason_code=code,
        reason=reason,
    )


def all_reports_excluded(inputs: AnalysisInputs, code: ExclusionCode, reason: str) -> Selection:
    """Accounting for an analysis that selects nothing (e.g. an unsupported measure)."""
    reasons = {rid: RecordReason(code, reason) for rid in inputs.eligible_record_ids}
    return Selection(units=[], excluded=[], record_reasons=reasons, notes=[])


def account_reports(
    inputs: AnalysisInputs,
    selection: Selection,
    included: dict[str, str],
    unit_exclusions: list[ExcludedStudy],
) -> tuple[list[ExcludedStudy], list[ReportAccount]]:
    """Cohort-level exclusions and the per-report ledger for one analysis.

    ``included`` maps each report used in the calculation to its unit. A study/cohort is
    listed as excluded once, with all of its reports, and only if none of its reports
    contributes; every eligible report appears exactly once in the ledger.
    """
    contributing = {inputs.cohort(rid) for rid in included}
    excluded = [*selection.excluded, *unit_exclusions]
    reasons: dict[str, tuple[ExclusionCode, str]] = {}
    for e in excluded:
        for rid in e.record_ids:
            reasons[rid] = (e.reason_code, e.reason)
    represented = contributing | {inputs.cohort(r) for e in excluded for r in e.record_ids}
    pending: dict[str, list[str]] = defaultdict(list)
    for rid, rr in selection.record_reasons.items():
        note = ""
        if inputs.cohort(rid) in contributing:
            note = (" (another report of this study/cohort contributes; this report is not "
                    "counted as a separate study)")  # fmt: skip
        reasons[rid] = (rr.code, rr.reason + note)
        if inputs.cohort(rid) not in represented:
            pending[inputs.cohort(rid)].append(rid)
    for _, rids in sorted(pending.items()):
        rids.sort(key=lambda r: (selection.record_reasons[r].code == "outcome_not_reported", r))
        first = selection.record_reasons[rids[0]]
        reason = first.reason if len(rids) == 1 else "; ".join(
            f"{inputs.label(r)}: {selection.record_reasons[r].reason}" for r in rids
        )  # fmt: skip
        arms = [a for r in rids for a in selection.record_reasons[r].arm_ids]
        excluded.append(_excluded_study(inputs, rids, first.code, reason, arms))
    ledger: list[ReportAccount] = []
    for rid in sorted(inputs.eligible_record_ids | set(included)):
        ids = {"record_id": rid, "study_id": inputs.study(rid), "cohort_id": inputs.cohort(rid),
               "label": inputs.label(rid)}  # fmt: skip
        if rid in included:
            ledger.append(
                ReportAccount.model_validate(ids | {"status": "included", "unit_id": included[rid]})
            )
        elif rid in reasons:
            code, reason = reasons[rid]
            ledger.append(
                ReportAccount.model_validate(
                    ids | {"status": "excluded", "reason_code": code, "reason": reason}
                )
            )
        else:  # pragma: no cover - every eligible report receives a reason in select_units
            raise StatisticsInputError(f"report {rid} is not accounted for")
    return sorted(excluded, key=lambda e: (e.label, e.study_id)), ledger


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
    inputs.validate()
    problems = spec.check_against_outcome(outcome)
    if problems:
        raise StatisticsInputError(f"analysis {spec.analysis_id}: " + "; ".join(problems))
    reasons: dict[str, RecordReason] = {}
    group_excluded: list[ExcludedStudy] = []
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

    for rid in sorted(inputs.eligible_record_ids):
        arms = by_record.get(rid, [])
        arm_ids = tuple(a.arm_id for a in arms)
        u = unavailable.get((rid, spec.outcome_id)) or unavailable.get((rid, None))
        if u is not None:
            reasons[rid] = RecordReason(
                "recorded_unavailable",
                f"recorded by {u.recorded_by} as unavailable ({u.reason_code}): {u.reason}",
                arm_ids,
            )
            continue
        if not arms:
            reasons[rid] = RecordReason(
                "outcome_not_reported",
                "eligible report without approved extracted data for this outcome; missing "
                "outcome data are never treated as zero events",
            )
            continue
        usable = [a for a in arms if a.estimate_type in estimators]
        if not usable:
            kinds = sorted({str(a.estimate_type or "raw counts/summaries") for a in arms})
            reasons[rid] = RecordReason(
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
                reasons[rid] = RecordReason(
                    "timepoint_missing",
                    f"no structured follow-up timepoint recorded; the plan requires {wanted}",
                    arm_ids,
                )
            else:
                reasons[rid] = RecordReason(
                    "timepoint_mismatch",
                    f"reports at {', '.join(reported)}; the plan requires {wanted} "
                    "(timepoints are never converted or approximated)",
                    arm_ids,
                )
            continue
        picked, problem = _pick_arms(rid, timed, roles, spec)
        if problem is not None:
            reasons[rid] = RecordReason(problem[0], problem[1], arm_ids)  # type: ignore[arg-type]
            continue
        chosen[rid] = picked

    units = _group_cohorts(spec, inputs, chosen, group_excluded, notes)
    _disambiguate_labels(units)
    return Selection(units=units, excluded=group_excluded, record_reasons=reasons, notes=notes)


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
    exclusions: list[ExcludedStudy],
    notes: list[str],
) -> list[Unit]:
    by_cohort: dict[str, list[str]] = defaultdict(list)
    for rid in chosen:
        by_cohort[inputs.cohort(rid)].append(rid)
    declared = [set(c.record_ids) for c in inputs.contributions if c.outcome_id == spec.outcome_id]
    units: list[Unit] = []
    for cohort, rids in sorted(by_cohort.items()):
        rids = sorted(rids)
        if len(rids) == 1:
            rid = rids[0]
            units.append(Unit(inputs.study(rid), (rid,), inputs.label(rid), chosen[rid]))
            continue
        if any(set(rids) <= d for d in declared):
            notes.append(
                f"reports {rids} of study/cohort {cohort} enter as separate units: a reviewer "
                "documented non-overlapping participants for this outcome"
            )
            for rid in rids:
                units.append(Unit(rid, (rid,), inputs.label(rid), chosen[rid]))
            continue
        arm_ids = [a.arm_id for rid in rids for a in chosen[rid].values()]
        exclusions.append(
            _excluded_study(
                inputs,
                rids,
                "overlapping_cohort",
                f"{len(rids)} reports of study/cohort {cohort} carry data for this analysis "
                "without a human decision on overlap; none is used (no double counting)",
                arm_ids,
            )
        )
    return units


def _disambiguate_labels(units: list[Unit]) -> None:
    counts: dict[str, int] = defaultdict(int)
    for u in units:
        counts[u.label] += 1
    for i, u in enumerate(units):
        if counts[u.label] > 1:
            units[i] = Unit(u.unit_id, u.record_ids, f"{u.label} [{u.record_ids[0][:8]}]", u.arms)
