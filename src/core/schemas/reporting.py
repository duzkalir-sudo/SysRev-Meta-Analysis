"""Session 7 contracts: PRISMA 2020 checklist decisions, GRADE certainty assessments,
structured synthesis without meta-analysis, and SWiM applicability.

Every record is append-only (the latest record per subject is current) and carries the
hashes of the inputs it was made on, so a relevant change to the evidence or the manuscript
makes it *stale*; a stale record is never reported and blocks the audit until a human
re-confirms or withdraws it. Nothing here is decided by a model or by the system: the
system offers decision support and computes only what follows mechanically from human
inputs (for example the certainty level implied by human domain judgements).

References (verified against the primary publications, PubMed Central):

* Page MJ, McKenzie JE, Bossuyt PM, et al. The PRISMA 2020 statement. BMJ 2021;372:n71.
  doi:10.1136/bmj.n71
* Campbell M, McKenzie JE, Sowden A, et al. Synthesis without meta-analysis (SWiM) in
  systematic reviews: reporting guideline. BMJ 2020;368:l6890. doi:10.1136/bmj.l6890
* Guyatt GH, Oxman AD, Sultan S, et al. GRADE guidelines: 9. Rating up the quality of
  evidence. J Clin Epidemiol 2011;64:1311-6. doi:10.1016/j.jclinepi.2011.06.004
* Schünemann HJ, Cuello C, Akl EA, et al. GRADE guidelines: 18. How ROBINS-I and other
  tools to assess risk of bias in nonrandomized studies should be used to rate the
  certainty of a body of evidence. J Clin Epidemiol 2019;111:105-114.
  doi:10.1016/j.jclinepi.2018.01.012
* Guyatt GH, Oxman AD, Santesso N, et al. GRADE guidelines: 12. Preparing summary of
  findings tables-binary outcomes. J Clin Epidemiol 2013;66:158-72.
  doi:10.1016/j.jclinepi.2012.01.012
* Iorio A, Spencer FA, Falavigna M, et al. Use of GRADE for assessment of evidence about
  prognosis. BMJ 2015;350:h870. doi:10.1136/bmj.h870
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import Field, computed_field, model_validator

from src.core.enums import ActorType
from src.core.schemas.base import (
    AwareDatetime,
    FrozenModel,
    Identifier,
    NonEmptyStr,
    StrictModel,
    new_id,
    utc_now,
)
from src.core.schemas.manuscript import SectionId

SHA = Field(pattern=r"^[0-9a-f]{64}$")

# --------------------------------------------------------------------------- checklists

ChecklistId = Literal["prisma_2020_main", "prisma_2020_abstract", "swim"]
# Statuses a *human* may record. The system itself only ever proposes "requires human
# review" (a candidate location was found) or "missing" (nothing was located); it never
# records "reported".
HumanChecklistStatus = Literal["reported", "partially_reported", "missing", "not_applicable"]
ChecklistStatus = Literal[
    "reported", "partially_reported", "missing", "not_applicable", "requires_human_review"
]


class ChecklistDecision(FrozenModel):
    """An authenticated reviewer's decision on one reporting-checklist item.

    The decision is bound to the manuscript version it was made on and to a *basis hash*:
    the content of the sections it locates plus the evidence components relevant to the
    item. When either changes the decision becomes stale and must be re-confirmed.
    """

    decision_id: Identifier = Field(default_factory=new_id)
    project_id: Identifier
    checklist: ChecklistId
    item_id: NonEmptyStr
    status: HumanChecklistStatus
    section_ids: list[SectionId] = Field(default_factory=list)
    location: str = Field(default="", max_length=500, description="precise location")
    justification: str = Field(min_length=10, max_length=4000)
    evidence_keys: list[NonEmptyStr] = Field(
        default_factory=list, description="evidence-package keys the reviewer relied on"
    )
    manuscript_version: int = Field(ge=1)
    manuscript_sha256: str = SHA
    evidence_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="content evidence fingerprint (SHA-256) displayed to the reviewer",
    )
    basis_sha256: str = SHA
    based_on_decision_id: Identifier | None = Field(
        default=None, description="the item's decision displayed to the reviewer (None: none)"
    )
    reviewer_id: Identifier
    reviewer_name: NonEmptyStr
    decided_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _located(self) -> ChecklistDecision:
        if self.status in ("reported", "partially_reported") and (
            not self.section_ids or not self.location.strip()
        ):
            raise ValueError(
                "a reported or partially reported item names the manuscript section(s) "
                "and the precise location where it is reported"
            )
        if len(set(self.section_ids)) != len(self.section_ids):
            raise ValueError("each section is named once")
        return self


# --------------------------------------------------------------------------- GRADE

CertaintyLevel = Literal["high", "moderate", "low", "very_low"]
LEVEL_VALUE: dict[str, int] = {"very_low": 1, "low": 2, "moderate": 3, "high": 4}
VALUE_LEVEL: dict[int, str] = {v: k for k, v in LEVEL_VALUE.items()}
LEVEL_LABEL: dict[str, str] = {
    "high": "high",
    "moderate": "moderate",
    "low": "low",
    "very_low": "very low",
}

GradeDomain = Literal[
    "risk_of_bias", "inconsistency", "indirectness", "imprecision", "publication_bias"
]
GRADE_DOMAINS: tuple[str, ...] = (
    "risk_of_bias",
    "inconsistency",
    "indirectness",
    "imprecision",
    "publication_bias",
)
DOMAIN_LABEL: dict[str, str] = {
    "risk_of_bias": "risk of bias",
    "inconsistency": "inconsistency",
    "indirectness": "indirectness",
    "imprecision": "imprecision",
    "publication_bias": "publication bias",
}
Concern = Literal["not_serious", "serious", "very_serious", "extremely_serious"]
CONCERN_LEVELS: dict[str, int] = {
    "not_serious": 0, "serious": 1, "very_serious": 2, "extremely_serious": 3,
}  # fmt: skip
# Rating down by three levels ("extremely serious") exists for risk of bias in bodies of
# non-randomised evidence assessed with ROBINS-I, which start at high certainty (GRADE
# guidelines 18: such evidence is generally rated down by at least two levels, to low or very
# low; reaching very low from high through risk of bias alone requires three). It is not
# available for any other domain or approach.
THREE_LEVEL_SCOPE = ("risk_of_bias", "interventions_nrsi_robins_i")

UpgradeFactor = Literal["large_effect", "dose_response", "plausible_confounding"]
UPGRADE_LABEL: dict[str, str] = {
    "large_effect": "large magnitude of effect",
    "dose_response": "dose-response gradient",
    "plausible_confounding": "plausible residual confounding would reduce the effect",
}

# The question the outcome answers; decided by the analysis, never by a reviewer's wish.
QuestionType = Literal["comparative_intervention_effect", "single_group_estimate"]

GradeApproach = Literal[
    # Randomised trials: the body of evidence starts at high certainty.
    "interventions_randomised",
    # Non-randomised studies of interventions assessed with ROBINS-I: start at high, with
    # risk of bias then rated by the ROBINS-I judgements (GRADE guidelines 18).
    "interventions_nrsi_robins_i",
    # Non-randomised studies assessed with other tools: start at low (GRADE guidelines 18).
    "interventions_nonrandomised_other",
    # GRADE for overall prognosis / event rates in broad categories of patients (Iorio et al.
    # 2015). Applies only to single-group estimates and only with a documented rationale; the
    # initial level is recorded by the reviewer with its justification (not hard-coded here).
    "overall_prognosis",
]
APPROACH_LABEL: dict[str, str] = {
    "interventions_randomised": "GRADE for intervention effects, randomised trials",
    "interventions_nrsi_robins_i": (
        "GRADE for intervention effects, non-randomised studies assessed with ROBINS-I"
    ),
    "interventions_nonrandomised_other": (
        "GRADE for intervention effects, non-randomised studies assessed with other tools"
    ),
    "overall_prognosis": "GRADE for overall prognosis (event rates in broad categories)",
}
APPROACH_QUESTION: dict[str, str] = {
    "interventions_randomised": "comparative_intervention_effect",
    "interventions_nrsi_robins_i": "comparative_intervention_effect",
    "interventions_nonrandomised_other": "comparative_intervention_effect",
    "overall_prognosis": "single_group_estimate",
}
FIXED_START: dict[str, str] = {
    "interventions_randomised": "high",
    "interventions_nrsi_robins_i": "high",
    "interventions_nonrandomised_other": "low",
}
# Rating up is considered for non-randomised evidence (GRADE guidelines 9 and 18).
UPGRADABLE = frozenset({"interventions_nrsi_robins_i", "interventions_nonrandomised_other"})

# Basing "publication bias not suspected" on a statistical test or a funnel plot is refused:
# the statistics engine performs no such test, and a non-significant test (or no funnel plot)
# does not demonstrate the absence of publication bias.
PUBLICATION_TEST_RE = re.compile(
    r"\bEgger|\bBegg|funnel|trim[- ]and[- ]fill|\bp\s*[>=≥]|non-?significant|"
    r"not\s+(statistically\s+)?significant|asymmetr",
    re.IGNORECASE,
)


class DomainRating(StrictModel):
    domain: GradeDomain
    concern: Concern
    rationale: str = Field(min_length=10, max_length=4000)

    @property
    def levels(self) -> int:
        return CONCERN_LEVELS[self.concern]


class UpgradeRating(StrictModel):
    factor: UpgradeFactor
    levels: int = Field(ge=1, le=2)
    rationale: str = Field(min_length=10, max_length=4000)

    @model_validator(mode="after")
    def _levels(self) -> UpgradeRating:
        if self.levels == 2 and self.factor != "large_effect":
            raise ValueError("only a large magnitude of effect may rate up by two levels")
        return self


class BaselineRisk(StrictModel):
    """An assumed comparator risk for absolute effects, entered by a human with its source.

    Absolute effects are never calculated without one (GRADE guidelines 12)."""

    label: NonEmptyStr = Field(max_length=200, description='e.g. "Study population"')
    risk: float = Field(gt=0, lt=1, description="proportion (0-1)")
    source: str = Field(min_length=10, max_length=2000)
    ref_key: Identifier | None = Field(default=None, description="reference registry key")


GradeStatus = Literal["draft", "approved", "not_assessed", "withdrawn"]


class GradeAssessment(FrozenModel):
    """One outcome-level certainty assessment (append-only; the latest per analysis is
    current). ``implied_certainty`` follows mechanically from the human inputs; the final
    certainty is the reviewer's, and any departure from the implied level needs a reason."""

    assessment_id: Identifier = Field(default_factory=new_id)
    project_id: Identifier
    analysis_id: Identifier
    outcome_id: Identifier
    question_type: QuestionType
    status: GradeStatus
    approach: GradeApproach | None = None
    framework_rationale: str = Field(default="", max_length=4000)
    initial_certainty: CertaintyLevel | None = None
    initial_rationale: str = Field(default="", max_length=4000)
    domains: list[DomainRating] = Field(default_factory=list)
    upgrades: list[UpgradeRating] = Field(default_factory=list)
    final_certainty: CertaintyLevel | None = None
    departure_rationale: str | None = Field(default=None, max_length=4000)
    summary: str = Field(default="", max_length=4000, description="overall explanation")
    baseline_risks: list[BaselineRisk] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    not_assessed_reason: (
        Literal["framework_not_applicable", "not_performed", "insufficient_information"] | None
    ) = None
    status_note: str | None = Field(default=None, max_length=4000)
    inputs_sha256: str = SHA
    based_on_assessment_id: Identifier | None = Field(
        default=None, description="the record displayed to the reviewer (None: none)"
    )
    created_by: NonEmptyStr
    actor_type: ActorType
    reviewer_id: Identifier | None = None
    created_at: AwareDatetime = Field(default_factory=utc_now)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def implied_certainty(self) -> CertaintyLevel | None:
        """Initial level − domain downgrades + upgrades, within very low … high."""
        if self.initial_certainty is None:
            return None
        value = LEVEL_VALUE[self.initial_certainty]
        value -= sum(d.levels for d in self.domains)
        value += sum(u.levels for u in self.upgrades)
        return VALUE_LEVEL[max(1, min(4, value))]  # type: ignore[return-value]

    def domain(self, name: str) -> DomainRating | None:
        return next((d for d in self.domains if d.domain == name), None)

    @model_validator(mode="wrap")
    @classmethod
    def _implied_is_derived(cls, data: object, handler: object) -> GradeAssessment:
        claimed = None
        if isinstance(data, dict) and "implied_certainty" in data:
            data = dict(data)
            claimed = data.pop("implied_certainty")
        result: GradeAssessment = handler(data)  # type: ignore[operator]
        if claimed is not None and claimed != result.implied_certainty:
            raise ValueError("claimed implied_certainty contradicts the domain judgements")
        return result

    @model_validator(mode="after")
    def _rules(self) -> GradeAssessment:
        human_needed = self.status in ("approved", "not_assessed", "withdrawn")
        if human_needed and (self.actor_type != ActorType.HUMAN or self.reviewer_id is None):
            raise ValueError(f"a {self.status} GRADE record is made by an authenticated human")
        if self.status == "withdrawn":
            if not (self.status_note and len(self.status_note) >= 10):
                raise ValueError("a withdrawal states its reason")
            return self
        if self.status == "not_assessed":
            if self.not_assessed_reason is None or not (
                self.status_note and len(self.status_note) >= 20
            ):
                raise ValueError(
                    "'not assessed' needs a reason code and an explanation (≥ 20 characters)"
                )
            if self.final_certainty is not None or self.domains or self.upgrades:
                raise ValueError("an unassessed outcome carries no certainty rating")
            return self
        if self.not_assessed_reason is not None:
            raise ValueError("not_assessed_reason only applies to 'not assessed' records")
        if self.approach is not None:
            self._approach_rules()
        for d in self.domains:
            if d.concern == "extremely_serious" and (d.domain, self.approach) != THREE_LEVEL_SCOPE:
                raise ValueError(
                    "'extremely serious' (rating down by three levels) applies only to risk of "
                    "bias with the ROBINS-I approach (non-randomised studies starting at high "
                    f"certainty); not to {d.domain.replace('_', ' ')} with "
                    f"{APPROACH_LABEL.get(str(self.approach), 'no approach')}"
                )
        names = [d.domain for d in self.domains]
        if len(names) != len(set(names)):
            raise ValueError("each GRADE domain is judged once")
        factors = [u.factor for u in self.upgrades]
        if len(factors) != len(set(factors)):
            raise ValueError("each upgrading factor is recorded once")
        pub = self.domain("publication_bias")
        if (
            pub is not None
            and pub.concern == "not_serious"
            and (PUBLICATION_TEST_RE.search(pub.rationale))
        ):
            raise ValueError(
                "publication bias cannot be judged absent on the basis of a statistical test "
                "or funnel plot: no such test was performed by the statistics engine, and a "
                "non-significant test or an unavailable funnel plot does not demonstrate the "
                "absence of publication bias; state the actual basis (e.g. search of "
                "registers and grey literature) or rate it as suspected"
            )
        if self.status == "approved":
            self._approved_rules()
        return self

    def _approach_rules(self) -> None:
        assert self.approach is not None
        if APPROACH_QUESTION[self.approach] != self.question_type:
            raise ValueError(
                f"{APPROACH_LABEL[self.approach]} does not apply to a "
                f"{self.question_type.replace('_', ' ')} question; a single-group estimate is "
                "not an intervention effect, and a comparative effect is not a prognosis"
            )
        fixed = FIXED_START.get(self.approach)
        if fixed is not None and self.initial_certainty not in (None, fixed):
            raise ValueError(
                f"{APPROACH_LABEL[self.approach]} starts at {LEVEL_LABEL[fixed]} certainty "
                "(GRADE guidelines 18); the initial level cannot be chosen freely"
            )
        if self.upgrades and self.approach not in UPGRADABLE:
            raise ValueError(
                "rating up applies to non-randomised evidence (GRADE guidelines 9 and 18); it "
                f"is not available for {APPROACH_LABEL[self.approach]}"
            )

    def _approved_rules(self) -> None:
        if self.approach is None:
            raise ValueError("an approved assessment records the GRADE approach used")
        if len(self.framework_rationale) < 20:
            raise ValueError("record why this GRADE approach fits the question (≥ 20 characters)")
        if self.approach == "overall_prognosis":
            if len(self.framework_rationale) < 40:
                raise ValueError(
                    "the prognosis framework needs an explicit methodological rationale "
                    "(≥ 40 characters) for treating this single-group estimate as an "
                    "event-rate (prognosis) question"
                )
            if self.initial_certainty is None or len(self.initial_rationale) < 20:
                raise ValueError(
                    "for the prognosis framework the reviewer records the initial certainty "
                    "and its justification; the system does not assume a starting level"
                )
        if self.initial_certainty is None:
            raise ValueError("an approved assessment has an initial certainty")
        if sorted(d.domain for d in self.domains) != sorted(GRADE_DOMAINS):
            raise ValueError("an approved assessment judges all five GRADE domains")
        if self.final_certainty is None:
            raise ValueError("an approved assessment records the final certainty")
        ceiling = min(4, LEVEL_VALUE[self.initial_certainty] + sum(u.levels for u in self.upgrades))
        if LEVEL_VALUE[self.final_certainty] > ceiling:
            raise ValueError(
                "the final certainty cannot exceed the initial level plus the recorded "
                "upgrades (a statistically significant result never raises certainty)"
            )
        if self.final_certainty != self.implied_certainty and not (
            self.departure_rationale and len(self.departure_rationale) >= 20
        ):
            raise ValueError(
                f"the final certainty ({LEVEL_LABEL[self.final_certainty]}) differs from the "
                f"level implied by the domain judgements "
                f"({LEVEL_LABEL[str(self.implied_certainty)]}); explain the departure"
            )
        if len(self.summary) < 10:
            raise ValueError("an approved assessment explains the overall certainty decision")
        if self.baseline_risks and self.question_type != "comparative_intervention_effect":
            raise ValueError("baseline risks apply only to comparative effects")
        labels = [b.label for b in self.baseline_risks]
        if len(labels) != len(set(labels)):
            raise ValueError("baseline-risk labels are unique")

    @property
    def rated_down(self) -> list[DomainRating]:
        return [d for d in self.domains if d.levels]


# --------------------------------------------------------------------------- synthesis

GroupingBasis = Literal[
    "population", "intervention", "comparator", "outcome", "study_design", "timepoint"
]
NotPooledReason = Literal[
    "heterogeneous_designs",
    "incompatible_measures",
    "insufficient_numerical_data",
    "qualitative_findings_only",
    "single_study",
    "other",
]
# The direction of effect is the side of the null on which the study's observed effect lies,
# irrespective of its statistical significance or confidence interval (vote counting based
# on the direction of effect; vote counting based on statistical significance is not an
# acceptable synthesis method — Cochrane Handbook ch. 12, McKenzie & Brennan; SWiM item 3;
# Boon MH, Thomson H. Res Synth Methods 2021;12:29-33, doi:10.1002/jrsm.1458, which applies
# that guidance and excludes studies with inconsistent direction from the count).
# Only "favours_*" enter a vote count. A point estimate exactly at the null, mixed results
# within a study and an unreportable direction are tabulated, never counted as votes.
Direction = Literal[
    "favours_intervention",
    "favours_comparator",
    "point_estimate_at_null",
    "mixed_or_conflicting",
    "direction_not_determinable",
    "not_applicable",
]
VOTE_DIRECTIONS = frozenset({"favours_intervention", "favours_comparator"})
DIRECTION_LABEL: dict[str, str] = {
    "favours_intervention": "favoured the intervention (observed direction)",
    "favours_comparator": "favoured the comparator (observed direction)",
    "point_estimate_at_null": "had a point estimate exactly at the null (not counted)",
    "mixed_or_conflicting": "had results in conflicting directions (not counted)",
    "direction_not_determinable": "did not report enough to establish a direction (not counted)",
    "not_applicable": "direction not applicable",
}
# A direction basis may not rest on statistical significance, P values or whether an
# interval includes the null: those classify by significance, not by direction.
SIGNIFICANCE_BASIS_RE = re.compile(
    r"significan|\bp\s*[<>=≤≥]|\bp[- ]?values?\b|confidence interval\w*\s+"
    r"(?:includ|cross|overlap|span|contain)|\bCI\s+(?:includ|cross|overlap|span|contain)|"
    r"\bnull\s+(?:not\s+)?(?:included|excluded)|\bchance\b",
    re.IGNORECASE,
)
# SWiM item 3 lists further methods (summary statistics of effect estimates, combining P
# values); they compute new statistics and are not implemented, so they cannot be chosen.
SynthesisMethod = Literal["structured_tabulation", "vote_counting_direction_of_effect"]
METHOD_LABEL: dict[str, str] = {
    "structured_tabulation": "structured tabulation of study-level findings without a "
    "quantitative summary",
    "vote_counting_direction_of_effect": "vote counting based on the direction of effect "
    "(tabulated counts of human-judged directions; no statistical test)",
}


class SynthesisFinding(StrictModel):
    """One study's source-linked finding within a synthesis group."""

    record_id: Identifier
    item_ids: list[Identifier] = Field(
        default_factory=list, description="accepted Master Extraction Table items"
    )
    arm_ids: list[Identifier] = Field(default_factory=list, description="approved arms")
    finding: str = Field(min_length=10, max_length=2000)
    direction: Direction = "not_applicable"
    direction_basis: str | None = Field(default=None, max_length=2000)
    selected_for_unit: bool = Field(
        default=True,
        description="this result represents its study/cohort (the voting unit); other "
        "results of the same unit are tabulated but never counted",
    )

    @model_validator(mode="after")
    def _sourced(self) -> SynthesisFinding:
        if not (self.item_ids or self.arm_ids):
            raise ValueError("a study-level finding links at least one extraction item or arm")
        if self.direction != "not_applicable" and not (
            self.direction_basis and len(self.direction_basis) >= 10
        ):
            raise ValueError("a direction of effect needs its basis (≥ 10 characters)")
        if self.direction_basis and SIGNIFICANCE_BASIS_RE.search(self.direction_basis):
            raise ValueError(
                f"the direction basis for report {self.record_id} relies on statistical "
                "significance, a P value or the confidence interval; the direction of effect "
                "is the side of the null on which the observed effect lies, regardless of "
                "significance (vote counting based on significance is not acceptable — "
                "Cochrane Handbook 12.2.2.1). State the observed direction, or record the "
                "result as direction not determinable"
            )
        return self


SynthesisStatus = Literal["draft", "approved", "withdrawn"]


class SynthesisGroup(FrozenModel):
    """A human-defined group of studies synthesised without meta-analysis (SWiM items 1-9)."""

    group_id: Identifier
    project_id: Identifier
    outcome_id: Identifier
    status: SynthesisStatus
    title: NonEmptyStr = Field(max_length=200)
    grouping_basis: list[GroupingBasis] = Field(min_length=1)
    description: str = Field(min_length=10, max_length=4000)
    rationale: str = Field(min_length=20, max_length=4000)
    protocol_change: bool = False
    protocol_change_rationale: str | None = Field(default=None, max_length=4000)
    reason_not_pooled: NotPooledReason
    reason_note: str = Field(min_length=10, max_length=2000)
    standardised_metric: str = Field(min_length=3, max_length=1000)
    synthesis_method: SynthesisMethod
    method_justification: str = Field(min_length=20, max_length=4000)
    prioritisation_criteria: str | None = Field(default=None, max_length=2000)
    heterogeneity_investigation: str | None = Field(default=None, max_length=2000)
    certainty_method: str = Field(min_length=10, max_length=2000)
    presentation: str = Field(min_length=10, max_length=2000)
    limitations: str = Field(min_length=20, max_length=4000)
    findings: list[SynthesisFinding] = Field(min_length=1)
    selection_rule: str | None = Field(
        default=None,
        max_length=2000,
        description="how one result was selected for a study/cohort with several reports or "
        "results (required whenever a result is not selected)",
    )
    summary: str = Field(min_length=20, max_length=4000)
    inputs_sha256: str = SHA
    revision_id: Identifier = Field(default_factory=new_id)
    based_on_revision_id: Identifier | None = Field(
        default=None, description="the revision displayed to the reviewer (None: new group)"
    )
    status_note: str | None = Field(default=None, max_length=4000)
    created_by: NonEmptyStr
    actor_type: ActorType
    reviewer_id: Identifier | None = None
    created_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _rules(self) -> SynthesisGroup:
        if self.status in ("approved", "withdrawn") and (
            self.actor_type != ActorType.HUMAN or self.reviewer_id is None
        ):
            raise ValueError(f"a {self.status} synthesis is recorded by an authenticated human")
        if self.status == "withdrawn" and not (self.status_note and len(self.status_note) >= 10):
            raise ValueError("a withdrawal states its reason")
        if self.protocol_change and not (
            self.protocol_change_rationale and len(self.protocol_change_rationale) >= 20
        ):
            raise ValueError("a change from the protocol's groups needs its rationale (SWiM 1b)")
        records = [f.record_id for f in self.findings]
        if len(records) != len(set(records)):
            raise ValueError("each report contributes one finding per group")
        if self.synthesis_method == "vote_counting_direction_of_effect" and any(
            f.direction == "not_applicable" for f in self.findings if f.selected_for_unit
        ):
            raise ValueError(
                "vote counting based on direction of effect needs a judged direction (or "
                "'direction not determinable') for every selected study result"
            )
        if any(not f.selected_for_unit for f in self.findings) and not (
            self.selection_rule and len(self.selection_rule) >= 20
        ):
            raise ValueError(
                "a result that does not represent its study/cohort needs the documented "
                "selection rule (≥ 20 characters) used to choose the counted result"
            )
        return self

    @property
    def record_ids(self) -> list[str]:
        return sorted(f.record_id for f in self.findings)


class SwimApplicability(FrozenModel):
    """A human decision on whether the SWiM guideline applies to the syntheses recorded.

    SWiM addresses quantitative intervention-effect data synthesised without meta-analysis;
    its applicability to descriptive single-group, prevalence or prognostic syntheses is a
    judgement the reviewer records with a rationale."""

    decision_id: Identifier = Field(default_factory=new_id)
    project_id: Identifier
    applicability: Literal["applicable", "partially_applicable", "not_applicable"]
    rationale: str = Field(min_length=20, max_length=4000)
    synthesis_sha256: str = SHA
    based_on_decision_id: Identifier | None = Field(
        default=None, description="the decision displayed to the reviewer (None: none)"
    )
    reviewer_id: Identifier
    reviewer_name: NonEmptyStr
    decided_at: AwareDatetime = Field(default_factory=utc_now)


__all__ = [
    "GRADE_DOMAINS",
    "BaselineRisk",
    "ChecklistDecision",
    "DomainRating",
    "GradeAssessment",
    "SwimApplicability",
    "SynthesisFinding",
    "SynthesisGroup",
    "UpgradeRating",
]
