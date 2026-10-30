"""Stage 3-4 contracts: full-text manifest, source-grounded extraction and risk of bias."""

from __future__ import annotations

import math
import re
from itertools import pairwise
from typing import Literal

from pydantic import Field, field_validator, model_validator

from src.core.enums import (
    ActorType,
    ArmRole,
    EstimateType,
    EvidenceLocationType,
    FullTextStatus,
    RoBTool,
    TimepointType,
    UploadMatchStatus,
)
from src.core.schemas.base import (
    DOI,
    PMID,
    AwareDatetime,
    FrozenModel,
    Identifier,
    NonEmptyStr,
    StrictModel,
    new_id,
    utc_now,
)
from src.core.schemas.protocol import FollowUpTimepoint

# --------------------------------------------------------------------------- #
# Stage 3 — full-text retrieval manifest
# --------------------------------------------------------------------------- #


AttemptOutcome = Literal[
    "retrieved",
    "not_available",
    "invalid_url",
    "blocked_url",
    "network_error",
    "http_error",
    "not_pdf",
    "too_large",
    "malformed",
    "duplicate",
    "skipped",
]


class RetrievalAttempt(FrozenModel):
    """One lawful retrieval attempt (an open-access service or a user upload)."""

    service: NonEmptyStr = Field(description='e.g. "europe_pmc_fulltext_xml", "unpaywall"')
    url: str | None = None
    outcome: AttemptOutcome
    detail: str | None = None
    attempted_at: AwareDatetime = Field(default_factory=utc_now)


_SHA256 = r"^[0-9a-f]{64}$"
_PDF_STATUSES = (FullTextStatus.OA_DOWNLOADED, FullTextStatus.USER_UPLOADED)


class FullTextItem(StrictModel):
    """Retrieval status of one included report. Missing items form the Gate 2 manifest.

    A report counts as *retrieved* only with a stored file whose SHA-256 is recorded. A PDF
    whose link to the record is uncertain or contradicted (``match_status``) blocks Gate 2
    until an authenticated human confirms it; an unobtainable report is "not retrieved"
    (PRISMA), never an eligibility exclusion.
    """

    record_id: Identifier
    first_author: str | None = None
    year: int | None = None
    journal: str | None = None
    title: NonEmptyStr
    pmid: PMID = None
    doi: DOI = None
    status: FullTextStatus = FullTextStatus.PENDING
    oa_url: str | None = None
    pdf_filename: str | None = None
    pdf_sha256: str | None = Field(default=None, pattern=_SHA256)
    unobtainable_reason: str | None = None
    # --- Session 2: provenance of the retrieved full text
    pmcid: str | None = None
    source_url: str | None = None
    source_kind: Literal["oa_pdf", "pmc_xml", "user_upload"] | None = None
    license: str | None = None
    xml_filename: str | None = None
    xml_sha256: str | None = Field(default=None, pattern=_SHA256)
    match_status: UploadMatchStatus | None = None
    match_notes: list[str] = Field(default_factory=list)
    confirmed_by: str | None = None
    confirmed_by_reviewer_id: Identifier | None = None
    recorded_by: str | None = Field(
        default=None, description="authenticated reviewer who uploaded / marked unobtainable"
    )
    recorded_by_reviewer_id: Identifier | None = None
    # XML-only full text: retrieved for PRISMA, but NOT usable for page-grounded extraction.
    # Gate 2 stays blocked until a human acknowledges that no usable PDF is available.
    xml_only_acknowledged_by: str | None = None
    xml_only_acknowledged_reviewer_id: Identifier | None = None
    xml_only_note: str | None = None
    attempts: list[RetrievalAttempt] = Field(default_factory=list)

    @model_validator(mode="after")
    def _rules(self) -> FullTextItem:
        has_pdf = self.status in _PDF_STATUSES
        if has_pdf and not (self.pdf_filename and self.pdf_sha256):
            raise ValueError(f"status {self.status} requires pdf_filename and pdf_sha256")
        if self.status == FullTextStatus.OA_STRUCTURED_TEXT and not (
            self.xml_filename and self.xml_sha256
        ):
            raise ValueError("structured full text requires xml_filename and xml_sha256")
        if self.status == FullTextStatus.UNOBTAINABLE and not self.unobtainable_reason:
            raise ValueError("unobtainable reports require unobtainable_reason")
        if self.match_status == UploadMatchStatus.HUMAN_CONFIRMED and not self.confirmed_by:
            raise ValueError("a human-confirmed PDF match requires confirmed_by")
        return self

    @property
    def is_retrieved(self) -> bool:
        return self.status in (*_PDF_STATUSES, FullTextStatus.OA_STRUCTURED_TEXT)

    @property
    def has_usable_pdf(self) -> bool:
        """A PDF exists and its identity is confirmed: the only basis for extraction."""
        return self.status in _PDF_STATUSES and not self.match_unconfirmed

    @property
    def xml_only_unacknowledged(self) -> bool:
        return (
            self.status == FullTextStatus.OA_STRUCTURED_TEXT
            and self.xml_only_acknowledged_by is None
        )

    @property
    def match_unconfirmed(self) -> bool:
        """A linked PDF whose identity was not positively verified (fail closed: an absent
        ``match_status`` means the check never ran, not that it passed)."""
        return self.status in _PDF_STATUSES and self.match_status not in (
            UploadMatchStatus.CONFIRMED,
            UploadMatchStatus.HUMAN_CONFIRMED,
        )

    @property
    def blocks_gate2(self) -> bool:
        return (
            self.status in (FullTextStatus.PENDING, FullTextStatus.MISSING)
            or self.match_unconfirmed
            or self.xml_only_unacknowledged
        )


# --------------------------------------------------------------------------- #
# Stage 4 — source-grounded extraction
# --------------------------------------------------------------------------- #

_WS_RE = re.compile(r"\s+")
_THOUSANDS_RE = re.compile(r"^[+-]?\d{1,3}(,\d{3})+(\.\d+)?$")


def _squash(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def parse_reported_number(text: str) -> float:
    """Parse a number exactly as typeset in a paper.

    Handles the Unicode minus (U+2212), the Lancet-style mid-dot decimal (U+00B7),
    thin/no-break spaces, thousands separators (``1,234``) and a lone decimal
    comma (``12,5``). Raises ``ValueError`` for anything else. Performs no arithmetic.
    """
    cleaned = (
        text.strip()
        .replace("−", "-")
        .replace("–", "-")
        .replace("·", ".")
        .replace(" ", "")
        .replace(" ", "")
        .replace(" ", "")
    )
    cleaned = cleaned.rstrip("%")
    if _THOUSANDS_RE.match(cleaned):
        cleaned = cleaned.replace(",", "")
    elif cleaned.count(",") == 1 and "." not in cleaned:
        cleaned = cleaned.replace(",", ".")
    value = float(cleaned)
    if not math.isfinite(value):
        raise ValueError(f"non-finite number: {text!r}")
    return value


class SourceEvidence(FrozenModel):
    """Exact provenance of an extracted value. Immutable once created."""

    evidence_id: Identifier = Field(default_factory=new_id)
    pdf_filename: NonEmptyStr
    pdf_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    page_number: int = Field(ge=1)
    location_type: EvidenceLocationType
    location_identifier: NonEmptyStr = Field(description='e.g. "Table 2", "Results ¶3"')
    verbatim_quote: str = Field(min_length=1, max_length=4000)
    extracted_by: ActorType
    extractor_identity: NonEmptyStr
    extracted_at: AwareDatetime = Field(default_factory=utc_now)

    @field_validator("verbatim_quote")
    @classmethod
    def _quote_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("verbatim_quote must not be blank")
        return value


class ExtractedValue(FrozenModel):
    """A single number copied verbatim from a source document.

    ``reported_text`` is the literal token as printed (e.g. ``"12·5"``). It must
    appear inside ``evidence.verbatim_quote``, and ``value`` must equal its parsed
    form — so a value that was computed rather than copied is rejected.
    """

    value: float
    reported_text: NonEmptyStr
    unit: str | None = None
    evidence: SourceEvidence

    @model_validator(mode="after")
    def _grounded(self) -> ExtractedValue:
        if not math.isfinite(self.value):
            raise ValueError("value must be finite")
        if _squash(self.reported_text) not in _squash(self.evidence.verbatim_quote):
            raise ValueError(
                f"reported_text {self.reported_text!r} does not occur in the verbatim quote"
            )
        try:
            parsed = parse_reported_number(self.reported_text)
        except ValueError as exc:
            raise ValueError(f"reported_text {self.reported_text!r} is not a number") from exc
        if not math.isclose(parsed, self.value, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(
                f"value {self.value} differs from reported_text {self.reported_text!r}; "
                "derived values must be computed by the stats engine, not extracted"
            )
        return self


_NUMERIC_FIELDS = (
    "n_randomized",
    "n_analyzed",
    "events",
    "mean",
    "sd",
    "se",
    "median",
    "q1",
    "q3",
    "minimum",
    "maximum",
    "reported_estimate",
    "ci_lower",
    "ci_upper",
    "n_at_risk",
    "follow_up_median",
    "person_time",
)
_COUNT_FIELDS = ("n_randomized", "n_analyzed", "events", "n_at_risk")
_SAMPLE_SIZE_FIELDS = frozenset({"n_randomized", "n_analyzed", "n_at_risk", "follow_up_median"})
# Estimates on a probability scale (bounded by 0-1 or 0-100 %).
PROBABILITY_ESTIMATES = frozenset(
    {
        EstimateType.CRUDE_PROPORTION,
        EstimateType.KAPLAN_MEIER,
        EstimateType.ACTUARIAL_LIFE_TABLE,
        EstimateType.CUMULATIVE_INCIDENCE,
    }
)
# Estimates defined at a specific time t (require a fixed timepoint).
TIME_T_ESTIMATES = frozenset(
    {
        EstimateType.KAPLAN_MEIER,
        EstimateType.ACTUARIAL_LIFE_TABLE,
        EstimateType.CUMULATIVE_INCIDENCE,
    }
)


class ExtractedStudyArm(StrictModel):
    """Arm-level outcome data for one study, outcome and timepoint.

    Comparative arms use ``arm_role`` intervention/control; single-arm studies and case
    series use ``single_group``. For non-randomised studies ``n_randomized`` holds the
    number enrolled.

    Raw counts (``events`` / ``n_analyzed``) and an author-reported summary
    (``reported_estimate`` + ``estimate_type``) are separate fields: a Kaplan-Meier
    5-year tumour-control probability is never stored as, or inferred from, a crude
    proportion. Nothing here is computed; every number is a verbatim extraction.
    """

    arm_id: Identifier = Field(default_factory=new_id)
    record_id: Identifier
    outcome_id: Identifier
    arm_label: NonEmptyStr
    arm_role: ArmRole
    timepoint: str | None = None

    n_randomized: ExtractedValue | None = None
    n_analyzed: ExtractedValue | None = None
    events: ExtractedValue | None = None
    mean: ExtractedValue | None = None
    sd: ExtractedValue | None = None
    se: ExtractedValue | None = None
    median: ExtractedValue | None = None
    q1: ExtractedValue | None = None
    q3: ExtractedValue | None = None
    minimum: ExtractedValue | None = None
    maximum: ExtractedValue | None = None

    # Author-reported summary estimate (e.g. "5-year actuarial control 92.3% (95% CI ...)")
    estimate_type: EstimateType | None = None
    estimate_scale: Literal["percent", "proportion"] | None = None
    reported_estimate: ExtractedValue | None = None
    ci_lower: ExtractedValue | None = None
    ci_upper: ExtractedValue | None = None
    ci_level: float | None = Field(default=None, gt=0, lt=100, description="e.g. 95")
    follow_up: FollowUpTimepoint | None = None
    n_at_risk: ExtractedValue | None = None
    follow_up_median: ExtractedValue | None = None
    person_time: ExtractedValue | None = None

    approved_by_human: bool = False
    approved_by: str | None = None
    approved_at: AwareDatetime | None = None

    @staticmethod
    def numeric_field_names() -> tuple[str, ...]:
        return _NUMERIC_FIELDS

    def extracted_values(self) -> dict[str, ExtractedValue]:
        return {name: v for name in _NUMERIC_FIELDS if (v := getattr(self, name)) is not None}

    def _num(self, name: str) -> float | None:
        v: ExtractedValue | None = getattr(self, name)
        return None if v is None else v.value

    @model_validator(mode="after")
    def _plausibility(self) -> ExtractedStudyArm:
        values = self.extracted_values()
        outcome_fields = set(values) - _SAMPLE_SIZE_FIELDS
        if not outcome_fields:
            raise ValueError(
                "an arm needs at least one outcome datum (events, mean, median, estimate, ...)"
            )
        for name in _COUNT_FIELDS:
            v = self._num(name)
            if v is not None and (v < 0 or not float(v).is_integer()):
                raise ValueError(f"{name} must be a non-negative integer count")
        for name in ("sd", "se", "follow_up_median", "person_time"):
            v = self._num(name)
            if v is not None and v < 0:
                raise ValueError(f"{name} must be non-negative")
        events, n_an, n_rand = (
            self._num("events"),
            self._num("n_analyzed"),
            self._num("n_randomized"),
        )
        if events is not None and n_an is not None and events > n_an:
            raise ValueError("events cannot exceed n_analyzed")
        if n_an is not None and n_rand is not None and n_an > n_rand:
            raise ValueError("n_analyzed cannot exceed n_randomized")
        at_risk = self._num("n_at_risk")
        if at_risk is not None and n_an is not None and at_risk > n_an:
            raise ValueError("n_at_risk cannot exceed n_analyzed")
        self._validate_reported_estimate()
        ordered = [
            (k, x)
            for k in ("minimum", "q1", "median", "q3", "maximum")
            if (x := self._num(k)) is not None
        ]
        for (k1, x1), (k2, x2) in pairwise(ordered):
            if x1 > x2:
                raise ValueError(f"{k1} ({x1}) must not exceed {k2} ({x2})")
        if self.approved_by_human and not (self.approved_by and self.approved_at):
            raise ValueError("human approval requires approved_by and approved_at")
        return self

    def _validate_reported_estimate(self) -> None:
        est = self.reported_estimate
        if (est is None) != (self.estimate_type is None):
            raise ValueError("reported_estimate and estimate_type must be given together")
        has_ci = self.ci_lower is not None or self.ci_upper is not None
        if has_ci:
            if est is None or self.ci_lower is None or self.ci_upper is None:
                raise ValueError("a confidence interval needs both bounds and a reported_estimate")
            if self.ci_level is None:
                raise ValueError("ci_level is required with a confidence interval (e.g. 95)")
            if not self.ci_lower.value <= est.value <= self.ci_upper.value:
                raise ValueError("reported_estimate must lie within its confidence interval")
        elif self.ci_level is not None:
            raise ValueError("ci_level given without a confidence interval")
        if est is None or self.estimate_type is None:
            if self.estimate_scale is not None:
                raise ValueError("estimate_scale is only meaningful with a reported_estimate")
            return
        etype = self.estimate_type
        if etype in PROBABILITY_ESTIMATES:
            if self.estimate_scale is None:
                raise ValueError(f"{etype} estimates need estimate_scale (percent or proportion)")
            upper = 100.0 if self.estimate_scale == "percent" else 1.0
            for name in ("reported_estimate", "ci_lower", "ci_upper"):
                v = self._num(name)
                if v is not None and not 0.0 <= v <= upper:
                    raise ValueError(f"{name} {v} is outside 0-{upper:g} for {etype}")
        elif self.estimate_scale is not None:
            raise ValueError(f"estimate_scale does not apply to {etype}")
        if self.follow_up is None and etype != EstimateType.MEDIAN_TIME_TO_EVENT:
            raise ValueError(f"{etype} estimates need the follow_up timepoint they refer to")
        if etype in TIME_T_ESTIMATES and (
            self.follow_up is None or self.follow_up.kind != TimepointType.FIXED
        ):
            raise ValueError(
                f"{etype} is a probability at a time t: follow_up must be a fixed timepoint"
            )
        if etype in (EstimateType.INCIDENCE_RATE, EstimateType.MEDIAN_TIME_TO_EVENT) and not (
            est.unit
        ):
            raise ValueError(f"{etype} estimates need a unit (e.g. per 100 person-years, months)")


# --------------------------------------------------------------------------- #
# Stage 4 — risk of bias
# --------------------------------------------------------------------------- #

# RoB 2: Sterne JAC et al. BMJ 2019;366:l4898.
ROB2_DOMAINS = (
    "randomization_process",
    "deviations_from_intended_interventions",
    "missing_outcome_data",
    "measurement_of_the_outcome",
    "selection_of_the_reported_result",
)
ROB2_JUDGEMENTS = ("low", "some_concerns", "high")

# ROBINS-I: Sterne JAC et al. BMJ 2016;355:i4919.
ROBINS_I_DOMAINS = (
    "confounding",
    "selection_of_participants",
    "classification_of_interventions",
    "deviations_from_intended_interventions",
    "missing_data",
    "measurement_of_outcomes",
    "selection_of_the_reported_result",
)
ROBINS_I_JUDGEMENTS = ("low", "moderate", "serious", "critical", "no_information")

# Newcastle-Ottawa Scale (Wells GA et al.): maximum stars per category.
NOS_MAX_STARS = {"selection": 4, "comparability": 2, "outcome_or_exposure": 3}


class DomainJudgement(StrictModel):
    domain: NonEmptyStr
    judgement: NonEmptyStr
    support: NonEmptyStr = Field(description="reviewer's justification")
    evidence: list[SourceEvidence] = Field(default_factory=list)
    missing_information: bool = Field(
        default=False, description="the report does not provide the information this domain needs"
    )
    needs_human_attention: list[str] = Field(
        default_factory=list, description="e.g. supporting quotes that failed verification"
    )


class RiskOfBiasAssessment(StrictModel):
    """A risk-of-bias assessment. Agent output is a *draft*: it may propose domain
    judgements (each backed by verified evidence or flagged as missing information) but
    may never set an overall judgement. Only an authenticated human approves.
    """

    assessment_id: Identifier = Field(default_factory=new_id)
    record_id: Identifier
    outcome_id: Identifier | None = None
    tool: RoBTool
    tool_rationale: str | None = None
    domains: list[DomainJudgement] = Field(default_factory=list)
    overall: str | None = None
    nos_stars: dict[str, int] | None = None
    assessor: NonEmptyStr
    assessed_by: ActorType
    approved_by_human: bool = False
    approved_by: str | None = None
    approved_reviewer_id: Identifier | None = None
    approved_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def _approval_rules(self) -> RiskOfBiasAssessment:
        stamps = (self.approved_by, self.approved_reviewer_id, self.approved_at)
        if self.approved_by_human:
            if any(v is None for v in stamps):
                raise ValueError(
                    "approval requires approved_by, approved_reviewer_id and approved_at"
                )
            if self.tool != RoBTool.NOS and self.overall is None:
                raise ValueError("an approved assessment needs an overall judgement")
        elif any(v is not None for v in stamps):
            raise ValueError("approval stamps are only allowed on approved assessments")
        if self.assessed_by != ActorType.HUMAN and not self.approved_by_human:
            if self.overall is not None:
                raise ValueError(
                    "automated drafts cannot set an overall judgement; a human must approve"
                )
            for d in self.domains:
                if not d.evidence and not d.missing_information:
                    raise ValueError(
                        f"draft domain {d.domain!r} needs supporting evidence or an explicit "
                        "missing_information flag"
                    )
        return self

    @model_validator(mode="after")
    def _tool_rules(self) -> RiskOfBiasAssessment:
        if self.tool == RoBTool.EXTERNAL:
            if self.assessed_by != ActorType.HUMAN:
                raise ValueError("external-instrument assessments are recorded by humans only")
            if not (self.tool_rationale and self.tool_rationale.strip()):
                raise ValueError("name the external instrument and why it was chosen")
            if self.nos_stars is not None:
                raise ValueError("nos_stars is only valid for NOS")
            return self
        if self.tool == RoBTool.NOS:
            self._validate_nos()
        else:
            domains, levels = (
                (ROB2_DOMAINS, ROB2_JUDGEMENTS)
                if self.tool == RoBTool.ROB2
                else (ROBINS_I_DOMAINS, ROBINS_I_JUDGEMENTS)
            )
            self._validate_domains(domains, levels)
        return self

    def _validate_nos(self) -> None:
        if self.domains:
            raise ValueError("NOS assessments use nos_stars, not domain judgements")
        if self.nos_stars is None or set(self.nos_stars) != set(NOS_MAX_STARS):
            raise ValueError(f"NOS requires star counts for {sorted(NOS_MAX_STARS)}")
        for category, stars in self.nos_stars.items():
            if not 0 <= stars <= NOS_MAX_STARS[category]:
                raise ValueError(f"NOS {category} stars must be 0..{NOS_MAX_STARS[category]}")

    def _validate_domains(self, domains: tuple[str, ...], levels: tuple[str, ...]) -> None:
        if self.nos_stars is not None:
            raise ValueError("nos_stars is only valid for NOS")
        names = [d.domain for d in self.domains]
        if sorted(names) != sorted(domains):
            raise ValueError(f"{self.tool} requires exactly the domains {list(domains)}")
        for d in self.domains:
            if d.judgement not in levels:
                raise ValueError(f"invalid {self.tool} judgement {d.judgement!r}")
        if self.overall is None:
            return
        if self.overall not in levels:
            raise ValueError(f"invalid overall judgement {self.overall!r}")
        severity = {lvl: i for i, lvl in enumerate(levels) if lvl != "no_information"}
        worst = max(
            (severity[d.judgement] for d in self.domains if d.judgement in severity), default=0
        )
        if self.overall == "no_information":
            # ROBINS-I: "no information" overall only when no domain is serious/critical.
            if worst >= severity["serious"]:
                raise ValueError(
                    "'no_information' overall is invalid with serious/critical domains"
                )
            return
        # Both tools: the overall judgement is at least as severe as the worst domain.
        if severity[self.overall] < worst:
            raise ValueError("overall judgement cannot be less severe than the worst domain")


# --------------------------------------------------------------------------- #
# Gate 3 — master extraction table
# --------------------------------------------------------------------------- #


class MasterExtractionTable(StrictModel):
    arms: list[ExtractedStudyArm] = Field(default_factory=list)
    risk_of_bias: list[RiskOfBiasAssessment] = Field(default_factory=list)
    status: Literal["draft", "approved"] = "draft"

    def unapproved_arm_ids(self) -> list[str]:
        return [a.arm_id for a in self.arms if not a.approved_by_human]
