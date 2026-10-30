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
    EvidenceLocationType,
    FullTextStatus,
    RoBTool,
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

# --------------------------------------------------------------------------- #
# Stage 3 — full-text retrieval manifest
# --------------------------------------------------------------------------- #


class FullTextItem(StrictModel):
    """Retrieval status of one included report. Missing items form the Gate 2 manifest."""

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
    pdf_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    unobtainable_reason: str | None = None

    @model_validator(mode="after")
    def _rules(self) -> FullTextItem:
        has_pdf = self.status in (FullTextStatus.OA_DOWNLOADED, FullTextStatus.USER_UPLOADED)
        if has_pdf and not (self.pdf_filename and self.pdf_sha256):
            raise ValueError(f"status {self.status} requires pdf_filename and pdf_sha256")
        if self.status == FullTextStatus.UNOBTAINABLE and not self.unobtainable_reason:
            raise ValueError("unobtainable reports require unobtainable_reason")
        return self

    @property
    def blocks_gate2(self) -> bool:
        return self.status in (FullTextStatus.PENDING, FullTextStatus.MISSING)


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
)
_COUNT_FIELDS = ("n_randomized", "n_analyzed", "events")


class ExtractedStudyArm(StrictModel):
    """Arm-level outcome data for one study, outcome and timepoint."""

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
        outcome_fields = set(values) - {"n_randomized", "n_analyzed"}
        if not outcome_fields:
            raise ValueError("an arm needs at least one outcome datum (events, mean, median, ...)")
        for name in _COUNT_FIELDS:
            v = self._num(name)
            if v is not None and (v < 0 or not float(v).is_integer()):
                raise ValueError(f"{name} must be a non-negative integer count")
        for name in ("sd", "se"):
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


class RiskOfBiasAssessment(StrictModel):
    assessment_id: Identifier = Field(default_factory=new_id)
    record_id: Identifier
    outcome_id: Identifier | None = None
    tool: RoBTool
    domains: list[DomainJudgement] = Field(default_factory=list)
    overall: str | None = None
    nos_stars: dict[str, int] | None = None
    assessor: NonEmptyStr
    assessed_by: ActorType
    approved_by_human: bool = False

    @model_validator(mode="after")
    def _tool_rules(self) -> RiskOfBiasAssessment:
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
