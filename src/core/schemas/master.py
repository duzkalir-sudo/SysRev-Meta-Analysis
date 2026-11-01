"""Stage 4 contracts: the human-reviewed Master Extraction Table.

An :class:`ExtractionItem` is one requested datum for one report (a study characteristic,
a baseline variable, or an arm-level outcome number). It records *what the source says*
(``data_status``), the verbatim evidence, the deterministic verification of that evidence
against the parsed document, and the human review of the item. Items are never deleted
or edited in place after review: a correction supersedes the old item with a new version,
and every review action is an append-only :class:`ExtractionReviewEvent`.

A missing datum is ``NOT_REPORTED`` with ``value=None`` — never zero — and a missing
result never excludes a study; exclusion is a screening decision only.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import Field, SerializerFunctionWrapHandler, model_serializer, model_validator

from src.core.enums import (
    ActorType,
    ArmRole,
    DataStatus,
    EstimateType,
    EvidenceCheckStatus,
    ExtractionDomain,
    ReviewStatus,
)
from src.core.schemas.base import (
    AwareDatetime,
    FrozenModel,
    Identifier,
    NonEmptyStr,
    StrictModel,
    new_id,
    utc_now,
)
from src.core.schemas.extraction import (
    ExtractedStudyArm,
    ExtractedValue,
    NumberFormat,
    NumberFormatStatus,
    SourceEvidence,
    number_format_status,
)
from src.core.schemas.protocol import FollowUpTimepoint

ARM_FIELDS = frozenset(ExtractedStudyArm.numeric_field_names())
ARM_DOMAINS = frozenset({ExtractionDomain.OUTCOME_DATA, ExtractionDomain.ADVERSE_EVENT})
ACCEPTABLE_CHECKS = frozenset({EvidenceCheckStatus.VERIFIED, EvidenceCheckStatus.HUMAN_ATTESTED})


class EvidenceCheck(FrozenModel):
    """Deterministic comparison of a SourceEvidence with the parsed document."""

    status: EvidenceCheckStatus
    detail: str = ""
    matched_pages: tuple[int, ...] = ()
    checked_against_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    checked_at: AwareDatetime = Field(default_factory=utc_now)

    @property
    def acceptable(self) -> bool:
        return self.status in ACCEPTABLE_CHECKS


class ExtractionItem(StrictModel):
    item_id: Identifier = Field(default_factory=new_id)
    version: int = Field(default=1, ge=1)
    supersedes: Identifier | None = None
    record_id: Identifier = Field(description="report-level identifier (a PRISMA report)")
    domain: ExtractionDomain
    field_name: NonEmptyStr = Field(description='arm field (e.g. "events") or a label')
    value_kind: Literal["numeric", "text"] = "numeric"

    # arm-level coordinates (outcome data and adverse events)
    outcome_id: Identifier | None = None
    arm_label: str | None = None
    arm_role: ArmRole | None = None
    timepoint: str | None = Field(default=None, description='verbatim, e.g. "5 years"')
    follow_up: FollowUpTimepoint | None = None
    estimate_type: EstimateType | None = None
    estimate_scale: Literal["percent", "proportion"] | None = None
    ci_level: float | None = Field(default=None, gt=0, lt=100)

    data_status: DataStatus
    reported_text: str | None = Field(default=None, description="number exactly as printed")
    value: float | None = None
    # Session 9 (M5): the recorded reading of an ambiguous printed number; None = default rules
    number_format: NumberFormat | None = None
    unit: str | None = None
    text_value: str | None = None
    evidence: SourceEvidence | None = None
    verification: EvidenceCheck | None = None
    flags: list[str] = Field(default_factory=list, description="discrepancy / attention flags")
    status_note: str | None = None

    proposed_by: ActorType
    proposer_identity: NonEmptyStr
    proposed_at: AwareDatetime = Field(default_factory=utc_now)

    review_status: ReviewStatus = ReviewStatus.PENDING
    reviewed_by: str | None = None
    reviewer_id: Identifier | None = None
    reviewed_at: AwareDatetime | None = None
    review_note: str | None = None

    @model_validator(mode="after")
    def _rules(self) -> ExtractionItem:
        if self.data_status == DataStatus.DERIVED:
            raise ValueError(
                "derived values are not authorised in Stage 4; deterministic derivations "
                "belong to the statistics engine with documented methods"
            )
        self._coordinates()
        if self.data_status == DataStatus.REPORTED:
            self._reported()
        elif self.value is not None:
            raise ValueError(
                f"{self.data_status} items carry no value (missing data is never zero)"
            )
        stamps = (self.reviewed_by, self.reviewer_id, self.reviewed_at)
        if self.review_status == ReviewStatus.PENDING:
            if any(v is not None for v in stamps):
                raise ValueError("pending items cannot carry review stamps")
        elif any(v is None for v in stamps):
            raise ValueError("reviewed items need reviewed_by, reviewer_id and reviewed_at")
        if (
            self.review_status == ReviewStatus.ACCEPTED
            and self.data_status == DataStatus.REPORTED
            and (self.verification is None or not self.verification.acceptable)
        ):
            raise ValueError(
                "a reported value can only be accepted once its evidence is verified against "
                "the document (or explicitly attested by a human)"
            )
        return self

    def _coordinates(self) -> None:
        if self.domain in ARM_DOMAINS and self.value_kind == "numeric":
            if self.field_name not in ARM_FIELDS:
                raise ValueError(f"{self.field_name!r} is not an arm-level numeric field")
            if not (self.outcome_id and self.arm_label and self.arm_role):
                raise ValueError("arm-level items need outcome_id, arm_label and arm_role")
        if self.field_name in ("reported_estimate", "ci_lower", "ci_upper") and (
            self.estimate_type is None
        ):
            raise ValueError("estimates need estimate_type (crude vs Kaplan-Meier etc.)")

    def _reported(self) -> None:
        if self.evidence is None:
            raise ValueError("reported items need SourceEvidence")
        if self.value_kind == "numeric":
            if self.reported_text is None or self.value is None:
                raise ValueError("reported numeric items need reported_text and value")
            # Reuse the core grounding rule: value == parse(reported_text), text in quote.
            ExtractedValue(
                value=self.value,
                reported_text=self.reported_text,
                unit=self.unit,
                evidence=self.evidence,
                number_format=self.number_format,
            )
            if not math.isfinite(self.value):
                raise ValueError("value must be finite")
        elif not (self.text_value and self.text_value.strip()):
            raise ValueError("reported text items need text_value")

    @property
    def arm_key(self) -> tuple[str, str, str, str]:
        """Grouping key for assembling arms (never mixes timepoints or estimate types)."""
        return (
            self.record_id,
            self.outcome_id or "",
            self.arm_label or "",
            "|".join((self.timepoint or "", self.estimate_type or "")),
        )

    def to_extracted_value(self) -> ExtractedValue:
        if self.data_status != DataStatus.REPORTED or self.value is None or not self.evidence:
            raise ValueError("only reported numeric items become ExtractedValues")
        assert self.reported_text is not None
        return ExtractedValue(
            value=self.value, reported_text=self.reported_text, unit=self.unit,
            evidence=self.evidence, number_format=self.number_format,
        )  # fmt: skip

    @model_serializer(mode="wrap")
    def _omit_absent_format(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Session 9 review (P1): see ``ExtractedValue``; items stored before Session 9 keep
        their serialised form and state hashes (arm bases, Gate 3 fingerprint)."""
        data: dict[str, Any] = handler(self)
        if self.number_format is None:
            data.pop("number_format", None)
        return data

    @property
    def format_status(self) -> NumberFormatStatus | None:
        """How the stored value reads its printed text (``None`` for non-numeric items)."""
        if self.data_status != DataStatus.REPORTED or self.value is None or not self.reported_text:
            return None
        return number_format_status(self.reported_text, self.value, self.number_format)


class ExtractionReviewEvent(FrozenModel):
    """Append-only history entry for one extraction item."""

    event_id: Identifier = Field(default_factory=new_id)
    item_id: Identifier
    action: Literal[
        "proposed", "accepted", "rejected", "corrected", "arms_built", "arms_invalidated"
    ]
    actor_type: ActorType
    actor: NonEmptyStr
    reviewer_id: Identifier | None = None
    note: str | None = None
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    occurred_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _human_actions_are_authenticated(self) -> ExtractionReviewEvent:
        if self.action != "proposed" and (
            self.actor_type != ActorType.HUMAN or self.reviewer_id is None
        ):
            raise ValueError(f"{self.action!r} requires an authenticated human reviewer")
        return self


class StudyReportLink(StrictModel):
    """Maps a report (record) to a study. Several reports may describe one cohort.

    Unlinked reports are their own study (``study_id == record_id``).
    """

    record_id: Identifier
    study_id: Identifier
    is_primary_report: bool = False
    linked_by: NonEmptyStr
    reviewer_id: Identifier
    linked_at: AwareDatetime = Field(default_factory=utc_now)
    note: str | None = None


class QuantitativeUnavailability(StrictModel):
    """An included report is eligible (qualitative synthesis) but contributes no data to
    ``outcome_id`` (or to any quantitative analysis when ``outcome_id`` is None)."""

    unavailability_id: Identifier = Field(default_factory=new_id)
    record_id: Identifier
    outcome_id: Identifier | None = None
    reason_code: Literal[
        "not_reported",
        "insufficient_detail",
        "incompatible_measure",
        "overlapping_cohort",
        "figure_only",
        "other",
    ]
    reason: NonEmptyStr
    recorded_by: NonEmptyStr
    reviewer_id: Identifier
    recorded_at: AwareDatetime = Field(default_factory=utc_now)


class CohortContributionDecision(StrictModel):
    """A human decision that several reports of one study/cohort contribute *different*
    participants to ``outcome_id`` (e.g. non-overlapping recruitment periods or centres),
    so their arms may be pooled together without double-counting.

    The alternative resolution — "use one report" — is recorded as
    :class:`QuantitativeUnavailability` (``reason_code="overlapping_cohort"``) for every
    redundant report.
    """

    decision_id: Identifier = Field(default_factory=new_id)
    study_id: Identifier
    outcome_id: Identifier
    record_ids: list[Identifier] = Field(min_length=2)
    decision: Literal["non_overlapping_samples"] = "non_overlapping_samples"
    justification: NonEmptyStr
    decided_by: NonEmptyStr
    reviewer_id: Identifier
    decided_at: AwareDatetime = Field(default_factory=utc_now)
