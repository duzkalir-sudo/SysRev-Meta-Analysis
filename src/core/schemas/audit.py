"""Stage 6 contracts: Auditor verification report.

An :class:`AuditReport` certifies nothing on its own. ``passed`` is **derived**: it is true
only when the report is *complete* (it was produced for a specific manuscript version and
evidence fingerprint, every required check scope was run, and the coverage list of expected
reportable outputs is non-empty) **and** no check, coverage item or finding is in a
blocking status. A report that lists one verified number, one verified citation or any
selectively populated subset of checks therefore does not pass.

Findings that cannot be settled automatically (``REQUIRES_HUMAN_REVIEW``) do not make
``passed`` false, but the Stage 6 completion guard additionally requires each of them to be
acknowledged by an authenticated human, and a human sign-off of the exact manuscript version.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import Field, ModelWrapValidatorHandler, computed_field, model_validator

from src.core.enums import CheckStatus
from src.core.schemas.base import (
    DOI,
    PMID,
    AwareDatetime,
    Identifier,
    NonEmptyStr,
    StrictModel,
    new_id,
    utc_now,
)

# Statuses that block approval and cannot be acknowledged away: they must be fixed.
BLOCKING_STATUSES = frozenset(
    {
        CheckStatus.MISMATCH,
        CheckStatus.NOT_FOUND,
        CheckStatus.MISSING,
        CheckStatus.UNSUPPORTED,
        CheckStatus.NOT_CHECKED,
    }
)

# Every scope the Stage 6 Auditor runs. A report missing any of them is incomplete.
REQUIRED_SCOPES = frozenset(
    {
        "provenance",  # manuscript bound to the current evidence package
        "structure",  # required sections, abstract headings, author-input markers
        "numeric_bindings",  # every inserted number re-resolved against its source key
        "literal_numbers",  # every number typed outside a binding
        "semantic_labels",  # pooled/single-study, estimator, model, scale, significance
        "coverage",  # every expected reportable output is reported
        "consistency",  # abstract/conclusions vs Results; tables and figures
        "methods",  # methods described vs methods recorded
        "claims",  # causal / overclaiming / recommendation language
        "citations",  # identifiers, numbering, bibliography, duplicates
        "claim_support",  # whether the cited source supports the sentence
        "narrative",  # author statements without bindings/citations -> human review
    }
)


class NumericClaimCheck(StrictModel):
    """A number found in the manuscript, cross-checked against an approved source key."""

    claim_text: NonEmptyStr
    manuscript_location: NonEmptyStr
    reported_value: float
    stats_key: str | None = None
    expected_value: float | None = None
    tolerance: float = Field(default=0.0, ge=0.0)
    status: CheckStatus

    @model_validator(mode="after")
    def _consistent(self) -> NumericClaimCheck:
        if self.status not in (CheckStatus.VERIFIED, CheckStatus.MISMATCH):
            return self
        if self.stats_key is None or self.expected_value is None:
            raise ValueError("verified/mismatch checks require stats_key and expected_value")
        within = abs(self.reported_value - self.expected_value) <= self.tolerance
        if within != (self.status == CheckStatus.VERIFIED):
            raise ValueError("status contradicts reported vs expected values and tolerance")
        return self


class CitationCheck(StrictModel):
    """A manuscript citation, cross-checked against the verified PMID/DOI registry."""

    citation_text: NonEmptyStr
    manuscript_location: NonEmptyStr
    pmid: PMID = None
    doi: DOI = None
    matched_record_id: Identifier | None = None
    status: CheckStatus

    @model_validator(mode="after")
    def _consistent(self) -> CitationCheck:
        if self.status == CheckStatus.VERIFIED and self.matched_record_id is None:
            raise ValueError("verified citations must reference a record in the database")
        if not (self.pmid or self.doi) and self.status == CheckStatus.VERIFIED:
            raise ValueError("verified citations require a PMID or DOI")
        return self


class AuditFinding(StrictModel):
    """One actionable audit result, pointing to the manuscript text and the source evidence."""

    finding_id: NonEmptyStr = Field(description="deterministic hash of the finding's content")
    scope: NonEmptyStr
    status: CheckStatus
    location: NonEmptyStr = Field(description="section and sentence, e.g. 'results §3'")
    excerpt: str = Field(default="", description="the manuscript text concerned")
    source_key: str | None = Field(default=None, description="approved data/statistics key")
    source_value: str | None = Field(default=None, description="what the source says")
    message: NonEmptyStr
    action: NonEmptyStr = Field(description="what the author must do")
    origin: Literal["deterministic"] = "deterministic"

    @property
    def blocking(self) -> bool:
        return self.status in BLOCKING_STATUSES

    @property
    def needs_human(self) -> bool:
        return self.status == CheckStatus.REQUIRES_HUMAN_REVIEW


class CoverageItem(StrictModel):
    """An output the evidence package says must be reported, and where it was found."""

    key: NonEmptyStr
    description: NonEmptyStr
    required_in: list[NonEmptyStr]
    status: Literal[CheckStatus.VERIFIED, CheckStatus.MISSING]
    found_in: list[str] = Field(default_factory=list)


class AuditReport(StrictModel):
    """Outcome of one Auditor pass. ``passed`` is derived, never asserted."""

    report_id: Identifier = Field(default_factory=new_id)
    project_id: Identifier
    manuscript_version: int = Field(ge=1)
    stats_output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generated_at: AwareDatetime = Field(default_factory=utc_now)
    numeric_checks: list[NumericClaimCheck] = Field(default_factory=list)
    citation_checks: list[CitationCheck] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    # Session 6: completeness. Reports without these fields (e.g. hand-built ones) are
    # incomplete and never pass.
    manuscript_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    evidence_fingerprint: dict[str, str] = Field(default_factory=dict)
    auditor_version: str | None = None
    scopes_checked: dict[str, int] = Field(
        default_factory=dict, description="scope -> number of units examined"
    )
    coverage: list[CoverageItem] = Field(default_factory=list)
    findings: list[AuditFinding] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="wrap")
    @classmethod
    def _derived_fields_are_not_inputs(
        cls, data: Any, handler: ModelWrapValidatorHandler[AuditReport]
    ) -> AuditReport:
        """Accept serialised reports, but reject any claimed verdict that the checks contradict."""
        claimed: dict[str, Any] = {}
        if isinstance(data, dict):
            data = dict(data)
            for key in ("passed", "failures", "complete", "requires_human_review"):
                if key in data:
                    claimed[key] = data.pop(key)
        report = handler(data)
        for key, value in claimed.items():
            if getattr(report, key) != value:
                raise ValueError(f"claimed {key}={value!r} contradicts the individual checks")
        return report

    def _statuses(self) -> list[CheckStatus]:
        return [
            *(c.status for c in self.numeric_checks),
            *(c.status for c in self.citation_checks),
            *(c.status for c in self.coverage),
            *(f.status for f in self.findings),
        ]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def complete(self) -> bool:
        """Produced for an identified manuscript and evidence state, with every scope run."""
        return (
            self.manuscript_sha256 is not None
            and bool(self.evidence_fingerprint)
            and set(self.scopes_checked) >= REQUIRED_SCOPES
            and bool(self.coverage)
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        statuses = self._statuses()
        return (
            self.complete and bool(statuses) and not any(s in BLOCKING_STATUSES for s in statuses)
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def failures(self) -> int:
        return sum(s in BLOCKING_STATUSES for s in self._statuses())

    @computed_field  # type: ignore[prop-decorator]
    @property
    def requires_human_review(self) -> int:
        return sum(s == CheckStatus.REQUIRES_HUMAN_REVIEW for s in self._statuses())

    def results_sha256(self) -> str:
        """Fingerprint of everything the audit concluded (not its id or timestamp)."""
        payload = self.model_dump(
            mode="json",
            exclude={
                "report_id", "generated_at", "passed", "failures", "complete",
                "requires_human_review",
            },
        )  # fmt: skip
        text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
