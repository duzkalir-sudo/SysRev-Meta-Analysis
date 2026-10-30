"""Stage 6 contracts: Auditor Agent verification report."""

from __future__ import annotations

from typing import Any

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


class NumericClaimCheck(StrictModel):
    """A number found in the manuscript, cross-checked against ``stats_output.json``."""

    claim_text: NonEmptyStr
    manuscript_location: NonEmptyStr
    reported_value: float
    stats_key: str | None = None
    expected_value: float | None = None
    tolerance: float = Field(default=0.0, ge=0.0)
    status: CheckStatus

    @model_validator(mode="after")
    def _consistent(self) -> NumericClaimCheck:
        if self.status == CheckStatus.NOT_FOUND:
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

    @model_validator(mode="wrap")
    @classmethod
    def _derived_fields_are_not_inputs(
        cls, data: Any, handler: ModelWrapValidatorHandler[AuditReport]
    ) -> AuditReport:
        """Accept serialised reports, but reject any claimed verdict that the checks contradict."""
        claimed: dict[str, Any] = {}
        if isinstance(data, dict):
            data = dict(data)
            for key in ("passed", "failures"):
                if key in data:
                    claimed[key] = data.pop(key)
        report = handler(data)
        for key, value in claimed.items():
            if getattr(report, key) != value:
                raise ValueError(f"claimed {key}={value!r} contradicts the individual checks")
        return report

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        checks: list[NumericClaimCheck | CitationCheck] = [
            *self.numeric_checks,
            *self.citation_checks,
        ]
        return bool(checks) and all(c.status == CheckStatus.VERIFIED for c in checks)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def failures(self) -> int:
        checks: list[NumericClaimCheck | CitationCheck] = [
            *self.numeric_checks,
            *self.citation_checks,
        ]
        return sum(c.status != CheckStatus.VERIFIED for c in checks)
