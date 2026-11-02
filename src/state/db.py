"""SQLAlchemy 2.0 persistence models.

Design choices:

* Query-relevant fields are real columns (with indexes/constraints); the full
  validated Pydantic payload is stored alongside as JSON so a record can be
  re-hydrated losslessly and re-validated on load.
* ``SourceEvidence`` is its own table, and ``extracted_values.evidence_id`` is
  ``NOT NULL`` — the database itself refuses an extracted number without provenance.
* ``state_transitions``, ``gate_decisions`` and ``source_evidence`` are
  append-only: ORM-level updates and deletes raise.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
    create_engine,
    event,
)
from sqlalchemy.engine import Dialect, Engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

from src.core.exceptions import SysRevError


class UTCDateTime(TypeDecorator[datetime]):
    """Stores timezone-aware datetimes and returns them as UTC-aware (SQLite drops tzinfo)."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("naive datetimes are not allowed")
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    type_annotation_map: ClassVar[dict[Any, Any]] = {datetime: UTCDateTime, dict[str, Any]: JSON}


class ProjectORM(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str] = mapped_column(Text)
    phase: Mapped[str] = mapped_column(String(64), index=True)
    prisma_counts: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    stats_output_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    manuscript_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=_now)
    updated_at: Mapped[datetime] = mapped_column(default=_now, onupdate=_now)
    version_id: Mapped[int] = mapped_column(Integer, nullable=False)

    # optimistic concurrency control
    __mapper_args__ = {"version_id_col": version_id}  # noqa: RUF012

    transitions: Mapped[list[StateTransitionORM]] = relationship(
        back_populates="project", order_by="StateTransitionORM.seq"
    )


class ProtocolORM(Base):
    __tablename__ = "protocols"
    __table_args__ = (UniqueConstraint("project_id", "version"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    version: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16))
    approved_by: Mapped[str | None] = mapped_column(Text, nullable=True)
    approved_at: Mapped[datetime | None] = mapped_column(nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


class StudyRecordORM(Base):
    __tablename__ = "study_records"
    __table_args__ = (
        Index("ix_records_project_doi", "project_id", "doi"),
        Index("ix_records_project_pmid", "project_id", "pmid"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    source_database: Mapped[str] = mapped_column(String(32))
    pmid: Mapped[str | None] = mapped_column(String(16), nullable=True)
    pmcid: Mapped[str | None] = mapped_column(String(16), nullable=True)
    doi: Mapped[str | None] = mapped_column(String(255), nullable=True)
    title: Mapped[str] = mapped_column(Text)
    normalized_title: Mapped[str] = mapped_column(Text)
    publication_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    first_author_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    duplicate_of: Mapped[str | None] = mapped_column(
        ForeignKey("study_records.id"), nullable=True, index=True
    )
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


class ScreeningDecisionORM(Base):
    __tablename__ = "screening_decisions"
    __table_args__ = (UniqueConstraint("record_id", "reviewer", "stage"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    record_id: Mapped[str] = mapped_column(ForeignKey("study_records.id"), index=True)
    reviewer: Mapped[str] = mapped_column(String(32))
    reviewer_identity: Mapped[str] = mapped_column(Text)
    is_human: Mapped[bool] = mapped_column(Boolean)
    stage: Mapped[str] = mapped_column(String(32))
    verdict: Mapped[str] = mapped_column(String(16))
    exclusion_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    rationale: Mapped[str] = mapped_column(Text)
    decided_at: Mapped[datetime] = mapped_column()
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


class ScreeningConflictORM(Base):
    __tablename__ = "screening_conflicts"
    __table_args__ = (UniqueConstraint("record_id", "stage"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    record_id: Mapped[str] = mapped_column(ForeignKey("study_records.id"))
    stage: Mapped[str] = mapped_column(String(32))
    resolved_verdict: Mapped[str | None] = mapped_column(String(16), nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


class CohortOverlapWarningORM(Base):
    __tablename__ = "cohort_overlap_warnings"
    __table_args__ = (UniqueConstraint("project_id", "record_id_a", "record_id_b"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    record_id_a: Mapped[str] = mapped_column(ForeignKey("study_records.id"))
    record_id_b: Mapped[str] = mapped_column(ForeignKey("study_records.id"))
    overlap_score: Mapped[float] = mapped_column(Float)
    severity: Mapped[str] = mapped_column(String(16))
    resolution: Mapped[str] = mapped_column(String(48))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


class DuplicateCandidateORM(Base):
    __tablename__ = "duplicate_candidates"
    __table_args__ = (UniqueConstraint("project_id", "record_id_a", "record_id_b"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    record_id_a: Mapped[str] = mapped_column(ForeignKey("study_records.id"))
    record_id_b: Mapped[str] = mapped_column(ForeignKey("study_records.id"))
    resolution: Mapped[str] = mapped_column(String(32))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


class FullTextItemORM(Base):
    __tablename__ = "fulltext_items"
    __table_args__ = (UniqueConstraint("project_id", "record_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    record_id: Mapped[str] = mapped_column(ForeignKey("study_records.id"))
    status: Mapped[str] = mapped_column(String(32), index=True)
    pdf_filename: Mapped[str | None] = mapped_column(Text, nullable=True)
    pdf_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


class SourceEvidenceORM(Base):
    __tablename__ = "source_evidence"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    pdf_filename: Mapped[str] = mapped_column(Text)
    pdf_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    page_number: Mapped[int] = mapped_column(Integer)
    location_type: Mapped[str] = mapped_column(String(16))
    location_identifier: Mapped[str] = mapped_column(Text)
    verbatim_quote: Mapped[str] = mapped_column(Text)
    extracted_by: Mapped[str] = mapped_column(String(16))
    extractor_identity: Mapped[str] = mapped_column(Text)
    extracted_at: Mapped[datetime] = mapped_column()


class ExtractedArmORM(Base):
    __tablename__ = "extracted_arms"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    record_id: Mapped[str] = mapped_column(ForeignKey("study_records.id"), index=True)
    outcome_id: Mapped[str] = mapped_column(String(64))
    arm_label: Mapped[str] = mapped_column(Text)
    arm_role: Mapped[str] = mapped_column(String(16))
    timepoint: Mapped[str | None] = mapped_column(Text, nullable=True)
    estimate_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    estimate_scale: Mapped[str | None] = mapped_column(String(16), nullable=True)
    ci_level: Mapped[float | None] = mapped_column(Float, nullable=True)
    follow_up: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    approved_by_human: Mapped[bool] = mapped_column(Boolean, default=False)
    approved_by: Mapped[str | None] = mapped_column(Text, nullable=True)
    approved_at: Mapped[datetime | None] = mapped_column(nullable=True)

    values: Mapped[list[ExtractedValueORM]] = relationship(
        back_populates="arm", cascade="all, delete-orphan"
    )


class ExtractedValueORM(Base):
    __tablename__ = "extracted_values"
    __table_args__ = (UniqueConstraint("arm_id", "field_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    arm_id: Mapped[str] = mapped_column(ForeignKey("extracted_arms.id"), index=True)
    field_name: Mapped[str] = mapped_column(String(32))
    value: Mapped[float] = mapped_column(Float)
    reported_text: Mapped[str] = mapped_column(Text)
    unit: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Session 9 (M5): recorded reading of an ambiguous printed number (None = default rules)
    number_format: Mapped[str | None] = mapped_column(String(20), nullable=True)
    evidence_id: Mapped[str] = mapped_column(ForeignKey("source_evidence.id"), nullable=False)

    arm: Mapped[ExtractedArmORM] = relationship(back_populates="values")
    evidence: Mapped[SourceEvidenceORM] = relationship()


class RiskOfBiasORM(Base):
    __tablename__ = "risk_of_bias"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    record_id: Mapped[str] = mapped_column(ForeignKey("study_records.id"))
    tool: Mapped[str] = mapped_column(String(16))
    overall: Mapped[str | None] = mapped_column(String(32), nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


class ReviewerORM(Base):
    """An enrolled human reviewer. Only a salted scrypt hash of the token is stored."""

    __tablename__ = "reviewers"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    display_name: Mapped[str] = mapped_column(Text)
    salt: Mapped[str] = mapped_column(String(64))
    token_hash: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(default=_now)
    revoked_at: Mapped[datetime | None] = mapped_column(nullable=True)


class GateDecisionORM(Base):
    __tablename__ = "gate_decisions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    gate: Mapped[str] = mapped_column(String(16))
    verdict: Mapped[str] = mapped_column(String(16))
    reviewer_id: Mapped[str] = mapped_column(ForeignKey("reviewers.id"), nullable=False)
    reviewer_name: Mapped[str] = mapped_column(Text)
    comments: Mapped[str | None] = mapped_column(Text, nullable=True)
    decided_at: Mapped[datetime] = mapped_column()


class StateTransitionORM(Base):
    __tablename__ = "state_transitions"

    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id: Mapped[str] = mapped_column(String(64), unique=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    from_phase: Mapped[str] = mapped_column(String(64))
    to_phase: Mapped[str] = mapped_column(String(64))
    actor_type: Mapped[str] = mapped_column(String(16))
    actor: Mapped[str] = mapped_column(Text)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    gate_decision_id: Mapped[str | None] = mapped_column(
        ForeignKey("gate_decisions.id"), nullable=True
    )
    occurred_at: Mapped[datetime] = mapped_column()

    project: Mapped[ProjectORM] = relationship(back_populates="transitions")


class FullTextDocumentORM(Base):
    """A stored full-text file (PDF or structured XML) linked to one report."""

    __tablename__ = "fulltext_documents"
    __table_args__ = (UniqueConstraint("project_id", "record_id", "sha256"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    record_id: Mapped[str] = mapped_column(ForeignKey("study_records.id"), index=True)
    kind: Mapped[str] = mapped_column(String(8))
    filename: Mapped[str] = mapped_column(Text)
    original_filename: Mapped[str | None] = mapped_column(Text, nullable=True)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    size_bytes: Mapped[int] = mapped_column(Integer)
    source_kind: Mapped[str] = mapped_column(String(16))
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    recorded_by: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=_now)


class ExtractionItemORM(Base):
    __tablename__ = "extraction_items"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    record_id: Mapped[str] = mapped_column(ForeignKey("study_records.id"), index=True)
    review_status: Mapped[str] = mapped_column(String(16), index=True)
    data_status: Mapped[str] = mapped_column(String(16))
    verification_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


class ExtractionReviewEventORM(Base):
    __tablename__ = "extraction_review_events"

    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id: Mapped[str] = mapped_column(String(64), unique=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    item_id: Mapped[str] = mapped_column(String(64), index=True)
    action: Mapped[str] = mapped_column(String(16))
    actor: Mapped[str] = mapped_column(Text)
    reviewer_id: Mapped[str | None] = mapped_column(ForeignKey("reviewers.id"), nullable=True)
    occurred_at: Mapped[datetime] = mapped_column()
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


class StudyReportLinkORM(Base):
    __tablename__ = "study_report_links"
    __table_args__ = (UniqueConstraint("project_id", "record_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    record_id: Mapped[str] = mapped_column(ForeignKey("study_records.id"))
    study_id: Mapped[str] = mapped_column(String(64), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


class QuantitativeUnavailabilityORM(Base):
    __tablename__ = "quantitative_unavailability"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    record_id: Mapped[str] = mapped_column(ForeignKey("study_records.id"), index=True)
    outcome_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


class CohortContributionDecisionORM(Base):
    __tablename__ = "cohort_contribution_decisions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    study_id: Mapped[str] = mapped_column(String(64))
    outcome_id: Mapped[str] = mapped_column(String(64))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


class AuditReportORM(Base):
    __tablename__ = "audit_reports"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    manuscript_version: Mapped[int] = mapped_column(Integer)
    passed: Mapped[bool] = mapped_column(Boolean)
    generated_at: Mapped[datetime] = mapped_column()
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


class AnalysisPlanORM(Base):
    """Stage 5 analysis plans. Versions are append-only; approval is a new version."""

    __tablename__ = "analysis_plans"

    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id: Mapped[str] = mapped_column(String(64), index=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    version: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16))
    approved_reviewer_id: Mapped[str | None] = mapped_column(
        ForeignKey("reviewers.id"), nullable=True
    )
    sha256: Mapped[str] = mapped_column(String(64))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(default=_now)


class StatsOutputORM(Base):
    """Every ``stats_output.json`` produced by the engine (append-only)."""

    __tablename__ = "stats_outputs"

    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id: Mapped[str] = mapped_column(String(64), unique=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    plan_id: Mapped[str] = mapped_column(String(64))
    plan_sha256: Mapped[str] = mapped_column(String(64))
    input_sha256: Mapped[str] = mapped_column(String(64))
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    content: Mapped[str] = mapped_column(Text, doc="canonical JSON exactly as hashed")
    created_at: Mapped[datetime] = mapped_column(default=_now)


class ProjectAISettingsORM(Base):
    """Per-project model configuration per AI task (append-only versions; no secrets)."""

    __tablename__ = "project_ai_settings"

    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    version: Mapped[int] = mapped_column(Integer)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(default=_now)


class LLMCallORM(Base):
    """Audit trail of every LLM call: model, effort, tokens, cost estimate (append-only)."""

    __tablename__ = "llm_calls"

    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id: Mapped[str] = mapped_column(String(64), unique=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    task: Mapped[str] = mapped_column(String(64), index=True)
    provider: Mapped[str] = mapped_column(String(32))
    requested_model: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(32))
    call_key: Mapped[str] = mapped_column(String(64), index=True)
    total_tokens: Mapped[int] = mapped_column(Integer)
    cost_usd_estimate: Mapped[float | None] = mapped_column(Float, nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(default=_now)


class LLMCallOutcomeORM(Base):
    """Marks that a call's reply became a validated, persisted workflow result
    (append-only). Billing rows in ``llm_calls`` are written regardless of outcome."""

    __tablename__ = "llm_call_outcomes"

    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    call_id: Mapped[str] = mapped_column(ForeignKey("llm_calls.id"), unique=True)
    call_key: Mapped[str] = mapped_column(String(64), index=True)
    result_ref: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=_now)


class AIVerificationORM(Base):
    """Advisory second-model checks of extracted values (append-only; never approvals)."""

    __tablename__ = "ai_verifications"

    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id: Mapped[str] = mapped_column(String(64), unique=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    item_id: Mapped[str] = mapped_column(String(64), index=True)
    verdict: Mapped[str] = mapped_column(String(32))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(default=_now)


class ProtocolEventORM(Base):
    """Protocol drafting, verification and approval history (append-only, full payload)."""

    __tablename__ = "protocol_events"

    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id: Mapped[str] = mapped_column(String(64), unique=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    protocol_version: Mapped[int] = mapped_column(Integer)
    action: Mapped[str] = mapped_column(String(32))
    reviewer_id: Mapped[str | None] = mapped_column(ForeignKey("reviewers.id"), nullable=True)
    event: Mapped[dict[str, Any]] = mapped_column(JSON)
    protocol_payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(default=_now)


class SearchRunORM(Base):
    """The search log: previews, executions, imports, human declarations (append-only)."""

    __tablename__ = "search_runs"

    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id: Mapped[str] = mapped_column(String(64), unique=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    kind: Mapped[str] = mapped_column(String(32), index=True)
    status: Mapped[str] = mapped_column(String(32))
    database: Mapped[str | None] = mapped_column(String(32), nullable=True)
    strategy_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    reviewer_id: Mapped[str | None] = mapped_column(ForeignKey("reviewers.id"), nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(default=_now)


class SearchRawRecordORM(Base):
    """The record exactly as parsed from the source, before any normalisation (append-only)."""

    __tablename__ = "search_raw_records"
    __table_args__ = (UniqueConstraint("record_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("search_runs.id"), index=True)
    record_id: Mapped[str] = mapped_column(ForeignKey("study_records.id"))
    position: Mapped[int] = mapped_column(Integer)
    raw: Mapped[dict[str, Any]] = mapped_column(JSON)


class ManuscriptVersionORM(Base):
    """Stage 6 manuscript versions (append-only; edits create a new version)."""

    __tablename__ = "manuscript_versions"
    __table_args__ = (UniqueConstraint("project_id", "version"),)

    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    version: Mapped[int] = mapped_column(Integer)
    content_sha256: Mapped[str] = mapped_column(String(64), index=True)
    actor_type: Mapped[str] = mapped_column(String(16))
    reviewer_id: Mapped[str | None] = mapped_column(ForeignKey("reviewers.id"), nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(default=_now)


class ReferenceEntryORM(Base):
    """Background literature imported by identifier (append-only; latest row per key wins)."""

    __tablename__ = "reference_entries"

    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    ref_key: Mapped[str] = mapped_column(String(64), index=True)
    reviewer_id: Mapped[str | None] = mapped_column(ForeignKey("reviewers.id"), nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(default=_now)


class ReferenceVerificationORM(Base):
    """Live PubMed/Crossref identity checks, including failed ('unavailable') attempts."""

    __tablename__ = "reference_verifications"

    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id: Mapped[str] = mapped_column(String(64), unique=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    ref_key: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(16))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(default=_now)


class EvidenceExcerptORM(Base):
    __tablename__ = "evidence_excerpts"

    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id: Mapped[str] = mapped_column(String(64), unique=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    ref_key: Mapped[str] = mapped_column(String(64), index=True)
    reviewer_id: Mapped[str] = mapped_column(ForeignKey("reviewers.id"))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(default=_now)


class FindingAcknowledgementORM(Base):
    __tablename__ = "finding_acknowledgements"

    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id: Mapped[str] = mapped_column(String(64), unique=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    audit_report_id: Mapped[str] = mapped_column(ForeignKey("audit_reports.id"), index=True)
    finding_id: Mapped[str] = mapped_column(String(64))
    reviewer_id: Mapped[str] = mapped_column(ForeignKey("reviewers.id"))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(default=_now)


class ManuscriptApprovalORM(Base):
    __tablename__ = "manuscript_approvals"

    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id: Mapped[str] = mapped_column(String(64), unique=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    manuscript_version: Mapped[int] = mapped_column(Integer)
    audit_report_id: Mapped[str] = mapped_column(ForeignKey("audit_reports.id"))
    reviewer_id: Mapped[str] = mapped_column(ForeignKey("reviewers.id"))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(default=_now)


class ClaimReviewORM(Base):
    """Advisory model opinions on claim support (never human assessments)."""

    __tablename__ = "claim_reviews"

    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id: Mapped[str] = mapped_column(String(64), unique=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    audit_report_id: Mapped[str] = mapped_column(String(64), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(default=_now)


class ReportingRecordORM(Base):
    """Session 7 reporting records (append-only; the latest record per subject is current):
    checklist decisions, GRADE assessments, synthesis groups and SWiM applicability."""

    __tablename__ = "reporting_records"

    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id: Mapped[str] = mapped_column(String(64), unique=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    kind: Mapped[str] = mapped_column(String(32), index=True)
    subject: Mapped[str] = mapped_column(String(128), index=True)
    reviewer_id: Mapped[str | None] = mapped_column(ForeignKey("reviewers.id"), nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    payload_sha256: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(default=_now)


class ModelCatalogueSnapshotORM(Base):
    """Session 8: candidate model catalogues (refresh / verification), never edited."""

    __tablename__ = "model_catalogue_snapshots"

    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id: Mapped[str] = mapped_column(String(64), unique=True)
    kind: Mapped[str] = mapped_column(String(32))
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    parent_sha256: Mapped[str] = mapped_column(String(64))
    created_by: Mapped[str] = mapped_column(Text)
    reviewer_id: Mapped[str | None] = mapped_column(ForeignKey("reviewers.id"), nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(default=_now)


class ModelCatalogueActivationORM(Base):
    """Session 8: which catalogue snapshot is active (the latest row), and who activated it."""

    __tablename__ = "model_catalogue_activations"

    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id: Mapped[str] = mapped_column(String(64), unique=True)
    snapshot_id: Mapped[str] = mapped_column(ForeignKey("model_catalogue_snapshots.id"))
    sha256: Mapped[str] = mapped_column(String(64))
    previous_sha256: Mapped[str] = mapped_column(String(64))
    reviewer_id: Mapped[str] = mapped_column(ForeignKey("reviewers.id"))
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    activated_at: Mapped[datetime] = mapped_column(default=_now)


class GovernanceRecordORM(Base):
    """Session 8 project records (append-only; meaning by ``kind``): protocol amendment
    proposals, decisions and re-reviews, arm invalidations after corrections, and the history
    of report-to-study links. Earlier tables are not altered (no schema migration needed)."""

    __tablename__ = "governance_records"

    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id: Mapped[str] = mapped_column(String(64), unique=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    kind: Mapped[str] = mapped_column(String(32), index=True)
    subject: Mapped[str] = mapped_column(String(128), index=True)
    reviewer_id: Mapped[str | None] = mapped_column(ForeignKey("reviewers.id"), nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    payload_sha256: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(default=_now)


APPEND_ONLY_MODELS: tuple[type[Base], ...] = (
    ProtocolEventORM,
    SearchRunORM,
    SearchRawRecordORM,
    StateTransitionORM,
    GateDecisionORM,
    SourceEvidenceORM,
    ExtractionReviewEventORM,
    AnalysisPlanORM,
    StatsOutputORM,
    ProjectAISettingsORM,
    LLMCallORM,
    LLMCallOutcomeORM,
    AIVerificationORM,
    AuditReportORM,
    ManuscriptVersionORM,
    ReferenceEntryORM,
    ReferenceVerificationORM,
    EvidenceExcerptORM,
    FindingAcknowledgementORM,
    ManuscriptApprovalORM,
    ClaimReviewORM,
    ReportingRecordORM,
    ModelCatalogueSnapshotORM,
    ModelCatalogueActivationORM,
    GovernanceRecordORM,
)


class AppendOnlyViolationError(RuntimeError):
    """Raised when code attempts to modify or delete an append-only audit record."""


def _forbid_mutation(mapper: Any, connection: Any, target: Base) -> None:
    raise AppendOnlyViolationError(
        f"{type(target).__name__} rows are append-only and cannot be updated or deleted"
    )


for _model in APPEND_ONLY_MODELS:
    event.listen(_model, "before_update", _forbid_mutation)
    event.listen(_model, "before_delete", _forbid_mutation)


def _enable_sqlite_foreign_keys(dbapi_connection: Any, connection_record: Any) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def make_engine(url: str = "sqlite:///:memory:", *, echo: bool = False) -> Engine:
    kwargs: dict[str, Any] = {"echo": echo}
    if url.startswith("sqlite") and ":memory:" in url:
        from sqlalchemy.pool import StaticPool

        kwargs |= {"connect_args": {"check_same_thread": False}, "poolclass": StaticPool}
    engine = create_engine(url, **kwargs)
    if engine.dialect.name == "sqlite":
        event.listen(engine, "connect", _enable_sqlite_foreign_keys)
    return engine


def prepare_sqlite_path(url: str) -> None:
    """Create the parent directory of a file-based SQLite URL with owner-only permissions.

    The database holds reviewer credential hashes and the audit trail; other OS users
    should not be able to read or write it. On POSIX ``mkdir(mode=0o700)`` creates the
    directory protected in one step; an existing directory is used as it is (unchanged).

    On Windows the mode has no reliable effect, so protection is a separate step that can
    fail (Session 10). It must fail closed across restarts (review of PR #15, P1): a new
    database is created only in a directory **verified** owner-only from its security
    descriptor (``src.state.acl``):

    * directories this call creates are restricted to the current user and verified; if
      that fails, the (empty) directories it created are removed again and the error is
      raised, so a retry starts from the beginning instead of finding a directory;
    * an existing directory without the database file is verified; an empty one is
      restricted first (an interrupted initialisation, or a folder made for the database);
      a non-empty one is never changed: if it is not owner-only, startup is refused;
    * an existing database file is opened as before; the preflight reports its directory.
    """
    if not url.startswith("sqlite:///") or ":memory:" in url:
        return
    path = Path(url.removeprefix("sqlite:///"))
    directory = path.parent
    if not _is_windows():
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        return
    created = _create_directories(directory)
    try:
        if not created:
            if path.exists():
                return
            if any(directory.iterdir()):
                _require_owner_only(directory, existing=True)
                return
        restrict_directory_to_current_user(directory)
        _require_owner_only(directory, existing=False)
    except BaseException:
        _remove_created(created)
        raise


def _is_windows() -> bool:
    return os.name == "nt"


def _create_directories(directory: Path) -> list[Path]:
    """``mkdir -p`` that returns the directories it created, outermost first."""
    missing = []
    d = directory
    while not d.exists():
        missing.append(d)
        if d.parent == d:
            break
        d = d.parent
    created = []
    try:
        for d in reversed(missing):
            try:
                d.mkdir(mode=0o700)
            except FileExistsError:
                if not d.is_dir():
                    raise
                continue  # created concurrently by someone else: not ours to remove
            created.append(d)
    except BaseException:
        _remove_created(created)
        raise
    return created


def _remove_created(created: list[Path]) -> None:
    """Remove the directories a failed initialisation created, innermost first, only while
    they are empty (``rmdir`` never deletes content)."""
    for d in reversed(created):
        try:
            d.rmdir()
        except OSError:
            return


def _require_owner_only(directory: Path, *, existing: bool) -> None:
    problems = windows_directory_problems(directory)
    if problems:
        how = (
            "It already exists and contains files, so it is not changed automatically. "
            "Choose a database location in a new folder (it is then created and restricted "
            "automatically) or restrict this folder to your account."
            if existing
            else "Its owner-only protection could not be applied or verified."
        )
        raise SysRevError(
            f"refusing to create the database in {directory}: it is not restricted to the "
            f"current user ({'; '.join(problems)}). {how}"
        )


def windows_directory_problems(directory: Path) -> list[str]:
    """Why ``directory`` is not owner-only on Windows (empty: verified owner-only)."""
    from src.state.acl import owner_only_problems, windows_sddl

    try:
        return owner_only_problems(windows_sddl(directory), current_user_sid())
    except (OSError, SysRevError, subprocess.SubprocessError) as exc:
        return [f"the security descriptor could not be read ({type(exc).__name__}: {exc})"]


def directory_protection_problems(directory: Path) -> list[str]:
    """Why ``directory`` is accessible to other accounts (empty: owner-only). POSIX: no
    group/other mode bits; Windows: ``windows_directory_problems``."""
    if _is_windows():
        return windows_directory_problems(directory)
    import stat

    mode = stat.S_IMODE(directory.stat().st_mode)
    return [] if mode & 0o077 == 0 else [f"mode {oct(mode)} gives access to group/other"]


def _windows_tool(name: str) -> str:
    """Absolute path of a Windows system tool (never resolved through PATH)."""
    return str(Path(os.environ.get("SYSTEMROOT", r"C:\Windows")) / "System32" / name)


def current_user_sid(run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> str:
    """SID of the current Windows user (``whoami /user``)."""
    out = run(
        [_windows_tool("whoami.exe"), "/user", "/fo", "csv", "/nh"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    ).stdout
    sids = re.findall(r'"(S-1-[0-9-]+)"', out)
    if len(sids) != 1:
        raise SysRevError(f"could not determine the current Windows user's SID from {out!r}")
    return str(sids[0])


def restrict_directory_to_current_user(
    directory: Path, run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run
) -> None:
    """Windows equivalent of ``chmod 0700``: remove inherited entries and grant full control
    to the current user only (inherited by files and subdirectories). Fails closed: if the
    ACL cannot be applied, the database is not opened there."""
    sid = current_user_sid(run)
    result = run(
        [
            _windows_tool("icacls.exe"),
            str(directory),
            "/inheritance:r",
            "/grant:r",
            f"*{sid}:(OI)(CI)F",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise SysRevError(
            f"could not restrict {directory} to the current user (icacls exit "
            f"{result.returncode}: {(result.stdout + result.stderr).strip()}); the database "
            "directory would be readable by other accounts. Choose a database location in "
            "your user profile or fix the folder permissions, then start again."
        )


# Nullable columns added after a table was first released. ``create_all`` never alters an
# existing table, so a database created by an earlier version gets them here; existing rows
# keep NULL, i.e. no recorded interpretation (Session 9, M5).
ADDED_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("extracted_values", "number_format", "VARCHAR(20)"),
)


def init_db(engine: Engine) -> None:
    Base.metadata.create_all(engine)
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    with engine.begin() as conn:
        for table, column, ddl in ADDED_COLUMNS:
            existing = {c["name"] for c in inspector.get_columns(table)}
            if column not in existing:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)
