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

from datetime import UTC, datetime
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


class GateDecisionORM(Base):
    __tablename__ = "gate_decisions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    gate: Mapped[str] = mapped_column(String(16))
    verdict: Mapped[str] = mapped_column(String(16))
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


class AuditReportORM(Base):
    __tablename__ = "audit_reports"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    manuscript_version: Mapped[int] = mapped_column(Integer)
    passed: Mapped[bool] = mapped_column(Boolean)
    generated_at: Mapped[datetime] = mapped_column()
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


APPEND_ONLY_MODELS: tuple[type[Base], ...] = (
    StateTransitionORM,
    GateDecisionORM,
    SourceEvidenceORM,
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


def init_db(engine: Engine) -> None:
    Base.metadata.create_all(engine)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)
