"""Session 5 contracts: the append-only search log (PRISMA-S) and protocol events.

A :class:`SearchRun` is written only *after* something actually happened: an API call
returned (preview or execution), a file was parsed and its records stored (import), or an
authenticated human recorded a decision (not searched, acknowledgement). A search that
was not executed or imported can therefore never appear as one.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import Field, model_validator

from src.core.enums import ActorType, SearchRunKind, SearchRunStatus, SourceDatabase
from src.core.schemas.base import (
    AwareDatetime,
    FrozenModel,
    Identifier,
    NonEmptyStr,
    new_id,
    utc_now,
)

Sha256 = str


class SearchFileRef(FrozenModel):
    """A preserved file: an imported export, or a raw API response body."""

    role: Literal["import_file", "api_response"]
    filename: NonEmptyStr = Field(description="stored, content-addressed file name")
    original_name: str | None = None
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    request: str | None = Field(default=None, description="method and URL (api_key removed)")


class DatePartition(FrozenModel):
    """One publication-date slice of a PubMed search partitioned below 10,000 records."""

    mindate: NonEmptyStr
    maxdate: NonEmptyStr
    count: int = Field(ge=0)


class SearchRun(FrozenModel):
    run_id: Identifier = Field(default_factory=new_id)
    kind: SearchRunKind
    status: SearchRunStatus = SearchRunStatus.COMPLETE
    database: SourceDatabase | None = None
    strategy_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    protocol_version: int | None = Field(default=None, ge=1)
    platform: str | None = None
    # Exactly what was sent (API) or documented as run in the vendor interface (import).
    query: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    endpoint: str | None = None
    query_translation: str | None = None
    executed_at: AwareDatetime = Field(default_factory=utc_now)
    searched_on: date | None = Field(
        default=None, description="date the human ran the search in the vendor interface"
    )
    hits_reported: int | None = Field(default=None, ge=0)
    records_received: int | None = Field(default=None, ge=0)
    records_saved: int = Field(default=0, ge=0)
    partitions: list[DatePartition] = Field(default_factory=list)
    import_format: Literal["ris", "bibtex", "csv"] | None = None
    column_map: dict[str, str] = Field(default_factory=dict)
    encoding: str | None = None
    files: list[SearchFileRef] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    problems: list[str] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    refers_to_run_id: Identifier | None = None
    actor: NonEmptyStr
    actor_type: ActorType
    reviewer_id: Identifier | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _shape(self) -> SearchRun:
        identifying = self.kind in (SearchRunKind.EXECUTION, SearchRunKind.IMPORT)
        needs_query = self.kind in (SearchRunKind.PREVIEW, SearchRunKind.EXECUTION,
                                    SearchRunKind.IMPORT)  # fmt: skip
        if needs_query and (self.database is None or not (self.query and self.query.strip())):
            raise ValueError(f"a {self.kind} needs the database and the exact query")
        if identifying and self.status == SearchRunStatus.COMPLETE and self.problems:
            raise ValueError("a run with problems cannot be complete")
        if self.status == SearchRunStatus.COUNT_MISMATCH and not self.problems:
            raise ValueError("a count mismatch must state the discrepancy")
        if self.status == SearchRunStatus.FAILED and self.records_saved:
            raise ValueError("a failed run cannot have saved records")
        if self.kind == SearchRunKind.IMPORT:
            if self.searched_on is None or self.import_format is None:
                raise ValueError("an import needs the search date and the file format")
            if not any(f.role == "import_file" for f in self.files):
                raise ValueError("an import must reference the preserved import file")
        human_only = (
            SearchRunKind.EXECUTION, SearchRunKind.IMPORT, SearchRunKind.NOT_SEARCHED,
            SearchRunKind.ACKNOWLEDGEMENT, SearchRunKind.DEDUPLICATION,
        )  # fmt: skip
        if self.kind in human_only and (
            self.actor_type != ActorType.HUMAN or self.reviewer_id is None
        ):
            raise ValueError(f"a {self.kind} must be performed by an authenticated reviewer")
        needs_note = self.kind in (SearchRunKind.NOT_SEARCHED, SearchRunKind.ACKNOWLEDGEMENT)
        if needs_note and not (self.note and self.note.strip()):
            raise ValueError(f"a {self.kind} entry requires a reason")
        if self.kind == SearchRunKind.NOT_SEARCHED and self.database is None:
            raise ValueError("name the database that was not searched")
        if self.kind == SearchRunKind.ACKNOWLEDGEMENT and self.refers_to_run_id is None:
            raise ValueError("an acknowledgement must refer to a run")
        return self

    @property
    def identifies_records(self) -> bool:
        return (
            self.kind in (SearchRunKind.EXECUTION, SearchRunKind.IMPORT)
            and self.status != SearchRunStatus.FAILED
        )


class ProtocolEvent(FrozenModel):
    """Append-only history of protocol drafting, verification and approval."""

    event_id: Identifier = Field(default_factory=new_id)
    protocol_id: Identifier
    protocol_version: int = Field(ge=1)
    action: Literal[
        "draft_saved", "mesh_verified", "syntax_validated", "syntax_attested", "approved"
    ]
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    material_changes: dict[str, list[str]] = Field(default_factory=dict)
    invalidated: list[str] = Field(default_factory=list)
    actor: NonEmptyStr
    actor_type: ActorType
    reviewer_id: Identifier | None = None
    note: str | None = None
    occurred_at: AwareDatetime = Field(default_factory=utc_now)
