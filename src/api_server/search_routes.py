"""Session 5 endpoints: protocol editor, database searches, imports, deduplication, PRISMA.

Every state-changing endpoint requires ``Authorization: Bearer <reviewer token>``; the
acting human comes from the token. Network searches run only when a reviewer requests them
and only for strategies of the approved protocol (see ``src.search.service``).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from datetime import date
from typing import Any, Literal, TypeVar

from fastapi import APIRouter, Body, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.api_server.app import ReviewerDep, SessionDep
from src.core.exceptions import SysRevError
from src.fulltext.storage import DocumentStore
from src.search import protocol_editor as editor
from src.search.databases import DATABASES
from src.search.importers import MAX_IMPORT_BYTES, ImportFormat, parse_file
from src.search.prisma_sync import derive_prisma
from src.search.reporting import report_csv, report_markdown, search_report
from src.search.service import SearchClients, SearchService
from src.state.repository import NotFoundError, Repository

T = TypeVar("T")


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NoteBody(_Body):
    note: str = Field(min_length=1)


class OptionalNoteBody(_Body):
    note: str | None = None


class AcknowledgeBody(_Body):
    note: str = Field(min_length=1)
    run_id: str | None = None
    strategy_sha256: str | None = None


class ZeroResultBody(_Body):
    searched_on: date
    note: str = Field(min_length=1)
    restrictions_confirmed: bool = False
    restrictions_note: str | None = None


class DuplicateResolutionBody(_Body):
    resolution: Literal["distinct", "duplicate_keep_a", "duplicate_keep_b"]
    note: str | None = None


def _call(fn: Callable[[], T]) -> T:
    try:
        return fn()
    except NotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except ValidationError as exc:
        errors = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()]
        raise HTTPException(422, {"message": "invalid protocol", "errors": errors}) from exc
    except (SysRevError, ValueError) as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc


def _protocol_view(repo: Repository, project_id: str) -> dict[str, Any]:
    _call(lambda: repo.get_project(project_id))
    protocol = repo.latest_protocol(project_id)
    if protocol is None:
        return {"protocol": None, "approval_blockers": ["protocol not drafted"], "strategies": []}
    return {
        "protocol": protocol.model_dump(mode="json"),
        "approval_blockers": protocol.approval_blockers(),
        "sha256": editor.protocol_sha256(protocol),
        "strategies": [
            {"sha256": s.sha256, "database": s.database.value,
             "label": DATABASES[s.database].label if s.database in DATABASES else s.database,
             "access": DATABASES[s.database].access if s.database in DATABASES else "unknown",
             "syntax_validated": s.syntax_validated, "validation_method": s.validation_method}
            for s in protocol.search_strategies
        ],
    }  # fmt: skip


def build_search_router(store: DocumentStore) -> APIRouter:
    router = APIRouter(prefix="/projects/{project_id}")

    def service(request: Request, repo: Repository, project_id: str) -> SearchService:
        clients: SearchClients | None = getattr(request.app.state, "search_clients", None)
        return SearchService(repo, project_id, store=store, clients=clients)

    # ------------------------------------------------------------------ protocol
    @router.get("/protocol")
    def get_protocol(project_id: str, session: SessionDep) -> dict[str, Any]:
        return _protocol_view(Repository(session), project_id)

    @router.get("/protocol/history")
    def protocol_history(project_id: str, session: SessionDep) -> list[dict[str, Any]]:
        events = Repository(session).list_protocol_events(project_id)
        return [e.model_dump(mode="json") | {"status": p.status} for e, p in events]

    @router.put("/protocol")
    def save_protocol(
        project_id: str,
        session: SessionDep,
        reviewer: ReviewerDep,
        body: dict[str, Any] = Body(...),  # noqa: B008
    ) -> dict[str, Any]:
        repo = Repository(session)
        result = _call(lambda: editor.save_draft(repo, project_id, body, reviewer=reviewer))
        return _protocol_view(repo, project_id) | {
            "new_version": result.new_version,
            "material_changes": result.material_changes,
            "invalidated": result.invalidated,
        }

    @router.post("/protocol/mesh/verify")
    def verify_mesh(
        project_id: str, request: Request, session: SessionDep, reviewer: ReviewerDep
    ) -> dict[str, Any]:
        repo = Repository(session)
        svc = service(request, repo, project_id)

        async def run() -> dict[str, bool]:
            async with svc.clients.pubmed() as client:
                return await editor.verify_mesh(repo, project_id, client, reviewer=reviewer)

        results = _call(lambda: asyncio.run(run()))
        return _protocol_view(repo, project_id) | {"mesh_results": results}

    @router.post("/protocol/strategies/{sha}/preview")
    def preview(
        project_id: str, sha: str, request: Request, session: SessionDep, reviewer: ReviewerDep
    ) -> dict[str, Any]:
        repo = Repository(session)
        svc = service(request, repo, project_id)
        run = _call(lambda: asyncio.run(svc.preview(sha, reviewer=reviewer)))
        return run.model_dump(mode="json")

    @router.post("/protocol/strategies/{sha}/attest")
    def attest(
        project_id: str, sha: str, body: NoteBody, session: SessionDep, reviewer: ReviewerDep
    ) -> dict[str, Any]:
        repo = Repository(session)

        def run() -> Any:
            protocol = repo.latest_protocol(project_id)
            strategy = protocol.strategy(sha) if protocol else None
            if strategy is not None and DATABASES[strategy.database].access == "api":
                raise SysRevError("API databases are validated by a preview, not an attestation")
            return editor.record_strategy_validation(
                repo, project_id, sha, method="human_attestation", reviewer=reviewer,
                note=body.note,
            )  # fmt: skip

        _call(run)
        return _protocol_view(repo, project_id)

    @router.post("/protocol/approve")
    def approve(
        project_id: str, body: OptionalNoteBody, session: SessionDep, reviewer: ReviewerDep
    ) -> dict[str, Any]:
        repo = Repository(session)
        _call(lambda: editor.approve(repo, project_id, reviewer=reviewer, note=body.note))
        return _protocol_view(repo, project_id)

    # ------------------------------------------------------------------ searches
    @router.get("/searches")
    def searches(project_id: str, session: SessionDep) -> dict[str, Any]:
        repo = Repository(session)
        _call(lambda: repo.get_project(project_id))
        return search_report(repo, project_id) | {
            "runs": [r.model_dump(mode="json") for r in repo.load_search_runs(project_id)]
        }

    @router.get("/searches/report")
    def report(
        project_id: str, session: SessionDep, format: Literal["json", "markdown", "csv"] = "json"
    ) -> Response:
        data = search_report(Repository(session), project_id)
        if format == "csv":
            return Response(report_csv(data), media_type="text/csv")
        if format == "markdown":
            return Response(report_markdown(data), media_type="text/markdown")
        return Response(json.dumps(data, indent=2), media_type="application/json")

    @router.post("/searches/strategies/{sha}/execute", status_code=status.HTTP_201_CREATED)
    def execute(
        project_id: str, sha: str, request: Request, session: SessionDep, reviewer: ReviewerDep
    ) -> dict[str, Any]:
        repo = Repository(session)
        svc = service(request, repo, project_id)
        run = _call(lambda: asyncio.run(svc.execute(sha, reviewer=reviewer)))
        return run.model_dump(mode="json")

    @router.post("/searches/import-preview")
    async def import_preview(
        request: Request,
        format: ImportFormat,
        columns: str | None = None,
    ) -> dict[str, Any]:
        """Parse an export without storing anything (column mapping, counts, problems)."""
        data = await request.body()
        mapping = json.loads(columns) if columns else None
        parsed = _call(lambda: parse_file(data, format, column_map=mapping))
        return {
            "entries": len(parsed.entries),
            "encoding": parsed.encoding,
            "column_map": parsed.column_map,
            "errors": parsed.errors[:50],
            "warnings": parsed.warnings[:50],
            "sample": [
                {"title": e.title, "year": e.year, "doi": e.doi, "pmid": e.pmid,
                 "first_author": e.authors[0].match_key if e.authors else None,
                 "source_id": e.source_id}
                for e in parsed.entries[:5]
            ],
        }  # fmt: skip

    @router.post("/searches/strategies/{sha}/import", status_code=status.HTTP_201_CREATED)
    async def import_file(
        project_id: str,
        sha: str,
        request: Request,
        session: SessionDep,
        reviewer: ReviewerDep,
        format: ImportFormat,
        filename: str,
        searched_on: date,
        hits_reported: int | None = None,
        columns: str | None = None,
        note: str | None = None,
        restrictions_confirmed: bool = False,
        restrictions_note: str | None = None,
    ) -> dict[str, Any]:
        """``hits_reported``: total the vendor interface showed for the whole search (omit if
        unavailable — completeness then needs an acknowledgement). ``restrictions_confirmed``:
        the reviewer confirms the protocol's limits/filters were applied in the interface."""
        data = await request.body()
        if len(data) > MAX_IMPORT_BYTES:
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "file too large")
        repo = Repository(session)
        svc = service(request, repo, project_id)
        mapping = json.loads(columns) if columns else None
        run = _call(
            lambda: svc.import_file(
                sha, data, filename=filename, fmt=format, searched_on=searched_on,
                reviewer=reviewer, hits_reported=hits_reported, column_map=mapping, note=note,
                restrictions_confirmed=restrictions_confirmed,
                restrictions_note=restrictions_note,
            )
        )  # fmt: skip
        return run.model_dump(mode="json")

    @router.post("/searches/strategies/{sha}/zero-results", status_code=status.HTTP_201_CREATED)
    def zero_results(
        project_id: str, sha: str, body: ZeroResultBody, request: Request, session: SessionDep,
        reviewer: ReviewerDep,
    ) -> dict[str, Any]:  # fmt: skip
        """The reviewer attests that the manual search ran and returned no records."""
        repo = Repository(session)
        svc = service(request, repo, project_id)
        run = _call(
            lambda: svc.record_zero_results(
                sha, searched_on=body.searched_on, note=body.note, reviewer=reviewer,
                restrictions_confirmed=body.restrictions_confirmed,
                restrictions_note=body.restrictions_note,
            )
        )  # fmt: skip
        return run.model_dump(mode="json")

    @router.post("/searches/strategies/{sha}/not-searched", status_code=status.HTTP_201_CREATED)
    def not_searched(
        project_id: str, sha: str, body: NoteBody, request: Request, session: SessionDep,
        reviewer: ReviewerDep,
    ) -> dict[str, Any]:  # fmt: skip
        repo = Repository(session)
        svc = service(request, repo, project_id)
        run = _call(lambda: svc.record_not_searched(sha, reason=body.note, reviewer=reviewer))
        return run.model_dump(mode="json")

    @router.post("/searches/acknowledge", status_code=status.HTTP_201_CREATED)
    def acknowledge(
        project_id: str, body: AcknowledgeBody, request: Request, session: SessionDep,
        reviewer: ReviewerDep,
    ) -> dict[str, Any]:  # fmt: skip
        repo = Repository(session)
        svc = service(request, repo, project_id)
        run = _call(
            lambda: svc.acknowledge(note=body.note, reviewer=reviewer, run_id=body.run_id,
                                    strategy_sha256=body.strategy_sha256)
        )  # fmt: skip
        return run.model_dump(mode="json")

    @router.post("/searches/prepare", status_code=status.HTTP_201_CREATED)
    def prepare(
        project_id: str, request: Request, session: SessionDep, reviewer: ReviewerDep
    ) -> dict[str, Any]:
        repo = Repository(session)
        svc = service(request, repo, project_id)
        run = _call(lambda: svc.prepare_for_screening(reviewer=reviewer))
        return run.model_dump(mode="json")

    # ------------------------------------------------------------------ records & duplicates
    @router.get("/records")
    def records(
        project_id: str, session: SessionDep, unique_only: bool = False, limit: int = 500
    ) -> list[dict[str, Any]]:
        rows = Repository(session).load_records(project_id, unique_only=unique_only)
        return [
            {"record_id": r.record_id, "source": r.source_database.value, "title": r.title,
             "year": r.publication_year, "doi": r.doi, "pmid": r.pmid,
             "duplicate_of": r.duplicate_of,
             "search_run_id": r.provenance.search_run_id if r.provenance else None}
            for r in rows[: max(0, limit)]
        ]  # fmt: skip

    @router.get("/duplicates")
    def duplicates(project_id: str, session: SessionDep) -> list[dict[str, Any]]:
        repo = Repository(session)
        return [c.model_dump(mode="json") for c in repo.load_duplicate_candidates(project_id)]

    @router.post("/duplicates/{candidate_id}/resolve")
    def resolve_duplicate(
        project_id: str, candidate_id: str, body: DuplicateResolutionBody, request: Request,
        session: SessionDep, reviewer: ReviewerDep,
    ) -> dict[str, Any]:  # fmt: skip
        repo = Repository(session)
        svc = service(request, repo, project_id)
        resolved = _call(
            lambda: svc.resolve_duplicate(candidate_id, resolution=body.resolution,
                                          reviewer=reviewer, note=body.note)
        )  # fmt: skip
        return dict(resolved.model_dump(mode="json"))

    # ------------------------------------------------------------------ PRISMA
    @router.get("/prisma")
    def prisma(project_id: str, session: SessionDep) -> dict[str, Any]:
        counts = _call(lambda: Repository(session).load_prisma(project_id))
        if counts is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no PRISMA snapshot recorded")
        return counts.model_dump(mode="json")

    @router.post("/prisma/recompute")
    def prisma_recompute(
        project_id: str, session: SessionDep, reviewer: ReviewerDep
    ) -> dict[str, Any]:
        """Re-derive the PRISMA counts from persisted records and decisions (never typed in)."""
        repo = Repository(session)
        del reviewer
        counts = _call(lambda: derive_prisma(repo, project_id))
        repo.save_prisma(project_id, counts)
        return counts.model_dump(mode="json")

    return router
