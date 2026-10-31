"""Session 8: protocol amendments after Gate 1 (see :mod:`src.search.amendments`).

Every state-changing endpoint requires a reviewer token and the hash of the state displayed
(the approved protocol for a proposal; the proposal revision for verification and decisions;
an open obligation for a re-review). A changed state is refused with 409 and nothing is saved.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, Literal, TypeVar

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.api_server.app import ReviewerDep, SessionDep
from src.core.enums import ScreeningVerdict
from src.core.exceptions import SysRevError
from src.fulltext.storage import DocumentStore
from src.search import amendments
from src.search.databases import DATABASES
from src.search.service import SearchClients, SearchService
from src.state.repository import NotFoundError, Repository

T = TypeVar("T")


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProposalBody(_Body):
    protocol: dict[str, Any]
    rationale: str = Field(min_length=1)
    expected_base_sha256: str = Field(min_length=64, max_length=64)


class RevisionBody(_Body):
    expected_sha256: str = Field(min_length=64, max_length=64)


class AttestBody(RevisionBody):
    note: str = Field(min_length=1)


class DecisionBody(RevisionBody):
    decision: Literal["approved", "rejected", "withdrawn"]
    comments: str = Field(min_length=1)


class ResolutionBody(_Body):
    resolution: Literal["confirmed", "revised", "not_applicable", "withdrawn"]
    note: str = Field(min_length=1)
    verdict: ScreeningVerdict | None = None
    exclusion_reason: str | None = None


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


def build_amendment_router(store: DocumentStore) -> APIRouter:
    router = APIRouter(prefix="/projects/{project_id}/amendments")

    def service(request: Request, repo: Repository, project_id: str) -> SearchService:
        clients: SearchClients | None = getattr(request.app.state, "search_clients", None)
        return SearchService(repo, project_id, store=store, clients=clients)

    @router.get("")
    def view(project_id: str, session: SessionDep) -> dict[str, Any]:
        repo = Repository(session)
        _call(lambda: repo.get_project(project_id))
        return amendments.state(repo, project_id)

    @router.post("", status_code=status.HTTP_201_CREATED)
    def propose(project_id: str, body: ProposalBody, session: SessionDep,
                reviewer: ReviewerDep) -> dict[str, Any]:  # fmt: skip
        repo = Repository(session)
        _call(
            lambda: amendments.propose(
                repo, project_id, body.protocol, rationale=body.rationale, reviewer=reviewer,
                expected_base_sha256=body.expected_base_sha256,
            )
        )  # fmt: skip
        return amendments.state(repo, project_id)

    @router.post("/{amendment_id}/mesh/verify")
    def verify_mesh(project_id: str, amendment_id: str, body: RevisionBody, request: Request,
                    session: SessionDep, reviewer: ReviewerDep) -> dict[str, Any]:  # fmt: skip
        repo = Repository(session)
        svc = service(request, repo, project_id)

        async def run() -> Any:
            async with svc.clients.pubmed() as client:
                return await amendments.verify_mesh(
                    repo, project_id, amendment_id, client, reviewer=reviewer,
                    expected_sha256=body.expected_sha256,
                )  # fmt: skip

        _call(lambda: asyncio.run(run()))
        return amendments.state(repo, project_id)

    @router.post("/{amendment_id}/strategies/{sha}/preview")
    def preview(project_id: str, amendment_id: str, sha: str, body: RevisionBody,
                request: Request, session: SessionDep,
                reviewer: ReviewerDep) -> dict[str, Any]:  # fmt: skip
        repo = Repository(session)
        svc = service(request, repo, project_id)
        run = _call(
            lambda: asyncio.run(
                svc.preview(
                    sha,
                    reviewer=reviewer,
                    amendment_id=amendment_id,
                    expected_amendment_sha256=body.expected_sha256,
                )
            )
        )
        return {"run": run.model_dump(mode="json")} | amendments.state(repo, project_id)

    @router.post("/{amendment_id}/strategies/{sha}/attest")
    def attest(project_id: str, amendment_id: str, sha: str, body: AttestBody,
               session: SessionDep, reviewer: ReviewerDep) -> dict[str, Any]:  # fmt: skip
        repo = Repository(session)

        def run() -> Any:
            pending = amendments.open_amendment(repo, project_id)
            strategy = pending.protocol.strategy(sha) if pending else None
            if strategy is not None and DATABASES[strategy.database].access == "api":
                raise SysRevError("API databases are validated by a preview, not an attestation")
            return amendments.record_strategy_validation(
                repo, project_id, amendment_id, sha, method="human_attestation",
                reviewer=reviewer, expected_sha256=body.expected_sha256, note=body.note,
            )  # fmt: skip

        _call(run)
        return amendments.state(repo, project_id)

    @router.post("/{amendment_id}/decision")
    def decide(project_id: str, amendment_id: str, body: DecisionBody, session: SessionDep,
               reviewer: ReviewerDep) -> dict[str, Any]:  # fmt: skip
        repo = Repository(session)
        _call(
            lambda: amendments.decide(
                repo, project_id, amendment_id, body.decision, reviewer=reviewer,
                comments=body.comments, expected_sha256=body.expected_sha256,
            )
        )  # fmt: skip
        return amendments.state(repo, project_id)

    @router.post("/obligations/{obligation_id}/resolve")
    def resolve(project_id: str, obligation_id: str, body: ResolutionBody,
                session: SessionDep, reviewer: ReviewerDep) -> dict[str, Any]:  # fmt: skip
        repo = Repository(session)
        _call(
            lambda: amendments.resolve(
                repo, project_id, obligation_id, body.resolution, reviewer=reviewer,
                note=body.note, verdict=body.verdict, exclusion_reason=body.exclusion_reason,
            )
        )  # fmt: skip
        return amendments.state(repo, project_id)

    return router
