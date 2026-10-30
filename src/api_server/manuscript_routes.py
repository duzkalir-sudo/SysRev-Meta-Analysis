"""Stage 6 endpoints: manuscript versions, references, audits, acknowledgements, sign-off.

Every state-changing endpoint requires an enrolled reviewer's bearer token; the acting human
is taken from the token. No endpoint calls a language model (model-assisted drafting and
advisory claim review are started from Python / the local UI, never over HTTP). Reference
import and verification call PubMed/Crossref only when a reviewer requests it; a failed
lookup is recorded as *unavailable*, never as a verification.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from fastapi import APIRouter, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.api_server.app import ReviewerDep, SessionDep
from src.apis.crossref import CrossrefClient
from src.apis.pubmed import PubMedClient
from src.core.enums import ActorType
from src.core.exceptions import SysRevError
from src.core.schemas.manuscript import EvidenceExcerpt, JournalSettings, StyleSettings
from src.manuscript.document import vancouver
from src.manuscript.export import (
    audit_markdown,
    manuscript_docx,
    manuscript_markdown,
    provenance_markdown,
)
from src.manuscript.references import import_reference, verify_reference
from src.manuscript.service import Actor, ManuscriptService
from src.state.db import ReviewerORM
from src.state.repository import NotFoundError, Repository

T = TypeVar("T")


@dataclass(frozen=True)
class ReferenceClients:
    """Factories for the literature clients used for reference import/verification
    (injectable for tests). ``None`` from a factory means "not configured"."""

    pubmed: Callable[[], PubMedClient | None]
    crossref: Callable[[], CrossrefClient | None]

    @classmethod
    def from_settings(cls) -> ReferenceClients:
        from src.core.config import get_settings

        s = get_settings()
        common: dict[str, Any] = {"timeout": s.http_timeout_seconds, "max_retries": 1}

        def pubmed() -> PubMedClient | None:
            if not s.ncbi_email:
                return None
            key = s.ncbi_api_key.get_secret_value() if s.ncbi_api_key else None
            return PubMedClient(email=s.ncbi_email, tool=s.ncbi_tool, api_key=key, **common)

        def crossref() -> CrossrefClient | None:
            return CrossrefClient(
                mailto=s.crossref_mailto,
                requests_per_second=s.crossref_requests_per_second,
                **common,
            )

        return cls(pubmed, crossref)


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DraftBody(_Body):
    style: StyleSettings | None = None
    journal: JournalSettings | None = None


class EditBody(_Body):
    changes: dict[str, str] = Field(min_length=1)
    summary: str = Field(min_length=3)
    style: StyleSettings | None = None
    journal: JournalSettings | None = None


class AckBody(_Body):
    decision: Literal["confirmed_supported", "accepted_as_author_statement"]
    note: str = Field(min_length=10)


class ApproveBody(_Body):
    statement: str = Field(min_length=10)


class ReferenceBody(_Body):
    pmid: str | None = None
    doi: str | None = None


class ExcerptBody(_Body):
    text: str = Field(min_length=20, max_length=5000)
    location: str = Field(min_length=1)


def _call(fn: Callable[[], T]) -> T:
    try:
        return fn()
    except NotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except ValidationError as exc:
        raise HTTPException(422, str(exc)) from exc
    except (SysRevError, ValueError) as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc


def _actor(reviewer: ReviewerORM) -> Actor:
    return Actor(reviewer.display_name, ActorType.HUMAN, reviewer.id)


def _summary(v: Any) -> dict[str, Any]:
    return {
        "version": v.version,
        "parent_version": v.parent_version,
        "content_sha256": v.content_sha256,
        "created_by": v.created_by,
        "actor_type": v.actor_type.value,
        "created_at": v.created_at.isoformat(),
        "change_summary": v.change_summary,
        "provenance": [p.model_dump(mode="json") for p in v.provenance],
    }


def build_manuscript_router() -> APIRouter:
    router = APIRouter(prefix="/projects/{project_id}/manuscript")

    def svc(session: Any, project_id: str) -> ManuscriptService:
        repo = Repository(session)
        _call(lambda: repo.get_project(project_id))
        return ManuscriptService(repo, project_id)

    def clients(request: Request) -> ReferenceClients:
        injected: ReferenceClients | None = getattr(request.app.state, "reference_clients", None)
        return injected or ReferenceClients.from_settings()

    # ------------------------------------------------------------------ read
    @router.get("/status")
    def get_status(project_id: str, session: SessionDep) -> dict[str, Any]:
        return svc(session, project_id).status()

    @router.get("/evidence")
    def evidence(project_id: str, session: SessionDep) -> dict[str, Any]:
        p = _call(lambda: svc(session, project_id).package())
        return {
            "fingerprint": p.fingerprint,
            "facts": [
                {
                    "key": f.key,
                    "text": f.text,
                    "description": f.description,
                    "source": f.source,
                    "category": f.category,
                }
                for f in p.facts.values()
            ],
            "templates": sorted(p.templates),
            "references": [
                {
                    "ref_key": k,
                    "origin": s.entry.origin,
                    "citation": vancouver(s.entry),
                    "identity": s.identity_status,
                    "basis": s.identity_basis,
                    "excerpts": len(s.excerpts),
                }
                for k, s in p.references.items()
            ],
            "expected_outputs": [
                {"keys": list(e.keys), "description": e.description, "sections": list(e.sections)}
                for e in p.expected
            ],
        }

    @router.get("/versions")
    def versions(project_id: str, session: SessionDep) -> list[dict[str, Any]]:
        return [_summary(v) for v in Repository(session).list_manuscript_versions(project_id)]

    @router.get("/versions/{version}")
    def version(project_id: str, version: int, session: SessionDep) -> dict[str, Any]:
        v = _call(lambda: Repository(session).get_manuscript_version(project_id, version))
        return v.model_dump(mode="json")

    @router.get("/diff")
    def diff(project_id: str, a: int, b: int, session: SessionDep) -> dict[str, str]:
        return {"diff": _call(lambda: svc(session, project_id).diff(a, b))}

    @router.get("/audits")
    def audits(project_id: str, session: SessionDep) -> list[dict[str, Any]]:
        return [
            r.model_dump(mode="json") for r in Repository(session).list_audit_reports(project_id)
        ]

    @router.get("/audits/{report_id}")
    def audit(project_id: str, report_id: str, session: SessionDep) -> dict[str, Any]:
        repo = Repository(session)
        report = _call(lambda: repo.get_audit_report(project_id, report_id))
        acks = [
            a.model_dump(mode="json")
            for a in repo.load_acknowledgements(project_id)
            if a.audit_report_id == report_id
        ]
        reviews = [
            r.model_dump(mode="json")
            for r in repo.load_claim_reviews(project_id)
            if r.audit_report_id == report_id
        ]
        return report.model_dump(mode="json") | {
            "acknowledgements": acks,
            "advisory_claim_reviews": reviews,
        }

    @router.get("/audits/{report_id}/export")
    def export_audit(
        project_id: str, report_id: str, session: SessionDep, format: Literal["md", "json"] = "md"
    ) -> Response:
        repo = Repository(session)
        report = _call(lambda: repo.get_audit_report(project_id, report_id))
        if format == "json":
            return Response(
                json.dumps(report.model_dump(mode="json"), indent=2), media_type="application/json"
            )
        text = audit_markdown(
            report, repo.load_acknowledgements(project_id), repo.load_claim_reviews(project_id)
        )
        return Response(text, media_type="text/markdown")

    @router.get("/export")
    def export(
        project_id: str,
        session: SessionDep,
        version: int | None = None,
        format: Literal["md", "docx", "provenance"] = "md",
    ) -> Response:
        repo = Repository(session)
        v = (
            _call(lambda: repo.get_manuscript_version(project_id, version))
            if version
            else repo.latest_manuscript_version(project_id)
        )
        if v is None:
            raise HTTPException(404, "no manuscript version")
        if format == "docx":
            return Response(
                manuscript_docx(v),
                media_type="application/vnd.openxmlformats-"
                "officedocument.wordprocessingml.document",
            )
        text = provenance_markdown(v) if format == "provenance" else manuscript_markdown(v)
        return Response(text, media_type="text/markdown")

    # ------------------------------------------------------------------ write (authenticated)
    @router.post("/draft")
    def draft(
        project_id: str, body: DraftBody, session: SessionDep, reviewer: ReviewerDep
    ) -> dict[str, Any]:
        s = svc(session, project_id)
        v = _call(
            lambda: s.create_template_draft(
                _actor(reviewer), style=body.style, journal=body.journal
            )
        )
        return _summary(v)

    @router.post("/versions")
    def edit(
        project_id: str, body: EditBody, session: SessionDep, reviewer: ReviewerDep
    ) -> dict[str, Any]:
        s = svc(session, project_id)
        v = _call(
            lambda: s.edit(
                _actor(reviewer), body.changes, body.summary, style=body.style, journal=body.journal
            )
        )
        return _summary(v)

    @router.post("/rerender")
    def rerender(project_id: str, session: SessionDep, reviewer: ReviewerDep) -> dict[str, Any]:
        s = svc(session, project_id)
        return _summary(_call(lambda: s.rerender(_actor(reviewer))))

    @router.post("/audit")
    def run_audit(project_id: str, session: SessionDep, reviewer: ReviewerDep) -> dict[str, Any]:
        del reviewer  # authenticated: an audit is a persisted project event
        s = svc(session, project_id)
        return _call(s.audit).model_dump(mode="json")

    @router.post("/audits/{report_id}/findings/{finding_id}/acknowledge")
    def acknowledge(
        project_id: str,
        report_id: str,
        finding_id: str,
        body: AckBody,
        session: SessionDep,
        reviewer: ReviewerDep,
    ) -> dict[str, Any]:
        s = svc(session, project_id)
        ack = _call(
            lambda: s.acknowledge(
                report_id,
                finding_id,
                decision=body.decision,
                note=body.note,
                reviewer_id=reviewer.id,
                reviewer_name=reviewer.display_name,
            )
        )
        return ack.model_dump(mode="json")

    @router.post("/approve")
    def approve(
        project_id: str, body: ApproveBody, session: SessionDep, reviewer: ReviewerDep
    ) -> dict[str, Any]:
        s = svc(session, project_id)
        approval = _call(
            lambda: s.approve(
                reviewer_id=reviewer.id,
                reviewer_name=reviewer.display_name,
                statement=body.statement,
            )
        )
        return approval.model_dump(mode="json")

    @router.post("/references")
    def add_reference(
        project_id: str,
        body: ReferenceBody,
        request: Request,
        session: SessionDep,
        reviewer: ReviewerDep,
    ) -> dict[str, Any]:
        repo = Repository(session)
        c = clients(request)
        entry = _call(
            lambda: asyncio.run(
                import_reference(
                    pmid=body.pmid,
                    doi=body.doi,
                    pubmed=c.pubmed(),
                    crossref=c.crossref(),
                    added_by=reviewer.display_name,
                    reviewer_id=reviewer.id,
                )
            )
        )
        _call(lambda: repo.save_reference_entry(project_id, entry))
        return entry.model_dump(mode="json")

    @router.post("/references/verify")
    def verify_references(
        project_id: str, request: Request, session: SessionDep, reviewer: ReviewerDep
    ) -> list[dict[str, Any]]:
        s = svc(session, project_id)
        package = _call(s.package)
        c = clients(request)

        async def run() -> list[Any]:
            out = []
            for state in package.references.values():
                out += await verify_reference(
                    state.entry,
                    pubmed=c.pubmed(),
                    crossref=c.crossref(),
                    checked_by=reviewer.display_name,
                )
            return out

        results = _call(lambda: asyncio.run(run()))
        for v in results:
            s.repo.save_reference_verification(project_id, v)
        return [v.model_dump(mode="json") for v in results]

    @router.post("/references/{ref_key}/excerpts")
    def add_excerpt(
        project_id: str, ref_key: str, body: ExcerptBody, session: SessionDep, reviewer: ReviewerDep
    ) -> dict[str, Any]:
        s = svc(session, project_id)
        if ref_key not in _call(s.package).references:
            raise HTTPException(404, f"reference {ref_key} is not in the registry")
        excerpt = EvidenceExcerpt(
            ref_key=ref_key,
            text=body.text,
            location=body.location,
            added_by=reviewer.display_name,
            reviewer_id=reviewer.id,
        )
        _call(lambda: s.repo.save_evidence_excerpt(project_id, excerpt))
        return excerpt.model_dump(mode="json")

    return router
