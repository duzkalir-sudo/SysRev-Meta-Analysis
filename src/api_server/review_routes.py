"""Stage 2-4 human-review endpoints (Session 2).

Every state-changing endpoint requires ``Authorization: Bearer <reviewer token>``; the
acting human is taken from the token, never from the request body. Read endpoints return
inspectable JSON or CSV. LLM runs (screening, retrieval, extraction drafts) are started
from Python (see README) — not over HTTP — so no endpoint can spend model credit.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any, Literal, TypeVar

from fastapi import APIRouter, Body, Header, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field

from src.api_server.app import ReviewerDep, SessionDep  # app.py imports this module lazily
from src.core.enums import ScreeningStage, ScreeningVerdict
from src.core.exceptions import SysRevError
from src.extraction import (
    ProposedValue,
    accept_item,
    approve_rob,
    build_arms,
    correct_item,
    export_csv,
    export_json,
    export_rows,
    link_reports_to_study,
    record_unavailability,
    reject_item,
)
from src.fulltext import (
    DocumentStore,
    UploadRejectedError,
    build_manifest,
    confirm_pdf_match,
    manifest_csv,
    manifest_json,
    mark_unobtainable,
    register_upload,
    report_flow,
)
from src.pdf_extractor import parse_pdf
from src.screening.llm_screening import adjudicate, agreement_summary
from src.screening.reporting import methods_statement, screening_reporting_facts
from src.state.repository import NotFoundError, Repository

T = TypeVar("T")
MAX_UPLOAD_BYTES = 100 * 1024 * 1024


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AdjudicationBody(_Body):
    verdict: ScreeningVerdict
    rationale: str = Field(min_length=1)
    exclusion_reason: str | None = None
    criteria_cited: list[str] = Field(default_factory=list)


class NoteBody(_Body):
    note: str = Field(min_length=1)


class OptionalNoteBody(_Body):
    note: str | None = None


class ReasonBody(_Body):
    reason: str = Field(min_length=1)


class CorrectionBody(_Body):
    correction: ProposedValue
    note: str = Field(min_length=1)
    attest_unverifiable: bool = False


class StudyLinkBody(_Body):
    record_ids: list[str] = Field(min_length=1)
    study_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$")
    primary_record_id: str
    note: str = Field(min_length=1)


class UnavailabilityBody(_Body):
    reason_code: Literal["not_reported", "insufficient_detail", "incompatible_measure",
                         "overlapping_cohort", "figure_only", "other"]  # fmt: skip
    reason: str = Field(min_length=1)
    outcome_id: str | None = None


class RoBApprovalBody(_Body):
    overall: str | None = None
    domain_judgements: dict[str, tuple[str, str]] | None = None
    nos_stars: dict[str, int] | None = None
    note: str | None = None


def _call(fn: Callable[[], T]) -> T:
    try:
        return fn()
    except NotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except (SysRevError, ValueError) as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc


def build_review_router(store: DocumentStore) -> APIRouter:
    router = APIRouter(prefix="/projects/{project_id}")

    def doc_for(repo: Repository, project_id: str, record_id: str) -> Any:
        item = repo.get_fulltext_item(project_id, record_id)
        if item is None or item.pdf_filename is None or item.pdf_sha256 is None:
            raise NotFoundError(f"record {record_id} has no linked PDF")
        data = store.read_verified(item.pdf_filename, item.pdf_sha256)
        return parse_pdf(data, filename=item.pdf_filename, expected_sha256=item.pdf_sha256)

    # ------------------------------------------------------------------ screening
    @router.get("/screening/{stage}/summary")
    def screening_summary(
        project_id: str, stage: ScreeningStage, session: SessionDep
    ) -> dict[str, Any]:
        repo = Repository(session)
        summary = agreement_summary(repo, project_id, stage)
        facts = screening_reporting_facts(repo, project_id, stage)
        return {
            "stage": stage,
            "paired_records": summary.paired_records,
            "agreement": summary.statistics.model_dump() if summary.statistics else None,
            "conflicts": summary.conflicts,
            "unresolved_conflicts": summary.unresolved_conflicts,
            "automated_passes": facts.automated_passes,
            "human_screeners": facts.human_screeners,
            "human_adjudicators": facts.human_adjudicators,
            "methods_statement": methods_statement(facts),
        }

    @router.get("/screening/{stage}/conflicts")
    def list_conflicts(
        project_id: str, stage: ScreeningStage, session: SessionDep
    ) -> list[dict[str, Any]]:
        conflicts = Repository(session).load_conflicts(project_id)
        return [c.model_dump(mode="json") for c in conflicts if c.stage == stage]

    @router.post("/screening/{stage}/records/{record_id}/adjudication",
                 status_code=status.HTTP_201_CREATED)  # fmt: skip
    def adjudication(
        project_id: str, stage: ScreeningStage, record_id: str, body: AdjudicationBody,
        session: SessionDep, reviewer: ReviewerDep,
    ) -> dict[str, Any]:  # fmt: skip
        repo = Repository(session)
        decision = _call(
            lambda: adjudicate(
                repo, project_id, record_id, stage, verdict=body.verdict,
                rationale=body.rationale, reviewer=reviewer,
                exclusion_reason=body.exclusion_reason, criteria_cited=body.criteria_cited,
            )
        )  # fmt: skip
        return decision.model_dump(mode="json")

    # ------------------------------------------------------------------ full text
    @router.get("/fulltext/manifest")
    def manifest(
        project_id: str, session: SessionDep, format: Literal["json", "csv"] = "json",
        only_open: bool = True,
    ) -> Response:  # fmt: skip
        rows = build_manifest(Repository(session), project_id, only_open=only_open)
        if format == "csv":
            return Response(manifest_csv(rows), media_type="text/csv")
        return Response(manifest_json(rows), media_type="application/json")

    @router.get("/fulltext/flow")
    def flow(project_id: str, session: SessionDep) -> dict[str, Any]:
        f = report_flow(Repository(session), project_id)
        return {k: getattr(f, k) for k in f.__dataclass_fields__}

    @router.put("/fulltext/{record_id}/pdf", status_code=status.HTTP_201_CREATED)
    async def upload(
        project_id: str, record_id: str, request: Request, session: SessionDep,
        reviewer: ReviewerDep, x_filename: Annotated[str, Header()] = "upload.pdf",
    ) -> dict[str, Any]:  # fmt: skip
        data = await request.body()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "PDF too large")
        repo = Repository(session)
        try:
            item = _call(
                lambda: register_upload(
                    repo, store, project_id, record_id, data,
                    original_filename=x_filename, reviewer=reviewer,
                )
            )  # fmt: skip
        except HTTPException as exc:
            if isinstance(exc.__cause__, UploadRejectedError):
                raise HTTPException(422, str(exc.__cause__)) from exc
            raise
        return item.model_dump(mode="json")

    @router.post("/fulltext/{record_id}/confirm-match")
    def confirm(project_id: str, record_id: str, body: NoteBody, session: SessionDep,
                reviewer: ReviewerDep) -> dict[str, Any]:  # fmt: skip
        repo = Repository(session)
        item = _call(lambda: confirm_pdf_match(repo, project_id, record_id, reviewer=reviewer,
                                               note=body.note))  # fmt: skip
        return item.model_dump(mode="json")

    @router.post("/fulltext/{record_id}/unobtainable")
    def unobtainable(project_id: str, record_id: str, body: ReasonBody, session: SessionDep,
                     reviewer: ReviewerDep) -> dict[str, Any]:  # fmt: skip
        repo = Repository(session)
        item = _call(lambda: mark_unobtainable(repo, project_id, record_id, reviewer=reviewer,
                                               reason=body.reason))  # fmt: skip
        return item.model_dump(mode="json")

    # ------------------------------------------------------------------ extraction
    @router.get("/extraction/items")
    def items(
        project_id: str, session: SessionDep, include_superseded: bool = False
    ) -> list[dict[str, Any]]:
        rows = Repository(session).load_extraction_items(
            project_id, include_superseded=include_superseded
        )
        return [i.model_dump(mode="json") for i in rows]

    @router.get("/extraction/items/{item_id}/history")
    def history(project_id: str, item_id: str, session: SessionDep) -> list[dict[str, Any]]:
        events = Repository(session).list_review_events(project_id, item_id)
        return [e.model_dump(mode="json") for e in events]

    @router.post("/extraction/items/{item_id}/accept")
    def accept(project_id: str, item_id: str, body: OptionalNoteBody, session: SessionDep,
               reviewer: ReviewerDep) -> dict[str, Any]:  # fmt: skip
        repo = Repository(session)
        item = _call(lambda: accept_item(repo, project_id, item_id, reviewer=reviewer,
                                         note=body.note))  # fmt: skip
        return item.model_dump(mode="json")

    @router.post("/extraction/items/{item_id}/reject")
    def reject(project_id: str, item_id: str, body: NoteBody, session: SessionDep,
               reviewer: ReviewerDep) -> dict[str, Any]:  # fmt: skip
        repo = Repository(session)
        item = _call(lambda: reject_item(repo, project_id, item_id, reviewer=reviewer,
                                         note=body.note))  # fmt: skip
        return item.model_dump(mode="json")

    @router.post("/extraction/items/{item_id}/correct")
    def correct(project_id: str, item_id: str, body: CorrectionBody, session: SessionDep,
                reviewer: ReviewerDep) -> dict[str, Any]:  # fmt: skip
        repo = Repository(session)

        def run() -> Any:
            old = repo.get_extraction_item(project_id, item_id)
            doc = doc_for(repo, project_id, old.record_id)
            return correct_item(
                repo, project_id, item_id, body.correction, doc, reviewer=reviewer,
                note=body.note, attest_unverifiable=body.attest_unverifiable,
            )  # fmt: skip

        return _call(run).model_dump(mode="json")  # type: ignore[no-any-return]

    @router.post("/extraction/records/{record_id}/arms", status_code=status.HTTP_201_CREATED)
    def arms(
        project_id: str, record_id: str, session: SessionDep, reviewer: ReviewerDep
    ) -> list[dict[str, Any]]:
        repo = Repository(session)
        built = _call(lambda: build_arms(repo, project_id, record_id, reviewer=reviewer))
        return [a.model_dump(mode="json") for a in built]

    @router.post("/extraction/records/{record_id}/unavailability",
                 status_code=status.HTTP_201_CREATED)  # fmt: skip
    def unavailability(project_id: str, record_id: str, body: UnavailabilityBody,
                       session: SessionDep, reviewer: ReviewerDep) -> dict[str, Any]:  # fmt: skip
        repo = Repository(session)
        entry = _call(
            lambda: record_unavailability(
                repo, project_id, record_id, reviewer=reviewer, reason_code=body.reason_code,
                reason=body.reason, outcome_id=body.outcome_id,
            )
        )  # fmt: skip
        return entry.model_dump(mode="json")

    @router.post("/studies/link", status_code=status.HTTP_201_CREATED)
    def link(project_id: str, body: StudyLinkBody, session: SessionDep,
             reviewer: ReviewerDep) -> list[dict[str, Any]]:  # fmt: skip
        repo = Repository(session)
        links = _call(
            lambda: link_reports_to_study(
                repo, project_id, body.record_ids, study_id=body.study_id,
                primary_record_id=body.primary_record_id, reviewer=reviewer, note=body.note,
            )
        )  # fmt: skip
        return [x.model_dump(mode="json") for x in links]

    @router.get("/extraction/export")
    def export(project_id: str, session: SessionDep,
               format: Literal["json", "csv"] = "json") -> Response:  # fmt: skip
        rows = export_rows(Repository(session), project_id)
        if format == "csv":
            return Response(export_csv(rows), media_type="text/csv")
        return Response(export_json(rows), media_type="application/json")

    # ------------------------------------------------------------------ risk of bias
    @router.get("/rob")
    def rob(project_id: str, session: SessionDep) -> list[dict[str, Any]]:
        return [a.model_dump(mode="json") for a in Repository(session).load_rob(project_id)]

    @router.post("/rob/{assessment_id}/approve")
    def rob_approve(project_id: str, assessment_id: str, session: SessionDep,
                    reviewer: ReviewerDep,
                    body: Annotated[RoBApprovalBody, Body()]) -> dict[str, Any]:  # fmt: skip
        repo = Repository(session)
        approved = _call(
            lambda: approve_rob(
                repo, project_id, assessment_id, reviewer=reviewer, overall=body.overall,
                domain_judgements=body.domain_judgements, nos_stars=body.nos_stars,
                note=body.note,
            )
        )  # fmt: skip
        return approved.model_dump(mode="json")

    return router
