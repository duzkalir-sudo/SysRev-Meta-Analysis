"""Session 7 endpoints: PRISMA 2020 flow diagram and checklists, GRADE and Summary of
Findings, structured synthesis without meta-analysis, and SWiM.

Read endpoints are open (like the other project views). Every decision endpoint requires an
enrolled reviewer's bearer token, and the acting human is taken from the token: a request
body never names a reviewer. No endpoint calls a language model, and no endpoint can mark a
checklist item, a certainty rating or a synthesis as approved on the system's behalf.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, TypeVar

from fastapi import APIRouter, HTTPException, Response, status
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.api_server.app import ReviewerDep, SessionDep
from src.core.exceptions import SysRevError
from src.reporting import prisma_flow as flow
from src.reporting.export import checklist_docx, checklist_xlsx, sof_docx, sof_markdown
from src.reporting.service import ReportingService
from src.state.repository import NotFoundError, Repository

T = TypeVar("T")
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
Checklist = Literal["prisma_2020_main", "prisma_2020_abstract", "swim"]


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


SHA = r"^[0-9a-f]{64}$"


class SeenChecklist(_Body):
    """The checklist screen the reviewer decided on (from GET /checklists/{name})."""

    manuscript_version: int = Field(ge=1)
    manuscript_sha256: str = Field(pattern=SHA)
    evidence_sha256: str = Field(pattern=SHA)


class ItemDecision(_Body):
    status: Literal["reported", "partially_reported", "missing", "not_applicable"]
    section_ids: list[str] = Field(default_factory=list)
    location: str = ""
    justification: str = Field(min_length=10)
    evidence_keys: list[str] = Field(default_factory=list)
    # required (may be null): the item's decision displayed to the reviewer
    based_on_decision_id: str | None


class ItemDecisionBody(ItemDecision):
    seen: SeenChecklist


class BulkItemBody(ItemDecision):
    item_id: str


class BulkDecisionBody(_Body):
    seen: SeenChecklist
    decisions: list[BulkItemBody] = Field(min_length=1, max_length=100)


class GradeBody(_Body):
    status: Literal["draft", "approved", "not_assessed"]
    approach: str | None = None
    framework_rationale: str = ""
    initial_certainty: str | None = None
    initial_rationale: str = ""
    domains: list[dict[str, Any]] = Field(default_factory=list)
    upgrades: list[dict[str, Any]] = Field(default_factory=list)
    final_certainty: str | None = None
    departure_rationale: str | None = None
    summary: str = ""
    baseline_risks: list[dict[str, Any]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    not_assessed_reason: str | None = None
    status_note: str | None = None
    # the state displayed with the decision support (from GET /grade); both required
    inputs_sha256: str = Field(pattern=SHA)
    based_on_assessment_id: str | None


class WithdrawGradeBody(_Body):
    reason: str = Field(min_length=10)
    based_on_assessment_id: str


class WithdrawGroupBody(_Body):
    reason: str = Field(min_length=10)
    based_on_revision_id: str


class SynthesisBody(_Body):
    group_id: str
    outcome_id: str
    status: Literal["draft", "approved"]
    title: str
    grouping_basis: list[str]
    description: str
    rationale: str
    protocol_change: bool = False
    protocol_change_rationale: str | None = None
    reason_not_pooled: str
    reason_note: str
    standardised_metric: str
    synthesis_method: str
    method_justification: str
    prioritisation_criteria: str | None = None
    heterogeneity_investigation: str | None = None
    certainty_method: str
    presentation: str
    limitations: str
    findings: list[dict[str, Any]] = Field(min_length=1)
    selection_rule: str | None = None
    summary: str


class SynthesisSaveBody(SynthesisBody):
    # from POST /synthesis/preview: the evidence state and group revision displayed
    inputs_sha256: str = Field(pattern=SHA)
    based_on_revision_id: str | None


class SwimBody(_Body):
    applicability: Literal["applicable", "partially_applicable", "not_applicable"]
    rationale: str = Field(min_length=20)
    # from GET /swim: the approved syntheses and decision displayed
    synthesis_sha256: str = Field(pattern=SHA)
    based_on_decision_id: str | None


def _call(fn: Callable[[], T]) -> T:
    try:
        return fn()
    except NotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except flow.UnsupportedVariantError as exc:
        raise HTTPException(
            422,
            {"message": str(exc), "variant": exc.variant, "unsupported_fields": exc.fields,
             "reason": exc.reason},
        ) from exc  # fmt: skip
    except ValidationError as exc:
        raise HTTPException(422, str(exc)) from exc
    except (SysRevError, ValueError, KeyError) as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc


def build_reporting_router() -> APIRouter:
    router = APIRouter(prefix="/projects/{project_id}/reporting")

    def svc(session: Any, project_id: str) -> ReportingService:
        repo = Repository(session)
        _call(lambda: repo.get_project(project_id))
        return ReportingService(repo, project_id)

    # ------------------------------------------------------------------ read
    @router.get("/status")
    def overview(project_id: str, session: SessionDep) -> dict[str, Any]:
        return _call(svc(session, project_id).overview)

    @router.get("/prisma/diagram")
    def diagram(
        project_id: str,
        session: SessionDep,
        format: Literal["svg", "png", "json"] = "svg",
        variant: str = flow.SUPPORTED_VARIANT,
    ) -> Response:
        s = svc(session, project_id)
        out = _call(lambda: s.flow(variant))
        headers = {"X-Snapshot-SHA256": out["spec"]["snapshot_sha256"],
                   "X-Diagram-Status": out["spec"]["status"],
                   "X-Generator": flow.GENERATOR}  # fmt: skip
        if format == "json":
            return Response(flow.spec_json(out["spec"]), media_type="application/json",
                            headers=headers)  # fmt: skip
        if format == "png":
            try:
                data = flow.png_from_spec(out["spec"])
            except flow.RenderingUnavailableError as exc:
                raise HTTPException(501, str(exc)) from exc
            return Response(data, media_type="image/png", headers=headers)
        return Response(out["svg"], media_type="image/svg+xml",
                        headers=headers | {"X-SVG-SHA256": out["svg_sha256"]})  # fmt: skip

    @router.get("/checklists/{checklist}")
    def checklist(project_id: str, checklist: Checklist, session: SessionDep) -> dict[str, Any]:
        s = svc(session, project_id)
        return _call(lambda: s.checklist(checklist))

    @router.get("/checklists/{checklist}/export")
    def export_checklist(
        project_id: str,
        checklist: Checklist,
        session: SessionDep,
        format: Literal["csv", "docx", "xlsx"] = "csv",
    ) -> Response:
        s = svc(session, project_id)
        if format == "csv":
            return Response(_call(lambda: s.checklist_csv(checklist)), media_type="text/csv")
        data = _call(lambda: s.checklist(checklist))
        if format == "docx":
            return Response(checklist_docx(data), media_type=DOCX)
        return Response(checklist_xlsx(data), media_type=XLSX)

    @router.get("/grade")
    def grade(project_id: str, session: SessionDep) -> list[dict[str, Any]]:
        return _call(svc(session, project_id).grade)

    @router.get("/sof")
    def sof(project_id: str, session: SessionDep, format: Literal["md", "docx"] = "md") -> Response:
        s = svc(session, project_id)
        package = _call(s.package)
        if format == "docx":
            return Response(sof_docx(package), media_type=DOCX)
        return Response(sof_markdown(package), media_type="text/markdown")

    @router.get("/synthesis")
    def syntheses(project_id: str, session: SessionDep) -> list[dict[str, Any]]:
        return _call(svc(session, project_id).syntheses)

    @router.get("/swim")
    def swim(project_id: str, session: SessionDep) -> dict[str, Any]:
        return _call(svc(session, project_id).swim)

    # ------------------------------------------------------------------ decisions
    @router.post("/checklists/{checklist}/items/{item_id}")
    def decide(
        project_id: str,
        checklist: Checklist,
        item_id: str,
        body: ItemDecisionBody,
        session: SessionDep,
        reviewer: ReviewerDep,
    ) -> dict[str, Any]:
        s = svc(session, project_id)
        data = body.model_dump()
        seen = data.pop("seen")
        d = _call(lambda: s.decide_item(checklist, item_id, **data, seen=seen,
                                        reviewer_id=reviewer.id,
                                        reviewer_name=reviewer.display_name))  # fmt: skip
        return d.model_dump(mode="json")

    @router.post("/checklists/{checklist}/items")
    def decide_many(
        project_id: str,
        checklist: Checklist,
        body: BulkDecisionBody,
        session: SessionDep,
        reviewer: ReviewerDep,
    ) -> list[dict[str, Any]]:
        """Several decisions in one request; each item still carries its own status,
        location and justification (no blanket approval)."""
        s = svc(session, project_id)
        rows = [d.model_dump() | {"checklist": checklist} for d in body.decisions]
        out = _call(lambda: s.decide_items(rows, seen=body.seen.model_dump(),
                                           reviewer_id=reviewer.id,
                                           reviewer_name=reviewer.display_name))  # fmt: skip
        return [d.model_dump(mode="json") for d in out]

    @router.post("/grade/{analysis_id}")
    def save_grade(
        project_id: str, analysis_id: str, body: GradeBody, session: SessionDep,
        reviewer: ReviewerDep,
    ) -> dict[str, Any]:  # fmt: skip
        s = svc(session, project_id)
        g = _call(lambda: s.save_grade(analysis_id, body.model_dump(), reviewer_id=reviewer.id,
                                       reviewer_name=reviewer.display_name))  # fmt: skip
        return g.model_dump(mode="json")

    @router.post("/grade/{analysis_id}/withdraw")
    def withdraw_grade(
        project_id: str, analysis_id: str, body: WithdrawGradeBody, session: SessionDep,
        reviewer: ReviewerDep,
    ) -> dict[str, Any]:  # fmt: skip
        s = svc(session, project_id)
        g = _call(lambda: s.withdraw_grade(
            analysis_id, body.reason, based_on_assessment_id=body.based_on_assessment_id,
            reviewer_id=reviewer.id, reviewer_name=reviewer.display_name))  # fmt: skip
        return g.model_dump(mode="json")

    @router.post("/synthesis/preview")
    def preview_group(project_id: str, body: SynthesisBody, session: SessionDep) -> dict[str, Any]:
        """Read-only: the linked evidence as it is now, its hash and the problems. A save must
        send back the returned ``inputs_sha256`` and ``based_on_revision_id``."""
        s = svc(session, project_id)
        return _call(lambda: s.preview_group(body.model_dump()))

    @router.post("/synthesis")
    def save_group(
        project_id: str, body: SynthesisSaveBody, session: SessionDep, reviewer: ReviewerDep
    ) -> dict[str, Any]:
        s = svc(session, project_id)
        g = _call(lambda: s.save_group(body.model_dump(), reviewer_id=reviewer.id,
                                       reviewer_name=reviewer.display_name))  # fmt: skip
        return g.model_dump(mode="json")

    @router.post("/synthesis/{group_id}/withdraw")
    def withdraw_group(
        project_id: str, group_id: str, body: WithdrawGroupBody, session: SessionDep,
        reviewer: ReviewerDep,
    ) -> dict[str, Any]:  # fmt: skip
        s = svc(session, project_id)
        g = _call(lambda: s.withdraw_group(
            group_id, body.reason, based_on_revision_id=body.based_on_revision_id,
            reviewer_id=reviewer.id, reviewer_name=reviewer.display_name))  # fmt: skip
        return g.model_dump(mode="json")

    @router.post("/swim")
    def decide_swim(
        project_id: str, body: SwimBody, session: SessionDep, reviewer: ReviewerDep
    ) -> dict[str, Any]:
        s = svc(session, project_id)
        d = _call(lambda: s.decide_swim(body.applicability, body.rationale,
                                        synthesis_sha256=body.synthesis_sha256,
                                        based_on_decision_id=body.based_on_decision_id,
                                        reviewer_id=reviewer.id,
                                        reviewer_name=reviewer.display_name))  # fmt: skip
        return d.model_dump(mode="json")

    return router
