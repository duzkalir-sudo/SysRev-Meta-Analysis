"""Stage 2-5 human-review endpoints (Sessions 2-3).

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
from src.core.enums import ReviewStatus, ScreeningStage, ScreeningVerdict
from src.core.exceptions import SysRevError
from src.core.schemas.base import utc_now
from src.core.schemas.stats import AnalysisPlan, AnalysisSpec
from src.extraction import (
    ProposedValue,
    accept_item,
    approve_rob,
    build_arms,
    correct_item,
    double_counting_state_sha256,
    export_csv,
    export_json,
    export_rows,
    item_state_sha256,
    link_reports_to_study,
    links_state_sha256,
    record_unavailability,
    reject_item,
    resolve_double_counting,
    revise_reviewed_item,
    unlink_reports,
)
from src.fulltext import (
    DocumentStore,
    UploadRejectedError,
    acknowledge_xml_only,
    build_manifest,
    confirm_pdf_match,
    manifest_csv,
    manifest_json,
    mark_unobtainable,
    register_upload,
    report_flow,
)
from src.pdf_extractor.cache import DEFAULT_CACHE
from src.screening.llm_screening import adjudicate, agreement_summary
from src.screening.reporting import methods_statement, screening_reporting_facts
from src.state.repository import NotFoundError, Repository
from src.stats_engine import plan_service

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
    # Session 8: hash of the item as displayed (``state_sha256`` of GET /extraction/items).
    # Required to correct an already reviewed value; optional for pending items (legacy).
    expected_state_sha256: str | None = None


class StudyLinkBody(_Body):
    record_ids: list[str] = Field(min_length=1)
    study_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$")
    primary_record_id: str
    note: str = Field(min_length=1)
    expected_state_sha256: str | None = None  # GET /studies ``state_sha256`` (Session 8)


class UnlinkBody(_Body):
    record_ids: list[str] = Field(min_length=1)
    note: str = Field(min_length=1)
    expected_state_sha256: str = Field(min_length=64, max_length=64)


class PlanDraftBody(_Body):
    analyses: list[AnalysisSpec] = Field(min_length=1)
    rationale: str | None = None
    expected_latest_sha256: str | None  # explicit null only when no plan exists yet


class PlanApprovalBody(_Body):
    expected_sha256: str = Field(min_length=64, max_length=64)
    expected_latest_sha256: str = Field(min_length=64, max_length=64)


class UnavailabilityBody(_Body):
    reason_code: Literal["not_reported", "insufficient_detail", "incompatible_measure",
                         "overlapping_cohort", "figure_only", "other"]  # fmt: skip
    reason: str = Field(min_length=1)
    outcome_id: str | None = None


class DoubleCountingBody(_Body):
    study_id: str = Field(min_length=1)
    outcome_id: str = Field(min_length=1)
    justification: str = Field(min_length=1)
    keep_record_id: str | None = None
    drop_record_ids: list[str] = Field(default_factory=list)
    non_overlapping_record_ids: list[str] = Field(default_factory=list)
    expected_state_sha256: str | None = None  # GET /extraction/double-counting/state


class RoBApprovalBody(_Body):
    overall: str | None = None
    domain_judgements: dict[str, tuple[str, str]] | None = None
    nos_stars: dict[str, int] | None = None
    note: str | None = None


class AnalysisPlanBody(_Body):
    analyses: list[AnalysisSpec] = Field(min_length=1)
    rationale: str | None = None


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
        return DEFAULT_CACHE.parse(
            data, filename=item.pdf_filename, expected_sha256=item.pdf_sha256
        )

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

    @router.get("/screening/{stage}/decisions")
    def list_decisions(
        project_id: str, stage: ScreeningStage, session: SessionDep
    ) -> list[dict[str, Any]]:
        """All decisions at ``stage``; automated passes carry ``is_human: false``."""
        decisions = Repository(session).load_screening_decisions(project_id)
        return [d.model_dump(mode="json") for d in decisions if d.stage == stage]

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

    @router.post("/fulltext/{record_id}/acknowledge-xml-only")
    def xml_only(project_id: str, record_id: str, body: NoteBody, session: SessionDep,
                 reviewer: ReviewerDep) -> dict[str, Any]:  # fmt: skip
        repo = Repository(session)
        item = _call(lambda: acknowledge_xml_only(repo, project_id, record_id,
                                                  reviewer=reviewer, note=body.note))  # fmt: skip
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
        return [i.model_dump(mode="json") | {"state_sha256": item_state_sha256(i)} for i in rows]

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
            if old.review_status == ReviewStatus.PENDING:
                return correct_item(
                    repo, project_id, item_id, body.correction, doc, reviewer=reviewer,
                    note=body.note, attest_unverifiable=body.attest_unverifiable,
                    expected_state_sha256=body.expected_state_sha256,
                )  # fmt: skip
            if body.expected_state_sha256 is None:
                raise ValueError(
                    "correcting a reviewed value requires expected_state_sha256 (the state "
                    "of the item as displayed)"
                )
            new, invalidated = revise_reviewed_item(
                repo, project_id, item_id, body.correction, doc, reviewer=reviewer,
                note=body.note, attest_unverifiable=body.attest_unverifiable,
                expected_state_sha256=body.expected_state_sha256,
            )  # fmt: skip
            return new, invalidated

        result = _call(run)
        if isinstance(result, tuple):
            item, invalidated = result
            return item.model_dump(mode="json") | {"invalidated_arms": invalidated}  # type: ignore[no-any-return]
        return result.model_dump(mode="json")  # type: ignore[no-any-return]

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
                expected_state_sha256=body.expected_state_sha256,
            )
        )  # fmt: skip
        return [x.model_dump(mode="json") for x in links]

    @router.get("/studies")
    def studies(project_id: str, session: SessionDep) -> dict[str, Any]:
        """Report-to-study links, their append-only history and the state hash to send back."""
        repo = Repository(session)
        _call(lambda: repo.get_project(project_id))
        included = sorted(repo.included_record_ids(project_id, ScreeningStage.FULL_TEXT))
        return {
            "links": [x.model_dump(mode="json") for x in repo.load_study_links(project_id)],
            "history": repo.governance_records(project_id, "study_link"),
            "full_text_included": included,
            "state_sha256": links_state_sha256(repo, project_id),
        }

    @router.post("/studies/unlink")
    def unlink(project_id: str, body: UnlinkBody, session: SessionDep,
               reviewer: ReviewerDep) -> dict[str, Any]:  # fmt: skip
        repo = Repository(session)
        _call(
            lambda: unlink_reports(
                repo, project_id, body.record_ids, reviewer=reviewer, note=body.note,
                expected_state_sha256=body.expected_state_sha256,
            )
        )  # fmt: skip
        return {"unlinked": body.record_ids, "state_sha256": links_state_sha256(repo, project_id)}

    @router.get("/extraction/double-counting")
    def double_counting(project_id: str, session: SessionDep) -> list[str]:
        return Repository(session).double_counting_conflicts(project_id)

    @router.get("/extraction/double-counting/state")
    def double_counting_state(project_id: str, session: SessionDep) -> dict[str, Any]:
        repo = Repository(session)
        _call(lambda: repo.get_project(project_id))
        return {
            "conflicts": repo.double_counting_conflicts(project_id),
            "state_sha256": double_counting_state_sha256(repo, project_id),
            "study_of": repo.study_ids(project_id),
        }

    @router.post("/extraction/double-counting/resolve", status_code=status.HTTP_201_CREATED)
    def resolve_overlap(project_id: str, body: DoubleCountingBody, session: SessionDep,
                        reviewer: ReviewerDep) -> list[str]:  # fmt: skip
        repo = Repository(session)
        _call(
            lambda: resolve_double_counting(
                repo, project_id, study_id=body.study_id, outcome_id=body.outcome_id,
                reviewer=reviewer, justification=body.justification,
                keep_record_id=body.keep_record_id, drop_record_ids=body.drop_record_ids,
                non_overlapping_record_ids=body.non_overlapping_record_ids,
                expected_state_sha256=body.expected_state_sha256,
            )
        )  # fmt: skip
        return repo.double_counting_conflicts(project_id)

    @router.get("/extraction/export")
    def export(project_id: str, session: SessionDep,
               format: Literal["json", "csv"] = "json") -> Response:  # fmt: skip
        rows = export_rows(Repository(session), project_id)
        if format == "csv":
            return Response(export_csv(rows), media_type="text/csv")
        return Response(export_json(rows), media_type="application/json")

    @router.get("/extraction/ai-verifications")
    def ai_verifications(project_id: str, session: SessionDep) -> list[dict[str, Any]]:
        """Advisory second-model checks; never a review decision."""
        rows = Repository(session).load_ai_verifications(project_id)
        return [v.model_dump(mode="json") for v in rows]

    # ------------------------------------------------------------------ AI usage (read-only)
    @router.get("/ai/settings")
    def ai_settings(project_id: str, session: SessionDep) -> dict[str, Any]:
        settings = Repository(session).latest_ai_settings(project_id)
        if settings is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no AI settings saved")
        return settings.model_dump(mode="json")

    @router.get("/ai/usage")
    def ai_usage(project_id: str, session: SessionDep) -> dict[str, Any]:
        """Estimated usage totals and the per-call audit trail (no prompts, no secrets)."""
        repo = Repository(session)
        _call(lambda: repo.get_project(project_id))
        return {
            "totals": repo.llm_usage_totals(project_id),
            "calls": [c.model_dump(mode="json") for c in repo.load_llm_calls(project_id)],
            "note": "costs are list-price estimates; the provider invoice is authoritative",
        }

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

    # ------------------------------------------------------------------ Stage 5 statistics
    @router.post("/stats/plan", status_code=status.HTTP_201_CREATED)
    def approve_plan(project_id: str, body: AnalysisPlanBody, session: SessionDep,
                     reviewer: ReviewerDep) -> dict[str, Any]:  # fmt: skip
        """The authenticated reviewer approves an analysis plan (a new append-only version)."""
        repo = Repository(session)
        previous = repo.latest_analysis_plan(project_id, approved_only=False)
        plan = _call(
            lambda: AnalysisPlan(
                version=1 if previous is None else previous.version + 1,
                analyses=body.analyses, rationale=body.rationale,
                drafted_by=reviewer.display_name, status="approved",
                approved_by=reviewer.display_name, approved_reviewer_id=reviewer.id,
                approved_at=utc_now(),
            )
        )  # fmt: skip
        _call(lambda: repo.save_analysis_plan(project_id, plan))
        return plan.model_dump(mode="json")

    @router.get("/stats/plans")
    def get_plans(project_id: str, session: SessionDep) -> dict[str, Any]:
        """Every stored plan version (drafts and approvals) and the hashes to send back."""
        repo = Repository(session)
        _call(lambda: repo.get_project(project_id))
        return plan_service.plan_state(repo, project_id)

    @router.post("/stats/plan/drafts", status_code=status.HTTP_201_CREATED)
    def save_plan_draft(project_id: str, body: PlanDraftBody, session: SessionDep,
                        reviewer: ReviewerDep) -> dict[str, Any]:  # fmt: skip
        repo = Repository(session)
        draft = _call(
            lambda: plan_service.save_draft(
                repo, project_id, body.analyses, rationale=body.rationale, reviewer=reviewer,
                expected_latest_sha256=body.expected_latest_sha256,
            )
        )  # fmt: skip
        return draft.model_dump(mode="json") | {
            "sha256": draft.sha256(),
            "problems": plan_service.plan_problems(repo, project_id, draft.analyses),
        }

    @router.post("/stats/plan/drafts/{plan_id}/approve", status_code=status.HTTP_201_CREATED)
    def approve_plan_draft(
        project_id: str,
        plan_id: str,
        body: PlanApprovalBody,
        session: SessionDep,
        reviewer: ReviewerDep,
    ) -> dict[str, Any]:
        repo = Repository(session)
        plan = _call(
            lambda: plan_service.approve_draft(
                repo, project_id, plan_id, reviewer=reviewer,
                expected_sha256=body.expected_sha256,
                expected_latest_sha256=body.expected_latest_sha256,
            )
        )  # fmt: skip
        return plan.model_dump(mode="json") | {"sha256": plan.sha256()}

    @router.get("/stats/plan")
    def get_plan(project_id: str, session: SessionDep) -> dict[str, Any]:
        plan = _call(lambda: Repository(session).latest_analysis_plan(project_id))
        if plan is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no approved analysis plan")
        return plan.model_dump(mode="json")

    @router.get("/stats/output")
    def get_stats(project_id: str, session: SessionDep) -> dict[str, Any]:
        """The current statistics artifact and any reason it is not a valid basis for Stage 6."""
        repo = Repository(session)
        output = _call(lambda: repo.current_stats_output(project_id))
        if output is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no statistics artifact recorded")
        return {
            "sha256": output.sha256(),
            "problems": repo.stats_output_problems(project_id),
            "stats_output": output.model_dump(mode="json"),
        }

    return router
