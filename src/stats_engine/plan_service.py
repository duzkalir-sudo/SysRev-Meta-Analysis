"""Analysis-plan drafting and approval (Session 8 UI operations).

Plans are append-only (:meth:`Repository.save_analysis_plan`). A reviewer saves a *draft*
version, may revise it (each revision is a new draft version), and approves one draft; the
approval stores the same plan id and version with ``status = approved``. Approval is bound to
the screen shown: the request carries the hash of the draft displayed and of the latest
stored plan version, and is refused (nothing saved) if either has changed — for example when
another reviewer saved a newer draft or approved a plan in the meantime.

Review of PR #13 (finding 3): the screen is also bound to the **protocol** it was composed
against (``protocol_sha256``: the approved protocol, including every outcome definition). An
amendment approved after the screen was loaded — e.g. a narrowed outcome that keeps its id —
makes draft saves and approvals from that screen fail with nothing stored, until the reviewer
reloads and reviews the current definitions. Each bound approval is recorded
(``plan_approval_basis``: plan, protocol hash and version) so that only such an approval can
close a plan re-review required by a protocol amendment.

A newly approved plan makes any statistics output computed from an earlier plan stale
(:meth:`Repository.stats_output_problems`), so Stage 5 must be re-run before Stage 6. Approval
is refused once Stage 6 has been reached (no route back to re-run Stage 5).
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select

from src.core.schemas.base import utc_now
from src.core.schemas.stats import AnalysisPlan, AnalysisSpec
from src.extraction.master_table import StaleStateError, check_expected
from src.state.db import AnalysisPlanORM, ReviewerORM
from src.state.repository import Repository


def list_plans(repo: Repository, project_id: str) -> list[AnalysisPlan]:
    rows = repo.session.scalars(
        select(AnalysisPlanORM)
        .where(AnalysisPlanORM.project_id == project_id)
        .order_by(AnalysisPlanORM.seq)
    ).all()
    return [AnalysisPlan.model_validate(r.payload) for r in rows]


def plan_problems(repo: Repository, project_id: str, analyses: list[AnalysisSpec]) -> list[str]:
    """What would refuse approval of these analyses against the current protocol."""
    protocol = repo.latest_protocol(project_id)
    if protocol is None:
        return ["no protocol: an analysis plan needs protocol outcomes"]
    outcomes = {o.outcome_id: o for o in protocol.outcomes}
    out: list[str] = []
    for spec in analyses:
        outcome = outcomes.get(spec.outcome_id)
        if outcome is None:
            out.append(f"{spec.analysis_id}: unknown outcome {spec.outcome_id}")
        else:
            out.extend(f"{spec.analysis_id}: {p}" for p in spec.check_against_outcome(outcome))
    return out


def protocol_basis(repo: Repository, project_id: str) -> tuple[str | None, int | None]:
    """(hash, version) of the current protocol; ``(None, None)`` before one exists."""
    from src.search.protocol_editor import protocol_sha256

    protocol = repo.latest_protocol(project_id)
    if protocol is None:
        return None, None
    return protocol_sha256(protocol), protocol.version


def _check_protocol(repo: Repository, project_id: str, expected: str | None) -> tuple[str, int]:
    sha, version = protocol_basis(repo, project_id)
    if sha is None or version is None:
        raise StaleStateError("no protocol: an analysis plan needs protocol outcomes")
    check_expected(sha, expected or "", "the protocol (outcome definitions)")
    return sha, version


def plan_state(repo: Repository, project_id: str) -> dict[str, Any]:
    plans = list_plans(repo, project_id)
    latest = plans[-1] if plans else None
    approved = next((p for p in reversed(plans) if p.status == "approved"), None)
    protocol = repo.latest_protocol(project_id)
    sha, version = protocol_basis(repo, project_id)
    return {
        "protocol_sha256": sha,
        "protocol_version": version,
        "outcomes": [o.model_dump(mode="json") for o in protocol.outcomes] if protocol else [],
        "plans": [p.model_dump(mode="json") | {"sha256": p.sha256()} for p in plans],
        "latest_sha256": latest.sha256() if latest else None,
        "latest_approved_sha256": approved.sha256() if approved else None,
        "draft_problems": (
            plan_problems(repo, project_id, latest.analyses)
            if latest is not None and latest.status == "draft"
            else []
        ),
    }


def _latest_sha(repo: Repository, project_id: str) -> str | None:
    plans = list_plans(repo, project_id)
    return plans[-1].sha256() if plans else None


def save_draft(
    repo: Repository,
    project_id: str,
    analyses: list[AnalysisSpec],
    *,
    rationale: str | None,
    reviewer: ReviewerORM,
    expected_latest_sha256: str | None,
    expected_protocol_sha256: str,
) -> AnalysisPlan:
    """A new draft version (the first, or a revision of what the reviewer saw)."""
    reviewer = repo.require_active_reviewer(reviewer.id)
    repo.lock_project(project_id)
    _check_protocol(repo, project_id, expected_protocol_sha256)
    latest = _latest_sha(repo, project_id)
    if latest != expected_latest_sha256:
        raise StaleStateError(
            "the analysis plan has changed since it was displayed; nothing was saved — reload"
        )
    previous = repo.latest_analysis_plan(project_id, approved_only=False)
    draft = AnalysisPlan(
        version=1 if previous is None else previous.version + 1,
        analyses=analyses,
        rationale=rationale,
        drafted_by=reviewer.display_name,
        status="draft",
    )
    repo.save_analysis_plan(project_id, draft)
    return draft


def approve_draft(
    repo: Repository,
    project_id: str,
    plan_id: str,
    *,
    reviewer: ReviewerORM,
    expected_sha256: str,
    expected_latest_sha256: str,
    expected_protocol_sha256: str,
) -> AnalysisPlan:
    reviewer = repo.require_active_reviewer(reviewer.id)
    repo.lock_project(project_id)
    protocol_sha, protocol_version = _check_protocol(repo, project_id, expected_protocol_sha256)
    plans = list_plans(repo, project_id)
    draft = next((p for p in plans if p.plan_id == plan_id and p.status == "draft"), None)
    if draft is None:
        raise StaleStateError(f"no draft plan {plan_id} to approve")
    check_expected(draft.sha256(), expected_sha256, "the displayed draft")
    check_expected(plans[-1].sha256(), expected_latest_sha256, "the analysis plan")
    if plans[-1].plan_id != plan_id:
        raise StaleStateError("a newer plan version exists; reload and review it")
    approved = AnalysisPlan.model_validate(
        draft.model_dump()
        | {
            "status": "approved",
            "approved_by": reviewer.display_name,
            "approved_reviewer_id": reviewer.id,
            "approved_at": utc_now(),
        }
    )
    repo.save_analysis_plan(project_id, approved)
    repo.add_governance_record(
        project_id, "plan_approval_basis", approved.plan_id,
        {"plan_id": approved.plan_id, "version": approved.version,
         "plan_sha256": approved.sha256(), "protocol_sha256": protocol_sha,
         "protocol_version": protocol_version},
        reviewer_id=reviewer.id,
    )  # fmt: skip
    return approved
