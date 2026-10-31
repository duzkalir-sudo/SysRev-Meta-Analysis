"""Session 8 model-catalogue endpoints ("Refresh Models", issue #9).

Read endpoints show the active catalogue with the four facts kept apart (account
availability from the last listing, adapter compatibility, verified capabilities, verified
prices), the stored snapshots and a change preview. Activation and human verification need a
reviewer token and are bound to the state the reviewer previewed: the request carries the
snapshot hash and the active catalogue hash shown, and is refused (nothing saved) if either
changed. Refreshing (listing provider models with an API key) is done in the local UI
process, not over HTTP, so no endpoint receives or uses provider credentials.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import text

from src.api_server.app import ReviewerDep, SessionDep
from src.core.exceptions import SysRevError
from src.llm.catalogue import ModelCatalogue
from src.llm.model_registry import (
    ModelVerification,
    VerificationRefusedError,
    diff,
    model_status,
    verification_snapshot,
)
from src.state.repository import NotFoundError, Repository


class ActivationBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_sha256: str = Field(min_length=64, max_length=64)
    expected_active_sha256: str = Field(min_length=64, max_length=64)
    note: str | None = None


class VerificationBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verification: ModelVerification
    expected_active_sha256: str = Field(min_length=64, max_length=64)


def build_model_router(base: ModelCatalogue) -> APIRouter:
    router = APIRouter(prefix="/models")

    def preview(repo: Repository, snapshot_id: str) -> dict[str, Any]:
        snapshot = repo.get_catalogue_snapshot(snapshot_id)
        active = repo.active_catalogue(base)
        return {
            "snapshot": snapshot.model_dump(mode="json"),
            "sha256": snapshot.sha256,
            "active_sha256": active.sha256,
            "based_on_active": snapshot.parent_sha256 == active.sha256,
            "changes": diff(active, snapshot.catalogue()),
        }

    @router.get("/catalogue")
    def catalogue(session: SessionDep) -> dict[str, Any]:
        repo = Repository(session)
        active = repo.active_catalogue(base)
        return {
            "active_sha256": active.sha256,
            "active_snapshot_id": active.snapshot_id or "built-in",
            "models": model_status(active, repo.latest_discovery()),
            "discovery": {k: v.model_dump(mode="json") for k, v in repo.latest_discovery().items()},
            "activations": [
                {
                    "snapshot_id": a.snapshot_id,
                    "sha256": a.sha256,
                    "previous_sha256": a.previous_sha256,
                    "reviewer_id": a.reviewer_id,
                    "note": a.note,
                    "activated_at": a.activated_at.isoformat(),
                }
                for a in repo.catalogue_activations()
            ],
            "note": "listing a model never makes it executable; only verified, activated, "
            "non-retired entries can run. Costs without a verified price are unavailable.",
        }

    @router.get("/snapshots")
    def snapshots(session: SessionDep) -> list[dict[str, Any]]:
        repo = Repository(session)
        active = repo.active_catalogue(base)
        return [
            {"snapshot_id": s.snapshot_id, "kind": s.kind, "created_at": s.created_at.isoformat(),
             "created_by": s.created_by, "sha256": s.sha256, "parent_sha256": s.parent_sha256,
             "models": len(s.models), "active": s.snapshot_id == active.snapshot_id,
             "discovery": [d.model_dump(mode="json") for d in s.discovery]}
            for s in repo.load_catalogue_snapshots()
        ]  # fmt: skip

    @router.get("/snapshots/{snapshot_id}/preview")
    def get_preview(snapshot_id: str, session: SessionDep) -> dict[str, Any]:
        try:
            return preview(Repository(session), snapshot_id)
        except NotFoundError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc

    @router.post("/snapshots/{snapshot_id}/activate")
    def activate(snapshot_id: str, body: ActivationBody, session: SessionDep,
                 reviewer: ReviewerDep) -> dict[str, Any]:  # fmt: skip
        repo = Repository(session)
        try:
            row = repo.activate_catalogue_snapshot(
                snapshot_id, base=base, expected_sha256=body.expected_sha256,
                expected_active_sha256=body.expected_active_sha256, reviewer_id=reviewer.id,
                note=body.note,
            )  # fmt: skip
        except NotFoundError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
        except SysRevError as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
        return {"activated": row.snapshot_id, "sha256": row.sha256,
                "previous_sha256": row.previous_sha256}  # fmt: skip

    @router.post("/verifications", status_code=status.HTTP_201_CREATED)
    def verify(body: VerificationBody, session: SessionDep,
               reviewer: ReviewerDep) -> dict[str, Any]:  # fmt: skip
        """Store a candidate snapshot with a human-verified entry (not active yet)."""
        repo = Repository(session)
        # write lock before the check (SQLite: one writer)
        repo.session.execute(text("UPDATE model_catalogue_activations SET note = note WHERE 1 = 0"))
        active = repo.active_catalogue(base)
        if active.sha256 != body.expected_active_sha256:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                "the active catalogue has changed since this screen was loaded; reload it",
            )
        try:
            snapshot = verification_snapshot(
                active, body.verification, created_by=reviewer.display_name,
                reviewer_id=reviewer.id,
            )  # fmt: skip
        except VerificationRefusedError as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
        repo.save_catalogue_snapshot(snapshot)
        return preview(repo, snapshot.snapshot_id)

    return router
