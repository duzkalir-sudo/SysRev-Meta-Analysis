"""FastAPI application: project state and HITL gate endpoints.

All phase changes go through :class:`PipelineStateMachine`; guard facts are
computed from the database by :meth:`Repository.build_guard_context`.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session, sessionmaker

from src import __version__
from src.core.enums import GateId, GateVerdict, Phase
from src.core.exceptions import GuardViolationError, StateTransitionError
from src.core.schemas.base import new_id
from src.core.schemas.pipeline import GateDecision, StateTransition
from src.state.db import init_db, make_engine, make_session_factory
from src.state.machine import GATE_OF_PHASE
from src.state.repository import NotFoundError, Repository


class CreateProjectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1)


class GateDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdict: GateVerdict
    reviewer_name: str = Field(min_length=1)
    comments: str | None = None


class ProjectView(BaseModel):
    project_id: str
    title: str
    phase: Phase
    awaiting_gate: GateId | None
    history: list[StateTransition]


def get_session(request: Request) -> Iterator[Session]:
    factory: sessionmaker[Session] = request.app.state.session_factory
    with factory() as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise


SessionDep = Annotated[Session, Depends(get_session)]


def create_app(database_url: str = "sqlite:///:memory:") -> FastAPI:
    engine = make_engine(database_url)
    init_db(engine)
    factory: sessionmaker[Session] = make_session_factory(engine)

    app = FastAPI(title="SysRev Meta-Analysis Agent", version=__version__)
    app.state.session_factory = factory

    def _view(repo: Repository, project_id: str) -> ProjectView:
        project = repo.get_project(project_id)
        phase = Phase(project.phase)
        return ProjectView(
            project_id=project.id,
            title=project.title,
            phase=phase,
            awaiting_gate=GATE_OF_PHASE.get(phase),
            history=repo.list_transitions(project_id),
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.post("/projects", status_code=status.HTTP_201_CREATED)
    def create_project(body: CreateProjectRequest, session: SessionDep) -> ProjectView:
        repo = Repository(session)
        project_id = new_id()
        repo.create_project(project_id, body.title)
        return _view(repo, project_id)

    @app.get("/projects/{project_id}")
    def get_project(project_id: str, session: SessionDep) -> ProjectView:
        repo = Repository(session)
        try:
            return _view(repo, project_id)
        except NotFoundError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc

    @app.get("/projects/{project_id}/guards")
    def get_guards(project_id: str, session: SessionDep) -> dict[str, list[str]]:
        """Blocking reasons for every target reachable from the current phase."""
        repo = Repository(session)
        try:
            machine = repo.load_state_machine(project_id)
        except NotFoundError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
        context = repo.build_guard_context(project_id)
        return {t.value: machine.check(t, context) for t in sorted(machine.allowed_targets())}

    @app.post("/projects/{project_id}/gates/{gate}")
    def decide_gate(
        project_id: str, gate: GateId, body: GateDecisionRequest, session: SessionDep
    ) -> ProjectView:
        repo = Repository(session)
        try:
            machine = repo.load_state_machine(project_id)
        except NotFoundError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
        try:
            decision = GateDecision(
                gate=gate,
                verdict=body.verdict,
                reviewer_name=body.reviewer_name,
                comments=body.comments,
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        try:
            transition = machine.submit_gate_decision(
                decision, repo.build_guard_context(project_id)
            )
        except GuardViolationError as exc:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                {"message": str(exc), "violations": exc.violations},
            ) from exc
        except StateTransitionError as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
        repo.record_transition(transition)
        return _view(repo, project_id)

    return app
