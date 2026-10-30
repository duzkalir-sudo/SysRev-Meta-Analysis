"""Deterministic test doubles for Session 2 (no network, no credentials).

All clinical content is SYNTHETIC and invented for tests.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

from src.core.enums import ReviewerRole, ScreeningStage, ScreeningVerdict
from src.core.schemas import ScreeningDecision, StudyRecord
from src.llm.provider import LLMConfig, LLMRequest, LLMResponse
from src.state.db import ReviewerORM
from src.state.repository import Repository
from tests.conftest import NOW, make_protocol

Reply = str | dict[str, Any] | Exception
Responder = Callable[[LLMRequest], Reply]


class ScriptedProvider:
    """Returns scripted replies; records every request it receives."""

    def __init__(self, model_id: str, responder: Responder | Sequence[Reply]) -> None:
        self._model_id = model_id
        self.requests: list[LLMRequest] = []
        if callable(responder):
            self._responder = responder
        else:
            replies = list(responder)
            self._responder = lambda _req: replies.pop(0)

    @property
    def provider_name(self) -> str:
        return "scripted"

    @property
    def model_id(self) -> str:
        return self._model_id

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        reply = self._responder(request)
        if isinstance(reply, Exception):
            raise reply
        text = reply if isinstance(reply, str) else json.dumps(reply)
        return LLMResponse(text=text, provider="scripted", model_id=self._model_id)


def config(label: str, model: str = "model-a") -> LLMConfig:
    return LLMConfig(provider="scripted", model_id=model, label=label)


def approved_protocol(**overrides: Any) -> Any:
    return make_protocol(status="approved", approved_by="Dr Human", approved_at=NOW, **overrides)


def include_at_title_abstract(
    repo: Repository, project_id: str, records: Sequence[StudyRecord]
) -> None:
    """Two automated passes agree to include each record (synthetic decisions)."""
    for r in records:
        for role in (ReviewerRole.REVIEWER_1, ReviewerRole.REVIEWER_2):
            repo.save_screening_decision(
                project_id,
                ScreeningDecision(
                    record_id=r.record_id,
                    reviewer=role,
                    reviewer_identity=f"agent-{role}",
                    stage=ScreeningStage.TITLE_ABSTRACT,
                    verdict=ScreeningVerdict.INCLUDE,
                    rationale="synthetic include",
                ),
            )


def enroll(repo: Repository, name: str = "Dr Human") -> ReviewerORM:
    reviewer_id, _ = repo.enroll_reviewer(name)
    return repo.require_active_reviewer(reviewer_id)
