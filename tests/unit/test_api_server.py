from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from src.api_server.app import create_app
from src.core.enums import ActorType, Phase
from src.state.repository import Repository
from tests.conftest import READY


@pytest.fixture
def c() -> Iterator[TestClient]:
    with TestClient(create_app("sqlite:///:memory:")) as client:  # runs lifespan teardown
        yield client


def _enroll(c: TestClient, name: str = "Dr R") -> tuple[str, str]:
    with c.app.state.session_factory() as session:  # type: ignore[attr-defined]
        reviewer_id, token = Repository(session).enroll_reviewer(name)
        session.commit()
    return reviewer_id, token


def _to_gate1(c: TestClient, pid: str) -> None:
    with c.app.state.session_factory() as session:  # type: ignore[attr-defined]
        repo = Repository(session)
        m = repo.load_state_machine(pid)
        repo.record_transition(
            m.transition(Phase.GATE1_SEARCH_STRATEGY, actor="orchestrator",
                         actor_type=ActorType.SYSTEM, context=READY)
        )  # fmt: skip
        session.commit()


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_health(c: TestClient) -> None:
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_create_and_get_project(c: TestClient) -> None:
    created = c.post("/projects", json={"title": "Synthetic review"})
    assert created.status_code == 201
    pid = created.json()["project_id"]
    got = c.get(f"/projects/{pid}").json()
    assert got["phase"] == "stage1_protocol"
    assert got["awaiting_gate"] is None
    assert c.get("/projects/nope").status_code == 404
    assert c.post("/projects", json={"title": ""}).status_code == 422
    assert c.post("/projects", json={"title": "x", "extra": 1}).status_code == 422


def test_gate_decision_rejected_when_not_at_gate(c: TestClient) -> None:
    _, token = _enroll(c)
    pid = c.post("/projects", json={"title": "x"}).json()["project_id"]
    r = c.post(f"/projects/{pid}/gates/gate_1", json={"verdict": "approved"},
               headers=_bearer(token))  # fmt: skip
    assert r.status_code == 409
    assert "not paused" in r.json()["detail"]


def test_guards_endpoint_lists_blockers(c: TestClient) -> None:
    pid = c.post("/projects", json={"title": "x"}).json()["project_id"]
    assert c.get(f"/projects/{pid}/guards").json() == {"gate1_search_strategy": []}
    assert c.get("/projects/nope/guards").status_code == 404


def test_rejection_without_comments_is_422_and_unknown_project_404(c: TestClient) -> None:
    _, token = _enroll(c)
    pid = c.post("/projects", json={"title": "x"}).json()["project_id"]
    r = c.post(f"/projects/{pid}/gates/gate_1", json={"verdict": "rejected"},
               headers=_bearer(token))  # fmt: skip
    assert r.status_code == 422
    r = c.post("/projects/nope/gates/gate_1", json={"verdict": "approved"},
               headers=_bearer(token))  # fmt: skip
    assert r.status_code == 404


# --------------------------------------------------------------------------- authentication
# Regression tests for the PR #1 review: an agent must not be able to approve a gate by
# sending an HTTP request that merely names the human reviewer.


def test_gate_approval_without_token_is_401_and_changes_nothing(c: TestClient) -> None:
    _enroll(c, "Dr R")
    pid = c.post("/projects", json={"title": "x"}).json()["project_id"]
    _to_gate1(c, pid)
    for headers in ({}, {"Authorization": "Basic abc"}, {"Authorization": "Bearer"}):
        r = c.post(f"/projects/{pid}/gates/gate_1", json={"verdict": "approved"},
                   headers=headers)  # fmt: skip
        assert r.status_code == 401
        assert r.headers["WWW-Authenticate"] == "Bearer"
    assert c.get(f"/projects/{pid}").json()["phase"] == "gate1_search_strategy"


def test_naming_a_reviewer_in_the_body_is_not_accepted(c: TestClient) -> None:
    _, token = _enroll(c, "Dr R")
    pid = c.post("/projects", json={"title": "x"}).json()["project_id"]
    _to_gate1(c, pid)
    body = {"verdict": "approved", "reviewer_name": "Dr R"}
    assert c.post(f"/projects/{pid}/gates/gate_1", json=body).status_code == 401
    # Even with a valid token, a body-supplied identity is rejected rather than ignored.
    r = c.post(f"/projects/{pid}/gates/gate_1", json=body, headers=_bearer(token))
    assert r.status_code == 422


def test_forged_tampered_and_revoked_tokens_are_401(c: TestClient) -> None:
    reviewer_id, token = _enroll(c, "Dr R")
    pid = c.post("/projects", json={"title": "x"}).json()["project_id"]
    _to_gate1(c, pid)
    forged = [
        "not-a-token",
        f"rvw_{reviewer_id}_" + "A" * 43,  # right id, wrong secret
        token[:-1] + ("A" if token[-1] != "A" else "B"),  # tampered secret
        "rvw_" + "0" * 32 + token[len("rvw_") + 32 :],  # right secret, unknown id
    ]
    for bad in forged:
        r = c.post(f"/projects/{pid}/gates/gate_1", json={"verdict": "approved"},
                   headers=_bearer(bad))  # fmt: skip
        assert r.status_code == 401, bad
    with c.app.state.session_factory() as session:  # type: ignore[attr-defined]
        Repository(session).revoke_reviewer(reviewer_id)
        session.commit()
    r = c.post(f"/projects/{pid}/gates/gate_1", json={"verdict": "approved"},
               headers=_bearer(token))  # fmt: skip
    assert r.status_code == 401


def test_identity_is_taken_from_token(c: TestClient) -> None:
    reviewer_id, token = _enroll(c, "Dr Real Human")
    pid = c.post("/projects", json={"title": "x"}).json()["project_id"]
    _to_gate1(c, pid)
    r = c.post(f"/projects/{pid}/gates/gate_1",
               json={"verdict": "rejected", "comments": "add Embase strategy"},
               headers=_bearer(token))  # fmt: skip
    assert r.status_code == 200
    decision = r.json()["history"][-1]["gate_decision"]
    assert decision["reviewer_name"] == "Dr Real Human"
    assert decision["reviewer_id"] == reviewer_id
    assert r.json()["phase"] == "stage1_protocol"
