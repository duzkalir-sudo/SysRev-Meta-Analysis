from __future__ import annotations

from fastapi.testclient import TestClient

from src.api_server.app import create_app


def _client() -> TestClient:
    return TestClient(create_app("sqlite:///:memory:"))


def test_health() -> None:
    r = _client().get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_create_and_get_project() -> None:
    c = _client()
    created = c.post("/projects", json={"title": "Synthetic review"})
    assert created.status_code == 201
    pid = created.json()["project_id"]
    got = c.get(f"/projects/{pid}").json()
    assert got["phase"] == "stage1_protocol"
    assert got["awaiting_gate"] is None
    assert c.get("/projects/nope").status_code == 404
    assert c.post("/projects", json={"title": ""}).status_code == 422
    assert c.post("/projects", json={"title": "x", "extra": 1}).status_code == 422


def test_gate_decision_rejected_when_not_at_gate() -> None:
    c = _client()
    pid = c.post("/projects", json={"title": "x"}).json()["project_id"]
    r = c.post(
        f"/projects/{pid}/gates/gate_1", json={"verdict": "approved", "reviewer_name": "Dr R"}
    )
    assert r.status_code == 409
    assert "not paused" in r.json()["detail"]


def test_guards_endpoint_lists_blockers() -> None:
    c = _client()
    pid = c.post("/projects", json={"title": "x"}).json()["project_id"]
    assert c.get(f"/projects/{pid}/guards").json() == {"gate1_search_strategy": []}
    assert c.get("/projects/nope/guards").status_code == 404


def test_rejection_without_comments_is_422() -> None:
    c = _client()
    pid = c.post("/projects", json={"title": "x"}).json()["project_id"]
    r = c.post(f"/projects/{pid}/gates/gate_1", json={"verdict": "rejected", "reviewer_name": "R"})
    assert r.status_code == 422
    assert (
        c.post(
            "/projects/nope/gates/gate_1", json={"verdict": "approved", "reviewer_name": "R"}
        ).status_code
        == 404
    )
