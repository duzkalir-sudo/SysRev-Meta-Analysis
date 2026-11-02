"""Session 10 review (PR #15), P1: an HTTP success must never precede the commit.

The Windows preflight received ``201 Created`` for ``POST /projects`` and lost the project
when the server was stopped immediately afterwards. ``get_session`` committed in the code
after ``yield``, and FastAPI's default (request-scoped) dependency exit runs after the
response has been sent. These tests use a real HTTP server (uvicorn on 127.0.0.1), the
normal ``create_app`` and a file-based SQLite database; only ``Session.commit`` is wrapped
to delay or fail it. The first test is the independent reviewer's reproduction
(``test_pr15_commit_timing.py``), kept with its oracle. Synthetic data only.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from src.api_server.app import create_app
from src.state.repository import Repository
from tests.search_helpers import draft_data
from validation.session10.preflight import free_port

ROOT = Path(__file__).resolve().parents[2]


class _Server:
    def __init__(self, app: FastAPI) -> None:
        self.app = app
        self.port = free_port()
        self.server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="error")
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


@pytest.fixture
def served(tmp_path: Path) -> Iterator[_Server]:
    app = create_app(f"sqlite:///{tmp_path / 'timing.sqlite'}", tmp_path / "data")
    srv = _Server(app)
    srv.thread.start()
    for _ in range(250):
        if srv.server.started:
            break
        time.sleep(0.02)
    assert srv.server.started
    try:
        yield srv
    finally:
        srv.server.should_exit = True
        srv.thread.join(timeout=10)
        app.state.session_factory.kw["bind"].dispose()


class _Barrier:
    """Wraps ``Session.commit``: holds the first ``hold`` commits until released."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, hold: int = 1) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.held = 0
        self.hold = hold
        original = Session.commit

        def commit(session: Session) -> None:
            if self.held < self.hold:
                self.held += 1
                self.entered.set()
                assert self.release.wait(10), "test commit barrier timed out"
            original(session)

        monkeypatch.setattr(Session, "commit", commit)


def _post(url: str, **kw: Any) -> httpx.Response:
    return httpx.post(url, timeout=15, trust_env=False, **kw)


def _put(url: str, **kw: Any) -> httpx.Response:
    return httpx.put(url, timeout=15, trust_env=False, **kw)


def test_http_success_waits_for_database_commit(
    served: _Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reviewer's reproduction: while the commit is blocked, no response is sent."""
    barrier = _Barrier(monkeypatch)
    with ThreadPoolExecutor(max_workers=1) as pool:
        try:
            response = pool.submit(_post, f"{served.url}/projects",
                                   json={"title": "synthetic commit timing"})  # fmt: skip
            assert barrier.entered.wait(5)
            time.sleep(0.2)
            with served.app.state.session_factory() as other:
                assert Repository(other).list_projects() == []  # not yet committed
            premature = response.done()
            detail = response.result().status_code if premature else "waiting"
            assert not premature, f"HTTP {detail} was returned while the commit was blocked"
            barrier.release.set()
            assert response.result(timeout=5).status_code == 201
        finally:
            barrier.release.set()
    with served.app.state.session_factory() as other:
        assert len(Repository(other).list_projects()) == 1


def test_authenticated_write_waits_for_commit(
    served: _Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A representative authenticated write (bearer token): saving a protocol draft."""
    factory = served.app.state.session_factory
    with factory() as s:
        repo = Repository(s)
        _, token = repo.enroll_reviewer("Dr Synthetic Reviewer")
        repo.create_project("p1", "Synthetic project")
        s.commit()
    auth = {"Authorization": f"Bearer {token}"}
    url = f"{served.url}/projects/p1/protocol"
    assert _put(url, json=draft_data()).status_code == 401  # authentication unchanged
    barrier = _Barrier(monkeypatch)
    with ThreadPoolExecutor(max_workers=1) as pool:
        try:
            response = pool.submit(_put, url, json=draft_data(), headers=auth)
            assert barrier.entered.wait(5)
            time.sleep(0.2)
            with factory() as other:
                assert Repository(other).list_protocol_events("p1") == []
            assert not response.done(), "HTTP response sent before the protocol was committed"
            barrier.release.set()
            ok = response.result(timeout=5)
            assert ok.status_code == 200, ok.text
            assert ok.json()["new_version"] is True
        finally:
            barrier.release.set()
    with factory() as other:
        events = Repository(other).list_protocol_events("p1")
    assert len(events) == 1


def test_failed_commit_is_an_error_and_stores_nothing(
    served: _Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = Session.commit
    failures = {"left": 1}

    def failing_commit(session: Session) -> None:
        if failures["left"]:
            failures["left"] -= 1
            raise OperationalError("COMMIT", {}, Exception("synthetic disk I/O error"))
        original(session)

    monkeypatch.setattr(Session, "commit", failing_commit)
    r = _post(f"{served.url}/projects", json={"title": "never stored"})
    assert r.status_code == 500
    assert "rolled back" in r.json()["detail"] and "nothing" in r.json()["detail"]
    with served.app.state.session_factory() as other:
        assert Repository(other).list_projects() == []
    # the server stays usable and the next request is stored
    again = _post(f"{served.url}/projects", json={"title": "stored"})
    assert again.status_code == 201
    with served.app.state.session_factory() as other:
        assert [p.title for p in Repository(other).list_projects()] == ["stored"]


def test_refused_request_is_rolled_back(served: _Server) -> None:
    """An endpoint error still rolls back (unchanged): no reviewer, no protocol."""
    with served.app.state.session_factory() as s:
        Repository(s).create_project("p2", "Synthetic project")
        s.commit()
    r = _put(f"{served.url}/projects/p2/protocol", json=draft_data(),
             headers={"Authorization": "Bearer not-a-token"})  # fmt: skip
    assert r.status_code == 401
    with served.app.state.session_factory() as other:
        assert Repository(other).list_protocol_events("p2") == []


def test_acknowledged_write_survives_an_immediate_hard_stop(tmp_path: Path) -> None:
    """What the Windows preflight did: 201, then the server is killed at once and restarted.

    ``Popen.kill`` is SIGKILL on POSIX and TerminateProcess on Windows: no graceful
    shutdown, so only data committed before the response can survive."""
    env = dict(os.environ)
    env["SYSREV_DATABASE_URL"] = f"sqlite:///{(tmp_path / 'db' / 'sysrev.db').as_posix()}"
    env["SYSREV_DATA_DIR"] = str(tmp_path / "data")
    env["PYTHONPATH"] = str(ROOT)
    port = free_port()
    cmd = [sys.executable, "-m", "uvicorn", "src.api_server.main:app", "--host", "127.0.0.1",
           "--port", str(port), "--log-level", "warning"]  # fmt: skip
    base = f"http://127.0.0.1:{port}"

    def start() -> subprocess.Popen[bytes]:
        proc = subprocess.Popen(cmd, env=env, cwd=tmp_path, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)  # fmt: skip
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                if httpx.get(f"{base}/health", timeout=2, trust_env=False).status_code == 200:
                    return proc
            except httpx.HTTPError:
                time.sleep(0.2)
        proc.kill()
        raise AssertionError("API server did not start")

    acknowledged = []
    for i in range(3):
        proc = start()
        try:
            r = _post(f"{base}/projects", json={"title": f"Kalıcı proje {i} ğüşİ"})
            assert r.status_code == 201
            acknowledged.append(r.json()["project_id"])
        finally:
            proc.kill()  # immediately after the acknowledgement
            proc.wait(timeout=20)
    proc = start()
    try:
        listed = httpx.get(f"{base}/projects", timeout=10, trust_env=False).json()
        titles = {p["project_id"]: p["title"] for p in listed}
    finally:
        proc.kill()
        proc.wait(timeout=20)
    assert [titles.get(p) for p in acknowledged] == [f"Kalıcı proje {i} ğüşİ" for i in range(3)]
