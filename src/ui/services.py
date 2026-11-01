"""Non-Streamlit helpers for the local UI (kept separate so they are unit-testable).

* :class:`ApiClient` talks to the existing FastAPI application — in-process by default
  (no second program to start on Windows), or to a running server via ``SYSREV_API_URL``.
  Every human decision is sent there with the reviewer's bearer token, so gate and
  approval authentication is exactly the API's.
* :class:`UIBackend` opens repository sessions for read-mostly views and for starting
  AI steps through :class:`src.workflows.AIWorkflow` (which records proposals and the
  usage audit trail, never human decisions).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy.orm import Session, sessionmaker

from src.core.config import get_settings
from src.fulltext.storage import DocumentStore
from src.llm.catalogue import ModelCatalogue
from src.llm.credentials import CredentialResolver, redact
from src.state.db import init_db, make_engine, make_session_factory, prepare_sqlite_path
from src.state.repository import Repository
from src.workflows import AIWorkflow


@dataclass(frozen=True, slots=True)
class ApiResult:
    ok: bool
    status: int
    data: Any
    error: str | None = None


class ApiClient:
    def __init__(self, client: httpx.Client | Any) -> None:
        self._client = client

    @classmethod
    def in_process(cls, database_url: str, data_dir: Path) -> ApiClient:
        from fastapi.testclient import TestClient

        from src.api_server.app import create_app

        return cls(TestClient(create_app(database_url, data_dir), raise_server_exceptions=False))

    @classmethod
    def remote(cls, base_url: str) -> ApiClient:
        return cls(httpx.Client(base_url=base_url, timeout=60.0))

    def request(
        self,
        method: str,
        path: str,
        *,
        token: str | None = None,
        json: Any = None,
        content: bytes | None = None,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
    ) -> ApiResult:
        hdrs = dict(headers or {})
        if token:
            hdrs["Authorization"] = f"Bearer {token.strip()}"
        resp = self._client.request(
            method, path, json=json, content=content, headers=hdrs, params=params
        )
        ctype = resp.headers.get("content-type", "")
        data: Any
        if "json" in ctype or not ctype:
            try:
                data = resp.json()
            except ValueError:
                data = resp.text
        elif ctype.startswith("text/") or "svg" in ctype:
            data = resp.text
        else:  # binary exports (PNG, DOCX, XLSX) are returned unchanged
            data = resp.content
        if resp.status_code >= 400:
            detail = data.get("detail", data) if isinstance(data, dict) else data
            return ApiResult(False, resp.status_code, data, redact(str(detail))[:2000])
        return ApiResult(True, resp.status_code, data)

    def get(self, path: str, **kw: Any) -> ApiResult:
        return self.request("GET", path, **kw)

    def post(self, path: str, **kw: Any) -> ApiResult:
        return self.request("POST", path, **kw)


class UIBackend:
    def __init__(self, database_url: str, data_dir: Path) -> None:
        prepare_sqlite_path(database_url)
        data_dir.mkdir(parents=True, exist_ok=True)
        self.database_url = database_url
        self.data_dir = data_dir
        engine = make_engine(database_url)
        init_db(engine)
        self._factory: sessionmaker[Session] = make_session_factory(engine)
        self.store = DocumentStore(data_dir / "fulltext")
        self.base_catalogue = ModelCatalogue.load(data_dir)  # built-in + local override
        api_url = os.environ.get("SYSREV_API_URL", "").strip()
        self.api = (
            ApiClient.remote(api_url) if api_url else ApiClient.in_process(database_url, data_dir)
        )

    @property
    def catalogue(self) -> ModelCatalogue:
        """The active catalogue (latest activated snapshot, else built-in + override), read
        afresh so an activation in another session applies to the next step."""
        with self.repo() as repo:
            return repo.active_catalogue(self.base_catalogue)

    @classmethod
    def from_settings(cls) -> UIBackend:
        s = get_settings()
        return cls(s.database_url, Path(s.data_dir))

    @contextmanager
    def repo(self) -> Iterator[Repository]:
        with self._factory() as session:
            try:
                yield Repository(session)
                session.commit()
            except Exception:
                session.rollback()
                raise

    def workflow(
        self, repo: Repository, project_id: str, credentials: CredentialResolver
    ) -> AIWorkflow:
        return AIWorkflow(
            repo, project_id, catalogue=self.catalogue, credentials=credentials, store=self.store
        )

    def stats_dir(self, project_id: str) -> Path:
        return self.data_dir / "output" / project_id / "stats"
