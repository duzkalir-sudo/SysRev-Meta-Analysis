"""Session 4: critical UI workflows, driven headlessly with Streamlit's AppTest.

The UI runs against a temporary SQLite database and the in-process FastAPI app.
No provider is called.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from src.core.config import get_settings
from src.llm.tasks import AITask
from src.state.db import init_db, make_engine, make_session_factory
from src.state.repository import Repository

APP = str(Path(__file__).resolve().parents[2] / "src" / "ui" / "app.py")
SECRET = "sk-ant-ui-test-0123456789abcdefSECRET"


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    import streamlit as st

    url = f"sqlite:///{tmp_path / 'ui.db'}"
    monkeypatch.setenv("SYSREV_DATABASE_URL", url)
    monkeypatch.setenv("SYSREV_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("SYSREV_API_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    get_settings.cache_clear()
    st.cache_resource.clear()
    yield url
    get_settings.cache_clear()
    st.cache_resource.clear()


def app(page: str, **state: object) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=30)
    for k, v in state.items():
        at.session_state[k] = v
    at.session_state["page"] = page
    return at.run()


def make_project(url: str, title: str = "Synthetic UI review") -> tuple[str, str]:
    engine = make_engine(url)
    init_db(engine)
    with make_session_factory(engine)() as s:
        repo = Repository(s)
        repo.create_project("uiproj", title)
        _, token = repo.enroll_reviewer("Dr Human")
        s.commit()
    return "uiproj", token


def test_create_and_select_project(db: str) -> None:
    at = app("Projects")
    assert not at.exception
    at.text_input(key="new_project_title").input("My synthetic review")
    at.button[0].click().run()  # "Create project" form submit
    assert not at.exception
    assert at.session_state["project_id"]
    at2 = app("Projects")
    assert "My synthetic review" in at2.selectbox[0].options[0]


def test_pipeline_shows_phase_blockers_and_gate_requires_token(db: str) -> None:
    pid, token = make_project(db)
    at = app("Pipeline status", project_id=pid)
    assert not at.exception
    assert at.metric[0].value == "stage1_protocol"
    # request the Stage 1 → Gate 1 transition without a token: refused by the API
    at.button[0].click().run()
    assert any("Not authenticated" in e.value for e in at.error)
    at = app("Pipeline status", project_id=pid, reviewer_token=token)
    at.button[0].click().run()
    assert any("Transition recorded" in s.value for s in at.success)
    at = app("Pipeline status", project_id=pid)
    assert at.metric[0].value == "gate1_search_strategy"
    # at a gate, an approval without a token is refused
    at.button[0].click().run()
    assert any("Not authenticated" in e.value for e in at.error)


def test_model_settings_saved_without_keys_and_session_key_not_persisted(db: str) -> None:
    pid, _ = make_project(db)
    at = app("AI models & keys", project_id=pid)
    assert not at.exception
    assert any("development credits" in i.value for i in at.info)
    at.text_input(key="key_anthropic").input(SECRET)
    at.button(key="use_anthropic").click().run()
    assert any("memory" in s.value for s in at.success)
    assert at.selectbox(key="extraction_model").value == "— choose a model —"
    at.selectbox(key="extraction_model").select("claude-sonnet-5-5").run()
    at.selectbox(key="extraction_effort").select("high")
    save = next(b for b in at.button if b.label == "Save model settings")
    save.click().run()
    assert any("settings version 1" in s.value for s in at.success)
    engine = create_engine(db)
    with engine.connect() as conn:
        dump = json.dumps([
            [str(v) for v in row]
            for table in ("project_ai_settings", "llm_calls", "projects")
            for row in conn.execute(text(f"SELECT * FROM {table}"))
        ])  # fmt: skip
    assert SECRET not in dump
    with make_session_factory(make_engine(db))() as s:
        saved = Repository(s).latest_ai_settings(pid)
    assert saved is not None and saved.tasks[AITask.EXTRACTION].effort == "high"
    assert AITask.SCREENING_REVIEWER_1 not in saved.tasks  # never chosen implicitly


def test_run_page_reports_missing_settings_and_preflight(db: str) -> None:
    pid, _ = make_project(db)
    at = app("Run AI steps", project_id=pid)
    assert any("Save model settings first" in w.value for w in at.warning)


def test_review_pages_render_and_usage_page_shows_estimates(db: str) -> None:
    pid, _ = make_project(db)
    for page in ("Screening", "Full text & extraction", "Risk of bias", "Statistics",
                 "Usage & costs"):  # fmt: skip
        at = app(page, project_id=pid)
        assert not at.exception, (page, at.exception)
    at = app("Usage & costs", project_id=pid)
    assert at.metric[0].value == "0.0000"
    assert any("invoice is authoritative" in i.value for i in at.info)


def test_adjudication_requires_authentication(db: str) -> None:
    pid, _ = make_project(db)
    at = app("Screening", project_id=pid)
    submit = next(b for b in at.button if "adjudication" in b.label)
    submit.click().run()
    assert not any("Adjudication recorded" in s.value for s in at.success)
