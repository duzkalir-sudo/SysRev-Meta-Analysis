"""Session 7: headless Streamlit checks of the reporting pages (SYNTHETIC project).

Each page must render without an exception, show the stored state, and refuse decisions
without a reviewer token (the decision is made by the API, never by the UI)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.config import get_settings
from src.manuscript.service import ManuscriptService
from src.state.db import init_db, make_engine, make_session_factory
from src.state.repository import Repository
from tests.manuscript_helpers import SYSTEM, comparative_project


@pytest.mark.parametrize(
    ("page", "expect", "button"),
    [
        ("PRISMA flow & checklists", "PRISMA 2020 flow diagram", "Record decision"),
        ("GRADE & Summary of Findings", "Summary of Findings", "Save (authenticated)"),
        ("Synthesis without MA & SWiM", "Synthesis without meta-analysis", "Preview"),
    ],
)
def test_reporting_pages_render_and_require_a_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, page: str, expect: str, button: str
) -> None:
    pytest.importorskip("streamlit")
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    url = f"sqlite:///{tmp_path / 'ui.db'}"
    monkeypatch.setenv("SYSREV_DATABASE_URL", url)
    monkeypatch.setenv("SYSREV_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("SYSREV_API_URL", raising=False)
    get_settings.cache_clear()
    st.cache_resource.clear()
    engine = make_engine(url)
    init_db(engine)
    with make_session_factory(engine)() as s:
        repo, _, _ = comparative_project(s)
        ManuscriptService(repo, "mc1").create_template_draft(SYSTEM)
        s.commit()
    app_path = str(Path(__file__).resolve().parents[2] / "src" / "ui" / "app.py")
    at = AppTest.from_file(app_path, default_timeout=60)
    at.session_state["project_id"] = "mc1"
    at.session_state["page"] = page
    at.run()
    assert not at.exception
    assert any(expect in h.value for h in at.header)
    target = [b for b in at.button if button in b.label]
    assert target
    target[0].click().run()
    assert not at.exception
    # a decision without a token is refused by the API; an incomplete synthesis is already
    # refused at the preview step, and no save is offered
    errors = [e.value for e in at.error]
    assert any(
        "Not authenticated" in e or "Refused" in e or "at least 1 item" in e for e in errors
    ), errors
    assert not [b for b in at.button if "Save the previewed group" in b.label]
    with make_session_factory(engine)() as s:
        repo = Repository(s)
        assert repo.load_checklist_decisions("mc1") == []
        assert repo.load_grade_assessments("mc1") == []
        assert repo.load_synthesis_groups("mc1") == []
    engine.dispose()
    get_settings.cache_clear()
    st.cache_resource.clear()
