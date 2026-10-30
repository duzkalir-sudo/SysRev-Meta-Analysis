"""Session 5 UI: protocol editor and searches page, headless (Streamlit AppTest).

Runs against a temporary SQLite database and the in-process API; PubMed is mocked.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from pathlib import Path

import pytest
import respx

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from src.apis.pubmed import EUTILS_BASE
from src.core.config import get_settings
from src.core.enums import Phase
from src.state.db import init_db, make_engine, make_session_factory
from src.state.repository import Repository
from tests.search_helpers import FakePubMed, approved_in_stage2

APP = str(Path(__file__).resolve().parents[2] / "src" / "ui" / "app.py")


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    import streamlit as st

    url = f"sqlite:///{tmp_path / 'ui.db'}"
    monkeypatch.setenv("SYSREV_DATABASE_URL", url)
    monkeypatch.setenv("SYSREV_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SYSREV_NCBI_EMAIL", "t@example.org")
    monkeypatch.delenv("SYSREV_API_URL", raising=False)
    get_settings.cache_clear()
    st.cache_resource.clear()
    yield url
    get_settings.cache_clear()
    st.cache_resource.clear()


def setup(url: str) -> tuple[str, str]:
    engine = make_engine(url)
    init_db(engine)
    with make_session_factory(engine)() as s:
        repo = Repository(s)
        repo.create_project("uip", "Synthetic UI review")
        _, token = repo.enroll_reviewer("Dr Human")
        s.commit()
    engine.dispose()
    return "uip", token


def run(page: str, **state: object) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    for k, v in state.items():
        at.session_state[k] = v
    at.session_state["page"] = page
    return at.run()


def button(at: AppTest, key: str) -> object:
    return next(b for b in at.button if b.key == key)


def submit(at: AppTest, label: str) -> AppTest:
    return next(b for b in at.button if b.label == label).click().run()


def test_protocol_editor_builds_saves_and_reports_validation(db: str) -> None:
    pid, token = setup(db)
    at = run("Protocol", project_id=pid)
    assert not at.exception
    at.text_input(key="pq_title").input("Synthetic drug X review")
    at.text_area(key="pq_rq").input("Does drug X reduce Z?")
    at.text_area(key="pq_p").input("Adults with Y")
    at.text_area(key="pq_i").input("Drug X")
    at.text_input(key="pq_c").input("Placebo")
    at = submit(at, "Apply question & design")
    at.text_input(key="po_id").input("hba1c")
    at.text_input(key="po_name").input("HbA1c change")
    at.selectbox(key="po_type").select("continuous")
    at.selectbox(key="po_em").select("MD")
    at = submit(at, "Add / replace outcome")
    assert not at.error
    at.multiselect(key="pe_designs").select("rct")
    at.text_area(key="pe_inc").input("Randomised trials\nAdults")
    at.text_area(key="pe_exc").input("Animal studies")
    at = submit(at, "Apply eligibility")
    at.text_area(key="pm_terms").input("Diabetes Mellitus")
    at = submit(at, "Apply MeSH terms")
    at.multiselect(key="pd_dbs").select("PubMed").select("Embase")
    at = submit(at, "Apply database selection")
    at.text_area(key="ps_query_pubmed").input('"drug x"[tiab]')
    at.text_input(key="ps_from_pubmed").input("2000-01-01")
    at = submit(at, "Apply PubMed strategy")
    at.text_area(key="ps_query_embase").input("'drug x'/exp")
    at = submit(at, "Apply Embase strategy")
    assert not at.exception and not at.error

    # Saving without a token is refused by the API.
    at = button(at, "proto_save").click().run()  # type: ignore[attr-defined]
    assert any("Not authenticated" in e.value for e in at.error)
    at.session_state["reviewer_token"] = token
    at = button(at, "proto_save").click().run()  # type: ignore[attr-defined]
    assert any("Draft saved" in s.value for s in at.success)
    engine = make_engine(db)
    with make_session_factory(engine)() as s:
        protocol = Repository(s).latest_protocol(pid)
    assert protocol is not None and protocol.status == "draft"
    assert [o.outcome_id for o in protocol.outcomes] == ["hba1c"]
    engine.dispose()
    assert {st.database.value for st in protocol.search_strategies} == {"pubmed", "embase"}
    assert protocol.search_strategies[0].date_from == date(2000, 1, 1)
    # Approval is refused while MeSH/strategies are unverified; reasons are shown.
    at = button(at, "proto_approve").click().run()  # type: ignore[attr-defined]
    assert any("cannot be approved" in e.value for e in at.error)
    assert any("not verified against NLM MeSH" in w.value for w in at.warning)


@respx.mock
def test_searches_page_executes_approved_pubmed_search(db: str) -> None:
    pid, token = setup(db)
    engine = make_engine(db)
    with make_session_factory(engine)() as s:
        repo = Repository(s)
        reviewer = repo.list_reviewers()[0]
        approved_in_stage2(
            repo,
            pid,
            reviewer,
            selected_databases=["pubmed"],
            search_strategies=[{"database": "pubmed", "query": "drug x"}],
        )
        assert repo.get_project(pid).phase == Phase.STAGE2_SEARCH_SCREENING
        sha = repo.latest_protocol(pid).search_strategies[0].sha256  # type: ignore[union-attr]
        s.commit()
    fake = FakePubMed({"96000001": date(2020, 1, 1), "96000002": date(2021, 1, 1)})
    respx.get(f"{EUTILS_BASE}esearch.fcgi").mock(side_effect=fake.esearch)
    respx.post(f"{EUTILS_BASE}efetch.fcgi").mock(side_effect=fake.efetch)
    at = run("Searches & import", project_id=pid, reviewer_token=token)
    assert not at.exception
    assert any("not yet searched" in e.label for e in at.expander)
    at = button(at, f"exe_{sha}").click().run()  # type: ignore[attr-defined]
    assert not at.exception
    assert any("Search executed" in s.value for s in at.success)
    at = button(at, "prepare").click().run()  # type: ignore[attr-defined]
    assert any("Records prepared" in s.value for s in at.success)
    with make_session_factory(engine)() as s:
        repo = Repository(s)
        assert len(repo.load_records(pid, unique_only=True)) == 2
        assert repo.search_identification_problems(pid) == []
        assert repo.load_prisma(pid).total_identified == 2  # type: ignore[union-attr]
    engine.dispose()
