"""Session 8 UI workflows (Streamlit AppTest, headless). Every decision is authenticated with a
real reviewer token and driven through the widgets; intervening changes are made by *another*
database session, and a submission from the older screen must be refused with nothing saved.
All data SYNTHETIC; provider list endpoints are mocked (respx); no model is called.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from sqlalchemy import func, select
from sqlalchemy.engine import Engine

from src.core.config import get_settings
from src.core.enums import Phase
from src.core.schemas.stats import AnalysisSpec
from src.extraction import link_reports_to_study, links_state_sha256
from src.llm.discovery import ANTHROPIC_MODELS_URL, OPENAI_MODELS_URL
from src.search import amendments
from src.search import protocol_editor as editor
from src.state.db import (
    ModelCatalogueActivationORM,
    init_db,
    make_engine,
    make_session_factory,
)
from src.state.repository import Repository
from src.stats_engine import plan_service
from tests.manuscript_helpers import comparative_project

pytest.importorskip("streamlit")
PID = "mc1"


@dataclass
class Env:
    engine: Engine
    rid: str
    token: str

    def other(self) -> Any:
        return make_session_factory(self.engine)()

    def app(self, page: str) -> Any:
        from streamlit.testing.v1 import AppTest

        path = str(Path(__file__).resolve().parents[2] / "src" / "ui" / "app.py")
        at = AppTest.from_file(path, default_timeout=60)
        at.session_state["project_id"] = PID
        at.session_state["page"] = page
        at.session_state["reviewer_token"] = self.token
        at.run()
        assert not at.exception, at.exception
        return at


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Env]:
    import streamlit as st

    url = f"sqlite:///{tmp_path / 'ui8.db'}"
    monkeypatch.setenv("SYSREV_DATABASE_URL", url)
    monkeypatch.setenv("SYSREV_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-ui-test-0123456789abcdefSECRET")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-uitest0123456789abcdefSECRET")
    monkeypatch.delenv("SYSREV_API_URL", raising=False)
    get_settings.cache_clear()
    st.cache_resource.clear()
    engine = make_engine(url)
    init_db(engine)
    with make_session_factory(engine)() as s:
        _, rid, token = comparative_project(s)
        s.commit()
    yield Env(engine, rid, token)
    engine.dispose()
    get_settings.cache_clear()
    st.cache_resource.clear()


def widget(at: Any, kind: str, label: str, index: int = 0) -> Any:
    found = [w for w in getattr(at, kind) if w.label == label]
    assert len(found) > index, f"no {kind} {label!r}: {[w.label for w in getattr(at, kind)]}"
    return found[index]


def click(at: Any, label_part: str, index: int = 0) -> Any:
    buttons = [b for b in at.button if label_part in b.label]
    assert len(buttons) > index, f"no button {label_part!r}: {[b.label for b in at.button]}"
    buttons[index].click().run()
    assert not at.exception, at.exception
    return at


def errors(at: Any) -> list[str]:
    return [e.value for e in at.error]


def successes(at: Any) -> list[str]:
    return [e.value for e in at.success]


# --------------------------------------------------------------------------- Refresh Models
@respx.mock
def test_refresh_preview_and_stale_activation_is_refused(env: Env) -> None:
    respx.get(ANTHROPIC_MODELS_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "claude-opus-9-test",
                        "display_name": "Claude Opus 9",
                        "type": "model",
                        "created_at": "2026-10-20T00:00:00Z",
                        "lifecycle": "active",
                    }
                ],
                "has_more": False,
                "last_id": "claude-opus-9-test",
            },
        )
    )
    respx.get(OPENAI_MODELS_URL).mock(return_value=httpx.Response(503, text="maintenance"))
    at = env.app("Model catalogue (Refresh Models)")
    click(at, "Refresh models (no paid inference)")
    assert any("1 newly listed (unverified)" in m for m in successes(at))
    assert not any(r.method == "POST" for r in [c.request for c in respx.calls])  # no inference
    # another reviewer activates the same candidate after this preview was loaded
    from src.llm.catalogue import ModelCatalogue

    with env.other() as s:
        repo = Repository(s)
        snap = repo.load_catalogue_snapshots()[-1]
        base = ModelCatalogue.load(Path(get_settings().data_dir))
        repo.activate_catalogue_snapshot(
            snap.snapshot_id,
            base=base,
            expected_sha256=snap.sha256,
            expected_active_sha256=base.sha256,
            reviewer_id=env.rid,
        )
        s.commit()
        before = s.scalar(select(func.count()).select_from(ModelCatalogueActivationORM))
    click(at, "Activate this snapshot (authenticated)")  # from the screen shown before
    assert any("has changed" in e for e in errors(at)), errors(at)
    with env.other() as s:  # nothing activated from the stale preview
        assert s.scalar(select(func.count()).select_from(ModelCatalogueActivationORM)) == before


# --------------------------------------------------------------------------- study links
def test_study_link_from_an_old_screen_is_refused(env: Env) -> None:
    with env.other() as s:  # links are Stage 5 inputs: changeable while Stage 5 can re-run
        Repository(s).get_project(PID).phase = Phase.STAGE5_META_ANALYSIS.value
        s.commit()
    at = env.app("Corrections, links & double counting")
    widget(at, "multiselect", "Reports of one study").set_value(["t1", "t2"])
    widget(at, "text_input", "Study id (letters, digits, . _ : -)").input("S-one")
    widget(at, "text_area", "Why these reports describe one study (required)").input("same trial")
    with env.other() as s:  # another session links t3 meanwhile
        repo = Repository(s)
        link_reports_to_study(
            repo,
            PID,
            ["t3"],
            study_id="S-three",
            primary_record_id="t3",
            reviewer=repo.require_active_reviewer(env.rid),
            note="other tab",
            expected_state_sha256=links_state_sha256(repo, PID),
        )
        s.commit()
    click(at, "Link reports (authenticated)")
    assert any("has changed" in e for e in errors(at))
    with env.other() as s:
        assert Repository(s).study_ids(PID, {"t1", "t2"}) == {"t1": "t1", "t2": "t2"}
    # after reloading the links screen (tabs: corrections, links, double counting)
    click(at, "Reload the current state", index=1)
    widget(at, "multiselect", "Reports of one study").set_value(["t1", "t2"])
    widget(at, "text_input", "Study id (letters, digits, . _ : -)").input("S-one")
    widget(at, "text_area", "Why these reports describe one study (required)").input("same trial")
    click(at, "Link reports (authenticated)")
    assert any("Reports linked" in m for m in successes(at)), errors(at)
    with env.other() as s:
        assert Repository(s).study_ids(PID, {"t1", "t2"}) == {"t1": "S-one", "t2": "S-one"}


# --------------------------------------------------------------------------- analysis plan
def test_plan_draft_and_approval_from_the_displayed_draft_only(env: Env) -> None:
    with env.other() as s:
        Repository(s).get_project(PID).phase = Phase.STAGE5_META_ANALYSIS.value
        s.commit()
    at = env.app("Analysis plan")
    widget(at, "text_input", "Analysis id").input("or1")
    widget(at, "selectbox", "Effect measure").set_value("OR")
    click(at, "Add analysis to the composition")
    widget(at, "text_area", "Rationale for this version").input("odds ratio as sensitivity")
    click(at, "Save as a new draft version (authenticated)")
    assert any("Draft saved" in m for m in successes(at)), errors(at)
    with env.other() as s:  # another session saves a newer draft
        repo = Repository(s)
        state = plan_service.plan_state(repo, PID)
        plan_service.save_draft(
            repo,
            PID,
            [
                AnalysisSpec(
                    analysis_id="rd",
                    outcome_id="complication",
                    effect_measure="RR",
                    random_effects_ci="wald",
                )
            ],
            rationale="other tab",
            reviewer=repo.require_active_reviewer(env.rid),
            expected_latest_sha256=state["latest_sha256"],
            expected_protocol_sha256=plan_service.protocol_basis(repo, "mc1")[0],
        )
        s.commit()
    click(at, "Approve this draft (authenticated)")
    assert any("has changed" in e for e in errors(at))
    with env.other() as s:
        plans = plan_service.list_plans(Repository(s), PID)
        assert plans[-1].status == "draft" and plans[-1].analyses[0].analysis_id == "rd"


# --------------------------------------------------------------------------- amendments
def test_amendment_decision_is_bound_to_the_displayed_revision(env: Env) -> None:
    at = env.app("Protocol amendments")
    widget(at, "text_area", "Review title").input("Corrected synthetic title (amended)")
    widget(at, "text_area", "Rationale for the amendment (required)").input("typo")
    click(at, "Submit proposal")
    assert any("Amendment proposed" in m for m in successes(at)), errors(at)
    with env.other() as s:
        repo = Repository(s)
        assert repo.latest_protocol(PID).version == 1  # type: ignore[union-attr]
        pending = amendments.open_amendment(repo, PID)
        assert pending is not None
        # another session records a strategy validation: a new revision of the proposal
        sha = pending.protocol.search_strategies[0].sha256
        amendments.record_strategy_validation(
            repo,
            PID,
            pending.amendment_id,
            sha,
            method="human_attestation",
            reviewer=repo.require_active_reviewer(env.rid),
            expected_sha256=pending.sha256,
            note="re-checked",
        )
        s.commit()
    widget(at, "text_area", "Comments (required)").input("approve")
    click(at, "Record decision on the displayed revision")
    assert any("has changed" in e for e in errors(at))
    with env.other() as s:
        assert Repository(s).latest_protocol(PID).version == 1  # type: ignore[union-attr]
    click(at, "Reload the current state")
    widget(at, "text_area", "Comments (required)").input("approve")
    click(at, "Record decision on the displayed revision")
    assert any("Amendment approved" in m for m in successes(at)), errors(at)
    with env.other() as s:
        repo = Repository(s)
        versions = [(p.version, p.status) for p in repo.list_protocols(PID)]
        assert versions == [(1, "superseded"), (2, "approved")]
        assert editor.protocol_sha256(repo.latest_protocol(PID))  # type: ignore[arg-type]


def test_screen_keys_are_typed_and_disjoint_from_session7(env: Env) -> None:
    at = env.app("Corrections, links & double counting")
    store = at.session_state["operation_screens"]
    assert all(isinstance(k, tuple) and k[1] == PID for k in store)
    assert {k[0] for k in store} <= {"extraction_items", "study_links", "double_counting"}
    assert "reporting_screens" not in at.session_state or all(
        k not in store for k in at.session_state["reporting_screens"]
    )


# --------------------------------------------------------------------------- corrections
@pytest.fixture
def stage4(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Env]:
    import streamlit as st

    from src.core.enums import ActorType, ScreeningStage, ScreeningVerdict
    from src.extraction import accept_item, build_arms, submit_proposals
    from src.fulltext import confirm_pdf_match, register_upload
    from src.fulltext.storage import DocumentStore
    from src.pdf_extractor import parse_pdf
    from src.screening.llm_screening import adjudicate
    from tests.conftest import make_record
    from tests.fakes import approved_protocol, include_at_title_abstract
    from tests.fixtures.pdf_factory import SYNTHETIC_DOI, SYNTHETIC_TITLE, synthetic_article
    from tests.unit.test_master_table import km

    url = f"sqlite:///{tmp_path / 'ui8c.db'}"
    monkeypatch.setenv("SYSREV_DATABASE_URL", url)
    monkeypatch.setenv("SYSREV_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("SYSREV_API_URL", raising=False)
    get_settings.cache_clear()
    st.cache_resource.clear()
    engine = make_engine(url)
    init_db(engine)
    with make_session_factory(engine)() as s:
        repo = Repository(s)
        repo.create_project(PID, "Synthetic correction review")
        repo.save_protocol(PID, approved_protocol())
        rec = make_record("r1", SYNTHETIC_TITLE, doi=SYNTHETIC_DOI)
        repo.save_records(PID, [rec])
        include_at_title_abstract(repo, PID, [rec])
        rid, token = repo.enroll_reviewer("Dr Corrector")
        reviewer = repo.require_active_reviewer(rid)
        store = DocumentStore(tmp_path / "data" / "fulltext")
        item = register_upload(
            repo,
            store,
            PID,
            "r1",
            synthetic_article(),
            original_filename="r1.pdf",
            reviewer=reviewer,
        )
        if item.match_unconfirmed:
            confirm_pdf_match(repo, PID, "r1", reviewer=reviewer, note="checked")
        adjudicate(
            repo,
            PID,
            "r1",
            ScreeningStage.FULL_TEXT,
            verdict=ScreeningVerdict.INCLUDE,
            rationale="eligible",
            reviewer=reviewer,
        )
        stored = repo.get_fulltext_item(PID, "r1")
        assert stored is not None and stored.pdf_filename and stored.pdf_sha256
        doc = parse_pdf(
            store.read_verified(stored.pdf_filename, stored.pdf_sha256),
            filename=stored.pdf_filename,
        )
        (proposed,) = submit_proposals(
            repo,
            PID,
            "r1",
            doc,
            [km("reported_estimate", "62.1", 1, 1, 3)],
            proposed_by=ActorType.AGENT,
            proposer_identity="llm:test",
        )
        accept_item(repo, PID, proposed.item_id, reviewer=reviewer, note=None)
        build_arms(repo, PID, "r1", reviewer=reviewer)
        repo.get_project(PID).phase = Phase.STAGE4_EXTRACTION_ROB.value
        s.commit()
    yield Env(engine, rid, token)
    engine.dispose()
    get_settings.cache_clear()
    st.cache_resource.clear()


def _fill_correction(at: Any) -> None:
    widget(at, "text_input", "Number exactly as printed").input("81.4")
    widget(at, "text_input", "Location id (e.g. p2.t1.r1c2)").input("p2.t1.r1c2")
    widget(at, "text_area", "Reason for the correction (required)").input(
        "the 3-year value was read from the wrong column"
    )


def test_value_correction_with_evidence_invalidates_the_arm_and_refuses_stale_screens(
    stage4: Env,
) -> None:
    from src.extraction import correction_state_sha256, revise_reviewed_item
    from tests.unit.test_master_table import km

    at = stage4.app("Corrections, links & double counting")
    _fill_correction(at)
    click(at, "Submit correction (authenticated)")
    assert any("Invalidated arm(s)" in m for m in successes(at)), errors(at)
    with stage4.other() as s:  # another session corrects the new version again
        repo = Repository(s)
        current = next(i for i in repo.load_extraction_items(PID))
        from src.fulltext.storage import DocumentStore
        from src.pdf_extractor import parse_pdf

        item = repo.get_fulltext_item(PID, "r1")
        assert item is not None and item.pdf_filename and item.pdf_sha256
        store = DocumentStore(Path(get_settings().data_dir) / "fulltext")
        doc = parse_pdf(
            store.read_verified(item.pdf_filename, item.pdf_sha256), filename=item.pdf_filename
        )
        revise_reviewed_item(
            repo,
            PID,
            current.item_id,
            km("reported_estimate", "62.1", 1, 1, 3),
            doc,
            reviewer=repo.require_active_reviewer(stage4.rid),
            note="reverted after discussion",
            expected_state_sha256=correction_state_sha256(repo, PID, current),
        )
        s.commit()
        versions = len(repo.load_extraction_items(PID, include_superseded=True))
    _fill_correction(at)
    click(at, "Submit correction (authenticated)")
    assert any("has changed" in e for e in errors(at)), errors(at)
    with stage4.other() as s:
        assert len(Repository(s).load_extraction_items(PID, include_superseded=True)) == versions


# --------------------------------------------------------------------------- PR #13 finding 3
def test_plan_approval_from_a_screen_of_the_old_protocol_is_refused_until_reload(
    env: Env,
) -> None:
    with env.other() as s:  # Stage 4 with a displayed draft
        repo = Repository(s)
        repo.get_project(PID).phase = Phase.STAGE4_EXTRACTION_ROB.value
        state = plan_service.plan_state(repo, PID)
        plan_service.save_draft(
            repo,
            PID,
            [AnalysisSpec(analysis_id="or", outcome_id="complication", effect_measure="OR")],
            rationale="odds ratio",
            reviewer=repo.require_active_reviewer(env.rid),
            expected_latest_sha256=state["latest_sha256"],
            expected_protocol_sha256=state["protocol_sha256"],
        )
        s.commit()
    at = env.app("Analysis plan")
    assert any("protocol version 1" in c.value for c in at.caption)
    with env.other() as s:  # another reviewer narrows the outcome definition (same id)
        repo = Repository(s)
        reviewer = repo.require_active_reviewer(env.rid)
        base = repo.latest_protocol(PID)
        assert base is not None
        data = base.model_dump(mode="json")
        data["outcomes"][0]["name"] = "Serious complication requiring re-operation"
        p = amendments.propose(
            repo,
            PID,
            data,
            rationale="narrowed",
            reviewer=reviewer,
            expected_base_sha256=editor.protocol_sha256(base),
        )
        amendments.decide(
            repo,
            PID,
            p.amendment_id,
            "approved",
            reviewer=reviewer,
            comments="ok",
            expected_sha256=p.sha256,
        )
        s.commit()
        n_plans = len(plan_service.list_plans(repo, PID))

    def plan_obligation() -> str:
        with env.other() as s:
            return str(
                next(
                    o["state"]
                    for o in amendments.obligation_status(Repository(s), PID)
                    if o["area"] == "analysis_plans"
                )
            )

    click(at, "Approve this draft (authenticated)")
    assert any("has changed" in e for e in errors(at)), errors(at)
    with env.other() as s:
        assert len(plan_service.list_plans(Repository(s), PID)) == n_plans  # nothing stored
    assert plan_obligation() == "open"
    click(at, "Reload the current state")  # explicit reload: the amended definitions are shown
    assert any("protocol version 2" in c.value for c in at.caption)
    click(at, "Approve this draft (authenticated)")
    assert any("Plan approved" in m for m in successes(at)), errors(at)
    assert plan_obligation() == "resolved"
