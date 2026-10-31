"""Session 7 (review of PR #12, round 3): the Streamlit UI must submit the state identifiers of
the screen the reviewer actually saw, never identifiers fetched again during the submission
rerun. Every flow is authenticated (a real reviewer token) and driven through the widgets;
the intervening change is made by *another* database session. All data SYNTHETIC."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.engine import Engine

from src.core.config import get_settings
from src.manuscript.service import ManuscriptService
from src.reporting.service import ReportingService
from src.state.db import init_db, make_engine, make_session_factory
from src.state.repository import Repository
from tests.manuscript_helpers import (
    comparative_project,
    complete_manuscript,
    human,
    save_grade,
    save_group,
)
from tests.unit.test_reporting_grade import _group, approval

pytest.importorskip("streamlit")
STALE = "has changed"


@dataclass
class Env:
    engine: Engine
    rid: str
    token: str
    tmp: Path

    def other_session(self) -> Any:
        """A separate database session (another reviewer, another browser tab)."""
        return make_session_factory(self.engine)()

    def app(self, page: str) -> Any:
        from streamlit.testing.v1 import AppTest

        path = str(Path(__file__).resolve().parents[2] / "src" / "ui" / "app.py")
        at = AppTest.from_file(path, default_timeout=60)
        at.session_state["project_id"] = "mc1"
        at.session_state["page"] = page
        at.session_state["reviewer_token"] = self.token
        at.run()
        assert not at.exception
        return at


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Env]:
    import streamlit as st

    url = f"sqlite:///{tmp_path / 'ui.db'}"
    monkeypatch.setenv("SYSREV_DATABASE_URL", url)
    monkeypatch.setenv("SYSREV_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("SYSREV_API_URL", raising=False)
    get_settings.cache_clear()
    st.cache_resource.clear()
    engine = make_engine(url)
    init_db(engine)
    with make_session_factory(engine)() as s:
        repo, rid, token = comparative_project(s)
        complete_manuscript(ManuscriptService(repo, "mc1"), human(repo, rid))
        s.commit()
    yield Env(engine, rid, token, tmp_path)
    engine.dispose()
    get_settings.cache_clear()
    st.cache_resource.clear()


def widget(at: Any, kind: str, label: str, index: int = 0) -> Any:
    found = [w for w in getattr(at, kind) if w.label == label]
    assert len(found) > index, f"no {kind} {label!r}"
    return found[index]


def click(at: Any, label_part: str, index: int = 0) -> Any:
    buttons = [b for b in at.button if label_part in b.label]
    assert len(buttons) > index, f"no button {label_part!r}"
    buttons[index].click().run()
    assert not at.exception
    return at


def errors(at: Any) -> list[str]:
    return [e.value for e in at.error]


# --------------------------------------------------------------------------- checklist


def _fill_checklist(at: Any, item: str) -> None:
    widget(at, "selectbox", "Item").set_value(item)
    widget(at, "selectbox", "Your decision").set_value("reported")
    widget(at, "multiselect", "Section(s) where it is reported").set_value(["methods"])
    widget(at, "text_input", "Precise location (e.g. 'Methods, Information sources')").input(
        "Methods, Information sources and search"
    )
    widget(at, "text_area", "Justification (required)").input("decided on the displayed screen")


def test_checklist_form_from_an_old_screen_is_refused(env: Env) -> None:
    at = env.app("PRISMA flow & checklists")  # the reviewer sees manuscript version 2
    assert any("manuscript version 2" in c.value for c in at.caption)
    with env.other_session() as s:  # another session edits the Methods: version 3
        repo = Repository(s)
        latest = repo.latest_manuscript_version("mc1")
        assert latest is not None and latest.section("methods") is not None
        ManuscriptService(repo, "mc1").edit(
            human(repo, env.rid),
            {"methods": latest.section("methods").source + " Edited elsewhere."},  # type: ignore[union-attr]
            "edit in another session",
        )
        s.commit()
        before = [d.decision_id for d in repo.load_checklist_decisions("mc1")]
    _fill_checklist(at, "6")
    click(at, "Record decision (authenticated)")
    assert any(STALE in e for e in errors(at)), errors(at)
    with env.other_session() as s:
        decisions = Repository(s).load_checklist_decisions("mc1")
        assert [d.decision_id for d in decisions] == before  # nothing stored
        assert not [d for d in decisions if d.manuscript_version == 3]
    # explicit review of the updated state, then the decision is accepted on version 3
    click(at, "Reload the current state")
    assert any("manuscript version 3" in c.value for c in at.caption)
    _fill_checklist(at, "6")
    click(at, "Record decision (authenticated)")
    assert not errors(at), errors(at)
    with env.other_session() as s:
        latest6 = [
            d
            for d in Repository(s).load_checklist_decisions("mc1")
            if d.item_id == "6" and d.checklist == "prisma_2020_main"
        ][-1]
        assert latest6.manuscript_version == 3


# --------------------------------------------------------------------------- synthesis


T1 = (
    "Complications occurred in 12 of 100 participants receiving X and in 20 of 100 "
    "receiving placebo."
)
T3 = "Complications occurred in 5 of 60 participants receiving X and in 9 of 58 receiving placebo."


def _fill_group(at: Any, arm_prefix: str, summary: str, gid: str = "g1", finding: str = T1) -> None:
    widget(at, "text_input", "Group id (letters, digits, -_.)").input(gid)
    widget(at, "text_input", "Title").input("Trials reporting complications by severity")
    widget(at, "multiselect", "Grouped by").set_value(["outcome"])
    widget(at, "text_area", "Description of the group (SWiM 1a)").input(
        "trials that reported complications only by severity grade"
    )
    widget(at, "text_area", "Rationale for the grouping (SWiM 1a)").input(
        "severity-graded counts are not compatible with the pre-specified outcome"
    )
    widget(at, "selectbox", "Why not meta-analysed").set_value("incompatible_measures")
    widget(at, "text_input", "Explanation").input("complications reported by severity only")
    widget(at, "text_input", "Standardised metric (SWiM 2)").input("direction of effect")
    widget(at, "text_area", "Justification of the method").input(
        "a single trial cannot be summarised quantitatively; it is tabulated"
    )
    widget(at, "text_input", "Certainty assessment of the synthesis (SWiM 6)").input(
        "not assessed with GRADE for this synthesis"
    )
    widget(at, "text_input", "Presentation and ordering of studies (SWiM 7)").input(
        "table ordered by risk of bias"
    )
    widget(at, "text_area", "Limitations of the synthesis (SWiM 9)").input(
        "only one trial; the direction of effect is uncertain"
    )
    arms = widget(at, "multiselect", "Approved arms (source)")
    arms.set_value([o for o in arms.options if o.startswith(arm_prefix)])
    widget(at, "text_area", "Finding (numbers only as extracted)").input(finding)
    widget(at, "selectbox", "Direction of effect").set_value("favours_intervention")
    widget(
        at, "text_input", "Basis for the direction (observed direction, not significance)"
    ).input("fewer complications observed in the X arm")
    widget(at, "text_area", "Narrative summary (SWiM 8)").input(summary)
    widget(at, "selectbox", "Record as").set_value("approved")


def test_synthesis_save_submits_the_frozen_preview(env: Env) -> None:
    at = env.app("Synthesis without MA & SWiM")
    _fill_group(at, "t1 ·", "The trial favoured the intervention (reviewer A).")
    click(at, "Preview the linked evidence (step 1 of 2)")  # the reviewer sees this preview
    assert not errors(at), errors(at)
    with env.other_session() as s:  # another reviewer saves g1 with a different summary
        repo = Repository(s)
        save_group(
            ReportingService(repo, "mc1"),
            _group(repo, env.rid, summary="The trial favoured the intervention (B)."),
            env.rid,
        )
        s.commit()
    click(at, "Save the previewed group (step 2 of 2")
    assert any(STALE in e for e in errors(at)), errors(at)
    with env.other_session() as s:
        groups = Repository(s).load_synthesis_groups("mc1")
        assert len(groups) == 1 and groups[0].summary.endswith("(B).")  # not overwritten


# --------------------------------------------------------------------------- GRADE


def _fill_grade(at: Any) -> None:
    widget(at, "selectbox", "Record as").set_value("approved")
    widget(at, "selectbox", "GRADE approach").set_value("interventions_randomised")
    widget(at, "text_area", "Why this approach fits the question").input(
        "all contributing studies are randomised trials"
    )
    for d in ("risk of bias", "inconsistency", "indirectness", "imprecision", "publication bias"):
        rationale = (
            "registers and grey literature searched (synthetic)"
            if d == "publication bias"
            else "synthetic rationale for this domain"
        )
        widget(at, "text_input", f"Rationale: {d}").input(rationale)
    widget(at, "selectbox", "Final certainty").set_value("high")
    widget(at, "text_area", "Overall explanation").input("no domain rated down (synthetic)")


def test_grade_form_and_withdrawal_from_an_old_screen_are_refused(env: Env) -> None:
    at = env.app("GRADE & Summary of Findings")  # the reviewer sees: no record yet
    with env.other_session() as s:
        repo = Repository(s)
        save_grade(ReportingService(repo, "mc1"), "comp", approval(), env.rid)  # record A
        s.commit()
    _fill_grade(at)
    click(at, "Save (authenticated)")
    assert any(STALE in e for e in errors(at)), errors(at)
    with env.other_session() as s:
        records = Repository(s).load_grade_assessments("mc1")
        assert len(records) == 1 and records[0].final_certainty == "moderate"  # A, untouched
    # withdrawal: the reviewer explicitly reloads and sees record A ...
    click(at, "Reload the current state", 0)
    with env.other_session() as s:  # ... then another session replaces it with record B
        repo = Repository(s)
        save_grade(
            ReportingService(repo, "mc1"),
            "comp",
            approval(
                final_certainty="low",
                departure_rationale="synthetic borderline concerns in two domains",
            ),
            env.rid,
        )
        s.commit()
    widget(at, "text_input", "Reason for withdrawing the current record").input(
        "withdrawing the record I reviewed (A)"
    )
    click(at, "Withdraw (authenticated)")
    assert any(STALE in e for e in errors(at)), errors(at)
    with env.other_session() as s:
        latest = Repository(s).load_grade_assessments("mc1")[-1]
        assert latest.status == "approved" and latest.final_certainty == "low"  # B stands


# --------------------------------------------------------------------------- SWiM


def test_swim_decision_from_an_old_screen_is_refused(env: Env) -> None:
    with env.other_session() as s:
        repo = Repository(s)
        save_group(ReportingService(repo, "mc1"), _group(repo, env.rid), env.rid)
        s.commit()
    at = env.app("Synthesis without MA & SWiM")  # the reviewer sees one approved synthesis
    with env.other_session() as s:  # another session approves a second synthesis
        repo = Repository(s)
        second = _group(repo, env.rid, group_id="g2", title="Second group (synthetic)")
        second["findings"][0] |= {
            "record_id": "t3",
            "arm_ids": [a.arm_id for a in repo.load_extracted_arms("mc1") if a.record_id == "t3"],
            "finding": "Complications occurred in 5 of 60 participants receiving X and in 9 of "
            "58 receiving placebo.",
        }
        second["summary"] = "The single trial in this group favoured the intervention."
        save_group(ReportingService(repo, "mc1"), second, env.rid)
        s.commit()
    widget(at, "selectbox", "SWiM applies to these syntheses").set_value("not_applicable")
    widget(at, "text_area", "Rationale").input("decided on the syntheses displayed (synthetic)")
    click(at, "Record (authenticated)")
    assert any(STALE in e for e in errors(at)), errors(at)
    with env.other_session() as s:
        assert Repository(s).load_swim_applicability("mc1") == []


def test_synthesis_saves_when_nothing_changed_and_withdrawal_uses_the_frozen_list(
    env: Env,
) -> None:
    at = env.app("Synthesis without MA & SWiM")
    _fill_group(at, "t1 ·", "The trial favoured the intervention (reviewer A).")
    click(at, "Preview the linked evidence (step 1 of 2)")
    click(at, "Save the previewed group (step 2 of 2")  # nothing changed in between
    assert not errors(at), errors(at)
    assert any("Synthesis group g1 saved" in m.value for m in at.success)
    with env.other_session() as s:
        groups = Repository(s).load_synthesis_groups("mc1")
        assert len(groups) == 1 and groups[0].summary.endswith("(reviewer A).")
    # the reviewer now sees revision 1 of g1; another session saves revision 2
    with env.other_session() as s:
        repo = Repository(s)
        save_group(
            ReportingService(repo, "mc1"),
            _group(repo, env.rid, summary="The trial favoured the intervention (B)."),
            env.rid,
        )
        s.commit()
    widget(at, "text_input", "Reason for withdrawing the group").input(
        "withdrawing the revision I reviewed"
    )
    click(at, "Withdraw group (authenticated)")
    assert any(STALE in e for e in errors(at)), errors(at)
    with env.other_session() as s:
        latest = Repository(s).load_synthesis_groups("mc1")[-1]
        assert latest.status == "approved" and latest.summary.endswith("(B).")


# --------------------------------------------------------------------------- key namespaces


def _groups_shown(at: Any) -> list[str]:
    return sorted(e.label.split(" · ")[0] for e in at.expander if " · revision " in e.label)


def _no_frozen_preview(at: Any) -> bool:
    return not [b for b in at.button if "Save the previewed group" in b.label]


@pytest.mark.parametrize("gid", ["active", "groups"])
def test_group_ids_never_collide_with_internal_screen_keys(env: Env, gid: str) -> None:
    """Valid group ids equal to internal control words ('active', 'groups') must behave like
    any other id: the preview, the active-preview pointer and the groups-list snapshot live in
    separate namespaces."""
    at = env.app("Synthesis without MA & SWiM")
    # 1. preview -> save, nothing changed in between
    _fill_group(at, "t1 ·", "The trial favoured the intervention (reviewer A).", gid=gid)
    click(at, "Preview the linked evidence (step 1 of 2)")
    assert not errors(at), errors(at)
    assert any(f"Preview of group {gid} (frozen at step 1)" in m.value for m in at.markdown)
    click(at, "Save the previewed group (step 2 of 2")
    assert not errors(at), errors(at)
    assert any(f"Synthesis group {gid} saved" in m.value for m in at.success)
    # 2. the groups list and the active-preview pointer are intact (also after another rerun)
    assert _groups_shown(at) == [gid] and _no_frozen_preview(at)
    at.run()
    assert not at.exception and _groups_shown(at) == [gid] and _no_frozen_preview(at)
    # 3. opening and saving another group still works
    _fill_group(
        at, "t3 ·", "The trial favoured the intervention (second group).", gid="g2", finding=T3
    )
    click(at, "Preview the linked evidence (step 1 of 2)")
    click(at, "Save the previewed group (step 2 of 2")
    assert not errors(at), errors(at)
    assert _groups_shown(at) == sorted([gid, "g2"]) and _no_frozen_preview(at)
    # 4. an intervening revision still makes the frozen preview of the same group refused
    _fill_group(at, "t1 ·", "The trial favoured the intervention (reviewer A, v2).", gid=gid)
    click(at, "Preview the linked evidence (step 1 of 2)")
    with env.other_session() as s:
        repo = Repository(s)
        save_group(
            ReportingService(repo, "mc1"),
            _group(repo, env.rid, group_id=gid, summary="The trial favoured the intervention (B)."),
            env.rid,
        )
        s.commit()
    click(at, "Save the previewed group (step 2 of 2")
    assert any(STALE in e for e in errors(at)), errors(at)
    at.run()  # the session is not corrupted
    assert not at.exception
    with env.other_session() as s:
        mine = [g for g in Repository(s).load_synthesis_groups("mc1") if g.group_id == gid]
        assert len(mine) == 2 and mine[-1].summary.endswith("(B).")  # not overwritten


@pytest.mark.parametrize(
    "gid", ["active", "groups", "synthesis_active", "synthesis_groups", "a:b", "mc1", ""]
)
def test_screen_keys_are_typed_and_disjoint(gid: str) -> None:
    """Any user-supplied id lands in a data position of a typed key: it can never equal an
    internal control key, and ':' inside an id cannot alias another key."""
    from src.ui import reporting_pages as rp

    controls = {
        rp._key(rp.SYNTHESIS_ACTIVE, "mc1"),
        rp._key(rp.SYNTHESIS_GROUPS, "mc1"),
        rp._key(rp.SWIM, "mc1"),
    }
    preview = rp._key(rp.SYNTHESIS_PREVIEW, "mc1", gid)
    assert preview not in controls
    assert preview[0] == rp.SYNTHESIS_PREVIEW and preview[-1] == gid
    assert rp._key(rp.GRADE, "p", "a:b") != rp._key(rp.GRADE, "p:a", "b")
    assert rp._widget_key("reload", rp._key(rp.GRADE, "p", "a:b")) != rp._widget_key(
        "reload", rp._key(rp.GRADE, "p:a", "b")
    )
