"""Frozen decision screens for the Session 8 pages (same rules as Session 7's reporting pages).

Streamlit reruns the script when a form is submitted, so anything fetched during that rerun is
state the reviewer has *not* seen. Each Session 8 decision screen (the model catalogue, a
snapshot preview, an extraction item, the study links, the double-counting conflicts, the
analysis plans, a protocol amendment) is therefore kept in its own session-state namespace,
``operation_screens`` — disjoint from Session 7's ``reporting_screens`` — under a *typed* key
``(kind, project, *scope)``. Kinds are internal literals in the first position; user-supplied
identifiers (an item id, a snapshot id) only fill data positions, so they can never collide
with another screen. A decision sends the identifiers of the snapshot the reviewer saw; the API
refuses them (409, "has changed") if the stored state moved on. A snapshot is replaced only by
an explicit reload or after the reviewer's own successful decision (followed by a rerun).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import streamlit as st

from src.ui.services import ApiResult

STORE = "operation_screens"
FLASH = "operation_flash"
ScreenKey = tuple[str, ...]
Show = Callable[..., bool]

# kinds (internal literals; never user input)
CATALOGUE = "catalogue"
SNAPSHOT = "snapshot_preview"
SNAPSHOT_ACTIVE = "snapshot_selected"
ITEMS = "extraction_items"
ARMS = "report_arms"
LINKS = "study_links"
OVERLAP = "double_counting"
PLANS = "analysis_plans"
PLAN_WORK = "analysis_plan_work"
AMENDMENT = "amendment"
AMEND_WORK = "amendment_work"
MANIFEST = "fulltext_manifest"  # Session 9 review (P1): PDF identity and replacement screen

RELOAD = "Reload the current state (review it before deciding)"
CHANGED = (
    "The stored state changed after this screen was loaded. A decision from this screen will "
    "be refused: reload, review the updated state, then decide."
)


def key(kind: str, pid: str, *scope: str) -> ScreenKey:
    return (kind, pid, *scope)


def store() -> dict[ScreenKey, Any]:
    data: dict[ScreenKey, Any] = st.session_state.setdefault(STORE, {})
    return data


def wkey(prefix: str, k: ScreenKey) -> str:
    """A Streamlit widget key derived from a screen key (JSON keeps the parts unambiguous)."""
    return f"s8:{prefix}:{json.dumps(k)}"


def frozen(k: ScreenKey, fetch: Callable[[], ApiResult]) -> tuple[Any, str | None]:
    s = store()
    if k not in s:
        res = fetch()
        if not res.ok:
            return None, res.error
        s[k] = res.data
    return s[k], None


def put(k: ScreenKey, value: Any) -> None:
    store()[k] = value


def drop(k: ScreenKey) -> None:
    store().pop(k, None)


def reload_button(k: ScreenKey, *also: ScreenKey) -> None:
    if st.button(RELOAD, key=wkey("reload", k)):
        for x in (k, *also):
            drop(x)
        st.rerun()


def warn_if_changed(ids_now: Any, ids_shown: Any) -> None:
    if ids_now != ids_shown:
        st.warning(CHANGED)


def flash() -> None:
    message = st.session_state.pop(FLASH, None)
    if message:
        st.success(message)


def submit(result: ApiResult, show: Show, keys: list[ScreenKey], message: str) -> bool:
    """Accepted: drop the snapshots and rerun (the next screen shows the new state).
    Refused because the state changed: say so; nothing was saved."""
    if result.ok:
        for k in keys:
            drop(k)
        st.session_state[FLASH] = message
        st.rerun()
    if result.status == 409 and "has changed" in (result.error or ""):
        st.error(f"Refused ({result.status}): {result.error}")
        st.warning("Nothing was saved. Use the reload button to review the updated state.")
        return False
    show(result)
    return False


def upload_headers(filename: str, displayed_sha256: str | None) -> dict[str, str]:
    """Headers of a PDF upload from the frozen manifest screen (Session 9 review, P1): a
    replacement names the PDF the reviewer saw as linked, so it is refused if another PDF
    has been linked since."""
    headers = {"X-Filename": filename}
    if displayed_sha256:
        headers["X-Replaces-SHA256"] = displayed_sha256
    return headers
