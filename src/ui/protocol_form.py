"""Pure helpers for the protocol editor page (no Streamlit import; unit-tested).

The page keeps a working copy of the protocol as plain JSON; these functions convert
between that copy and the text a reviewer types. Nothing here validates scientific
content: the server validates every draft with :class:`PICOProtocol`.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from src.core.enums import ReviewDesign

_UNITS = {"day": "days", "days": "days", "week": "weeks", "weeks": "weeks",
          "month": "months", "months": "months", "year": "years", "years": "years"}  # fmt: skip
_FIXED = re.compile(r"^(\d+(?:\.\d+)?)\s*([a-z]+)$")
_WINDOW = re.compile(r"^(\d+(?:\.\d+)?)\s*(?:-|–|to)\s*(\d+(?:\.\d+)?)\s*([a-z]+)$")
_LAST = {"last follow-up", "last follow up", "last followup", "last"}


def empty_protocol() -> dict[str, Any]:
    return {
        "review_title": "", "research_question": "", "population": "", "intervention": "",
        "review_design": ReviewDesign.COMPARATIVE.value, "comparator": None, "outcomes": [],
        "study_designs": [], "inclusion_criteria": [], "exclusion_criteria": [],
        "mesh_terms": [], "selected_databases": [], "search_strategies": [],
        "prospero_id": None,
    }  # fmt: skip


def lines(text: str) -> list[str]:
    return [line.strip() for line in (text or "").splitlines() if line.strip()]


def parse_timepoints(text: str) -> list[dict[str, Any]]:
    """``"5 years; 24-36 months; last follow-up"`` → FollowUpTimepoint dicts."""
    out: list[dict[str, Any]] = []
    for part in (p.strip().lower() for p in (text or "").split(";")):
        if not part:
            continue
        if part in _LAST:
            out.append({"kind": "last_follow_up"})
            continue
        if m := _WINDOW.match(part):
            unit = _UNITS.get(m.group(3))
            if unit:
                out.append({"kind": "window", "window_min": float(m.group(1)),
                            "window_max": float(m.group(2)), "unit": unit})  # fmt: skip
                continue
        if m := _FIXED.match(part):
            unit = _UNITS.get(m.group(2))
            if unit:
                out.append({"kind": "fixed", "value": float(m.group(1)), "unit": unit})
                continue
        raise ValueError(
            f"cannot read timepoint {part!r}: use e.g. '5 years', '24-36 months' or "
            "'last follow-up', separated by ';'"
        )
    return out


def format_timepoints(timepoints: list[dict[str, Any]]) -> str:
    parts = []
    for t in timepoints:
        if t["kind"] == "fixed":
            parts.append(f"{t['value']:g} {t['unit']}")
        elif t["kind"] == "window":
            parts.append(f"{t['window_min']:g}-{t['window_max']:g} {t['unit']}")
        else:
            parts.append("last follow-up")
    return "; ".join(parts)


def parse_mesh(text: str, previous: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """One descriptor per line; ``[noexp]`` = do not explode, ``[majr]`` = major topic only.

    Verification state is kept in the working copy for display only; the server keeps or
    resets it itself.
    """
    known = {m["descriptor"].casefold(): m for m in previous or []}
    out = []
    for line in lines(text):
        flags = {f.lower() for f in re.findall(r"\[(\w+)\]", line)}
        descriptor = re.sub(r"\s*\[\w+\]", "", line).strip()
        if not descriptor:
            continue
        item = {"descriptor": descriptor, "explode": "noexp" not in flags,
                "major_topic_only": "majr" in flags}  # fmt: skip
        old = known.get(descriptor.casefold())
        if old and old.get("verified"):
            item |= {"verified": True, "verified_at": old.get("verified_at")}
        out.append(item)
    return out


def format_mesh(terms: list[dict[str, Any]]) -> str:
    return "\n".join(
        m["descriptor"] + ("" if m.get("explode", True) else " [noexp]")
        + (" [majr]" if m.get("major_topic_only") else "")
        for m in terms
    )  # fmt: skip


def parse_date(text: str) -> str | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        raise ValueError(f"dates must be YYYY-MM-DD, got {text!r}") from None


def upsert_outcome(outcomes: list[dict[str, Any]], outcome: dict[str, Any]) -> list[dict[str, Any]]:
    """Add an outcome, or replace the one with the same ``outcome_id``."""
    rest = [o for o in outcomes if o["outcome_id"] != outcome["outcome_id"]]
    return [*rest, outcome]


def set_strategy(
    strategies: list[dict[str, Any]], database: str, strategy: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """The page edits one strategy per database; ``None`` removes it."""
    rest = [s for s in strategies if s["database"] != database]
    return rest if strategy is None else [*rest, strategy]
