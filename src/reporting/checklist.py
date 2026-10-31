"""Reporting checklists (PRISMA 2020 main and abstract, SWiM): suggestions, bases, status.

* **Suggestions** are deterministic: an item has a *candidate location* in a section when the
  section binds an evidence key the item is about (e.g. ``{{search.statement}}`` for item 6)
  or, as a weaker signal, contains one of the item's cue phrases. A candidate yields the
  status ``requires_human_review``; no candidate yields ``missing``. The system never records
  ``reported``: keyword or key presence is not methodological compliance.
* **Decisions** are made by authenticated reviewers. Each is bound to a *basis hash*: the
  source and frozen values of the sections it locates (for reported/partially reported
  items) and the evidence-fingerprint components the item depends on. When the manuscript
  sections or that evidence change, the decision is *stale* and must be re-confirmed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.core.schemas.manuscript import ManuscriptVersion
from src.core.schemas.reporting import ChecklistDecision
from src.core.schemas.stats import canonical_sha256
from src.reporting.standards import CATALOGUES, ChecklistItem

FACT_KEY_RE = re.compile(r"\{\{\s*([A-Za-z][A-Za-z0-9_.:\-]*)\s*\}\}")


def section_basis(version: ManuscriptVersion, section_id: str) -> str:
    sec = version.section(section_id)
    return canonical_sha256(
        {
            "source": sec.source if sec else None,
            "bindings": [
                [b.index, b.key, b.rendered_text]
                for b in version.bindings
                if b.section_id == section_id
            ],
        }
    )


def decision_basis(
    item: ChecklistItem,
    version: ManuscriptVersion,
    section_ids: list[str],
    fingerprint: dict[str, str],
    status: str,
) -> str:
    sections = section_ids if status in ("reported", "partially_reported") else []
    return canonical_sha256(
        {
            "checklist": item.checklist,
            "item": item.item_id,
            "status": status,
            "sections": {s: section_basis(version, s) for s in sorted(sections)},
            "evidence": {k: fingerprint.get(k) for k in sorted(item.evidence)},
        }
    )


@dataclass
class ItemState:
    item: ChecklistItem
    suggested_status: str  # "requires_human_review" | "missing"
    candidates: list[str] = field(default_factory=list)
    decision: ChecklistDecision | None = None
    stale: bool = False
    stale_reason: str | None = None

    @property
    def status(self) -> str:
        """The item's current status: the human decision if current, else the suggestion."""
        if self.decision is not None and not self.stale:
            return self.decision.status
        return self.suggested_status

    @property
    def decided(self) -> bool:
        return self.decision is not None and not self.stale


def suggest(item: ChecklistItem, version: ManuscriptVersion) -> list[str]:
    """Candidate locations ('section: reason'), deterministically."""
    out: list[str] = []
    for sid in item.sections:
        sec = version.section(sid)
        if sec is None or not sec.source.strip():
            continue
        keys = {b.key for b in version.bindings if b.section_id == sid}
        keys |= set(FACT_KEY_RE.findall(sec.source))
        hits = sorted(
            k
            for k in keys
            if any(k == c or (c.endswith((".", ":")) and k.startswith(c)) for c in item.cue_keys)
        )
        if hits:
            out.append(f"{sid}: binds {', '.join(hits[:4])}{' …' if len(hits) > 4 else ''}")
            continue
        low = sec.source.lower()
        terms = [t for t in item.cue_terms if t in low]
        if terms:
            out.append(f"{sid}: mentions '{terms[0]}' (wording only — not evidence of compliance)")
    return out


def item_states(
    checklist: str,
    version: ManuscriptVersion,
    fingerprint: dict[str, str],
    decisions: list[ChecklistDecision],
) -> list[ItemState]:
    latest: dict[str, ChecklistDecision] = {}
    for d in decisions:  # chronological
        if d.checklist == checklist:
            latest[d.item_id] = d
    states: list[ItemState] = []
    for it in CATALOGUES[checklist]:
        cands = suggest(it, version)
        st = ItemState(it, "requires_human_review" if cands else "missing", cands)
        dec = latest.get(it.item_id)
        if dec is not None:
            st.decision = dec
            current = decision_basis(it, version, list(dec.section_ids), fingerprint, dec.status)
            if current != dec.basis_sha256:
                st.stale = True
                changed = [s for s in d.section_ids if version.section(s) is None]
                st.stale_reason = f"decided on manuscript version {d.manuscript_version}; " + (
                    f"section(s) {changed} no longer exist"
                    if changed
                    else "the located section text or the relevant evidence changed since"
                )
        states.append(st)
    return states


def summary(states: list[ItemState]) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in states:
        key = s.status if s.decided else f"undecided ({s.suggested_status})"
        if s.stale:
            key = "stale (re-confirm)"
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def complete(states: list[ItemState]) -> bool:
    """Every item has a current human decision, and none is 'missing'. This is a statement
    about the checklist record only — never a claim that the review is PRISMA-compliant."""
    return all(s.decided and s.status != "missing" for s in states)
