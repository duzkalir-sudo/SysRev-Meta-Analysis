"""Deterministic rendering of manuscript source into journal text.

Markers are replaced by approved facts and citations are numbered in order of first
appearance (Vancouver). Rendering records, for every character span, whether it came from a
fact binding or a citation, so the Auditor can tell numbers inserted from the evidence
package from numbers typed by a person or a model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.core.schemas.manuscript import (
    SECTION_ORDER,
    ManuscriptSection,
    NumericBinding,
    ReferenceEntry,
)
from src.manuscript.evidence import EvidencePackage

FACT_RE = re.compile(r"\{\{\s*([A-Za-z][A-Za-z0-9_.:\-]*)\s*\}\}")
CITE_RE = re.compile(r"\[(@[A-Za-z0-9_.:\-]+(?:\s*;\s*@[A-Za-z0-9_.:\-]+)*)\]")
AUTHOR_INPUT_RE = re.compile(r"\[\[\s*AUTHOR INPUT\s*:([^\]]*)\]\]", re.IGNORECASE)
MARKER_RE = re.compile(f"{FACT_RE.pattern}|{CITE_RE.pattern}")
MAX_NESTING = 3


@dataclass(frozen=True, slots=True)
class Span:
    start: int
    end: int
    kind: str  # "fact" | "citation"
    key: str  # fact key, or ";"-joined ref keys
    binding_index: int | None = None


@dataclass
class RenderedSection:
    section_id: str
    heading: str
    text: str
    spans: list[Span] = field(default_factory=list)
    unknown_facts: list[str] = field(default_factory=list)
    unknown_refs: list[str] = field(default_factory=list)


@dataclass
class Rendered:
    sections: list[RenderedSection]
    bindings: list[NumericBinding]
    citation_order: list[str]  # ref keys in numbering order
    bibliography: list[tuple[int, str, str]]  # (number, ref_key, Vancouver text)
    markdown: str
    templates_used: list[tuple[str, str]] = field(default_factory=list)  # (section, key)
    drift: list[str] = field(default_factory=list)  # frozen bindings that no longer align

    def number_of(self, ref_key: str) -> int | None:
        try:
            return self.citation_order.index(ref_key) + 1
        except ValueError:
            return None


def vancouver(entry: ReferenceEntry) -> str:
    """NLM/Vancouver style from stored fields only; absent fields are omitted, never guessed."""
    authors = entry.authors[:6]
    names = ", ".join(authors) + (", et al" if len(entry.authors) > 6 else "")
    parts: list[str] = []
    if names:
        parts.append(names + ".")
    parts.append(entry.title.rstrip(".") + ".")
    tail = entry.journal.rstrip(".") + "." if entry.journal else ""
    if entry.year:
        tail += f" {entry.year}"
        if entry.volume:
            tail += f";{entry.volume}"
            if entry.issue:
                tail += f"({entry.issue})"
        if entry.pages:
            tail += f":{entry.pages}"
        tail += "."
    if tail.strip():
        parts.append(tail.strip())
    if entry.doi:
        parts.append(f"doi:{entry.doi}")
    if entry.pmid:
        parts.append(f"PMID: {entry.pmid}")
    return " ".join(parts)


def _numbers_label(nums: list[int]) -> str:
    """[1], [1,2], [1–3], [1,3–5]."""
    nums = sorted(set(nums))
    out: list[str] = []
    i = 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(f"{nums[i]}–{nums[j]}" if j - i >= 2 else ",".join(map(str, nums[i : j + 1])))
        i = j + 1
    return "[" + ",".join(out) + "]"


def expand(
    source: str, package: EvidencePackage, depth: int = 0, used: list[str] | None = None
) -> str:
    """Expand table/figure templates (which themselves contain markers)."""
    if depth >= MAX_NESTING:
        return source

    def repl(m: re.Match[str]) -> str:
        key = m.group(1)
        tpl = package.templates.get(key)
        if tpl is None:
            return m.group(0)
        if used is not None:
            used.append(key)
        return expand(tpl, package, depth + 1, used)

    return FACT_RE.sub(repl, source)


def render(
    sections: list[ManuscriptSection],
    package: EvidencePackage,
    frozen: list[NumericBinding] | None = None,
) -> Rendered:
    """Render ``sections``. With ``frozen`` (a stored version's bindings), each marker shows
    the text that was frozen into the version, so the result reproduces what the stored
    manuscript says; markers that no longer align with a frozen binding are listed in
    ``drift``."""
    ordered = sorted(sections, key=lambda s: SECTION_ORDER.index(s.section_id))
    order: list[str] = []
    out: list[RenderedSection] = []
    bindings: list[NumericBinding] = []
    templates_used: list[tuple[str, str]] = []
    drift: list[str] = []
    frozen_at = {(b.section_id, b.index): b for b in frozen or []}
    seen: set[tuple[str, int]] = set()
    for sec in ordered:
        used: list[str] = []
        src = expand(sec.source, package, used=used)
        templates_used += [(sec.section_id, k) for k in used]
        rs = RenderedSection(sec.section_id, sec.heading, "")
        text_parts: list[str] = []
        pos = 0
        last = 0
        marker_index = 0
        for m in MARKER_RE.finditer(src):
            literal = src[last : m.start()]
            text_parts.append(literal)
            pos += len(literal)
            if m.group(1) is not None:  # fact marker
                key = m.group(1)
                fact = package.facts.get(key)
                if fact is None:
                    rs.unknown_facts.append(key)
                    rendered = m.group(0)  # left visible; the audit reports it
                    rs.spans.append(Span(pos, pos + len(rendered), "unknown", key))
                else:
                    rendered, sha = fact.text, fact.sha256()
                    if frozen is not None:
                        old = frozen_at.get((sec.section_id, marker_index))
                        seen.add((sec.section_id, marker_index))
                        if old is not None and old.key == key:
                            rendered, sha = old.rendered_text, old.fact_sha256
                        else:
                            drift.append(
                                f"{sec.section_id} marker {marker_index}: {key} has no "
                                "matching frozen binding"
                            )
                    bindings.append(
                        NumericBinding(
                            section_id=sec.section_id,
                            index=marker_index,
                            key=key,
                            rendered_text=rendered,
                            fact_sha256=sha,
                        )
                    )
                    rs.spans.append(Span(pos, pos + len(rendered), "fact", key, len(bindings) - 1))
                marker_index += 1
            else:  # citation marker
                keys = [k.strip().lstrip("@") for k in m.group(2).split(";")]
                nums: list[int] = []
                for k in keys:
                    if k not in package.references:
                        rs.unknown_refs.append(k)
                        continue
                    if k not in order:
                        order.append(k)
                    nums.append(order.index(k) + 1)
                rendered = _numbers_label(nums) if nums else m.group(0)
                rs.spans.append(Span(pos, pos + len(rendered), "citation", ";".join(keys)))
            text_parts.append(rendered)
            pos += len(rendered)
            last = m.end()
        text_parts.append(src[last:])
        rs.text = "".join(text_parts)
        out.append(rs)
    if frozen is not None:
        drift += [
            f"{sid} marker {i}: frozen binding {b.key} no longer has a marker"
            for (sid, i), b in sorted(frozen_at.items())
            if (sid, i) not in seen
        ]
    bib = [(i + 1, k, vancouver(package.references[k].entry)) for i, k in enumerate(order)]
    return Rendered(out, bindings, order, bib, to_markdown(out, bib), templates_used, drift)


def to_markdown(sections: list[RenderedSection], bib: list[tuple[int, str, str]]) -> str:
    lines: list[str] = []
    for s in sections:
        if s.section_id == "title":
            lines += [f"# {s.text.strip()}", ""]
            continue
        lines += [f"## {s.heading}", "", s.text.strip(), ""]
    lines += ["## References", ""]
    lines += [f"{n}. {text}" for n, _, text in bib] or ["(no references cited)"]
    return "\n".join(lines).rstrip()
