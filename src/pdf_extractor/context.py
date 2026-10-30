"""Select only the relevant parts of a parsed document for an LLM prompt.

Sending whole articles to a model is expensive and dilutes attention. This deterministic
keyword scorer picks the text blocks and tables most likely to hold the requested data
and labels each with its physical page and identifier, so proposals can cite them.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from src.pdf_extractor.models import ParsedDocument

DEFAULT_TERMS = (
    "patients", "participants", "enrolled", "included", "median", "mean", "follow-up",
    "table", "outcome", "control", "obliteration", "response", "relief", "complication",
    "adverse", "toxicity", "kaplan", "actuarial", "survival", "cumulative", "%",
)  # fmt: skip


@dataclass(frozen=True, slots=True)
class ContextSnippet:
    page: int
    location_id: str
    kind: str
    text: str
    score: int


def select_context(
    doc: ParsedDocument,
    terms: Iterable[str] = DEFAULT_TERMS,
    *,
    max_chars: int = 12000,
) -> list[ContextSnippet]:
    words = [t.casefold() for t in terms if t.strip()]
    candidates: list[ContextSnippet] = []
    for page in doc.pages:
        for block in page.blocks:
            if block.kind == "page_number":
                continue
            score = sum(block.text.casefold().count(w) for w in words)
            if block.kind in ("table_caption", "figure_legend"):
                score += 2
            candidates.append(
                ContextSnippet(page.physical_page, block.block_id, block.kind, block.text, score)
            )
        for table in page.tables:
            text = "\n".join(
                f"[{table.cell_id(r, 0).rsplit('c', 1)[0]}] " + " | ".join(c or "" for c in row)
                for r, row in enumerate(table.rows)
            )
            score = 5 + sum(text.casefold().count(w) for w in words)
            candidates.append(
                ContextSnippet(page.physical_page, table.table_id, "table", text, score)
            )
    chosen: list[ContextSnippet] = []
    used = 0
    for snip in sorted(candidates, key=lambda s: (-s.score, s.page, s.location_id)):
        if snip.score == 0 or used + len(snip.text) > max_chars:
            continue
        chosen.append(snip)
        used += len(snip.text)
    return sorted(chosen, key=lambda s: (s.page, s.location_id))


def render_context(snippets: Iterable[ContextSnippet]) -> str:
    return "\n\n".join(
        f"<<page {s.page} | {s.location_id} | {s.kind}>>\n{s.text}" for s in snippets
    )
