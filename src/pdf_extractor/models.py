"""Parsed-document model: page-indexed text blocks and tables with stable identifiers.

Identifiers (all physical-page based, 1-indexed):

* text block ``p{page}.b{n}``  — paragraph, heading, figure legend, table caption, ...
* table      ``p{page}.t{n}``
* table cell ``p{page}.t{n}.r{row}c{col}`` (row 0 is the first extracted row)

``physical_page`` is the page's position in the PDF file. ``page_label`` is the logical
label from the PDF ``/PageLabels`` dictionary (if defined) and ``printed_page_text`` is a
number printed in the page header/footer (heuristic). Evidence always cites the physical
page; labels are recorded only so humans can cross-reference the printed journal page.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

BlockKind = Literal[
    "paragraph", "heading", "figure_legend", "table_caption", "supplementary", "page_number"
]

_ID_RE = re.compile(r"^p(?P<page>\d+)\.(?P<kind>[bt])(?P<n>\d+)(?:\.r(?P<r>\d+)c(?P<c>\d+))?$")


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TextBlock(_Frozen):
    block_id: str
    page: int = Field(ge=1)
    kind: BlockKind
    text: str


class ParsedTable(_Frozen):
    table_id: str
    page: int = Field(ge=1)
    caption: str | None = None
    rows: tuple[tuple[str | None, ...], ...]
    bbox: tuple[float, float, float, float]
    unreadable: bool = False
    issues: tuple[str, ...] = ()

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        return max((len(r) for r in self.rows), default=0)

    @property
    def header(self) -> tuple[str | None, ...]:
        return self.rows[0] if self.rows else ()

    def cell(self, row: int, col: int) -> str | None:
        if 0 <= row < len(self.rows) and 0 <= col < len(self.rows[row]):
            return self.rows[row][col]
        return None

    def cell_id(self, row: int, col: int) -> str:
        return f"{self.table_id}.r{row}c{col}"

    def as_text(self) -> str:
        return "\n".join(" ".join(c or "" for c in row) for row in self.rows)


class ParsedPage(_Frozen):
    physical_page: int = Field(ge=1)
    page_label: str | None = None
    printed_page_text: str | None = None
    text: str
    char_count: int = Field(ge=0)
    image_count: int = Field(ge=0)
    is_image_only: bool = False
    blocks: tuple[TextBlock, ...] = ()
    tables: tuple[ParsedTable, ...] = ()
    issues: tuple[str, ...] = ()

    @property
    def needs_manual_review(self) -> bool:
        return bool(self.issues) or self.is_image_only


class ParsedDocument(_Frozen):
    pdf_filename: str
    pdf_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    n_pages: int = Field(ge=0)
    has_page_labels: bool = False
    pages: tuple[ParsedPage, ...]
    parser: str = "pdfplumber"
    issues: tuple[str, ...] = ()

    def page(self, physical_page: int) -> ParsedPage | None:
        if 1 <= physical_page <= len(self.pages):
            return self.pages[physical_page - 1]
        return None

    @property
    def pages_needing_review(self) -> list[int]:
        return [p.physical_page for p in self.pages if p.needs_manual_review]

    def find_block(self, block_id: str) -> TextBlock | None:
        parsed = parse_location_id(block_id)
        if parsed is None or parsed[1] != "b":
            return None
        page = self.page(parsed[0])
        return next((b for b in page.blocks if b.block_id == block_id), None) if page else None

    def find_table(self, table_id: str) -> ParsedTable | None:
        parsed = parse_location_id(table_id)
        if parsed is None or parsed[1] != "t":
            return None
        page = self.page(parsed[0])
        base = table_id.split(".r")[0]
        return next((t for t in page.tables if t.table_id == base), None) if page else None

    @property
    def full_text(self) -> str:
        return "\n".join(p.text for p in self.pages)


def parse_location_id(identifier: str) -> tuple[int, str, int, int | None, int | None] | None:
    """``"p4.t1.r2c3"`` → ``(4, "t", 1, 2, 3)``; ``None`` for free-text labels."""
    m = _ID_RE.match(identifier.strip())
    if not m:
        return None
    r, c = m.group("r"), m.group("c")
    return (
        int(m.group("page")),
        m.group("kind"),
        int(m.group("n")),
        int(r) if r is not None else None,
        int(c) if c is not None else None,
    )
