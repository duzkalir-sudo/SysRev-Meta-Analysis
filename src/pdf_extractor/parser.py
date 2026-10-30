"""Deterministic PDF parsing with pdfplumber (text, tables) and pypdf (page labels).

pdfplumber (MIT, built on pdfminer.six) is used because it exposes per-line positions and
ruling-line table detection without native dependencies, and works unchanged on Windows.
No OCR is performed: a page without a text layer is flagged ``is_image_only`` and listed
for manual review, so no unverified text is ever generated. Tables are detected from
ruling lines; a table caption with no detectable structure is flagged, not guessed.
"""

from __future__ import annotations

import hashlib
import io
import re
import statistics
from pathlib import Path
from typing import Any

import pdfplumber
from pypdf import PdfReader
from pypdf.generic import DictionaryObject

from src.core.exceptions import SysRevError
from src.pdf_extractor.models import (
    BlockKind,
    ParsedDocument,
    ParsedPage,
    ParsedTable,
    TextBlock,
)

MIN_TEXT_CHARS = 25  # fewer characters than this → treated as no usable text layer
UNREADABLE_EMPTY_FRACTION = 0.5

_TABLE_CAPTION = re.compile(r"^(table|tab\.)\s*[0-9ivxlc]+[a-z]?\b", re.IGNORECASE)
_FIGURE_LEGEND = re.compile(r"^(fig\.|figure)\s*[0-9]+[a-z]?\b", re.IGNORECASE)
_SUPPLEMENT = re.compile(
    r"^(supplementa(ry|l)\b|appendix\b|e?table\s*s\d|efigure|online resource)", re.IGNORECASE
)
_PAGE_NUMBER = re.compile(r"^(page\s+)?\d{1,5}$", re.IGNORECASE)


class PDFParseError(SysRevError):
    """The file is not a readable PDF."""


class HashMismatchError(SysRevError):
    """The bytes do not have the SHA-256 recorded for them."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_pdf_file(path: Path, *, expected_sha256: str | None = None) -> ParsedDocument:
    return parse_pdf(path.read_bytes(), filename=path.name, expected_sha256=expected_sha256)


def parse_pdf(
    data: bytes,
    *,
    filename: str,
    expected_sha256: str | None = None,
    max_pages: int | None = None,
) -> ParsedDocument:
    """Parse ``data``; raise :class:`HashMismatchError` if it is not the expected file."""
    sha = sha256_bytes(data)
    if expected_sha256 is not None and sha != expected_sha256:
        raise HashMismatchError(
            f"{filename}: SHA-256 {sha} does not match the recorded {expected_sha256}"
        )
    doc_issues: list[str] = []
    labels, has_labels = _page_labels(data, doc_issues)
    try:
        pdf = pdfplumber.open(io.BytesIO(data))
    except Exception as exc:  # pdfminer raises many exception types for corrupt files
        raise PDFParseError(f"{filename}: not a readable PDF ({type(exc).__name__})") from exc
    pages: list[ParsedPage] = []
    with pdf:
        all_pages = pdf.pages if max_pages is None else pdf.pages[:max_pages]
        if not pdf.pages:
            raise PDFParseError(f"{filename}: PDF has no pages")
        for index, page in enumerate(all_pages):
            label = labels[index] if has_labels and index < len(labels) else None
            pages.append(_parse_page(page, index + 1, label))
    return ParsedDocument(
        pdf_filename=filename,
        pdf_sha256=sha,
        n_pages=len(pages),
        has_page_labels=has_labels,
        pages=tuple(pages),
        issues=tuple(doc_issues),
    )


def _page_labels(data: bytes, issues: list[str]) -> tuple[list[str], bool]:
    try:
        reader = PdfReader(io.BytesIO(data))
        root = reader.trailer["/Root"].get_object()
        if not isinstance(root, DictionaryObject) or "/PageLabels" not in root:
            return [], False
        return list(reader.page_labels), True
    except Exception as exc:  # labels are optional metadata; never fatal
        issues.append(f"page labels unreadable ({type(exc).__name__})")
        return [], False


def _parse_page(page: Any, number: int, label: str | None) -> ParsedPage:
    issues: list[str] = []
    try:
        chars = len(page.chars)
        images = len(page.images)
        lines: list[dict[str, Any]] = page.extract_text_lines(return_chars=False)
        found = page.find_tables()
    except Exception as exc:
        return ParsedPage(
            physical_page=number, page_label=label, text="", char_count=0, image_count=0,
            issues=(f"page could not be parsed ({type(exc).__name__}); manual review",),
        )  # fmt: skip

    image_only = chars < MIN_TEXT_CHARS and images > 0
    if image_only:
        issues.append("image-only page: no text layer (OCR not performed); manual review")
    elif chars < MIN_TEXT_CHARS:
        issues.append("no extractable text on this page; manual review")

    table_boxes = [tuple(float(v) for v in t.bbox) for t in found]
    body = [ln for ln in lines if not _inside_any(ln, table_boxes)]
    blocks, printed = _blocks(body, number)
    tables = [_table(t, number, i + 1, blocks) for i, t in enumerate(found)]
    for t in tables:
        issues.extend(f"{t.table_id}: {msg}" for msg in t.issues)
    captions = [b for b in blocks if b.kind == "table_caption"]
    if captions and not tables:
        issues.append(
            "table caption present but no ruled table structure detected; read the table manually"
        )
    text = "\n".join(str(ln["text"]) for ln in lines)
    return ParsedPage(
        physical_page=number,
        page_label=label,
        printed_page_text=printed,
        text=text,
        char_count=chars,
        image_count=images,
        is_image_only=image_only,
        blocks=tuple(blocks),
        tables=tuple(tables),
        issues=tuple(issues),
    )


def _inside_any(line: dict[str, Any], boxes: list[tuple[float, ...]]) -> bool:
    mid = (float(line["top"]) + float(line["bottom"])) / 2
    x_mid = (float(line["x0"]) + float(line["x1"])) / 2
    return any(b[0] <= x_mid <= b[2] and b[1] <= mid <= b[3] for b in boxes)


def _classify(text: str) -> BlockKind:
    if _TABLE_CAPTION.match(text):
        return "table_caption"
    if _FIGURE_LEGEND.match(text):
        return "figure_legend"
    if _SUPPLEMENT.match(text):
        return "supplementary"
    words = text.split()
    if len(words) <= 8 and len(text) < 70 and not text.endswith((".", ":", ";", ",")):
        return "heading" if text[:1].isupper() else "paragraph"
    return "paragraph"


def _blocks(lines: list[dict[str, Any]], page: int) -> tuple[list[TextBlock], str | None]:
    printed: str | None = None
    if lines:
        for candidate in (lines[0], lines[-1]):
            text = str(candidate["text"]).strip()
            if _PAGE_NUMBER.match(text):
                printed = text
                break
    heights = [float(ln["bottom"]) - float(ln["top"]) for ln in lines] or [10.0]
    gap_limit = 0.8 * statistics.median(heights)
    groups: list[list[str]] = []
    prev_bottom: float | None = None
    for ln in lines:
        text = str(ln["text"]).strip()
        if not text:
            continue
        if printed is not None and text == printed and ln in (lines[0], lines[-1]):
            groups.append([text])
            prev_bottom = None
            continue
        new_block = (
            prev_bottom is None
            or float(ln["top"]) - prev_bottom > gap_limit
            or _classify(text) in ("table_caption", "figure_legend", "supplementary")
        )
        if new_block:
            groups.append([text])
        else:
            groups[-1].append(text)
        prev_bottom = float(ln["bottom"])
    blocks: list[TextBlock] = []
    for i, group in enumerate(groups, start=1):
        text = "\n".join(group)
        kind: BlockKind = "page_number" if text == printed else _classify(group[0])
        blocks.append(TextBlock(block_id=f"p{page}.b{i}", page=page, kind=kind, text=text))
    return blocks, printed


def _table(found: Any, page: int, n: int, blocks: list[TextBlock]) -> ParsedTable:
    issues: list[str] = []
    try:
        raw: list[list[str | None]] = found.extract()
    except Exception as exc:
        raw = []
        issues.append(f"table cells could not be extracted ({type(exc).__name__})")
    rows = tuple(tuple(c.strip() if isinstance(c, str) else None for c in row) for row in raw)
    cells = [c for row in rows for c in row]
    empty = sum(1 for c in cells if not c)
    unreadable = not cells or empty / len(cells) > UNREADABLE_EMPTY_FRACTION
    if unreadable:
        issues.append("table unreadable (mostly empty cells); manual review")
    if len({len(r) for r in rows}) > 1:
        issues.append("ragged rows (merged or split cells); check alignment manually")
    bbox = tuple(float(v) for v in found.bbox)
    caption = next(
        (b.text for b in reversed(blocks) if b.kind == "table_caption"),
        None,
    )
    return ParsedTable(
        table_id=f"p{page}.t{n}",
        page=page,
        caption=caption,
        rows=rows,
        bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
        unreadable=unreadable,
        issues=tuple(issues),
    )
