"""Verify SourceEvidence against the actual parsed document — not against the claim itself.

Matching tolerates PDF-extraction artefacts that do not change meaning (Unicode
compatibility forms and ligatures, dash and quote variants, soft hyphens, words
hyphenated across a line break, whitespace next to punctuation, letter case) but never
alters digits: whitespace *between two digits* is preserved, so ``"12 34"`` (two table
cells) can never match a claimed ``"1234"``, and numbers must match as whole tokens, so
``"12"`` is not found inside ``"125"`` or ``"12.5"``.
"""

from __future__ import annotations

import re
import unicodedata

from src.core.enums import EvidenceCheckStatus, EvidenceLocationType
from src.core.schemas.extraction import SourceEvidence
from src.core.schemas.master import EvidenceCheck
from src.pdf_extractor.models import ParsedDocument, ParsedPage, parse_location_id

_DASHES = dict.fromkeys(map(ord, "‐‑‒–—−﹣－"), "-")
_QUOTES = {ord("‘"): "'", ord("’"): "'", ord("“"): '"', ord("”"): '"'}
_HYPHEN_BREAK = re.compile(r"([A-Za-z])-[ \t]*\n[ \t]*([a-z])")
_MIDDOT_DECIMAL = re.compile(r"(?<=\d)·(?=\d)")
_SPACE_NOT_BETWEEN_DIGITS = re.compile(r"(?<!\d)\s+|\s+(?!\d)")
_WS = re.compile(r"\s+")


def match_key(text: str) -> str:
    """Canonical form for robust, digit-preserving substring matching."""
    t = unicodedata.normalize("NFKC", text).replace("­", "")
    t = _HYPHEN_BREAK.sub(r"\1\2", t)
    t = t.translate(_DASHES).translate(_QUOTES)
    t = _MIDDOT_DECIMAL.sub(".", t)
    t = _WS.sub(" ", t).strip()
    t = _SPACE_NOT_BETWEEN_DIGITS.sub("", t)
    return t.casefold()


def number_in_text(reported_text: str, text: str) -> bool:
    """True when ``reported_text`` occurs in ``text`` as a whole numeric token."""
    needle = match_key(reported_text)
    if not needle:
        return False
    hay = match_key(text)
    pattern = rf"(?<![\d.]){re.escape(needle)}(?!\d|\.\d)"
    return re.search(pattern, hay) is not None


def quote_on_page(quote: str, page: ParsedPage) -> bool:
    key = match_key(quote)
    if not key:
        return False
    if key in match_key(page.text):
        return True
    return any(key in match_key(t.as_text()) for t in page.tables)


def verify_evidence(
    evidence: SourceEvidence,
    doc: ParsedDocument,
    *,
    reported_text: str | None = None,
) -> EvidenceCheck:
    """Check file identity, page, quote and (optionally) the number against ``doc``."""

    def result(
        status: EvidenceCheckStatus, detail: str, pages: tuple[int, ...] = ()
    ) -> EvidenceCheck:
        return EvidenceCheck(
            status=status, detail=detail, matched_pages=pages,
            checked_against_sha256=doc.pdf_sha256,
        )  # fmt: skip

    if evidence.pdf_sha256 is None:
        return result(EvidenceCheckStatus.HASH_MISMATCH, "evidence does not record a SHA-256")
    if evidence.pdf_sha256 != doc.pdf_sha256:
        return result(
            EvidenceCheckStatus.HASH_MISMATCH,
            f"evidence cites SHA-256 {evidence.pdf_sha256[:12]}…, document is "
            f"{doc.pdf_sha256[:12]}…",
        )
    if evidence.pdf_filename != doc.pdf_filename:
        return result(
            EvidenceCheckStatus.FILE_MISMATCH,
            f"evidence cites {evidence.pdf_filename!r}, document is {doc.pdf_filename!r}",
        )
    page = doc.page(evidence.page_number)
    if page is None:
        return result(
            EvidenceCheckStatus.PAGE_OUT_OF_RANGE,
            f"page {evidence.page_number} does not exist ({doc.n_pages} pages)",
        )
    if not quote_on_page(evidence.verbatim_quote, page):
        if page.is_image_only or page.char_count == 0:
            return result(
                EvidenceCheckStatus.PAGE_UNREADABLE,
                f"page {page.physical_page} has no text layer; a human must verify",
            )
        others = tuple(
            p.physical_page
            for p in doc.pages
            if p.physical_page != page.physical_page and quote_on_page(evidence.verbatim_quote, p)
        )
        if others:
            return result(
                EvidenceCheckStatus.WRONG_PAGE,
                f"quote is not on page {page.physical_page}; found on page(s) {list(others)}",
                others,
            )
        return result(
            EvidenceCheckStatus.QUOTE_NOT_FOUND,
            "quote does not occur anywhere in the parsed document (possible fabrication or "
            "paraphrase)",
        )
    if reported_text is not None and not number_in_text(reported_text, evidence.verbatim_quote):
        return result(
            EvidenceCheckStatus.NUMBER_NOT_IN_QUOTE,
            f"{reported_text!r} is not a whole number token in the quote",
        )
    location_problem = _location_problem(evidence, doc, reported_text)
    if location_problem:
        return result(EvidenceCheckStatus.LOCATION_MISMATCH, location_problem)
    note = _label_note(evidence, page)
    return result(EvidenceCheckStatus.VERIFIED, note, (page.physical_page,))


def _location_problem(
    evidence: SourceEvidence, doc: ParsedDocument, reported_text: str | None
) -> str | None:
    parsed = parse_location_id(evidence.location_identifier)
    if parsed is None:
        return None
    page_no, kind, _, row, col = parsed
    if page_no != evidence.page_number:
        return f"location {evidence.location_identifier} is not on page {evidence.page_number}"
    key = match_key(evidence.verbatim_quote)
    if kind == "b":
        block = doc.find_block(evidence.location_identifier)
        if block is None:
            return f"block {evidence.location_identifier} does not exist"
        return None if key in match_key(block.text) else "quote is not in the cited block"
    if evidence.location_type not in (EvidenceLocationType.TABLE, EvidenceLocationType.SUPPLEMENT):
        return "a table identifier was cited for non-table evidence"
    table = doc.find_table(evidence.location_identifier)
    if table is None:
        return f"table {evidence.location_identifier} does not exist"
    if key not in match_key(table.as_text()):
        return "quote is not in the cited table"
    if row is not None and col is not None:
        cell = table.cell(row, col)
        if cell is None:
            return f"cell {evidence.location_identifier} does not exist"
        if reported_text is not None and not number_in_text(reported_text, cell):
            return f"cell {evidence.location_identifier} reads {cell!r}, not {reported_text!r}"
    return None


def _label_note(evidence: SourceEvidence, page: ParsedPage) -> str:
    label = evidence.location_identifier
    if parse_location_id(label) is not None:
        return f"verified at {label}"
    if match_key(label) and match_key(label) not in match_key(page.text):
        return f"quote verified on page {page.physical_page}; label {label!r} not seen on page"
    return f"quote verified on page {page.physical_page}"
