"""Source-grounded PDF parsing (Stage 4).

``parser`` produces a page-indexed :class:`ParsedDocument` (text blocks, ruled tables,
page labels, image-only page flags) without OCR or any model. ``grounding`` verifies a
``SourceEvidence`` against that parsed document. ``context`` selects only the relevant
blocks/tables for an LLM prompt.
"""

from src.pdf_extractor.context import ContextSnippet, render_context, select_context
from src.pdf_extractor.grounding import match_key, number_in_text, verify_evidence
from src.pdf_extractor.models import (
    ParsedDocument,
    ParsedPage,
    ParsedTable,
    TextBlock,
    parse_location_id,
)
from src.pdf_extractor.parser import (
    HashMismatchError,
    PDFParseError,
    parse_pdf,
    parse_pdf_file,
    sha256_bytes,
)

__all__ = [
    "ContextSnippet",
    "HashMismatchError",
    "PDFParseError",
    "ParsedDocument",
    "ParsedPage",
    "ParsedTable",
    "TextBlock",
    "match_key",
    "number_in_text",
    "parse_location_id",
    "parse_pdf",
    "parse_pdf_file",
    "render_context",
    "select_context",
    "sha256_bytes",
    "verify_evidence",
]
