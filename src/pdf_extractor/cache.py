"""Reuse of parsed PDFs within one process (issue #4: parse each document once).

A :class:`ParsedDocument` is a deterministic function of the PDF bytes, the filename it is
cited under, and the parser (this module's rules plus the pdfplumber/pypdf versions). The
cache key contains exactly these, so a replaced PDF (different SHA-256), a different
filename or an upgraded parser can never be served a stale parse. The bytes are still read
and hashed on every request — only the expensive layout analysis is reused — so a tampered
file on disk is detected as before. The cache is in memory, bounded (least recently used
entries are evicted) and never persisted; parsed documents are immutable (frozen models).
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from importlib import metadata

from src.pdf_extractor.models import ParsedDocument
from src.pdf_extractor.parser import parse_pdf, sha256_bytes

PARSER_RULES_VERSION = "1"  # bump when src.pdf_extractor.parser changes its output


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:  # pragma: no cover - both are dependencies
        return "unknown"


PARSER_ID = (
    f"sysrev-parser-{PARSER_RULES_VERSION}/pdfplumber-{_version('pdfplumber')}"
    f"/pypdf-{_version('pypdf')}"
)


class ParsedDocumentCache:
    def __init__(self, maxsize: int = 32) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be at least 1")
        self._maxsize = maxsize
        self._items: OrderedDict[tuple[str, str, str], ParsedDocument] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def parse(self, data: bytes, *, filename: str, expected_sha256: str) -> ParsedDocument:
        """``parse_pdf`` with reuse. Raises exactly what ``parse_pdf`` raises."""
        sha = sha256_bytes(data)
        key = (sha, filename, PARSER_ID)
        with self._lock:
            found = self._items.get(key)
            if found is not None and sha == expected_sha256:
                self._items.move_to_end(key)
                self.hits += 1
                return found
        doc = parse_pdf(data, filename=filename, expected_sha256=expected_sha256)
        with self._lock:
            self.misses += 1
            self._items[key] = doc
            self._items.move_to_end(key)
            while len(self._items) > self._maxsize:
                self._items.popitem(last=False)
        return doc

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


DEFAULT_CACHE = ParsedDocumentCache()
