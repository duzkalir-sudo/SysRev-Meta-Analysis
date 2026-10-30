"""Lawful open-access PDF download with streaming size limits and file validation.

Only URLs supplied by open-access services (Unpaywall, Europe PMC) are fetched; nothing
here circumvents authentication, paywalls or robots restrictions. The response is
streamed to a partial file with a hard byte limit, the ``%PDF-`` header is checked (HTML
landing pages are often served with HTTP 200), the SHA-256 is computed while streaming,
and the file is opened with pypdf before it is accepted.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import httpx
from pypdf import PdfReader

from src import __version__
from src.fulltext.storage import DocumentStore, StoredFile

PDF_MAGIC = b"%PDF-"
HEADER_WINDOW = 1024  # the PDF header may be preceded by junk bytes (ISO 32000-1 §7.5.2)

FailureKind = Literal["invalid_url", "network_error", "http_error", "not_pdf", "too_large",
                      "malformed"]  # fmt: skip


@dataclass(frozen=True, slots=True)
class DownloadLimits:
    max_bytes: int = 50 * 1024 * 1024
    timeout_seconds: float = 60.0


@dataclass(frozen=True, slots=True)
class DownloadResult:
    url: str
    stored: StoredFile | None = None
    failure: FailureKind | None = None
    detail: str | None = None

    @property
    def ok(self) -> bool:
        return self.stored is not None


def _validate_pdf(path: Path) -> str | None:
    try:
        reader = PdfReader(str(path))
        if len(reader.pages) == 0:
            return "PDF has no pages"
    except Exception as exc:  # pypdf raises a variety of errors for corrupt files
        return f"PDF could not be opened ({type(exc).__name__})"
    return None


def validate_pdf_bytes(data: bytes, store: DocumentStore) -> tuple[StoredFile | None, str | None]:
    """Validate an in-memory PDF (e.g. a user upload) and store it if valid."""
    if PDF_MAGIC not in data[:HEADER_WINDOW]:
        return None, "file does not start with a PDF header"
    partial = store.new_partial()
    partial.write_bytes(data)
    problem = _validate_pdf(partial)
    if problem:
        partial.unlink(missing_ok=True)
        return None, problem
    return store.commit_partial(partial, hashlib.sha256(data).hexdigest(), ".pdf"), None


class PDFDownloader:
    def __init__(
        self,
        store: DocumentStore,
        *,
        limits: DownloadLimits | None = None,
        http_client: httpx.AsyncClient | None = None,
        contact_email: str | None = None,
    ) -> None:
        self.store = store
        self.limits = limits or DownloadLimits()
        ua = f"sysrev-meta-analysis/{__version__}"
        if contact_email:
            ua += f" (mailto:{contact_email})"
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(
            timeout=self.limits.timeout_seconds,
            headers={"User-Agent": ua},
            follow_redirects=True,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def fetch(self, url: str) -> DownloadResult:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return DownloadResult(url, failure="invalid_url", detail="only http(s) URLs")
        partial = self.store.new_partial()
        try:
            return await self._stream(url, partial)
        except httpx.HTTPError as exc:
            return DownloadResult(url, failure="network_error", detail=type(exc).__name__)
        finally:
            partial.unlink(missing_ok=True)  # no-op once committed

    async def _stream(self, url: str, partial: Path) -> DownloadResult:
        limit = self.limits.max_bytes
        async with self._client.stream("GET", url) as response:
            if not response.is_success:
                return DownloadResult(
                    url, failure="http_error", detail=f"HTTP {response.status_code}"
                )
            declared = response.headers.get("Content-Length")
            if declared and declared.isdigit() and int(declared) > limit:
                return DownloadResult(url, failure="too_large", detail=f"{declared} bytes")
            digest = hashlib.sha256()
            size = 0
            head = b""
            with partial.open("wb") as fh:
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > limit:
                        return DownloadResult(url, failure="too_large", detail=f">{limit} bytes")
                    if len(head) < HEADER_WINDOW:
                        head += chunk[: HEADER_WINDOW - len(head)]
                    digest.update(chunk)
                    fh.write(chunk)
        if PDF_MAGIC not in head:
            return DownloadResult(url, failure="not_pdf", detail="no %PDF- header")
        problem = await asyncio.to_thread(_validate_pdf, partial)
        if problem:
            return DownloadResult(url, failure="malformed", detail=problem)
        stored = self.store.commit_partial(partial, digest.hexdigest(), ".pdf")
        return DownloadResult(url, stored=stored)
