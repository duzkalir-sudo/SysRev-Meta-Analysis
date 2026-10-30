"""Unpaywall REST API client (Open Access location resolution) and safe PDF download.

Reference: https://unpaywall.org/products/api — ``GET /v2/{doi}?email=...``.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from src.apis.base import BaseAPIClient
from src.apis.rate_limit import AsyncRateLimiter
from src.core.exceptions import APIClientError
from src.core.identifiers import normalize_doi

UNPAYWALL_BASE = "https://api.unpaywall.org/v2/"
PDF_MAGIC = b"%PDF-"
MAX_PDF_BYTES = 100 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class OALocation:
    url: str | None
    url_for_pdf: str | None
    host_type: str | None
    license: str | None
    version: str | None


@dataclass(frozen=True, slots=True)
class UnpaywallResult:
    doi: str
    is_oa: bool
    oa_status: str | None
    best_oa_location: OALocation | None
    oa_locations: list[OALocation]

    @property
    def pdf_candidates(self) -> list[str]:
        """Direct PDF URLs, best location first, without duplicates."""
        urls: list[str] = []
        locations = ([self.best_oa_location] if self.best_oa_location else []) + self.oa_locations
        for loc in locations:
            if loc.url_for_pdf and loc.url_for_pdf not in urls:
                urls.append(loc.url_for_pdf)
        return urls


@dataclass(frozen=True, slots=True)
class DownloadedPDF:
    path: Path
    sha256: str
    size_bytes: int
    source_url: str


def _write_file(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _location(raw: dict[str, Any] | None) -> OALocation | None:
    if not raw:
        return None
    return OALocation(
        url=raw.get("url"),
        url_for_pdf=raw.get("url_for_pdf"),
        host_type=raw.get("host_type"),
        license=raw.get("license"),
        version=raw.get("version"),
    )


class UnpaywallClient(BaseAPIClient):
    base_url = UNPAYWALL_BASE
    name = "Unpaywall"

    def __init__(
        self,
        *,
        email: str,
        requests_per_second: float = 5.0,
        rate_limiter: AsyncRateLimiter | None = None,
        http_client: httpx.AsyncClient | None = None,
        **kwargs: Any,
    ) -> None:
        if not email:
            raise ValueError("Unpaywall requires an email parameter on every request")
        super().__init__(
            rate_limiter=rate_limiter or AsyncRateLimiter.per_second(requests_per_second),
            http_client=http_client,
            contact_email=email,
            **kwargs,
        )
        self.email = email

    async def lookup(self, doi: str) -> UnpaywallResult | None:
        """Return OA status for ``doi`` or ``None`` if Unpaywall does not know the DOI."""
        norm = normalize_doi(doi)
        assert norm is not None
        data = await self._get_json(quote(norm, safe="/"), {"email": self.email}, allow_404=True)
        if data is None:
            return None
        if not isinstance(data, dict):
            raise APIClientError("unexpected Unpaywall response shape")
        locations = [loc for raw in data.get("oa_locations") or [] if (loc := _location(raw))]
        return UnpaywallResult(
            doi=str(data.get("doi") or norm).lower(),
            is_oa=bool(data.get("is_oa")),
            oa_status=data.get("oa_status"),
            best_oa_location=_location(data.get("best_oa_location")),
            oa_locations=locations,
        )

    async def download_pdf(self, url: str, destination: Path) -> DownloadedPDF:
        """Download ``url`` to ``destination`` only if the payload is really a PDF.

        Landing pages served with HTTP 200 are common; the ``%PDF-`` magic bytes
        are checked rather than trusting the Content-Type header.
        """
        response = await self._request("GET", url)
        assert response is not None
        content = response.content
        if len(content) > MAX_PDF_BYTES:
            raise APIClientError(f"PDF exceeds {MAX_PDF_BYTES} bytes", url=url)
        if not content.lstrip()[:5].startswith(PDF_MAGIC):
            raise APIClientError("downloaded content is not a PDF", url=url)
        await asyncio.to_thread(_write_file, destination, content)
        return DownloadedPDF(
            path=destination,
            sha256=hashlib.sha256(content).hexdigest(),
            size_bytes=len(content),
            source_url=url,
        )
