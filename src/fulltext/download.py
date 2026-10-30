"""Lawful open-access PDF download with streaming size limits and file validation.

Only URLs supplied by open-access services (Unpaywall, Europe PMC) are fetched; nothing
here circumvents authentication, paywalls or robots restrictions. The response is
streamed to a partial file with a hard byte limit, the ``%PDF-`` header is checked (HTML
landing pages are often served with HTTP 200), the SHA-256 is computed while streaming,
and the file is opened with pypdf before it is accepted.

SSRF protection. OA URLs come from third-party metadata, so they are untrusted. Every
request target — the initial URL and each redirect hop — must use http(s), carry no
credentials, and resolve *only* to globally routable addresses (loopback, private,
link-local, carrier-grade NAT, multicast, reserved and unspecified ranges are refused).
Redirects are followed manually (at most ``max_redirects``) so each hop is checked before
it is requested. Residual risk: DNS rebinding between this check and httpx's own
resolution is not prevented (it would require pinning the connection to the checked IP).
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import socket
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urljoin, urlparse

import httpx
from pypdf import PdfReader

from src import __version__
from src.fulltext.storage import DocumentStore, StoredFile

PDF_MAGIC = b"%PDF-"
HEADER_WINDOW = 1024  # the PDF header may be preceded by junk bytes (ISO 32000-1 §7.5.2)

FailureKind = Literal["invalid_url", "blocked_url", "network_error", "http_error", "not_pdf",
                      "too_large", "malformed"]  # fmt: skip
Resolver = Callable[[str], Awaitable[list[str]]]


async def system_resolver(host: str) -> list[str]:
    infos = await asyncio.get_running_loop().getaddrinfo(host, None, type=socket.SOCK_STREAM)
    return [str(info[4][0]) for info in infos]


def is_public_address(address: str) -> bool:
    """True only for globally routable unicast addresses (IPv4-mapped IPv6 unwrapped)."""
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


@dataclass(frozen=True, slots=True)
class DownloadLimits:
    max_bytes: int = 50 * 1024 * 1024
    timeout_seconds: float = 60.0
    max_redirects: int = 5


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
        resolver: Resolver = system_resolver,
    ) -> None:
        self.store = store
        self.limits = limits or DownloadLimits()
        self._resolve = resolver
        ua = f"sysrev-meta-analysis/{__version__}"
        if contact_email:
            ua += f" (mailto:{contact_email})"
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(
            timeout=self.limits.timeout_seconds,
            headers={"User-Agent": ua},
            follow_redirects=False,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def check_url(self, url: str) -> DownloadResult | None:
        """``None`` if ``url`` may be requested, else the failure to report."""
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            return DownloadResult(url, failure="invalid_url", detail="only http(s) URLs")
        if parsed.username or parsed.password:
            return DownloadResult(url, failure="blocked_url", detail="credentials in URL")
        host = parsed.hostname
        try:
            ipaddress.ip_address(host)
            addresses = [host]
        except ValueError:
            try:
                addresses = await self._resolve(host)
            except OSError as exc:
                return DownloadResult(url, failure="network_error", detail=f"DNS: {exc}")
        if not addresses or not all(is_public_address(a) for a in addresses):
            return DownloadResult(
                url, failure="blocked_url", detail=f"{host} resolves to a non-public address"
            )
        return None

    async def fetch(self, url: str) -> DownloadResult:
        partial = self.store.new_partial()
        try:
            current = url
            for _ in range(self.limits.max_redirects + 1):
                blocked = await self.check_url(current)
                if blocked is not None:
                    return DownloadResult(url, failure=blocked.failure,
                                          detail=f"{current}: {blocked.detail}")  # fmt: skip
                outcome = await self._stream(current, partial)
                if isinstance(outcome, DownloadResult):
                    return outcome
                current = urljoin(current, outcome)
            return DownloadResult(url, failure="http_error", detail="too many redirects")
        except httpx.HTTPError as exc:
            return DownloadResult(url, failure="network_error", detail=type(exc).__name__)
        finally:
            partial.unlink(missing_ok=True)  # no-op once committed

    async def _stream(self, url: str, partial: Path) -> DownloadResult | str:
        """Download one hop; a ``str`` result is a redirect location still to be checked."""
        limit = self.limits.max_bytes
        async with self._client.stream("GET", url, follow_redirects=False) as response:
            if response.is_redirect:
                location = response.headers.get("Location")
                if not location:
                    return DownloadResult(url, failure="http_error", detail="empty redirect")
                return str(location)
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
