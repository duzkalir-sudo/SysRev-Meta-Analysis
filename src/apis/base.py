"""Shared async HTTP client with rate limiting, retries and back-off."""

from __future__ import annotations

import asyncio
import email.utils
import logging
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from types import TracebackType
from typing import Any, Self

import httpx

from src import __version__
from src.apis.rate_limit import AsyncRateLimiter
from src.core.exceptions import APIClientError

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


def _parse_retry_after(value: str | None) -> float | None:
    """Parse an HTTP ``Retry-After`` header (delta-seconds or HTTP-date, RFC 9110 §10.2.3)."""
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(UTC)).total_seconds())


class BaseAPIClient:
    """Base class for literature API clients.

    Subclasses set ``base_url`` and build requests with :meth:`_request`. A
    client may be used as an async context manager; if an ``httpx.AsyncClient``
    is injected, the caller owns its lifecycle.
    """

    base_url: str = ""
    name: str = "api"

    def __init__(
        self,
        *,
        rate_limiter: AsyncRateLimiter,
        http_client: httpx.AsyncClient | None = None,
        timeout: float = 30.0,
        max_retries: int = 5,
        backoff_base: float = 1.0,
        backoff_max: float = 60.0,
        contact_email: str | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._limiter = rate_limiter
        self._owns_client = http_client is None
        ua = f"sysrev-meta-analysis/{__version__}"
        if contact_email:
            ua += f" (mailto:{contact_email})"
        self._client = http_client or httpx.AsyncClient(
            timeout=timeout, headers={"User-Agent": ua}, follow_redirects=True
        )
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self._sleep = sleep

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _backoff(self, attempt: int) -> float:
        return float(min(self.backoff_max, self.backoff_base * (2**attempt)))

    async def _request(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        data: Mapping[str, Any] | None = None,
        allow_404: bool = False,
    ) -> httpx.Response | None:
        """Send a request through the rate limiter, retrying transient failures.

        Returns ``None`` for 404 when ``allow_404`` is set; raises
        :class:`APIClientError` for other non-success responses.
        """
        full_url = url if url.startswith("http") else f"{self.base_url}{url}"
        last_error: str = ""
        for attempt in range(self.max_retries + 1):
            await self._limiter.acquire()
            try:
                response = await self._client.request(method, full_url, params=params, data=data)
            except httpx.TransportError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt == self.max_retries:
                    break
                delay = self._backoff(attempt)
                logger.warning("%s transport error (%s); retry in %.1fs", self.name, exc, delay)
                await self._sleep(delay)
                continue

            if response.status_code == 404 and allow_404:
                return None
            if response.status_code in RETRYABLE_STATUS and attempt < self.max_retries:
                retry_after = _parse_retry_after(response.headers.get("Retry-After"))
                delay = retry_after if retry_after is not None else self._backoff(attempt)
                delay = min(delay, self.backoff_max)
                logger.warning(
                    "%s HTTP %s; retry %d/%d in %.1fs",
                    self.name,
                    response.status_code,
                    attempt + 1,
                    self.max_retries,
                    delay,
                )
                await self._sleep(delay)
                continue
            if response.is_success:
                return response
            raise APIClientError(
                f"{self.name} returned HTTP {response.status_code}: {response.text[:300]}",
                status_code=response.status_code,
                url=str(response.request.url),
            )
        raise APIClientError(
            f"{self.name} request failed after {self.max_retries + 1} attempts: {last_error}",
            url=full_url,
        )

    async def _get_json(
        self, url: str, params: Mapping[str, Any] | None = None, *, allow_404: bool = False
    ) -> Any:
        response = await self._request("GET", url, params=params, allow_404=allow_404)
        if response is None:
            return None
        try:
            return response.json()
        except ValueError as exc:
            raise APIClientError(f"{self.name} returned invalid JSON", url=url) from exc
