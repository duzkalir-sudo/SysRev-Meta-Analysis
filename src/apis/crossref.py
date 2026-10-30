"""Crossref REST API client, used by the Auditor to verify that cited DOIs exist.

Reference: https://api.crossref.org — ``GET /works/{doi}``; a ``mailto``
parameter routes requests to the polite pool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx

from src.apis.base import BaseAPIClient
from src.apis.rate_limit import AsyncRateLimiter
from src.core.exceptions import APIClientError
from src.core.identifiers import normalize_doi

CROSSREF_BASE = "https://api.crossref.org/"


@dataclass(frozen=True, slots=True)
class CrossrefWork:
    doi: str
    title: str | None
    container_title: str | None
    year: int | None
    first_author_family: str | None
    type: str | None


class CrossrefClient(BaseAPIClient):
    base_url = CROSSREF_BASE
    name = "Crossref"

    def __init__(
        self,
        *,
        mailto: str | None = None,
        requests_per_second: float = 5.0,
        rate_limiter: AsyncRateLimiter | None = None,
        http_client: httpx.AsyncClient | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            rate_limiter=rate_limiter or AsyncRateLimiter.per_second(requests_per_second),
            http_client=http_client,
            contact_email=mailto,
            **kwargs,
        )
        self.mailto = mailto

    async def get_work(self, doi: str) -> CrossrefWork | None:
        norm = normalize_doi(doi)
        assert norm is not None
        params = {"mailto": self.mailto} if self.mailto else None
        data = await self._get_json(f"works/{quote(norm, safe='/')}", params, allow_404=True)
        if data is None:
            return None
        if not isinstance(data, dict) or "message" not in data:
            raise APIClientError("unexpected Crossref response shape")
        msg: dict[str, Any] = data["message"]
        titles = msg.get("title") or []
        containers = msg.get("container-title") or []
        authors = msg.get("author") or []
        year = None
        for key in ("published-print", "published-online", "issued"):
            parts = (msg.get(key) or {}).get("date-parts") or []
            if parts and parts[0] and isinstance(parts[0][0], int):
                year = parts[0][0]
                break
        return CrossrefWork(
            doi=str(msg.get("DOI", norm)).lower(),
            title=titles[0] if titles else None,
            container_title=containers[0] if containers else None,
            year=year,
            first_author_family=authors[0].get("family") if authors else None,
            type=msg.get("type"),
        )
