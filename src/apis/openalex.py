"""OpenAlex Works API client.

Reference: https://docs.openalex.org — ``/works`` with ``search`` or ``filter``,
``per-page`` ≤ 200 and cursor paging (``cursor=*``). A ``mailto`` parameter
routes requests to the polite pool. OpenAlex abstracts are distributed as an
inverted index and are reconstructed here verbatim (word positions only).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import httpx

from src.apis.base import BaseAPIClient
from src.apis.rate_limit import AsyncRateLimiter
from src.core.enums import SourceDatabase
from src.core.exceptions import APIClientError, InvalidIdentifierError
from src.core.identifiers import (
    extract_registry_ids,
    normalize_doi,
    normalize_pmcid,
    normalize_pmid,
    strip_markup,
)
from src.core.schemas.study import Author, SearchProvenance, StudyRecord

OPENALEX_BASE = "https://api.openalex.org/"
MAX_PER_PAGE = 200


def reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str | None:
    """Rebuild abstract text from OpenAlex's ``abstract_inverted_index``."""
    if not inverted_index:
        return None
    positions: dict[int, str] = {}
    for word, idxs in inverted_index.items():
        for i in idxs:
            positions[i] = word
    if not positions:
        return None
    return " ".join(positions[i] for i in sorted(positions))


class OpenAlexClient(BaseAPIClient):
    base_url = OPENALEX_BASE
    name = "OpenAlex"

    def __init__(
        self,
        *,
        mailto: str | None = None,
        api_key: str | None = None,
        requests_per_second: float = 8.0,
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
        self.api_key = api_key

    def _common(self) -> dict[str, str]:
        params: dict[str, str] = {}
        if self.mailto:
            params["mailto"] = self.mailto
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    async def iter_works(
        self,
        *,
        search: str | None = None,
        filter_: str | None = None,
        per_page: int = MAX_PER_PAGE,
        max_records: int | None = None,
    ) -> AsyncIterator[tuple[int, dict[str, Any]]]:
        if not (search or filter_):
            raise ValueError("provide search and/or filter_")
        params: dict[str, Any] = {
            **self._common(),
            "per-page": max(1, min(per_page, MAX_PER_PAGE)),
        }
        if search:
            params["search"] = search
        if filter_:
            params["filter"] = filter_
        cursor: str | None = "*"
        yielded = 0
        while cursor:
            data = await self._get_json("works", {**params, "cursor": cursor})
            if not isinstance(data, dict):
                raise APIClientError("unexpected OpenAlex response shape")
            meta = data.get("meta") or {}
            count = int(meta.get("count") or 0)
            results = data.get("results") or []
            for raw in results:
                yield count, raw
                yielded += 1
                if max_records is not None and yielded >= max_records:
                    return
            cursor = meta.get("next_cursor") if results else None

    async def search(
        self,
        *,
        search: str | None = None,
        filter_: str | None = None,
        max_records: int | None = None,
    ) -> tuple[int, list[StudyRecord]]:
        query = " | ".join(
            p
            for p in (f"search={search}" if search else "", f"filter={filter_}" if filter_ else "")
            if p
        )
        executed_at = datetime.now(UTC)
        total = 0
        records: list[StudyRecord] = []
        async for total, raw in self.iter_works(
            search=search, filter_=filter_, max_records=max_records
        ):
            record = parse_openalex_work(
                raw, query=query, executed_at=executed_at, total_hits=total
            )
            if record is not None:
                records.append(record)
        return total, records

    async def get_work_by_doi(self, doi: str) -> StudyRecord | None:
        norm = normalize_doi(doi)
        raw = await self._get_json(f"works/https://doi.org/{norm}", self._common(), allow_404=True)
        if raw is None:
            return None
        return parse_openalex_work(
            raw, query=f"doi:{norm}", executed_at=datetime.now(UTC), total_hits=1
        )


def _safe(fn: Any, value: Any) -> Any:
    try:
        return fn(value)
    except InvalidIdentifierError:
        return None


def _split_display_name(name: str) -> tuple[str | None, str | None]:
    parts = name.split()
    if len(parts) < 2:
        return (parts[0] if parts else None), None
    return parts[-1], " ".join(parts[:-1])


def parse_openalex_work(
    raw: dict[str, Any], *, query: str, executed_at: datetime, total_hits: int
) -> StudyRecord | None:
    title = raw.get("title") or raw.get("display_name")
    if not title:
        return None
    title = " ".join(strip_markup(title).split())
    ids = raw.get("ids") or {}
    abstract = reconstruct_abstract(raw.get("abstract_inverted_index"))

    authors: list[Author] = []
    for authorship in raw.get("authorships") or []:
        author = authorship.get("author") or {}
        display = author.get("display_name") or authorship.get("raw_author_name")
        if not display:
            continue
        last, fore = _split_display_name(display)
        affs: list[str] = list(authorship.get("raw_affiliation_strings") or [])
        if not affs:
            affs = [
                inst["display_name"]
                for inst in authorship.get("institutions") or []
                if inst.get("display_name")
            ]
        orcid_url = author.get("orcid") or ""
        orcid = orcid_url.rsplit("/", 1)[-1] if orcid_url else None
        authors.append(
            Author(
                last_name=last,
                fore_name=fore,
                full_name=display,
                orcid=orcid or None,
                affiliations=[a.strip() for a in affs if a and a.strip()],
                name_parsed_heuristically=True,
            )
        )

    source = (raw.get("primary_location") or {}).get("source") or {}
    openalex_id = (raw.get("id") or "").rsplit("/", 1)[-1] or None
    year = raw.get("publication_year")

    return StudyRecord(
        source_database=SourceDatabase.OPENALEX,
        provenance=SearchProvenance(
            database=SourceDatabase.OPENALEX,
            query=query or "openalex",
            endpoint=f"{OPENALEX_BASE}works",
            executed_at=executed_at,
            total_hits_reported=total_hits,
            source_record_id=openalex_id or "unknown",
        ),
        openalex_id=openalex_id,
        pmid=_safe(normalize_pmid, ids.get("pmid")),
        pmcid=_safe(normalize_pmcid, ids.get("pmcid")),
        doi=_safe(normalize_doi, raw.get("doi") or ids.get("doi")),
        title=title,
        abstract=abstract,
        authors=authors,
        journal=source.get("display_name"),
        publication_year=year if isinstance(year, int) else None,
        publication_types=[raw["type"]] if raw.get("type") else [],
        language=raw.get("language"),
        registry_ids=extract_registry_ids(abstract),
    )
