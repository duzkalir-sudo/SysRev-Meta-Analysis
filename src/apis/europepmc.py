"""Europe PMC RESTful Web Service client.

Reference: https://europepmc.org/RestfulWebService — ``/search`` with
``format=json``, ``resultType=core`` and ``cursorMark`` deep paging
(``pageSize`` ≤ 1000).
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

EUROPEPMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/"
MAX_PAGE_SIZE = 1000


class EuropePMCClient(BaseAPIClient):
    base_url = EUROPEPMC_BASE
    name = "Europe PMC"

    def __init__(
        self,
        *,
        requests_per_second: float = 5.0,
        rate_limiter: AsyncRateLimiter | None = None,
        http_client: httpx.AsyncClient | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            rate_limiter=rate_limiter or AsyncRateLimiter.per_second(requests_per_second),
            http_client=http_client,
            **kwargs,
        )

    async def iter_results(
        self, query: str, *, page_size: int = MAX_PAGE_SIZE, max_records: int | None = None
    ) -> AsyncIterator[tuple[int, dict[str, Any]]]:
        """Yield ``(hitCount, raw_result)`` for every hit using cursorMark paging."""
        page_size = max(1, min(page_size, MAX_PAGE_SIZE))
        cursor = "*"
        yielded = 0
        while True:
            data = await self._get_json(
                "search",
                {
                    "query": query,
                    "format": "json",
                    "resultType": "core",
                    "pageSize": page_size,
                    "cursorMark": cursor,
                },
            )
            if not isinstance(data, dict):
                raise APIClientError("unexpected Europe PMC response shape")
            hit_count = int(data.get("hitCount", 0))
            results = data.get("resultList", {}).get("result", [])
            for raw in results:
                yield hit_count, raw
                yielded += 1
                if max_records is not None and yielded >= max_records:
                    return
            next_cursor = data.get("nextCursorMark")
            if not results or not next_cursor or next_cursor == cursor:
                return
            cursor = next_cursor

    async def search(
        self, query: str, *, max_records: int | None = None
    ) -> tuple[int, list[StudyRecord]]:
        executed_at = datetime.now(UTC)
        hit_count = 0
        records: list[StudyRecord] = []
        async for hit_count, raw in self.iter_results(query, max_records=max_records):
            record = parse_europepmc_result(
                raw, query=query, executed_at=executed_at, total_hits=hit_count
            )
            if record is not None:
                records.append(record)
        return hit_count, records

    async def find_pmcid(self, pmid: str) -> str | None:
        """PMCID of a MEDLINE record (``EXT_ID:<pmid> AND SRC:MED``), or ``None``."""
        norm = normalize_pmid(pmid)
        data = await self._get_json(
            "search",
            {"query": f"EXT_ID:{norm} AND SRC:MED", "format": "json", "resultType": "lite"},
        )
        if not isinstance(data, dict):
            raise APIClientError("unexpected Europe PMC response shape")
        for raw in data.get("resultList", {}).get("result", []):
            if str(raw.get("pmid")) == norm and raw.get("pmcid"):
                return _safe(normalize_pmcid, raw["pmcid"])  # type: ignore[no-any-return]
        return None

    async def fetch_fulltext_xml(self, pmcid: str) -> bytes | None:
        """JATS XML of an open-access article (``/{PMCID}/fullTextXML``); ``None`` if absent.

        Europe PMC serves full text only for its open-access subset; other articles 404.
        """
        norm = normalize_pmcid(pmcid)
        response = await self._request("GET", f"{norm}/fullTextXML", allow_404=True)
        if response is None or not response.content.strip():
            return None
        return response.content


def _safe(fn: Any, value: Any) -> Any:
    try:
        return fn(value)
    except InvalidIdentifierError:
        return None


def parse_europepmc_result(
    raw: dict[str, Any], *, query: str, executed_at: datetime, total_hits: int
) -> StudyRecord | None:
    title = raw.get("title")
    if not title:
        return None
    title = " ".join(strip_markup(title).split())
    abstract = raw.get("abstractText")
    abstract = " ".join(strip_markup(abstract).split()) if abstract else None

    authors: list[Author] = []
    for a in (raw.get("authorList") or {}).get("author", []) or []:
        affs = [
            aff["affiliation"].strip()
            for aff in (a.get("authorAffiliationDetailsList") or {}).get("authorAffiliation", [])
            if aff.get("affiliation")
        ]
        if not affs and a.get("affiliation"):
            affs = [a["affiliation"].strip()]
        orcid = (
            (a.get("authorId") or {}).get("value")
            if (a.get("authorId") or {}).get("type") == "ORCID"
            else None
        )
        last, full, collective = a.get("lastName"), a.get("fullName"), a.get("collectiveName")
        if not (last or full or collective):
            continue
        authors.append(
            Author(
                last_name=last,
                fore_name=a.get("firstName"),
                initials=a.get("initials"),
                full_name=full,
                collective_name=collective,
                orcid=orcid,
                affiliations=affs,
            )
        )

    journal_info = raw.get("journalInfo") or {}
    journal = (journal_info.get("journal") or {}).get("title")
    year_raw = journal_info.get("yearOfPublication") or raw.get("pubYear")
    year = int(str(year_raw)) if str(year_raw or "").isdigit() else None

    mesh = [
        m["descriptorName"]
        for m in (raw.get("meshHeadingList") or {}).get("meshHeading", [])
        if m.get("descriptorName")
    ]
    keywords = list((raw.get("keywordList") or {}).get("keyword", []))
    pub_types = list((raw.get("pubTypeList") or {}).get("pubType", []))
    source_id = f"{raw.get('source', 'UNK')}:{raw.get('id', '')}"

    return StudyRecord(
        source_database=SourceDatabase.EUROPE_PMC,
        provenance=SearchProvenance(
            database=SourceDatabase.EUROPE_PMC,
            query=query,
            endpoint=f"{EUROPEPMC_BASE}search",
            executed_at=executed_at,
            total_hits_reported=total_hits,
            source_record_id=source_id,
        ),
        pmid=_safe(normalize_pmid, raw.get("pmid")),
        pmcid=_safe(normalize_pmcid, raw.get("pmcid")),
        doi=_safe(normalize_doi, raw.get("doi")),
        title=title,
        abstract=abstract,
        authors=authors,
        journal=journal,
        publication_year=year,
        publication_types=pub_types,
        mesh_headings=mesh,
        keywords=keywords,
        language=raw.get("language"),
        registry_ids=extract_registry_ids(abstract),
    )
