"""NCBI Entrez (PubMed E-utilities) client.

Reference: Sayers E. *A General Introduction to the E-utilities*. In: Entrez
Programming Utilities Help. NCBI; https://www.ncbi.nlm.nih.gov/books/NBK25497/

Policy honoured here: ``tool`` and ``email`` are sent on every request; the
rate is capped at 3 requests/second (10 with an API key); large ID sets are
fetched with HTTP POST in batches.
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx
from lxml import etree

from src.apis.base import BaseAPIClient
from src.apis.rate_limit import AsyncRateLimiter
from src.core.enums import SourceDatabase
from src.core.exceptions import APIClientError, InvalidIdentifierError, SearchResultTooLargeError
from src.core.identifiers import extract_registry_ids, normalize_doi, normalize_pmcid
from src.core.schemas.study import Author, SearchProvenance, StudyRecord

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
# ESearch cannot retrieve records beyond the first 10,000 for PubMed.
PUBMED_MAX_RETRIEVABLE = 10_000
EFETCH_BATCH_SIZE = 200

_SAFE_PARSER = etree.XMLParser(
    resolve_entities=False, no_network=True, huge_tree=False, remove_comments=True
)
_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2}|2100)\b")
_ORCID_RE = re.compile(r"(\d{4}-\d{4}-\d{4}-\d{3}[\dX])")


@dataclass(frozen=True, slots=True)
class ESearchResult:
    query: str
    count: int
    pmids: list[str]
    query_translation: str | None
    executed_at: datetime


def _text(el: etree._Element | None) -> str | None:
    """All descendant text of an element (handles inline markup such as <i>, <sup>)."""
    if el is None:
        return None
    value = " ".join("".join(str(t) for t in el.itertext()).split())
    return value or None


class PubMedClient(BaseAPIClient):
    base_url = EUTILS_BASE
    name = "PubMed E-utilities"

    def __init__(
        self,
        *,
        email: str,
        tool: str = "sysrev-meta-analysis",
        api_key: str | None = None,
        rate_limiter: AsyncRateLimiter | None = None,
        http_client: httpx.AsyncClient | None = None,
        **kwargs: Any,
    ) -> None:
        if not email:
            raise ValueError("NCBI requires a contact email for E-utilities requests")
        limiter = rate_limiter or AsyncRateLimiter(10 if api_key else 3, 1.0)
        super().__init__(
            rate_limiter=limiter, http_client=http_client, contact_email=email, **kwargs
        )
        self.email = email
        self.tool = tool
        self.api_key = api_key

    def _common(self) -> dict[str, str]:
        params = {"tool": self.tool, "email": self.email}
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    async def esearch(
        self,
        query: str,
        *,
        mindate: str | None = None,
        maxdate: str | None = None,
        datetype: str = "pdat",
    ) -> ESearchResult:
        """Run a PubMed search and return *all* matching PMIDs.

        Raises:
            SearchResultTooLargeError: if the query matches more than 10,000
                records; the caller must partition the search (e.g. by date).
        """
        executed_at = datetime.now(UTC)
        base: dict[str, Any] = {
            **self._common(),
            "db": "pubmed",
            "term": query,
            "retmode": "json",
        }
        if mindate or maxdate:
            base |= {
                "datetype": datetype,
                "mindate": mindate or "1800",
                "maxdate": maxdate or "3000",
            }

        first = await self._get_json("esearch.fcgi", {**base, "retmax": 0})
        result = self._esearch_payload(first)
        count = int(result["count"])
        if count > PUBMED_MAX_RETRIEVABLE:
            raise SearchResultTooLargeError(
                f"PubMed query matched {count} records; E-utilities can only page through "
                f"the first {PUBMED_MAX_RETRIEVABLE}. Partition the search by date range."
            )
        translation = result.get("querytranslation")
        pmids: list[str] = []
        page = 5_000
        retstart = 0
        while retstart < count:
            data = await self._get_json(
                "esearch.fcgi", {**base, "retstart": retstart, "retmax": page}
            )
            ids = self._esearch_payload(data).get("idlist", [])
            if not ids:
                break
            pmids.extend(ids)
            retstart += len(ids)
        # Preserve order, drop any repeats that can appear if the index changes mid-paging.
        unique = list(dict.fromkeys(pmids))
        return ESearchResult(query, count, unique, translation, executed_at)

    @staticmethod
    def _esearch_payload(data: Any) -> dict[str, Any]:
        if not isinstance(data, dict) or "esearchresult" not in data:
            raise APIClientError("unexpected ESearch response shape")
        result: dict[str, Any] = data["esearchresult"]
        if "ERROR" in result:
            raise APIClientError(f"ESearch error: {result['ERROR']}")
        return result

    async def efetch_records(
        self, pmids: Sequence[str], *, query: str = "", total_hits: int | None = None
    ) -> list[StudyRecord]:
        """Fetch PubMed XML for ``pmids`` (POST, batched) and parse to ``StudyRecord``s."""
        executed_at = datetime.now(UTC)
        records: list[StudyRecord] = []
        for i in range(0, len(pmids), EFETCH_BATCH_SIZE):
            batch = list(pmids[i : i + EFETCH_BATCH_SIZE])
            response = await self._request(
                "POST",
                "efetch.fcgi",
                data={**self._common(), "db": "pubmed", "id": ",".join(batch), "retmode": "xml"},
            )
            assert response is not None
            records.extend(
                parse_pubmed_xml(
                    response.content,
                    query=query,
                    executed_at=executed_at,
                    total_hits=total_hits if total_hits is not None else len(pmids),
                )
            )
        return records

    async def search(
        self, query: str, *, mindate: str | None = None, maxdate: str | None = None
    ) -> tuple[ESearchResult, list[StudyRecord]]:
        """ESearch + EFetch in one call."""
        result = await self.esearch(query, mindate=mindate, maxdate=maxdate)
        records = await self.efetch_records(result.pmids, query=query, total_hits=result.count)
        return result, records

    async def mesh_descriptor_exists(self, descriptor: str) -> bool:
        """Check that a MeSH descriptor exists in the NLM MeSH database (db=mesh)."""
        cleaned = descriptor.replace('"', "").strip()
        if not cleaned:
            return False
        data = await self._get_json(
            "esearch.fcgi",
            {**self._common(), "db": "mesh", "term": f'"{cleaned}"[MeSH Terms]', "retmode": "json"},
        )
        return int(self._esearch_payload(data).get("count", 0)) > 0


def parse_pubmed_xml(
    content: bytes,
    *,
    query: str = "",
    executed_at: datetime | None = None,
    total_hits: int = 0,
) -> list[StudyRecord]:
    """Parse an EFetch ``PubmedArticleSet`` document. Only fields present are mapped."""
    try:
        root = etree.fromstring(content, parser=_SAFE_PARSER)
    except etree.XMLSyntaxError as exc:
        raise APIClientError(f"malformed PubMed XML: {exc}") from exc
    executed_at = executed_at or datetime.now(UTC)
    records: list[StudyRecord] = []
    for article in root.iter("PubmedArticle"):
        record = _parse_article(article, query, executed_at, total_hits)
        if record is not None:
            records.append(record)
    return records


def _parse_article(
    article: etree._Element, query: str, executed_at: datetime, total_hits: int
) -> StudyRecord | None:
    citation = article.find("MedlineCitation")
    if citation is None:
        return None
    pmid = _text(citation.find("PMID"))
    art = citation.find("Article")
    if pmid is None or art is None:
        return None
    title = _text(art.find("ArticleTitle")) or _text(art.find("VernacularTitle"))
    if not title:
        return None

    abstract_parts: list[str] = []
    for node in art.findall("Abstract/AbstractText"):
        text = _text(node)
        if not text:
            continue
        label = node.get("Label")
        abstract_parts.append(f"{label}: {text}" if label else text)
    abstract = "\n".join(abstract_parts) or None

    authors: list[Author] = []
    for a in art.findall("AuthorList/Author"):
        if a.get("ValidYN", "Y") != "Y":
            continue
        affiliations = [t for n in a.findall("AffiliationInfo/Affiliation") if (t := _text(n))]
        orcid = None
        for ident in a.findall("Identifier"):
            if ident.get("Source") == "ORCID" and (m := _ORCID_RE.search(_text(ident) or "")):
                orcid = m.group(1)
        last, collective = _text(a.find("LastName")), _text(a.find("CollectiveName"))
        if not (last or collective):
            continue
        authors.append(
            Author(
                last_name=last,
                fore_name=_text(a.find("ForeName")),
                initials=_text(a.find("Initials")),
                collective_name=collective,
                orcid=orcid,
                affiliations=affiliations,
            )
        )

    year = _publication_year(art)
    doi = pmcid = None
    for aid in article.findall("PubmedData/ArticleIdList/ArticleId"):
        kind, value = aid.get("IdType"), _text(aid)
        try:
            if kind == "doi" and doi is None:
                doi = normalize_doi(value)
            elif kind == "pmc" and pmcid is None:
                pmcid = normalize_pmcid(value)
        except InvalidIdentifierError:
            continue
    if doi is None:
        for eloc in art.findall("ELocationID"):
            if eloc.get("EIdType") == "doi" and eloc.get("ValidYN", "Y") == "Y":
                with contextlib.suppress(InvalidIdentifierError):
                    doi = normalize_doi(_text(eloc))
                break

    registry: set[str] = set(extract_registry_ids(abstract))
    for acc in art.findall("DataBankList/DataBank/AccessionNumberList/AccessionNumber"):
        registry.update(extract_registry_ids(_text(acc)))

    return StudyRecord(
        source_database=SourceDatabase.PUBMED,
        provenance=SearchProvenance(
            database=SourceDatabase.PUBMED,
            query=query or f"PMID:{pmid}",
            endpoint=f"{EUTILS_BASE}efetch.fcgi",
            executed_at=executed_at,
            total_hits_reported=total_hits,
            source_record_id=pmid,
        ),
        pmid=pmid,
        pmcid=pmcid,
        doi=doi,
        title=title,
        abstract=abstract,
        authors=authors,
        journal=_text(art.find("Journal/Title")),
        publication_year=year,
        publication_types=[
            t for n in art.findall("PublicationTypeList/PublicationType") if (t := _text(n))
        ],
        mesh_headings=[
            t
            for n in citation.findall("MeshHeadingList/MeshHeading/DescriptorName")
            if (t := _text(n))
        ],
        keywords=[t for n in citation.findall("KeywordList/Keyword") if (t := _text(n))],
        language=_text(art.find("Language")),
        registry_ids=sorted(registry),
    )


def _publication_year(art: etree._Element) -> int | None:
    pub_date = art.find("Journal/JournalIssue/PubDate")
    if pub_date is not None:
        year = _text(pub_date.find("Year"))
        if year and year.isdigit():
            return int(year)
        medline = _text(pub_date.find("MedlineDate"))
        if medline and (m := _YEAR_RE.search(medline)):
            return int(m.group(1))
    article_date = _text(art.find("ArticleDate/Year"))
    return int(article_date) if article_date and article_date.isdigit() else None
