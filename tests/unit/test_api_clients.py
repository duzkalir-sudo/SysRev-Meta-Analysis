"""Client tests against respx-mocked endpoints. Response bodies are synthetic but follow
the documented JSON/XML shapes of each API."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import httpx
import pytest
import respx

from src.apis.crossref import CrossrefClient
from src.apis.europepmc import EUROPEPMC_BASE, EuropePMCClient
from src.apis.openalex import OPENALEX_BASE, OpenAlexClient, reconstruct_abstract
from src.apis.pubmed import EUTILS_BASE, PubMedClient, parse_pubmed_xml
from src.apis.rate_limit import AsyncRateLimiter
from src.apis.unpaywall import UNPAYWALL_BASE, UnpaywallClient
from src.core.enums import SourceDatabase
from src.core.exceptions import APIClientError, SearchResultTooLargeError

FIXTURES = Path(__file__).parent.parent / "fixtures"
PUBMED_XML = (FIXTURES / "pubmed_efetch_synthetic.xml").read_bytes()

Sleep = Callable[[float], Awaitable[None]]


def esearch_json(count: int, ids: list[str], translation: str = "") -> dict[str, Any]:
    return {
        "header": {"type": "esearch", "version": "0.3"},
        "esearchresult": {
            "count": str(count),
            "retmax": str(len(ids)),
            "retstart": "0",
            "idlist": ids,
            "querytranslation": translation,
        },
    }


# --------------------------------------------------------------------------- PubMed


def test_parse_pubmed_xml_maps_only_present_fields() -> None:
    records = parse_pubmed_xml(PUBMED_XML, query="q", total_hits=3)
    assert len(records) == 2  # third article lacks a title and is skipped
    r1, r2 = records

    assert r1.pmid == "90000001"
    assert r1.doi == "10.5555/synth.0001"
    assert r1.pmcid == "PMC9000001"
    assert r1.title == "Effect of drug X on HbA1c: a randomised controlled trial."
    assert r1.abstract is not None
    assert r1.abstract.startswith("BACKGROUND: Drug X")
    assert "RESULTS: HbA1c fell by 0·5%." in r1.abstract
    assert [a.match_key for a in r1.authors] == [
        "synthetica a",
        "placebo b",
        "synth 1 trial investigators",
    ]  # ValidYN="N" author dropped
    assert r1.authors[0].orcid == "0000-0002-1825-009X"
    assert r1.authors[0].affiliations == [
        "Department of Endocrinology, Fictional University Hospital, Exampleton, Testland."
    ]
    assert r1.journal == "Journal of Synthetic Medicine"
    assert r1.publication_year == 2021
    assert r1.publication_types == ["Randomized Controlled Trial", "Journal Article"]
    assert r1.mesh_headings == ["Diabetes Mellitus, Type 2", "Glycated Hemoglobin"]
    assert r1.keywords == ["synthetic"]
    assert r1.registry_ids == ["NCT01234567", "NCT07654321"]
    assert r1.provenance is not None
    assert r1.provenance.total_hits_reported == 3
    assert r1.provenance.source_record_id == "90000001"
    assert r1.source_database == SourceDatabase.PUBMED

    assert r2.publication_year == 2019  # from MedlineDate
    assert r2.doi is None  # invalid DOI discarded, not "repaired"
    assert r2.abstract is None


def test_parse_pubmed_xml_rejects_malformed() -> None:
    with pytest.raises(APIClientError, match="malformed"):
        parse_pubmed_xml(b"<PubmedArticleSet><broken>")


def test_parse_pubmed_xml_does_not_expand_entities() -> None:
    evil = (
        b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY e SYSTEM "file:///etc/passwd">]>'
        b"<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>1</PMID><Article>"
        b"<ArticleTitle>T &e;</ArticleTitle></Article></MedlineCitation></PubmedArticle>"
        b"</PubmedArticleSet>"
    )
    records = parse_pubmed_xml(evil)
    assert "root:" not in records[0].title


def test_pubmed_client_requires_email() -> None:
    with pytest.raises(ValueError, match="email"):
        PubMedClient(email="")


@respx.mock
async def test_pubmed_search_end_to_end(no_wait_limiter: AsyncRateLimiter, no_sleep: Sleep) -> None:
    esearch = respx.get(f"{EUTILS_BASE}esearch.fcgi").mock(
        side_effect=[
            httpx.Response(200, json=esearch_json(3, [], "translated")),
            httpx.Response(200, json=esearch_json(3, ["90000001", "90000002", "90000003"])),
        ]
    )
    efetch = respx.post(f"{EUTILS_BASE}efetch.fcgi").mock(
        return_value=httpx.Response(200, content=PUBMED_XML)
    )
    async with PubMedClient(
        email="t@example.org", api_key="KEY", rate_limiter=no_wait_limiter, sleep=no_sleep
    ) as client:
        result, records = await client.search("diabetes AND drug x", mindate="2015")
    assert result.count == 3
    assert result.pmids == ["90000001", "90000002", "90000003"]
    assert result.query_translation == "translated"
    assert len(records) == 2

    params = esearch.calls[0].request.url.params
    assert params["tool"] == "sysrev-meta-analysis"
    assert params["email"] == "t@example.org"
    assert params["api_key"] == "KEY"
    assert params["db"] == "pubmed"
    assert params["mindate"] == "2015"
    assert params["datetype"] == "pdat"
    body = parse_qs(efetch.calls[0].request.content.decode())
    assert body["id"] == ["90000001,90000002,90000003"]
    assert body["retmode"] == ["xml"]


@respx.mock
async def test_pubmed_efetch_batches_ids(
    no_wait_limiter: AsyncRateLimiter, no_sleep: Sleep
) -> None:
    efetch = respx.post(f"{EUTILS_BASE}efetch.fcgi").mock(
        return_value=httpx.Response(200, content=b"<PubmedArticleSet/>")
    )
    ids = [str(i) for i in range(1, 451)]
    async with PubMedClient(email="t@x.org", rate_limiter=no_wait_limiter, sleep=no_sleep) as c:
        await c.efetch_records(ids)
    assert efetch.call_count == 3  # 200 + 200 + 50


@respx.mock
async def test_pubmed_refuses_silent_truncation(
    no_wait_limiter: AsyncRateLimiter, no_sleep: Sleep
) -> None:
    respx.get(f"{EUTILS_BASE}esearch.fcgi").mock(
        return_value=httpx.Response(200, json=esearch_json(25_000, []))
    )
    async with PubMedClient(email="t@x.org", rate_limiter=no_wait_limiter, sleep=no_sleep) as c:
        with pytest.raises(SearchResultTooLargeError, match="Partition"):
            await c.esearch("cancer")


@respx.mock
async def test_pubmed_esearch_error_payload(
    no_wait_limiter: AsyncRateLimiter, no_sleep: Sleep
) -> None:
    respx.get(f"{EUTILS_BASE}esearch.fcgi").mock(
        return_value=httpx.Response(200, json={"esearchresult": {"ERROR": "Invalid query"}})
    )
    async with PubMedClient(email="t@x.org", rate_limiter=no_wait_limiter, sleep=no_sleep) as c:
        with pytest.raises(APIClientError, match="Invalid query"):
            await c.esearch("((")


@respx.mock
async def test_pubmed_mesh_descriptor_exists(
    no_wait_limiter: AsyncRateLimiter, no_sleep: Sleep
) -> None:
    route = respx.get(f"{EUTILS_BASE}esearch.fcgi").mock(
        side_effect=[
            httpx.Response(200, json=esearch_json(1, ["68003920"])),
            httpx.Response(200, json=esearch_json(0, [])),
        ]
    )
    async with PubMedClient(email="t@x.org", rate_limiter=no_wait_limiter, sleep=no_sleep) as c:
        assert await c.mesh_descriptor_exists("Diabetes Mellitus")
        assert not await c.mesh_descriptor_exists("Hallucinated Heading")
        assert not await c.mesh_descriptor_exists('  "" ')
    params = route.calls[0].request.url.params
    assert params["db"] == "mesh"
    assert params["term"] == '"Diabetes Mellitus"[MeSH Terms]'


def test_default_ncbi_rate_depends_on_api_key() -> None:
    assert PubMedClient(email="t@x.org")._limiter.max_calls == 3
    assert PubMedClient(email="t@x.org", api_key="k")._limiter.max_calls == 10


# --------------------------------------------------------------------------- Europe PMC

EPMC_RESULT = {
    "id": "90000001",
    "source": "MED",
    "pmid": "90000001",
    "pmcid": "PMC9000001",
    "doi": "10.5555/SYNTH.0001",
    "title": "Effect of drug <i>X</i> on HbA1c: a randomised controlled trial.",
    "abstractText": "<h4>Methods</h4>Adults randomised (ISRCTN12345678).",
    "authorList": {
        "author": [
            {
                "fullName": "Synthetica A",
                "firstName": "Ana",
                "lastName": "Synthetica",
                "initials": "A",
                "authorId": {"type": "ORCID", "value": "0000-0002-1825-009X"},
                "authorAffiliationDetailsList": {
                    "authorAffiliation": [{"affiliation": "Fictional University Hospital"}]
                },
            },
            {"collectiveName": "SYNTH-1 Investigators"},
            {"initials": "Z"},
        ]
    },
    "journalInfo": {"yearOfPublication": 2021, "journal": {"title": "J Synth Med"}},
    "pubYear": "2021",
    "pubTypeList": {"pubType": ["Randomized Controlled Trial"]},
    "meshHeadingList": {"meshHeading": [{"descriptorName": "Glycated Hemoglobin"}]},
    "keywordList": {"keyword": ["synthetic"]},
    "language": "eng",
}


@respx.mock
async def test_europepmc_cursor_paging(no_wait_limiter: AsyncRateLimiter, no_sleep: Sleep) -> None:
    second = dict(EPMC_RESULT, id="X2", pmid=None, doi=None, title="Second synthetic trial")
    route = respx.get(f"{EUROPEPMC_BASE}search").mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "hitCount": 3,
                    "nextCursorMark": "AoE1",
                    "resultList": {"result": [EPMC_RESULT]},
                },
            ),
            httpx.Response(
                200,
                json={
                    "hitCount": 3,
                    "nextCursorMark": "AoE2",
                    "resultList": {"result": [second, {"id": "X3"}]},
                },
            ),
            httpx.Response(
                200, json={"hitCount": 3, "nextCursorMark": "AoE2", "resultList": {"result": []}}
            ),
        ]
    )
    async with EuropePMCClient(rate_limiter=no_wait_limiter, sleep=no_sleep) as c:
        hits, records = await c.search("drug x")
    assert hits == 3
    assert len(records) == 2  # untitled result skipped
    r = records[0]
    assert r.source_database == SourceDatabase.EUROPE_PMC
    assert (r.pmid, r.pmcid, r.doi) == ("90000001", "PMC9000001", "10.5555/synth.0001")
    assert r.title == "Effect of drug X on HbA1c: a randomised controlled trial."
    assert r.abstract == "Methods Adults randomised (ISRCTN12345678)."
    assert r.registry_ids == ["ISRCTN12345678"]
    assert len(r.authors) == 2
    assert r.authors[0].orcid == "0000-0002-1825-009X"
    assert r.authors[0].affiliations == ["Fictional University Hospital"]
    assert r.publication_year == 2021
    assert r.mesh_headings == ["Glycated Hemoglobin"]
    assert r.provenance is not None and r.provenance.source_record_id == "MED:90000001"
    cursors = [call.request.url.params["cursorMark"] for call in route.calls]
    assert cursors == ["*", "AoE1", "AoE2"]
    assert route.calls[0].request.url.params["resultType"] == "core"


@respx.mock
async def test_europepmc_max_records(no_wait_limiter: AsyncRateLimiter, no_sleep: Sleep) -> None:
    respx.get(f"{EUROPEPMC_BASE}search").mock(
        return_value=httpx.Response(
            200,
            json={
                "hitCount": 500,
                "nextCursorMark": "n",
                "resultList": {"result": [EPMC_RESULT, EPMC_RESULT]},
            },
        )
    )
    async with EuropePMCClient(rate_limiter=no_wait_limiter, sleep=no_sleep) as c:
        _, records = await c.search("x", max_records=1)
    assert len(records) == 1


# --------------------------------------------------------------------------- OpenAlex

OPENALEX_WORK = {
    "id": "https://openalex.org/W1234567890",
    "doi": "https://doi.org/10.5555/synth.0001",
    "title": "Effect of drug X on HbA1c: a randomised controlled trial",
    "display_name": "Effect of drug X on HbA1c: a randomised controlled trial",
    "publication_year": 2021,
    "type": "article",
    "language": "en",
    "ids": {
        "openalex": "https://openalex.org/W1234567890",
        "doi": "https://doi.org/10.5555/synth.0001",
        "pmid": "https://pubmed.ncbi.nlm.nih.gov/90000001",
        "pmcid": "https://www.ncbi.nlm.nih.gov/pmc/articles/9000001",
    },
    "primary_location": {"source": {"display_name": "Journal of Synthetic Medicine"}},
    "authorships": [
        {
            "author": {
                "display_name": "Ana Synthetica",
                "orcid": "https://orcid.org/0000-0002-1825-009X",
            },
            "institutions": [{"display_name": "Fictional University"}],
            "raw_affiliation_strings": ["Dept of Endocrinology, Fictional University Hospital"],
        },
        {"author": {"display_name": "Bruno Placebo"}, "institutions": [{"display_name": "IIR"}]},
        {"author": {}},
    ],
    "abstract_inverted_index": {
        "Drug": [0],
        "X": [1, 4],
        "lowers": [2],
        "HbA1c;": [3],
        "works.": [5],
    },
}


def test_reconstruct_abstract() -> None:
    assert reconstruct_abstract(OPENALEX_WORK["abstract_inverted_index"]) == (  # type: ignore[arg-type]
        "Drug X lowers HbA1c; X works."
    )
    assert reconstruct_abstract(None) is None
    assert reconstruct_abstract({}) is None


@respx.mock
async def test_openalex_search_cursor_paging(
    no_wait_limiter: AsyncRateLimiter, no_sleep: Sleep
) -> None:
    route = respx.get(f"{OPENALEX_BASE}works").mock(
        side_effect=[
            httpx.Response(
                200, json={"meta": {"count": 2, "next_cursor": "c2"}, "results": [OPENALEX_WORK]}
            ),
            httpx.Response(
                200,
                json={
                    "meta": {"count": 2, "next_cursor": "c3"},
                    "results": [{"id": "W9", "title": None}],
                },
            ),
            httpx.Response(200, json={"meta": {"count": 2, "next_cursor": None}, "results": []}),
        ]
    )
    async with OpenAlexClient(
        mailto="t@example.org", rate_limiter=no_wait_limiter, sleep=no_sleep
    ) as c:
        total, records = await c.search(search="drug x", filter_="publication_year:2015-2024")
    assert total == 2
    assert len(records) == 1
    r = records[0]
    assert r.source_database == SourceDatabase.OPENALEX
    assert r.openalex_id == "W1234567890"
    assert (r.doi, r.pmid, r.pmcid) == ("10.5555/synth.0001", "90000001", "PMC9000001")
    assert r.abstract == "Drug X lowers HbA1c; X works."
    assert r.journal == "Journal of Synthetic Medicine"
    assert [a.last_name for a in r.authors] == ["Synthetica", "Placebo"]
    assert all(a.name_parsed_heuristically for a in r.authors)
    assert r.authors[0].affiliations == ["Dept of Endocrinology, Fictional University Hospital"]
    assert r.authors[1].affiliations == ["IIR"]
    params = route.calls[0].request.url.params
    assert params["mailto"] == "t@example.org"
    assert params["cursor"] == "*"
    assert params["per-page"] == "200"
    assert params["filter"] == "publication_year:2015-2024"
    assert route.calls[1].request.url.params["cursor"] == "c2"


async def test_openalex_requires_query() -> None:
    async with OpenAlexClient() as c:
        with pytest.raises(ValueError, match="search"):
            await c.search()


@respx.mock
async def test_openalex_get_by_doi(no_wait_limiter: AsyncRateLimiter, no_sleep: Sleep) -> None:
    respx.get(f"{OPENALEX_BASE}works/https://doi.org/10.5555/synth.0001").mock(
        return_value=httpx.Response(200, json=OPENALEX_WORK)
    )
    respx.get(f"{OPENALEX_BASE}works/https://doi.org/10.5555/missing").mock(
        return_value=httpx.Response(404)
    )
    async with OpenAlexClient(api_key="k", rate_limiter=no_wait_limiter, sleep=no_sleep) as c:
        rec = await c.get_work_by_doi("10.5555/SYNTH.0001")
        assert rec is not None and rec.pmid == "90000001"
        assert await c.get_work_by_doi("10.5555/missing") is None


# --------------------------------------------------------------------------- Unpaywall

UNPAYWALL_JSON = {
    "doi": "10.5555/synth.0001",
    "is_oa": True,
    "oa_status": "green",
    "best_oa_location": {
        "url": "https://repo.example.test/landing",
        "url_for_pdf": "https://repo.example.test/paper.pdf",
        "host_type": "repository",
        "license": "cc-by",
        "version": "acceptedVersion",
    },
    "oa_locations": [
        {"url_for_pdf": "https://repo.example.test/paper.pdf"},
        {"url_for_pdf": "https://mirror.example.test/paper.pdf"},
        {"url": "https://landing-only.example.test"},
    ],
}


@respx.mock
async def test_unpaywall_lookup(no_wait_limiter: AsyncRateLimiter, no_sleep: Sleep) -> None:
    route = respx.get(f"{UNPAYWALL_BASE}10.5555/synth.0001").mock(
        return_value=httpx.Response(200, json=UNPAYWALL_JSON)
    )
    respx.get(f"{UNPAYWALL_BASE}10.5555/unknown").mock(return_value=httpx.Response(404))
    async with UnpaywallClient(email="t@x.org", rate_limiter=no_wait_limiter, sleep=no_sleep) as c:
        res = await c.lookup("https://doi.org/10.5555/SYNTH.0001")
        assert res is not None and res.is_oa and res.oa_status == "green"
        assert res.pdf_candidates == [
            "https://repo.example.test/paper.pdf",
            "https://mirror.example.test/paper.pdf",
        ]
        assert await c.lookup("10.5555/unknown") is None
    assert route.calls[0].request.url.params["email"] == "t@x.org"


def test_unpaywall_requires_email() -> None:
    with pytest.raises(ValueError, match="email"):
        UnpaywallClient(email="")


@respx.mock
async def test_unpaywall_download_validates_pdf(
    tmp_path: Path, no_wait_limiter: AsyncRateLimiter, no_sleep: Sleep
) -> None:
    respx.get("https://repo.example.test/paper.pdf").mock(
        return_value=httpx.Response(200, content=b"%PDF-1.7\n%synthetic\n%%EOF")
    )
    respx.get("https://repo.example.test/landing").mock(
        return_value=httpx.Response(200, content=b"<html>Sign in</html>")
    )
    async with UnpaywallClient(email="t@x.org", rate_limiter=no_wait_limiter, sleep=no_sleep) as c:
        pdf = await c.download_pdf("https://repo.example.test/paper.pdf", tmp_path / "a" / "x.pdf")
        assert pdf.path.read_bytes().startswith(b"%PDF-")
        assert len(pdf.sha256) == 64
        with pytest.raises(APIClientError, match="not a PDF"):
            await c.download_pdf("https://repo.example.test/landing", tmp_path / "y.pdf")
    assert not (tmp_path / "y.pdf").exists()


# --------------------------------------------------------------------------- Crossref


@respx.mock
async def test_crossref_get_work(no_wait_limiter: AsyncRateLimiter, no_sleep: Sleep) -> None:
    respx.get("https://api.crossref.org/works/10.5555/synth.0001").mock(
        return_value=httpx.Response(
            200,
            json={
                "status": "ok",
                "message": {
                    "DOI": "10.5555/SYNTH.0001",
                    "title": ["Effect of drug X"],
                    "container-title": ["Journal of Synthetic Medicine"],
                    "author": [{"family": "Synthetica", "given": "Ana"}],
                    "issued": {"date-parts": [[2021, 3]]},
                    "type": "journal-article",
                },
            },
        )
    )
    respx.get("https://api.crossref.org/works/10.5555/nope").mock(return_value=httpx.Response(404))
    async with CrossrefClient(mailto="t@x.org", rate_limiter=no_wait_limiter, sleep=no_sleep) as c:
        work = await c.get_work("10.5555/SYNTH.0001")
        assert work is not None
        assert (work.doi, work.year, work.first_author_family) == (
            "10.5555/synth.0001",
            2021,
            "Synthetica",
        )
        assert await c.get_work("10.5555/nope") is None
