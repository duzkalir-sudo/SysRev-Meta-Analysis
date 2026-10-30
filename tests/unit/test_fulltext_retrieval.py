"""Session 2: Stage 3 retrieval, downloads, uploads, identity matching, manifest, resume.

All HTTP is respx-mocked (no live calls); records and PDFs are SYNTHETIC.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from pathlib import Path

import httpx
import pytest
import respx
from sqlalchemy.orm import Session

from src.apis.europepmc import EUROPEPMC_BASE, EuropePMCClient
from src.apis.rate_limit import AsyncRateLimiter
from src.apis.unpaywall import UNPAYWALL_BASE, UnpaywallClient
from src.core.enums import (
    FullTextStatus,
    Phase,
    ScreeningStage,
    ScreeningVerdict,
    UploadMatchStatus,
)
from src.core.exceptions import SysRevError
from src.fulltext import (
    DocumentStore,
    DownloadLimits,
    FullTextRetriever,
    PDFDownloader,
    UploadRejectedError,
    assess_match,
    build_manifest,
    confirm_pdf_match,
    manifest_csv,
    mark_unobtainable,
    register_upload,
    report_flow,
)
from src.pdf_extractor import parse_pdf
from src.screening.llm_screening import adjudicate
from src.state.db import init_db, make_engine, make_session_factory
from src.state.machine import PipelineStateMachine
from src.state.repository import Repository
from tests.conftest import make_author, make_record
from tests.fakes import approved_protocol, enroll, include_at_title_abstract
from tests.fixtures.pdf_factory import SYNTHETIC_DOI, SYNTHETIC_TITLE, synthetic_article

Sleep = Callable[[float], Awaitable[None]]
PID = "p1"
PDF = synthetic_article()
OTHER_PDF = synthetic_article(
    title="An unrelated synthetic paper about something else", doi="10.5555/other.9999"
)


async def public_dns(host: str) -> list[str]:
    """Offline stand-in for DNS: every synthetic test host is a public address."""
    return ["93.184.215.14"]


def records() -> list[object]:
    return [
        make_record(
            "r1",
            SYNTHETIC_TITLE,
            doi=SYNTHETIC_DOI,
            pmid="90000001",
            authors=[make_author("Doe", "Alex")],
        ),
        make_record("r2", "Second synthetic report", doi="10.5555/second", pmid="90000002"),
        make_record("r3", "Third synthetic report without identifiers"),
    ]


def setup_project(repo: Repository) -> None:
    repo.create_project(PID, "Synthetic review")
    repo.save_protocol(PID, approved_protocol())
    recs = records()
    repo.save_records(PID, recs)  # type: ignore[arg-type]
    include_at_title_abstract(repo, PID, recs)  # type: ignore[arg-type]


@pytest.fixture
def repo(session: Session) -> Repository:
    r = Repository(session)
    setup_project(r)
    return r


def clients(
    limiter: AsyncRateLimiter, sleep: Sleep, store: DocumentStore
) -> tuple[EuropePMCClient, UnpaywallClient, PDFDownloader]:
    http = httpx.AsyncClient()
    common = {"rate_limiter": limiter, "http_client": http, "max_retries": 0, "sleep": sleep}
    return (
        EuropePMCClient(**common),  # type: ignore[arg-type]
        UnpaywallClient(email="test@example.org", **common),  # type: ignore[arg-type]
        PDFDownloader(
            store,
            http_client=http,
            limits=DownloadLimits(max_bytes=2_000_000),
            resolver=public_dns,
        ),
    )


def mock_services(pdf_route: httpx.Response | Exception | None = None) -> None:
    pdf_route = pdf_route or httpx.Response(200, content=PDF)

    def epmc_search(request: httpx.Request) -> httpx.Response:
        pmid = request.url.params["query"].split(":")[1].split()[0]
        hits = [{"pmid": pmid, "pmcid": "PMC1000001"}] if pmid == "90000001" else []
        return httpx.Response(200, json={"hitCount": len(hits), "resultList": {"result": hits}})

    respx.get(f"{EUROPEPMC_BASE}search").mock(side_effect=epmc_search)
    respx.get(f"{EUROPEPMC_BASE}PMC1000001/fullTextXML").mock(
        return_value=httpx.Response(
            200, content=b"<article><body><p>synthetic</p></body></article>"
        )
    )
    oa = {
        "url": "https://oa.example.org/r1.pdf",
        "url_for_pdf": "https://oa.example.org/r1.pdf",
        "host_type": "repository",
        "license": "cc-by",
        "version": "publishedVersion",
    }
    respx.get(f"{UNPAYWALL_BASE}{SYNTHETIC_DOI}").mock(
        return_value=httpx.Response(
            200,
            json={
                "doi": SYNTHETIC_DOI,
                "is_oa": True,
                "best_oa_location": oa,
                "oa_locations": [oa],
            },
        )
    )
    respx.get(f"{UNPAYWALL_BASE}10.5555/second").mock(return_value=httpx.Response(404))
    route = respx.get("https://oa.example.org/r1.pdf")
    if isinstance(pdf_route, Exception):
        route.mock(side_effect=pdf_route)
    else:
        route.mock(return_value=pdf_route)


# --------------------------------------------------------------------------- downloader


@respx.mock
@pytest.mark.parametrize(
    ("response", "failure"),
    [
        (httpx.Response(200, content=b"<html>landing page</html>"), "not_pdf"),
        (httpx.Response(403), "http_error"),
        (httpx.Response(200, content=b"%PDF-1.4\n" + b"x" * 3000), "too_large"),
        (httpx.Response(200, content=b"%PDF-1.4\ncorrupt body"), "malformed"),
    ],
)
async def test_download_failures_are_classified(
    tmp_path: Path, response: httpx.Response, failure: str
) -> None:
    store = DocumentStore(tmp_path)
    respx.get("https://oa.example.org/x.pdf").mock(return_value=response)
    dl = PDFDownloader(store, limits=DownloadLimits(max_bytes=2000), resolver=public_dns)
    result = await dl.fetch("https://oa.example.org/x.pdf")
    await dl.aclose()
    assert not result.ok and result.failure == failure
    assert list(store.root.glob("*.pdf")) == []  # nothing half-written is kept
    assert list(store.partial_dir.glob("*.part")) == []


@respx.mock
async def test_download_success_network_error_and_invalid_url(tmp_path: Path) -> None:
    store = DocumentStore(tmp_path)
    respx.get("https://oa.example.org/ok.pdf").mock(return_value=httpx.Response(200, content=PDF))
    respx.get("https://oa.example.org/down.pdf").mock(side_effect=httpx.ConnectError("boom"))
    dl = PDFDownloader(store, resolver=public_dns)
    ok = await dl.fetch("https://oa.example.org/ok.pdf")
    assert ok.ok and ok.stored is not None
    assert ok.stored.path.read_bytes() == PDF and ok.stored.filename == f"{ok.stored.sha256}.pdf"
    down = await dl.fetch("https://oa.example.org/down.pdf")
    assert down.failure == "network_error"
    assert (await dl.fetch("file:///etc/passwd")).failure == "invalid_url"
    await dl.aclose()


def test_store_detects_tampering_and_cleans_partials(tmp_path: Path) -> None:
    store = DocumentStore(tmp_path)
    stored = store.put_bytes(PDF, ".pdf")
    assert store.put_bytes(PDF, ".pdf").path == stored.path  # identical bytes stored once
    (store.new_partial()).write_bytes(b"interrupted")
    assert store.cleanup_partials() == 1
    stored.path.write_bytes(PDF + b"tampered")
    with pytest.raises(SysRevError, match="hash"):
        store.read_verified(stored.filename, stored.sha256)


# --------------------------------------------------------------------------- matching


def test_match_assessment() -> None:
    doc = parse_pdf(PDF, filename="a.pdf")
    rec = make_record("r1", SYNTHETIC_TITLE, doi=SYNTHETIC_DOI)
    assert assess_match(rec, doc).status == UploadMatchStatus.CONFIRMED
    wrong = make_record("r2", "Completely different topic", doi="10.5555/zzz")
    result = assess_match(wrong, doc)
    assert result.status == UploadMatchStatus.MISMATCH and result.other_dois == (SYNTHETIC_DOI,)
    no_doi = make_record("r3", SYNTHETIC_TITLE, authors=[make_author("Roe", "B")])
    assert assess_match(no_doi, doc).status == UploadMatchStatus.CONFIRMED
    similar = make_record("r4", "Synthetic radiosurgery cohort", authors=[make_author("Zed", "Q")])
    assert assess_match(similar, doc).status in (
        UploadMatchStatus.UNCERTAIN,
        UploadMatchStatus.CONFIRMED,
    )


# --------------------------------------------------------------------------- retrieval


@respx.mock
async def test_retrieval_prefers_xml_and_records_provenance(
    repo: Repository, tmp_path: Path, no_wait_limiter: AsyncRateLimiter, no_sleep: Sleep
) -> None:
    mock_services()
    store = DocumentStore(tmp_path)
    epmc, upw, dl = clients(no_wait_limiter, no_sleep, store)
    summary = await FullTextRetriever(
        repo, PID, store, downloader=dl, europepmc=epmc, unpaywall=upw
    ).run()
    assert summary.pdf == ["r1"] and sorted(summary.missing) == ["r2", "r3"]
    item = repo.get_fulltext_item(PID, "r1")
    assert item is not None and item.status == FullTextStatus.OA_DOWNLOADED
    assert item.pmcid == "PMC1000001" and item.xml_sha256 and item.license == "cc-by"
    assert item.match_status == UploadMatchStatus.CONFIRMED
    assert item.source_url == "https://oa.example.org/r1.pdf"
    r2 = repo.get_fulltext_item(PID, "r2")
    assert r2 is not None and r2.status == FullTextStatus.MISSING
    assert {a.outcome for a in r2.attempts} == {"skipped", "not_available"}
    rows = build_manifest(repo, PID)
    assert [r.record_id for r in rows] == ["r2", "r3"]
    csv_text = manifest_csv(rows)
    assert csv_text.splitlines()[0].startswith("record_id,title,authors,year,journal,doi,pmid")
    assert "10.5555/second" in csv_text and "Smith J" in csv_text


@respx.mock
async def test_network_failure_then_resume_after_restart(
    tmp_path: Path, no_wait_limiter: AsyncRateLimiter, no_sleep: Sleep
) -> None:
    url = f"sqlite:///{tmp_path / 'review.db'}"
    store = DocumentStore(tmp_path / "files")
    engine = make_engine(url)
    init_db(engine)
    with make_session_factory(engine)() as s:
        repo = Repository(s)
        setup_project(repo)
        s.commit()
        mock_services(pdf_route=httpx.ConnectError("connection reset"))
        epmc, upw, dl = clients(no_wait_limiter, no_sleep, store)
        first = await FullTextRetriever(
            repo, PID, store, downloader=dl, europepmc=epmc, unpaywall=upw
        ).run(limit=1)
        # the PDF download failed but the open-access XML was retrieved
        assert first.processed == ["r1"] and first.xml_only == ["r1"]
    engine.dispose()  # simulated application restart

    respx.reset()
    mock_services()
    engine = make_engine(url)
    with make_session_factory(engine)() as s:
        repo = Repository(s)
        item = repo.get_fulltext_item(PID, "r1")
        assert item is not None and item.status == FullTextStatus.OA_STRUCTURED_TEXT
        assert any(a.outcome == "network_error" for a in item.attempts)
        # retrieved for PRISMA, but no usable PDF: Gate 2 waits for a human decision
        assert item.is_retrieved and not item.has_usable_pdf and item.blocks_gate2
        epmc, upw, dl = clients(no_wait_limiter, no_sleep, store)
        second = await FullTextRetriever(
            repo, PID, store, downloader=dl, europepmc=epmc, unpaywall=upw
        ).run()
        assert second.pdf == ["r1"] and sorted(second.processed) == ["r1", "r2", "r3"]
        third = await FullTextRetriever(
            repo, PID, store, downloader=dl, europepmc=epmc, unpaywall=upw
        ).run(retry_missing=False)
        assert third.processed == [] and len(third.skipped) == 3
    engine.dispose()


# --------------------------------------------------------------------------- uploads


def test_upload_matching_mismatch_confirmation_and_gate2(repo: Repository, tmp_path: Path) -> None:
    store = DocumentStore(tmp_path)
    reviewer = enroll(repo)
    with pytest.raises(UploadRejectedError):
        register_upload(
            repo, store, PID, "r2", b"not a pdf", original_filename="x.pdf", reviewer=reviewer
        )
    good = register_upload(
        repo, store, PID, "r1", PDF, original_filename="doe2020.pdf", reviewer=reviewer
    )
    assert good.match_status == UploadMatchStatus.CONFIRMED and good.recorded_by == "Dr Human"
    wrong = register_upload(
        repo, store, PID, "r2", OTHER_PDF, original_filename="x.pdf", reviewer=reviewer
    )
    assert wrong.match_status == UploadMatchStatus.MISMATCH and wrong.blocks_gate2
    dup = register_upload(
        repo, store, PID, "r3", PDF, original_filename="copy.pdf", reviewer=reviewer
    )
    assert dup.match_status != UploadMatchStatus.CONFIRMED
    assert any("already linked to record(s) ['r1']" in n for n in dup.match_notes)
    assert repo.document_original_name(PID, good.pdf_sha256 or "") in ("doe2020.pdf", "copy.pdf")

    ctx = repo.build_guard_context(PID)
    assert ctx.fulltext_items_blocking == 2
    assert PipelineStateMachine(PID, Phase.GATE2_MISSING_FULLTEXT).check(
        Phase.STAGE4_EXTRACTION_ROB, ctx
    )
    other = enroll(repo, "Dr Second")
    repo.revoke_reviewer(other.id)
    with pytest.raises(SysRevError, match="revoked"):
        confirm_pdf_match(repo, PID, "r2", reviewer=other, note="looks fine")
    confirm_pdf_match(repo, PID, "r2", reviewer=reviewer, note="checked authors and journal")
    assert repo.build_guard_context(PID).fulltext_items_blocking == 1


def test_unobtainable_is_not_retrieved_never_excluded(repo: Repository, tmp_path: Path) -> None:
    store = DocumentStore(tmp_path)
    reviewer = enroll(repo)
    register_upload(repo, store, PID, "r1", PDF, original_filename="a.pdf", reviewer=reviewer)
    register_upload(repo, store, PID, "r2", OTHER_PDF, original_filename="b.pdf", reviewer=reviewer)
    confirm_pdf_match(repo, PID, "r2", reviewer=reviewer, note="publisher PDF, title differs")
    with pytest.raises(SysRevError, match="already has a retrieved"):
        mark_unobtainable(repo, PID, "r1", reviewer=reviewer, reason="x")
    mark_unobtainable(
        repo, PID, "r3", reviewer=reviewer, reason="no library access; author contacted twice"
    )
    # eligibility after full-text assessment for the two retrieved reports
    adjudicate(
        repo,
        PID,
        "r1",
        ScreeningStage.FULL_TEXT,
        verdict=ScreeningVerdict.INCLUDE,
        rationale="eligible",
        reviewer=reviewer,
    )
    adjudicate(
        repo,
        PID,
        "r2",
        ScreeningStage.FULL_TEXT,
        verdict=ScreeningVerdict.EXCLUDE,
        rationale="wrong population",
        exclusion_reason="E1: Animal studies",
        reviewer=reviewer,
    )
    flow = report_flow(repo, PID)
    assert flow.sought == 3 and flow.not_retrieved == ("r3",) and flow.assessed == 2
    assert flow.excluded_by_reason == {"E1: Animal studies": 1}
    assert flow.reports_included == 1 and flow.awaiting_retrieval == ()
    assert not any(
        d.record_id == "r3"
        for d in repo.load_screening_decisions(PID)
        if d.stage == ScreeningStage.FULL_TEXT
    )
    assert repo.build_guard_context(PID).fulltext_items_blocking == 0
    assert json.loads(json.dumps(flow.excluded_by_reason))
