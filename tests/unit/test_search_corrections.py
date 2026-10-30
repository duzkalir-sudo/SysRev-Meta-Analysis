"""PR #10 review corrections: effective search status, vendor-count reconciliation,
restriction provenance, plus defects found while verifying them.

Simulated tests only: mocked HTTP (respx) and synthetic RIS exports generated below
(10.5555 test-prefix DOIs; invented titles). Not real vendor exports.
"""

from __future__ import annotations

import asyncio
import csv
import io
from datetime import date
from pathlib import Path

import httpx
import pytest
import respx
from sqlalchemy.orm import Session

from src.apis.pubmed import EUTILS_BASE
from src.core.enums import SearchRunKind
from src.fulltext.storage import DocumentStore
from src.search.identification import strategy_state
from src.search.reporting import report_csv, report_markdown, search_report
from src.search.service import SearchError, SearchService
from src.state.db import ReviewerORM
from src.state.repository import Repository
from tests.fakes import enroll
from tests.search_helpers import FakePubMed, approved_in_stage2, fast_clients

PID = "corr"
DAY = date(2026, 10, 1)


def ris(n: int, tag: str, *, year: int = 2015) -> bytes:
    """``n`` synthetic RIS entries with distinct titles (and distinct test DOIs)."""
    entries = [
        f"TY  - JOUR\nTI  - Synthetic {tag} report number {i} on drug X\nAU  - Fict{tag}, A\n"
        f"PY  - {year}\nDO  - 10.5555/{tag.lower()}.{i}\nAN  - {tag}{i:05d}\nER  - \n"
        for i in range(n)
    ]
    return "\n".join(entries).encode()


@pytest.fixture
def repo(session: Session) -> Repository:
    r = Repository(session)
    r.create_project(PID, "Synthetic correction review")
    return r


@pytest.fixture
def human(repo: Repository) -> ReviewerORM:
    return enroll(repo, "Dr Human")


def service(repo: Repository, tmp_path: Path) -> SearchService:
    return SearchService(
        repo, PID, store=DocumentStore(tmp_path / "searches"), clients=fast_clients()
    )


def sha(repo: Repository, db: str) -> str:
    protocol = repo.latest_protocol(PID)
    assert protocol is not None
    return next(s.sha256 for s in protocol.search_strategies if s.database == db)


def csv_rows(report: dict[str, object]) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(report_csv(report))))  # type: ignore[arg-type]


def problems_for(repo: Repository, db: str) -> list[str]:
    return [p for p in repo.search_identification_problems(PID) if p.startswith(db)]


# ====================================================================== 1. effective status
def test_not_searched_then_successful_import_supersedes_declaration(
    repo: Repository, human: ReviewerORM, tmp_path: Path
) -> None:
    approved_in_stage2(repo, PID, human)
    svc = service(repo, tmp_path)
    emb = sha(repo, "embase")
    first = svc.record_not_searched(emb, reason="licence expired", reviewer=human)
    second = svc.record_not_searched(emb, reason="licence expired (re-checked)", reviewer=human)
    state = strategy_state(repo.load_search_runs(PID), emb)
    assert state.status == "not_searched"
    assert state.effective_declaration is not None
    assert state.effective_declaration.run_id == second.run_id  # latest declaration applies

    svc.import_file(
        emb,
        ris(3, "EMB"),
        filename="embase.ris",
        fmt="ris",
        searched_on=DAY,
        reviewer=human,
        hits_reported=3,
    )
    state = strategy_state(repo.load_search_runs(PID), emb)
    assert state.status == "searched" and len(state.declarations) == 2
    assert "2 earlier 'not searched' declaration(s) superseded" in state.detail
    assert problems_for(repo, "embase") == []
    # Both declarations stay in the append-only log.
    kinds = [r.kind for r in repo.load_search_runs(PID) if r.strategy_sha256 == emb]
    assert kinds == [SearchRunKind.NOT_SEARCHED, SearchRunKind.NOT_SEARCHED, SearchRunKind.IMPORT]

    report = search_report(repo, PID)
    assert "Embase" in report["databases_searched"]
    assert all(d["database"] != "Embase" for d in report["databases_not_searched"])
    row = next(s for s in report["strategies"] if s["database"] == "Embase")
    assert row["current_status"] == "searched" and row["not_searched_reason"] is None
    by_id = {r["note"]: r for r in csv_rows(report) if r["kind"] == "not_searched"}
    # A later success supersedes every earlier declaration (checked before replacement).
    assert by_id["licence expired"]["current_effect"] == (
        "superseded by a later successful search/import"
    )
    assert by_id["licence expired (re-checked)"]["current_effect"] == (
        "superseded by a later successful search/import"
    )
    assert {r["strategy_status"] for r in csv_rows(report) if r["database"] == "Embase"} == {
        "searched"
    }
    md = report_markdown(report)
    assert "Not searched: Embase" not in md and "Embase" in md.split("\n")[2]
    assert first.run_id != second.run_id
    # Declaring a searched strategy as not searched is refused.
    with pytest.raises(SearchError, match="already searched"):
        svc.record_not_searched(emb, reason="mistake", reviewer=human)


@respx.mock
def test_not_searched_then_failed_execution_stays_not_searched(
    repo: Repository, human: ReviewerORM, tmp_path: Path
) -> None:
    respx.get(f"{EUTILS_BASE}esearch.fcgi").mock(return_value=httpx.Response(500, text="down"))
    approved_in_stage2(repo, PID, human)
    svc = service(repo, tmp_path)
    pm = sha(repo, "pubmed")
    svc.record_not_searched(pm, reason="NCBI outage on the planned date", reviewer=human)
    failed = asyncio.run(svc.execute(pm, reviewer=human))
    assert failed.status.value == "failed" and not failed.identifies_records
    state = strategy_state(repo.load_search_runs(PID), pm)
    assert state.status == "not_searched" and state.failed_attempts == 1
    assert problems_for(repo, "pubmed") == []  # a reasoned declaration remains in effect

    report = search_report(repo, PID)
    assert "PubMed" not in report["databases_searched"]
    assert report["databases_not_searched"][0] == {
        "database": "PubMed",
        "reasons": ["NCBI outage on the planned date"],
    }
    rows = [r for r in csv_rows(report) if r["database"] == "PubMed"]
    assert {r["kind"]: r["current_effect"] for r in rows} == {
        "not_searched": "in effect",
        "execution": "failed attempt: nothing identified, not a search",
    }
    assert {r["strategy_status"] for r in rows} == {"not_searched"}
    md = report_markdown(report)
    assert "Not searched: PubMed — NCBI outage on the planned date" in md
    assert "Databases searched: none" in md


@respx.mock
def test_failed_execution_alone_is_never_reported_as_searched(
    repo: Repository, human: ReviewerORM, tmp_path: Path
) -> None:
    respx.get(f"{EUTILS_BASE}esearch.fcgi").mock(return_value=httpx.Response(400, text="bad"))
    approved_in_stage2(repo, PID, human)
    pm = sha(repo, "pubmed")
    asyncio.run(service(repo, tmp_path).execute(pm, reviewer=human))
    assert strategy_state(repo.load_search_runs(PID), pm).status == "not_yet_searched"
    assert any("1 failed attempt(s) identified nothing" in p for p in problems_for(repo, "no "))
    report = search_report(repo, PID)
    assert "PubMed" in report["databases_not_yet_searched"]
    assert "Not yet searched: " in report_markdown(report)


# ====================================================================== 2. vendor totals
def _scopus_stage2(repo: Repository, human: ReviewerORM) -> str:
    approved_in_stage2(
        repo,
        PID,
        human,
        selected_databases=["scopus"],
        search_strategies=[{"database": "scopus", "query": 'TITLE-ABS-KEY("drug x")'}],
    )
    return sha(repo, "scopus")


def test_150_reported_100_imported_blocks_readiness(
    repo: Repository, human: ReviewerORM, tmp_path: Path
) -> None:
    s = _scopus_stage2(repo, human)
    svc = service(repo, tmp_path)
    svc.import_file(
        s,
        ris(100, "SCA"),
        filename="scopus1.ris",
        fmt="ris",
        searched_on=DAY,
        reviewer=human,
        hits_reported=150,
    )
    svc.prepare_for_screening(reviewer=human)
    msg = problems_for(repo, "scopus")
    assert msg and "reported 150 records but 100 were imported" in msg[0]
    assert any("scopus" in p for p in repo.build_guard_context(PID).search_problems)
    st = search_report(repo, PID)["strategies"][0]
    assert (st["reconciliation"], st["interface_reported"], st["imported_records"]) == (
        "mismatch",
        150,
        100,
    )
    assert "150 vs 100 (mismatch)" in report_markdown(search_report(repo, PID))


def test_unknown_total_is_never_a_match_and_needs_reasoned_acknowledgement(
    repo: Repository, human: ReviewerORM, tmp_path: Path
) -> None:
    s = _scopus_stage2(repo, human)
    svc = service(repo, tmp_path)
    run = svc.import_file(
        s, ris(100, "SCB"), filename="s.ris", fmt="ris", searched_on=DAY, reviewer=human
    )
    assert run.hits_reported is None
    assert any("total reported by the vendor interface was not recorded" in w for w in run.warnings)
    # The imported count is never substituted for the database-reported total.
    assert {r.provenance.total_hits_reported for r in repo.load_records(PID)} == {None}  # type: ignore[union-attr]
    svc.prepare_for_screening(reviewer=human)
    assert "not recorded for the 100 imported records" in problems_for(repo, "scopus")[0]
    report = search_report(repo, PID)
    assert report["strategies"][0]["reconciliation"] == "unknown"
    assert "None vs 100 (unknown)" in report_markdown(report)
    csv_import = next(r for r in csv_rows(report) if r["kind"] == "import")
    assert csv_import["hits_reported"] == "" and "not recorded" in csv_import["warnings"]

    with pytest.raises(ValueError, match="requires a reason"):
        svc.acknowledge(strategy_sha256=s, note="  ", reviewer=human)
    ack = svc.acknowledge(
        strategy_sha256=s, reviewer=human, note="Scopus showed no total for this export (synthetic)"
    )
    assert ack.summary["state"] == "unknown" and ack.summary["reported"] is None
    assert problems_for(repo, "scopus") == []
    st = search_report(repo, PID)["strategies"][0]
    assert st["reconciliation"] == "unknown" and st["reconciliation_acknowledged"]
    assert "acknowledged: Scopus showed no total" in report_markdown(search_report(repo, PID))


def test_multiple_files_reconcile_to_combined_total(
    repo: Repository, human: ReviewerORM, tmp_path: Path
) -> None:
    s = _scopus_stage2(repo, human)
    svc = service(repo, tmp_path)
    svc.import_file(
        s,
        ris(100, "SCC"),
        filename="part1.ris",
        fmt="ris",
        searched_on=DAY,
        reviewer=human,
        hits_reported=150,
    )
    svc.import_file(
        s, ris(50, "SCD"), filename="part2.ris", fmt="ris", searched_on=DAY, reviewer=human
    )  # total already recorded with part 1
    svc.prepare_for_screening(reviewer=human)
    assert problems_for(repo, "scopus") == []
    st = search_report(repo, PID)["strategies"][0]
    assert (st["reconciliation"], st["import_files"], st["imported_records"]) == ("matched", 2, 150)
    with pytest.raises(SearchError, match="reconciled"):
        svc.acknowledge(strategy_sha256=s, note="nothing to explain", reviewer=human)
    assert repo.load_prisma(PID).identified_by_database["scopus"] == 150  # type: ignore[union-attr,index]


def test_acknowledgement_is_invalidated_by_a_further_import(
    repo: Repository, human: ReviewerORM, tmp_path: Path
) -> None:
    s = _scopus_stage2(repo, human)
    svc = service(repo, tmp_path)
    svc.import_file(
        s,
        ris(100, "SCE"),
        filename="a.ris",
        fmt="ris",
        searched_on=DAY,
        reviewer=human,
        hits_reported=150,
    )
    ack = svc.acknowledge(strategy_sha256=s, note="export limit (synthetic)", reviewer=human)
    svc.prepare_for_screening(reviewer=human)
    assert problems_for(repo, "scopus") == []
    svc.import_file(
        s,
        ris(30, "SCF"),
        filename="b.ris",
        fmt="ris",
        searched_on=DAY,
        reviewer=human,
        hits_reported=150,
    )
    svc.prepare_for_screening(reviewer=human)
    msg = problems_for(repo, "scopus")
    assert msg and "reported 150 records but 130 were imported" in msg[0]
    effects = {
        r["note"]: r["current_effect"]
        for r in csv_rows(search_report(repo, PID))
        if r["kind"] == "acknowledgement"
    }
    assert effects["export limit (synthetic)"] == "no longer applies (imports changed after it)"
    svc.import_file(s, ris(20, "SCG"), filename="c.ris", fmt="ris", searched_on=DAY, reviewer=human)
    svc.prepare_for_screening(reviewer=human)
    assert problems_for(repo, "scopus") == []  # 150 of 150 imported
    assert ack.summary["records"] == 100


def test_conflicting_declared_totals_need_acknowledgement(
    repo: Repository, human: ReviewerORM, tmp_path: Path
) -> None:
    s = _scopus_stage2(repo, human)
    svc = service(repo, tmp_path)
    svc.import_file(
        s,
        ris(10, "SCH"),
        filename="a.ris",
        fmt="ris",
        searched_on=DAY,
        reviewer=human,
        hits_reported=20,
    )
    svc.import_file(
        s,
        ris(10, "SCI"),
        filename="b.ris",
        fmt="ris",
        searched_on=DAY,
        reviewer=human,
        hits_reported=21,
    )
    svc.prepare_for_screening(reviewer=human)
    assert (
        "different totals were declared for the imports (20, 21)"
        in (problems_for(repo, "scopus")[0])
    )
    st = search_report(repo, PID)["strategies"][0]
    assert st["reconciliation"] == "conflicting" and st["interface_reported"] is None


# ====================================================================== 3. restrictions
def test_manual_import_preserves_protocol_restrictions_and_their_application(
    repo: Repository, human: ReviewerORM, tmp_path: Path
) -> None:
    query = "'drug x'/exp AND 'randomized controlled trial'/de"
    approved_in_stage2(
        repo,
        PID,
        human,
        selected_databases=["embase", "web_of_science"],
        search_strategies=[
            {
                "database": "embase",
                "platform": "Embase.com",
                "query": query,
                "date_from": "2010-01-01",
                "date_to": "2020-12-31",
                "languages": ["English"],
                "filters": ["[article]/lim", "[humans]/lim"],
            },
            {
                "database": "web_of_science",
                "platform": "Web of Science Core Collection",
                "query": 'TS=("drug x")',
                "filters": ["Document Types: Article"],
            },
        ],
    )
    svc = service(repo, tmp_path)
    emb, wos = sha(repo, "embase"), sha(repo, "web_of_science")
    run = svc.import_file(
        emb,
        ris(3, "EMR", year=2015) + b"\n" + ris(1, "EMO", year=2005),
        filename="embase.ris",
        fmt="ris",
        searched_on=DAY,
        reviewer=human,
        hits_reported=4,
        restrictions_confirmed=True,
        restrictions_note="Embase.com Limits panel: Article, Humans, English, 2010-2020",
    )
    svc.import_file(
        wos,
        ris(2, "WOS"),
        filename="wos.ris",
        fmt="ris",
        searched_on=DAY,
        reviewer=human,
        hits_reported=2,
    )
    stored = {r.run_id: r for r in repo.load_search_runs(PID)}[run.run_id]  # round trip
    assert stored.query == query  # never rewritten
    assert stored.platform == "Embase.com" and stored.searched_on == DAY
    assert stored.protocol_version == 1 and stored.strategy_sha256 == emb
    assert stored.protocol_restrictions == {
        "date_from": "2010-01-01",
        "date_to": "2020-12-31",
        "languages": ["English"],
        "filters": ["[article]/lim", "[humans]/lim"],
    }
    assert stored.restrictions_applied == "confirmed_by_reviewer"
    assert stored.parameters == {}  # nothing was sent by the system for a manual search
    # Record-level evidence check: one record lies outside the protocol date limits.
    assert any("1 imported record(s) have a publication year outside" in w for w in stored.warnings)

    report = search_report(repo, PID)
    rows = {r["database"]: r for r in csv_rows(report) if r["kind"] == "import"}
    e = rows["Embase"]
    assert e["platform"] == "Embase.com" and e["query_as_run"] == query
    assert e["protocol_date_limits"] == "2010-01-01 to 2020-12-31"
    assert e["protocol_languages"] == "English"
    assert e["protocol_filters"] == "[article]/lim | [humans]/lim"
    assert e["restrictions_applied"].startswith("confirmed by the reviewer")
    assert "Limits panel" in e["restrictions_note"]
    assert e["searched_on"] == DAY.isoformat() and e["strategy_sha256"] == emb
    assert e["protocol_version"] == "1"
    w = rows["Web of Science"]
    assert w["protocol_filters"] == "Document Types: Article"
    assert w["restrictions_applied"] == "NOT confirmed as applied in the vendor interface"
    md = report_markdown(report)
    assert "Protocol filters (separate from the query): [article]/lim | [humans]/lim" in md
    assert "Protocol language limits: English" in md
    assert "Restrictions applied: NOT confirmed as applied in the vendor interface" in md
    assert "Restrictions applied: confirmed by the reviewer" in md


@respx.mock
def test_api_execution_records_restrictions_applied_by_system(
    repo: Repository, human: ReviewerORM, tmp_path: Path
) -> None:
    fake = FakePubMed({"97000001": date(2012, 1, 1)})
    respx.get(f"{EUTILS_BASE}esearch.fcgi").mock(side_effect=fake.esearch)
    respx.post(f"{EUTILS_BASE}efetch.fcgi").mock(side_effect=fake.efetch)
    approved_in_stage2(
        repo,
        PID,
        human,
        selected_databases=["pubmed"],
        search_strategies=[
            {
                "database": "pubmed",
                "query": "drug x",
                "filters": ["humans[mh]"],
                "date_from": "2010-01-01",
            }
        ],
    )
    run = asyncio.run(service(repo, tmp_path).execute(sha(repo, "pubmed"), reviewer=human))
    assert run.restrictions_applied == "applied_by_system"
    assert run.query == "(drug x) AND (humans[mh])" and run.parameters["mindate"] == "2010/01/01"
    assert run.protocol_restrictions["filters"] == ["humans[mh]"]


# ====================================================================== 4. further defects
def test_zero_result_manual_search_is_searched_not_not_searched(
    repo: Repository, human: ReviewerORM, tmp_path: Path
) -> None:
    approved_in_stage2(repo, PID, human)
    svc = service(repo, tmp_path)
    ct = sha(repo, "clinicaltrials_gov")
    with pytest.raises(ValueError, match="attestation"):
        svc.record_zero_results(ct, searched_on=DAY, note=" ", reviewer=human)
    svc.record_zero_results(
        ct, searched_on=DAY, note="0 studies found in ClinicalTrials.gov", reviewer=human
    )
    state = strategy_state(repo.load_search_runs(PID), ct)
    assert state.status == "searched" and state.imports.state == "matched"
    assert problems_for(repo, "clinicaltrials_gov") == []
    assert "ClinicalTrials.gov" in search_report(repo, PID)["databases_searched"]
    assert repo.load_records(PID) == []


def test_reprepare_after_duplicate_decision_keeps_decision_and_records(
    repo: Repository, human: ReviewerORM, tmp_path: Path
) -> None:
    """Formerly a deadlock: a further import after a human duplicate decision made
    re-preparation impossible while the guard demanded it."""
    approved_in_stage2(
        repo,
        PID,
        human,
        selected_databases=["embase", "scopus"],
        search_strategies=[
            {"database": "embase", "query": "x"},
            {"database": "scopus", "query": "y"},
        ],
    )
    svc = service(repo, tmp_path)
    shared = (
        b"TY  - JOUR\nTI  - Alpha synthetic trial of drug X\nAU  - Aa, A\nPY  - 2020\n"
        b"DO  - 10.5555/shared.1\nER  - \n"
    )
    other = (
        b"TY  - JOUR\nTI  - Unrelated synthetic paper about something else\nAU  - Bb, B\n"
        b"PY  - 2020\nDO  - 10.5555/shared.1\nER  - \n"
    )
    svc.import_file(
        sha(repo, "embase"),
        shared,
        filename="e.ris",
        fmt="ris",
        searched_on=DAY,
        reviewer=human,
        hits_reported=1,
    )
    svc.import_file(
        sha(repo, "scopus"),
        other,
        filename="s1.ris",
        fmt="ris",
        searched_on=DAY,
        reviewer=human,
        hits_reported=2,
    )
    svc.prepare_for_screening(reviewer=human)
    cand = repo.load_duplicate_candidates(PID)[0]
    svc.resolve_duplicate(
        cand.candidate_id,
        resolution="duplicate_keep_a",
        reviewer=human,
        note="same report; the Scopus title is a database error",
    )
    svc.import_file(
        sha(repo, "scopus"),
        ris(1, "LATE"),
        filename="s2.ris",
        fmt="ris",
        searched_on=DAY,
        reviewer=human,
    )
    prep = svc.prepare_for_screening(reviewer=human)  # previously refused -> deadlock
    assert prep.summary["human_decisions_reapplied"] == 1
    assert prep.summary["possible_duplicates_for_review"] == 0
    records = {r.record_id: r for r in repo.load_records(PID)}
    assert len(records) == 3  # nothing lost
    assert records[cand.record_id_b].duplicate_of == cand.record_id_a
    assert repo.load_prisma(PID).duplicates_removed == 1  # type: ignore[union-attr]
    assert [c.resolution for c in repo.load_duplicate_candidates(PID)] == ["duplicate_keep_a"]
    assert repo.search_identification_problems(PID) == []


def test_derived_prisma_full_flow_agrees_with_guard_cross_check(
    repo: Repository, human: ReviewerORM, tmp_path: Path
) -> None:
    """PRISMA derived from stored data must satisfy the guard's independent cross-check
    through screening, retrieval (incl. not retrieved) and full-text exclusion reasons."""
    from src.core.enums import FullTextStatus, ReviewerRole, ScreeningStage, ScreeningVerdict
    from src.core.schemas import FullTextItem, ScreeningDecision
    from src.fulltext.retrieval import mark_unobtainable
    from src.search.prisma_sync import derive_prisma

    approved_in_stage2(
        repo,
        PID,
        human,
        selected_databases=["embase"],
        search_strategies=[{"database": "embase", "query": "x"}],
    )
    svc = service(repo, tmp_path)
    svc.import_file(
        sha(repo, "embase"),
        ris(5, "PRS"),
        filename="p.ris",
        fmt="ris",
        searched_on=DAY,
        reviewer=human,
        hits_reported=5,
    )
    svc.prepare_for_screening(reviewer=human)
    ids = [r.record_id for r in repo.load_records(PID, unique_only=True)]

    def decide(
        rid: str, stage: ScreeningStage, verdict: ScreeningVerdict, reason: str | None = None
    ) -> None:
        for role in (ReviewerRole.REVIEWER_1, ReviewerRole.REVIEWER_2):
            repo.save_screening_decision(
                PID,
                ScreeningDecision(
                    record_id=rid,
                    reviewer=role,
                    reviewer_identity=f"llm-{role}",
                    stage=stage,
                    verdict=verdict,
                    rationale="synthetic",
                    exclusion_reason=reason,
                ),
            )

    ta, ft = ScreeningStage.TITLE_ABSTRACT, ScreeningStage.FULL_TEXT
    inc, exc = ScreeningVerdict.INCLUDE, ScreeningVerdict.EXCLUDE
    decide(ids[0], ta, exc, "wrong population")
    for rid in ids[1:]:
        decide(rid, ta, inc)
    for rid in ids[1:4]:
        repo.save_fulltext_item(
            PID,
            FullTextItem(
                record_id=rid,
                title="t",
                status=FullTextStatus.USER_UPLOADED,
                pdf_filename=f"{rid}.pdf",
                pdf_sha256="a" * 64,
                recorded_by="Dr Human",
                recorded_by_reviewer_id=human.id,
            ),
        )
    repo.save_fulltext_item(PID, FullTextItem(record_id=ids[4], title="t"))
    mark_unobtainable(repo, PID, ids[4], reviewer=human, reason="no library access")
    decide(ids[1], ft, inc)
    decide(ids[2], ft, exc, "no comparator")
    decide(ids[3], ft, exc, "no comparator")

    counts = derive_prisma(repo, PID)
    assert counts.identified_by_database == {"embase": 5}
    assert (counts.records_screened, counts.records_excluded, counts.reports_sought) == (5, 1, 4)
    assert (counts.reports_not_retrieved, counts.reports_assessed) == (1, 3)
    assert counts.reports_excluded_by_reason == {"no comparator": 2}
    assert (counts.reports_included, counts.studies_included) == (1, 1)
    assert counts.is_final
    repo.save_prisma(PID, counts)
    assert repo.build_guard_context(PID).prisma_violations == ()


def test_zero_result_endpoint_requires_token_and_records_search(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from src.api_server.app import create_app

    app = create_app("sqlite:///:memory:", data_dir=tmp_path, search_clients=fast_clients())
    with TestClient(app) as client, app.state.session_factory() as s:
        repo = Repository(s)
        repo.create_project(PID, "HTTP zero")
        reviewer = enroll(repo, "Dr Zero")
        _, token = repo.enroll_reviewer("Dr Token")
        approved_in_stage2(repo, PID, reviewer)
        ct = sha(repo, "clinicaltrials_gov")
        s.commit()
        url = f"/projects/{PID}/searches/strategies/{ct}/zero-results"
        body = {"searched_on": "2026-10-01", "note": "0 studies (synthetic)"}
        assert client.post(url, json=body).status_code == 401
        ok = client.post(url, json=body, headers={"Authorization": f"Bearer {token}"})
        assert ok.status_code == 201 and ok.json()["hits_reported"] == 0
        assert ok.json()["actor"] == "Dr Token"
