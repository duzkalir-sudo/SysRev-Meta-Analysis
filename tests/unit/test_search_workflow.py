"""Session 5: protocol editor, search execution/import, deduplication, PRISMA, guards.

All HTTP traffic is mocked with respx; all records are SYNTHETIC.
"""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from pathlib import Path

import httpx
import pytest
import respx
from sqlalchemy.orm import Session

from src.apis.europepmc import EUROPEPMC_BASE
from src.apis.openalex import OPENALEX_BASE
from src.apis.pubmed import EUTILS_BASE
from src.core.enums import (
    Phase,
    ReviewerRole,
    ScreeningStage,
    ScreeningVerdict,
    SearchRunKind,
    SearchRunStatus,
    SourceDatabase,
)
from src.core.exceptions import GuardViolationError, SysRevError
from src.core.schemas import ScreeningDecision, SearchStrategy
from src.fulltext.storage import DocumentStore
from src.search import protocol_editor as editor
from src.search.importers import ImportFileError
from src.search.prisma_sync import derive_prisma
from src.search.reporting import report_csv, report_markdown, search_report
from src.search.service import SearchError, SearchService, executable_query
from src.state.db import AppendOnlyViolationError, ReviewerORM, SearchRunORM
from src.state.repository import Repository
from src.workflows.ai_steps import AIWorkflow, WorkflowStepError
from tests.fakes import enroll
from tests.search_helpers import (
    PUBMED_QUERY,
    FakePubMed,
    MeSHAlwaysFound,
    advance,
    approved_in_stage2,
    draft_data,
    fast_clients,
)

FIX = Path(__file__).parents[1] / "fixtures" / "search"
PID = "s5"


@pytest.fixture
def repo(session: Session) -> Repository:
    r = Repository(session)
    r.create_project(PID, "Synthetic Session 5 review")
    return r


@pytest.fixture
def human(repo: Repository) -> ReviewerORM:
    return enroll(repo, "Dr Human")


def svc(repo: Repository, tmp_path: Path) -> SearchService:
    return SearchService(
        repo, PID, store=DocumentStore(tmp_path / "searches"), clients=fast_clients()
    )


def strategy(repo: Repository, db: str) -> SearchStrategy:
    protocol = repo.latest_protocol(PID)
    assert protocol is not None
    return next(s for s in protocol.search_strategies if s.database == db)


# ---------------------------------------------------------------------- protocol editor
def test_draft_ignores_client_supplied_approval_and_verification(
    repo: Repository, human: ReviewerORM
) -> None:
    data = draft_data(status="approved", approved_by="Someone", version=7)
    data["mesh_terms"] = [
        {"descriptor": "Diabetes Mellitus", "verified": True, "verified_at": "2026-01-01T00:00:00Z"}
    ]
    data["search_strategies"][0] |= {"syntax_validated": True, "validation_method": "api_preview"}
    result = editor.save_draft(repo, PID, data, reviewer=human)
    p = result.protocol
    assert (p.status, p.version, p.approved_by) == ("draft", 1, None)
    assert not p.mesh_terms[0].verified
    assert not any(s.syntax_validated for s in p.search_strategies)
    assert "not syntax-validated" in "; ".join(p.approval_blockers())
    with pytest.raises(editor.ProtocolEditError, match="cannot be approved"):
        editor.approve(repo, PID, reviewer=human)


def test_invalid_drafts_are_refused(repo: Repository, human: ReviewerORM) -> None:
    with pytest.raises(ValueError, match="comparator"):
        editor.save_draft(repo, PID, draft_data(comparator=None), reviewer=human)
    bad = draft_data()
    bad["outcomes"][0]["preferred_effect_measure"] = "PROPORTION"
    with pytest.raises(ValueError, match="invalid for"):
        editor.save_draft(repo, PID, bad, reviewer=human)
    assert repo.latest_protocol(PID) is None


def test_verification_survives_only_unchanged_items_and_amendment_voids_approval(
    repo: Repository, human: ReviewerORM
) -> None:
    editor.save_draft(repo, PID, draft_data(), reviewer=human)
    checker = MeSHAlwaysFound(missing={"Not A Heading"})
    data = draft_data(
        mesh_terms=[{"descriptor": "Diabetes Mellitus"}, {"descriptor": "Not A Heading"}]
    )
    editor.save_draft(repo, PID, data, reviewer=human)
    results = asyncio.run(editor.verify_mesh(repo, PID, checker, reviewer=human))
    assert results == {"Diabetes Mellitus": True, "Not A Heading": False}
    editor.save_draft(repo, PID, draft_data(), reviewer=human)  # drop the unknown heading
    for s in repo.latest_protocol(PID).search_strategies:  # type: ignore[union-attr]
        editor.record_strategy_validation(
            repo, PID, s.sha256, method="human_attestation", reviewer=human, note="checked"
        )
    approved = editor.approve(repo, PID, reviewer=human)
    assert approved.status == "approved" and approved.approved_reviewer_id == human.id
    with pytest.raises(editor.ProtocolEditError, match="nothing changed"):
        editor.save_draft(repo, PID, draft_data(), reviewer=human)

    changed = draft_data()
    changed["search_strategies"][0]["query"] = PUBMED_QUERY + " AND english[la]"
    changed["inclusion_criteria"] = ["Randomised trials in adults", "Published after 2000"]
    result = editor.save_draft(repo, PID, changed, reviewer=human)
    assert result.new_version and result.protocol.version == 2
    assert result.material_changes == {
        "eligibility": ["inclusion_criteria"],
        "search": ["search_strategies"],
    }
    assert any("pubmed strategy" in i for i in result.invalidated)
    assert any("approval of version 1" in i for i in result.invalidated)
    p2 = result.protocol
    assert p2.mesh_terms[0].verified  # unchanged descriptor keeps its NLM verification
    by_db = {s.database: s for s in p2.search_strategies}
    assert not by_db[SourceDatabase.PUBMED].syntax_validated
    assert by_db[SourceDatabase.EMBASE].syntax_validated  # unchanged strategy keeps it
    # Gate 1 refuses the unapproved amendment.
    advance(repo, PID, Phase.GATE1_SEARCH_STRATEGY, human)
    blockers = repo.build_guard_context(PID).protocol_blockers
    assert any("not been approved by an authenticated" in b for b in blockers)
    actions = [e.action for e, _ in repo.list_protocol_events(PID)]
    assert actions[0] == "draft_saved" and "approved" in actions and "mesh_verified" in actions
    assert [p.status for p in repo.list_protocols(PID)] == ["approved", "draft"]


def test_protocol_locked_after_gate1_and_events_append_only(
    repo: Repository, human: ReviewerORM, session: Session
) -> None:
    approved_in_stage2(repo, PID, human)
    assert repo.get_project(PID).phase == Phase.STAGE2_SEARCH_SCREENING
    with pytest.raises(editor.ProtocolEditError, match="locked"):
        editor.save_draft(repo, PID, draft_data(population="Children"), reviewer=human)
    approval = next(e for e, _ in repo.list_protocol_events(PID) if e.action == "approved")
    assert approval.reviewer_id == human.id


def test_revoked_reviewer_approval_does_not_release_gate1(
    repo: Repository, human: ReviewerORM
) -> None:
    editor.save_draft(repo, PID, draft_data(mesh_terms=[]), reviewer=human)
    for s in repo.latest_protocol(PID).search_strategies:  # type: ignore[union-attr]
        editor.record_strategy_validation(
            repo, PID, s.sha256, method="human_attestation", reviewer=human
        )
    editor.approve(repo, PID, reviewer=human)
    assert repo.build_guard_context(PID).protocol_blockers == ()
    repo.revoke_reviewer(human.id)
    assert repo.build_guard_context(PID).protocol_blockers


def test_selected_databases_must_match_strategies() -> None:
    from tests.conftest import make_protocol

    p = make_protocol(selected_databases=[SourceDatabase.PUBMED, SourceDatabase.SCOPUS])
    assert "selected database scopus has no search strategy" in p.approval_blockers()


# ---------------------------------------------------------------------- previews
@respx.mock
def test_stage1_preview_validates_pubmed_strategy_only_without_errors(
    repo: Repository, human: ReviewerORM, tmp_path: Path
) -> None:
    fake = FakePubMed({"90000001": date(2021, 3, 1), "90000002": date(1999, 1, 1)})
    respx.get(f"{EUTILS_BASE}esearch.fcgi").mock(side_effect=fake.esearch)
    editor.save_draft(repo, PID, draft_data(), reviewer=human)
    sha = strategy(repo, "pubmed").sha256
    run = asyncio.run(svc(repo, tmp_path).preview(sha, reviewer=human))
    assert run.kind == SearchRunKind.PREVIEW and run.hits_reported == 1  # date limit applied
    assert fake.esearch_calls[-1]["mindate"] == "2000/01/01"
    assert run.query_translation and run.files and run.files[0].role == "api_response"
    assert "api_key" not in (run.files[0].request or "")
    assert strategy(repo, "pubmed").validation_method == "api_preview"

    fake.errorlist = {"phrasesnotfound": ["drgu x"]}
    editor.save_draft(
        repo,
        PID,
        draft_data(
            search_strategies=[{"database": "pubmed", "query": "drgu x"}],
            selected_databases=["pubmed"],
        ),
        reviewer=human,
    )
    sha2 = strategy(repo, "pubmed").sha256
    run2 = asyncio.run(svc(repo, tmp_path).preview(sha2, reviewer=human))
    assert "error: phrasesnotfound: drgu x" in run2.warnings
    assert not strategy(repo, "pubmed").syntax_validated


def test_executable_query_translation() -> None:
    pm = SearchStrategy(
        database=SourceDatabase.PUBMED,
        query="a OR b",
        filters=["humans[mh]"],
        date_from=date(2000, 1, 1),
    )
    q = executable_query(pm)
    assert q.query == "(a OR b) AND (humans[mh])"
    assert q.parameters == {"datetype": "pdat", "mindate": "2000/01/01", "maxdate": "3000"}
    ep = executable_query(
        SearchStrategy(database=SourceDatabase.EUROPE_PMC, query="a", date_to=date(2020, 12, 31))
    )
    assert ep.query == "(a) AND (FIRST_PDATE:[1800-01-01 TO 2020-12-31])"
    oa = executable_query(
        SearchStrategy(
            database=SourceDatabase.OPENALEX,
            query="drug x",
            filters=["type:article"],
            date_from=date(2010, 1, 1),
        )
    )
    assert oa.parameters == {"filter": "type:article,from_publication_date:2010-01-01"}
    with pytest.raises(SearchError, match="import an export"):
        executable_query(SearchStrategy(database=SourceDatabase.EMBASE, query="x"))
    with pytest.raises(SearchError, match="language limits"):
        executable_query(
            SearchStrategy(database=SourceDatabase.PUBMED, query="x", languages=["eng"])
        )


# ---------------------------------------------------------------------- execution
@respx.mock
def test_pubmed_execution_stores_records_provenance_and_raw_responses(
    repo: Repository, human: ReviewerORM, tmp_path: Path
) -> None:
    fake = FakePubMed({f"9100000{i}": date(2010 + i, 1, 1) for i in range(5)})
    respx.get(f"{EUTILS_BASE}esearch.fcgi").mock(side_effect=fake.esearch)
    respx.post(f"{EUTILS_BASE}efetch.fcgi").mock(side_effect=fake.efetch)
    service = svc(repo, tmp_path)
    sha = None
    with pytest.raises(SearchError, match="stage2_search_screening"):
        asyncio.run(service.execute("0" * 64, reviewer=human))
    approved_in_stage2(repo, PID, human)
    sha = strategy(repo, "pubmed").sha256
    run = asyncio.run(service.execute(sha, reviewer=human))
    assert run.status == SearchRunStatus.COMPLETE
    assert (run.hits_reported, run.records_saved) == (5, 5)
    assert run.query == PUBMED_QUERY and run.parameters["mindate"] == "2000/01/01"
    assert run.protocol_version == 1 and run.actor == "Dr Human"
    records = repo.load_records(PID)
    assert len(records) == 5
    assert {r.provenance.search_run_id for r in records} == {run.run_id}  # type: ignore[union-attr]
    stored = DocumentStore(tmp_path / "searches")
    for f in run.files:  # every raw response preserved and checksummed
        assert stored.read_verified(f.filename, f.sha256)
    with pytest.raises(SearchError, match="already executed"):
        asyncio.run(service.execute(sha, reviewer=human))
    # identification is incomplete until Embase/CT.gov are handled and records are prepared
    problems = repo.search_identification_problems(PID)
    assert any("embase strategy" in p for p in problems)
    assert any("not been prepared" in p for p in problems)
    with pytest.raises(GuardViolationError, match="searches: "):
        advance(repo, PID, Phase.GATE1_5_SCREENING_CONFLICTS, human)


@respx.mock
def test_pubmed_over_limit_is_partitioned_by_date_and_verified(
    repo: Repository, human: ReviewerORM, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.apis.pubmed as pubmed

    monkeypatch.setattr(pubmed, "PUBMED_MAX_RETRIEVABLE", 10)
    base = date(2000, 1, 1)
    fake = FakePubMed({f"92{i:06d}": base + timedelta(days=300 * i) for i in range(25)})
    respx.get(f"{EUTILS_BASE}esearch.fcgi").mock(side_effect=fake.esearch)
    respx.post(f"{EUTILS_BASE}efetch.fcgi").mock(side_effect=fake.efetch)
    approved_in_stage2(repo, PID, human)
    run = asyncio.run(svc(repo, tmp_path).execute(strategy(repo, "pubmed").sha256, reviewer=human))
    assert run.status == SearchRunStatus.COMPLETE and run.records_saved == 25
    assert len(run.partitions) >= 3 and sum(p.count for p in run.partitions) == 25
    assert all(p.count <= 10 for p in run.partitions)


@respx.mock
def test_incomplete_partitioning_fails_without_saving(
    repo: Repository, human: ReviewerORM, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.apis.pubmed as pubmed

    monkeypatch.setattr(pubmed, "PUBMED_MAX_RETRIEVABLE", 10)
    fake = FakePubMed({f"93{i:06d}": date(2005, 1, 1) + timedelta(days=i) for i in range(15)})
    fake.total_override = 16  # PubMed reports one record that no date slice returns
    respx.get(f"{EUTILS_BASE}esearch.fcgi").mock(side_effect=fake.esearch)
    approved_in_stage2(
        repo,
        PID,
        human,
        search_strategies=[{"database": "pubmed", "query": "x"}],
        selected_databases=["pubmed"],
    )
    run = asyncio.run(svc(repo, tmp_path).execute(strategy(repo, "pubmed").sha256, reviewer=human))
    assert run.status == SearchRunStatus.FAILED and "16 records" in run.problems[0]
    assert repo.load_records(PID) == []
    assert not run.identifies_records


@respx.mock
def test_efetch_shortfall_is_a_count_mismatch_needing_acknowledgement(
    repo: Repository, human: ReviewerORM, tmp_path: Path
) -> None:
    fake = FakePubMed(
        {"94000001": date(2020, 1, 1), "94000002": date(2020, 1, 2)}, drop_from_efetch={"94000002"}
    )
    respx.get(f"{EUTILS_BASE}esearch.fcgi").mock(side_effect=fake.esearch)
    respx.post(f"{EUTILS_BASE}efetch.fcgi").mock(side_effect=fake.efetch)
    approved_in_stage2(
        repo,
        PID,
        human,
        search_strategies=[{"database": "pubmed", "query": "x"}],
        selected_databases=["pubmed"],
    )
    service = svc(repo, tmp_path)
    run = asyncio.run(service.execute(strategy(repo, "pubmed").sha256, reviewer=human))
    assert run.status == SearchRunStatus.COUNT_MISMATCH and run.records_saved == 1
    assert "94000002" in run.problems[0]
    service.prepare_for_screening(reviewer=human)
    assert any("acknowledge" in p for p in repo.search_identification_problems(PID))
    service.acknowledge(run_id=run.run_id, note="record withdrawn by PubMed", reviewer=human)
    assert repo.search_identification_problems(PID) == []


@respx.mock
def test_europepmc_and_openalex_execution(
    repo: Repository, human: ReviewerORM, tmp_path: Path
) -> None:
    from tests.unit.test_api_clients import EPMC_RESULT, OPENALEX_WORK

    epmc = respx.get(f"{EUROPEPMC_BASE}search").mock(
        side_effect=[
            httpx.Response(200, json={"hitCount": 2}),
            httpx.Response(
                200,
                json={
                    "hitCount": 2,
                    "nextCursorMark": "n1",
                    "resultList": {"result": [EPMC_RESULT, {"id": "X", "source": "MED"}]},
                },
            ),
            httpx.Response(
                200, json={"hitCount": 2, "nextCursorMark": "n1", "resultList": {"result": []}}
            ),
        ]
    )
    oa = respx.get(f"{OPENALEX_BASE}works").mock(
        side_effect=[
            httpx.Response(200, json={"meta": {"count": 1}, "results": []}),
            httpx.Response(
                200, json={"meta": {"count": 1, "next_cursor": None}, "results": [OPENALEX_WORK]}
            ),
        ]
    )
    approved_in_stage2(
        repo,
        PID,
        human,
        selected_databases=["europe_pmc", "openalex"],
        search_strategies=[
            {"database": "europe_pmc", "query": "drug x", "date_from": "2001-01-01"},
            {"database": "openalex", "query": "drug x", "filters": ["type:article"]},
        ],
    )
    service = svc(repo, tmp_path)
    r1 = asyncio.run(service.execute(strategy(repo, "europe_pmc").sha256, reviewer=human))
    assert r1.status == SearchRunStatus.COUNT_MISMATCH  # one result without a title
    assert any("without a title" in p for p in r1.problems) and r1.records_saved == 1
    assert "FIRST_PDATE:[2001-01-01 TO 3000-12-31]" in epmc.calls[0].request.url.params["query"]
    r2 = asyncio.run(service.execute(strategy(repo, "openalex").sha256, reviewer=human))
    assert r2.status == SearchRunStatus.COMPLETE and r2.records_saved == 1
    assert oa.calls[1].request.url.params["filter"] == "type:article"


@respx.mock
def test_api_failure_is_logged_as_failed_attempt_not_as_search(
    repo: Repository, human: ReviewerORM, tmp_path: Path
) -> None:
    respx.get(f"{EUTILS_BASE}esearch.fcgi").mock(return_value=httpx.Response(400, text="bad"))
    approved_in_stage2(repo, PID, human)
    run = asyncio.run(svc(repo, tmp_path).execute(strategy(repo, "pubmed").sha256, reviewer=human))
    assert run.status == SearchRunStatus.FAILED and repo.load_records(PID) == []
    report = search_report(repo, PID)
    assert report["databases_searched"] == []
    assert any("pubmed strategy" in p for p in report["problems"])


# ---------------------------------------------------------------------- imports & dedup
@respx.mock
def test_multi_database_identification_dedup_and_prisma(
    repo: Repository, human: ReviewerORM, tmp_path: Path, session: Session
) -> None:
    fixture = (Path(__file__).parents[1] / "fixtures" / "pubmed_efetch_synthetic.xml").read_bytes()
    fake = FakePubMed({"90000001": date(2021, 3, 1), "90000002": date(2021, 6, 1)})
    respx.get(f"{EUTILS_BASE}esearch.fcgi").mock(side_effect=fake.esearch)
    respx.post(f"{EUTILS_BASE}efetch.fcgi").mock(return_value=httpx.Response(200, content=fixture))
    approved_in_stage2(
        repo,
        PID,
        human,
        selected_databases=[
            "pubmed",
            "embase",
            "scopus",
            "cochrane_central",
            "clinicaltrials_gov",
            "web_of_science",
        ],
        search_strategies=[
            {"database": "pubmed", "query": PUBMED_QUERY},
            {"database": "embase", "platform": "Embase.com", "query": "'drug x'/exp"},
            {"database": "scopus", "query": 'TITLE-ABS-KEY("drug x")'},
            {"database": "cochrane_central", "query": "drug x"},
            {"database": "clinicaltrials_gov", "query": "drug X"},
            {"database": "web_of_science", "query": 'TS=("drug x")'},
        ],
    )
    service = svc(repo, tmp_path)
    sha = {
        db: strategy(repo, db).sha256
        for db in (
            "pubmed",
            "embase",
            "scopus",
            "cochrane_central",
            "clinicaltrials_gov",
            "web_of_science",
        )
    }
    pm = asyncio.run(service.execute(sha["pubmed"], reviewer=human))
    assert pm.records_saved == 2
    day = date(2026, 10, 1)
    emb = service.import_file(
        sha["embase"],
        (FIX / "embase_synthetic.ris").read_bytes(),
        filename="embase.ris",
        fmt="ris",
        searched_on=day,
        reviewer=human,
        hits_reported=4,
    )
    assert emb.records_saved == 3 and emb.files[0].original_name == "embase.ris"
    with pytest.raises(ImportFileError, match="already imported"):
        service.import_file(
            sha["embase"],
            (FIX / "embase_synthetic.ris").read_bytes(),
            filename="again.ris",
            fmt="ris",
            searched_on=day,
            reviewer=human,
        )
    with pytest.raises(ImportFileError, match="needs a title"):
        service.import_file(
            sha["embase"],
            b"TY  - JOUR\nAU  - X, Y\nER  - \n",
            filename="bad.ris",
            fmt="ris",
            searched_on=day,
            reviewer=human,
        )
    with pytest.raises(ImportFileError, match="future"):
        service.import_file(
            sha["embase"],
            b"TY  - JOUR\nTI  - T\nER  - \n",
            filename="f.ris",
            fmt="ris",
            searched_on=date(2999, 1, 1),
            reviewer=human,
        )
    service.import_file(
        sha["scopus"],
        (FIX / "scopus_synthetic.bib").read_bytes(),
        filename="scopus.bib",
        fmt="bibtex",
        searched_on=day,
        reviewer=human,
        hits_reported=2,
    )
    service.import_file(
        sha["cochrane_central"],
        (FIX / "central_synthetic.ris").read_bytes(),
        filename="central.ris",
        fmt="ris",
        searched_on=day,
        reviewer=human,
        hits_reported=1,
    )
    ct = service.import_file(
        sha["clinicaltrials_gov"],
        (FIX / "ctgov_synthetic.csv").read_bytes(),
        filename="ctgov.csv",
        fmt="csv",
        searched_on=day,
        reviewer=human,
        hits_reported=2,
    )
    assert ct.column_map["title"] == "Study Title"
    service.record_not_searched(
        sha["web_of_science"], reason="no institutional access", reviewer=human
    )

    problems = repo.search_identification_problems(PID)
    assert any("reported 4 records but 3 were imported" in p for p in problems)
    service.acknowledge(
        strategy_sha256=sha["embase"],
        reviewer=human,
        note="one record could not be exported (synthetic)",
    )
    assert repo.search_identification_problems(PID) == [
        "records have not been prepared for screening (deduplicated) since the last search or "
        "import"
    ]
    prep = service.prepare_for_screening(reviewer=human)
    assert repo.search_identification_problems(PID) == []
    # 2 PubMed + 3 Embase + 2 Scopus + 1 CENTRAL + 2 CT.gov = 10 records.
    assert prep.summary["records"] == 10
    # Duplicates: Embase#1 and Scopus ref1 (DOI/PMID of PubMed 90000001); CENTRAL#1 (title of
    # PubMed 90000002). Embase#3 / Scopus ref2 share a DOI but have unrelated titles.
    assert prep.summary["duplicates_removed"] == 3
    assert prep.summary["possible_duplicates_for_review"] == 1
    candidate = repo.load_duplicate_candidates(PID)[0]
    assert "dissimilar titles" in candidate.conflict
    by_source = prep.summary["by_source"]
    assert by_source["pubmed"] == {"identified": 2, "duplicates_removed": 0, "unique": 2}
    assert by_source["embase"]["duplicates_removed"] == 1
    assert by_source["clinicaltrials_gov"] == {
        "identified": 2,
        "duplicates_removed": 0,
        "unique": 2,
    }
    prisma = repo.load_prisma(PID)
    assert prisma is not None
    assert prisma.identified_from_registers == 2  # ClinicalTrials.gov is a register
    assert prisma.identified_by_database[SourceDatabase.EMBASE] == 3
    assert SourceDatabase.WEB_OF_SCIENCE not in prisma.identified_by_database
    assert prisma.total_identified == 10 and prisma.duplicates_removed == 3
    assert prisma.records_awaiting_screening == 7
    raw = repo.raw_record(
        PID,
        next(
            r.record_id
            for r in repo.load_records(PID)
            if r.source_database == SourceDatabase.CLINICALTRIALS_GOV
        ),
    )
    assert raw is not None and "NCT Number" in raw

    report = search_report(repo, PID)
    assert "Web of Science" not in report["databases_searched"]
    assert report["databases_not_searched"] == [
        {"database": "Web of Science", "reasons": ["no institutional access"]}
    ]
    md = report_markdown(report)
    assert "Not searched: Web of Science — no institutional access" in md
    assert "embase.ris sha256:" in report_csv(report)

    # The screening pool is the unique records; the AI step accepts it now.
    unique = repo.load_records(PID, unique_only=True)
    assert len(unique) == 7
    # Human adjudicates the ambiguous pair as distinct; PRISMA re-derived.
    service.resolve_duplicate(candidate.candidate_id, resolution="distinct", reviewer=human)
    again = service.prepare_for_screening(reviewer=human)  # the decision is kept, not re-asked
    assert again.summary["human_decisions_reapplied"] == 1
    assert again.summary["possible_duplicates_for_review"] == 0
    assert [c.resolution for c in repo.load_duplicate_candidates(PID)] == ["distinct"]
    # Screening starts. Session 8: a later import is an *update search*, prepared
    # incrementally — the screened pool is never re-deduplicated (test_update_searches.py).
    repo.save_screening_decision(
        PID,
        ScreeningDecision(
            record_id=unique[0].record_id,
            reviewer=ReviewerRole.REVIEWER_1,
            reviewer_identity="llm-pass-1",
            stage=ScreeningStage.TITLE_ABSTRACT,
            verdict=ScreeningVerdict.INCLUDE,
            rationale="synthetic",
        ),
    )
    late = service.import_file(
        sha["embase"],
        b"TY  - JOUR\nTI  - Late\nER  - \n",
        filename="late.ris",
        fmt="ris",
        searched_on=day,
        reviewer=human,
    )
    assert late.records_saved == 1
    assert any("not been prepared" in p for p in repo.search_identification_problems(PID))
    update = service.prepare_for_screening(reviewer=human)
    assert update.summary["update_search"] and update.summary["new_unique_records_to_screen"] == 1
    counts = derive_prisma(repo, PID)
    assert counts.records_awaiting_screening == 8 and counts.records_screened == 0
    # The search log is append-only.
    row = session.query(SearchRunORM).first()
    assert row is not None
    row.status = "complete"
    with pytest.raises(AppendOnlyViolationError):
        session.flush()


def test_ai_screening_refuses_until_identification_complete(
    repo: Repository, human: ReviewerORM, tmp_path: Path
) -> None:
    from src.llm.catalogue import ModelCatalogue
    from src.llm.credentials import CredentialResolver

    approved_in_stage2(repo, PID, human)
    service = svc(repo, tmp_path)
    service.import_file(
        strategy(repo, "embase").sha256,
        (FIX / "embase_synthetic.ris").read_bytes(),
        filename="e.ris",
        fmt="ris",
        searched_on=date(2026, 10, 1),
        reviewer=human,
    )
    wf = AIWorkflow(
        repo,
        PID,
        catalogue=ModelCatalogue.load(),
        credentials=CredentialResolver(),
        store=DocumentStore(tmp_path / "ft"),
    )
    with pytest.raises(WorkflowStepError, match="identification is not complete"):
        wf.run_title_abstract_screening()


def test_search_actions_require_an_enrolled_reviewer(
    repo: Repository, human: ReviewerORM, tmp_path: Path
) -> None:
    approved_in_stage2(repo, PID, human)
    repo.revoke_reviewer(human.id)
    with pytest.raises(SysRevError, match="not enrolled or was revoked"):
        svc(repo, tmp_path).record_not_searched(
            strategy(repo, "embase").sha256, reason="x", reviewer=human
        )
