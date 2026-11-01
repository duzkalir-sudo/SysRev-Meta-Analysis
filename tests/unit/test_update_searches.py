"""Session 8: additional/update searches after screening has begun (Stage 2).

Records already prepared for screening are frozen: an update search's records are
deduplicated against them incrementally, never the reverse, and PRISMA is re-derived so the
update's records, duplicates and new screening workload are counted. All data SYNTHETIC.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from src.core.enums import ReviewerRole, ScreeningStage, ScreeningVerdict
from src.core.schemas import ScreeningDecision
from src.fulltext.storage import DocumentStore
from src.search import amendments
from src.search import protocol_editor as editor
from src.search.prisma_sync import derive_prisma
from src.search.service import SearchError, SearchService
from src.state.db import ReviewerORM
from src.state.repository import Repository
from tests.fakes import enroll
from tests.search_helpers import approved_in_stage2, fast_clients

FIX = Path(__file__).parents[1] / "fixtures" / "search"
PID = "u8"
DAY = date(2026, 10, 1)
UPDATE_RIS = b"""TY  - JOUR
TI  - Effect of drug X on HbA1c: a randomised controlled trial
AU  - Synthetica, Ana
PY  - 2021
DO  - 10.5555/synth.0001
ER  -

TY  - JOUR
TI  - Drug X for condition Y: a new synthetic trial published after the first search
AU  - Novel, Nora
PY  - 2026
DO  - 10.5555/synth.0100
ER  -

TY  - JOUR
TI  - Long-term drug X in condition Y: synthetic extension study
AU  - Later, Lee
PY  - 2026
DO  - 10.5555/synth.0101
ER  -

TY  - JOUR
TI  - Long-term drug X in condition Y: synthetic extension study
AU  - Later, Lee
PY  - 2026
DO  - 10.5555/synth.0101
ER  -
"""


@pytest.fixture
def repo(session: Session) -> Repository:
    r = Repository(session)
    r.create_project(PID, "Synthetic update-search review")
    return r


@pytest.fixture
def human(repo: Repository) -> ReviewerORM:
    return enroll(repo, "Dr Update")


def _service(repo: Repository, tmp_path: Path) -> SearchService:
    return SearchService(repo, PID, store=DocumentStore(tmp_path / "s"), clients=fast_clients())


def _screened_pool(repo: Repository, human: ReviewerORM, tmp_path: Path) -> SearchService:
    approved_in_stage2(repo, PID, human)
    svc = _service(repo, tmp_path)
    embase = next(
        s
        for s in repo.latest_protocol(PID).search_strategies  # type: ignore[union-attr]
        if s.database == "embase"
    )
    svc.import_file(
        embase.sha256,
        (FIX / "embase_synthetic.ris").read_bytes(),
        filename="e.ris",
        fmt="ris",
        searched_on=DAY,
        reviewer=human,
        hits_reported=3,
    )
    svc.prepare_for_screening(reviewer=human)
    for r in repo.load_records(PID, unique_only=True):
        for role in (ReviewerRole.REVIEWER_1, ReviewerRole.REVIEWER_2):
            repo.save_screening_decision(
                PID,
                ScreeningDecision(
                    record_id=r.record_id,
                    reviewer=role,
                    reviewer_identity=f"llm-{role}",
                    stage=ScreeningStage.TITLE_ABSTRACT,
                    verdict=ScreeningVerdict.INCLUDE,
                    rationale="synthetic",
                ),
            )
    return svc


def _add_strategy_by_amendment(repo: Repository, human: ReviewerORM) -> str:
    protocol = repo.latest_protocol(PID)
    assert protocol is not None
    data = protocol.model_dump(mode="json")
    data["search_strategies"] = [
        *data["search_strategies"],
        {
            "database": "scopus",
            "platform": "Scopus",
            "query": 'TITLE-ABS-KEY("drug x")',
            "date_from": "2025-01-01",
        },
    ]
    data["selected_databases"] = [*data["selected_databases"], "scopus"]
    p = amendments.propose(
        repo,
        PID,
        data,
        rationale="top-up search of a database added after peer review of the protocol",
        reviewer=human,
        expected_base_sha256=editor.protocol_sha256(protocol),
    )
    new_sha = next(s.sha256 for s in p.protocol.search_strategies if s.database == "scopus")
    p = amendments.record_strategy_validation(
        repo,
        PID,
        p.amendment_id,
        new_sha,
        method="human_attestation",
        reviewer=human,
        expected_sha256=p.sha256,
        note="checked",
    )
    amendments.decide(
        repo,
        PID,
        p.amendment_id,
        "approved",
        reviewer=human,
        comments="ok",
        expected_sha256=p.sha256,
    )
    return new_sha


def test_update_search_is_deduplicated_against_the_frozen_pool(
    repo: Repository, human: ReviewerORM, tmp_path: Path
) -> None:
    svc = _screened_pool(repo, human, tmp_path)
    frozen = {r.record_id: r.model_dump(mode="json") for r in repo.load_records(PID)}
    decisions_before = [d.model_dump(mode="json") for d in repo.load_screening_decisions(PID)]
    before = derive_prisma(repo, PID)
    assert (before.total_identified, before.records_screened) == (3, 3)
    new_sha = _add_strategy_by_amendment(repo, human)
    obligation = next(o for o in amendments.obligation_status(repo, PID) if o["area"] == "searches")
    assert obligation["subject"] == new_sha and obligation["state"] == "open"
    run = svc.import_file(
        new_sha,
        UPDATE_RIS,
        filename="scopus-update.ris",
        fmt="ris",
        searched_on=DAY,
        reviewer=human,
        hits_reported=4,
    )
    assert run.records_saved == 4
    assert (
        next(o for o in amendments.obligation_status(repo, PID) if o["area"] == "searches")["state"]
        == "resolved"
    )
    update = svc.prepare_for_screening(reviewer=human)
    s = update.summary
    assert s["update_search"] and s["new_records"] == 4
    assert s["duplicates_of_screened_records"] == 1  # same DOI as a screened record
    assert s["duplicates_among_new_records"] == 1  # the two copies of the extension study
    assert s["new_unique_records_to_screen"] == 2
    # frozen records and their decisions are unchanged
    after = {r.record_id: r.model_dump(mode="json") for r in repo.load_records(PID)}
    assert all(after[rid] == data for rid, data in frozen.items())
    assert [d.model_dump(mode="json") for d in repo.load_screening_decisions(PID)] == (
        decisions_before
    )
    dup = next(
        r
        for r in repo.load_records(PID)
        if r.doi == "10.5555/synth.0001" and r.record_id not in frozen
    )
    assert dup.duplicate_of in frozen  # the screened record is kept
    prisma = derive_prisma(repo, PID)
    assert prisma.total_identified == 7 and prisma.duplicates_removed == 2
    assert prisma.records_screened == 3 and prisma.records_awaiting_screening == 2
    assert repo.load_prisma(PID) == prisma  # snapshot re-derived by the preparation
    ctx = repo.build_guard_context(PID)
    assert ctx.ta_missing_decisions == 2
    # the update search is logged and prepared (the fixture never searched PubMed/CT.gov)
    assert not any("prepared" in p or new_sha[:12] in p for p in ctx.search_problems)


def test_a_screened_record_is_never_dropped_for_an_unscreened_duplicate(
    repo: Repository, human: ReviewerORM, tmp_path: Path
) -> None:
    from src.core.schemas import DuplicateCandidate

    svc = _screened_pool(repo, human, tmp_path)
    screened = repo.load_records(PID, unique_only=True)[0].record_id
    new_sha = _add_strategy_by_amendment(repo, human)
    svc.import_file(
        new_sha,
        UPDATE_RIS,
        filename="u.ris",
        fmt="ris",
        searched_on=DAY,
        reviewer=human,
        hits_reported=4,
    )
    svc.prepare_for_screening(reviewer=human)
    fresh = next(
        r.record_id
        for r in repo.load_records(PID, unique_only=True)
        if r.doi == "10.5555/synth.0100"
    )
    a, b = sorted((screened, fresh))
    candidate = DuplicateCandidate(
        record_id_a=a, record_id_b=b, match_basis="manual check", conflict="synthetic"
    )
    repo.save_duplicate_candidate(PID, candidate)
    drop_screened = "duplicate_keep_a" if a == fresh else "duplicate_keep_b"
    with pytest.raises(SearchError, match="keep the screened record"):
        svc.resolve_duplicate(candidate.candidate_id, resolution=drop_screened, reviewer=human)


def test_update_preparation_needs_new_records(
    repo: Repository, human: ReviewerORM, tmp_path: Path
) -> None:
    svc = _screened_pool(repo, human, tmp_path)
    with pytest.raises(SearchError, match="no search or import was logged"):
        svc.prepare_for_screening(reviewer=human)
