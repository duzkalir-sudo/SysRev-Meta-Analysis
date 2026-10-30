"""Shared Session 5 test helpers: a fake PubMed index behind respx, protocol drafts.

Everything is SYNTHETIC: PMIDs 9xxxxxxx, DOIs with the 10.5555 test prefix.
"""

from __future__ import annotations

import asyncio
from datetime import date
from typing import Any
from urllib.parse import parse_qs

import httpx

from src.apis.europepmc import EuropePMCClient
from src.apis.openalex import OpenAlexClient
from src.apis.pubmed import PubMedClient
from src.apis.rate_limit import AsyncRateLimiter
from src.core.enums import ActorType, GateId, GateVerdict, Phase
from src.core.schemas.pipeline import GateDecision
from src.search import protocol_editor as editor
from src.search.service import SearchClients
from src.state.db import ReviewerORM
from src.state.repository import Repository
from tests.conftest import FakeClock


class MeSHAlwaysFound:
    def __init__(self, missing: set[str] | None = None) -> None:
        self.missing = missing or set()
        self.asked: list[str] = []

    async def mesh_descriptor_exists(self, descriptor: str) -> bool:
        self.asked.append(descriptor)
        return descriptor not in self.missing


def fast_clients() -> SearchClients:
    def limiter() -> AsyncRateLimiter:
        clock = FakeClock()
        return AsyncRateLimiter(1000, 1.0, clock=clock, sleep=clock.sleep)

    async def no_sleep(_: float) -> None:
        return None

    kw: dict[str, Any] = {"sleep": no_sleep}
    return SearchClients(
        pubmed=lambda: PubMedClient(email="t@example.org", rate_limiter=limiter(), **kw),
        europepmc=lambda: EuropePMCClient(rate_limiter=limiter(), **kw),
        openalex=lambda: OpenAlexClient(mailto="t@example.org", rate_limiter=limiter(), **kw),
    )


class FakePubMed:
    """Answers ESearch (pubmed + mesh) and EFetch from an in-memory index of dated PMIDs."""

    def __init__(
        self,
        records: dict[str, date],
        *,
        mesh: set[str] | None = None,
        errorlist: dict[str, list[str]] | None = None,
        total_override: int | None = None,
        drop_from_efetch: set[str] | None = None,
    ) -> None:
        self.records = dict(sorted(records.items()))
        self.mesh = mesh or set()
        self.errorlist = errorlist
        self.total_override = total_override
        self.drop = drop_from_efetch or set()
        self.esearch_calls: list[dict[str, str]] = []

    @staticmethod
    def _date(value: str, end: bool) -> date:
        parts = [int(p) for p in value.split("/")]
        if len(parts) == 1:
            return date(min(parts[0], 9999), 12, 31) if end else date(parts[0], 1, 1)
        return date(parts[0], parts[1], parts[2])

    def esearch(self, request: httpx.Request) -> httpx.Response:
        params = {k: v[-1] for k, v in parse_qs(request.url.query.decode()).items()}
        self.esearch_calls.append(params)
        if params.get("db") == "mesh":
            term = params["term"].split('"')[1]
            count = 1 if term in self.mesh else 0
            return httpx.Response(200, json={"esearchresult": {"count": str(count), "idlist": []}})
        ids = list(self.records)
        if "mindate" in params:
            lo, hi = self._date(params["mindate"], False), self._date(params["maxdate"], True)
            ids = [p for p in ids if lo <= self.records[p] <= hi]
        count = (
            len(ids) if self.total_override is None or "mindate" in params else self.total_override
        )
        start, retmax = int(params.get("retstart", 0)), int(params.get("retmax", 20))
        result: dict[str, Any] = {
            "count": str(count),
            "idlist": ids[start : start + retmax],
            "querytranslation": f"translated({params['term']})",
        }
        if self.errorlist:
            result["errorlist"] = self.errorlist
        return httpx.Response(200, json={"esearchresult": result})

    def efetch(self, request: httpx.Request) -> httpx.Response:
        form = parse_qs(request.content.decode())
        ids = [i for i in form["id"][0].split(",") if i not in self.drop]
        articles = "".join(
            f"<PubmedArticle><MedlineCitation><PMID>{pmid}</PMID><Article>"
            f"<Journal><JournalIssue><PubDate><Year>{self.records[pmid].year}</Year></PubDate>"
            f"</JournalIssue><Title>Synthetic Journal</Title></Journal>"
            f"<ArticleTitle>Synthetic record {pmid} on drug X</ArticleTitle>"
            f"<AuthorList><Author><LastName>Author{pmid[-3:]}</LastName></Author></AuthorList>"
            f"</Article></MedlineCitation></PubmedArticle>"
            for pmid in ids
        )
        xml = f"<?xml version='1.0'?><PubmedArticleSet>{articles}</PubmedArticleSet>"
        return httpx.Response(200, content=xml.encode(), headers={"content-type": "text/xml"})


PUBMED_QUERY = '("drug x"[tiab]) AND (randomized controlled trial[pt])'


def draft_data(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "review_title": "Synthetic drug X versus placebo for condition Y",
        "research_question": "Does drug X reduce outcome Z in adults with condition Y?",
        "population": "Adults with condition Y",
        "intervention": "Drug X",
        "comparator": "Placebo",
        "review_design": "comparative",
        "outcomes": [
            {
                "outcome_id": "hba1c",
                "name": "HbA1c change",
                "kind": "primary",
                "data_type": "continuous",
                "preferred_effect_measure": "MD",
            }
        ],
        "study_designs": ["rct"],
        "inclusion_criteria": ["Randomised trials in adults"],
        "exclusion_criteria": ["Animal studies"],
        "mesh_terms": [{"descriptor": "Diabetes Mellitus"}],
        "selected_databases": ["pubmed", "embase", "clinicaltrials_gov"],
        "search_strategies": [
            {
                "database": "pubmed",
                "platform": "PubMed",
                "query": PUBMED_QUERY,
                "date_from": "2000-01-01",
            },
            {
                "database": "embase",
                "platform": "Embase.com",
                "query": "'drug x'/exp AND 'randomized controlled trial'/de",
            },
            {"database": "clinicaltrials_gov", "query": "drug X AND condition Y"},
        ],
    }
    data.update(overrides)
    return data


def advance(repo: Repository, pid: str, target: Phase, reviewer: ReviewerORM) -> None:
    m = repo.load_state_machine(pid)
    repo.record_transition(
        m.transition(
            target,
            actor=reviewer.display_name,
            actor_type=ActorType.HUMAN,
            context=repo.build_guard_context(pid),
        )
    )


def pass_gate1(repo: Repository, pid: str, reviewer: ReviewerORM) -> None:
    advance(repo, pid, Phase.GATE1_SEARCH_STRATEGY, reviewer)
    m = repo.load_state_machine(pid)
    decision = GateDecision(
        gate=GateId.GATE_1,
        verdict=GateVerdict.APPROVED,
        reviewer_name=reviewer.display_name,
        reviewer_id=reviewer.id,
    )
    repo.record_transition(m.submit_gate_decision(decision, repo.build_guard_context(pid)))


def approved_in_stage2(repo: Repository, pid: str, reviewer: ReviewerORM, **overrides: Any) -> None:
    """Draft → verify MeSH/strategies (directly) → approve → Gate 1 → Stage 2."""
    editor.save_draft(repo, pid, draft_data(**overrides), reviewer=reviewer)
    protocol = repo.latest_protocol(pid)
    assert protocol is not None
    for s in protocol.search_strategies:
        method = "api_preview" if s.database == "pubmed" else "human_attestation"
        editor.record_strategy_validation(
            repo,
            pid,
            s.sha256,
            method=method,  # type: ignore[arg-type]
            reviewer=reviewer,
            note="synthetic validation",
        )
    asyncio.run(editor.verify_mesh(repo, pid, MeSHAlwaysFound(), reviewer=reviewer))
    editor.approve(repo, pid, reviewer=reviewer)
    pass_gate1(repo, pid, reviewer)
