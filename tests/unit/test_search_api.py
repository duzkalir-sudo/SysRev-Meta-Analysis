"""Session 5 over HTTP: protocol → approval → Gate 1 → search → import → screening pool.

Mocked literature APIs (respx); synthetic data only. Every write needs the reviewer token.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from pathlib import Path

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from src.api_server.app import create_app
from src.apis.pubmed import EUTILS_BASE
from src.screening.deduplication import Deduplicator
from src.state.repository import Repository
from tests.conftest import make_record
from tests.search_helpers import FakePubMed, draft_data, fast_clients

FIX = Path(__file__).parents[1] / "fixtures" / "search"


@pytest.fixture
def api(tmp_path: Path) -> Iterator[tuple[TestClient, dict[str, str]]]:
    app = create_app("sqlite:///:memory:", data_dir=tmp_path, search_clients=fast_clients())
    with TestClient(app) as client:
        with app.state.session_factory() as s:
            _, token = Repository(s).enroll_reviewer("Dr API")
            s.commit()
        yield client, {"Authorization": f"Bearer {token}"}


@respx.mock
def test_end_to_end_protocol_search_import_and_screening_pool(
    api: tuple[TestClient, dict[str, str]],
) -> None:
    client, auth = api
    fake = FakePubMed(
        {"95000001": date(2021, 1, 1), "95000002": date(2022, 1, 1)}, mesh={"Diabetes Mellitus"}
    )
    respx.get(f"{EUTILS_BASE}esearch.fcgi").mock(side_effect=fake.esearch)
    respx.post(f"{EUTILS_BASE}efetch.fcgi").mock(side_effect=fake.efetch)
    pid = client.post("/projects", json={"title": "Synthetic HTTP review"}).json()["project_id"]
    base = f"/projects/{pid}"
    data = draft_data(
        selected_databases=["pubmed", "embase"],
        search_strategies=draft_data()["search_strategies"][:2],
    )

    # Writes without a token are refused.
    assert client.put(f"{base}/protocol", json=data).status_code == 401
    bad = client.put(f"{base}/protocol", json=data | {"comparator": None}, headers=auth)
    assert bad.status_code == 422 and "comparator" in str(bad.json()["detail"]["errors"])

    saved = client.put(f"{base}/protocol", json=data, headers=auth).json()
    assert saved["protocol"]["status"] == "draft" and saved["new_version"]
    assert "one or more MeSH terms are not verified against NLM MeSH" in saved["approval_blockers"]
    verified = client.post(f"{base}/protocol/mesh/verify", headers=auth).json()
    assert verified["mesh_results"] == {"Diabetes Mellitus": True}
    shas = {s["database"]: s["sha256"] for s in saved["strategies"]}
    preview = client.post(f"{base}/protocol/strategies/{shas['pubmed']}/preview", headers=auth)
    assert preview.json()["hits_reported"] == 2
    refused = client.post(
        f"{base}/protocol/strategies/{shas['pubmed']}/attest", headers=auth, json={"note": "x"}
    )
    assert refused.status_code == 409  # API strategies are validated by a preview
    client.post(
        f"{base}/protocol/strategies/{shas['embase']}/attest",
        headers=auth,
        json={"note": "run in Embase.com, 4 hits, no syntax errors"},
    )
    approved = client.post(f"{base}/protocol/approve", headers=auth, json={}).json()
    assert approved["protocol"]["status"] == "approved"
    assert approved["protocol"]["approved_by"] == "Dr API"
    history = client.get(f"{base}/protocol/history").json()
    assert [h["action"] for h in history][-1] == "approved"

    # Executing before Gate 1 is refused; the gate then requires the token.
    early = client.post(f"{base}/searches/strategies/{shas['pubmed']}/execute", headers=auth)
    assert early.status_code == 409
    client.post(f"{base}/advance", json={"target": "gate1_search_strategy"}, headers=auth)
    gate = client.post(f"{base}/gates/gate_1", json={"verdict": "approved"}, headers=auth)
    assert gate.json()["phase"] == "stage2_search_screening"
    assert client.put(f"{base}/protocol", json=data, headers=auth).status_code == 409  # locked

    run = client.post(f"{base}/searches/strategies/{shas['pubmed']}/execute", headers=auth)
    assert run.status_code == 201 and run.json()["records_saved"] == 2
    ris = (FIX / "embase_synthetic.ris").read_bytes()
    check = client.post(f"{base}/searches/import-preview", params={"format": "ris"}, content=ris)
    assert check.json()["entries"] == 3 and check.json()["errors"] == []
    params = {
        "format": "ris",
        "filename": "embase.ris",
        "searched_on": "2026-10-01",
        "hits_reported": 3,
    }
    assert (
        client.post(
            f"{base}/searches/strategies/{shas['embase']}/import", params=params, content=ris
        ).status_code
        == 401
    )
    imp = client.post(
        f"{base}/searches/strategies/{shas['embase']}/import",
        params=params,
        content=ris,
        headers=auth,
    )
    assert imp.status_code == 201 and imp.json()["records_saved"] == 3

    blocked = client.post(
        f"{base}/advance", json={"target": "gate1_5_screening_conflicts"}, headers=auth
    )
    assert any("not been prepared" in v for v in blocked.json()["detail"]["violations"])
    prep = client.post(f"{base}/searches/prepare", headers=auth).json()
    assert prep["summary"]["records"] == 5
    unique = client.get(f"{base}/records", params={"unique_only": True}).json()
    assert len(unique) == prep["summary"]["unique"]
    assert all(r["search_run_id"] for r in unique)
    prisma = client.get(f"{base}/prisma").json()
    assert prisma["identified_by_database"] == {"pubmed": 2, "embase": 3}
    assert prisma["records_awaiting_screening"] == len(unique)
    guards = client.get(f"{base}/guards").json()
    # Only screening remains: no search problems, PRISMA consistent.
    assert guards["gate1_5_screening_conflicts"] == [
        f"{len(unique)} unique record(s) lack title/abstract decisions from both reviewers"
    ]
    md = client.get(f"{base}/searches/report", params={"format": "markdown"}).text
    assert "Databases searched: Embase, PubMed" in md and PUBMED_IN(md)
    assert client.post(f"{base}/prisma/recompute").status_code == 401


def PUBMED_IN(md: str) -> bool:
    return '("drug x"[tiab]) AND (randomized controlled trial[pt])' in md


def test_duplicate_resolution_endpoint_requires_token(
    api: tuple[TestClient, dict[str, str]],
) -> None:
    client, auth = api
    pid = client.post("/projects", json={"title": "Dup"}).json()["project_id"]
    app = client.app
    with app.state.session_factory() as s:  # type: ignore[attr-defined]
        repo = Repository(s)
        recs = [
            make_record("a", "Alpha", doi="10.5555/d"),
            make_record("b", "Beta zeta", doi="10.5555/d"),
        ]
        result = Deduplicator(exact_match_title_floor=60).deduplicate(recs)
        assert result.duplicates_removed == 0 and len(result.possible_duplicates) == 1
        repo.save_records(pid, result.all_records)
        cand = result.possible_duplicates[0].to_candidate()
        repo.save_duplicate_candidate(pid, cand)
        project = repo.get_project(pid)
        project.phase = "stage2_search_screening"
        s.commit()
    url = f"/projects/{pid}/duplicates/{cand.candidate_id}/resolve"
    assert client.post(url, json={"resolution": "distinct"}).status_code == 401
    ok = client.post(
        url, json={"resolution": "distinct", "note": "different reports"}, headers=auth
    )
    assert ok.status_code == 200 and ok.json()["resolved_by"] == "Dr API"
    assert client.get(f"/projects/{pid}/prisma").json()["duplicates_removed"] == 0
    assert client.get(f"/projects/{pid}/duplicates").json()[0]["resolution"] == "distinct"


def test_dedup_title_floor_keeps_genuine_identifier_matches() -> None:
    a = make_record(
        "a", "Effect of drug X on HbA1c: a randomised controlled trial", doi="10.5555/e"
    )
    b = make_record(
        "b", "Effect of drug X on HbA1c: a randomized controlled trial.", doi="10.5555/E"
    )
    assert Deduplicator(exact_match_title_floor=60).deduplicate([a, b]).duplicate_of == {"b": "a"}


def test_databases_and_unknown_project(api: tuple[TestClient, dict[str, str]]) -> None:
    client, _ = api
    assert client.get("/projects/nope/protocol").status_code == 404
    assert client.get("/projects/nope/searches").status_code == 404
    _ = httpx  # imported for respx-mocked tests above
