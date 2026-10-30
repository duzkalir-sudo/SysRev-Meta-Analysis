"""Session 2 HTTP endpoints: authentication on every human action, inspectable outputs.

Synthetic data only.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api_server.app import create_app
from src.core.enums import ActorType, ReviewerRole, ScreeningStage, ScreeningVerdict
from src.core.schemas import ScreeningDecision
from src.extraction import submit_proposals
from src.pdf_extractor import parse_pdf
from src.state.repository import Repository
from tests.conftest import make_record
from tests.fakes import approved_protocol, include_at_title_abstract
from tests.fixtures.pdf_factory import SYNTHETIC_DOI, SYNTHETIC_TITLE, synthetic_article
from tests.unit.test_master_table import km

PID = "p1"


@pytest.fixture
def c(tmp_path: Path) -> Iterator[TestClient]:
    with TestClient(create_app("sqlite:///:memory:", data_dir=tmp_path)) as client:
        with client.app.state.session_factory() as s:  # type: ignore[attr-defined]
            repo = Repository(s)
            repo.create_project(PID, "Synthetic review")
            repo.save_protocol(PID, approved_protocol())
            recs = [
                make_record("r1", SYNTHETIC_TITLE, doi=SYNTHETIC_DOI),
                make_record("r2", "Second synthetic report"),
            ]
            repo.save_records(PID, [*recs, make_record("r3", "Third")])
            include_at_title_abstract(repo, PID, recs)
            for role, verdict in (
                (ReviewerRole.REVIEWER_1, ScreeningVerdict.INCLUDE),
                (ReviewerRole.REVIEWER_2, ScreeningVerdict.UNCERTAIN),
            ):
                repo.save_screening_decision(
                    PID,
                    ScreeningDecision(
                        record_id="r3",
                        reviewer=role,
                        reviewer_identity=f"llm:{role}",
                        stage=ScreeningStage.TITLE_ABSTRACT,
                        verdict=verdict,
                        rationale="synthetic",
                    ),
                )
            s.commit()
        yield client


def token(c: TestClient, name: str = "Dr API") -> str:
    with c.app.state.session_factory() as s:  # type: ignore[attr-defined]
        _, tok = Repository(s).enroll_reviewer(name)
        s.commit()
    return tok


def auth(tok: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {tok}"}


def test_every_human_action_requires_a_reviewer_token(c: TestClient) -> None:
    base = f"/projects/{PID}"
    calls = [
        (
            "post",
            f"{base}/screening/title_abstract/records/r3/adjudication",
            {"verdict": "include", "rationale": "x"},
        ),
        ("post", f"{base}/fulltext/r1/confirm-match", {"note": "x"}),
        ("post", f"{base}/fulltext/r1/unobtainable", {"reason": "x"}),
        ("post", f"{base}/extraction/items/abc/accept", {}),
        ("post", f"{base}/extraction/items/abc/reject", {"note": "x"}),
        ("post", f"{base}/extraction/records/r1/arms", None),
        ("post", f"{base}/rob/abc/approve", {"overall": "low"}),
        (
            "post",
            f"{base}/studies/link",
            {"record_ids": ["r1"], "study_id": "S1", "primary_record_id": "r1", "note": "x"},
        ),
    ]
    for method, url, body in calls:
        for headers in ({}, auth("rvw_" + "0" * 32 + "_" + "x" * 43)):
            r = getattr(c, method)(url, json=body, headers=headers)
            assert r.status_code == 401, url
    r = c.put(f"{base}/fulltext/r1/pdf", content=synthetic_article())
    assert r.status_code == 401


def test_screening_adjudication_and_summary(c: TestClient) -> None:
    summary = c.get(f"/projects/{PID}/screening/title_abstract/summary").json()
    assert summary["paired_records"] == 3 and summary["unresolved_conflicts"] == 0
    assert "not equivalent to independent human reviewers" in summary["methods_statement"]
    tok = token(c)
    url = f"/projects/{PID}/screening/title_abstract/records/r3/adjudication"
    r = c.post(
        url,
        json={
            "verdict": "exclude",
            "rationale": "animal model",
            "exclusion_reason": "E1: Animal studies",
        },
        headers=auth(tok),
    )
    assert r.status_code == 201 and r.json()["reviewer_identity"] == "Dr API"
    assert r.json()["is_human"] is True
    again = c.post(url, json={"verdict": "include", "rationale": "x"}, headers=auth(tok))
    assert again.status_code == 409


def test_upload_manifest_extraction_review_and_export(c: TestClient) -> None:
    tok = token(c)
    manifest = c.get(f"/projects/{PID}/fulltext/manifest", params={"format": "csv"})
    rows = list(csv.DictReader(io.StringIO(manifest.text)))
    assert [r["record_id"] for r in rows] == ["r1", "r2"]
    assert rows[0]["doi"] == SYNTHETIC_DOI and rows[0]["status"] == "not_attempted"
    bad = c.put(f"/projects/{PID}/fulltext/r1/pdf", content=b"<html>", headers=auth(tok))
    assert bad.status_code == 422
    up = c.put(
        f"/projects/{PID}/fulltext/r1/pdf",
        content=synthetic_article(),
        headers=auth(tok) | {"X-Filename": "doe.pdf"},
    )
    assert up.status_code == 201 and up.json()["match_status"] == "confirmed"
    unob = c.post(
        f"/projects/{PID}/fulltext/r2/unobtainable",
        json={"reason": "not held by any library"},
        headers=auth(tok),
    )
    assert unob.status_code == 200 and unob.json()["recorded_by"] == "Dr API"
    flow = c.get(f"/projects/{PID}/fulltext/flow").json()
    assert flow["not_retrieved"] == ["r2"] and flow["assessed"] == 1

    with c.app.state.session_factory() as s:  # type: ignore[attr-defined]
        repo = Repository(s)
        item = repo.get_fulltext_item(PID, "r1")
        assert item is not None and item.pdf_filename and item.pdf_sha256
        doc = parse_pdf(synthetic_article(), filename=item.pdf_filename)
        (proposed,) = submit_proposals(
            repo,
            PID,
            "r1",
            doc,
            [km("reported_estimate", "62.1", 1, 1, 3)],
            proposed_by=ActorType.AGENT,
            proposer_identity="llm:t",
        )
        s.commit()
    items = c.get(f"/projects/{PID}/extraction/items").json()
    assert items[0]["verification"]["status"] == "verified"
    correction = km("reported_estimate", "62.1", 1, 1, 3).model_dump(mode="json")
    correction["page"] = 1  # wrong page: the API must refuse an ungrounded correction
    r = c.post(
        f"/projects/{PID}/extraction/items/{proposed.item_id}/correct",
        json={"correction": correction, "note": "x"},
        headers=auth(tok),
    )
    assert r.status_code == 409 and "wrong_page" in r.json()["detail"]
    ok = c.post(
        f"/projects/{PID}/extraction/items/{proposed.item_id}/accept", json={}, headers=auth(tok)
    )
    assert ok.status_code == 200 and ok.json()["reviewed_by"] == "Dr API"
    hist = c.get(f"/projects/{PID}/extraction/items/{proposed.item_id}/history").json()
    assert [h["action"] for h in hist] == ["proposed", "accepted"]
    arms = c.post(f"/projects/{PID}/extraction/records/r1/arms", headers=auth(tok))
    assert arms.status_code == 201 and arms.json()[0]["approved_by"] == "Dr API"
    export = c.get(f"/projects/{PID}/extraction/export", params={"format": "csv"})
    assert export.headers["content-type"].startswith("text/csv")
    row = next(csv.DictReader(io.StringIO(export.text)))
    assert row["page"] == "2" and row["verification_status"] == "verified"
    assert c.get(f"/projects/{PID}/extraction/export").json()[0]["reported_text"] == "62.1"


def test_analysis_plan_approval_is_authenticated(c: TestClient) -> None:
    base = f"/projects/{PID}/stats"
    body = {"analyses": [{"analysis_id": "md", "outcome_id": "hba1c", "effect_measure": "MD"}]}
    assert c.post(f"{base}/plan", json=body).status_code == 401
    assert c.get(f"{base}/plan").status_code == 404
    tok = token(c)
    impersonation = body | {"approved_by": "Someone Else"}
    assert c.post(f"{base}/plan", json=impersonation, headers=auth(tok)).status_code == 422
    wrong = {"analyses": [{"analysis_id": "rr", "outcome_id": "hba1c", "effect_measure": "RR"}]}
    r = c.post(f"{base}/plan", json=wrong, headers=auth(tok))
    assert r.status_code == 409 and "not admissible" in r.json()["detail"]
    r = c.post(f"{base}/plan", json=body, headers=auth(tok))
    assert r.status_code == 201 and r.json()["approved_by"] == "Dr API"
    again = c.post(f"{base}/plan", json=body, headers=auth(tok))
    assert again.status_code == 201 and again.json()["version"] == 2
    assert c.get(f"{base}/plan").json()["version"] == 2
    assert c.get(f"{base}/output").status_code == 404
