"""Session 6: authenticated Stage 6 HTTP endpoints and the Streamlit manuscript page.

All data SYNTHETIC. Reference lookups use in-memory fake clients (no network).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api_server.app import create_app
from src.api_server.manuscript_routes import ReferenceClients
from src.core.config import get_settings
from src.core.enums import Phase
from src.state.repository import Repository
from tests.manuscript_helpers import fill_author_input, single_arm_project
from tests.unit.test_manuscript_workflow import MATCH, WORK, FakeCrossref, FakePubMed


@pytest.fixture
def api(tmp_path: Path) -> Iterator[tuple[TestClient, dict[str, str]]]:
    app = create_app(f"sqlite:///{tmp_path / 'm.db'}", tmp_path / "data")
    app.state.reference_clients = ReferenceClients(
        pubmed=lambda: FakePubMed([MATCH]),  # type: ignore[arg-type,return-value]
        crossref=lambda: FakeCrossref(WORK),  # type: ignore[arg-type,return-value]
    )
    with app.state.session_factory() as session:
        _, _, token = single_arm_project(session)
        session.commit()
    with TestClient(app) as client:
        yield client, {"Authorization": f"Bearer {token}"}


BASE = "/projects/ms1/manuscript"


def test_writes_require_authentication(api: tuple[TestClient, dict[str, str]]) -> None:
    client, _ = api
    for path, body in (
        (f"{BASE}/draft", {}),
        (f"{BASE}/audit", None),
        (f"{BASE}/approve", {"statement": "approve without token"}),
        (f"{BASE}/references", {"pmid": "91111111"}),
    ):
        assert client.post(path, json=body).status_code == 401
    # a body cannot name the acting reviewer
    assert (
        client.post(
            f"{BASE}/approve",
            json={"statement": "x" * 20, "reviewer_id": "r"},
            headers={"Authorization": "Bearer bad"},
        ).status_code
        == 401
    )


def test_http_workflow_draft_audit_ack_sign_off(api: tuple[TestClient, dict[str, str]]) -> None:
    client, auth = api
    assert client.post(f"{BASE}/draft", json={}, headers=auth).status_code == 200
    v1 = client.get(f"{BASE}/versions/1").json()
    changes = {s["section_id"]: fill_author_input(s["source"]) for s in v1["sections"]}
    changes["discussion"] += " Radiosurgery causes radiation necrosis."
    r = client.post(
        f"{BASE}/versions", json={"changes": changes, "summary": "author text"}, headers=auth
    )
    assert r.status_code == 200 and r.json()["version"] == 2
    assert "@@" in client.get(f"{BASE}/diff", params={"a": 1, "b": 2}).json()["diff"]
    # Session 7: every PRISMA 2020 item needs a current human decision (via the reporting API)
    for checklist in ("prisma_2020_main", "prisma_2020_abstract"):
        screen = client.get(f"/projects/ms1/reporting/checklists/{checklist}").json()
        rows = screen["items"]
        decisions = [
            {
                "item_id": r["item_id"],
                "status": "reported",
                "section_ids": [r["candidates"][0].split(":")[0]],
                "location": r["candidates"][0],
                "justification": "synthetic test decision",
                "based_on_decision_id": None,
            }
            if r["candidates"]
            else {
                "item_id": r["item_id"],
                "status": "not_applicable",
                "justification": "synthetic test decision",
                "based_on_decision_id": None,
            }
            for r in rows
        ]
        seen = {
            k: screen[k] for k in ("manuscript_version", "manuscript_sha256", "evidence_sha256")
        }
        url = f"/projects/ms1/reporting/checklists/{checklist}/items"
        assert client.post(url, json={"decisions": decisions, "seen": seen}).status_code == 401
        ok = client.post(url, json={"decisions": decisions, "seen": seen}, headers=auth)
        assert ok.status_code == 200, ok.text
    audit = client.post(f"{BASE}/audit", headers=auth).json()
    assert audit["passed"] and audit["requires_human_review"] >= 2
    reviews = [f["finding_id"] for f in audit["findings"] if f["status"] == "requires_human_review"]
    # sign-off is refused in the drafting phase and before acknowledgement
    assert (
        client.post(
            f"{BASE}/approve", json={"statement": "I approve version 2."}, headers=auth
        ).status_code
        == 409
    )
    ack = client.post(
        f"{BASE}/audits/{audit['report_id']}/findings/{reviews[0]}/acknowledge",
        json={"decision": "accepted_as_author_statement", "note": "supported by cited trials"},
        headers=auth,
    )
    assert ack.status_code == 200 and ack.json()["reviewer_name"] == "Dr Stats Reviewer"
    verified_id = next(f["finding_id"] for f in audit["findings"] if f["status"] == "verified")
    refused = client.post(  # all-or-nothing: a non-review finding refuses the whole request
        f"{BASE}/audits/{audit['report_id']}/acknowledge",
        json={
            "finding_ids": [*reviews[1:], verified_id],
            "decision": "confirmed_supported",
            "note": "bulk assessment by the author",
        },
        headers=auth,
    )
    assert refused.status_code == 409
    bulk = client.post(
        f"{BASE}/audits/{audit['report_id']}/acknowledge",
        json={
            "finding_ids": reviews[1:],
            "decision": "confirmed_supported",
            "note": "author statements checked against the sources",
        },
        headers=auth,
    )
    assert bulk.status_code == 200 and len(bulk.json()) == len(reviews) - 1
    adv = client.post(
        "/projects/ms1/advance", json={"target": Phase.STAGE6_AUDIT.value}, headers=auth
    )
    assert adv.status_code == 200
    blocked = client.post("/projects/ms1/advance", json={"target": "completed"}, headers=auth)
    assert blocked.status_code == 409 and "sign-off" in str(blocked.json())
    ok = client.post(f"{BASE}/approve", json={"statement": "I approve version 2."}, headers=auth)
    assert ok.status_code == 200
    done = client.post("/projects/ms1/advance", json={"target": "completed"}, headers=auth)
    assert done.status_code == 200 and done.json()["phase"] == "completed"
    md = client.get(f"{BASE}/export", params={"format": "md"})
    assert md.status_code == 200 and md.text.startswith("# Radiation necrosis")
    docx = client.get(f"{BASE}/export", params={"format": "docx"})
    assert docx.status_code == 200 and docx.content[:2] == b"PK"
    exported = client.get(f"{BASE}/audits/{audit['report_id']}/export").text
    assert "Human acknowledgement by Dr Stats Reviewer" in exported
    # completed: the manuscript is frozen
    assert client.post(f"{BASE}/rerender", headers=auth).status_code == 409


def test_reference_import_verification_and_excerpt(api: tuple[TestClient, dict[str, str]]) -> None:
    client, auth = api
    r = client.post(f"{BASE}/references", json={"pmid": "91111111"}, headers=auth)
    assert r.status_code == 200 and r.json()["metadata_source"] == "pubmed_efetch"
    assert client.post(f"{BASE}/references", json={}, headers=auth).status_code == 409
    results = client.post(f"{BASE}/references/verify", headers=auth).json()
    assert {v["status"] for v in results if v["ref_key"] == "pmid-91111111"} == {"verified"}
    ex = client.post(
        f"{BASE}/references/pmid-91111111/excerpts",
        headers=auth,
        json={"text": "Synthetic excerpt of the cited source text.", "location": "p 2"},
    )
    assert ex.status_code == 200
    missing = client.post(
        f"{BASE}/references/nope/excerpts",
        headers=auth,
        json={"text": "Synthetic excerpt of the cited source.", "location": "p"},
    )
    assert missing.status_code == 404
    refs = {x["ref_key"]: x for x in client.get(f"{BASE}/evidence").json()["references"]}
    assert refs["pmid-91111111"]["identity"] == "verified" and refs["pmid-91111111"]["excerpts"]


def test_ui_manuscript_page(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("streamlit")
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    from src.manuscript.service import ManuscriptService
    from src.state.db import init_db, make_engine, make_session_factory
    from tests.manuscript_helpers import SYSTEM

    url = f"sqlite:///{tmp_path / 'ui.db'}"
    monkeypatch.setenv("SYSREV_DATABASE_URL", url)
    monkeypatch.setenv("SYSREV_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("SYSREV_API_URL", raising=False)
    get_settings.cache_clear()
    st.cache_resource.clear()
    engine = make_engine(url)
    init_db(engine)
    with make_session_factory(engine)() as s:
        repo, _, _ = single_arm_project(s)
        ManuscriptService(repo, "ms1").create_template_draft(SYSTEM)
        s.commit()
    app_path = str(Path(__file__).resolve().parents[2] / "src" / "ui" / "app.py")
    at = AppTest.from_file(app_path, default_timeout=60)
    at.session_state["project_id"] = "ms1"
    at.session_state["page"] = "Manuscript"
    at.run()
    assert not at.exception
    assert any("Radiation necrosis" in m.value for m in at.markdown)
    blocked = [b for b in at.button if "audit" in b.label.lower()]
    assert blocked
    blocked[0].click().run()  # no reviewer token: refused by the API
    assert not at.exception
    assert any("Not authenticated" in e.value for e in at.error)
    with make_session_factory(engine)() as s:
        assert Repository(s).latest_audit_report("ms1") is None
    engine.dispose()
    get_settings.cache_clear()
    st.cache_resource.clear()
