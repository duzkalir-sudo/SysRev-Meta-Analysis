"""Session 7: authenticated reporting API and compatibility with Session 6 records.
All data SYNTHETIC; no network access."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api_server.app import create_app
from src.core.schemas.audit import AuditReport
from src.manuscript.service import ManuscriptService, completion_problems
from src.state.db import AuditReportORM, init_db, make_engine, make_session_factory
from src.state.repository import Repository
from tests.manuscript_helpers import comparative_project, complete_manuscript, human

BASE = "/projects/mc1/reporting"


@pytest.fixture
def api(tmp_path: Path) -> Iterator[tuple[TestClient, dict[str, str]]]:
    url = f"sqlite:///{tmp_path / 'api.db'}"
    engine = make_engine(url)
    init_db(engine)
    with make_session_factory(engine)() as s:
        repo, rid, token = comparative_project(s)
        complete_manuscript(ManuscriptService(repo, "mc1"), human(repo, rid))
        s.commit()
    client = TestClient(create_app(url, tmp_path / "data"))
    yield client, {"Authorization": f"Bearer {token}"}
    engine.dispose()


def test_diagram_exports_and_unsupported_variants(api: tuple[TestClient, dict[str, str]]) -> None:
    client, _ = api
    svg = client.get(f"{BASE}/prisma/diagram")
    assert svg.status_code == 200 and svg.headers["content-type"].startswith("image/svg+xml")
    assert svg.headers["x-diagram-status"] == "consistent" and "Studies included" in svg.text
    png = client.get(f"{BASE}/prisma/diagram", params={"format": "png"})
    assert png.status_code == 200 and png.content[:4] == b"\x89PNG"
    spec = client.get(f"{BASE}/prisma/diagram", params={"format": "json"}).json()
    assert spec["counts"]["studies_included"] == 3 and spec["variant"].startswith("new_review")
    bad = client.get(
        f"{BASE}/prisma/diagram", params={"variant": "updated_review_databases_registers"}
    )
    assert bad.status_code == 422
    assert any("previous version" in f for f in bad.json()["detail"]["unsupported_fields"])


def test_decisions_require_a_token_and_never_name_the_reviewer(
    api: tuple[TestClient, dict[str, str]],
) -> None:
    client, auth = api
    body = {"status": "not_applicable", "justification": "no funding for this synthetic review"}
    url = f"{BASE}/checklists/prisma_2020_main/items/25"
    assert client.post(url, json=body).status_code == 401
    assert client.post(url, json=body | {"reviewer_name": "x"}, headers=auth).status_code == 422
    ok = client.post(url, json=body, headers=auth)
    assert ok.status_code == 200 and ok.json()["reviewer_name"] == "Dr Stats Reviewer"
    grade = {
        "status": "approved",
        "approach": "interventions_randomised",
        "framework_rationale": "all contributing studies are randomised trials",
        "domains": [
            {
                "domain": d,
                "concern": "not_serious",
                "rationale": "synthetic rationale for this domain",
            }
            for d in (
                "risk_of_bias",
                "inconsistency",
                "indirectness",
                "imprecision",
                "publication_bias",
            )
        ],
        "final_certainty": "high",
        "summary": "no domain rated down (synthetic)",
    }
    assert client.post(f"{BASE}/grade/comp", json=grade).status_code == 401
    r = client.post(f"{BASE}/grade/comp", json=grade, headers=auth)
    assert r.status_code == 200 and r.json()["implied_certainty"] == "high"
    view = client.get(f"{BASE}/grade").json()[0]
    assert view["status"] == "approved"
    sof = client.get(f"{BASE}/sof")
    assert sof.status_code == 200 and "Summary of Findings" in sof.text
    assert client.get(f"{BASE}/sof", params={"format": "docx"}).content[:2] == b"PK"
    status = client.get(f"{BASE}/status").json()
    assert status["grade"] == {"comp": "approved"} and status["blockers"]
    for fmt in ("csv", "docx", "xlsx"):
        exp = client.get(f"{BASE}/checklists/prisma_2020_main/export", params={"format": fmt})
        assert exp.status_code == 200 and exp.content
    assert (
        client.post(
            f"{BASE}/swim",
            json={
                "applicability": "applicable",
                "rationale": "no synthesis without MA here (synthetic)",
            },
            headers=auth,
        ).status_code
        == 409
    )


def test_legacy_audit_reports_still_load_but_cannot_authorise_completion(
    api: tuple[TestClient, dict[str, str]], tmp_path: Path
) -> None:
    client, auth = api
    current = client.post("/projects/mc1/manuscript/audit", headers=auth).json()
    legacy = {
        k: v
        for k, v in current.items()
        if k not in ("passed", "failures", "complete", "requires_human_review")
    }
    legacy |= {
        "report_id": "legacy1",
        "auditor_version": "stage6-auditor-1.2.0",
        "findings": [f for f in current["findings"] if f["scope"] != "reporting"],
    }
    legacy["scopes_checked"] = {
        k: v for k, v in current["scopes_checked"].items() if k != "reporting"
    }
    report = AuditReport.model_validate(legacy)
    assert report.complete  # judged by the scopes its own Auditor version required
    engine = make_engine(f"sqlite:///{tmp_path / 'api.db'}")
    with make_session_factory(engine)() as s:
        s.add(
            AuditReportORM(
                id="legacy1",
                project_id="mc1",
                manuscript_version=2,
                passed=True,
                generated_at=report.generated_at.replace(year=2099),
                payload=report.model_dump(mode="json"),
            )
        )
        s.commit()
        repo = Repository(s)
        assert repo.latest_audit_report("mc1").report_id == "legacy1"  # type: ignore[union-attr]
        assert any("not the current Auditor" in p for p in completion_problems(repo, "mc1"))
    engine.dispose()
