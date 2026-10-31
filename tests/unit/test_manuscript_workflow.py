"""Session 6: model-assisted drafting and advisory claim review (mocked HTTP only), live
reference verification failure modes (mocked clients), and exports. All data SYNTHETIC.

No real API call is made: the Anthropic adapter is exercised against respx-mocked HTTP, and
PubMed/Crossref lookups use in-memory fakes.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from sqlalchemy.orm import Session

from src.apis.crossref import CrossrefWork
from src.core.enums import CheckStatus
from src.core.schemas.manuscript import EvidenceExcerpt, ReferenceEntry
from src.core.schemas.study import StudyRecord
from src.fulltext.storage import DocumentStore
from src.llm.tasks import AITask, ProjectAISettings, TaskModelSettings
from src.manuscript.drafting import validate_section_text
from src.manuscript.evidence import build_evidence_package
from src.manuscript.export import audit_markdown, manuscript_docx, manuscript_markdown
from src.manuscript.references import import_reference, verify_reference
from src.manuscript.service import ManuscriptService
from src.workflows import AIWorkflow
from tests.conftest import NOW, make_author, make_record
from tests.manuscript_helpers import (
    REVIEWER,
    SYSTEM,
    complete_manuscript,
    human,
    single_arm_project,
)
from tests.unit.test_ai_workflow import CAT, CREDS, KEY, a_reply
from tests.unit.test_llm_adapters import ANTHROPIC_URL

MODEL = "claude-opus-5-5"


def _settings() -> ProjectAISettings:
    cfg = TaskModelSettings(
        provider="anthropic", model_id=MODEL, effort="low", max_output_tokens=4096
    )
    return ProjectAISettings(tasks={AITask.MANUSCRIPT_DRAFTING: cfg, AITask.CLAIM_REVIEW: cfg})


def _workflow(session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    repo, rid, _ = single_arm_project(session)
    repo.save_ai_settings("ms1", _settings())
    wf = AIWorkflow(repo, "ms1", catalogue=CAT, credentials=CREDS(), store=DocumentStore(tmp_path))
    return repo, rid, wf


GOOD_DISCUSSION = (
    "**Principal findings.** Across {{stats.nec.k}} case series, the pooled crude proportion of "
    "radiation necrosis was {{stats.nec.pooled_estimate}} [@study-s1; @study-s2]. "
    "[[AUTHOR INPUT: comparison with verified literature]]"
)


def test_old_settings_without_new_tasks_remain_valid() -> None:
    old = {"tasks": {"extraction": {"provider": "anthropic", "model_id": MODEL}}}
    s = ProjectAISettings.model_validate(old)
    assert AITask.MANUSCRIPT_DRAFTING not in s.tasks
    assert {"manuscript_drafting", "claim_review"} <= {t.value for t in AITask}


@respx.mock
def test_model_drafting_saves_validated_section_with_provenance(
    session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, _, wf = _workflow(session, tmp_path, monkeypatch)
    ManuscriptService(repo, "ms1").create_template_draft(SYSTEM)
    route = respx.post(ANTHROPIC_URL).mock(
        side_effect=lambda req: a_reply(
            json.dumps({"text": GOOD_DISCUSSION, "author_input_needed": []}),
            json.loads(req.content)["model"],
        )
    )
    report = wf.run_manuscript_drafting(["discussion"])
    assert route.call_count == 1 and not report.failed, report
    sent = json.loads(route.calls[0].request.content)
    assert sent["model"] == MODEL and sent["output_config"]["effort"] == "low"
    prompt = sent["messages"][0]["content"]
    assert "stats.nec.pooled_estimate" in prompt and KEY not in prompt
    v2 = repo.latest_manuscript_version("ms1")
    assert v2 is not None and v2.version == 2
    disc = v2.section("discussion")
    assert disc is not None and disc.origin == "llm"
    prov = v2.provenance[0]
    assert (prov.method, prov.provider, prov.model_id, prov.effort) == (
        "llm",
        "anthropic",
        MODEL,
        "low",
    )
    calls = repo.load_llm_calls("ms1")
    assert [c.call_id for c in calls] == prov.llm_call_ids
    assert calls[0].task == "manuscript_drafting" and calls[0].input_tokens == 1000
    assert "5.9%" in v2.rendered_markdown  # inserted by the renderer, not typed by the model
    # the model's draft is not approved by anything: the audit still runs independently
    audit = ManuscriptService(repo, "ms1").audit()
    assert not audit.passed  # other sections still contain author-input markers


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("Necrosis occurred in about 6% of patients.", "number was typed"),
        ("Necrosis was common [@pmid-12345678].", "unknown reference"),
        ("Smith et al. found {{stats.nec.pooled_estimate}}.", "author citation"),
        ("Necrosis was {{stats.nec.pooled_estimate_typo}}.", "unknown fact key"),
        ("See doi: 10.5555/abc for details.", "number was typed"),
        ("As shown before [3], necrosis is rare.", "typed numbers"),
    ],
)
@respx.mock
def test_model_drafting_rejects_fabrication(
    session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str, fragment: str
) -> None:
    repo, _, wf = _workflow(session, tmp_path, monkeypatch)
    ManuscriptService(repo, "ms1").create_template_draft(SYSTEM)
    respx.post(ANTHROPIC_URL).mock(
        side_effect=lambda req: a_reply(
            json.dumps({"text": text}), json.loads(req.content)["model"]
        )
    )
    report = wf.run_manuscript_drafting(["introduction"])
    assert report.failed and fragment in report.failed[0], report
    latest = repo.latest_manuscript_version("ms1")
    assert latest is not None and latest.version == 1  # nothing saved
    assert repo.load_llm_calls("ms1")[0].status == "ok"  # billed call is still logged


def test_drafting_requires_configuration_and_stage(session: Session, tmp_path: Path) -> None:
    repo, _, _ = single_arm_project(session)
    wf = AIWorkflow(repo, "ms1", catalogue=CAT, credentials=CREDS(), store=DocumentStore(tmp_path))
    ManuscriptService(repo, "ms1").create_template_draft(SYSTEM)
    assert wf.preflight(AITask.MANUSCRIPT_DRAFTING)  # no settings: nothing can run or bill
    repo.save_ai_settings("ms1", ProjectAISettings())
    with pytest.raises(Exception, match="no model configured"):
        wf.run_manuscript_drafting(["introduction"])


def test_validator_accepts_named_tools_but_not_numbers(session: Session) -> None:
    repo, _, _ = single_arm_project(session)
    p = build_evidence_package(repo, "ms1")
    assert validate_section_text("Reported following PRISMA 2020 (Table 1, Figure 2).", p) == []
    assert validate_section_text("Twelve of {{prisma.identified}} records.", p) == []
    assert validate_section_text("Of 12 records.", p)


@respx.mock
def test_advisory_claim_review_never_changes_findings(
    session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, rid, wf = _workflow(session, tmp_path, monkeypatch)
    svc = ManuscriptService(repo, "ms1")
    repo.save_reference_entry(
        "ms1",
        ReferenceEntry(
            ref_key="pmid-91111111",
            origin="imported_literature",
            pmid="91111111",
            title="Synthetic necrosis cohort",
            metadata_source="pubmed_efetch",
            retrieved_at=NOW,
            added_by=REVIEWER,
            reviewer_id=rid,
        ),
    )
    repo.save_evidence_excerpt(
        "ms1",
        EvidenceExcerpt(
            ref_key="pmid-91111111",
            text="Radiation necrosis risk increased with lesion volume.",
            location="p 3",
            added_by=REVIEWER,
            reviewer_id=rid,
        ),
    )
    actor = human(repo, rid)
    complete_manuscript(
        svc,
        actor,
        introduction="Radiation necrosis risk increased with lesion "
        "volume [@pmid-91111111]. {{protocol.question}}",
    )
    report = svc.audit()
    target = [f for f in report.findings if f.scope == "claim_support" and f.needs_human]
    assert target
    respx.post(ANTHROPIC_URL).mock(
        side_effect=lambda req: a_reply(
            json.dumps(
                {"verdict": "does_not_support", "rationale": "the excerpt is about volume only"}
            ),
            json.loads(req.content)["model"],
        )
    )
    step = wf.run_claim_review()
    assert step.done and not step.failed, step
    reviews = repo.load_claim_reviews("ms1")
    assert reviews[0].is_human is False and reviews[0].verdict == "does_not_support"
    again = repo.latest_audit_report("ms1")
    assert again is not None and again.results_sha256() == report.results_sha256()
    assert "Advisory model review (not a human assessment" in audit_markdown(report, [], reviews)


# --------------------------------------------------------------------------- references


class FakePubMed:
    def __init__(self, result: list[StudyRecord] | Exception) -> None:
        self.result = result

    async def efetch_records(
        self, pmids: Any, *, query: str = "", total_hits: int | None = None
    ) -> list[StudyRecord]:
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeCrossref:
    def __init__(self, result: CrossrefWork | Exception | None) -> None:
        self.result = result

    async def get_work(self, doi: str) -> CrossrefWork | None:
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


ENTRY = ReferenceEntry(
    ref_key="pmid-91111111",
    origin="imported_literature",
    pmid="91111111",
    doi="10.5555/synthetic.ref",
    title="Synthetic cohort of radiosurgery outcomes",
    authors=["Hazel C"],
    journal="Synthetic Neuro-Oncology",
    year=2019,
    metadata_source="pubmed_efetch",
    retrieved_at=NOW,
    added_by=REVIEWER,
    reviewer_id="r",
)
MATCH = make_record(
    "x",
    "Synthetic cohort of radiosurgery outcomes",
    pmid="91111111",
    doi="10.5555/synthetic.ref",
    publication_year=2019,
    authors=[make_author("Hazel", "Cara")],
    journal="Synthetic Neuro-Oncology",
)
WORK = CrossrefWork(
    doi="10.5555/synthetic.ref",
    title="Synthetic cohort of radiosurgery outcomes",
    container_title="Synthetic Neuro-Oncology",
    year=2019,
    first_author_family="Hazel",
    type="journal-article",
)


async def test_verification_statuses() -> None:
    ok = await verify_reference(
        ENTRY, pubmed=FakePubMed([MATCH]), crossref=FakeCrossref(WORK), checked_by="t"
    )
    assert {v.status for v in ok} == {"verified"}
    down = await verify_reference(
        ENTRY,
        pubmed=FakePubMed(httpx.ConnectError("offline")),
        crossref=FakeCrossref(httpx.ConnectError("offline")),
        checked_by="t",
    )
    assert {v.status for v in down} == {"unavailable"}  # never a verification
    gone = await verify_reference(
        ENTRY, pubmed=FakePubMed([]), crossref=FakeCrossref(None), checked_by="t"
    )
    assert {v.status for v in gone} == {"not_found"}
    wrong = MATCH.model_copy(
        update={"title": "An entirely different synthetic paper", "doi": "10.5555/other"}
    )
    bad = await verify_reference(
        ENTRY, pubmed=FakePubMed([wrong]), crossref=FakeCrossref(WORK), checked_by="t"
    )
    pub = next(v for v in bad if v.source == "pubmed")
    assert pub.status == "mismatch" and "MISMATCH" in pub.field_results["title"]
    assert "MISMATCH" in pub.field_results["doi"]  # DOI/PMID inconsistency between sources
    none = await verify_reference(ENTRY, pubmed=None, crossref=None, checked_by="t")
    assert {v.status for v in none} == {"unavailable"}


async def test_import_reference_never_from_memory() -> None:
    with pytest.raises(Exception, match="never entered from memory"):
        await import_reference(
            pmid=None, doi=None, pubmed=None, crossref=None, added_by="x", reviewer_id=None
        )
    with pytest.raises(Exception, match="not found"):
        await import_reference(
            pmid="91111111",
            doi=None,
            pubmed=FakePubMed([]),
            crossref=None,
            added_by="x",
            reviewer_id=None,
        )
    entry = await import_reference(
        pmid="91111111",
        doi=None,
        pubmed=FakePubMed([MATCH]),
        crossref=None,
        added_by="x",
        reviewer_id=None,
    )
    assert (entry.title, entry.metadata_source, entry.authors) == (
        MATCH.title,
        "pubmed_efetch",
        ["Hazel C"],
    )


def test_unavailable_or_mismatched_verification_blocks_audit(session: Session) -> None:
    repo, rid, _ = single_arm_project(session)
    svc = ManuscriptService(repo, "ms1")
    entry = ENTRY.model_copy(update={"reviewer_id": rid})
    repo.save_reference_entry("ms1", entry)
    from src.core.schemas.manuscript import ReferenceVerification

    for status, expected in (
        ("unavailable", CheckStatus.NOT_CHECKED),
        ("mismatch", CheckStatus.MISMATCH),
        ("not_found", CheckStatus.NOT_FOUND),
    ):
        repo.save_reference_verification(
            "ms1",
            ReferenceVerification(
                ref_key=entry.ref_key,
                metadata_sha256=entry.metadata_sha256(),
                source="crossref",
                status=status,
                checked_by="t",  # type: ignore[arg-type]
            ),
        )
        complete_manuscript(
            svc, human(repo, rid), introduction="Background [@pmid-91111111]. {{protocol.question}}"
        )
        report = svc.audit()
        statuses = [c.status for c in report.citation_checks if c.pmid == "91111111"]
        assert statuses == [expected] and not report.passed


# --------------------------------------------------------------------------- exports


def test_exports(session: Session) -> None:
    docx = pytest.importorskip("docx")
    repo, rid, _ = single_arm_project(session)
    svc = ManuscriptService(repo, "ms1")
    version = complete_manuscript(svc, human(repo, rid))
    md = manuscript_markdown(version)
    assert md.startswith("# Radiation necrosis") and "## References" in md
    document = docx.Document(io.BytesIO(manuscript_docx(version)))
    text = "\n".join(p.text for p in document.paragraphs)
    assert "References" in text and "5.9% (95% CI" in text
    # Table 1 characteristics, Table 2 analyses, Table 3 single-group evidence summary
    assert len(document.tables) == 3
    cells = [c.text for row in document.tables[1].rows for c in row.cells]
    assert "5.9% (95% CI 3.3% to 10.3%)" in cells  # identical to the stats-derived value
    report = svc.audit()
    out = audit_markdown(report)
    assert "deterministic checks passed: **True**" in out and "Limitations of the audit" in out
