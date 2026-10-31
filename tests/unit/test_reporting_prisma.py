"""Session 7: deterministic PRISMA 2020 flow diagrams and the PRISMA 2020 / SWiM checklists.

All projects, records and numbers are SYNTHETIC. Expected counts are written out by hand from
the accounting identities of the PRISMA 2020 flow diagram (Page MJ, et al. BMJ 2021;372:n71).
"""

from __future__ import annotations

import io
import json

import pytest
from sqlalchemy.orm import Session

from src.core.enums import CheckStatus, SourceDatabase
from src.core.exceptions import SysRevError
from src.core.schemas.pipeline import PRISMACounts
from src.core.schemas.reporting import ChecklistDecision
from src.manuscript.evidence import build_evidence_package
from src.manuscript.service import ManuscriptService, completion_problems
from src.reporting import prisma_flow as flow
from src.reporting.export import checklist_docx, checklist_xlsx
from src.reporting.service import ReportingService
from src.reporting.standards import PRISMA_ABSTRACT, PRISMA_MAIN, SWIM_ITEMS
from tests.manuscript_helpers import (
    REVIEWER,
    SYSTEM,
    complete_manuscript,
    confirm_checklists,
    decide_item,
    fill_author_input,
    human,
    single_arm_project,
)


def counts(**kw: object) -> PRISMACounts:
    base: dict[str, object] = {
        "identified_by_database": {SourceDatabase.PUBMED: 120, SourceDatabase.EMBASE: 80},
        "identified_from_registers": 15,
        "duplicates_removed": 45,
        "records_screened": 170,
        "records_excluded": 140,
        "reports_sought": 30,
        "reports_not_retrieved": 3,
        "reports_assessed": 27,
        "reports_excluded_by_reason": {"wrong population": 9, "no outcome data": 6},
        "reports_included": 12,
        "studies_included": 10,
    }
    base.update(kw)
    return PRISMACounts.model_validate(base)


def box(spec: dict, box_id: str) -> list[str]:
    return next(b["lines"] for b in spec["boxes"] if b["id"] == box_id)


SEARCHED = frozenset({"pubmed", "embase", "clinicaltrials_gov"})


# --------------------------------------------------------------------------- diagram


def test_diagram_counts_and_accounting_identities() -> None:
    c = counts()
    spec = flow.build_flow_spec(c, searched_sources=SEARCHED, problems=[])
    assert spec["status"] == "consistent" and not spec["warnings"]
    k = spec["counts"]
    # identification: 120 + 80 databases, 15 registers = 215 records
    assert box(spec, "identified")[:4] == [
        "Records identified from*:",
        "Databases (n = 200)",
        "  Embase (n = 80)",
        "  PubMed (n = 120)",
    ]
    assert "Registers (n = 15)" in box(spec, "identified")
    # 215 − 45 duplicates − 0 − 0 = 170 screened; 170 − 140 = 30 sought; 30 − 3 = 27 assessed;
    # 27 − (9 + 6) = 12 reports of 10 studies
    assert k["identified_databases"] + k["identified_registers"] - k["duplicates_removed"] == 170
    assert k["records_screened"] - k["records_excluded"] == k["reports_sought"] == 30
    assert k["reports_sought"] - k["reports_not_retrieved"] == k["reports_assessed"] == 27
    assert k["reports_assessed"] - k["reports_excluded"] == k["reports_included"] == 12
    assert (k["studies_included"], k["reports_included"]) == (10, 12)  # multiple reports
    assert box(spec, "included") == [
        "Studies included in review",
        "(n = 10)",
        "Reports of included studies",
        "(n = 12)",
    ]
    # zero-valued legitimate fields are shown as zero, not dropped
    removed = " ".join(x.strip() for x in box(spec, "removed"))
    assert "Records marked as ineligible by automation tools (n = 0)" in removed
    assert "Records removed for other reasons (n = 0)" in removed
    # reasons sorted by count
    assert box(spec, "reports_excluded")[1:] == [
        "  wrong population (n = 9)",
        "  no outcome data (n = 6)",
    ]


def test_registers_not_searched_are_not_shown_as_zero() -> None:
    c = counts(identified_from_registers=0, duplicates_removed=30)
    spec = flow.build_flow_spec(c, searched_sources=frozenset({"pubmed", "embase"}), problems=[])
    assert "Registers: none searched" in box(spec, "identified")
    searched = flow.build_flow_spec(c, searched_sources=SEARCHED, problems=[])
    assert "Registers (n = 0)" in box(searched, "identified")  # searched, nothing found


def test_svg_is_deterministic_and_regenerable_from_source() -> None:
    spec = flow.build_flow_spec(counts(), searched_sources=SEARCHED, problems=[])
    svg = flow.svg_from_spec(spec)
    assert svg == flow.svg_from_spec(json.loads(flow.spec_json(spec)))
    assert svg.startswith("<?xml") and "Studies included in review" in svg
    assert "WARNING" not in svg
    assert spec["snapshot_sha256"] in svg  # provenance in <desc>
    png = flow.png_from_spec(spec, dpi=72)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_inconsistent_or_non_final_snapshot_shows_a_visible_warning() -> None:
    pending = counts(records_screened=160, records_awaiting_screening=10, records_excluded=130)
    spec = flow.build_flow_spec(
        pending,
        searched_sources=SEARCHED,
        problems=["PRISMA records_screened is 160 but ... imply 170"],
    )
    assert spec["status"] == "warning"
    assert any("not final" in w for w in spec["warnings"])
    svg = flow.svg_from_spec(spec)
    assert "WARNING: PRISMA records_screened" in svg and "WARNING: the PRISMA snapshot" in svg


@pytest.mark.parametrize("variant", [v for v in flow.VARIANTS if v != flow.SUPPORTED_VARIANT])
def test_unsupported_variants_report_their_fields(variant: str) -> None:
    with pytest.raises(flow.UnsupportedVariantError) as exc:
        flow.build_flow_spec(counts(), searched_sources=SEARCHED, problems=[], variant=variant)
    assert exc.value.fields and exc.value.reason
    if "other_methods" in variant:
        assert any("Citation searching" in f for f in exc.value.fields)
    if variant.startswith("updated"):
        assert any("previous version" in f for f in exc.value.fields)


def test_unclassified_source_and_automation_breakdown() -> None:
    c = counts(identified_by_database={SourceDatabase.PUBMED: 120, SourceDatabase.MANUAL: 80})
    spec = flow.build_flow_spec(
        c,
        searched_sources=SEARCHED,
        problems=[],
        breakdown=flow.ExclusionBreakdown(by_human=100, by_automation=40),
    )
    assert any("manual" in w and "source type is not recorded" in w for w in spec["warnings"])
    assert "  by automation tools (n = 40)" in box(spec, "excluded")
    none = flow.build_flow_spec(
        counts(), searched_sources=SEARCHED, problems=[], breakdown=flow.ExclusionBreakdown(140, 0)
    )
    assert box(none, "excluded") == ["Records excluded**", "(n = 140)"]
    bad = flow.build_flow_spec(
        counts(), searched_sources=SEARCHED, problems=[], breakdown=flow.ExclusionBreakdown(10, 5)
    )
    assert any("does not add up" in w for w in bad["warnings"])


# --------------------------------------------------------------------------- project


def test_project_diagram_matches_persisted_records_and_study_links(session: Session) -> None:
    """Six reports; s1 and s2 are documented reports of one study → 5 studies, 6 reports."""
    repo, _, _ = single_arm_project(session, study_links=[["s1", "s2"]])
    out = ReportingService(repo, "ms1").flow()
    k = out["spec"]["counts"]
    stored = repo.load_prisma("ms1")
    assert stored is not None
    assert k == flow.flow_counts(stored)
    assert (k["studies_included"], k["reports_included"]) == (5, 6)
    assert out["spec"]["status"] == "consistent"
    assert "Registers: none searched" in box(out["spec"], "identified")


def test_stale_prisma_snapshot_blocks_and_is_visible(session: Session) -> None:
    repo, rid, _ = single_arm_project(session)
    stored = repo.load_prisma("ms1")
    assert stored is not None
    # an internally consistent but wrong snapshot: one study fewer than the links document
    repo.save_prisma("ms1", stored.model_copy(update={"studies_included": 5}))
    out = ReportingService(repo, "ms1").flow()
    assert out["spec"]["status"] == "warning" and "WARNING" in out["svg"]
    svc = ManuscriptService(repo, "ms1")
    complete_manuscript(svc, human(repo, rid))
    report = svc.audit()
    assert not report.passed
    assert any(
        f.scope == "reporting" and "flow diagram warning" in f.message
        for f in report.findings
        if f.blocking
    )


# --------------------------------------------------------------------------- checklists


def test_catalogues_are_complete_and_verbatim() -> None:
    assert len(PRISMA_MAIN) == 42 and len(PRISMA_ABSTRACT) == 12 and len(SWIM_ITEMS) == 10
    ids = [i.item_id for i in PRISMA_MAIN]
    assert ids[:2] == ["1", "2"] and "13f" in ids and "24c" in ids and ids[-1] == "27"
    d13 = next(i for i in PRISMA_MAIN if i.item_id == "13d")
    assert d13.text.startswith(
        "Describe any methods used to synthesise results and provide a rationale for the choice(s)."
    )
    assert [i.item_id for i in SWIM_ITEMS][:3] == ["1a", "1b", "2"]


def _draft(session: Session) -> tuple:
    repo, rid, _ = single_arm_project(session)
    svc = ManuscriptService(repo, "ms1")
    svc.create_template_draft(SYSTEM)
    return repo, rid, svc, ReportingService(repo, "ms1")


def test_keyword_presence_is_never_compliance(session: Session) -> None:
    repo, rid, svc, rs = _draft(session)
    latest = repo.latest_manuscript_version("ms1")
    assert latest is not None
    svc.edit(
        human(repo, rid), {"title": "A systematic review of radiosurgery (synthetic)"}, "title"
    )
    data = rs.checklist("prisma_2020_main")
    title = next(r for r in data["items"] if r["item_id"] == "1")
    assert title["candidates"] and title["status"] == "requires_human_review"
    assert all(r["status"] != "reported" for r in data["items"])  # nothing auto-reported
    assert not data["complete"] and "not a claim" in data["note"]


def test_decisions_need_location_reviewer_and_current_basis(session: Session) -> None:
    repo, rid, svc, rs = _draft(session)
    with pytest.raises(ValueError, match="location"):
        decide_item(
            rs,
            "prisma_2020_main",
            "6",
            rid,
            status="reported",
            section_ids=[],
            location="",
            justification="search described",
            evidence_keys=[],
        )
    with pytest.raises(SysRevError):  # an unknown (e.g. automated) identity
        decide_item(
            rs,
            "prisma_2020_main",
            "6",
            "agent",
            status="reported",
            section_ids=["methods"],
            location="Methods, Information sources",
            justification="search described",
            evidence_keys=[],
        )
    d = decide_item(
        rs,
        "prisma_2020_main",
        "6",
        rid,
        status="reported",
        section_ids=["methods"],
        location="Methods, Information sources and search",
        justification="all databases and the last search date are stated",
        evidence_keys=["search.statement"],
    )
    # a forged decision (wrong basis hash) is refused by the repository
    forged = ChecklistDecision.model_validate(
        d.model_dump() | {"decision_id": "x1", "item_id": "7", "basis_sha256": "0" * 64}
    )
    with pytest.raises(SysRevError, match="basis hash"):
        repo.save_checklist_decision(forged)
    item6 = next(r for r in rs.checklist("prisma_2020_main")["items"] if r["item_id"] == "6")
    assert item6["status"] == "reported" and item6["decided"]
    # editing the located section makes the decision stale
    latest = repo.latest_manuscript_version("ms1")
    assert latest is not None
    methods = latest.section("methods")
    assert methods is not None
    svc.edit(human(repo, rid), {"methods": methods.source + " Added sentence."}, "edit")
    item6 = next(r for r in rs.checklist("prisma_2020_main")["items"] if r["item_id"] == "6")
    assert item6["stale"] and item6["status"] != "reported"


def test_missing_items_block_and_complete_record_passes(session: Session) -> None:
    repo, rid, _ = single_arm_project(session)
    svc = ManuscriptService(repo, "ms1")
    latest = svc.create_template_draft(SYSTEM)
    svc.edit(
        human(repo, rid),
        {s.section_id: fill_author_input(s.source) for s in latest.sections},
        "author input",
    )
    report = svc.audit()
    missing = [f for f in report.findings if f.scope == "reporting" and f.blocking]
    assert len(missing) == 54 and all("no human decision" in f.message for f in missing)
    confirm_checklists(repo, "ms1", rid)
    rs = ReportingService(repo, "ms1")
    # a reviewer records an item as not reported: it blocks
    decide_item(
        rs,
        "prisma_2020_main",
        "25",
        rid,
        status="missing",
        section_ids=[],
        location="",
        justification="funding statement not yet written",
        evidence_keys=[],
    )
    r = svc.audit()
    assert [f.message for f in r.findings if f.blocking] == [
        "a reviewer recorded PRISMA 2020 item 25 (Support) as not reported"
    ]
    decide_item(
        rs,
        "prisma_2020_main",
        "25",
        rid,
        status="partially_reported",
        section_ids=["supplementary"],
        location="Supplementary material",
        justification="sources of support listed; role of funders not described",
        evidence_keys=[],
    )
    r = svc.audit()
    assert r.passed and any(f.scope == "reporting" and f.needs_human for f in r.findings)


def test_checklist_decision_after_audit_invalidates_it(session: Session) -> None:
    repo, rid, _ = single_arm_project(session)
    svc = ManuscriptService(repo, "ms1")
    complete_manuscript(svc, human(repo, rid))
    report = svc.audit()
    assert report.passed
    assert not any("evidence changed" in p for p in completion_problems(repo, "ms1"))
    decide_item(
        ReportingService(repo, "ms1"),
        "prisma_2020_main",
        "26",
        rid,
        status="missing",
        section_ids=[],
        location="",
        justification="competing interests not declared",
        evidence_keys=[],
    )
    problems = completion_problems(repo, "ms1")
    assert any("evidence changed" in p and "reporting_decisions" in p for p in problems)
    # the manuscript text is unchanged, so the version itself is not stale
    assert svc.status()["evidence_current_for_latest_version"]


def test_compliance_claim_requires_complete_checklist(session: Session) -> None:
    repo, rid, _ = single_arm_project(session)
    svc = ManuscriptService(repo, "ms1")
    claim = "This review was reported in accordance with the PRISMA 2020 statement."
    latest = svc.create_template_draft(SYSTEM)
    methods = latest.section("methods")
    assert methods is not None
    svc.edit(
        human(repo, rid),
        {s.section_id: fill_author_input(s.source) for s in latest.sections}
        | {"methods": fill_author_input(methods.source) + " " + claim},
        "claim",
    )
    r = svc.audit()
    assert any(
        "checklist is incomplete" in f.message
        for f in r.findings
        if f.status == CheckStatus.UNSUPPORTED
    )
    confirm_checklists(repo, "ms1", rid)
    r = svc.audit()
    hit = [f for f in r.findings if f.scope == "reporting" and "adherence" in f.message]
    assert hit and all(f.status == CheckStatus.REQUIRES_HUMAN_REVIEW for f in hit)


def test_checklist_exports_keep_provenance(session: Session) -> None:
    repo, rid, _ = single_arm_project(session)
    complete_manuscript(ManuscriptService(repo, "ms1"), human(repo, rid))
    rs = ReportingService(repo, "ms1")
    csv_text = rs.checklist_csv("prisma_2020_main")
    assert csv_text.count("\n") == 43  # header + 42 items
    assert "synthetic test decision" in csv_text and REVIEWER in csv_text
    data = rs.checklist("prisma_2020_abstract")
    import docx
    from openpyxl import load_workbook

    doc = docx.Document(io.BytesIO(checklist_docx(data)))
    assert len(doc.tables[0].rows) == 13
    wb = load_workbook(io.BytesIO(checklist_xlsx(data)))
    assert wb["checklist"].max_row == 13
    assert any(r[0].value == "manuscript_sha256" for r in wb["provenance"].iter_rows())


def test_swim_checklist_refused_when_not_triggered(session: Session) -> None:
    repo, rid, _ = single_arm_project(session)
    complete_manuscript(ManuscriptService(repo, "ms1"), human(repo, rid))
    with pytest.raises(SysRevError, match="SWiM checklist is not required"):
        decide_item(
            ReportingService(repo, "ms1"),
            "swim",
            "3",
            rid,
            status="not_applicable",
            section_ids=[],
            location="",
            justification="no synthesis without meta-analysis",
            evidence_keys=[],
        )
    package = build_evidence_package(repo, "ms1")
    assert package.reporting is not None and not package.reporting.swim_required
