"""Builders for Stage 6 tests. All studies, numbers and bibliographic data are SYNTHETIC.

Two projects are built through the real persistence layer and the real statistics engine,
so every number in the manuscript has a known, deterministic source:

* a **single-arm** review (crude proportion of radiation necrosis, 6 case series, plus a
  Kaplan-Meier local-control analysis from 3 series and a crude-control outcome with no
  data), and
* a **comparative** review (risk ratio of a complication, 3 trials).

DOIs use the 10.5555 test prefix and PMIDs are invented (9xxxxxxx).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Any

from sqlalchemy.orm import Session

from src.core.enums import (
    ActorType,
    ArmRole,
    FullTextStatus,
    Phase,
    ReviewDesign,
    ReviewerRole,
    ScreeningStage,
    ScreeningVerdict,
    SearchRunKind,
    SourceDatabase,
    StudyDesign,
    UploadMatchStatus,
)
from src.core.schemas import FullTextItem, ScreeningDecision, StudyRecord
from src.core.schemas.search import SearchRun
from src.core.schemas.stats import AnalysisSpec
from src.core.schemas.study import SearchProvenance
from src.extraction import link_reports_to_study
from src.manuscript.service import Actor, ManuscriptService
from src.search.prisma_sync import derive_prisma
from src.state.repository import Repository
from src.stats_engine.engine import run_plan
from tests.conftest import NOW, approved_by, approved_rob, make_author, make_record
from tests.stats_helpers import LAST_FU, OUTCOMES, YEARS_5, km_arm, plan, stats_arm

REVIEWER = "Dr Stats Reviewer"
QUERY = '("radiosurgery"[MeSH Terms]) AND (necrosis)'
NECROSIS = [
    ("s1", "0", "25"),
    ("s2", "3", "40"),
    ("s3", "1", "18"),
    ("s4", "0", "12"),
    ("s5", "5", "60"),
    ("s6", "2", "33"),
]
KM = [
    ("s1", "91.2", "80.1", "96.3"),
    ("s2", "88.0", "75.5", "94.4"),
    ("s3", "93.5", "70.2", "98.5"),
]
SURNAMES = ["Alder", "Birch", "Cedar", "Dogwood", "Elm", "Fir"]


def pubmed_record(project_rid: str, i: int, run_id: str | None = None) -> StudyRecord:
    pmid = str(90000001 + i)
    return make_record(
        project_rid,
        f"Synthetic case series {i + 1} of radiosurgery for brain metastases",
        pmid=pmid,
        doi=f"10.5555/synthetic.{i + 1}",
        journal="Journal of Synthetic Oncology",
        publication_year=2015 + i,
        authors=[make_author(SURNAMES[i], "Ann"), make_author("Quince", "Bo")],
        provenance=SearchProvenance(
            database=SourceDatabase.PUBMED,
            query=QUERY,
            endpoint="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            executed_at=NOW,
            total_hits_reported=6,
            source_record_id=pmid,
            search_run_id=run_id,
        ),
    )


def include_everywhere(
    repo: Repository, pid: str, records: list[StudyRecord], reviewer_id: str, name: str
) -> None:
    for rec in records:
        for stage in (ScreeningStage.TITLE_ABSTRACT, ScreeningStage.FULL_TEXT):
            for role in (ReviewerRole.REVIEWER_1, ReviewerRole.REVIEWER_2):
                repo.save_screening_decision(
                    pid,
                    ScreeningDecision(
                        record_id=rec.record_id,
                        reviewer=role,
                        reviewer_identity=f"agent-{role}",
                        stage=stage,
                        verdict=ScreeningVerdict.INCLUDE,
                        rationale="synthetic",
                    ),
                )
        repo.save_screening_decision(
            pid,
            ScreeningDecision(
                record_id=rec.record_id,
                reviewer=ReviewerRole.ADJUDICATOR,
                is_human=True,
                reviewer_identity=name,
                reviewer_id=reviewer_id,
                stage=ScreeningStage.FULL_TEXT,
                verdict=ScreeningVerdict.INCLUDE,
                rationale="confirmed",
            ),
        )
        repo.save_fulltext_item(
            pid,
            FullTextItem(
                record_id=rec.record_id,
                title=rec.title,
                status=FullTextStatus.OA_DOWNLOADED,
                pdf_filename=f"{rec.record_id}.pdf",
                pdf_sha256="b" * 64,
                match_status=UploadMatchStatus.CONFIRMED,
            ),
        )


def _search_run(repo: Repository, pid: str, reviewer_id: str, name: str) -> str:
    protocol = repo.latest_protocol(pid)
    assert protocol is not None
    strategy = protocol.search_strategies[0]
    run = SearchRun(
        kind=SearchRunKind.EXECUTION,
        database=SourceDatabase.PUBMED,
        strategy_sha256=strategy.sha256,
        protocol_version=protocol.version,
        query=strategy.query,
        endpoint="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        executed_at=NOW,
        searched_on=date(2026, 1, 15),
        hits_reported=6,
        records_received=6,
        records_saved=6,
        restrictions_applied="none_defined",
        actor=name,
        actor_type=ActorType.HUMAN,
        reviewer_id=reviewer_id,
    )
    repo.add_search_run(pid, run)
    return run.run_id


def single_arm_project(
    session: Session,
    pid: str = "ms1",
    *,
    extra_arms: list[Any] | None = None,
    record_overrides: dict[str, dict[str, Any]] | None = None,
    study_links: list[list[str]] | None = None,
    before_stats: Callable[[Repository, str], None] | None = None,
) -> tuple[Repository, str, str]:
    """A single-arm review in Stage 6 with a verified statistics artifact.

    Returns (repository, reviewer_id, reviewer token)."""
    repo = Repository(session)
    repo.create_project(pid, "Synthetic SRS review")
    rid, token = repo.enroll_reviewer(REVIEWER)
    protocol = approved_by(
        rid,
        name=REVIEWER,
        review_design=ReviewDesign.SINGLE_ARM,
        comparator=None,
        review_title="Radiation necrosis after radiosurgery for brain metastases (synthetic)",
        research_question="What is the incidence of radiation necrosis after radiosurgery?",
        population="Adults with brain metastases",
        intervention="Stereotactic radiosurgery",
        study_designs=[StudyDesign.CASE_SERIES],
        outcomes=[OUTCOMES["necrosis"], OUTCOMES["control_km"], OUTCOMES["control_crude"]],
    )
    repo.save_protocol(pid, protocol)
    run_id = _search_run(repo, pid, rid, REVIEWER)
    records = [pubmed_record(r, i, run_id) for i, (r, _, _) in enumerate(NECROSIS)]
    records = [
        rec.model_copy(update=(record_overrides or {}).get(rec.record_id, {})) for rec in records
    ]
    repo.save_records(pid, records)
    include_everywhere(repo, pid, records, rid, REVIEWER)
    for arm in extra_arms or []:
        repo.save_extracted_arm(pid, arm)
    for r, x, n in NECROSIS:
        repo.save_extracted_arm(
            pid, stats_arm(r, "necrosis", follow_up=LAST_FU, events=x, n_analyzed=n)
        )
    for r, est, lo, hi in KM:
        repo.save_extracted_arm(pid, km_arm(r, est, lo, hi))
    for rec in records:
        repo.save_rob(pid, approved_rob(rec.record_id, rid, name=REVIEWER))
    for group in study_links or []:  # documented before Stage 5 (the links are an input)
        link_reports_to_study(
            repo,
            pid,
            group,
            study_id=f"study-{group[0]}",
            primary_record_id=group[0],
            reviewer=repo.require_active_reviewer(rid),
            note="synthetic: reports of one study",
        )
    if before_stats is not None:  # further human decisions that are Stage 5 inputs
        before_stats(repo, rid)
    repo.save_prisma(pid, derive_prisma(repo, pid))
    _to_stage6(
        repo,
        pid,
        rid,
        [
            AnalysisSpec(
                analysis_id="nec",
                outcome_id="necrosis",
                effect_measure="PROPORTION",
                timepoints=[LAST_FU],
            ),
            AnalysisSpec(
                analysis_id="km5",
                outcome_id="control_km",
                effect_measure="KM_PROBABILITY",
                timepoints=[YEARS_5],
                reported_ci_scale="log_log",
            ),
            AnalysisSpec(
                analysis_id="ctl",
                outcome_id="control_crude",
                effect_measure="PROPORTION",
                timepoints=[LAST_FU],
            ),
        ],
    )
    return repo, rid, token


def comparative_project(
    session: Session, pid: str = "mc1", *, study_links: list[list[str]] | None = None
) -> tuple[Repository, str, str]:
    repo = Repository(session)
    repo.create_project(pid, "Synthetic comparative review")
    rid, token = repo.enroll_reviewer(REVIEWER)
    repo.save_protocol(
        pid,
        approved_by(
            rid,
            name=REVIEWER,
            outcomes=[OUTCOMES["complication"]],
            review_title="Intervention X versus placebo and complications (synthetic)",
        ),
    )
    run_id = _search_run(repo, pid, rid, REVIEWER)
    trials = [
        ("t1", "12", "100", "20", "100"),
        ("t2", "8", "80", "15", "82"),
        ("t3", "5", "60", "9", "58"),
    ]
    records = [pubmed_record(t[0], i, run_id) for i, t in enumerate(trials)]
    repo.save_records(pid, records)
    include_everywhere(repo, pid, records, rid, REVIEWER)
    for r, e1, n1, e0, n0 in trials:
        repo.save_extracted_arm(
            pid,
            stats_arm(
                r, "complication", role=ArmRole.INTERVENTION, label="X", events=e1, n_analyzed=n1
            ),
        )
        repo.save_extracted_arm(
            pid,
            stats_arm(
                r, "complication", role=ArmRole.CONTROL, label="Placebo", events=e0, n_analyzed=n0
            ),
        )
    for rec in records:
        repo.save_rob(pid, approved_rob(rec.record_id, rid, name=REVIEWER))
    for group in study_links or []:  # documented before Stage 5 (the links are an input)
        link_reports_to_study(
            repo,
            pid,
            group,
            study_id=f"study-{group[0]}",
            primary_record_id=group[0],
            reviewer=repo.require_active_reviewer(rid),
            note="synthetic: reports of one study",
        )
    repo.save_prisma(pid, derive_prisma(repo, pid))
    _to_stage6(
        repo,
        pid,
        rid,
        [AnalysisSpec(analysis_id="comp", outcome_id="complication", effect_measure="RR")],
    )
    return repo, rid, token


def _to_stage6(repo: Repository, pid: str, rid: str, specs: list[AnalysisSpec]) -> None:
    project = repo.get_project(pid)
    project.phase = Phase.STAGE5_META_ANALYSIS.value
    approved = plan(*specs, reviewer_id=rid)
    repo.save_analysis_plan(pid, approved)
    output = run_plan(approved, repo.load_stage5_inputs(pid), project_id=pid, generated_at=NOW)
    repo.record_stats_output(pid, output)
    project.phase = Phase.STAGE6_MANUSCRIPT.value
    repo.session.flush()


def human(repo: Repository, rid: str) -> Actor:
    return Actor(repo.require_active_reviewer(rid).display_name, ActorType.HUMAN, rid)


SYSTEM = Actor("system", ActorType.SYSTEM)

# Author text that replaces every [[AUTHOR INPUT]] marker of the template (synthetic prose,
# no numbers, no causal or recommendation language).
AUTHOR_TEXT: dict[str, str] = {}


def fill_author_input(source: str, replacement: str = "Synthetic author-written text.") -> str:
    import re

    return re.sub(r"\[\[AUTHOR INPUT:[^\]]*\]\]", replacement, source)


def confirm_checklists(
    repo: Repository,
    pid: str,
    rid: str,
    checklists: tuple[str, ...] = (
        "prisma_2020_main",
        "prisma_2020_abstract",
    ),
) -> None:
    """SYNTHETIC reviewer decisions on every checklist item of the latest version: an item with
    a candidate location is recorded as reported there; any other as not applicable. (A real
    reviewer reads each item; tests need a complete, current record.)"""
    from src.reporting.service import ReportingService

    rs = ReportingService(repo, pid)
    name = repo.require_active_reviewer(rid).display_name
    for checklist in checklists:
        screen = rs.checklist(checklist)  # decisions are made on this displayed state
        rows = []
        for row in screen["items"]:
            if row["decided"]:
                continue
            if row["candidates"]:
                section = row["candidates"][0].split(":", 1)[0]
                status, sections, loc = "reported", [section], row["candidates"][0]
            else:
                status, sections, loc = "not_applicable", [], ""
            rows.append(
                {
                    "checklist": checklist,
                    "item_id": row["item_id"],
                    "status": status,
                    "section_ids": sections,
                    "location": loc,
                    "justification": "synthetic test decision",
                    "based_on_decision_id": (row["decision"] or {}).get("decision_id"),
                }
            )
        if rows:
            rs.decide_items(rows, seen=seen(screen), reviewer_id=rid, reviewer_name=name)


def seen(screen: dict[str, Any]) -> dict[str, Any]:
    """The state identifiers a checklist screen displayed (sent back with a decision)."""
    return {k: screen[k] for k in ("manuscript_version", "manuscript_sha256", "evidence_sha256")}


def decide_item(rs: Any, checklist: str, item_id: str, rid: str, **kw: Any) -> Any:
    """Read the checklist screen, then decide one item on that displayed state."""
    screen = rs.checklist(checklist)
    row = next(r for r in screen["items"] if r["item_id"] == item_id)
    kw.setdefault("evidence_keys", [])
    kw.setdefault("section_ids", [])
    kw.setdefault("location", "")
    return rs.decide_item(
        checklist,
        item_id,
        **kw,
        seen=seen(screen),
        based_on_decision_id=(row["decision"] or {}).get("decision_id"),
        reviewer_id=rid,
        reviewer_name=rs.repo.require_active_reviewer(rid).display_name,
    )


def save_grade(rs: Any, analysis_id: str, body: dict[str, Any], rid: str) -> Any:
    """Read the GRADE screen, then save on the displayed inputs and record."""
    row = next(r for r in rs.grade() if r["analysis_id"] == analysis_id)
    shown = {
        "inputs_sha256": row["inputs_sha256"],
        "based_on_assessment_id": row["based_on_assessment_id"],
    }
    return rs.save_grade(
        analysis_id,
        shown | body,
        reviewer_id=rid,
        reviewer_name=rs.repo.require_active_reviewer(rid).display_name,
    )


def save_group(rs: Any, body: dict[str, Any], rid: str) -> Any:
    """Preview the group (the evidence a reviewer sees), then save on that state."""
    preview = rs.preview_group(body)
    shown = {k: preview[k] for k in ("inputs_sha256", "based_on_revision_id")}
    return rs.save_group(
        shown | body,
        reviewer_id=rid,
        reviewer_name=rs.repo.require_active_reviewer(rid).display_name,
    )


def decide_swim(rs: Any, applicability: str, rationale: str, rid: str) -> Any:
    screen = rs.swim()
    return rs.decide_swim(
        applicability,
        rationale,
        synthesis_sha256=screen["synthesis_sha256"],
        based_on_decision_id=screen["based_on_decision_id"],
        reviewer_id=rid,
        reviewer_name=rs.repo.require_active_reviewer(rid).display_name,
    )


def complete_manuscript(svc: ManuscriptService, actor: Actor, **extra: str) -> Any:
    """A template draft with every author-input marker filled (no other change), with every
    PRISMA 2020 checklist item decided (synthetically) for that version."""
    latest = svc.repo.latest_manuscript_version(svc.project_id)
    if latest is None:
        latest = svc.create_template_draft(SYSTEM)
    changes = {s.section_id: fill_author_input(s.source) for s in latest.sections}
    changes.update(extra)
    version = svc.edit(actor, changes, "author input added")
    assert actor.reviewer_id is not None
    confirm_checklists(svc.repo, svc.project_id, actor.reviewer_id)
    return version
