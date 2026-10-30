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
)
from src.core.schemas import FullTextItem, ScreeningDecision, StudyRecord
from src.core.schemas.search import SearchRun
from src.core.schemas.stats import AnalysisSpec
from src.core.schemas.study import SearchProvenance
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


def single_arm_project(session: Session, pid: str = "ms1") -> tuple[Repository, str, str]:
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
    repo.save_records(pid, records)
    include_everywhere(repo, pid, records, rid, REVIEWER)
    for r, x, n in NECROSIS:
        repo.save_extracted_arm(
            pid, stats_arm(r, "necrosis", follow_up=LAST_FU, events=x, n_analyzed=n)
        )
    for r, est, lo, hi in KM:
        repo.save_extracted_arm(pid, km_arm(r, est, lo, hi))
    for rec in records:
        repo.save_rob(pid, approved_rob(rec.record_id, rid, name=REVIEWER))
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


def comparative_project(session: Session, pid: str = "mc1") -> tuple[Repository, str, str]:
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


def complete_manuscript(svc: ManuscriptService, actor: Actor, **extra: str) -> Any:
    """A template draft with every author-input marker filled (no other change)."""
    latest = svc.repo.latest_manuscript_version(svc.project_id)
    if latest is None:
        latest = svc.create_template_draft(SYSTEM)
    changes = {s.section_id: fill_author_input(s.source) for s in latest.sections}
    changes.update(extra)
    return svc.edit(actor, changes, "author input added")
