"""Derive the PRISMA 2020 counts from persisted records and decisions (never typed in).

Identification is counted per source from the stored records (each record belongs to one
execution or import); ClinicalTrials.gov records go to the *registers* box. Screening,
retrieval and eligibility use the same fail-closed status functions as the guards, so the
snapshot always agrees with the cross-check in :meth:`Repository.build_guard_context`.
"""

from __future__ import annotations

from collections import Counter

from src.core.enums import FullTextStatus, ReviewerRole, ScreeningStage, ScreeningVerdict
from src.core.schemas.pipeline import PRISMACounts
from src.core.schemas.screening import ScreeningDecision
from src.screening.agreement import screening_status
from src.screening.prisma import PRISMATracker
from src.search.databases import DATABASES
from src.state.repository import Repository


def _exclusion_reason(decisions: list[ScreeningDecision]) -> str:
    by_role = {d.reviewer: d for d in decisions}
    adjudication = by_role.get(ReviewerRole.ADJUDICATOR)
    if adjudication is not None and adjudication.is_human and adjudication.exclusion_reason:
        return adjudication.exclusion_reason
    reasons = sorted({d.exclusion_reason for d in decisions
                      if d.exclusion_reason and d.verdict == ScreeningVerdict.EXCLUDE})  # fmt: skip
    return " / ".join(reasons) or "reason not recorded"


def identification_by_source(repo: Repository, project_id: str) -> dict[str, dict[str, int]]:
    """Per source: records identified, duplicates removed, unique records contributed."""
    out: dict[str, dict[str, int]] = {}
    for r in repo.load_records(project_id):
        row = out.setdefault(r.source_database.value,
                             {"identified": 0, "duplicates_removed": 0, "unique": 0})  # fmt: skip
        row["identified"] += 1
        row["duplicates_removed" if r.duplicate_of else "unique"] += 1
    return dict(sorted(out.items()))


def derive_prisma(repo: Repository, project_id: str) -> PRISMACounts:
    records = repo.load_records(project_id)
    tracker = PRISMATracker()
    for db, n in sorted(Counter(r.source_database for r in records).items()):
        info = DATABASES.get(db)
        if info is not None and info.is_register:
            tracker.record_register_hits(n)
        else:
            tracker.record_identified(db, n)
    tracker.record_duplicates_removed(sum(1 for r in records if r.duplicate_of))
    unique = {r.record_id for r in records if r.duplicate_of is None}
    tracker.register_screening_pool(unique)

    decisions = repo.load_screening_decisions(project_id)
    conflicts = repo.load_conflicts(project_id)
    ta = screening_status(unique, decisions, conflicts, ScreeningStage.TITLE_ABSTRACT)
    for rid, verdict in sorted(ta.final.items()):
        tracker.record_title_abstract_decision(rid, include=verdict == ScreeningVerdict.INCLUDE)

    included = ta.ids_with(ScreeningVerdict.INCLUDE)
    items = {i.record_id: i for i in repo.load_fulltext_items(project_id)}
    retrieved: set[str] = set()
    for rid in sorted(included & set(items)):
        if items[rid].is_retrieved:
            tracker.record_retrieval(rid, retrieved=True)
            retrieved.add(rid)
        elif items[rid].status == FullTextStatus.UNOBTAINABLE:
            tracker.record_retrieval(rid, retrieved=False)

    ft = screening_status(retrieved, decisions, conflicts, ScreeningStage.FULL_TEXT)
    ft_decisions: dict[str, list[ScreeningDecision]] = {}
    for d in decisions:
        if d.stage == ScreeningStage.FULL_TEXT:
            ft_decisions.setdefault(d.record_id, []).append(d)
    for rid, verdict in sorted(ft.final.items()):
        include = verdict == ScreeningVerdict.INCLUDE
        reason = None if include else _exclusion_reason(ft_decisions.get(rid, []))
        tracker.record_fulltext_decision(rid, include=include, reason=reason)
    ft_included = ft.ids_with(ScreeningVerdict.INCLUDE)
    studies = repo.study_ids(project_id, set(ft_included))
    tracker.set_studies_included(len({studies.get(rid, rid) for rid in ft_included}))
    return tracker.snapshot()
