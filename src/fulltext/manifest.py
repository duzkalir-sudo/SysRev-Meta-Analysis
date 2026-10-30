"""Missing-full-text manifest (Gate 2) and the report-level PRISMA retrieval flow."""

from __future__ import annotations

import csv
import io
import json
from collections import Counter
from dataclasses import asdict, dataclass

from src.core.enums import ReviewerRole, ScreeningStage, ScreeningVerdict
from src.screening.agreement import screening_status
from src.state.repository import Repository


@dataclass(frozen=True, slots=True)
class ManifestRow:
    record_id: str
    title: str
    authors: str
    year: int | None
    journal: str | None
    doi: str | None
    pmid: str | None
    pmcid: str | None
    status: str
    usable_pdf: bool
    match_status: str | None
    last_attempt: str | None
    notes: str


def build_manifest(
    repo: Repository, project_id: str, *, only_open: bool = True
) -> list[ManifestRow]:
    """One row per title/abstract-included report. ``only_open`` keeps rows that still
    block Gate 2 (not retrieved, or a PDF whose identity is unconfirmed)."""
    items = {i.record_id: i for i in repo.load_fulltext_items(project_id)}
    rows: list[ManifestRow] = []
    for rid in sorted(repo.included_record_ids(project_id, ScreeningStage.TITLE_ABSTRACT)):
        record = repo.get_record(project_id, rid)
        item = items.get(rid)
        if only_open and item is not None and not item.blocks_gate2:
            continue
        last = item.attempts[-1] if item and item.attempts else None
        authors = "; ".join(
            " ".join(x for x in (a.last_name, a.initials or a.fore_name) if x)
            or a.collective_name or a.full_name or ""
            for a in record.authors
        )  # fmt: skip
        rows.append(
            ManifestRow(
                record_id=rid,
                title=record.title,
                authors=authors,
                year=record.publication_year,
                journal=record.journal,
                doi=record.doi,
                pmid=record.pmid,
                pmcid=record.pmcid or (item.pmcid if item else None),
                status=item.status.value if item else "not_attempted",
                usable_pdf=bool(item and item.has_usable_pdf),
                match_status=item.match_status.value if item and item.match_status else None,
                last_attempt=f"{last.service}: {last.outcome}" if last else None,
                notes=" | ".join(item.match_notes) if item else "",
            )
        )
    return rows


def manifest_csv(rows: list[ManifestRow]) -> str:
    out = io.StringIO()
    fields = list(ManifestRow.__dataclass_fields__)
    writer = csv.DictWriter(out, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(asdict(row))
    return out.getvalue()


def manifest_json(rows: list[ManifestRow]) -> str:
    return json.dumps([asdict(r) for r in rows], indent=2, ensure_ascii=False)


@dataclass(frozen=True, slots=True)
class ReportFlow:
    """PRISMA 2020 report-level boxes derived from persisted facts (no typed-in numbers).

    ``not_retrieved`` (sought but unobtainable) is kept separate from reports that were
    retrieved and then excluded at full-text eligibility (``excluded_by_reason``).
    Retrieved reports are split into those with a usable (identity-confirmed) PDF and
    those available only as structured XML, which cannot yet support page-grounded
    numeric extraction.
    """

    sought: int
    not_retrieved: tuple[str, ...]
    retrieved_with_pdf: tuple[str, ...]
    retrieved_structured_text_only: tuple[str, ...]
    awaiting_retrieval: tuple[str, ...]
    assessed: int
    excluded_by_reason: dict[str, int]
    awaiting_assessment: tuple[str, ...]
    reports_included: int
    studies_included: int


def report_flow(repo: Repository, project_id: str) -> ReportFlow:
    sought = repo.included_record_ids(project_id, ScreeningStage.TITLE_ABSTRACT)
    items = {i.record_id: i for i in repo.load_fulltext_items(project_id)}
    retrieved = {rid for rid in sought if rid in items and items[rid].is_retrieved}
    unobtainable = {
        rid for rid in sought if rid in items and items[rid].status.value == "unobtainable"
    }
    decisions = repo.load_screening_decisions(project_id)
    ft = screening_status(
        retrieved, decisions, repo.load_conflicts(project_id), ScreeningStage.FULL_TEXT
    )
    reasons: Counter[str] = Counter()
    by_role = {
        (d.record_id, d.reviewer): d for d in decisions if d.stage == ScreeningStage.FULL_TEXT
    }
    for rid in sorted(ft.ids_with(ScreeningVerdict.EXCLUDE)):
        source = by_role.get((rid, ReviewerRole.ADJUDICATOR)) or by_role.get(
            (rid, ReviewerRole.REVIEWER_1)
        )
        reasons[(source.exclusion_reason if source else None) or "reason not recorded"] += 1
    included = ft.ids_with(ScreeningVerdict.INCLUDE)
    studies = set(repo.study_ids(project_id, included).values())
    return ReportFlow(
        sought=len(sought),
        retrieved_with_pdf=tuple(sorted(r for r in retrieved if items[r].has_usable_pdf)),
        retrieved_structured_text_only=tuple(
            sorted(r for r in retrieved if items[r].status.value == "oa_structured_text")
        ),
        not_retrieved=tuple(sorted(unobtainable)),
        awaiting_retrieval=tuple(sorted(sought - retrieved - unobtainable)),
        assessed=len(retrieved),
        excluded_by_reason=dict(reasons),
        awaiting_assessment=tuple(sorted(set(ft.missing) | set(ft.unresolved))),
        reports_included=len(included),
        studies_included=len(studies),
    )
