"""PRISMA-S search summary built only from the search log (Rethlefsen et al. 2021).

A database is reported as searched only if an execution or import is recorded for it; a
selected database without one is reported as *not searched*, with the reviewer's reason
when one was recorded. Previews are listed as search development, never as searches.
"""

from __future__ import annotations

import csv
import io
from typing import Any

from src.core.enums import SearchRunKind
from src.search.databases import DATABASES
from src.search.identification import import_totals, strategy_status
from src.search.prisma_sync import identification_by_source
from src.state.repository import Repository

CSV_COLUMNS = (
    "database", "platform", "kind", "status", "query_as_run", "parameters", "searched_on",
    "executed_at", "hits_reported", "records_received", "records_saved", "files", "actor",
    "note", "problems",
)  # fmt: skip


def _label(db: str | None) -> str:
    if db is None:
        return ""
    for d, i in DATABASES.items():
        if d.value == db:
            return i.label
    return db


def search_report(repo: Repository, project_id: str) -> dict[str, Any]:
    runs = repo.load_search_runs(project_id)
    protocol = repo.latest_protocol(project_id)
    strategies = []
    for s in protocol.search_strategies if protocol else []:
        totals = import_totals(runs, s.sha256)
        strategies.append({
            "database": _label(s.database), "strategy_sha256": s.sha256,
            "platform": s.platform, "query": s.query,
            "date_from": s.date_from.isoformat() if s.date_from else None,
            "date_to": s.date_to.isoformat() if s.date_to else None,
            "filters": s.filters, "syntax_validated": s.syntax_validated,
            "validation_method": s.validation_method, "status": strategy_status(runs, s),
            "imported_records": totals.records, "interface_reported": totals.reported,
        })  # fmt: skip
    searched = [r for r in runs if r.identifies_records]
    rows = []
    for r in runs:
        if r.kind in (SearchRunKind.PREVIEW, SearchRunKind.DEDUPLICATION):
            continue
        rows.append({
            "database": _label(r.database.value if r.database else None),
            "platform": r.platform or "", "kind": r.kind.value, "status": r.status.value,
            "query_as_run": r.query or "",
            "parameters": "; ".join(f"{k}={v}" for k, v in sorted(r.parameters.items())),
            "searched_on": (r.searched_on or r.executed_at.date()).isoformat(),
            "executed_at": r.executed_at.isoformat(),
            "hits_reported": r.hits_reported, "records_received": r.records_received,
            "records_saved": r.records_saved,
            "files": "; ".join(f"{f.original_name or f.filename} sha256:{f.sha256}"
                               for f in r.files if f.role == "import_file")
                     or (f"{len(r.files)} raw API response(s) preserved" if r.files else ""),
            "actor": r.actor, "note": r.note or "", "problems": "; ".join(r.problems),
        })  # fmt: skip
    dedup = [r for r in runs if r.kind == SearchRunKind.DEDUPLICATION]
    return {
        "strategies": strategies,
        "databases_searched": sorted({_label(r.database.value) for r in searched if r.database}),
        "databases_not_searched": [
            {"database": _label(r.database.value if r.database else None), "reason": r.note}
            for r in runs if r.kind == SearchRunKind.NOT_SEARCHED
        ],
        "log": rows,
        "previews": sum(1 for r in runs if r.kind == SearchRunKind.PREVIEW),
        "identification_by_source": identification_by_source(repo, project_id),
        "deduplication": dedup[-1].summary if dedup else None,
        "problems": repo.search_identification_problems(project_id),
    }  # fmt: skip


def report_csv(report: dict[str, Any]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for row in report["log"]:
        writer.writerow(row)
    return buf.getvalue()


def report_markdown(report: dict[str, Any]) -> str:
    out = ["# Search summary (PRISMA-S)", ""]
    out.append("Databases searched: " + (", ".join(report["databases_searched"]) or "none"))
    for item in report["databases_not_searched"]:
        out.append(f"Not searched: {item['database']} — {item['reason']}")
    out += ["", "## Searches as run", ""]
    for row in report["log"]:
        if row["kind"] not in ("execution", "import"):
            continue
        out += [
            f"### {row['database']} ({row['platform'] or 'platform not recorded'}) — "
            f"{row['kind']}, {row['status']}",
            f"* Date of search: {row['searched_on']}",
            f"* Records reported by the database: {row['hits_reported']}; "
            f"records retrieved/saved: {row['records_saved']}",
            f"* Limits/parameters: {row['parameters'] or 'none'}",
            f"* Files: {row['files'] or 'none'}",
            f"* Run by: {row['actor']}",
            "", "```", row["query_as_run"], "```", "",
        ]  # fmt: skip
        if row["problems"]:
            out += [f"* Problems: {row['problems']}", ""]
    out += ["## Identification by source", "", "| Source | Identified | Duplicates removed | "
            "Unique |", "|---|---|---|---|"]  # fmt: skip
    for db, c in report["identification_by_source"].items():
        out.append(f"| {_label(db)} | {c['identified']} | {c['duplicates_removed']} | "
                   f"{c['unique']} |")  # fmt: skip
    if report["problems"]:
        out += ["", "## Open problems", "", *(f"* {p}" for p in report["problems"])]
    return "\n".join(out) + "\n"
