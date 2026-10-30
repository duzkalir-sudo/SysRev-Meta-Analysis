"""PRISMA-S search summary built only from the search log (Rethlefsen et al. 2021).

Every status is derived by :mod:`src.search.identification` from the chronological log, so
the strategy table, the database lists, the Markdown and the CSV export always agree with
the screening-readiness checks:

* a database is **searched** if at least one of its protocol strategies has a successful
  execution or import (a recorded zero-result search counts; a failed attempt does not);
* it is **not searched** if none of its strategies was searched and a reviewer declared at
  least one of them not searched (the reason in effect is reported);
* otherwise it is **not yet searched**.

Historical declarations stay in the log with their current effect (e.g. "superseded by a
later successful search/import"). Protocol-defined restrictions are reported separately
from whether they were applied (by the system, confirmed by a reviewer, or not confirmed).
Previews are search development, never searches.
"""

from __future__ import annotations

import csv
import io
from typing import Any

from src.core.enums import SearchRunKind
from src.core.schemas.search import SearchRun
from src.search.databases import DATABASES
from src.search.identification import run_effects, strategy_state
from src.search.prisma_sync import identification_by_source
from src.state.repository import Repository

CSV_COLUMNS = (
    "database",
    "platform",
    "kind",
    "status",
    "current_effect",
    "strategy_status",
    "strategy_sha256",
    "protocol_version",
    "query_as_run",
    "parameters_sent",
    "protocol_date_limits",
    "protocol_languages",
    "protocol_filters",
    "restrictions_applied",
    "restrictions_note",
    "searched_on",
    "executed_at",
    "hits_reported",
    "records_received",
    "records_saved",
    "files",
    "actor",
    "note",
    "warnings",
    "problems",
)

APPLIED_TEXT = {
    "applied_by_system": "applied by the system in the API request",
    "confirmed_by_reviewer": "confirmed by the reviewer as applied in the vendor interface",
    "not_confirmed": "NOT confirmed as applied in the vendor interface",
    "none_defined": "none defined in the protocol",
    None: "not recorded (entry created before Session 5 corrections)",
}


def _label(db: str | None) -> str:
    if db is None:
        return ""
    for d, i in DATABASES.items():
        if d.value == db:
            return i.label
    return db


def _date_limits(r: dict[str, Any]) -> str:
    if not (r.get("date_from") or r.get("date_to")):
        return "none"
    return f"{r.get('date_from') or 'open'} to {r.get('date_to') or 'open'}"


def _log_row(r: SearchRun, effect: str, strategy_status: str) -> dict[str, Any]:
    pr = r.protocol_restrictions
    return {
        "database": _label(r.database.value if r.database else None),
        "platform": r.platform or "",
        "kind": r.kind.value,
        "status": r.status.value,
        "current_effect": effect,
        "strategy_status": strategy_status,
        "strategy_sha256": r.strategy_sha256 or "",
        "protocol_version": r.protocol_version,
        "query_as_run": r.query or "",
        "parameters_sent": "; ".join(f"{k}={v}" for k, v in sorted(r.parameters.items())),
        "protocol_date_limits": _date_limits(pr) if pr else "",
        "protocol_languages": "; ".join(pr.get("languages", [])) if pr else "",
        "protocol_filters": " | ".join(pr.get("filters", [])) if pr else "",
        "restrictions_applied": (
            APPLIED_TEXT[r.restrictions_applied]
            if r.kind in (SearchRunKind.EXECUTION, SearchRunKind.IMPORT, SearchRunKind.PREVIEW)
            else ""
        ),
        "restrictions_note": r.restrictions_note or "",
        "searched_on": (r.searched_on or r.executed_at.date()).isoformat(),
        "executed_at": r.executed_at.isoformat(),
        "hits_reported": r.hits_reported,
        "records_received": r.records_received,
        "records_saved": r.records_saved,
        "files": "; ".join(
            f"{f.original_name or f.filename} sha256:{f.sha256}"
            for f in r.files
            if f.role == "import_file"
        )
        or (f"{len(r.files)} raw API response(s) preserved" if r.files else ""),
        "actor": r.actor,
        "note": r.note or "",
        "warnings": "; ".join(r.warnings[:20]),
        "problems": "; ".join(r.problems),
    }


def search_report(repo: Repository, project_id: str) -> dict[str, Any]:
    runs = repo.load_search_runs(project_id)
    protocol = repo.latest_protocol(project_id)
    effects = run_effects(runs)
    strategies: list[dict[str, Any]] = []
    per_db: dict[str, list[dict[str, Any]]] = {}
    for s in protocol.search_strategies if protocol else []:
        st = strategy_state(runs, s.sha256)
        t = st.imports
        row: dict[str, Any] = {
            "database": _label(s.database),
            "database_id": s.database.value,
            "strategy_sha256": s.sha256,
            "protocol_version": protocol.version if protocol else None,
            "platform": s.platform,
            "query": s.query,
            "date_from": s.date_from.isoformat() if s.date_from else None,
            "date_to": s.date_to.isoformat() if s.date_to else None,
            "languages": s.languages,
            "filters": s.filters,
            "syntax_validated": s.syntax_validated,
            "validation_method": s.validation_method,
            "current_status": st.status,
            "status": st.detail,
            "not_searched_reason": st.effective_declaration.note
            if st.effective_declaration
            else None,
            "failed_attempts": st.failed_attempts,
            "imported_records": t.records,
            "import_files": t.files,
            "interface_reported": t.reported,
            "declared_totals": list(t.declared),
            "reconciliation": t.state,
            "reconciliation_acknowledged": st.import_acknowledgement is not None,
            "acknowledgement_note": st.import_acknowledgement.note
            if st.import_acknowledgement
            else None,
        }
        strategies.append(row)
        per_db.setdefault(_label(s.database), []).append(row)
    searched, not_searched, pending = [], [], []
    for db, rows in sorted(per_db.items()):
        statuses = {r["current_status"] for r in rows}
        if "searched" in statuses:
            searched.append(db)
        elif "not_searched" in statuses:
            not_searched.append(
                {
                    "database": db,
                    "reasons": [
                        r["not_searched_reason"]
                        for r in rows
                        if r["current_status"] == "not_searched"
                    ],
                }
            )
        else:
            pending.append(db)
    status_by_sha = {r["strategy_sha256"]: r["current_status"] for r in strategies}
    rows = [
        _log_row(r, effects.get(r.run_id, ""), status_by_sha.get(r.strategy_sha256 or "", ""))
        for r in runs
        if r.kind not in (SearchRunKind.PREVIEW, SearchRunKind.DEDUPLICATION)
    ]
    dedup = [r for r in runs if r.kind == SearchRunKind.DEDUPLICATION]
    return {
        "strategies": strategies,
        "databases_searched": searched,
        "databases_not_searched": not_searched,
        "databases_not_yet_searched": pending,
        "log": rows,
        "previews": sum(1 for r in runs if r.kind == SearchRunKind.PREVIEW),
        "identification_by_source": identification_by_source(repo, project_id),
        "deduplication": dedup[-1].summary if dedup else None,
        "problems": repo.search_identification_problems(project_id),
    }


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
        out.append(f"Not searched: {item['database']} — {'; '.join(item['reasons'])}")
    if report["databases_not_yet_searched"]:
        out.append("Not yet searched: " + ", ".join(report["databases_not_yet_searched"]))
    out += ["", "## Strategies (current status)", ""]
    for s in report["strategies"]:
        reconciliation = s["reconciliation"]
        if s["reconciliation_acknowledged"]:
            reconciliation += f" (acknowledged: {s['acknowledgement_note']})"
        out += [
            f"### {s['database']} — strategy {s['strategy_sha256'][:12]} "
            f"(protocol version {s['protocol_version']})",
            f"* Platform: {s['platform'] or 'not recorded'}",
            f"* Current status: {s['status']}",
            f"* Protocol date limits: {_date_limits(s)}",
            f"* Protocol language limits: {'; '.join(s['languages']) or 'none'}",
            f"* Protocol filters (separate from the query): {' | '.join(s['filters']) or 'none'}",
            f"* Vendor-reported total vs imported: {s['interface_reported']} vs "
            f"{s['imported_records']} ({reconciliation})"
            if s["import_files"] or s["reconciliation"] != "no_imports"
            else "* Imports: none",
            "",
            "```",
            s["query"],
            "```",
            "",
        ]
    out += ["## Search log as run (append-only)", ""]
    for row in report["log"]:
        if row["kind"] not in ("execution", "import", "not_searched"):
            continue
        out += [
            f"### {row['database']} ({row['platform'] or 'platform not recorded'}) — "
            f"{row['kind']}, {row['status']} — {row['current_effect']}",
            f"* Date of search: {row['searched_on']}; recorded by {row['actor']}",
        ]
        if row["kind"] == "not_searched":
            out += [f"* Reason: {row['note']}", ""]
            continue
        out += [
            f"* Records reported by the database: "
            f"{row['hits_reported'] if row['hits_reported'] is not None else 'not recorded'}; "
            f"records saved: {row['records_saved']}",
            f"* Parameters sent by the system: {row['parameters_sent'] or 'none'}",
            f"* Protocol restrictions: dates {row['protocol_date_limits'] or 'n/a'}; "
            f"languages {row['protocol_languages'] or 'none'}; "
            f"filters {row['protocol_filters'] or 'none'}",
            f"* Restrictions applied: {row['restrictions_applied']}"
            + (f" — {row['restrictions_note']}" if row["restrictions_note"] else ""),
            f"* Files: {row['files'] or 'none'}",
            "",
            "```",
            row["query_as_run"],
            "```",
            "",
        ]
        for label in ("warnings", "problems"):
            if row[label]:
                out += [f"* {label.capitalize()}: {row[label]}", ""]
    out += [
        "## Identification by source",
        "",
        "| Source | Identified | Duplicates removed | Unique |",
        "|---|---|---|---|",
    ]
    for db, c in report["identification_by_source"].items():
        out.append(
            f"| {_label(db)} | {c['identified']} | {c['duplicates_removed']} | {c['unique']} |"
        )
    if report["problems"]:
        out += ["", "## Open problems", "", *(f"* {p}" for p in report["problems"])]
    return "\n".join(out) + "\n"
