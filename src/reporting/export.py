"""Exports of the Session 7 reports. Nothing is recomputed here: tables are rendered from the
evidence package with the same marker renderer as the manuscript, and every value is listed
with the evidence key and source it came from.

* PRISMA 2020 / SWiM checklists: CSV (``ReportingService.checklist_csv``), editable DOCX and
  XLSX, with the item wording, status, location, justification, reviewer, decision time,
  manuscript version and hashes.
* Summary of Findings (comparative outcomes) and the single-group evidence summary:
  Markdown and DOCX, followed by a source-key appendix.
"""

from __future__ import annotations

import io
import re
from typing import Any

from src.core.schemas.manuscript import ManuscriptSection
from src.manuscript.document import FACT_RE, render
from src.reporting.standards import PRISMA_REFERENCE, SWIM_REFERENCE

SOF_TITLE = (
    "Summary of Findings (GRADE). Relative effects are those of the project's statistics "
    "output; absolute effects are shown only where an assumed comparator risk with a stated "
    "source was recorded. Certainty ratings and reasons are the approving reviewers' "
    "judgements."
)
SUMMARY_TITLE = (
    "Evidence summary for single-group outcomes. This is not a comparative Summary of "
    "Findings table: there is no comparator, so no relative or absolute effects are presented, "
    "and any certainty rating uses the GRADE approach for overall prognosis."
)
HEADERS = (
    "checklist",
    "item",
    "section",
    "topic",
    "requirement",
    "status",
    "location",
    "justification",
    "reviewer",
    "decided at",
    "manuscript version",
    "stale",
)


def _render_template(package: Any, name: str) -> str:
    sec = ManuscriptSection(
        section_id="tables", heading="Tables", source=f"{{{{{name}}}}}", origin="template"
    )
    rendered = render([sec], package)
    return rendered.sections[0].text


def _keys(package: Any, name: str) -> list[str]:
    from src.manuscript.document import expand

    return sorted(set(FACT_RE.findall(expand(package.templates[name], package))))


def sof_markdown(package: Any) -> str:
    parts = ["# Summary of Findings and evidence summaries", ""]
    keys: list[str] = []
    if "table:sof" in package.templates:
        parts += [SOF_TITLE, "", _render_template(package, "table:sof"), ""]
        keys += _keys(package, "table:sof")
    else:
        parts += [
            "No comparative outcome has an approved, current GRADE assessment; no Summary "
            "of Findings table was produced.",
            "",
        ]
    if "table:evidence_summary" in package.templates:
        parts += [SUMMARY_TITLE, "", _render_template(package, "table:evidence_summary"), ""]
        keys += _keys(package, "table:evidence_summary")
    parts += ["## Source keys", "", "| Key | Value | Source |", "|---|---|---|"]
    for k in sorted(set(keys)):
        f = package.facts.get(k)
        if f is not None:
            parts.append(f"| `{k}` | {f.text.replace('|', '/')} | {f.source.replace('|', '/')} |")
    parts += ["", f"Evidence fingerprint: `{package.fingerprint_sha256}`"]
    return "\n".join(parts) + "\n"


def _md_table_rows(md: str) -> list[list[str]]:
    rows = []
    for line in md.splitlines():
        if not line.startswith("|") or re.fullmatch(r"\|(\s*-+\s*\|)+", line.strip()):
            continue
        rows.append([c.strip().replace("<br>", "\n") for c in line.strip().strip("|").split("|")])
    return rows


def _docx() -> Any:
    try:
        import docx
    except ImportError as exc:  # pragma: no cover - depends on the installation
        raise RuntimeError(
            "DOCX export needs python-docx (pip install -e '.[manuscript]')"
        ) from exc
    return docx.Document()


def _add_table(doc: Any, rows: list[list[str]]) -> None:
    if not rows:
        return
    table = doc.add_table(rows=len(rows), cols=max(len(r) for r in rows))
    table.style = "Table Grid"
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            table.cell(i, j).text = cell


def sof_docx(package: Any) -> bytes:
    doc = _docx()
    doc.add_heading("Summary of Findings and evidence summaries", level=1)
    for name, title in (("table:sof", SOF_TITLE), ("table:evidence_summary", SUMMARY_TITLE)):
        if name in package.templates:
            doc.add_paragraph(title)
            _add_table(doc, _md_table_rows(_render_template(package, name)))
    doc.add_heading("Source keys", level=2)
    keys = sorted(
        {
            k
            for n in ("table:sof", "table:evidence_summary")
            if n in package.templates
            for k in _keys(package, n)
        }
    )
    _add_table(
        doc,
        [["Key", "Value", "Source"]]
        + [[k, package.facts[k].text, package.facts[k].source] for k in keys if k in package.facts],
    )
    doc.add_paragraph(f"Evidence fingerprint: {package.fingerprint_sha256}")
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _rows(data: dict[str, Any]) -> list[list[str]]:
    out = []
    for r in data["items"]:
        d = r["decision"] or {}
        out.append(
            [
                data["checklist"],
                r["item_id"],
                r["section"],
                r["topic"],
                r["requirement"],
                r["status"].replace("_", " "),
                d.get("location", ""),
                d.get("justification", ""),
                d.get("reviewer_name", ""),
                str(d.get("decided_at", "")),
                str(d.get("manuscript_version", "")),
                "yes" if r["stale"] else "no",
            ]
        )
    return out


def checklist_docx(data: dict[str, Any]) -> bytes:
    doc = _docx()
    doc.add_heading(data["standard"], level=1)
    doc.add_paragraph(
        f"Manuscript version {data['manuscript_version']} (SHA-256 {data['manuscript_sha256']}); "
        f"evidence SHA-256 {data['evidence_sha256']}. {data['note']}"
    )
    _add_table(doc, [list(HEADERS[1:])] + [r[1:] for r in _rows(data)])
    ref = SWIM_REFERENCE if data["checklist"] == "swim" else PRISMA_REFERENCE
    doc.add_paragraph(f"Item wording reproduced from: {ref}")
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def checklist_xlsx(data: dict[str, Any]) -> bytes:
    try:
        from openpyxl import Workbook
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("XLSX export needs openpyxl") from exc
    wb = Workbook()
    ws = wb.active
    ws.title = "checklist"
    ws.append(list(HEADERS))
    for r in _rows(data):
        ws.append(r)
    meta = wb.create_sheet("provenance")
    for k in ("standard", "manuscript_version", "manuscript_sha256", "evidence_sha256", "note"):
        meta.append([k, str(data[k])])
    ref = SWIM_REFERENCE if data["checklist"] == "swim" else PRISMA_REFERENCE
    meta.append(["item wording", ref])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
