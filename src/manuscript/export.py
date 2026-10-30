"""Exports: the manuscript as Markdown or editable Word (.docx), and the audit report.

Exports reproduce a stored version's rendered text exactly; nothing is recomputed. Figures
are not embedded: forest plots are the Stage 5 SVG artifacts (``plots/<analysis>.svg``), to
be submitted as separate figure files, and the legends name those files. The audit export
lists every finding with its location, source key and the action required, together with
human acknowledgements and (clearly labelled) advisory model reviews.
"""

from __future__ import annotations

import io
import re

from src.core.schemas.audit import AuditReport
from src.core.schemas.manuscript import ClaimReview, FindingAcknowledgement, ManuscriptVersion

BOLD_RE = re.compile(r"\*\*(.+?)\*\*")


def manuscript_markdown(version: ManuscriptVersion) -> str:
    return version.rendered_markdown + "\n"


def provenance_markdown(version: ManuscriptVersion) -> str:
    """A separate provenance note (not part of the submitted manuscript)."""
    lines = [
        f"# Provenance of manuscript version {version.version}",
        "",
        f"* Content SHA-256: `{version.content_sha256}`",
        f"* Created {version.created_at.isoformat()} by {version.created_by} "
        f"({version.actor_type.value}); parent version: {version.parent_version}",
        f"* Change: {version.change_summary}",
        "* Evidence fingerprint:",
    ]
    lines += [f"  * {k}: `{v}`" for k, v in sorted(version.evidence_fingerprint.items())]
    for p in version.provenance:
        model = f" {p.provider}/{p.model_id} (effort {p.effort or 'provider default'})"
        lines.append(
            f"* Generation: {p.method}{model if p.model_id else ''}; sections "
            f"{', '.join(p.sections) or 'all'}; LLM calls "
            f"{', '.join(p.llm_call_ids) or 'none'}"
        )
    return "\n".join(lines) + "\n"


def audit_markdown(
    report: AuditReport,
    acknowledgements: list[FindingAcknowledgement] | None = None,
    reviews: list[ClaimReview] | None = None,
) -> str:
    acks = {
        a.finding_id: a for a in acknowledgements or [] if a.audit_report_id == report.report_id
    }
    advisory: dict[str, list[ClaimReview]] = {}
    for r in reviews or []:
        if r.audit_report_id == report.report_id:
            advisory.setdefault(r.finding_id, []).append(r)
    lines = [
        f"# Audit report {report.report_id}",
        "",
        f"* Manuscript version {report.manuscript_version} (SHA-256 `{report.manuscript_sha256}`)",
        f"* Statistics output SHA-256 `{report.stats_output_sha256}`",
        f"* Auditor {report.auditor_version}; generated {report.generated_at.isoformat()}",
        f"* Complete: **{report.complete}**; deterministic checks passed: **{report.passed}**; "
        f"blocking: {report.failures}; requiring human review: {report.requires_human_review}",
        "",
        "Only deterministic checks decide this report. Advisory model reviews are listed "
        "separately and never change a finding. Final approval requires an authenticated "
        "human sign-off.",
        "",
        "## Scopes checked",
        "",
    ]
    lines += [f"* {k}: {v} unit(s)" for k, v in sorted(report.scopes_checked.items())]
    lines += [
        "",
        "## Coverage of expected outputs",
        "",
        "| Output | Required in | Found in | Status |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {c.description} | {', '.join(c.required_in)} | {', '.join(c.found_in)} | "
        f"{c.status.value} |"
        for c in report.coverage
    ]
    order = {
        "mismatch": 0,
        "not_found": 1,
        "missing": 2,
        "unsupported": 3,
        "not_checked": 4,
        "requires_human_review": 5,
        "verified": 6,
    }
    lines += ["", "## Findings", ""]
    for f in sorted(report.findings, key=lambda f: (order.get(f.status.value, 9), f.location)):
        if f.status.value == "verified":
            continue
        lines.append(f"### [{f.status.value}] {f.scope} — {f.location}")
        lines.append("")
        lines.append(f"* {f.message}")
        if f.excerpt:
            lines.append(f"* Text: “{f.excerpt}”")
        if f.source_key:
            lines.append(
                f"* Source key: `{f.source_key}`"
                + (f" — {f.source_value}" if f.source_value else "")
            )
        lines.append(f"* Action: {f.action}")
        if f.finding_id in acks:
            a = acks[f.finding_id]
            lines.append(
                f"* Human acknowledgement by {a.reviewer_name} "
                f"({a.acknowledged_at.date()}): {a.decision} — {a.note}"
            )
        for r in advisory.get(f.finding_id, []):
            lines.append(
                f"* Advisory model review (not a human assessment; {r.provider}/"
                f"{r.model_id}): {r.verdict} — {r.rationale}"
            )
        lines.append("")
    verified = sum(f.status.value == "verified" for f in report.findings)
    lines += [
        f"{verified} finding(s) verified; numeric checks: {len(report.numeric_checks)}; "
        f"citation checks: {len(report.citation_checks)}.",
        "",
        "## Limitations of the audit",
        "",
    ] + [f"* {x}" for x in report.limitations]
    return "\n".join(lines) + "\n"


def _add_runs(paragraph: object, text: str) -> None:
    pos = 0
    for m in BOLD_RE.finditer(text):
        if m.start() > pos:
            paragraph.add_run(text[pos : m.start()])  # type: ignore[attr-defined]
        paragraph.add_run(m.group(1)).bold = True  # type: ignore[attr-defined]
        pos = m.end()
    if pos < len(text):
        paragraph.add_run(text[pos:])  # type: ignore[attr-defined]


def manuscript_docx(version: ManuscriptVersion) -> bytes:
    """Editable Word document of the rendered manuscript (requires ``python-docx``)."""
    try:
        import docx
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("DOCX export needs python-docx: pip install -e '.[manuscript]'") from exc
    doc = docx.Document()
    lines = version.rendered_markdown.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if line.startswith("# "):
            doc.add_heading(line[2:], level=0)
        elif line.startswith("## "):
            doc.add_heading(line[3:], level=1)
        elif line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i])
                i += 1
            rows = [
                [c.strip() for c in r.strip().strip("|").split("|")]
                for r in block
                if not re.fullmatch(r"\|[\s|:-]+\|", r.strip())
            ]
            if rows:
                table = doc.add_table(rows=len(rows), cols=max(len(r) for r in rows))
                table.style = "Table Grid"
                for r, row in enumerate(rows):
                    for c, cell in enumerate(row):
                        target = table.cell(r, c).paragraphs[0]
                        _add_runs(target, cell.replace("<br>", "; "))
                        if r == 0:
                            for run in target.runs:
                                run.bold = True
            continue
        elif line.strip():  # incl. references, which keep their exact numbers as text
            _add_runs(doc.add_paragraph(), line)
        i += 1
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
