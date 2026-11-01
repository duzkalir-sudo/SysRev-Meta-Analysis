"""Session 10: collect the read-only evidence of a pilot project from the running API.

Usage (the API server must be running, e.g. ``uvicorn src.api_server.main:app``):

    python -m validation.session10.collect_pilot_evidence <project_id> \
        [--api http://127.0.0.1:8000] [--out session10_results/evidence]

Only ``GET`` endpoints are called and no reviewer token is sent, so nothing in the project
can change. Every response body is saved unchanged with its SHA-256 in ``index.json``, so
the validation report can cite identifiers and hashes instead of copying private files.
The bodies contain bibliographic data, verbatim quotations and file names from your pilot:
keep the output directory out of Git (``session10_results/`` is ignored).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

P = "/projects/{pid}"
ENDPOINTS: list[tuple[str, str, dict[str, str]]] = [
    ("project", P, {}),
    ("guards", P + "/guards", {}),
    ("protocol", P + "/protocol", {}),
    ("protocol_history", P + "/protocol/history", {}),
    ("searches", P + "/searches", {}),
    ("search_report_md", P + "/searches/report", {"format": "markdown"}),
    ("search_report_csv", P + "/searches/report", {"format": "csv"}),
    ("records", P + "/records", {}),
    ("duplicates", P + "/duplicates", {}),
    ("prisma", P + "/prisma", {}),
    ("screening_ta_summary", P + "/screening/title_abstract/summary", {}),
    ("screening_ta_decisions", P + "/screening/title_abstract/decisions", {}),
    ("screening_ta_conflicts", P + "/screening/title_abstract/conflicts", {}),
    ("screening_ft_summary", P + "/screening/full_text/summary", {}),
    ("screening_ft_decisions", P + "/screening/full_text/decisions", {}),
    ("fulltext_manifest", P + "/fulltext/manifest", {"only_open": "false"}),
    ("fulltext_manifest_csv", P + "/fulltext/manifest", {"only_open": "false", "format": "csv"}),
    ("fulltext_flow", P + "/fulltext/flow", {}),
    ("extraction_items", P + "/extraction/items", {}),
    ("extraction_export_csv", P + "/extraction/export", {"format": "csv"}),
    ("ai_verifications", P + "/extraction/ai-verifications", {}),
    ("studies", P + "/studies", {}),
    ("double_counting", P + "/extraction/double-counting", {}),
    ("rob", P + "/rob", {}),
    ("ai_settings", P + "/ai/settings", {}),
    ("ai_usage", P + "/ai/usage", {}),
    ("stats_plans", P + "/stats/plans", {}),
    ("stats_plan", P + "/stats/plan", {}),
    ("stats_output", P + "/stats/output", {}),
    ("reporting_status", P + "/reporting/status", {}),
    ("prisma_diagram", P + "/reporting/prisma/diagram", {}),
    ("checklist_prisma_main", P + "/reporting/checklists/prisma_2020_main", {}),
    ("checklist_prisma_abstract", P + "/reporting/checklists/prisma_2020_abstract", {}),
    ("checklist_swim", P + "/reporting/checklists/swim", {}),
    ("grade", P + "/reporting/grade", {}),
    ("sof", P + "/reporting/sof", {}),
    ("synthesis", P + "/reporting/synthesis", {}),
    ("swim", P + "/reporting/swim", {}),
    ("manuscript_status", P + "/manuscript/status", {}),
    ("manuscript_versions", P + "/manuscript/versions", {}),
    ("manuscript_audits", P + "/manuscript/audits", {}),
    ("manuscript_md", P + "/manuscript/export", {"format": "md"}),
    ("manuscript_docx", P + "/manuscript/export", {"format": "docx"}),
    ("manuscript_provenance", P + "/manuscript/export", {"format": "provenance"}),
    ("model_catalogue", "/models/catalogue", {}),
]
EXT = {
    "application/json": "json",
    "text/csv": "csv",
    "text/markdown": "md",
    "image/svg+xml": "svg",
    "image/png": "png",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
}


def _summary(name: str, body: Any) -> Any:
    """A short, structure-agnostic description (lengths and top-level keys only)."""
    if isinstance(body, list):
        return {"items": len(body)}
    if isinstance(body, dict):
        return {"keys": sorted(body)[:25]}
    return None


def main(argv: list[str] | None = None) -> int:
    import httpx

    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("project_id")
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--out", type=Path, default=Path("session10_results") / "evidence")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    args.out.mkdir(parents=True, exist_ok=True)
    index: dict[str, Any] = {
        "project_id": args.project_id,
        "api": args.api,
        "collected_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "responses": [],
    }
    with httpx.Client(base_url=args.api, timeout=60, trust_env=False) as client:
        index["health"] = client.get("/health").json()
        for name, path, params in ENDPOINTS:
            url = path.format(pid=args.project_id)
            r = client.get(url, params=params)
            ctype = r.headers.get("content-type", "").split(";")[0].strip()
            entry: dict[str, Any] = {
                "name": name,
                "url": url,
                "params": params,
                "http_status": r.status_code,
                "content_type": ctype,
            }
            if r.status_code == 200:
                fname = f"{name}.{EXT.get(ctype, 'bin')}"
                (args.out / fname).write_bytes(r.content)
                entry.update(
                    file=fname, bytes=len(r.content), sha256=hashlib.sha256(r.content).hexdigest()
                )
                if ctype == "application/json":
                    entry["summary"] = _summary(name, r.json())
            else:
                entry["detail"] = r.text[:300]
            index["responses"].append(entry)
            print(f"{r.status_code} {name}")
    (args.out / "index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    missing = [e["name"] for e in index["responses"] if e["http_status"] != 200]
    print(f"saved {len(index['responses']) - len(missing)} responses; not available: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
