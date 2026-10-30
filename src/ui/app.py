"""SysRev local reviewer UI (Streamlit). Start with ``python -m src.ui``.

A thin client: human decisions (gates, stage advances, adjudication, PDF upload and
identity confirmation, extraction accept/reject, arms, risk-of-bias approval) are API
calls authenticated with the reviewer's bearer token. AI steps run the existing runners
in-process and only ever create proposals plus an audit trail of the calls. No API key
or reviewer token is written to the database, a file or a log.
"""

from __future__ import annotations

import base64
from datetime import date
from typing import Any

import streamlit as st

from src.core.enums import (
    ClinicalOutcomeType,
    ComparisonType,
    EffectMeasure,
    GateVerdict,
    ReviewDesign,
    ScreeningStage,
    ScreeningVerdict,
    SourceDatabase,
    StudyDesign,
)
from src.core.schemas.protocol import ALLOWED_MEASURES
from src.llm.credentials import BILLING_NOTE, ENV_VARS, CredentialResolver, install_log_redaction
from src.llm.tasks import TASK_LABELS, AITask, ProjectAISettings, TaskModelSettings
from src.search.databases import DATABASES
from src.ui import protocol_form as pf
from src.ui.services import ApiResult, UIBackend

PROVIDER_DEFAULT = "(provider default — not sent)"
NOT_CONFIGURED = "— choose a model —"
PAGES = (
    "Projects", "Protocol", "Searches & import", "Pipeline status", "AI models & keys",
    "Run AI steps", "Screening", "Full text & extraction", "Risk of bias", "Statistics",
    "Usage & costs",
)  # fmt: skip


@st.cache_resource
def backend() -> UIBackend:
    install_log_redaction()
    return UIBackend.from_settings()


def credentials() -> CredentialResolver:
    if "credentials" not in st.session_state:
        st.session_state["credentials"] = CredentialResolver()
    resolver: CredentialResolver = st.session_state["credentials"]
    return resolver


def token() -> str | None:
    value = st.session_state.get("reviewer_token") or ""
    return str(value) if value else None


def show(result: ApiResult, success: str = "Done.") -> bool:
    if result.ok:
        st.success(success)
    elif result.status == 401:
        st.error("Not authenticated: enter a valid reviewer token in the sidebar.")
    else:
        st.error(f"Refused ({result.status}): {result.error}")
    return result.ok


def project_id() -> str | None:
    value = st.session_state.get("project_id")
    return str(value) if value else None


def require_project() -> str:
    pid = project_id()
    if pid is None:
        st.info("Select or create a project first (page 'Projects').")
        st.stop()
    return str(pid)


# ---------------------------------------------------------------------- pages
def page_projects(be: UIBackend) -> None:
    st.header("Projects")
    projects = be.api.get("/projects").data or []
    if projects:
        labels = {f"{p['title']}  ·  {p['phase']}  ·  {p['project_id'][:8]}": p["project_id"]
                  for p in projects}  # fmt: skip
        choice = st.selectbox("Select project", list(labels))
        if st.button("Open selected project") and choice:
            st.session_state["project_id"] = labels[choice]
            st.rerun()
    with st.form("create_project"):
        title = st.text_input("New project title", key="new_project_title")
        if st.form_submit_button("Create project") and title.strip():
            result = be.api.post("/projects", json={"title": title.strip()})
            if show(result, "Project created."):
                st.session_state["project_id"] = result.data["project_id"]
    st.caption(
        "Next: draft and approve the protocol (page 'Protocol'), pass Gate 1 ('Pipeline "
        "status'), then run or import the searches ('Searches & import')."
    )


# ---------------------------------------------------------------------- protocol (Session 5)
def _work(pid: str, server: dict[str, Any] | None) -> dict[str, Any]:
    """The page's working copy of the protocol (reloaded from the server after each save)."""
    key = f"proto_work_{pid}"
    if key not in st.session_state:
        st.session_state[key] = dict(server) if server else pf.empty_protocol()
    work: dict[str, Any] = st.session_state[key]
    return work


def _save_work(pid: str, work: dict[str, Any]) -> None:
    st.session_state[f"proto_work_{pid}"] = work


def page_protocol(be: UIBackend) -> None:
    pid = require_project()
    st.header("Protocol (Stage 1)")
    view = be.api.get(f"/projects/{pid}/protocol").data or {}
    project = be.api.get(f"/projects/{pid}").data or {}
    server = view.get("protocol")
    editable = project.get("phase") == "stage1_protocol"
    if server:
        st.markdown(
            f"**Version {server['version']} · {server['status']}**"
            + (
                f" · approved by {server['approved_by']} at {server['approved_at']}"
                if server.get("approved_by")
                else ""
            )
        )
    for b in view.get("approval_blockers", []):
        st.warning(f"Not ready for approval: {b}")
    if not editable:
        st.info(
            "The protocol can be edited only in Stage 1. After Gate 1 the approved version "
            "is locked (amendments after Gate 1 are not supported yet)."
        )
    work = _work(pid, server)

    st.subheader("1 · Review question and design")
    with st.form("proto_question"):
        title = st.text_input("Review title", work["review_title"], key="pq_title")
        question = st.text_area("Research question", work["research_question"], key="pq_rq")
        designs = [d.value for d in ReviewDesign]
        design = st.radio(
            "Review design",
            designs,
            horizontal=True,
            key="pq_design",
            index=designs.index(work["review_design"]),
        )
        population = st.text_area("Population (P)", work["population"], key="pq_p")
        intervention = st.text_area("Intervention (I)", work["intervention"], key="pq_i")
        comparator = st.text_input(
            "Comparator (C; leave empty for single-arm reviews)",
            work.get("comparator") or "",
            key="pq_c",
        )
        prospero = st.text_input(
            "PROSPERO ID (optional, CRD + 11 digits)",
            work.get("prospero_id") or "",
            key="pq_prospero",
        )
        if st.form_submit_button("Apply question & design"):
            work |= {
                "review_title": title,
                "research_question": question,
                "review_design": design,
                "population": population,
                "intervention": intervention,
                "comparator": comparator.strip() or None,
                "prospero_id": prospero.strip() or None,
            }
            _save_work(pid, work)

    st.subheader("2 · Outcomes, effect measures and follow-up timepoints")
    st.dataframe(
        [
            {
                "id": o["outcome_id"],
                "name": o["name"],
                "kind": o["kind"],
                "data type": o["data_type"],
                "comparison": o.get("comparison"),
                "effect measure": o["preferred_effect_measure"],
                "timepoints": pf.format_timepoints(o.get("timepoints", [])),
            }
            for o in work["outcomes"]
        ]
    )
    with st.form("proto_outcome"):
        c = st.columns(3)
        oid = c[0].text_input("Outcome ID (letters, digits, _ . : -)", key="po_id")
        name = c[1].text_input("Name", key="po_name")
        kind = c[2].radio("Kind", ["primary", "secondary"], horizontal=True, key="po_kind")
        c = st.columns(4)
        data_type = c[0].selectbox(
            "Data type", ["dichotomous", "continuous", "time_to_event", "count_rate"], key="po_type"
        )
        comparison = c[1].selectbox("Comparison", [x.value for x in ComparisonType], key="po_cmp")
        measure = c[2].selectbox("Effect measure", [m.value for m in EffectMeasure], key="po_em")
        clinical = c[3].selectbox(
            "Clinical type", [x.value for x in ClinicalOutcomeType], key="po_clin"
        )
        timepoints = st.text_input(
            "Timepoints (e.g. '5 years; 24-36 months; last follow-up')", key="po_tp"
        )
        definition = st.text_input("Operational definition (optional)", key="po_def")
        grading = st.text_input("Grading system (optional, e.g. CTCAE v5.0)", key="po_grade")
        if st.form_submit_button("Add / replace outcome"):
            try:
                outcome = {
                    "outcome_id": oid.strip(),
                    "name": name.strip(),
                    "kind": kind,
                    "data_type": data_type,
                    "comparison": comparison,
                    "preferred_effect_measure": measure,
                    "clinical_type": clinical,
                    "timepoints": pf.parse_timepoints(timepoints),
                    "definition": definition.strip() or None,
                    "grading_system": grading.strip() or None,
                }
                work["outcomes"] = pf.upsert_outcome(work["outcomes"], outcome)
                _save_work(pid, work)
            except ValueError as exc:
                st.error(str(exc))
    with st.expander("Admissible effect measures"):
        st.dataframe(
            [
                {
                    "comparison": k[0].value,
                    "data type": k[1],
                    "measures": ", ".join(sorted(m.value for m in v)),
                }
                for k, v in ALLOWED_MEASURES.items()
            ]
        )
    if work["outcomes"]:
        with st.form("proto_outcome_remove"):
            drop = st.selectbox(
                "Outcome to remove", [o["outcome_id"] for o in work["outcomes"]], key="po_drop"
            )
            if st.form_submit_button("Remove outcome"):
                work["outcomes"] = [o for o in work["outcomes"] if o["outcome_id"] != drop]
                _save_work(pid, work)

    st.subheader("3 · Eligibility criteria and study designs")
    with st.form("proto_eligibility"):
        sd = st.multiselect(
            "Eligible study designs",
            [d.value for d in StudyDesign],
            default=work["study_designs"],
            key="pe_designs",
        )
        inc = st.text_area(
            "Inclusion criteria (one per line)", "\n".join(work["inclusion_criteria"]), key="pe_inc"
        )
        excl = st.text_area(
            "Exclusion criteria (one per line)", "\n".join(work["exclusion_criteria"]), key="pe_exc"
        )
        if st.form_submit_button("Apply eligibility"):
            work |= {
                "study_designs": sd,
                "inclusion_criteria": pf.lines(inc),
                "exclusion_criteria": pf.lines(excl),
            }
            _save_work(pid, work)

    st.subheader("4 · MeSH terms")
    with st.form("proto_mesh"):
        mesh = st.text_area(
            "One descriptor per line; add [noexp] or [majr] if needed",
            pf.format_mesh(work["mesh_terms"]),
            key="pm_terms",
        )
        if st.form_submit_button("Apply MeSH terms"):
            work["mesh_terms"] = pf.parse_mesh(mesh, work["mesh_terms"])
            _save_work(pid, work)
    if server and server["mesh_terms"]:
        st.dataframe(
            [
                {
                    "descriptor": m["descriptor"],
                    "verified in NLM MeSH": m["verified"],
                    "verified at": m.get("verified_at"),
                }
                for m in server["mesh_terms"]
            ]
        )

    st.subheader("5 · Databases and database-specific search strategies")
    with st.form("proto_dbs"):
        labels = {DATABASES[d].label: d.value for d in DATABASES}
        chosen = st.multiselect(
            "Databases to search",
            list(labels),
            key="pd_dbs",
            default=[DATABASES[SourceDatabase(d)].label for d in work["selected_databases"]],
        )
        if st.form_submit_button("Apply database selection"):
            work["selected_databases"] = [labels[c] for c in chosen]
            work["search_strategies"] = [
                s for s in work["search_strategies"] if s["database"] in work["selected_databases"]
            ]
            _save_work(pid, work)
    for db_value in work["selected_databases"]:
        dbi = DATABASES[SourceDatabase(db_value)]
        existing = [s for s in work["search_strategies"] if s["database"] == db_value]
        current = existing[0] if existing else {}
        mode = "direct API search" if dbi.access == "api" else "manual export & import"
        with st.expander(
            f"{dbi.label} — {mode}",
            expanded=not existing,
        ):
            st.caption(dbi.note)
            if len(existing) > 1:
                st.warning(
                    "More than one strategy exists for this database; edit them "
                    "through the API (PUT /projects/{id}/protocol)."
                )
                continue
            with st.form(f"proto_strategy_{db_value}"):
                platform = st.text_input(
                    "Platform / interface",
                    current.get("platform") or dbi.platform,
                    key=f"ps_platform_{db_value}",
                )
                query = st.text_area(
                    "Search string (exactly as it will be run)",
                    current.get("query", ""),
                    key=f"ps_query_{db_value}",
                )
                c = st.columns(2)
                dfrom = c[0].text_input(
                    "Date from (YYYY-MM-DD, optional)",
                    current.get("date_from") or "",
                    key=f"ps_from_{db_value}",
                )
                dto = c[1].text_input(
                    "Date to (YYYY-MM-DD, optional)",
                    current.get("date_to") or "",
                    key=f"ps_to_{db_value}",
                )
                filters = st.text_area(
                    "Filters (one per line; API databases: ANDed to the query"
                    " or OpenAlex filter expressions)",
                    "\n".join(current.get("filters", [])),
                    key=f"ps_filters_{db_value}",
                )
                if st.form_submit_button(f"Apply {dbi.label} strategy"):
                    try:
                        strategy = (
                            None
                            if not query.strip()
                            else {
                                "database": db_value,
                                "platform": platform.strip() or None,
                                "query": query.strip(),
                                "date_from": pf.parse_date(dfrom),
                                "date_to": pf.parse_date(dto),
                                "filters": pf.lines(filters),
                                "generated_by": "human",
                            }
                        )
                        work["search_strategies"] = pf.set_strategy(
                            work["search_strategies"], db_value, strategy
                        )
                        _save_work(pid, work)
                    except ValueError as exc:
                        st.error(str(exc))

    st.subheader("6 · Save, verify and approve")
    st.caption(
        "Saving validates the whole protocol on the server. Verification flags in the "
        "working copy are ignored: MeSH terms are verified against NLM MeSH and "
        "strategies by a preview or an attestation below."
    )
    cols = st.columns(3)
    if cols[0].button("Save draft (authenticated)", disabled=not editable, key="proto_save"):
        result = be.api.request("PUT", f"/projects/{pid}/protocol", token=token(), json=work)
        if show(result, "Draft saved."):
            for item in result.data.get("invalidated", []):
                st.warning(f"No longer valid: {item}")
            if result.data.get("material_changes"):
                st.info(f"Material changes: {result.data['material_changes']}")
            _save_work(pid, result.data["protocol"])
        elif isinstance(result.data, dict) and isinstance(result.data.get("detail"), dict):
            for err in result.data["detail"].get("errors", []):
                st.error(err)
    if cols[1].button(
        "Verify MeSH terms in NLM MeSH (authenticated)",
        disabled=not editable,
        key="proto_mesh_verify",
    ):
        result = be.api.post(f"/projects/{pid}/protocol/mesh/verify", token=token())
        if show(result, "MeSH lookup finished."):
            for d, ok in result.data["mesh_results"].items():
                (st.success if ok else st.error)(f"{d}: {'found' if ok else 'NOT found'} in MeSH")
    if cols[2].button(
        "Approve protocol (authenticated)", disabled=not editable, key="proto_approve"
    ):
        show(
            be.api.post(f"/projects/{pid}/protocol/approve", token=token(), json={}),
            "Protocol approved. Next: 'Pipeline status' → advance to Gate 1.",
        )
    for s in view.get("strategies", []):
        label = f"{s['label']} strategy {s['sha256'][:12]}"
        state = (
            f"validated ({s['validation_method']})" if s["syntax_validated"] else "not validated"
        )
        st.markdown(f"**{label}** — {state}")
        if s["access"] == "api":
            if st.button(
                f"Preview count: {label}", key=f"preview_{s['sha256']}", disabled=not editable
            ):
                _show_preview(
                    be.api.post(
                        f"/projects/{pid}/protocol/strategies/{s['sha256']}/preview", token=token()
                    )
                )
        else:
            with st.form(f"attest_{s['sha256']}"):
                note = st.text_input(
                    "Attestation (e.g. 'run in Embase.com on 2026-10-30, "
                    "no syntax errors, 1,234 hits')",
                    key=f"att_{s['sha256']}",
                )
                if st.form_submit_button("Attest syntax checked (authenticated)") and note:
                    show(
                        be.api.post(
                            f"/projects/{pid}/protocol/strategies/{s['sha256']}/attest",
                            token=token(),
                            json={"note": note},
                        )
                    )
    with st.expander("Version history (append-only)"):
        st.dataframe(
            [
                {
                    k: e.get(k)
                    for k in (
                        "occurred_at",
                        "protocol_version",
                        "action",
                        "status",
                        "actor",
                        "note",
                        "material_changes",
                        "invalidated",
                    )
                }
                for e in be.api.get(f"/projects/{pid}/protocol/history").data or []
            ]
        )


def _show_preview(result: ApiResult) -> None:
    if not show(result, "Preview recorded in the search log."):
        return
    run = result.data
    if run["status"] == "failed":
        st.error("Preview failed: " + "; ".join(run["problems"]))
        return
    st.metric("Records reported by the database", run["hits_reported"])
    if run.get("query_translation"):
        st.caption(f"Query translation: {run['query_translation']}")
    for w in run.get("warnings", []):
        st.warning(w)


# ---------------------------------------------------------------------- searches (Session 5)
def page_searches(be: UIBackend) -> None:
    pid = require_project()
    st.header("Searches & import (Stage 2)")
    project = be.api.get(f"/projects/{pid}").data or {}
    data = be.api.get(f"/projects/{pid}/searches").data or {}
    open_ = project.get("phase") == "stage2_search_screening"
    if not open_:
        st.info(
            "Searches are executed or imported in Stage 2, after Gate 1 approved the "
            f"protocol. Current phase: {project.get('phase')}."
        )
    for p in data.get("problems", []):
        st.warning(p)
    st.subheader("Approved strategies")
    for s in data.get("strategies", []):
        sha = s["strategy_sha256"]
        dbi = next(i for i in DATABASES.values() if i.label == s["database"])
        with st.expander(f"{s['database']} · {s['status']} · {sha[:12]}"):
            st.code(s["query"], language=None)
            st.caption(
                f"Protocol restrictions — dates: {s['date_from'] or '—'} to {s['date_to'] or '—'}; "
                f"languages: {'; '.join(s['languages']) or 'none'}; filters: "
                f"{' | '.join(s['filters']) or 'none'}. Platform: {s['platform'] or '—'}. "
                f"{dbi.note}"
            )
            if dbi.access == "api":
                c = st.columns(2)
                if c[0].button("Preview count", key=f"spv_{sha}", disabled=not open_):
                    _show_preview(
                        be.api.post(
                            f"/projects/{pid}/protocol/strategies/{sha}/preview", token=token()
                        )
                    )
                if c[1].button(
                    "Execute approved search (authenticated)", key=f"exe_{sha}", disabled=not open_
                ):
                    r = be.api.post(
                        f"/projects/{pid}/searches/strategies/{sha}/execute", token=token()
                    )
                    if show(r, "Search executed and recorded."):
                        _run_summary(r.data)
            _import_form(be, pid, sha, dbi.label, open_)
            with st.form(f"notsearched_{sha}"):
                reason = st.text_input("Reason this database was not searched", key=f"ns_{sha}")
                if st.form_submit_button("Record as NOT searched (authenticated)") and reason:
                    show(
                        be.api.post(
                            f"/projects/{pid}/searches/strategies/{sha}/not-searched",
                            token=token(),
                            json={"note": reason},
                        )
                    )
    _acknowledgements(be, pid, data)
    st.subheader("Prepare records for screening")
    st.caption(
        "Deterministic deduplication (DOI/PMID/PMCID/OpenAlex ID, then near-identical "
        "titles). Pairs with conflicting evidence are not merged: they are listed below "
        "for a human decision."
    )
    if st.button(
        "Deduplicate and prepare for screening (authenticated)", disabled=not open_, key="prepare"
    ):
        r = be.api.post(f"/projects/{pid}/searches/prepare", token=token())
        if show(r, "Records prepared; PRISMA snapshot recorded."):
            st.json({k: v for k, v in r.data["summary"].items() if k != "match_reasons"})
    by_source = data.get("identification_by_source") or {}
    if by_source:
        st.dataframe([{"source": k, **v} for k, v in by_source.items()])
    _duplicates(be, pid)
    prisma = be.api.get(f"/projects/{pid}/prisma")
    st.subheader("PRISMA 2020 counts (derived from stored records and decisions)")
    if prisma.ok:
        st.json(prisma.data)
    if st.button("Recompute PRISMA from records (authenticated)", key="prisma_recompute"):
        show(be.api.post(f"/projects/{pid}/prisma/recompute", token=token()), "Recomputed.")
    with st.expander("Search log (append-only, including previews)"):
        st.dataframe(
            [
                {
                    k: r.get(k)
                    for k in (
                        "executed_at",
                        "kind",
                        "status",
                        "database",
                        "hits_reported",
                        "records_saved",
                        "actor",
                        "note",
                    )
                }
                for r in data.get("runs", [])
            ]
        )
    md = be.api.get(f"/projects/{pid}/searches/report", params={"format": "markdown"})
    st.download_button(
        "Download PRISMA-S search summary (Markdown)", md.data or "", file_name="search_summary.md"
    )


def _run_summary(run: dict[str, Any]) -> None:
    if run["status"] == "failed":
        st.error("Search failed, nothing identified: " + "; ".join(run["problems"]))
        return
    st.write(
        f"Reported by the database: {run['hits_reported']} · saved: {run['records_saved']}"
        f" · status: {run['status']}"
    )
    for p in run.get("problems", []):
        st.warning(p)


def _import_form(be: UIBackend, pid: str, sha: str, label: str, open_: bool) -> None:
    with st.form(f"import_{sha}"):
        st.markdown(f"**Import an export file from {label}**")
        upload = st.file_uploader(
            "RIS, BibTeX or CSV export", type=["ris", "bib", "csv"], key=f"imp_file_{sha}"
        )
        c = st.columns(3)
        fmt = c[0].selectbox("Format", ["ris", "bibtex", "csv"], key=f"imp_fmt_{sha}")
        searched_on = c[1].text_input(
            "Date the search was run (YYYY-MM-DD)", date.today().isoformat(), key=f"imp_date_{sha}"
        )
        hits = c[2].number_input(
            "Total records the interface reported for the whole search (not this file)",
            min_value=1,
            value=None,
            key=f"imp_hits_{sha}",
        )
        no_total = st.checkbox(
            "The interface did not show a total (completeness must then be acknowledged)",
            key=f"imp_nototal_{sha}",
        )
        confirmed = st.checkbox(
            "I applied the protocol's date limits, language limits and filters (shown above) "
            "in the vendor interface",
            key=f"imp_restr_{sha}",
        )
        restr_note = st.text_input(
            "How they were applied (optional, e.g. 'Embase Limits: Article, 2000-2025')",
            key=f"imp_restr_note_{sha}",
        )
        columns = st.text_input(
            'CSV column overrides (JSON, e.g. {"title": "Study Title"})', key=f"imp_cols_{sha}"
        )
        c2 = st.columns(2)
        check = c2[0].form_submit_button("Check file (nothing stored)")
        do_import = c2[1].form_submit_button("Import (authenticated)", disabled=not open_)
        if (check or do_import) and upload is not None:
            params: dict[str, Any] = {"format": fmt}
            if columns.strip():
                params["columns"] = columns.strip()
            if check:
                r = be.api.post(
                    f"/projects/{pid}/searches/import-preview",
                    content=upload.getvalue(),
                    params=params,
                )
                if show(r, f"{r.data.get('entries') if r.ok else 0} entries read."):
                    st.json(r.data)
            elif hits is None and not no_total:
                st.error("Enter the total the interface reported, or tick that it showed none.")
            else:
                params |= {
                    "filename": upload.name,
                    "searched_on": searched_on,
                    "restrictions_confirmed": confirmed,
                }
                if restr_note.strip():
                    params["restrictions_note"] = restr_note.strip()
                if hits is not None and not no_total:
                    params["hits_reported"] = int(hits)
                r = be.api.post(
                    f"/projects/{pid}/searches/strategies/{sha}/import",
                    token=token(),
                    content=upload.getvalue(),
                    params=params,
                )
                if show(r, "File imported and preserved."):
                    _run_summary(r.data)
    with st.form(f"zero_{sha}"):
        st.markdown(f"**The {label} search ran but returned no records**")
        zdate = st.text_input(
            "Date the search was run (YYYY-MM-DD)", date.today().isoformat(), key=f"zero_date_{sha}"
        )
        znote = st.text_input(
            "Attestation (e.g. 'run in Embase.com, 0 results')", key=f"zero_note_{sha}"
        )
        zconf = st.checkbox("The protocol's limits and filters were applied", key=f"zero_r_{sha}")
        if (
            st.form_submit_button("Record zero-result search (authenticated)", disabled=not open_)
            and znote
        ):
            r = be.api.post(
                f"/projects/{pid}/searches/strategies/{sha}/zero-results",
                token=token(),
                json={"searched_on": zdate, "note": znote, "restrictions_confirmed": zconf},
            )
            show(r, "Zero-result search recorded.")


def _acknowledgements(be: UIBackend, pid: str, data: dict[str, Any]) -> None:
    mismatched = [r for r in data.get("runs", []) if r["status"] == "count_mismatch"]
    shortfalls = [
        s
        for s in data.get("strategies", [])
        if s["reconciliation"] in ("mismatch", "unknown", "conflicting")
        and not s["reconciliation_acknowledged"]
    ]
    if not (mismatched or shortfalls):
        return
    st.subheader("Count discrepancies (need a human acknowledgement)")
    with st.form("acknowledge"):
        options = {
            f"execution {r['run_id'][:12]} ({r['database']})": ("run", r["run_id"])
            for r in mismatched
        }
        options |= {
            f"imports for {s['database']} {s['strategy_sha256'][:12]} ({s['reconciliation']})": (
                "strategy",
                s["strategy_sha256"],
            )
            for s in shortfalls
        }
        choice = st.selectbox("Discrepancy", list(options), key="ack_choice")
        note = st.text_area(
            "Explanation (e.g. 'Embase lists 3 conference records it cannot export')",
            key="ack_note",
        )
        if st.form_submit_button("Acknowledge (authenticated)") and note and choice:
            kind, ref = options[choice]
            body = {"note": note, ("run_id" if kind == "run" else "strategy_sha256"): ref}
            show(be.api.post(f"/projects/{pid}/searches/acknowledge", token=token(), json=body))


def _duplicates(be: UIBackend, pid: str) -> None:
    candidates = be.api.get(f"/projects/{pid}/duplicates").data or []
    open_items = [c for c in candidates if c["resolution"] == "unresolved"]
    st.subheader(f"Possible duplicates for human decision ({len(open_items)} unresolved)")
    if not candidates:
        return
    st.dataframe(
        [
            {
                k: c.get(k)
                for k in (
                    "candidate_id",
                    "record_id_a",
                    "record_id_b",
                    "match_basis",
                    "conflict",
                    "title_similarity",
                    "resolution",
                    "resolved_by",
                )
            }
            for c in candidates
        ]
    )
    if open_items:
        with st.form("dup_resolve"):
            cid = st.selectbox("Pair", [c["candidate_id"] for c in open_items], key="dup_id")
            resolution = st.radio(
                "Decision", ["distinct", "duplicate_keep_a", "duplicate_keep_b"], key="dup_res"
            )
            note = st.text_input("Note", key="dup_note")
            if st.form_submit_button("Record decision (authenticated)") and cid:
                show(
                    be.api.post(
                        f"/projects/{pid}/duplicates/{cid}/resolve",
                        token=token(),
                        json={"resolution": resolution, "note": note or None},
                    )
                )


def page_pipeline(be: UIBackend) -> None:
    pid = require_project()
    st.header("Pipeline status")
    view = be.api.get(f"/projects/{pid}")
    if not view.ok:
        st.error(view.error)
        return
    st.metric("Current phase", view.data["phase"])
    gate = view.data["awaiting_gate"]
    guards = be.api.get(f"/projects/{pid}/guards").data or {}
    for target, reasons in guards.items():
        if reasons:
            st.warning(f"Blocked → {target}:\n\n" + "\n".join(f"- {r}" for r in reasons))
        else:
            st.success(f"No blocking reasons for → {target}")
    if gate:
        st.subheader(f"Human gate {gate}")
        with st.form("gate"):
            verdict = st.radio("Decision", [v.value for v in GateVerdict], horizontal=True)
            comments = st.text_area("Comments (required for rejection)")
            if st.form_submit_button("Submit gate decision"):
                show(be.api.post(f"/projects/{pid}/gates/{gate}", token=token(),
                                 json={"verdict": verdict, "comments": comments or None}),
                     "Gate decision recorded.")  # fmt: skip
    elif guards:
        with st.form("advance"):
            target = st.selectbox("Advance to", list(guards))
            reason = st.text_input("Reason (optional)")
            if st.form_submit_button("Request transition (authenticated)"):
                show(be.api.post(f"/projects/{pid}/advance", token=token(),
                                 json={"target": target, "reason": reason or None}),
                     "Transition recorded.")  # fmt: skip
    with st.expander("Transition history"):
        st.dataframe(view.data["history"])


def _task_editor(
    be: UIBackend, task: AITask, current: TaskModelSettings | None
) -> TaskModelSettings | None:
    st.markdown(f"**{TASK_LABELS[task]}**")
    providers = ["anthropic", "openai"]
    cols = st.columns(4)
    provider = cols[0].selectbox(
        "Provider", providers, key=f"{task}_prov",
        index=providers.index(current.provider) if current else 0,
    )  # fmt: skip
    models = be.catalogue.models(provider)
    ids = [m.model_id for m in models]
    if not ids:
        st.warning(f"No {provider} models in the catalogue.")
        return None
    options = [NOT_CONFIGURED, *ids]
    model_id = cols[1].selectbox(
        "Exact API model ID", options, key=f"{task}_model",
        index=options.index(current.model_id) if current and current.model_id in ids else 0,
    )  # fmt: skip
    if model_id == NOT_CONFIGURED:
        st.caption("Not configured: this task cannot run until a model is chosen explicitly.")
        return None
    cap = be.catalogue.get(provider, model_id)
    efforts = [PROVIDER_DEFAULT, *cap.efforts]
    cur_effort = current.effort if current and current.effort in cap.efforts else None
    effort = cols[2].selectbox(
        "Effort (sent as-is)", efforts, key=f"{task}_effort",
        index=efforts.index(cur_effort) if cur_effort else 0,
    )  # fmt: skip
    max_out = cols[3].number_input(
        "Max output tokens", min_value=256, max_value=cap.max_output_tokens or 128000,
        value=current.max_output_tokens if current else 16000, step=256, key=f"{task}_max",
    )  # fmt: skip
    if not cap.verified:
        st.warning(f"Unverified catalogue entry: {cap.source}")
    else:
        st.caption(f"Catalogue source: {cap.source}")
    return TaskModelSettings(
        provider=provider, model_id=model_id,
        effort=None if effort == PROVIDER_DEFAULT else effort, max_output_tokens=int(max_out),
    )  # fmt: skip


def page_models(be: UIBackend) -> None:
    pid = require_project()
    st.header("AI models & API keys")
    st.info(BILLING_NOTE)
    st.subheader("API keys (never stored in the project, Git, logs or exports)")
    creds = credentials()
    for provider in ("anthropic", "openai"):
        source = creds.source(provider)
        st.markdown(f"**{provider}** — key: {'available via ' + source if source else 'not set'}"
                    f" (environment variable {ENV_VARS[provider]})")  # fmt: skip
        key = st.text_input(f"{provider} API key", type="password", key=f"key_{provider}")
        c1, c2, c3 = st.columns(3)
        if c1.button("Use for this session only", key=f"use_{provider}") and key:
            creds.set_session_key(provider, key)
            st.success("Key held in memory for this browser session only.")
        if creds.keyring_available():
            if c2.button("Save in OS credential store", key=f"save_{provider}") and key:
                creds.save_to_keyring(provider, key)
                st.success("Saved in the operating-system credential store.")
            if c3.button("Remove from OS store", key=f"del_{provider}"):
                creds.delete_from_keyring(provider)
        else:
            c2.caption("Install 'keyring' to use the OS credential store.")
    st.subheader("Model per task")
    with be.repo() as repo:
        current = repo.latest_ai_settings(pid) or ProjectAISettings()
    tasks: dict[AITask, TaskModelSettings] = {}
    for task in AITask:
        edited = _task_editor(be, task, current.tasks.get(task))
        if edited is not None:
            tasks[task] = edited
    st.subheader("Local usage warnings and stop threshold (not a provider-enforced cap)")
    c = st.columns(4)
    warn_usd = c[0].number_input("Warn at USD (estimate)", min_value=0.0,
                                 value=current.warn_usd or 0.0)  # fmt: skip
    stop_usd = c[1].number_input("Stop new calls at USD (estimate)", min_value=0.0,
                                 value=current.stop_usd or 0.0)  # fmt: skip
    warn_tok = c[2].number_input("Warn at tokens", min_value=0, value=current.warn_tokens or 0)
    stop_tok = c[3].number_input("Stop new calls at tokens", min_value=0,
                                 value=current.stop_tokens or 0)  # fmt: skip
    st.caption(
        "0 = no limit. The stop threshold refuses to *start* new calls once the recorded "
        "estimate reaches it; a call in flight completes and estimates can differ from the "
        "invoice. Set a spending limit in the provider console for a guaranteed ceiling."
    )
    if st.button("Save model settings"):
        try:
            settings = ProjectAISettings(
                tasks=tasks, warn_usd=warn_usd or None, stop_usd=stop_usd or None,
                warn_tokens=int(warn_tok) or None, stop_tokens=int(stop_tok) or None,
            )  # fmt: skip
            with be.repo() as repo:
                version = repo.save_ai_settings(pid, settings)
            st.success(f"Saved as settings version {version}.")
        except ValueError as exc:
            st.error(str(exc))


def _report(report: Any) -> None:
    data = report.__dict__ if hasattr(report, "__dict__") else report
    if getattr(report, "halted", None):
        st.error(f"Run halted: {report.halted}")
    st.json({k: v for k, v in data.items() if v not in (None, [], 0)})


def page_run(be: UIBackend) -> None:
    pid = require_project()
    st.header("Run AI steps (paid API calls)")
    creds = credentials()
    with be.repo() as repo:
        wf = be.workflow(repo, pid, creds)
        if repo.latest_ai_settings(pid) is None:
            st.warning("Save model settings first.")
            return
        for task in AITask:
            problems = wf.preflight(task)
            st.write(("✅ " if not problems else "⛔ ") + TASK_LABELS[task]
                     + (f" — {problems[0]}" if problems else ""))  # fmt: skip
        if note := wf.screening_independence_note():
            st.warning(note)
        totals = repo.llm_usage_totals(pid)
        st.caption(f"Recorded so far: ${totals['cost_usd_estimate']:.4f} (estimate), "
                   f"{totals['tokens']} tokens, {totals['calls']} calls.")  # fmt: skip
        confirm = st.checkbox("I understand these steps make paid API calls and that their "
                              "output is a proposal requiring human review.")  # fmt: skip
        limit = st.number_input("Max records per run (0 = all)", min_value=0, value=10)
        cols = st.columns(2)
        if cols[0].button("Title/abstract screening", disabled=not confirm):
            _report(wf.run_title_abstract_screening(limit=int(limit) or None))
        if cols[1].button("Full-text screening", disabled=not confirm):
            _report(wf.run_full_text_screening(limit=int(limit) or None))
        with_pdf = [i.record_id for i in repo.load_fulltext_items(pid) if i.pdf_sha256]
        chosen = st.multiselect("Records for extraction", with_pdf)
        force = st.checkbox("Force repeat of identical requests (pays again)")
        if st.button("Propose extraction values", disabled=not confirm or not chosen):
            _report(wf.run_extraction(chosen, force=force))
        if st.button("Advisory AI verification of pending values", disabled=not confirm):
            _report(wf.run_verification(force=force))
        rob_record = st.selectbox("Record for a risk-of-bias draft", ["", *with_pdf])
        design = st.selectbox("Study design (human judgement)", [d.value for d in StudyDesign])
        if st.button("Draft risk of bias", disabled=not confirm or not rob_record):
            _report(wf.run_rob_draft(rob_record, StudyDesign(design), force=force))


def page_screening(be: UIBackend) -> None:
    pid = require_project()
    st.header("Screening results and human review queue")
    stage = st.radio("Stage", [s.value for s in ScreeningStage], horizontal=True)
    summary = be.api.get(f"/projects/{pid}/screening/{stage}/summary")
    if summary.ok:
        st.json({k: v for k, v in summary.data.items() if k != "methods_statement"})
        st.caption(summary.data["methods_statement"])
    decisions = be.api.get(f"/projects/{pid}/screening/{stage}/decisions").data or []
    st.dataframe([
        {"record": d["record_id"], "by": "HUMAN" if d["is_human"] else "automated (AI)",
         "role": d["reviewer"], "identity": d["reviewer_identity"], "verdict": d["verdict"],
         "rationale": d["rationale"][:200]}
        for d in decisions
    ])  # fmt: skip
    conflicts = [c for c in be.api.get(f"/projects/{pid}/screening/{stage}/conflicts").data or []
                 if c.get("resolved_verdict") is None]  # fmt: skip
    st.subheader(f"Human review queue ({len(conflicts)} unresolved)")
    st.dataframe(conflicts)
    with st.form("adjudicate"):
        record = st.selectbox("Record", [c["record_id"] for c in conflicts] or [""])
        verdict = st.radio(
            "Human decision",
            [ScreeningVerdict.INCLUDE.value, ScreeningVerdict.EXCLUDE.value],
            horizontal=True,
        )
        rationale = st.text_area("Rationale")
        reason = st.text_input("Exclusion reason (if excluding)")
        if st.form_submit_button("Submit adjudication (authenticated)") and record:
            body = {"verdict": verdict, "rationale": rationale,
                    "exclusion_reason": reason or None}  # fmt: skip
            path = f"/projects/{pid}/screening/{stage}/records/{record}/adjudication"
            show(be.api.post(path, token=token(), json=body), "Adjudication recorded.")


def page_extraction(be: UIBackend) -> None:
    pid = require_project()
    st.header("Full text and extraction")
    manifest = be.api.get(f"/projects/{pid}/fulltext/manifest", params={"only_open": False})
    st.subheader("Full-text manifest")
    st.dataframe(manifest.data if manifest.ok else [])
    with st.form("upload"):
        record = st.text_input("Record ID for PDF upload")
        pdf = st.file_uploader("PDF", type=["pdf"])
        if st.form_submit_button("Upload (authenticated)") and record and pdf is not None:
            show(be.api.request("PUT", f"/projects/{pid}/fulltext/{record}/pdf", token=token(),
                                content=pdf.getvalue(), headers={"X-Filename": pdf.name}),
                 "PDF stored.")  # fmt: skip
    with st.form("confirm"):
        record = st.text_input("Record ID whose PDF identity you confirm")
        note = st.text_input("Note")
        if st.form_submit_button("Confirm PDF identity (authenticated)") and record and note:
            show(be.api.post(f"/projects/{pid}/fulltext/{record}/confirm-match", token=token(),
                             json={"note": note}))  # fmt: skip
    st.subheader("Extraction items")
    items = be.api.get(f"/projects/{pid}/extraction/items").data or []
    advisories: dict[str, list[str]] = {}
    for v in be.api.get(f"/projects/{pid}/extraction/ai-verifications").data or []:
        advisories.setdefault(v["item_id"], []).append(f"{v['verdict']} ({v['model_id']})")
    st.dataframe([
        {"item": i["item_id"], "record": i["record_id"], "field": i["field_name"],
         "outcome": i["outcome_id"], "arm": i["arm_label"], "as printed": i["reported_text"],
         "status": i["data_status"], "review": i["review_status"],
         "proposed by": i["proposer_identity"],
         "evidence check": (i.get("verification") or {}).get("status"),
         "AI advisory (not a review)": "; ".join(advisories.get(i["item_id"], [])),
         "flags": "; ".join(i["flags"])[:200]}
        for i in items
    ])  # fmt: skip
    pending = [i["item_id"] for i in items if i["review_status"] == "pending"]
    with st.form("review_item"):
        item = st.selectbox("Pending item", pending or [""])
        action = st.radio("Action", ["accept", "reject"], horizontal=True)
        note = st.text_input("Note (required to reject)")
        if st.form_submit_button("Submit review (authenticated)") and item:
            body = {"note": note or None} if action == "accept" else {"note": note}
            show(be.api.post(f"/projects/{pid}/extraction/items/{item}/{action}", token=token(),
                             json=body))  # fmt: skip
    with st.form("arms"):
        record = st.text_input("Record ID: build approved arms from accepted items")
        if st.form_submit_button("Build arms (authenticated)") and record:
            show(be.api.post(f"/projects/{pid}/extraction/records/{record}/arms", token=token()))
    st.download_button("Export extraction table (CSV)",
                       be.api.get(f"/projects/{pid}/extraction/export",
                                  params={"format": "csv"}).data or "",
                       file_name="extraction.csv")  # fmt: skip


def page_rob(be: UIBackend) -> None:
    pid = require_project()
    st.header("Risk of bias")
    rows = be.api.get(f"/projects/{pid}/rob").data or []
    for a in rows:
        status = "APPROVED by " + str(a["approved_by"]) if a["approved_by_human"] else "AI draft"
        with st.expander(f"{a['record_id']} · {a['tool']} · {status}"):
            st.json(a)
    drafts = [a["assessment_id"] for a in rows if not a["approved_by_human"]]
    with st.form("rob_approve"):
        aid = st.selectbox("Draft to approve", drafts or [""])
        overall = st.text_input("Overall judgement (human)")
        note = st.text_input("Note")
        if st.form_submit_button("Approve (authenticated)") and aid:
            show(be.api.post(f"/projects/{pid}/rob/{aid}/approve", token=token(),
                             json={"overall": overall or None, "note": note or None}))  # fmt: skip


def page_stats(be: UIBackend) -> None:
    pid = require_project()
    st.header("Statistics (deterministic engine; no LLM)")
    plan = be.api.get(f"/projects/{pid}/stats/plan")
    st.subheader("Approved analysis plan")
    st.json(plan.data if plan.ok else {"status": plan.error})
    if st.button("Run Stage 5 meta-analysis"):
        with be.repo() as repo:
            try:
                artifacts = be.workflow(repo, pid, credentials()).run_statistics(be.stats_dir(pid))
                st.success(f"stats_output.json sha256 {artifacts.sha256}")
            except Exception as exc:
                st.error(str(exc))
    output = be.api.get(f"/projects/{pid}/stats/output")
    if output.ok:
        if output.data["problems"]:
            st.error("Not a valid basis for Stage 6: " + "; ".join(output.data["problems"]))
        for analysis in output.data["stats_output"].get("analyses", []):
            with st.expander(f"Analysis {analysis.get('analysis_id')}"):
                st.json(analysis)
    plots = sorted((be.stats_dir(pid) / "plots").glob("*.svg"))
    for svg in plots:
        st.markdown(f"**Forest plot: {svg.stem}**")
        encoded = base64.b64encode(svg.read_bytes()).decode("ascii")
        st.markdown(f'<img src="data:image/svg+xml;base64,{encoded}" style="max-width:100%">',
                    unsafe_allow_html=True)  # fmt: skip


def page_usage(be: UIBackend) -> None:
    pid = require_project()
    st.header("API usage and estimated costs")
    st.info("Costs are list-price estimates from reported tokens; the provider invoice is "
            "authoritative. " + BILLING_NOTE)  # fmt: skip
    usage = be.api.get(f"/projects/{pid}/ai/usage")
    if not usage.ok:
        st.error(usage.error)
        return
    totals = usage.data["totals"]
    c = st.columns(4)
    c[0].metric("Estimated cost (USD)", f"{totals['cost_usd_estimate']:.4f}")
    c[1].metric("Tokens", totals["tokens"])
    c[2].metric("Calls", totals["calls"])
    c[3].metric("Failed calls", totals["failed_calls"])
    if totals["unpriced_successful_calls"]:
        st.warning(f"{totals['unpriced_successful_calls']} call(s) had no list price: "
                   "the estimate is understated.")  # fmt: skip
    st.dataframe([
        {k: call.get(k) for k in ("created_at", "task", "provider", "requested_model",
                                  "served_model", "effort_requested", "effort_effective",
                                  "status", "input_tokens", "output_tokens", "reasoning_tokens",
                                  "cache_read_tokens", "cost_usd_estimate", "attempts",
                                  "error_message")}
        for call in usage.data["calls"]
    ])  # fmt: skip


def main() -> None:
    st.set_page_config(page_title="SysRev reviewer", layout="wide")
    be = backend()
    with st.sidebar:
        st.title("SysRev")
        st.text_input("Reviewer token (for approvals)", type="password", key="reviewer_token",
                      help="Issued once by 'python -m src.api_server.reviewer_cli enroll'. "
                      "Kept in this browser session only.")  # fmt: skip
        page = st.radio("Page", PAGES, key="page")
        if pid := project_id():
            st.caption(f"Project {pid[:8]}")
        st.caption("AI output is always a proposal. Only an authenticated human can approve.")
    handlers = {
        "Projects": page_projects, "Protocol": page_protocol,
        "Searches & import": page_searches, "Pipeline status": page_pipeline,
        "AI models & keys": page_models, "Run AI steps": page_run,
        "Screening": page_screening, "Full text & extraction": page_extraction,
        "Risk of bias": page_rob, "Statistics": page_stats, "Usage & costs": page_usage,
    }  # fmt: skip
    handlers[str(page)](be)


main()
