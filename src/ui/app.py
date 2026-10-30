"""SysRev local reviewer UI (Streamlit). Start with ``python -m src.ui``.

A thin client: human decisions (gates, stage advances, adjudication, PDF upload and
identity confirmation, extraction accept/reject, arms, risk-of-bias approval) are API
calls authenticated with the reviewer's bearer token. AI steps run the existing runners
in-process and only ever create proposals plus an audit trail of the calls. No API key
or reviewer token is written to the database, a file or a log.
"""

from __future__ import annotations

import base64
from typing import Any

import streamlit as st

from src.core.enums import GateVerdict, ScreeningStage, ScreeningVerdict, StudyDesign
from src.llm.credentials import BILLING_NOTE, ENV_VARS, CredentialResolver, install_log_redaction
from src.llm.tasks import TASK_LABELS, AITask, ProjectAISettings, TaskModelSettings
from src.ui.services import ApiResult, UIBackend

PROVIDER_DEFAULT = "(provider default — not sent)"
NOT_CONFIGURED = "— choose a model —"
PAGES = (
    "Projects", "Pipeline status", "AI models & keys", "Run AI steps", "Screening",
    "Full text & extraction", "Risk of bias", "Statistics", "Usage & costs",
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
        "Protocol drafting and literature searches are run from Python (README, Stage 1–2); "
        "this interface does not yet provide them."
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
        "Projects": page_projects, "Pipeline status": page_pipeline,
        "AI models & keys": page_models, "Run AI steps": page_run,
        "Screening": page_screening, "Full text & extraction": page_extraction,
        "Risk of bias": page_rob, "Statistics": page_stats, "Usage & costs": page_usage,
    }  # fmt: skip
    handlers[str(page)](be)


main()
