"""Session 7 pages of the local reviewer UI: PRISMA flow diagram and checklists, GRADE and
Summary of Findings, synthesis without meta-analysis and SWiM.

Every decision is sent to the authenticated reporting API with the reviewer's token; the UI
never records a decision itself, never pre-fills a GRADE judgement and never offers to mark
a whole checklist as reported. Read views use the same API.

**Screens are frozen.** Streamlit reruns this script when a form is submitted, so anything
fetched during that rerun is state the reviewer has *not* seen. Each decision screen
(a checklist, one GRADE analysis, the synthesis groups, one synthesis preview, the SWiM
decision) is therefore kept in ``st.session_state`` under a key scoped by project and by
checklist, analysis, synthesis group or screen. It is rendered from that snapshot, and a
decision sends only the snapshot's state identifiers; the API refuses them if the stored
state changed. The snapshot is replaced only by an explicit "Reload the current state" or
after the reviewer's own successful decision (followed by a rerun, so the display and the
next submission agree). A live comparison warns when the stored state has moved on.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from typing import Any

import streamlit as st

from src.core.schemas.reporting import GRADE_DOMAINS
from src.ui.services import ApiResult, UIBackend

Show = Callable[..., bool]
SECTIONS = (
    "title",
    "abstract",
    "introduction",
    "methods",
    "results",
    "discussion",
    "limitations",
    "conclusions",
    "tables",
    "figures",
    "supplementary",
)
CHECKLIST_LABEL = {
    "prisma_2020_main": "PRISMA 2020 (main checklist)",
    "prisma_2020_abstract": "PRISMA 2020 for Abstracts",
    "swim": "SWiM",
}


RELOAD = "Reload the current state (review it before deciding)"
CHANGED = (
    "The stored state changed after this screen was loaded. A decision from this screen will "
    "be refused: reload, review the updated state, then decide."
)


def _key(pid: str, *scope: str) -> str:
    return ":".join(("screen", pid, *scope))


def _frozen(key: str, fetch: Callable[[], ApiResult]) -> tuple[Any, str | None]:
    """The screen the reviewer is looking at: fetched once, then kept until an explicit
    reload (never refreshed by the rerun that processes a submission)."""
    if key not in st.session_state:
        res = fetch()
        if not res.ok:
            return None, res.error
        st.session_state[key] = res.data
    return st.session_state[key], None


def _reload_button(key: str) -> None:
    if st.button(RELOAD, key=f"reload:{key}"):
        st.session_state.pop(key, None)
        st.rerun()


def _accepted(key: str, message: str) -> None:
    """After the reviewer's own successful decision: show the new state (rerun) so the next
    submission is made from what is displayed."""
    st.session_state.pop(key, None)
    st.session_state["reporting_flash"] = message
    st.rerun()


def _flash() -> None:
    message = st.session_state.pop("reporting_flash", None)
    if message:
        st.success(message)


def _submit(result: ApiResult, show: Show, key: str, message: str) -> None:
    if result.ok:
        _accepted(key, message)
    elif result.status == 409 and "has changed" in (result.error or ""):
        st.error(f"Refused ({result.status}): {result.error}")
        st.warning("Nothing was saved. Use the reload button to review the updated state.")
    else:
        show(result)


def _blockers(be: UIBackend, pid: str) -> None:
    res = be.api.get(f"/projects/{pid}/reporting/status")
    if not res.ok:
        st.error(res.error)
        return
    data = res.data
    c = st.columns(3)
    c[0].metric("Blocking reporting items", len(data["blockers"]))
    c[1].metric("Items requiring human review", len(data["requires_human_review"]))
    c[2].metric("PRISMA diagram", data["flow"]["status"])
    for w in data["flow"]["warnings"]:
        st.error(f"PRISMA flow diagram warning: {w}")
    with st.expander("What still blocks the audit (reporting scope)"):
        st.dataframe(data["blockers"] or [{"status": "none"}])
    st.caption(data["note"])


def _checklist_ids(data: dict[str, Any]) -> tuple[Any, ...]:
    return (
        data["manuscript_version"],
        data["manuscript_sha256"],
        data["evidence_sha256"],
        tuple((r["item_id"], (r["decision"] or {}).get("decision_id")) for r in data["items"]),
    )


def checklist_panel(be: UIBackend, pid: str, name: str, token: str | None, show: Show) -> None:
    key = _key(pid, "checklist", name)
    url = f"/projects/{pid}/reporting/checklists/{name}"
    data, error = _frozen(key, lambda: be.api.get(url))
    if data is None:
        st.info(error)
        return
    live = be.api.get(url)
    if live.ok and _checklist_ids(live.data) != _checklist_ids(data):
        st.warning(CHANGED)
    _reload_button(key)
    st.caption(
        f"{data['standard']} · manuscript version {data['manuscript_version']} · {data['note']}"
    )
    st.write(data["summary"])
    st.dataframe(
        [
            {
                "item": r["item_id"],
                "topic": r["topic"],
                "status": r["status"],
                "decided": r["decided"],
                "stale": r["stale"],
                "candidate locations (suggestions only)": "; ".join(r["candidates"]),
                "requirement": r["requirement"],
            }
            for r in data["items"]
        ]
    )
    ids = [r["item_id"] for r in data["items"]]
    with st.form(f"decide_{name}"):
        st.subheader("Record a decision on one item")
        item_id = st.selectbox("Item", ids)
        row = next(r for r in data["items"] if r["item_id"] == item_id)
        st.caption(row["requirement"])
        status = st.selectbox(
            "Your decision", ["reported", "partially_reported", "missing", "not_applicable"]
        )
        sections = st.multiselect("Section(s) where it is reported", SECTIONS)
        location = st.text_input("Precise location (e.g. 'Methods, Information sources')")
        justification = st.text_area("Justification (required)")
        if st.form_submit_button("Record decision (authenticated)"):
            _submit(
                be.api.post(
                    f"{url}/items/{item_id}",
                    token=token,
                    json={
                        "status": status,
                        "section_ids": sections,
                        "location": location,
                        "justification": justification,
                        # identifiers of the displayed (frozen) screen, never refetched
                        "based_on_decision_id": (row["decision"] or {}).get("decision_id"),
                        "seen": {
                            k: data[k]
                            for k in ("manuscript_version", "manuscript_sha256", "evidence_sha256")
                        },
                    },
                ),
                show,
                key,
                f"Decision on item {item_id} recorded.",
            )
    cols = st.columns(3)
    for col, fmt in zip(cols, ("csv", "docx", "xlsx"), strict=True):
        exp = be.api.get(f"{url}/export", params={"format": fmt})
        if exp.ok:
            col.download_button(f"Download {fmt.upper()}", _bytes(exp), file_name=f"{name}.{fmt}")


def _bytes(res: ApiResult) -> bytes:
    data = res.data
    if isinstance(data, str):
        return data.encode("utf-8")
    if isinstance(data, dict | list):  # JSON exports (e.g. the diagram source)
        return (json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()
    return bytes(data or b"")


def page_prisma(be: UIBackend, pid: str, token: str | None, show: Show) -> None:
    st.header("PRISMA 2020 flow diagram and checklists")
    st.caption(
        "The diagram is generated from the stored PRISMA snapshot; counts are never "
        "typed in or repaired. Checklist suggestions only point to candidate locations: "
        "every item needs your decision."
    )
    _flash()
    _blockers(be, pid)
    variant = st.selectbox(
        "Template variant",
        [
            "new_review_databases_registers",
            "new_review_databases_registers_other_methods",
            "updated_review_databases_registers",
            "updated_review_databases_registers_other_methods",
        ],
    )
    svg = be.api.get(f"/projects/{pid}/reporting/prisma/diagram", params={"variant": variant})
    if svg.ok:
        encoded = base64.b64encode(_bytes(svg)).decode("ascii")
        st.markdown(
            f'<img src="data:image/svg+xml;base64,{encoded}" style="max-width:100%">',
            unsafe_allow_html=True,
        )
        cols = st.columns(3)
        cols[0].download_button("Download SVG", _bytes(svg), file_name="prisma_flow.svg")
        for col, fmt in ((cols[1], "png"), (cols[2], "json")):
            res = be.api.get(
                f"/projects/{pid}/reporting/prisma/diagram",
                params={"format": fmt, "variant": variant},
            )
            if res.ok:
                col.download_button(
                    f"Download {fmt.upper()}", _bytes(res), file_name=f"prisma_flow.{fmt}"
                )
            else:
                col.caption(res.error or "")
    else:
        st.error(f"This variant cannot be drawn from the project's data: {svg.error}")
    for name in ("prisma_2020_main", "prisma_2020_abstract"):
        with st.expander(CHECKLIST_LABEL[name], expanded=name == "prisma_2020_main"):
            checklist_panel(be, pid, name, token, show)


def page_grade(be: UIBackend, pid: str, token: str | None, show: Show) -> None:
    st.header("GRADE certainty of evidence and Summary of Findings")
    st.caption(
        "Statistical results are shown as decision support only. Every domain judgement, "
        "rationale and the final certainty are yours; a statistically significant result "
        "never raises certainty, and a missing or non-significant test is not evidence "
        "against publication bias."
    )
    _flash()
    res = be.api.get(f"/projects/{pid}/reporting/grade")
    if not res.ok:
        st.error(res.error)
        return
    rows = res.data
    st.caption("Current status of every analysis (live overview):")
    st.dataframe(
        [
            {k: r[k] for k in ("analysis_id", "outcome", "question_type", "status", "stale")}
            for r in rows
        ]
    )
    if not rows:
        return
    aid = st.selectbox("Outcome analysis", [r["analysis_id"] for r in rows])
    live = next(r for r in rows if r["analysis_id"] == aid)
    key = _key(pid, "grade", aid)
    # the analysis screen the reviewer works on (frozen; reload to review a newer state)
    row = st.session_state.setdefault(key, live)
    if (row["inputs_sha256"], row["based_on_assessment_id"]) != (
        live["inputs_sha256"],
        live["based_on_assessment_id"],
    ):
        st.warning(CHANGED)
    _reload_button(key)
    st.caption(f"Record displayed: {row['based_on_assessment_id'] or 'none'}")
    if row["stale"]:
        st.error("The current record was made on inputs that have changed: re-assess or withdraw.")
    for p in row["problems"]:
        st.error(p)
    with st.expander("Decision support (does not determine any judgement)"):
        st.json(row["decision_support"])
    approaches = row["applicable_approaches"]
    if not approaches:
        st.warning(
            "No GRADE approach applies to this analysis as recorded (for example a "
            "single-group estimate without a documented prognosis rationale, or no "
            "estimate). You may record it as not assessed with a reason."
        )
    with st.form(f"grade_{aid}"):
        status = st.selectbox("Record as", ["approved", "draft", "not_assessed"])
        approach = st.selectbox("GRADE approach", ["(none)", *approaches])
        framework = st.text_area("Why this approach fits the question")
        initial = st.selectbox(
            "Initial certainty (prognosis framework only; fixed otherwise)",
            ["(fixed by approach)", "high", "moderate", "low", "very_low"],
        )
        initial_why = st.text_input("Justification of the initial level (prognosis framework)")
        domains = []
        for d in GRADE_DOMAINS:
            c = st.columns([1, 3])
            options = ["not_serious", "serious", "very_serious"]
            if d == "risk_of_bias":  # three levels: ROBINS-I approach only (checked by the API)
                options.append("extremely_serious")
            concern = c[0].selectbox(d.replace("_", " "), options, key=f"{aid}_{d}")
            why = c[1].text_input(f"Rationale: {d.replace('_', ' ')}", key=f"{aid}_{d}_why")
            domains.append({"domain": d, "concern": concern, "rationale": why})
        upgrades = []
        for f in ("large_effect", "dose_response", "plausible_confounding"):
            c = st.columns([1, 3])
            levels = c[0].selectbox(
                f"Rate up: {f.replace('_', ' ')}",
                [0, 1, 2] if f == "large_effect" else [0, 1],
                key=f"{aid}_{f}",
            )
            why = c[1].text_input(f"Rationale: {f.replace('_', ' ')}", key=f"{aid}_{f}_why")
            if levels:
                upgrades.append({"factor": f, "levels": levels, "rationale": why})
        final = st.selectbox("Final certainty", ["high", "moderate", "low", "very_low"])
        departure = st.text_input("If it differs from the level implied by the domains: why")
        summary = st.text_area("Overall explanation")
        st.markdown("**Assumed comparator risk for absolute effects (optional, RR/OR only)**")
        b_label = st.text_input("Label (e.g. 'Study population')")
        b_risk = st.number_input("Risk (0-1)", min_value=0.0, max_value=0.999, value=0.0)
        b_source = st.text_input("Source of this baseline risk (required if given)")
        reason = st.selectbox(
            "If not assessed: reason",
            ["framework_not_applicable", "not_performed", "insufficient_information"],
        )
        note = st.text_input("If not assessed: explanation")
        if st.form_submit_button("Save (authenticated)"):
            body: dict[str, Any] = {
                "status": status,
                # the decision support and record this screen displayed
                "inputs_sha256": row["inputs_sha256"],
                "based_on_assessment_id": row["based_on_assessment_id"],
            }
            if status == "not_assessed":
                body |= {"not_assessed_reason": reason, "status_note": note}
            else:
                body |= {
                    "approach": None if approach == "(none)" else approach,
                    "framework_rationale": framework,
                    "domains": domains,
                    "upgrades": upgrades,
                    "final_certainty": final,
                    "departure_rationale": departure or None,
                    "summary": summary,
                    "initial_rationale": initial_why,
                }
                if initial != "(fixed by approach)":
                    body["initial_certainty"] = initial
                if b_risk > 0:
                    body["baseline_risks"] = [
                        {
                            "label": b_label or "Assumed comparator risk",
                            "risk": b_risk,
                            "source": b_source,
                        }
                    ]
            _submit(
                be.api.post(f"/projects/{pid}/reporting/grade/{aid}", token=token, json=body),
                show,
                key,
                f"GRADE record for {aid} saved.",
            )
    with st.form(f"withdraw_{aid}"):
        why = st.text_input("Reason for withdrawing the current record")
        if st.form_submit_button("Withdraw (authenticated)"):
            _submit(
                be.api.post(
                    f"/projects/{pid}/reporting/grade/{aid}/withdraw",
                    token=token,
                    json={"reason": why, "based_on_assessment_id": row["based_on_assessment_id"]},
                ),
                show,
                key,
                f"GRADE record for {aid} withdrawn.",
            )
    st.subheader("Summary of Findings and evidence summaries (preview)")
    sof = be.api.get(f"/projects/{pid}/reporting/sof")
    if sof.ok:
        st.markdown(sof.data)
        st.download_button("Download Markdown", _bytes(sof), file_name="summary_of_findings.md")
        docx = be.api.get(f"/projects/{pid}/reporting/sof", params={"format": "docx"})
        if docx.ok:
            st.download_button("Download DOCX", _bytes(docx), file_name="summary_of_findings.docx")


def page_synthesis(be: UIBackend, pid: str, token: str | None, show: Show) -> None:
    st.header("Synthesis without meta-analysis and SWiM")
    st.caption(
        "Groups, study-level findings, directions and the narrative are yours. Numbers "
        "must come from accepted extraction items or approved arms; pooled estimates, "
        "significance-based vote counts and unsupported directional claims are refused."
    )
    _flash()
    _synthesis_groups(be, pid, token, show)
    with be.repo() as repo:
        protocol = repo.latest_protocol(pid)
        arms = [a for a in repo.load_extracted_arms(pid) if a.approved_by_human]
    outcomes = [o.outcome_id for o in protocol.outcomes] if protocol else []
    with st.form("synthesis_group"):
        gid = st.text_input("Group id (letters, digits, -_.)")
        outcome = st.selectbox("Outcome", outcomes or [""])
        title = st.text_input("Title")
        basis = st.multiselect(
            "Grouped by",
            ["population", "intervention", "comparator", "outcome", "study_design", "timepoint"],
        )
        description = st.text_area("Description of the group (SWiM 1a)")
        rationale = st.text_area("Rationale for the grouping (SWiM 1a)")
        change = st.checkbox("The grouping changed from the protocol (SWiM 1b)")
        change_why = st.text_input("Rationale for the change")
        reason = st.selectbox(
            "Why not meta-analysed",
            [
                "heterogeneous_designs",
                "incompatible_measures",
                "insufficient_numerical_data",
                "qualitative_findings_only",
                "single_study",
                "other",
            ],
        )
        reason_note = st.text_input("Explanation")
        metric = st.text_input("Standardised metric (SWiM 2)")
        method = st.selectbox(
            "Synthesis method (SWiM 3)",
            ["structured_tabulation", "vote_counting_direction_of_effect"],
        )
        method_why = st.text_area("Justification of the method")
        prioritisation = st.text_input("Prioritisation criteria (SWiM 4, optional)")
        heterogeneity = st.text_input("Investigation of heterogeneity (SWiM 5, optional)")
        certainty = st.text_input("Certainty assessment of the synthesis (SWiM 6)")
        presentation = st.text_input("Presentation and ordering of studies (SWiM 7)")
        limitations = st.text_area("Limitations of the synthesis (SWiM 9)")
        n = st.number_input("Number of study-level findings", min_value=1, max_value=8, value=1)
        findings = []
        for i in range(int(n)):
            st.markdown(f"**Finding {i + 1}**")
            choices = {
                f"{a.record_id} · {a.outcome_id} · {a.arm_label} ({a.arm_id[:8]})": a for a in arms
            }
            picked = st.multiselect("Approved arms (source)", list(choices), key=f"arms{i}")
            text = st.text_area("Finding (numbers only as extracted)", key=f"text{i}")
            direction = st.selectbox(
                "Direction of effect",
                [
                    "not_applicable",
                    "favours_intervention",
                    "favours_comparator",
                    "point_estimate_at_null",
                    "mixed_or_conflicting",
                    "direction_not_determinable",
                ],
                key=f"dir{i}",
                help="The observed direction of the effect estimate, irrespective of statistical "
                "significance or the confidence interval. Results at the null, mixed or "
                "undeterminable directions are tabulated and never counted as votes.",
            )
            why = st.text_input(
                "Basis for the direction (observed direction, not significance)", key=f"why{i}"
            )
            unit = st.checkbox(
                "This result represents its study/cohort (counted)", value=True, key=f"unit{i}"
            )
            sel = [choices[k] for k in picked]
            if sel:
                findings.append(
                    {
                        "record_id": sel[0].record_id,
                        "arm_ids": [a.arm_id for a in sel],
                        "finding": text,
                        "direction": direction,
                        "direction_basis": why or None,
                        "selected_for_unit": unit,
                    }
                )
        selection_rule = st.text_input(
            "Selection rule when a study/cohort has several reports or results (required then)"
        )
        summary = st.text_area("Narrative summary (SWiM 8)")
        status = st.selectbox("Record as", ["draft", "approved"])
        if st.form_submit_button("Preview the linked evidence (step 1 of 2)"):
            draft = {
                "group_id": gid,
                "outcome_id": outcome,
                "status": status,
                "title": title,
                "grouping_basis": basis,
                "description": description,
                "rationale": rationale,
                "protocol_change": change,
                "protocol_change_rationale": change_why or None,
                "reason_not_pooled": reason,
                "reason_note": reason_note,
                "standardised_metric": metric,
                "synthesis_method": method,
                "method_justification": method_why,
                "prioritisation_criteria": prioritisation or None,
                "heterogeneity_investigation": heterogeneity or None,
                "certainty_method": certainty,
                "presentation": presentation,
                "limitations": limitations,
                "findings": findings,
                "selection_rule": selection_rule or None,
                "summary": summary,
            }
            # step 1: freeze the preview response the reviewer is about to read
            preview = be.api.post(f"/projects/{pid}/reporting/synthesis/preview", json=draft)
            gkey = _key(pid, "synthesis", str(draft["group_id"] or "(no id)"))
            st.session_state[gkey] = {
                "draft": draft,
                "preview": preview.data if preview.ok else None,
                "error": None if preview.ok else preview.error,
            }
            st.session_state[_key(pid, "synthesis", "active")] = gkey
    _synthesis_preview(be, pid, token, show)
    st.subheader("SWiM reporting guideline")
    key = _key(pid, "swim")
    data, error = _frozen(key, lambda: be.api.get(f"/projects/{pid}/reporting/swim"))
    if data is None:
        st.error(error)
        return
    live = be.api.get(f"/projects/{pid}/reporting/swim")
    if live.ok and (live.data["synthesis_sha256"], live.data["based_on_decision_id"]) != (
        data["synthesis_sha256"],
        data["based_on_decision_id"],
    ):
        st.warning(CHANGED)
    _reload_button(key)
    if not data["required"]:
        st.info("No synthesis without meta-analysis is approved: SWiM is not triggered.")
        return
    st.caption(f"Suggestion (you decide): {data['suggestion']}")
    if data["stale"]:
        st.warning("The syntheses changed after the applicability decision: decide again.")
    st.json(data["decision"] or {"decision": "none yet"})
    with st.form("swim_applicability"):
        choice = st.selectbox(
            "SWiM applies to these syntheses",
            ["applicable", "partially_applicable", "not_applicable"],
        )
        why = st.text_area("Rationale")
        if st.form_submit_button("Record (authenticated)"):
            _submit(
                be.api.post(
                    f"/projects/{pid}/reporting/swim",
                    token=token,
                    json={
                        "applicability": choice,
                        "rationale": why,
                        # the syntheses and decision this (frozen) screen displayed
                        "synthesis_sha256": data["synthesis_sha256"],
                        "based_on_decision_id": data["based_on_decision_id"],
                    },
                ),
                show,
                key,
                "SWiM applicability recorded.",
            )
    if data["checklist_required"]:
        checklist_panel(be, pid, "swim", token, show)


def _synthesis_groups(be: UIBackend, pid: str, token: str | None, show: Show) -> None:
    """The recorded groups (frozen screen) and their withdrawal."""
    key = _key(pid, "synthesis", "groups")
    url = f"/projects/{pid}/reporting/synthesis"
    groups, error = _frozen(key, lambda: be.api.get(url))
    if groups is None:
        st.error(error)
        return
    live = be.api.get(url)
    if live.ok and [(g["group_id"], g["revision_id"]) for g in live.data] != [
        (g["group_id"], g["revision_id"]) for g in groups
    ]:
        st.warning(CHANGED)
    _reload_button(key)
    for g in groups:
        with st.expander(f"{g['group_id']} · {g['status']} · revision {g['revision_id'][:8]}"):
            for prob in g["problems"]:
                st.error(prob)
            st.json(g["record"])
    withdrawable = [g for g in groups if g["status"] != "withdrawn"]
    if withdrawable:
        with st.form("synthesis_withdraw"):
            gid = st.selectbox("Group to withdraw", [g["group_id"] for g in withdrawable])
            why = st.text_input("Reason for withdrawing the group")
            if st.form_submit_button("Withdraw group (authenticated)"):
                shown = next(g for g in withdrawable if g["group_id"] == gid)
                _submit(
                    be.api.post(
                        f"{url}/{gid}/withdraw",
                        token=token,
                        json={"reason": why, "based_on_revision_id": shown["revision_id"]},
                    ),
                    show,
                    key,
                    f"Synthesis group {gid} withdrawn.",
                )


def _synthesis_preview(be: UIBackend, pid: str, token: str | None, show: Show) -> None:
    """Step 2 shows the *frozen* preview (step 1's response) and submits its identifiers;
    nothing is previewed again during the submission."""
    active = st.session_state.get(_key(pid, "synthesis", "active"))
    gkey = str(active) if active else ""
    frozen = st.session_state.get(gkey) if gkey else None
    if not frozen:
        return
    draft, preview = frozen["draft"], frozen["preview"]
    st.markdown(f"**Preview of group {draft['group_id'] or '(no id)'} (frozen at step 1)**")
    if st.button("Discard this preview", key=f"discard:{gkey}"):
        st.session_state.pop(gkey, None)
        st.session_state.pop(_key(pid, "synthesis", "active"), None)
        st.rerun()
    if preview is None:
        st.error(frozen["error"])
        return
    st.json(preview["evidence"])
    st.caption(
        f"Independent study units: {preview['voting_units']} · group revision displayed: "
        f"{preview['based_on_revision_id'] or 'none (new group)'}"
    )
    for prob in preview["problems"]:
        st.error(prob)
    if st.button("Save the previewed group (step 2 of 2, authenticated)"):
        result = be.api.post(
            f"/projects/{pid}/reporting/synthesis",
            token=token,
            json=draft
            | {
                "inputs_sha256": preview["inputs_sha256"],
                "based_on_revision_id": preview["based_on_revision_id"],
            },
        )
        if result.ok:
            st.session_state.pop(_key(pid, "synthesis", "active"), None)
            st.session_state.pop(_key(pid, "synthesis", "groups"), None)
            st.session_state.pop(_key(pid, "swim"), None)
            _accepted(gkey, f"Synthesis group {draft['group_id']} saved.")
        else:
            _submit(result, show, gkey, "")
            if result.status == 409:
                st.warning("Discard this preview and preview again to review the updated state.")
