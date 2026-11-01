"""Session 8 pages of the local reviewer UI.

* **Model catalogue (Refresh Models)** — list models through the providers' official list
  endpoints (no inference), preview the candidate catalogue, verify a model's capabilities and
  prices with their sources, and activate a snapshot (authenticated).
* **Corrections, study links & double counting** — correct extracted values with source
  evidence (pending or already reviewed; history shown), link/unlink reports of one study, and
  resolve double counting of a cohort.
* **Analysis plan** — create, revise and approve analysis plans.
* **Protocol amendments** — propose, verify, approve/reject/withdraw amendments after Gate 1 and
  re-review the work they affect.

Every decision is an authenticated API call that carries the identifiers of the frozen screen
the reviewer saw (:mod:`src.ui.frozen`); the API refuses it if the state changed. Refreshing the
model list runs in this process with the reviewer's API keys (never sent over HTTP).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from typing import Any

import streamlit as st

from src.core.enums import ComparisonType, EffectMeasure, StudyDesign
from src.core.schemas.protocol import ALLOWED_MEASURES
from src.llm.credentials import CredentialResolver
from src.ui import frozen as fz
from src.ui.services import UIBackend

Show = Callable[..., bool]
EFFORT_VOCABULARY = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]
CORRECTION_COPY = (
    "domain",
    "field_name",
    "value_kind",
    "outcome_id",
    "arm_label",
    "arm_role",
    "timepoint",
    "follow_up",
    "estimate_type",
    "estimate_scale",
    "ci_level",
    "unit",
)
DATA_STATUSES = ["reported", "not_reported", "ambiguous", "contradictory", "inaccessible"]
LOCATION_TYPES = ["table", "figure", "paragraph", "supplement"]


# =========================================================================== models
def page_catalogue(
    be: UIBackend, pid: str, token: str | None, show: Show, creds: CredentialResolver
) -> None:
    st.header("Model catalogue — Refresh Models")
    st.info(
        "Refreshing calls only the providers' official model-list endpoints (GET /v1/models) "
        "with your API keys: no inference, nothing billed as a model call. A listed model is "
        "never executable by itself — its capabilities must be verified against official "
        "documentation and the catalogue snapshot activated by an enrolled reviewer. Unknown "
        "prices are shown as unavailable, never as zero."
    )
    fz.flash()
    ck = fz.key(fz.CATALOGUE, pid)
    data, error = fz.frozen(ck, lambda: be.api.get("/models/catalogue"))
    if data is None:
        st.error(error)
        return
    live = be.api.get("/models/catalogue")
    if live.ok:
        fz.warn_if_changed(live.data["active_sha256"], data["active_sha256"])
    fz.reload_button(ck)
    st.caption(
        f"Active catalogue {data['active_sha256'][:12]} (snapshot "
        f"{data['active_snapshot_id']}); {len(data['activations'])} activation(s) recorded."
    )
    rows = data["models"]
    st.dataframe(
        [
            {
                "provider": r["provider"],
                "model": r["model_id"],
                "executable": r["executable"],
                "account availability (last listing)": r["account_availability"],
                "adapter compatibility": r["adapter_compatibility"],
                "capabilities": r["capabilities"][:90],
                "efforts": ", ".join(r["efforts"]),
                "pricing": r["pricing"][:90],
                "lifecycle": r["lifecycle"],
                "why not executable": "; ".join(r["problems"])[:200],
            }
            for r in rows
        ]
    )
    for provider, d in data["discovery"].items():
        if d["status"] != "ok":
            st.warning(f"Last {provider} listing failed ({d['status']}): {d['error']}")

    st.subheader("1 · Refresh (list models; creates a candidate snapshot, not active)")
    if st.button("Refresh models (no paid inference)", key=fz.wkey("refresh", ck)):
        from src.llm.discovery import ModelLister
        from src.workflows.models import refresh_catalogue

        with be.repo() as repo:
            snapshot, changes = refresh_catalogue(repo, be.base_catalogue, ModelLister(creds))
        st.session_state[fz.FLASH] = (
            f"Candidate snapshot {snapshot.snapshot_id[:12]} stored: "
            f"{len(changes['added'])} newly listed (unverified), "
            f"{len(changes['changed'])} changed. Review it below before activating."
        )
        fz.put(fz.key(fz.SNAPSHOT_ACTIVE, pid), snapshot.snapshot_id)
        fz.drop(fz.key(fz.SNAPSHOT, pid, snapshot.snapshot_id))
        st.rerun()

    st.subheader("2 · Preview and activate a candidate snapshot")
    snaps = be.api.get("/models/snapshots").data or []
    if not snaps:
        st.caption("No snapshots stored yet.")
    else:
        # every snapshot stays selectable (the active one is marked), so a frozen preview is
        # still on screen when its submission is processed — the API then refuses it if stale
        labels = {
            f"{x['created_at'][:19]} · {x['kind']} · {x['snapshot_id'][:12]}"
            + (" · ACTIVE" if x["active"] else ""): x["snapshot_id"]
            for x in reversed(snaps)
        }
        selected = fz.store().get(fz.key(fz.SNAPSHOT_ACTIVE, pid))
        options = list(labels)
        index = next((i for i, v in enumerate(labels.values()) if v == selected), 0)
        choice = st.selectbox("Snapshot", options, index=index, key=fz.wkey("snap", ck))
        _snapshot_preview(be, pid, labels[choice], token, show)

    st.subheader("3 · Verify a model (capabilities and prices with their sources)")
    _verification_form(be, pid, data, token, show)


def _snapshot_preview(
    be: UIBackend, pid: str, snapshot_id: str, token: str | None, show: Show
) -> None:
    sk = fz.key(fz.SNAPSHOT, pid, snapshot_id)
    view, error = fz.frozen(sk, lambda: be.api.get(f"/models/snapshots/{snapshot_id}/preview"))
    if view is None:
        st.error(error)
        return
    fz.reload_button(sk, fz.key(fz.CATALOGUE, pid))
    ch = view["changes"]
    if not view["based_on_active"]:
        st.warning(
            "This snapshot was built on a different catalogue than the active one; it "
            "cannot be activated. Refresh or verify again."
        )
    st.write(
        {
            "added (unverified)": ch["added"],
            "changed": list(ch["changed"]),
            "newly executable": ch["newly_executable"],
            "newly blocked": ch["newly_blocked"],
            "removed": ch["removed"],
        }
    )
    with st.expander("Field-level changes"):
        st.json(ch["changed"])
    with st.form(fz.wkey("activate", sk)):
        note = st.text_input("Activation note (optional)")
        if st.form_submit_button("Activate this snapshot (authenticated)"):
            fz.submit(
                be.api.post(
                    f"/models/snapshots/{snapshot_id}/activate",
                    token=token,
                    json={
                        "expected_sha256": view["sha256"],
                        "expected_active_sha256": view["active_sha256"],
                        "note": note or None,
                    },
                ),
                show,
                [sk, fz.key(fz.CATALOGUE, pid)],
                f"Snapshot {snapshot_id[:12]} activated.",
            )


def _verification_form(
    be: UIBackend, pid: str, data: dict[str, Any], token: str | None, show: Show
) -> None:
    ck = fz.key(fz.CATALOGUE, pid)
    choices = [f"{r['provider']}/{r['model_id']}" for r in data["models"]]
    if not choices:
        return
    unverified = [
        c for c, r in zip(choices, data["models"], strict=True) if r["capabilities"] == "UNVERIFIED"
    ]
    target = st.selectbox(
        "Model",
        unverified + [c for c in choices if c not in unverified],
        key=fz.wkey("verify_model", ck),
    )
    row = data["models"][choices.index(target)]
    reported = row["provider_reported"]
    if reported:
        st.caption(f"Provider-reported metadata (a cross-check, not proof): {reported}")
    provider = row["provider"]
    with st.form(fz.wkey("verify", ck)):
        name = st.text_input("Display name", row["display_name"].split(" (UNVERIFIED")[0])
        efforts = st.multiselect(
            "Effort values (exact provider strings)",
            EFFORT_VOCABULARY,
            default=reported.get("effort_levels") or row["efforts"],
        )
        default = st.selectbox("Provider default effort", ["(not documented)", *EFFORT_VOCABULARY])
        thinking = st.radio(
            "Thinking mode sent by the adapter",
            ["adaptive", "none"] if provider == "anthropic" else ["none"],
            horizontal=True,
        )
        max_out = st.number_input(
            "Max output tokens (documented)",
            min_value=0,
            value=int(reported.get("max_tokens") or row["max_output_tokens"] or 0),
        )
        source = st.text_input("Capability source (official documentation URL, https://…)")
        verified_on = st.date_input("Capabilities verified on", value=date.today())
        st.markdown("**Prices (optional; leave input empty for 'unavailable')**")
        c = st.columns(4)
        p_in = c[0].text_input("Input USD / MTok")
        p_out = c[1].text_input("Output USD / MTok")
        p_read = c[2].text_input("Cache read USD / MTok")
        p_write = c[3].text_input("Cache write USD / MTok")
        p_source = st.text_input("Pricing source (official pricing page URL, https://…)")
        p_on = st.date_input("Prices verified on", value=date.today())
        note = st.text_input("Note")
        if st.form_submit_button("Store verified entry as a candidate snapshot (authenticated)"):
            pricing = None
            if p_in.strip():
                try:
                    pricing = {"input": float(p_in), "output": float(p_out)}
                    if p_read.strip():
                        pricing["cache_read"] = float(p_read)
                    if p_write.strip():
                        pricing["cache_write"] = float(p_write)
                except ValueError:
                    st.error("Prices must be numbers.")
                    return
            body = {
                "expected_active_sha256": data["active_sha256"],
                "verification": {
                    "provider": provider,
                    "model_id": row["model_id"],
                    "display_name": name,
                    "efforts": efforts,
                    "provider_default_effort": None if default.startswith("(") else default,
                    "thinking": thinking,
                    "max_output_tokens": int(max_out) or None,
                    "adapter": "anthropic.messages"
                    if provider == "anthropic"
                    else "openai.responses",
                    "capability_source": source,
                    "capabilities_verified_on": verified_on.isoformat(),
                    "pricing": pricing,
                    "pricing_source": p_source or None if pricing else None,
                    "pricing_verified_on": p_on.isoformat() if pricing else None,
                    "note": note or None,
                },
            }
            result = be.api.post("/models/verifications", token=token, json=body)
            if result.ok:
                fz.put(fz.key(fz.SNAPSHOT_ACTIVE, pid), result.data["snapshot"]["snapshot_id"])
            fz.submit(
                result,
                show,
                [ck],
                "Verified entry stored as a candidate snapshot; preview and activate it.",
            )


# =========================================================================== corrections
def page_corrections(be: UIBackend, pid: str, token: str | None, show: Show) -> None:
    st.header("Corrections, study links and double counting")
    st.caption(
        "Corrections must be grounded in the report's linked PDF (page and verbatim quote) or "
        "explicitly attested when the source cannot be machine-checked. Correcting an already "
        "accepted value (Stage 4 / Gate 3) invalidates the arm built from it: rebuild and "
        "re-approve the arms of that report. Statistics, GRADE and syntheses that used the old "
        "data become stale automatically."
    )
    fz.flash()
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Correct a value", "Report → study links", "Double counting", "Withdraw arms"]
    )
    with tab4:
        _withdraw(be, pid, token, show)
    with tab1:
        _correction(be, pid, token, show)
    with tab2:
        _links(be, pid, token, show)
    with tab3:
        _overlap(be, pid, token, show)


def _correction(be: UIBackend, pid: str, token: str | None, show: Show) -> None:
    ik = fz.key(fz.ITEMS, pid)
    items, error = fz.frozen(ik, lambda: be.api.get(f"/projects/{pid}/extraction/items"))
    if items is None:
        st.error(error)
        return
    live = be.api.get(f"/projects/{pid}/extraction/items")
    if live.ok:
        fz.warn_if_changed(
            sorted(i["state_sha256"] for i in live.data), sorted(i["state_sha256"] for i in items)
        )
    fz.reload_button(ik)
    if not items:
        st.info("No extraction items.")
        return
    labels = {
        f"{i['record_id']} · {i['field_name']} · {i.get('outcome_id') or '-'} · "
        f"{i.get('arm_label') or '-'} · {i.get('timepoint') or '-'} · {i['review_status']} · "
        f"v{i['version']} · {i['item_id'][:8]}": i
        for i in items
    }
    choice = st.selectbox("Item", list(labels), key=fz.wkey("item", ik))
    item = labels[choice]
    ev = item.get("evidence") or {}
    st.write(
        {
            "as printed": item.get("reported_text"),
            "status": item["data_status"],
            "review": item["review_status"],
            "reviewed by": item.get("reviewed_by"),
            "page": ev.get("page_number"),
            "location": ev.get("location_identifier"),
            "quote": ev.get("verbatim_quote"),
            "evidence check": (item.get("verification") or {}).get("status"),
            "linked PDF now (decision bound to it)": (item.get("source") or {}).get("pdf_sha256"),
            "PDF identity": (item.get("source") or {}).get("match_status"),
        }
    )
    hist = be.api.get(f"/projects/{pid}/extraction/items/{item['item_id']}/history")
    with st.expander("Audit trail of this item"):
        st.dataframe(
            [
                {k: h.get(k) for k in ("occurred_at", "action", "actor", "note")}
                for h in (hist.data or [])
            ]
        )
    if item["review_status"] == "superseded":
        st.info("This version is superseded; correct its current version.")
        return
    with st.form(fz.wkey("correct", (*ik, item["item_id"]))):
        status = st.selectbox(
            "Data status", DATA_STATUSES, index=DATA_STATUSES.index(item["data_status"])
        )
        text = st.text_input("Number exactly as printed", item.get("reported_text") or "")
        page = st.number_input(
            "PDF page (physical, 1-based)", min_value=1, value=int(ev.get("page_number") or 1)
        )
        loc_type = st.selectbox(
            "Location type",
            LOCATION_TYPES,
            index=LOCATION_TYPES.index(ev.get("location_type") or "paragraph"),
        )
        loc_id = st.text_input("Location id (e.g. p2.t1.r1c2)", ev.get("location_identifier") or "")
        quote = st.text_area("Verbatim quote containing the number", ev.get("verbatim_quote") or "")
        attest = st.checkbox(
            "The source cannot be machine-checked (figure / image-only page): I attest the value"
        )
        note = st.text_area("Reason for the correction (required)")
        if st.form_submit_button("Submit correction (authenticated)"):
            correction = {k: item[k] for k in CORRECTION_COPY if item.get(k) is not None} | {
                "data_status": status,
                "reported_text": text or None,
                "page": int(page),
                "location_type": loc_type,
                "location_id": loc_id or None,
                "quote": quote or None,
            }
            result = be.api.post(
                f"/projects/{pid}/extraction/items/{item['item_id']}/correct",
                token=token,
                json={
                    "correction": correction,
                    "note": note,
                    "attest_unverifiable": attest,
                    "expected_state_sha256": item["state_sha256"],
                },
            )
            invalid = (result.data or {}).get("invalidated_arms") if result.ok else None
            fz.submit(
                result,
                show,
                [ik],
                "Correction recorded."
                + (
                    f" Invalidated arm(s): {', '.join(invalid)} — rebuild the arms of this report."
                    if invalid
                    else ""
                ),
            )


def _withdraw(be: UIBackend, pid: str, token: str | None, show: Show) -> None:
    """Retire the arms of a report that is no longer included at full text."""
    st.caption(
        "For a report excluded at full text after its arms were built (e.g. on re-review after "
        "an eligibility amendment). The arms and their evidence stay stored; the withdrawal is "
        "recorded with your name and reason. Included reports are corrected instead."
    )
    rid = st.text_input("Report (record id)", key=fz.wkey("withdraw_rid", fz.key(fz.ARMS, pid)))
    if not rid:
        return
    wk = fz.key(fz.ARMS, pid, rid)
    view, error = fz.frozen(
        wk, lambda: be.api.get(f"/projects/{pid}/extraction/records/{rid}/arms-state")
    )
    if view is None:
        st.error(error)
        return
    live = be.api.get(f"/projects/{pid}/extraction/records/{rid}/arms-state")
    if live.ok:
        fz.warn_if_changed(live.data["state_sha256"], view["state_sha256"])
    fz.reload_button(wk)
    st.write({"included at full text": view["included"], "active arms": len(view["arms"])})
    st.dataframe(
        [
            {k: a.get(k) for k in ("arm_id", "outcome_id", "arm_label", "timepoint")}
            for a in view["arms"]
        ]
    )
    with st.form(fz.wkey("withdraw", wk)):
        reason = st.text_area("Reason for the withdrawal (required)")
        if st.form_submit_button("Withdraw these arms (authenticated)"):
            fz.submit(
                be.api.post(
                    f"/projects/{pid}/extraction/records/{rid}/withdraw-arms",
                    token=token,
                    json={"reason": reason, "expected_state_sha256": view["state_sha256"]},
                ),
                show,
                [wk],
                "Arms withdrawn.",
            )


def _links(be: UIBackend, pid: str, token: str | None, show: Show) -> None:
    lk = fz.key(fz.LINKS, pid)
    view, error = fz.frozen(lk, lambda: be.api.get(f"/projects/{pid}/studies"))
    if view is None:
        st.error(error)
        return
    live = be.api.get(f"/projects/{pid}/studies")
    if live.ok:
        fz.warn_if_changed(live.data["state_sha256"], view["state_sha256"])
    fz.reload_button(lk, fz.key(fz.OVERLAP, pid))
    st.dataframe(
        [
            {k: x[k] for k in ("record_id", "study_id", "is_primary_report", "linked_by", "note")}
            for x in view["links"]
        ]
    )
    with st.expander("Link history (append-only)"):
        st.dataframe(view["history"])
    included = view["full_text_included"]
    with st.form(fz.wkey("link", lk)):
        records = st.multiselect("Reports of one study", included)
        study = st.text_input("Study id (letters, digits, . _ : -)")
        primary = st.selectbox("Primary report", records or [""])
        note = st.text_area("Why these reports describe one study (required)")
        if st.form_submit_button("Link reports (authenticated)") and records:
            fz.submit(
                be.api.post(
                    f"/projects/{pid}/studies/link",
                    token=token,
                    json={
                        "record_ids": records,
                        "study_id": study,
                        "primary_record_id": primary,
                        "note": note,
                        "expected_state_sha256": view["state_sha256"],
                    },
                ),
                show,
                [lk, fz.key(fz.OVERLAP, pid)],
                "Reports linked.",
            )
    linked = [x["record_id"] for x in view["links"]]
    with st.form(fz.wkey("unlink", lk)):
        records = st.multiselect("Reports to unlink (each becomes its own study)", linked)
        note = st.text_area("Reason (required)", key=fz.wkey("unlink_note", lk))
        if st.form_submit_button("Unlink (authenticated)") and records:
            fz.submit(
                be.api.post(
                    f"/projects/{pid}/studies/unlink",
                    token=token,
                    json={
                        "record_ids": records,
                        "note": note,
                        "expected_state_sha256": view["state_sha256"],
                    },
                ),
                show,
                [lk, fz.key(fz.OVERLAP, pid)],
                "Reports unlinked.",
            )


def _overlap(be: UIBackend, pid: str, token: str | None, show: Show) -> None:
    ok = fz.key(fz.OVERLAP, pid)
    view, error = fz.frozen(
        ok, lambda: be.api.get(f"/projects/{pid}/extraction/double-counting/state")
    )
    if view is None:
        st.error(error)
        return
    live = be.api.get(f"/projects/{pid}/extraction/double-counting/state")
    if live.ok:
        fz.warn_if_changed(live.data["state_sha256"], view["state_sha256"])
    fz.reload_button(ok)
    if not view["conflicts"]:
        st.success("No unresolved double counting.")
        return
    for c in view["conflicts"]:
        st.warning(c)
    with st.form(fz.wkey("overlap", ok)):
        study = st.text_input("Study / cohort id (as shown above)")
        outcome = st.text_input("Outcome id (as shown above)")
        mode = st.radio(
            "Decision", ["keep one report", "reports contribute different participants"]
        )
        keep = st.text_input("Report to keep (first mode)")
        others = st.text_input("Other report ids, comma-separated (dropped, or non-overlapping)")
        justification = st.text_area("Justification (required)")
        if st.form_submit_button("Record decision (authenticated)"):
            ids = [x.strip() for x in others.split(",") if x.strip()]
            body: dict[str, Any] = {
                "study_id": study,
                "outcome_id": outcome,
                "justification": justification,
                "expected_state_sha256": view["state_sha256"],
            }
            if mode == "keep one report":
                body |= {"keep_record_id": keep, "drop_record_ids": ids}
            else:
                body |= {"non_overlapping_record_ids": ids}
            fz.submit(
                be.api.post(
                    f"/projects/{pid}/extraction/double-counting/resolve", token=token, json=body
                ),
                show,
                [ok],
                "Decision recorded.",
            )


# =========================================================================== plans
def page_plan(be: UIBackend, pid: str, token: str | None, show: Show) -> None:
    st.header("Analysis plan (draft, revise, approve)")
    st.caption(
        "Each save is a new draft version; approval stores that exact draft as approved and "
        "makes statistics computed from an earlier plan stale. Approval is refused from Stage 6 "
        "on (Stage 5 could not be re-run)."
    )
    fz.flash()
    pk = fz.key(fz.PLANS, pid)
    view, error = fz.frozen(pk, lambda: be.api.get(f"/projects/{pid}/stats/plans"))
    if view is None:
        st.error(error)
        return
    live = be.api.get(f"/projects/{pid}/stats/plans")
    if live.ok:
        fz.warn_if_changed(
            (live.data["latest_sha256"], live.data["protocol_sha256"]),
            (view["latest_sha256"], view["protocol_sha256"]),
        )
    fz.reload_button(pk, fz.key(fz.PLAN_WORK, pid))
    # The outcome definitions shown are those of the frozen screen; every save or approval
    # sends their protocol hash and is refused if an amendment changed them meanwhile.
    st.caption(
        f"Composed against protocol version {view['protocol_version']} "
        f"({(view['protocol_sha256'] or '')[:12]})."
    )
    with st.expander("Outcome definitions of this protocol version"):
        st.json(view["outcomes"])
    st.dataframe(
        [
            {
                "version": p["version"],
                "status": p["status"],
                "analyses": len(p["analyses"]),
                "drafted by": p["drafted_by"],
                "approved by": p.get("approved_by"),
                "sha256": p["sha256"][:12],
            }
            for p in view["plans"]
        ]
    )
    latest = view["plans"][-1] if view["plans"] else None
    if latest is not None and latest["status"] == "draft":
        st.subheader(f"Latest draft (version {latest['version']})")
        st.json(latest["analyses"])
        for p in view["draft_problems"]:
            st.error(f"Would be refused: {p}")
        with st.form(fz.wkey("approve_plan", pk)):
            st.caption("You approve exactly the draft displayed above.")
            if st.form_submit_button("Approve this draft (authenticated)"):
                fz.submit(
                    be.api.post(
                        f"/projects/{pid}/stats/plan/drafts/{latest['plan_id']}/approve",
                        token=token,
                        json={
                            "expected_sha256": latest["sha256"],
                            "expected_latest_sha256": view["latest_sha256"],
                            "expected_protocol_sha256": view["protocol_sha256"],
                        },
                    ),
                    show,
                    [pk],
                    "Plan approved.",
                )
    outcomes = {o["outcome_id"]: o for o in view["outcomes"]}
    st.subheader("Compose a new draft version")
    wk = fz.key(fz.PLAN_WORK, pid)
    work: list[dict[str, Any]] = fz.store().setdefault(
        wk, [dict(a) for a in (latest or {}).get("analyses", [])]
    )
    st.json(work)
    if not outcomes:
        st.info("The protocol defines no outcomes yet.")
        return
    with st.form(fz.wkey("add_analysis", pk)):
        aid = st.text_input("Analysis id")
        oid = st.selectbox("Outcome", list(outcomes))
        o = outcomes[oid]
        comparison = ComparisonType(o.get("comparison") or "comparative")
        allowed = ALLOWED_MEASURES.get((comparison, o["data_type"]), frozenset(EffectMeasure))
        measures = sorted(m.value for m in allowed)
        measure = st.selectbox("Effect measure", measures)
        tps = o.get("timepoints") or []
        chosen = st.multiselect(
            "Timepoints (pre-specified in the protocol)",
            list(range(len(tps))),
            format_func=lambda i: json.dumps(tps[i]),
        )
        model = st.selectbox("Primary model", ["random_effects", "common_effect"])
        ci = st.selectbox("Random-effects CI", ["hksj", "hksj_adhoc", "wald"])
        tau = st.selectbox("τ² estimator", ["REML", "DL"])
        scale = st.selectbox(
            "Declared CI scale (KM / cumulative incidence only)",
            ["", "log_log", "log", "logit", "plain"],
        )
        extra = st.text_area("Advanced settings as JSON (e.g. arm_selections), optional", "{}")
        if st.form_submit_button("Add analysis to the composition"):
            try:
                spec = {
                    "analysis_id": aid,
                    "outcome_id": oid,
                    "effect_measure": measure,
                    "timepoints": [tps[i] for i in chosen],
                    "primary_model": model,
                    "random_effects_ci": ci,
                    "tau2_estimator": tau,
                } | json.loads(extra or "{}")
            except json.JSONDecodeError as exc:
                st.error(f"Advanced settings are not valid JSON: {exc}")
                return
            if scale:
                spec["reported_ci_scale"] = scale
            work.append(spec)
            st.rerun()
    if st.button("Clear the composition", key=fz.wkey("clear_plan", pk)):
        fz.put(wk, [])
        st.rerun()
    with st.form(fz.wkey("save_plan", pk)):
        rationale = st.text_area("Rationale for this version")
        if st.form_submit_button("Save as a new draft version (authenticated)") and work:
            fz.submit(
                be.api.post(
                    f"/projects/{pid}/stats/plan/drafts",
                    token=token,
                    json={
                        "analyses": work,
                        "rationale": rationale or None,
                        "expected_latest_sha256": view["latest_sha256"],
                        "expected_protocol_sha256": view["protocol_sha256"],
                    },
                ),
                show,
                [pk, wk],
                "Draft saved.",
            )


# =========================================================================== amendments
EDITABLE_TEXT = (
    "review_title",
    "research_question",
    "population",
    "intervention",
    "comparator",
    "prospero_id",
)
EDITABLE_JSON = ("outcomes", "mesh_terms", "selected_databases", "search_strategies")


def page_amendments(be: UIBackend, pid: str, token: str | None, show: Show) -> None:
    st.header("Protocol amendments (after Gate 1)")
    st.caption(
        "The approved protocol is never edited. An amendment is a proposal with a rationale; it "
        "becomes protocol version n+1 only when an enrolled reviewer approves it, and the work it "
        "affects becomes pending re-review (forward transitions are blocked until each item is "
        "resolved). Earlier versions and decisions stay stored."
    )
    fz.flash()
    ak = fz.key(fz.AMENDMENT, pid)
    view, error = fz.frozen(ak, lambda: be.api.get(f"/projects/{pid}/amendments"))
    if view is None:
        st.error(error)
        return
    live = be.api.get(f"/projects/{pid}/amendments")
    if live.ok:
        fz.warn_if_changed(_amend_ids(live.data), _amend_ids(view))
    fz.reload_button(ak, fz.key(fz.AMEND_WORK, pid))
    for p in view["problems"]:
        st.warning(p)
    with st.expander("Which changes affect what (rules)"):
        st.json({"affected work": view["impact_rules"], "allowed phases": view["phase_rules"]})
    base = view["approved_protocol"]
    if base is None or base.get("status") != "approved":
        st.info("Amendments start from an approved protocol (after Gate 1).")
        return
    st.markdown(f"**Approved protocol version {base['version']}** · {view['approved_sha256'][:12]}")
    pending = view["open"]
    if pending is None:
        _proposal_form(pid, base, view, token, show, be)
    else:
        _pending_amendment(pid, pending, token, show, be)
    _obligations(pid, view, token, show, be)
    with st.expander("Amendment history (append-only)"):
        st.dataframe(
            [
                {
                    k: h.get(k)
                    for k in (
                        "amendment_id",
                        "revision",
                        "proposed_by",
                        "proposed_at",
                        "rationale",
                        "revision_note",
                    )
                }
                for h in view["history"]
            ]
        )
        st.dataframe(
            [
                {
                    k: d.get(k)
                    for k in (
                        "amendment_id",
                        "decision",
                        "decided_by",
                        "decided_at",
                        "protocol_version",
                        "comments",
                    )
                }
                for d in view["decisions"]
            ]
        )


def _amend_ids(v: dict[str, Any]) -> tuple[Any, ...]:
    return (
        v["approved_sha256"],
        (v["open"] or {}).get("sha256"),
        tuple((o["obligation_id"], o["state"]) for o in v["obligations"]),
    )


def _proposal_form(
    pid: str,
    base: dict[str, Any],
    view: dict[str, Any],
    token: str | None,
    show: Show,
    be: UIBackend,
) -> None:
    st.subheader("Propose an amendment")
    with st.form(fz.wkey("propose", fz.key(fz.AMENDMENT, pid))):
        edited: dict[str, Any] = {}
        for f in EDITABLE_TEXT:
            edited[f] = st.text_area(f.replace("_", " ").capitalize(), base.get(f) or "")
        edited["study_designs"] = st.multiselect(
            "Eligible study designs", [d.value for d in StudyDesign], default=base["study_designs"]
        )
        for f in ("inclusion_criteria", "exclusion_criteria"):
            edited[f] = st.text_area(
                f"{f.replace('_', ' ').capitalize()} (one per line)", "\n".join(base.get(f) or [])
            )
        st.caption(
            "Outcomes, MeSH terms, databases and search strategies (JSON; search changes "
            "are accepted in Stage 2 only)."
        )
        for f in EDITABLE_JSON:
            edited[f] = st.text_area(
                f, json.dumps(base.get(f), indent=1, ensure_ascii=False), height=160
            )
        rationale = st.text_area("Rationale for the amendment (required)")
        if st.form_submit_button("Submit proposal (authenticated; nothing changes until approved)"):
            data = dict(base)
            try:
                for f in EDITABLE_TEXT:
                    data[f] = edited[f].strip() or None
                data["study_designs"] = edited["study_designs"]
                for f in ("inclusion_criteria", "exclusion_criteria"):
                    data[f] = [x.strip() for x in edited[f].splitlines() if x.strip()]
                for f in EDITABLE_JSON:
                    data[f] = json.loads(edited[f])
            except json.JSONDecodeError as exc:
                st.error(f"Invalid JSON: {exc}")
                return
            fz.submit(
                be.api.post(
                    f"/projects/{pid}/amendments",
                    token=token,
                    json={
                        "protocol": data,
                        "rationale": rationale,
                        "expected_base_sha256": view["approved_sha256"],
                    },
                ),
                show,
                [fz.key(fz.AMENDMENT, pid)],
                "Amendment proposed.",
            )


def _pending_amendment(
    pid: str, pending: dict[str, Any], token: str | None, show: Show, be: UIBackend
) -> None:
    ak = fz.key(fz.AMENDMENT, pid)
    p = pending["proposal"]
    aid, sha = p["amendment_id"], pending["sha256"]
    st.subheader(f"Open amendment {aid[:12]} · revision {p['revision']}")
    st.write(
        {
            "proposed by": p["proposed_by"],
            "proposed at": p["proposed_at"],
            "rationale": p["rationale"],
            "changes": p["material_changes"],
            "affected work": {i["group"]: i["affected"] for i in p["impact"]},
        }
    )
    st.markdown(
        f"**Work that becomes pending re-review if approved:** "
        f"{len(pending['obligations_if_approved'])} item(s)"
    )
    st.dataframe(
        [
            {k: o[k] for k in ("area", "subject", "previous", "reason")}
            for o in pending["obligations_if_approved"]
        ]
    )
    for b in pending["approval_blockers"]:
        st.error(f"Blocks approval: {b}")
    proposed = p["proposed"]
    cols = st.columns(2)
    if cols[0].button("Verify new MeSH terms (NLM)", key=fz.wkey("mesh", ak)):
        fz.submit(
            be.api.post(
                f"/projects/{pid}/amendments/{aid}/mesh/verify",
                token=token,
                json={"expected_sha256": sha},
            ),
            show,
            [ak],
            "MeSH checked.",
        )
    for s in proposed["search_strategies"]:
        if s.get("syntax_validated"):
            continue
        strategy_sha = _strategy_sha(s)
        if strategy_sha is None:
            continue
        label = f"{s['database']} strategy {strategy_sha[:12]}"
        c = st.columns(2)
        if c[0].button(f"API preview: {label}", key=fz.wkey(f"prev:{strategy_sha}", ak)):
            fz.submit(
                be.api.post(
                    f"/projects/{pid}/amendments/{aid}/strategies/{strategy_sha}/preview",
                    token=token,
                    json={"expected_sha256": sha},
                ),
                show,
                [ak],
                "Preview recorded.",
            )
        note = c[1].text_input(f"Attestation note: {label}", key=fz.wkey(f"att:{strategy_sha}", ak))
        if (
            c[1].button(
                f"Attest syntax (non-API database): {label}",
                key=fz.wkey(f"attb:{strategy_sha}", ak),
            )
            and note
        ):
            fz.submit(
                be.api.post(
                    f"/projects/{pid}/amendments/{aid}/strategies/{strategy_sha}/attest",
                    token=token,
                    json={"expected_sha256": sha, "note": note},
                ),
                show,
                [ak],
                "Attestation recorded.",
            )
    with st.form(fz.wkey("decide", ak)):
        decision = st.radio("Decision", ["approved", "rejected", "withdrawn"], horizontal=True)
        comments = st.text_area("Comments (required)")
        if st.form_submit_button("Record decision on the displayed revision (authenticated)"):
            fz.submit(
                be.api.post(
                    f"/projects/{pid}/amendments/{aid}/decision",
                    token=token,
                    json={"decision": decision, "comments": comments, "expected_sha256": sha},
                ),
                show,
                [ak],
                f"Amendment {decision}.",
            )


def _strategy_sha(s: dict[str, Any]) -> str | None:
    from src.core.schemas.protocol import SearchStrategy

    try:
        return SearchStrategy.model_validate(s).sha256
    except ValueError:
        return None


def _obligations(
    pid: str, view: dict[str, Any], token: str | None, show: Show, be: UIBackend
) -> None:
    ak = fz.key(fz.AMENDMENT, pid)
    rows = view["obligations"]
    if not rows:
        return
    st.subheader("Re-review after approved amendments")
    st.dataframe(
        [{k: o[k] for k in ("area", "subject", "previous", "state", "detail")} for o in rows]
    )
    open_rows = [o for o in rows if o["state"] == "open" and o["area"] != "searches"]
    if not open_rows:
        return
    labels = {f"{o['area']} · {o['subject']} · was {o['previous'] or '-'}": o for o in open_rows}
    with st.form(fz.wkey("resolve", ak)):
        choice = st.selectbox("Item", list(labels))
        resolution = st.radio(
            "Resolution",
            ["confirmed", "revised", "not_applicable", "withdrawn"],
            horizontal=True,
            help="withdrawn: extraction data that do not meet an amended outcome definition "
            "are recorded as unavailable for that outcome (data and evidence stay stored)",
        )
        verdict = st.selectbox(
            "Revised screening verdict (screening items only)", ["", "include", "exclude"]
        )
        reason = st.text_input("Exclusion reason (if excluding)")
        note = st.text_area("Re-review note (required)")
        if st.form_submit_button("Record re-review (authenticated)"):
            o = labels[choice]
            fz.submit(
                be.api.post(
                    f"/projects/{pid}/amendments/obligations/{o['obligation_id']}/resolve",
                    token=token,
                    json={
                        "resolution": resolution,
                        "note": note,
                        "verdict": verdict or None,
                        "exclusion_reason": reason or None,
                    },
                ),
                show,
                [ak],
                "Re-review recorded.",
            )
