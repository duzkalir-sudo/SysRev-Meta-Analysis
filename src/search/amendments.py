"""Protocol amendments after Gate 1 (Session 8).

The approved protocol is never unlocked or edited. An amendment is a separate, append-only
proposal (``governance_records``, kind ``amendment_proposal``) holding the complete proposed
protocol, the rationale, the proposer and the computed impact. Until an enrolled reviewer
approves it, the approved protocol stays current for every step of the pipeline. On approval
the proposal becomes protocol version *n*+1 (status ``approved``; version *n* becomes
``superseded`` and stays stored), and the downstream work it affects becomes **pending
re-review** — recorded as obligations that block every forward transition until a human
resolves each one. Historical decisions are never rewritten: a re-review that changes a
screening verdict is a new, authenticated human decision recorded under the amendment.

Which changes affect what (``IMPACT``):

=================  ===========================================================================
change group       affected work (and how it is enforced)
=================  ===========================================================================
search             searches — new or changed strategies must be executed, imported or declared
                   not searched (Stage 2 guard); new records are deduplicated incrementally and
                   screened; PRISMA counts are re-derived. **Stage 2 only.**
eligibility        title/abstract and full-text screening — every record with a final decision
                   becomes a re-review obligation (confirm, or record a revised verdict);
                   extraction, risk of bias, statistics, reporting, audit and sign-off follow
                   from the included set. **Stage 2 to Gate 3.**
outcomes           outcome definitions, extraction (one obligation per report and changed or
                   removed outcome), analysis plans (an approved plan that uses a changed
                   outcome must be re-approved or confirmed), statistics (the input hash covers
                   the outcome definitions), GRADE/SoF (input hash), reporting, audit, sign-off.
                   **Stage 2 to Gate 3.**
administrative     reporting, audit and sign-off (the manuscript evidence fingerprint covers the
                   protocol, so audits and sign-offs made before the change no longer count).
                   **Stage 2 to the audit phase.**
=================  ===========================================================================

Statistics, GRADE/Summary-of-Findings and syntheses, audits and sign-offs are invalidated by
their existing input hashes and the evidence fingerprint; they are listed as affected work but
need no separate obligation. Unsupported, refused explicitly: amendments before Gate 1 (edit
the draft instead), search changes after Stage 2 and eligibility/outcome changes after Gate 3
(the state machine has no route back to the stages that would have to be repeated), and
amendments after completion.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, Literal, Protocol

from pydantic import Field

from src.core.enums import ActorType, Phase, ReviewStatus, ScreeningStage, ScreeningVerdict
from src.core.exceptions import SysRevError
from src.core.schemas.base import AwareDatetime, NonEmptyStr, StrictModel, new_id, utc_now
from src.core.schemas.protocol import PICOProtocol
from src.core.schemas.search import ProtocolEvent
from src.search.protocol_editor import (
    MESH_VERIFICATION_KEYS,
    PROTECTED_KEYS,
    STRATEGY_VALIDATION_KEYS,
    material_changes,
    protocol_sha256,
)
from src.state.repository import Repository

ImpactArea = Literal[
    "searches", "title_abstract_screening", "full_text_screening", "extraction",
    "outcome_definitions", "analysis_plans", "statistics", "reporting", "audit", "sign_off",
]  # fmt: skip
ObligationArea = Literal[
    "title_abstract_screening", "full_text_screening", "extraction", "analysis_plans", "searches"
]
IMPACT: dict[str, tuple[ImpactArea, ...]] = {
    "search": ("searches", "title_abstract_screening", "reporting", "audit", "sign_off"),
    "eligibility": (
        "title_abstract_screening", "full_text_screening", "extraction", "statistics",
        "reporting", "audit", "sign_off",
    ),
    "outcomes": (
        "outcome_definitions", "extraction", "analysis_plans", "statistics", "reporting",
        "audit", "sign_off",
    ),
    "administrative": ("reporting", "audit", "sign_off"),
}  # fmt: skip
STAGE2 = {Phase.STAGE2_SEARCH_SCREENING}
UP_TO_GATE3 = {
    Phase.STAGE2_SEARCH_SCREENING, Phase.GATE1_5_SCREENING_CONFLICTS,
    Phase.STAGE3_FULLTEXT_RETRIEVAL, Phase.GATE2_MISSING_FULLTEXT,
    Phase.STAGE4_EXTRACTION_ROB, Phase.GATE3_EXTRACTION_APPROVAL,
}  # fmt: skip
UP_TO_AUDIT = UP_TO_GATE3 | {
    Phase.STAGE5_META_ANALYSIS, Phase.STAGE6_MANUSCRIPT, Phase.STAGE6_AUDIT,
}  # fmt: skip
ALLOWED_PHASES: dict[str, set[Phase]] = {
    "search": STAGE2,
    "eligibility": UP_TO_GATE3,
    "outcomes": UP_TO_GATE3,
    "administrative": UP_TO_AUDIT,
}
# Note shown with each group; the reason a phase is not allowed.
PHASE_REASON: dict[str, str] = {
    "search": "update searches are supported in Stage 2 only (the pipeline cannot return to "
    "identification and screening from later stages)",
    "eligibility": "eligibility changes need re-screening and possibly re-extraction, which "
    "cannot be repeated after Gate 3",
    "outcomes": "outcome changes need re-extraction and a re-approved analysis plan, which "
    "cannot be repeated after Gate 3",
    "administrative": "administrative changes are accepted until the audit phase",
}


class AmendmentError(SysRevError):
    """The amendment cannot be proposed, revised or decided in the current state."""


class Reviewer(Protocol):
    @property
    def id(self) -> str: ...
    @property
    def display_name(self) -> str: ...


class ImpactItem(StrictModel):
    group: str
    fields: list[str]
    affected: list[ImpactArea]


class Obligation(StrictModel):
    """One piece of work that must be re-reviewed by a human because of an amendment."""

    obligation_id: str
    area: ObligationArea
    subject: str  # record id, "record|outcome", plan id or strategy sha
    stage: ScreeningStage | None = None
    previous: str | None = None  # e.g. the verdict that was final before the amendment
    reason: str


class AmendmentProposal(StrictModel):
    amendment_id: str = Field(default_factory=new_id)
    revision: int = Field(default=1, ge=1)
    base_version: int = Field(ge=1)
    base_sha256: str
    proposed: dict[str, Any]
    proposed_sha256: str
    rationale: NonEmptyStr
    proposed_by: NonEmptyStr
    reviewer_id: str
    proposed_at: AwareDatetime = Field(default_factory=utc_now)
    phase: str
    material_changes: dict[str, list[str]]
    impact: list[ImpactItem]
    revision_note: str | None = None

    @property
    def protocol(self) -> PICOProtocol:
        return PICOProtocol.model_validate(self.proposed)

    @property
    def sha256(self) -> str:
        return _sha(self.model_dump(mode="json"))


class AmendmentDecision(StrictModel):
    amendment_id: str
    revision: int
    decision: Literal["approved", "rejected", "withdrawn"]
    decided_by: NonEmptyStr
    reviewer_id: str
    comments: NonEmptyStr
    decided_at: AwareDatetime = Field(default_factory=utc_now)
    protocol_version: int | None = None
    obligations: list[Obligation] = Field(default_factory=list)


class ObligationResolution(StrictModel):
    obligation_id: str
    amendment_id: str
    area: ObligationArea
    subject: str
    stage: ScreeningStage | None = None
    resolution: Literal["confirmed", "revised", "not_applicable"]
    verdict: ScreeningVerdict | None = None  # for a revised screening verdict
    exclusion_reason: str | None = None
    note: NonEmptyStr
    resolved_by: NonEmptyStr
    reviewer_id: str
    resolved_at: AwareDatetime = Field(default_factory=utc_now)


def _sha(data: Any) -> str:
    text = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      default=str)  # fmt: skip
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- reading state
def proposals(repo: Repository, project_id: str) -> list[AmendmentProposal]:
    return [
        AmendmentProposal.model_validate({k: v for k, v in r.items() if not k.startswith("_")})
        for r in repo.governance_records(project_id, "amendment_proposal")
    ]


def decisions(repo: Repository, project_id: str) -> list[AmendmentDecision]:
    return [
        AmendmentDecision.model_validate({k: v for k, v in r.items() if not k.startswith("_")})
        for r in repo.governance_records(project_id, "amendment_decision")
    ]


def resolutions(repo: Repository, project_id: str) -> list[ObligationResolution]:
    return [
        ObligationResolution.model_validate({k: v for k, v in r.items() if not k.startswith("_")})
        for r in repo.governance_records(project_id, "amendment_resolution")
    ]


def latest_revisions(repo: Repository, project_id: str) -> dict[str, AmendmentProposal]:
    out: dict[str, AmendmentProposal] = {}
    for p in proposals(repo, project_id):
        out[p.amendment_id] = p
    return out


def open_amendment(repo: Repository, project_id: str) -> AmendmentProposal | None:
    decided = {d.amendment_id for d in decisions(repo, project_id)}
    open_ = [p for a, p in latest_revisions(repo, project_id).items() if a not in decided]
    return open_[-1] if open_ else None


def _approved_protocol(repo: Repository, project_id: str) -> PICOProtocol:
    protocol = repo.latest_protocol(project_id)
    if protocol is None or protocol.status != "approved":
        raise AmendmentError("amendments start from an approved protocol (after Gate 1)")
    return protocol


# --------------------------------------------------------------------------- impact
def impact(changes: Mapping[str, list[str]]) -> list[ImpactItem]:
    return [
        ImpactItem(group=group, fields=list(fields), affected=list(IMPACT[group]))
        for group, fields in changes.items()
    ]


def _changed_outcomes(base: PICOProtocol, new: PICOProtocol) -> set[str]:
    old = {o.outcome_id: o.model_dump(mode="json") for o in base.outcomes}
    cur = {o.outcome_id: o.model_dump(mode="json") for o in new.outcomes}
    return {oid for oid in old if cur.get(oid) != old[oid]}


def compute_obligations(
    repo: Repository, project_id: str, base: PICOProtocol, new: PICOProtocol, amendment_id: str
) -> list[Obligation]:
    """Concrete work that must be re-reviewed under the amended protocol."""
    changes = material_changes(base, new)
    out: list[Obligation] = []

    def oid(*parts: str) -> str:
        return _sha([amendment_id, *parts])[:32]

    if "eligibility" in changes:
        from src.screening.agreement import screening_status

        unique = {r.record_id for r in repo.load_records(project_id, unique_only=True)}
        decided = repo.load_screening_decisions(project_id)
        ta = screening_status(unique, decided, [], ScreeningStage.TITLE_ABSTRACT)
        for rid, verdict in sorted(ta.final.items()):
            out.append(Obligation(
                obligation_id=oid("ta", rid), area="title_abstract_screening", subject=rid,
                stage=ScreeningStage.TITLE_ABSTRACT, previous=verdict.value,
                reason=f"eligibility changed ({', '.join(changes['eligibility'])}): confirm "
                "or revise the title/abstract decision against the amended criteria",
            ))  # fmt: skip
        included = {r for r, v in ta.final.items() if v == ScreeningVerdict.INCLUDE}
        ft = screening_status(included, decided, [], ScreeningStage.FULL_TEXT)
        for rid, verdict in sorted(ft.final.items()):
            out.append(Obligation(
                obligation_id=oid("ft", rid), area="full_text_screening", subject=rid,
                stage=ScreeningStage.FULL_TEXT, previous=verdict.value,
                reason="eligibility changed: confirm or revise the full-text decision",
            ))  # fmt: skip
    changed = _changed_outcomes(base, new)
    if changed:
        pairs = sorted({
            (i.record_id, i.outcome_id) for i in repo.load_extraction_items(project_id)
            if i.outcome_id in changed and i.review_status != ReviewStatus.SUPERSEDED
        })  # fmt: skip
        for rid, outcome in pairs:
            assert outcome is not None
            out.append(Obligation(
                obligation_id=oid("ex", rid, outcome), area="extraction",
                subject=f"{rid}|{outcome}",
                reason=f"outcome {outcome} was changed or removed: re-check the extracted data "
                "of this report against the amended definition (correct values with the "
                "correction workflow), then resolve",
            ))  # fmt: skip
        plan = repo.latest_analysis_plan(project_id)
        if plan is not None and any(s.outcome_id in changed for s in plan.analyses):
            out.append(Obligation(
                obligation_id=oid("plan", plan.plan_id), area="analysis_plans",
                subject=plan.plan_id,
                reason=f"approved plan version {plan.version} analyses a changed or removed "
                "outcome: approve a revised plan, or confirm it still applies",
            ))  # fmt: skip
    old_strategies = {s.sha256 for s in base.search_strategies}
    for s in new.search_strategies:
        if s.sha256 not in old_strategies:
            out.append(Obligation(
                obligation_id=oid("search", s.sha256), area="searches", subject=s.sha256,
                reason=f"new or changed {s.database} strategy: execute or import it (or record "
                "it as not searched), then prepare the new records for screening",
            ))  # fmt: skip
    return out


def obligation_status(repo: Repository, project_id: str) -> list[dict[str, Any]]:
    """Every obligation of every approved amendment, with whether it is still open."""
    from src.search.identification import strategy_state

    runs = repo.load_search_runs(project_id)
    resolved = {r.obligation_id: r for r in resolutions(repo, project_id)}
    approved_plan = repo.latest_analysis_plan(project_id)
    rows = []
    for d in decisions(repo, project_id):
        if d.decision != "approved":
            continue
        for ob in d.obligations:
            state, detail = "open", ""
            res = resolved.get(ob.obligation_id)
            if res is not None:
                state, detail = "resolved", f"{res.resolution} by {res.resolved_by}: {res.note}"
            elif ob.area == "searches":
                st = strategy_state(runs, ob.subject)
                if st.status != "not_yet_searched":
                    state, detail = "resolved", f"strategy {st.status} (search log)"
            elif (
                ob.area == "analysis_plans"
                and approved_plan is not None
                and approved_plan.plan_id != ob.subject
                and approved_plan.approved_at is not None
                and approved_plan.approved_at > d.decided_at
            ):
                state, detail = "resolved", f"plan version {approved_plan.version} approved later"
            rows.append(ob.model_dump(mode="json") | {
                "amendment_id": d.amendment_id, "state": state, "detail": detail,
            })  # fmt: skip
    return rows


def amendment_problems(repo: Repository, project_id: str) -> list[str]:
    """Why forward transitions are blocked by amendments (empty = nothing outstanding)."""
    out: list[str] = []
    pending = open_amendment(repo, project_id)
    if pending is not None:
        out.append(
            f"protocol amendment {pending.amendment_id[:12]} (revision {pending.revision}) "
            "awaits a decision: approve, reject or withdraw it first"
        )
    open_obligations = [o for o in obligation_status(repo, project_id) if o["state"] == "open"]
    by_area: dict[str, int] = {}
    for o in open_obligations:
        by_area[o["area"]] = by_area.get(o["area"], 0) + 1
    for area, n in sorted(by_area.items()):
        out.append(
            f"{n} {area.replace('_', ' ')} item(s) are pending re-review after a protocol amendment"
        )
    return out


# --------------------------------------------------------------------------- building
def _strip(items: list[Any], keys: frozenset[str]) -> list[dict[str, Any]]:
    out = []
    for item in items:
        data = item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
        out.append({k: v for k, v in data.items() if k not in keys})
    return out


def build_proposed(base: PICOProtocol, data: Mapping[str, Any]) -> PICOProtocol:
    """The proposed protocol: submitted content, version base+1, verification carried over
    only for unchanged MeSH descriptors and unchanged strategies (as in Stage 1 drafting)."""
    clean = {k: v for k, v in dict(data).items() if k not in PROTECTED_KEYS}
    clean["mesh_terms"] = _strip(list(clean.get("mesh_terms") or []), MESH_VERIFICATION_KEYS)
    clean["search_strategies"] = _strip(
        list(clean.get("search_strategies") or []), STRATEGY_VALIDATION_KEYS
    )
    draft = PICOProtocol.model_validate(
        clean | {"protocol_id": new_id(), "version": base.version + 1, "status": "draft"}
    )
    verified = {m.descriptor.casefold(): m for m in base.mesh_terms if m.verified}
    mesh = []
    for m in draft.mesh_terms:
        old = verified.get(m.descriptor.casefold())
        mesh.append(
            m.model_copy(update={"verified": True, "verified_at": old.verified_at})
            if old is not None and old.descriptor_ui == m.descriptor_ui
            else m
        )
    validated = {s.sha256: s for s in base.search_strategies if s.syntax_validated}
    strategies = []
    for s in draft.search_strategies:
        old_s = validated.get(s.sha256)
        strategies.append(
            s.model_copy(update={k: getattr(old_s, k) for k in STRATEGY_VALIDATION_KEYS})
            if old_s is not None
            else s
        )
    return PICOProtocol.model_validate(
        draft.model_dump() | {"mesh_terms": mesh, "search_strategies": strategies}
    )


def _check_phase(phase: Phase, changes: Mapping[str, list[str]]) -> None:
    refused = [g for g in changes if phase not in ALLOWED_PHASES[g]]
    if refused:
        raise AmendmentError(
            f"not supported in {phase}: "
            + "; ".join(
                f"{g} changes ({', '.join(changes[g])}) — {PHASE_REASON[g]}" for g in refused
            )
        )


def _store(repo: Repository, project_id: str, p: AmendmentProposal) -> AmendmentProposal:
    repo.add_governance_record(
        project_id, "amendment_proposal", p.amendment_id, p.model_dump(mode="json"),
        reviewer_id=p.reviewer_id,
    )  # fmt: skip
    return p


def propose(
    repo: Repository,
    project_id: str,
    data: Mapping[str, Any],
    *,
    rationale: str,
    reviewer: Reviewer,
    expected_base_sha256: str,
) -> AmendmentProposal:
    """Store a proposal (not active). Bound to the approved protocol the reviewer saw."""
    active = repo.require_active_reviewer(reviewer.id)
    repo.lock_project(project_id)
    phase = Phase(repo.get_project(project_id).phase)
    if phase in (Phase.STAGE1_PROTOCOL, Phase.GATE1_SEARCH_STRATEGY):
        raise AmendmentError("before Gate 1 the protocol is edited as a draft, not amended")
    if phase == Phase.COMPLETED:
        raise AmendmentError("the review is completed; amendments are not supported")
    base = _approved_protocol(repo, project_id)
    if protocol_sha256(base) != expected_base_sha256:
        raise AmendmentError(
            "the approved protocol has changed since it was displayed; reload before proposing"
        )
    if open_amendment(repo, project_id) is not None:
        raise AmendmentError("another amendment is open: decide or withdraw it first")
    proposed = build_proposed(base, data)
    changes = material_changes(base, proposed)
    if not changes:
        raise AmendmentError("nothing changed: the proposal equals the approved protocol")
    _check_phase(phase, changes)
    return _store(repo, project_id, AmendmentProposal(
        base_version=base.version, base_sha256=protocol_sha256(base),
        proposed=proposed.model_dump(mode="json"), proposed_sha256=protocol_sha256(proposed),
        rationale=rationale, proposed_by=active.display_name, reviewer_id=active.id,
        phase=phase.value, material_changes=changes, impact=impact(changes),
    ))  # fmt: skip


def _revise(
    repo: Repository,
    project_id: str,
    current: AmendmentProposal,
    proposed: PICOProtocol,
    reviewer: Reviewer,
    note: str,
) -> AmendmentProposal:
    active = repo.require_active_reviewer(reviewer.id)
    return _store(repo, project_id, current.model_copy(update={
        "revision": current.revision + 1, "proposed": proposed.model_dump(mode="json"),
        "proposed_sha256": protocol_sha256(proposed), "proposed_at": utc_now(),
        "revision_note": f"{note} (by {active.display_name})",
    }))  # fmt: skip


def _current(repo: Repository, project_id: str, amendment_id: str, sha: str) -> AmendmentProposal:
    current = open_amendment(repo, project_id)
    if current is None or current.amendment_id != amendment_id:
        raise AmendmentError(f"amendment {amendment_id} is not open")
    if current.sha256 != sha:
        raise AmendmentError(
            "the amendment has changed since it was displayed (a newer revision exists); reload"
        )
    return current


async def verify_mesh(
    repo: Repository,
    project_id: str,
    amendment_id: str,
    checker: Any,
    *,
    reviewer: Reviewer,
    expected_sha256: str,
) -> AmendmentProposal:
    """Verify new MeSH descriptors of the proposal against NLM MeSH (a new revision)."""
    repo.lock_project(project_id)
    current = _current(repo, project_id, amendment_id, expected_sha256)
    proposed = current.protocol
    terms, missing = [], []
    for m in proposed.mesh_terms:
        if not m.verified and await checker.mesh_descriptor_exists(m.descriptor):
            m = m.model_copy(update={"verified": True, "verified_at": utc_now()})
        elif not m.verified:
            missing.append(m.descriptor)
        terms.append(m)
    updated = PICOProtocol.model_validate(proposed.model_dump() | {"mesh_terms": terms})
    note = "MeSH verified" + (f"; not found: {', '.join(missing)}" if missing else "")
    return _revise(repo, project_id, current, updated, reviewer, note)


def record_strategy_validation(
    repo: Repository,
    project_id: str,
    amendment_id: str,
    strategy_sha256: str,
    *,
    method: Literal["api_preview", "api_accepted", "human_attestation"],
    reviewer: Reviewer,
    expected_sha256: str,
    note: str | None = None,
) -> AmendmentProposal:
    current = _current(repo, project_id, amendment_id, expected_sha256)
    proposed = current.protocol
    if proposed.strategy(strategy_sha256) is None:
        raise AmendmentError(f"no strategy {strategy_sha256[:12]} in the amendment")
    active = repo.require_active_reviewer(reviewer.id)
    strategies = [
        s.model_copy(update={"syntax_validated": True, "validation_method": method,
                             "validated_by": active.display_name, "validated_at": utc_now(),
                             "validation_note": note})
        if s.sha256 == strategy_sha256 else s
        for s in proposed.search_strategies
    ]  # fmt: skip
    updated = PICOProtocol.model_validate(proposed.model_dump() | {"search_strategies": strategies})
    return _revise(repo, project_id, current, updated, reviewer,
                   f"strategy {strategy_sha256[:12]} validated ({method})")  # fmt: skip


def decide(
    repo: Repository,
    project_id: str,
    amendment_id: str,
    decision: Literal["approved", "rejected", "withdrawn"],
    *,
    reviewer: Reviewer,
    comments: str,
    expected_sha256: str,
) -> AmendmentDecision:
    """Approve (activating protocol version n+1), reject or withdraw the displayed revision."""
    active = repo.require_active_reviewer(reviewer.id)
    if not comments.strip():
        raise AmendmentError("a decision needs comments (the reason)")
    repo.lock_project(project_id)
    current = _current(repo, project_id, amendment_id, expected_sha256)
    base = _approved_protocol(repo, project_id)
    if protocol_sha256(base) != current.base_sha256:
        raise AmendmentError("the approved protocol changed after this amendment was proposed")
    record = AmendmentDecision(
        amendment_id=amendment_id, revision=current.revision, decision=decision,
        decided_by=active.display_name, reviewer_id=active.id, comments=comments,
    )  # fmt: skip
    if decision == "approved":
        phase = Phase(repo.get_project(project_id).phase)
        proposed = current.protocol
        changes = material_changes(base, proposed)
        _check_phase(phase, changes)
        blockers = proposed.approval_blockers()
        if blockers:
            raise AmendmentError("the amended protocol cannot be approved: " + "; ".join(blockers))
        approved = PICOProtocol.model_validate(
            proposed.model_dump()
            | {"status": "approved", "approved_by": active.display_name,
               "approved_at": utc_now(), "approved_reviewer_id": active.id}
        )  # fmt: skip
        obligations = compute_obligations(repo, project_id, base, approved, amendment_id)
        repo.save_protocol(project_id, approved)
        repo.supersede_older_protocols(project_id, approved.version)
        repo.add_protocol_event(
            project_id,
            ProtocolEvent(
                protocol_id=approved.protocol_id, protocol_version=approved.version,
                action="amendment_approved", protocol_sha256=protocol_sha256(approved),
                material_changes=changes,
                invalidated=[f"{o.area}: {o.subject}" for o in obligations],
                actor=active.display_name, actor_type=ActorType.HUMAN, reviewer_id=active.id,
                note=f"amendment {amendment_id} (revision {current.revision}): "
                f"{current.rationale}",
            ),
            approved,
        )  # fmt: skip
        record = record.model_copy(
            update={"protocol_version": approved.version, "obligations": obligations}
        )
    repo.add_governance_record(
        project_id, "amendment_decision", amendment_id, record.model_dump(mode="json"),
        reviewer_id=active.id,
    )  # fmt: skip
    return record


def resolve(
    repo: Repository,
    project_id: str,
    obligation_id: str,
    resolution: Literal["confirmed", "revised", "not_applicable"],
    *,
    reviewer: Reviewer,
    note: str,
    verdict: ScreeningVerdict | None = None,
    exclusion_reason: str | None = None,
    expected_open: bool = True,
) -> ObligationResolution:
    """A human re-review of one obligation. A *revised* screening verdict becomes the
    effective human decision for that record and stage (the earlier decisions stay stored)."""
    active = repo.require_active_reviewer(reviewer.id)
    repo.lock_project(project_id)
    row = next(
        (o for o in obligation_status(repo, project_id) if o["obligation_id"] == obligation_id),
        None,
    )
    if row is None:
        raise AmendmentError(f"no obligation {obligation_id}")
    if expected_open and row["state"] != "open":
        raise AmendmentError(
            f"obligation {obligation_id} has changed since it was displayed (now {row['state']})"
        )
    screening = row["area"] in ("title_abstract_screening", "full_text_screening")
    if resolution == "revised" and screening:
        if verdict not in (ScreeningVerdict.INCLUDE, ScreeningVerdict.EXCLUDE):
            raise AmendmentError("a revised screening decision must include or exclude")
        if verdict == ScreeningVerdict.EXCLUDE and not (exclusion_reason or "").strip():
            raise AmendmentError("an exclusion needs a reason")
        if verdict.value == row["previous"]:
            raise AmendmentError("the verdict is unchanged: resolve as 'confirmed'")
    elif verdict is not None:
        raise AmendmentError("a verdict is recorded only for a revised screening decision")
    if resolution == "not_applicable" and screening:
        raise AmendmentError("a screening re-review is confirmed or revised")
    if row["area"] == "analysis_plans" and resolution == "confirmed":
        from src.stats_engine.plan_service import plan_problems

        plan = repo.latest_analysis_plan(project_id)
        problems = plan_problems(repo, project_id, plan.analyses) if plan else ["no plan"]
        if problems:
            raise AmendmentError(
                "the approved plan contradicts the amended protocol: " + "; ".join(problems)
            )
    res = ObligationResolution(
        obligation_id=obligation_id, amendment_id=row["amendment_id"], area=row["area"],
        subject=row["subject"], stage=row["stage"], resolution=resolution,
        verdict=verdict, exclusion_reason=exclusion_reason, note=note,
        resolved_by=active.display_name, reviewer_id=active.id,
    )  # fmt: skip
    repo.add_governance_record(
        project_id, "amendment_resolution", obligation_id, res.model_dump(mode="json"),
        reviewer_id=active.id,
    )  # fmt: skip
    return res


def state(repo: Repository, project_id: str) -> dict[str, Any]:
    """Everything the amendment screen shows, with the hashes a decision must send back."""
    base = repo.latest_protocol(project_id)
    pending = open_amendment(repo, project_id)
    preview: dict[str, Any] | None = None
    if pending is not None and base is not None:
        proposed = pending.protocol
        preview = {
            "proposal": pending.model_dump(mode="json"),
            "sha256": pending.sha256,
            "approval_blockers": proposed.approval_blockers(),
            "obligations_if_approved": [
                o.model_dump(mode="json")
                for o in compute_obligations(repo, project_id, base, proposed, pending.amendment_id)
            ],
        }
    return {
        "approved_protocol": base.model_dump(mode="json") if base else None,
        "approved_sha256": protocol_sha256(base) if base else None,
        "open": preview,
        "history": [p.model_dump(mode="json") for p in proposals(repo, project_id)],
        "decisions": [d.model_dump(mode="json") for d in decisions(repo, project_id)],
        "obligations": obligation_status(repo, project_id),
        "problems": amendment_problems(repo, project_id),
        "impact_rules": {g: list(a) for g, a in IMPACT.items()},
        "phase_rules": PHASE_REASON,
    }
