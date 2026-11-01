"""Human-reviewed Master Extraction Table: propose → verify → review → assemble arms.

Every state change is made on behalf of an authenticated, enrolled reviewer and leaves an
append-only :class:`ExtractionReviewEvent`. Reviewed items are never edited: a correction
supersedes the old item with a new, re-verified version. Accepted, verified numeric items
are assembled into :class:`ExtractedStudyArm` rows (one per report, outcome, arm,
timepoint and estimator — crude counts and Kaplan-Meier estimates are never merged into
one estimate) and approved by the reviewer, which is what Gate 3 checks.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import defaultdict
from collections.abc import Iterator, Sequence
from typing import Any, Literal

from src.core.enums import (
    ActorType,
    ComparisonType,
    DataStatus,
    EffectMeasure,
    EstimateType,
    EvidenceCheckStatus,
    Phase,
    ReviewStatus,
    ScreeningStage,
)
from src.core.exceptions import SysRevError
from src.core.schemas.base import utc_now
from src.core.schemas.extraction import ExtractedStudyArm, RiskOfBiasAssessment
from src.core.schemas.master import (
    ARM_DOMAINS,
    CohortContributionDecision,
    EvidenceCheck,
    ExtractionItem,
    ExtractionReviewEvent,
    QuantitativeUnavailability,
    StudyReportLink,
)
from src.core.schemas.protocol import OutcomeDefinition
from src.extraction.proposals import ProposedValue, proposal_to_item
from src.pdf_extractor.grounding import verify_evidence
from src.pdf_extractor.models import ParsedDocument
from src.state.db import ExtractedArmORM, ReviewerORM
from src.state.repository import Repository

EventAction = Literal[
    "proposed", "accepted", "rejected", "corrected", "arms_built", "arms_invalidated"
]
# Corrections of already reviewed values are possible while Gate 3 can still enforce that arms
# are rebuilt and re-approved; later phases have no route back to Stage 4.
REVISION_PHASES = frozenset({Phase.STAGE4_EXTRACTION_ROB, Phase.GATE3_EXTRACTION_APPROVAL})


class ReviewError(SysRevError, ValueError):
    """A review action is not allowed in the item's current state."""


class StaleStateError(ReviewError):
    """The reviewer decided on a screen whose state has changed since it was displayed."""


def state_sha256(data: Any) -> str:
    """Hash of a displayed state (canonical JSON) that a decision is bound to."""
    text = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      default=str)  # fmt: skip
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def item_state_sha256(item: ExtractionItem) -> str:
    """Hash of the item alone (its content and review status)."""
    return state_sha256(item.model_dump(mode="json"))


def source_state(repo: Repository, project_id: str, record_id: str) -> dict[str, Any]:
    """The evidence source a correction screen shows: the report's linked PDF (content hash,
    stored filename) and the state of its identity check and human confirmation."""
    ft = repo.get_fulltext_item(project_id, record_id)
    if ft is None:
        return {"record_id": record_id, "linked_pdf": None}
    return {
        "record_id": record_id,
        "pdf_filename": ft.pdf_filename,
        "pdf_sha256": ft.pdf_sha256,
        "match_status": ft.match_status.value if ft.match_status else None,
        "match_notes": list(ft.match_notes),
        "confirmed_by_reviewer_id": ft.confirmed_by_reviewer_id,
        "match_unconfirmed": ft.match_unconfirmed,
    }


def correction_state_sha256(
    repo: Repository,
    project_id: str,
    item: ExtractionItem,
    source: dict[str, Any] | None = None,
) -> str:
    """What a correction decision is bound to (review of PR #13, finding 1): the item *and*
    the linked PDF and identity confirmation displayed with it. A replaced PDF, a changed
    identity check or a changed confirmation therefore makes an older screen stale.

    A screen must pass the ``source`` it displays (follow-up review, finding 1): the hash is
    then computed from exactly that captured state, never from a second read that another
    session could have changed in between. Without ``source`` the current state is read."""
    shown = source if source is not None else source_state(repo, project_id, item.record_id)
    if shown.get("record_id") != item.record_id:
        raise ValueError("the source shown belongs to another report")
    return state_sha256({"item": item.model_dump(mode="json"), "source": shown})


def check_expected(actual: str, expected: str | None, what: str) -> None:
    """Refuse (nothing saved) when the state shown to the reviewer is no longer current.
    ``expected=None`` means the caller sent no displayed state (legacy API use)."""
    if expected is not None and expected != actual:
        raise StaleStateError(
            f"{what} has changed since it was displayed; nothing was saved — reload, review "
            "the current state, then decide"
        )


def _require_linked_document(repo: Repository, project_id: str, record_id: str,
                             doc: ParsedDocument) -> None:  # fmt: skip
    item = repo.get_fulltext_item(project_id, record_id)
    if item is None or item.pdf_sha256 is None:
        raise ReviewError(f"record {record_id} has no linked PDF")
    if item.pdf_sha256 != doc.pdf_sha256 or item.pdf_filename != doc.pdf_filename:
        raise ReviewError(
            f"document {doc.pdf_filename} ({doc.pdf_sha256[:12]}…) is not the PDF linked to "
            f"record {record_id} ({item.pdf_filename}, {item.pdf_sha256[:12]}…)"
        )
    if item.match_unconfirmed:
        raise ReviewError(f"the PDF linked to {record_id} has an unconfirmed identity")


def require_current_source(repo: Repository, project_id: str, item: ExtractionItem) -> None:
    """Session 9 (H3): an item may only be accepted against the report's *current*, confirmed
    PDF. The evidence must cite that PDF (hash and stored filename) and the machine check must
    have been made against it; a successful check of an earlier, replaced PDF never authorises
    acceptance. Correct the item against the linked PDF instead."""
    if item.evidence is None:
        return
    ft = repo.get_fulltext_item(project_id, item.record_id)
    if ft is None or ft.pdf_sha256 is None:
        raise ReviewError(f"record {item.record_id} has no linked PDF; the evidence cannot be "
                          "checked against the authoritative source")  # fmt: skip
    if ft.match_unconfirmed:
        raise ReviewError(f"the PDF linked to {item.record_id} has an unconfirmed identity")
    ev = item.evidence
    if ev.pdf_sha256 != ft.pdf_sha256 or ev.pdf_filename != ft.pdf_filename:
        raise ReviewError(
            f"the evidence cites {ev.pdf_filename} ({str(ev.pdf_sha256)[:12]}…) but the PDF now "
            f"linked to {item.record_id} is {ft.pdf_filename} ({ft.pdf_sha256[:12]}…): the "
            "source was replaced; correct the item against the linked PDF (nothing was saved)"
        )
    check = item.verification
    if check is not None and check.checked_against_sha256 not in (None, ft.pdf_sha256):
        raise ReviewError(
            "the evidence check was made against another PDF than the one now linked; "
            "re-verify by correcting the item (nothing was saved)"
        )


def arm_basis(repo: Repository, project_id: str, arm_id: str) -> dict[str, str] | None:
    """The accepted item versions an arm represents (``item_id -> item state hash``).

    Session 9 (H1): recorded when the arm is built (``arm_basis``). For arms built before
    Session 9 it is reconstructed from the append-only ``arms_built`` review events (item
    ids only, hash ``""``). ``None`` for arms that were never built from master-table items
    (e.g. stored directly through the repository)."""
    records = repo.governance_records(project_id, "arm_basis", arm_id)
    if records:
        items = records[-1].get("items")
        return {str(k): str(v) for k, v in items.items()} if isinstance(items, dict) else {}
    built = {
        e.item_id
        for e in repo.list_review_events(project_id)
        if e.action == "arms_built" and e.note == f"arm {arm_id}"
    }
    return dict.fromkeys(built, "") if built else None


def _arm_items(item: ExtractionItem) -> bool:
    return (
        item.domain in ARM_DOMAINS
        and item.review_status == ReviewStatus.ACCEPTED
        and item.data_status == DataStatus.REPORTED
        and item.value_kind == "numeric"
    )


def _invalidate_if_unrepresented(
    repo: Repository, project_id: str, item: ExtractionItem, reviewer: ReviewerORM, why: str
) -> list[str]:
    """Session 9 (H1): a newly accepted arm-level value that the current arm does not represent
    makes that arm stale. The arm is invalidated (kept as history) so that Gate 3 requires the
    arms of the report to be rebuilt from the current accepted values and approved again."""
    if not _arm_items(item):
        return []
    arm_id = current_arm_id(repo, project_id, item.arm_key)
    if arm_id is None:
        return []
    basis = arm_basis(repo, project_id, arm_id)
    if basis is not None and item.item_id in basis:
        return []
    repo.invalidate_arm(
        project_id, arm_id, reviewer_id=reviewer.id, item_id=item.item_id,
        reason=f"{why}: {item.field_name} = {item.reported_text!r} is not represented by the arm",
        kind="stale_basis",
    )  # fmt: skip
    _event(repo, project_id, item, "arms_invalidated", reviewer,
           f"arm {arm_id} invalidated ({why}); rebuild the arms of {item.record_id}",
           None)  # fmt: skip
    return [arm_id]


def arm_representation_problems(repo: Repository, project_id: str, included: set[str]) -> list[str]:
    """Session 9 (H1): accepted arm-level values of included reports that no valid arm
    represents, and arms whose recorded basis is no longer the set of accepted values."""
    excluded = {(u.record_id, u.outcome_id) for u in repo.load_unavailability(project_id)}
    groups: dict[tuple[str, str, str, str], list[ExtractionItem]] = defaultdict(list)
    for item in repo.load_extraction_items(project_id):
        if (
            _arm_items(item)
            and item.record_id in included
            and (item.record_id, item.outcome_id) not in excluded
            and (item.record_id, None) not in excluded
        ):
            groups[item.arm_key].append(item)
    problems: list[str] = []
    for key, group in sorted(groups.items()):
        label = f"{key[0]} / {key[1]} / {key[2]} ({key[3].strip('|') or 'no timepoint'})"
        arm_id = current_arm_id(repo, project_id, key)
        if arm_id is None:
            problems.append(
                f"{label}: {len(group)} accepted value(s) are not represented by any approved "
                "arm; build the arms of the report"
            )
            continue
        basis = arm_basis(repo, project_id, arm_id)
        if basis is None:
            continue  # arm not built from master-table items (cannot be compared)
        current = {i.item_id: item_state_sha256(i) for i in group}
        missing = sorted(set(current) - set(basis))
        gone = sorted(set(basis) - set(current))
        changed = sorted(
            i for i in set(current) & set(basis) if basis[i] and basis[i] != current[i]
        )
        if missing or gone or changed:
            problems.append(
                f"{label}: arm {arm_id[:12]}… no longer matches the accepted values "
                f"(not represented: {len(missing)}, no longer accepted: {len(gone)}, "
                f"changed: {len(changed)}); rebuild and approve the arms again"
            )
    return problems


def source_and_format_problems(repo: Repository, project_id: str, included: set[str]) -> list[str]:
    """Session 9 (H3, M5): for included reports, accepted values and active arm values that
    cite a PDF other than the one now linked, the latest approved risk-of-bias assessment
    citing another PDF, and numbers whose printed form has no recorded reading."""
    linked = {
        i.record_id: i.pdf_sha256 for i in repo.load_fulltext_items(project_id) if i.pdf_sha256
    }
    problems: list[str] = []
    stale_items: dict[str, int] = defaultdict(int)
    unresolved: dict[str, int] = defaultdict(int)
    for item in repo.load_extraction_items(project_id):
        if item.record_id not in included or item.review_status != ReviewStatus.ACCEPTED:
            continue
        ev = item.evidence
        if ev is not None and ev.pdf_sha256 != linked.get(item.record_id):
            stale_items[item.record_id] += 1
        if item.format_status in ("ambiguous_uninterpreted", "legacy_reading_differs"):
            unresolved[item.record_id] += 1
    stale_arms: dict[str, int] = defaultdict(int)
    for arm in repo.load_extracted_arms(project_id):
        if arm.record_id not in included:
            continue
        for value in arm.extracted_values().values():
            if value.evidence.pdf_sha256 != linked.get(arm.record_id):
                stale_arms[arm.record_id] += 1
            if value.format_status in ("ambiguous_uninterpreted", "legacy_reading_differs"):
                unresolved[arm.record_id] += 1
    latest_rob: dict[str, RiskOfBiasAssessment] = {}
    for a in repo.load_rob(project_id):
        if a.approved_by_human and a.approved_at is not None:
            prev = latest_rob.get(a.record_id)
            if prev is None or (prev.approved_at is not None and a.approved_at > prev.approved_at):
                latest_rob[a.record_id] = a
    for rid in sorted(included):
        if stale_items[rid]:
            problems.append(
                f"{rid}: {stale_items[rid]} accepted value(s) cite a PDF that is no longer the "
                "linked one; re-verify them against the linked PDF (correct each item)"
            )
        if stale_arms[rid]:
            problems.append(
                f"{rid}: {stale_arms[rid]} value(s) of an approved arm cite a PDF that is no "
                "longer linked; rebuild the arms from re-verified values"
            )
        rob = latest_rob.get(rid)
        if rob is not None and any(
            e.pdf_sha256 is not None and e.pdf_sha256 != linked.get(rid)
            for d in rob.domains
            for e in d.evidence
        ):
            problems.append(
                f"{rid}: the approved risk-of-bias assessment cites a PDF that is no longer "
                "linked; a new assessment must be approved"
            )
        if unresolved[rid]:
            problems.append(
                f"{rid}: {unresolved[rid]} number(s) have an ambiguous printed format without a "
                "recorded reading, or were stored with the reading used before Session 9; "
                "correct them with number_format and a note citing the source context"
            )
    return problems


def submit_proposals(
    repo: Repository,
    project_id: str,
    record_id: str,
    doc: ParsedDocument,
    proposals: Sequence[ProposedValue],
    *,
    proposed_by: ActorType,
    proposer_identity: str,
) -> list[ExtractionItem]:
    """Verify proposals against the record's linked PDF and store them as pending items."""
    _require_linked_document(repo, project_id, record_id, doc)
    items: list[ExtractionItem] = []
    for proposal in proposals:
        item = proposal_to_item(
            proposal, doc, record_id=record_id, proposed_by=proposed_by,
            proposer_identity=proposer_identity,
        )  # fmt: skip
        repo.save_extraction_item(project_id, item)
        repo.add_review_event(
            project_id,
            ExtractionReviewEvent(
                item_id=item.item_id, action="proposed", actor_type=proposed_by,
                actor=proposer_identity, after=item.model_dump(mode="json"),
            ),
        )  # fmt: skip
        items.append(item)
    return items


def _pending(repo: Repository, project_id: str, item_id: str) -> ExtractionItem:
    item = repo.get_extraction_item(project_id, item_id)
    if item.review_status != ReviewStatus.PENDING:
        raise ReviewError(f"item {item_id} is already {item.review_status}")
    return item


def _stamped(item: ExtractionItem, status: ReviewStatus, reviewer: ReviewerORM,
             note: str | None) -> ExtractionItem:  # fmt: skip
    return ExtractionItem.model_validate(
        item.model_dump()
        | {
            "review_status": status,
            "reviewed_by": reviewer.display_name,
            "reviewer_id": reviewer.id,
            "reviewed_at": utc_now(),
            "review_note": note,
        }
    )


def _event(
    repo: Repository,
    project_id: str,
    item: ExtractionItem,
    action: EventAction,
    reviewer: ReviewerORM,
    note: str | None,
    before: ExtractionItem | None,
) -> None:
    repo.add_review_event(
        project_id,
        ExtractionReviewEvent(
            item_id=item.item_id, action=action, actor_type=ActorType.HUMAN,
            actor=reviewer.display_name, reviewer_id=reviewer.id, note=note,
            before=before.model_dump(mode="json") if before else None,
            after=item.model_dump(mode="json"),
        ),
    )  # fmt: skip


def accept_item(repo: Repository, project_id: str, item_id: str, *, reviewer: ReviewerORM,
                note: str | None = None) -> ExtractionItem:  # fmt: skip
    reviewer = repo.require_active_reviewer(reviewer.id)
    item = _pending(repo, project_id, item_id)
    if item.data_status == DataStatus.REPORTED and not (
        item.verification and item.verification.acceptable
    ):
        raise ReviewError(
            "evidence did not verify against the document; correct the item (with grounded "
            "evidence) or reject it"
        )
    if item.flags and not (note and note.strip()):
        raise ReviewError("flagged items can only be accepted with a note explaining why")
    _require_resolved_number(item)
    repo.require_gate3_open(project_id, "accepting an extraction item")
    require_current_source(repo, project_id, item)
    accepted = _stamped(item, ReviewStatus.ACCEPTED, reviewer, note)
    repo.save_extraction_item(project_id, accepted)
    _event(repo, project_id, accepted, "accepted", reviewer, note, item)
    _invalidate_if_unrepresented(repo, project_id, accepted, reviewer,
                                 "value accepted after the arm was built")  # fmt: skip
    return accepted


def _require_resolved_number(item: ExtractionItem) -> None:
    """Session 9 (M5): an uninterpreted ambiguous number, or a value stored with the reading
    used before Session 9 that differs from the current one, is never accepted or built."""
    status = item.format_status
    if status in ("ambiguous_uninterpreted", "legacy_reading_differs"):
        raise ReviewError(
            f"{item.reported_text!r} has no recorded number format ({status}); correct the item "
            "with number_format and a note citing the source context"
        )


def reject_item(repo: Repository, project_id: str, item_id: str, *, reviewer: ReviewerORM,
                note: str) -> ExtractionItem:  # fmt: skip
    reviewer = repo.require_active_reviewer(reviewer.id)
    if not note.strip():
        raise ReviewError("a rejection needs a reason")
    item = _pending(repo, project_id, item_id)
    rejected = _stamped(item, ReviewStatus.REJECTED, reviewer, note)
    repo.save_extraction_item(project_id, rejected)
    _event(repo, project_id, rejected, "rejected", reviewer, note, item)
    return rejected


def correct_item(
    repo: Repository,
    project_id: str,
    item_id: str,
    correction: ProposedValue,
    doc: ParsedDocument,
    *,
    reviewer: ReviewerORM,
    note: str,
    attest_unverifiable: bool = False,
    expected_state_sha256: str | None = None,
) -> ExtractionItem:
    """Supersede a pending item with a human-entered, re-verified version.

    The correction must itself be grounded in ``doc``. Only when the source cannot be
    machine-checked (e.g. an image-only page or a figure) may the reviewer attest it, and
    the item then records ``HUMAN_ATTESTED`` together with the failed check.
    """
    reviewer = repo.require_active_reviewer(reviewer.id)
    if not note.strip():
        raise ReviewError("a correction needs a note")
    repo.lock_project(project_id)
    check_expected(
        correction_state_sha256(repo, project_id, repo.get_extraction_item(project_id, item_id)),
        expected_state_sha256,
        f"item {item_id} or its linked PDF",
    )
    old = _pending(repo, project_id, item_id)
    _require_linked_document(repo, project_id, old.record_id, doc)
    new = _grounded_correction(old, correction, doc, reviewer, attest_unverifiable)
    superseded = _stamped(old, ReviewStatus.SUPERSEDED, reviewer, note)
    repo.save_extraction_item(project_id, superseded)
    accepted = _stamped(new, ReviewStatus.ACCEPTED, reviewer, note)
    repo.save_extraction_item(project_id, accepted)
    _event(repo, project_id, superseded, "corrected", reviewer, note, old)
    _event(repo, project_id, accepted, "accepted", reviewer, note, None)
    _invalidate_if_unrepresented(repo, project_id, accepted, reviewer,
                                 "corrected value accepted after the arm was built")  # fmt: skip
    return accepted


def _grounded_correction(
    old: ExtractionItem,
    correction: ProposedValue,
    doc: ParsedDocument,
    reviewer: ReviewerORM,
    attest_unverifiable: bool,
) -> ExtractionItem:
    """The corrected item, grounded in ``doc`` (or explicitly attested when the source cannot
    be machine-checked, keeping the failed check on record)."""
    new = proposal_to_item(
        correction, doc, record_id=old.record_id, proposed_by=ActorType.HUMAN,
        proposer_identity=reviewer.display_name, item_version=old.version + 1,
        supersedes=old.item_id,
    )  # fmt: skip
    if new.data_status.value != correction.data_status:
        # Grounding downgraded the reviewer's entry (e.g. the number is not in the quote):
        # refuse instead of silently storing something other than what was submitted.
        raise ReviewError(
            f"the correction was entered as {correction.data_status!r} but cannot be stored as "
            f"such ({'; '.join(new.flags) or new.data_status.value}); fix the entry"
        )
    check = new.verification
    if new.data_status == DataStatus.REPORTED and not (check and check.acceptable):
        if not attest_unverifiable:
            raise ReviewError(
                f"correction does not verify ({check.status.value if check else 'no evidence'}"
                f"{': ' + check.detail if check else ''}); fix it or attest explicitly"
            )
        assert check is not None
        new = ExtractionItem.model_validate(
            new.model_dump()
            | {
                "verification": EvidenceCheck(
                    status=EvidenceCheckStatus.HUMAN_ATTESTED,
                    detail=f"attested by {reviewer.display_name}; machine check: "
                    f"{check.status.value} ({check.detail})",
                    checked_against_sha256=doc.pdf_sha256,
                ),
                "flags": [*new.flags, "human-attested value (not machine-verifiable)"],
            }
        )
    return new


def _arm_id_chain(key: tuple[str, ...]) -> Iterator[str]:
    """Arm ids for one arm key: the original id, then one per rebuild after invalidation."""
    yield _arm_id(key)
    n = 1
    while True:
        yield _arm_id((*key, f"rebuild-{n}"))
        n += 1


def current_arm_id(repo: Repository, project_id: str, key: tuple[str, ...]) -> str | None:
    """The valid (not invalidated) arm built for ``key``, if any."""
    invalid = repo.invalidated_arm_ids(project_id)
    for arm_id in _arm_id_chain(key):
        if repo.session.get(ExtractedArmORM, arm_id) is None:
            return None
        if arm_id not in invalid:
            return arm_id
    return None  # pragma: no cover - the chain is unbounded but rows are finite


def _next_arm_id(repo: Repository, project_id: str, key: tuple[str, ...]) -> str | None:
    """A fresh id for building ``key`` (None: a valid arm already exists)."""
    invalid = repo.invalidated_arm_ids(project_id)
    for arm_id in _arm_id_chain(key):
        if repo.session.get(ExtractedArmORM, arm_id) is None:
            return arm_id
        if arm_id not in invalid:
            return None
    return None  # pragma: no cover


def revise_reviewed_item(
    repo: Repository,
    project_id: str,
    item_id: str,
    correction: ProposedValue,
    doc: ParsedDocument,
    *,
    reviewer: ReviewerORM,
    note: str,
    expected_state_sha256: str,
    attest_unverifiable: bool = False,
) -> tuple[ExtractionItem, list[str]]:
    """Correct an item that was already accepted or rejected (Session 8).

    The reviewed item is superseded (its earlier content stays in the append-only review
    history), the correction is grounded in the linked PDF (or attested) and accepted by the
    correcting reviewer, and any arm built from the old value is **invalidated**: Gate 3 then
    requires the arms of that report to be rebuilt and approved, and statistics, GRADE and
    syntheses computed from the old arm become stale through their input hashes. Returns the
    new item and the invalidated arm ids. Allowed in Stage 4 and Gate 3 only.
    """
    reviewer = repo.require_active_reviewer(reviewer.id)
    if not note.strip():
        raise ReviewError("a correction needs a note")
    repo.lock_project(project_id)
    phase = Phase(repo.current_phase(project_id))
    if phase not in REVISION_PHASES:
        raise ReviewError(
            f"reviewed values can be corrected in Stage 4 or at Gate 3 (project is in {phase}); "
            "later phases have no route back to rebuild and re-approve the arms"
        )
    old = repo.get_extraction_item(project_id, item_id)
    check_expected(
        correction_state_sha256(repo, project_id, old),
        expected_state_sha256,
        f"item {item_id} or its linked PDF",
    )
    if old.review_status == ReviewStatus.PENDING:
        raise ReviewError(f"item {item_id} is pending: use the ordinary correction")
    if old.review_status not in (ReviewStatus.ACCEPTED, ReviewStatus.REJECTED):
        raise ReviewError(f"item {item_id} is {old.review_status}; correct its current version")
    _require_linked_document(repo, project_id, old.record_id, doc)
    new = _grounded_correction(old, correction, doc, reviewer, attest_unverifiable)
    invalidated: list[str] = []
    if old.review_status == ReviewStatus.ACCEPTED and old.domain in ARM_DOMAINS:
        arm_id = current_arm_id(repo, project_id, old.arm_key)
        if arm_id is not None:
            repo.invalidate_arm(
                project_id, arm_id, reviewer_id=reviewer.id, item_id=old.item_id,
                reason=f"value corrected after acceptance: {note}",
            )  # fmt: skip
            invalidated.append(arm_id)
    superseded = _stamped(old, ReviewStatus.SUPERSEDED, reviewer, note)
    repo.supersede_reviewed_item(project_id, superseded)
    accepted = _stamped(new, ReviewStatus.ACCEPTED, reviewer, note)
    repo.save_extraction_item(project_id, accepted)
    _event(repo, project_id, superseded, "corrected", reviewer, note, old)
    _event(repo, project_id, accepted, "accepted", reviewer, note, None)
    if not invalidated:
        # Session 9 (H1): a REJECTED value corrected and accepted (or a value moved to another
        # arm) is not represented by the arm built without it.
        invalidated += _invalidate_if_unrepresented(
            repo, project_id, accepted, reviewer, "corrected value accepted after the arm was built"
        )
    if invalidated and old.review_status == ReviewStatus.ACCEPTED:
        _event(repo, project_id, superseded, "arms_invalidated", reviewer,
               f"arm(s) {', '.join(invalidated)} invalidated; rebuild the arms of "
               f"{old.record_id}", None)  # fmt: skip
    return accepted, invalidated


def reverify_items(
    repo: Repository, project_id: str, record_id: str, doc: ParsedDocument
) -> list[tuple[str, EvidenceCheck]]:
    """Re-check stored evidence against ``doc`` (e.g. after a PDF was replaced).

    Returns ``(item_id, check)`` for every item whose evidence no longer verifies.
    """
    failures: list[tuple[str, EvidenceCheck]] = []
    for item in repo.load_extraction_items(project_id):
        if item.record_id != record_id or item.evidence is None:
            continue
        if item.verification and item.verification.status == EvidenceCheckStatus.HUMAN_ATTESTED:
            continue
        check = verify_evidence(
            item.evidence, doc,
            reported_text=item.reported_text if item.value is not None else None,
        )  # fmt: skip
        if not check.acceptable:
            failures.append((item.item_id, check))
    return failures


# --------------------------------------------------------------------------- arms


def _arm_id(key: tuple[str, ...]) -> str:
    return hashlib.sha256("\x1f".join(key).encode()).hexdigest()[:32]


def build_arms(
    repo: Repository, project_id: str, record_id: str, *, reviewer: ReviewerORM
) -> list[ExtractedStudyArm]:
    """Assemble accepted, verified arm-level items into approved ExtractedStudyArms."""
    reviewer = repo.require_active_reviewer(reviewer.id)
    if record_id in repo.withdrawn_arm_records(project_id) and record_id not in (
        repo.included_record_ids(project_id, ScreeningStage.FULL_TEXT)
    ):
        raise ReviewError(
            f"the arms of {record_id} were withdrawn because the report is not included at "
            "full text; they are rebuilt only if it is included again"
        )
    items = [
        i
        for i in repo.load_extraction_items(project_id)
        if i.record_id == record_id
        and i.domain in ARM_DOMAINS
        and i.review_status == ReviewStatus.ACCEPTED
        and i.data_status == DataStatus.REPORTED
        and i.value_kind == "numeric"
    ]
    if any(
        i.review_status == ReviewStatus.PENDING
        for i in repo.load_extraction_items(project_id)
        if i.record_id == record_id
    ):
        raise ReviewError(f"record {record_id} still has pending extraction items")
    repo.require_gate3_open(project_id, "building approved arms")
    for item in items:
        # Session 9 (H3, M5): every value must rest on the current, confirmed PDF and have
        # a resolved number format before it can enter an approved arm.
        require_current_source(repo, project_id, item)
        _require_resolved_number(item)
    groups: dict[tuple[str, str, str, str], list[ExtractionItem]] = defaultdict(list)
    for item in items:
        groups[item.arm_key].append(item)
    built: list[ExtractedStudyArm] = []
    for key, group in sorted(groups.items()):
        next_id = _next_arm_id(repo, project_id, key)
        if next_id is None:  # a valid arm was already built from these items
            continue
        arm_id = next_id
        fields: dict[str, Any] = {}
        for item in group:
            if item.field_name in fields:
                raise ReviewError(f"two accepted values for {item.field_name} in arm {key}")
            fields[item.field_name] = item.to_extracted_value()
        first = group[0]
        # arm coordinates are guaranteed by the ExtractionItem schema for arm domains
        assert first.outcome_id is not None
        assert first.arm_label is not None
        assert first.arm_role is not None
        for attr in ("arm_role", "follow_up", "estimate_scale", "ci_level"):
            values = {json.dumps(getattr(i, attr), default=str) for i in group
                      if getattr(i, attr) is not None}  # fmt: skip
            if len(values) > 1:
                raise ReviewError(f"inconsistent {attr} across items of arm {key}")
        has_estimate = "reported_estimate" in fields
        arm = ExtractedStudyArm(
            arm_id=arm_id,
            record_id=record_id,
            outcome_id=first.outcome_id,
            arm_label=first.arm_label,
            arm_role=first.arm_role,
            timepoint=first.timepoint,
            estimate_type=first.estimate_type if has_estimate else None,
            estimate_scale=next((i.estimate_scale for i in group if i.estimate_scale), None)
            if has_estimate else None,
            ci_level=next((i.ci_level for i in group if i.ci_level), None),
            follow_up=next((i.follow_up for i in group if i.follow_up), None),
            approved_by_human=True,
            approved_by=reviewer.display_name,
            approved_at=utc_now(),
            **fields,
        )  # fmt: skip
        repo.save_extracted_arm(project_id, arm)
        ft = repo.get_fulltext_item(project_id, record_id)
        repo.add_governance_record(
            project_id, "arm_basis", arm_id,
            {"arm_id": arm_id, "record_id": record_id, "arm_key": list(key),
             "items": {i.item_id: item_state_sha256(i) for i in group},
             "pdf_sha256": ft.pdf_sha256 if ft else None},
            reviewer_id=reviewer.id,
        )  # fmt: skip
        for item in group:
            repo.add_review_event(
                project_id,
                ExtractionReviewEvent(
                    item_id=item.item_id, action="arms_built", actor_type=ActorType.HUMAN,
                    actor=reviewer.display_name, reviewer_id=reviewer.id,
                    note=f"arm {arm_id}",
                ),
            )  # fmt: skip
        built.append(arm)
    return built


def analysis_compatibility(arm: ExtractedStudyArm, outcome: OutcomeDefinition) -> tuple[bool, str]:
    """Can ``arm`` inform ``outcome``'s summary measure *without conversion*?

    Crude proportions, Kaplan-Meier/actuarial probabilities and competing-risks cumulative
    incidences are different estimands and are never substituted for one another.
    """
    measure = outcome.preferred_effect_measure
    if outcome.comparison == ComparisonType.SINGLE_GROUP:
        if measure == EffectMeasure.PROPORTION:
            if arm.events is not None and arm.n_analyzed is not None:
                return True, "crude events / n available"
            return False, "crude proportion needs events and n_analyzed (not an actuarial estimate)"
        if measure == EffectMeasure.KM_PROBABILITY:
            ok = arm.estimate_type in (EstimateType.KAPLAN_MEIER, EstimateType.ACTUARIAL_LIFE_TABLE)
            return (ok, "Kaplan-Meier/actuarial estimate available") if ok else (
                False, "needs an author-reported Kaplan-Meier/actuarial estimate at time t"
            )  # fmt: skip
        if measure == EffectMeasure.CUMULATIVE_INCIDENCE:
            ok = arm.estimate_type == EstimateType.CUMULATIVE_INCIDENCE
            return (ok, "cumulative incidence available") if ok else (
                False, "needs an author-reported competing-risks cumulative incidence"
            )  # fmt: skip
    return True, "not restricted at extraction stage; checked by the statistics engine"


# --------------------------------------------------------------------------- studies, export


def arms_state_sha256(repo: Repository, project_id: str, record_id: str) -> str:
    """The active arms of one report and its full-text inclusion, as displayed."""
    return state_sha256(
        {
            "arms": sorted(
                a.arm_id for a in repo.load_extracted_arms(project_id) if a.record_id == record_id
            ),
            "included": record_id in repo.included_record_ids(project_id, ScreeningStage.FULL_TEXT),
        }
    )


def withdraw_arms(
    repo: Repository,
    project_id: str,
    record_id: str,
    *,
    reviewer: ReviewerORM,
    reason: str,
    expected_state_sha256: str,
) -> list[str]:
    """Retire the active arms of a report that is **no longer included** at full text (e.g.
    excluded on re-review after an eligibility amendment). Review of PR #13, finding 4.

    The arms and their evidence stay stored; an append-only ``arm_invalidation`` record of kind
    ``withdrawal`` names the reviewer and reason, and every reader ignores the arms from then
    on. Data of an included report are never withdrawn here (correct them, or record them as
    unavailable for an outcome). Allowed in Stage 4 and at Gate 3, where Gate 3 re-checks the
    extraction state; refused (nothing written) if the displayed state changed.
    """
    reviewer = repo.require_active_reviewer(reviewer.id)
    if not reason.strip():
        raise ReviewError("a withdrawal needs a reason")
    repo.lock_project(project_id)
    phase = Phase(repo.current_phase(project_id))
    if phase not in REVISION_PHASES:
        raise ReviewError(
            f"arms are withdrawn in Stage 4 or at Gate 3 (project is in {phase}); later phases "
            "have no route back to re-check the extraction"
        )
    check_expected(arms_state_sha256(repo, project_id, record_id), expected_state_sha256,
                   f"the arms or inclusion of {record_id}")  # fmt: skip
    if record_id in repo.included_record_ids(project_id, ScreeningStage.FULL_TEXT):
        raise ReviewError(
            f"{record_id} is included at full text: its data are corrected, or recorded as "
            "unavailable for an outcome, not withdrawn"
        )
    arms = [a.arm_id for a in repo.load_extracted_arms(project_id) if a.record_id == record_id]
    if not arms:
        raise ReviewError(f"{record_id} has no active arms")
    for arm_id in arms:
        repo.invalidate_arm(
            project_id, arm_id, reviewer_id=reviewer.id, item_id=None,
            reason=f"withdrawn: report not included at full text — {reason}", kind="withdrawal",
        )  # fmt: skip
    return arms


def link_reports_to_study(
    repo: Repository,
    project_id: str,
    record_ids: Sequence[str],
    *,
    study_id: str,
    primary_record_id: str,
    reviewer: ReviewerORM,
    note: str,
    expected_state_sha256: str | None = None,
) -> list[StudyReportLink]:
    """Declare that several reports describe one study/cohort (human decision). A report
    already linked elsewhere is re-linked; every change is kept in the link history."""
    reviewer = repo.require_active_reviewer(reviewer.id)
    if primary_record_id not in record_ids:
        raise ValueError("the primary report must be one of the linked reports")
    repo.lock_project(project_id)
    check_expected(links_state_sha256(repo, project_id), expected_state_sha256,
                   "the report-to-study links")  # fmt: skip
    before = repo.study_ids(project_id, set(record_ids))
    links = [
        StudyReportLink(
            record_id=rid, study_id=study_id, is_primary_report=rid == primary_record_id,
            linked_by=reviewer.display_name, reviewer_id=reviewer.id, note=note,
        )
        for rid in record_ids
    ]  # fmt: skip
    for link in links:
        repo.save_study_link(project_id, link)
        _link_history(repo, project_id, link.record_id, before[link.record_id], study_id,
                      reviewer, note)  # fmt: skip
    return links


def links_state_sha256(repo: Repository, project_id: str) -> str:
    return state_sha256(
        [(x.record_id, x.study_id, x.is_primary_report) for x in repo.load_study_links(project_id)]
    )


def _link_history(repo: Repository, project_id: str, record_id: str, before: str, after: str,
                  reviewer: ReviewerORM, note: str) -> None:  # fmt: skip
    action = "unlinked" if after == record_id else "linked" if before == record_id else "relinked"
    repo.add_governance_record(
        project_id, "study_link", record_id,
        {"action": action, "record_id": record_id, "study_before": before,
         "study_after": after, "reviewer": reviewer.display_name, "note": note},
        reviewer_id=reviewer.id,
    )  # fmt: skip


def unlink_reports(
    repo: Repository,
    project_id: str,
    record_ids: Sequence[str],
    *,
    reviewer: ReviewerORM,
    note: str,
    expected_state_sha256: str,
) -> None:
    """Undo a report-to-study link (the report becomes its own study again). The history
    keeps the earlier link; statistics, PRISMA study counts and syntheses that used it become
    stale through their input hashes."""
    reviewer = repo.require_active_reviewer(reviewer.id)
    if not note.strip():
        raise ReviewError("unlinking needs a note")
    repo.lock_project(project_id)
    check_expected(links_state_sha256(repo, project_id), expected_state_sha256,
                   "the report-to-study links")  # fmt: skip
    before = repo.study_ids(project_id, set(record_ids))
    for rid in record_ids:
        repo.remove_study_link(project_id, rid, reviewer_id=reviewer.id)
        _link_history(repo, project_id, rid, before[rid], rid, reviewer, note)


def double_counting_state_sha256(repo: Repository, project_id: str) -> str:
    return state_sha256(repo.double_counting_conflicts(project_id))


def record_unavailability(
    repo: Repository,
    project_id: str,
    record_id: str,
    *,
    reviewer: ReviewerORM,
    reason_code: str,
    reason: str,
    outcome_id: str | None = None,
) -> QuantitativeUnavailability:
    """Eligible study without usable data for ``outcome_id`` (None: qualitative only)."""
    reviewer = repo.require_active_reviewer(reviewer.id)
    entry = QuantitativeUnavailability(
        record_id=record_id, outcome_id=outcome_id, reason_code=reason_code,  # type: ignore[arg-type]
        reason=reason, recorded_by=reviewer.display_name, reviewer_id=reviewer.id,
    )  # fmt: skip
    repo.save_unavailability(project_id, entry)
    return entry


def double_counting_risks(repo: Repository, project_id: str) -> list[str]:
    """Same cohort, same outcome/arm/timepoint/estimator, arm data from >1 report.

    These block Gate 3 → Stage 5 (``unresolved_double_counting``) until a human resolves
    them with :func:`resolve_double_counting`.
    """
    return repo.double_counting_conflicts(project_id)


def resolve_double_counting(
    repo: Repository,
    project_id: str,
    *,
    study_id: str,
    outcome_id: str,
    reviewer: ReviewerORM,
    justification: str,
    keep_record_id: str | None = None,
    drop_record_ids: Sequence[str] = (),
    non_overlapping_record_ids: Sequence[str] = (),
    expected_state_sha256: str | None = None,
) -> None:
    """Document how one cohort contributes to ``outcome_id`` (exactly one mode):

    * ``keep_record_id`` + ``drop_record_ids``: the dropped reports are recorded as
      unavailable for this outcome (``overlapping_cohort``), so only one report's patients
      are pooled;
    * ``non_overlapping_record_ids``: a human attests the reports contribute different
      participants (e.g. distinct recruitment periods), so all may be pooled.
    """
    reviewer = repo.require_active_reviewer(reviewer.id)
    if not justification.strip():
        raise ReviewError("a justification is required")
    repo.lock_project(project_id)
    check_expected(double_counting_state_sha256(repo, project_id), expected_state_sha256,
                   "the double-counting conflicts")  # fmt: skip
    if bool(keep_record_id) == bool(non_overlapping_record_ids):
        raise ReviewError("choose either keep_record_id or non_overlapping_record_ids")
    if keep_record_id:
        if not drop_record_ids or keep_record_id in drop_record_ids:
            raise ReviewError("list the redundant reports to drop (not the kept one)")
        for rid in drop_record_ids:
            record_unavailability(
                repo, project_id, rid, reviewer=reviewer, reason_code="overlapping_cohort",
                reason=f"same cohort as {keep_record_id} (study {study_id}): {justification}",
                outcome_id=outcome_id,
            )  # fmt: skip
        return
    repo.save_contribution_decision(
        project_id,
        CohortContributionDecision(
            study_id=study_id, outcome_id=outcome_id,
            record_ids=list(non_overlapping_record_ids), justification=justification,
            decided_by=reviewer.display_name, reviewer_id=reviewer.id,
        ),
    )  # fmt: skip


EXPORT_FIELDS = (
    "study_id", "record_id", "item_id", "version", "supersedes", "domain", "field_name",
    "outcome_id", "arm_label", "arm_role", "timepoint", "estimate_type", "estimate_scale",
    "ci_level", "data_status", "reported_text", "value", "unit", "text_value",
    "pdf_filename", "pdf_sha256", "page", "location_type", "location_id", "verbatim_quote",
    "extracted_by", "extractor_identity", "verification_status", "verification_detail",
    "flags", "review_status", "reviewed_by", "reviewed_at", "review_note",
)  # fmt: skip


def export_rows(
    repo: Repository, project_id: str, *, include_superseded: bool = True
) -> list[dict[str, Any]]:
    items = repo.load_extraction_items(project_id, include_superseded=include_superseded)
    unavailable = repo.load_unavailability(project_id)
    studies = repo.study_ids(
        project_id, {i.record_id for i in items} | {u.record_id for u in unavailable}
    )
    rows: list[dict[str, Any]] = []
    for i in items:
        ev, chk = i.evidence, i.verification
        rows.append({
            "study_id": studies[i.record_id], "record_id": i.record_id, "item_id": i.item_id,
            "version": i.version, "supersedes": i.supersedes, "domain": i.domain.value,
            "field_name": i.field_name, "outcome_id": i.outcome_id, "arm_label": i.arm_label,
            "arm_role": i.arm_role.value if i.arm_role else None, "timepoint": i.timepoint,
            "estimate_type": i.estimate_type.value if i.estimate_type else None,
            "estimate_scale": i.estimate_scale, "ci_level": i.ci_level,
            "data_status": i.data_status.value, "reported_text": i.reported_text,
            "value": i.value, "unit": i.unit, "text_value": i.text_value,
            "pdf_filename": ev.pdf_filename if ev else None,
            "pdf_sha256": ev.pdf_sha256 if ev else None,
            "page": ev.page_number if ev else None,
            "location_type": ev.location_type.value if ev else None,
            "location_id": ev.location_identifier if ev else None,
            "verbatim_quote": ev.verbatim_quote if ev else None,
            "extracted_by": i.proposed_by.value, "extractor_identity": i.proposer_identity,
            "verification_status": chk.status.value if chk else None,
            "verification_detail": chk.detail if chk else None,
            "flags": " | ".join(i.flags), "review_status": i.review_status.value,
            "reviewed_by": i.reviewed_by,
            "reviewed_at": i.reviewed_at.isoformat() if i.reviewed_at else None,
            "review_note": i.review_note,
        })  # fmt: skip
    for u in unavailable:
        row = dict.fromkeys(EXPORT_FIELDS)
        row |= {
            "study_id": studies[u.record_id], "record_id": u.record_id,
            "item_id": u.unavailability_id, "domain": "quantitative_availability",
            "field_name": "unavailable_for_quantitative_synthesis",
            "outcome_id": u.outcome_id or "ALL", "data_status": "not_reported",
            "text_value": f"{u.reason_code}: {u.reason}", "review_status": "accepted",
            "reviewed_by": u.recorded_by, "reviewed_at": u.recorded_at.isoformat(),
        }  # fmt: skip
        rows.append(row)
    return rows


def export_csv(rows: list[dict[str, Any]]) -> str:
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(EXPORT_FIELDS), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def export_json(rows: list[dict[str, Any]]) -> str:
    return json.dumps(rows, indent=2, ensure_ascii=False, default=str)
