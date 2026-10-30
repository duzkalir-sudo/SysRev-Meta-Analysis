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
from collections.abc import Sequence
from typing import Any, Literal

from src.core.enums import (
    ActorType,
    ComparisonType,
    DataStatus,
    EffectMeasure,
    EstimateType,
    EvidenceCheckStatus,
    ReviewStatus,
)
from src.core.exceptions import SysRevError
from src.core.schemas.base import utc_now
from src.core.schemas.extraction import ExtractedStudyArm
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

EventAction = Literal["proposed", "accepted", "rejected", "corrected", "arms_built"]


class ReviewError(SysRevError, ValueError):
    """A review action is not allowed in the item's current state."""


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
    accepted = _stamped(item, ReviewStatus.ACCEPTED, reviewer, note)
    repo.save_extraction_item(project_id, accepted)
    _event(repo, project_id, accepted, "accepted", reviewer, note, item)
    return accepted


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
) -> ExtractionItem:
    """Supersede a pending item with a human-entered, re-verified version.

    The correction must itself be grounded in ``doc``. Only when the source cannot be
    machine-checked (e.g. an image-only page or a figure) may the reviewer attest it, and
    the item then records ``HUMAN_ATTESTED`` together with the failed check.
    """
    reviewer = repo.require_active_reviewer(reviewer.id)
    if not note.strip():
        raise ReviewError("a correction needs a note")
    old = _pending(repo, project_id, item_id)
    _require_linked_document(repo, project_id, old.record_id, doc)
    new = proposal_to_item(
        correction, doc, record_id=old.record_id, proposed_by=ActorType.HUMAN,
        proposer_identity=reviewer.display_name, item_version=old.version + 1,
        supersedes=old.item_id,
    )  # fmt: skip
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
    superseded = _stamped(old, ReviewStatus.SUPERSEDED, reviewer, note)
    repo.save_extraction_item(project_id, superseded)
    accepted = _stamped(new, ReviewStatus.ACCEPTED, reviewer, note)
    repo.save_extraction_item(project_id, accepted)
    _event(repo, project_id, superseded, "corrected", reviewer, note, old)
    _event(repo, project_id, accepted, "accepted", reviewer, note, None)
    return accepted


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
    groups: dict[tuple[str, str, str, str], list[ExtractionItem]] = defaultdict(list)
    for item in items:
        groups[item.arm_key].append(item)
    built: list[ExtractedStudyArm] = []
    for key, group in sorted(groups.items()):
        arm_id = _arm_id(key)
        if repo.session.get(ExtractedArmORM, arm_id) is not None:
            continue
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


def link_reports_to_study(
    repo: Repository,
    project_id: str,
    record_ids: Sequence[str],
    *,
    study_id: str,
    primary_record_id: str,
    reviewer: ReviewerORM,
    note: str,
) -> list[StudyReportLink]:
    """Declare that several reports describe one study/cohort (human decision)."""
    reviewer = repo.require_active_reviewer(reviewer.id)
    if primary_record_id not in record_ids:
        raise ValueError("the primary report must be one of the linked reports")
    links = [
        StudyReportLink(
            record_id=rid, study_id=study_id, is_primary_report=rid == primary_record_id,
            linked_by=reviewer.display_name, reviewer_id=reviewer.id, note=note,
        )
        for rid in record_ids
    ]  # fmt: skip
    for link in links:
        repo.save_study_link(project_id, link)
    return links


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
