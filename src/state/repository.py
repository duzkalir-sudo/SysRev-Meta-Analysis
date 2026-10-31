"""Persistence gateway: converts validated Pydantic contracts to ORM rows and back.

Every load re-validates the stored payload through Pydantic, so corrupted or
hand-edited rows fail loudly instead of propagating into the analysis.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Sequence

from sqlalchemy import Select, func, select, text
from sqlalchemy.orm import Session

from src.core.enums import (
    ActorType,
    CheckStatus,
    FullTextStatus,
    OverlapResolution,
    Phase,
    ReviewerRole,
    ReviewStatus,
    ScreeningStage,
    ScreeningVerdict,
    SearchRunKind,
    UploadMatchStatus,
)
from src.core.exceptions import SysRevError
from src.core.schemas import (
    AuditReport,
    CohortOverlapWarning,
    DuplicateCandidate,
    ExtractedStudyArm,
    ExtractedValue,
    FullTextItem,
    GateDecision,
    PICOProtocol,
    PRISMACounts,
    RiskOfBiasAssessment,
    ScreeningConflict,
    ScreeningDecision,
    SourceEvidence,
    StateTransition,
    StudyRecord,
)
from src.core.schemas.ai import AIVerification
from src.core.schemas.base import utc_now
from src.core.schemas.manuscript import (
    ClaimReview,
    EvidenceExcerpt,
    FindingAcknowledgement,
    ManuscriptApproval,
    ManuscriptVersion,
    ReferenceEntry,
    ReferenceVerification,
    fingerprint_sha256,
)
from src.core.schemas.master import (
    CohortContributionDecision,
    ExtractionItem,
    ExtractionReviewEvent,
    QuantitativeUnavailability,
    StudyReportLink,
)
from src.core.schemas.reporting import (
    ChecklistDecision,
    GradeAssessment,
    SwimApplicability,
    SynthesisGroup,
)
from src.core.schemas.search import ProtocolEvent, SearchRun
from src.core.schemas.stats import AnalysisPlan, StatsOutput
from src.core.security import (
    format_token,
    generate_secret,
    hash_secret,
    new_salt,
    parse_token,
    verify_secret,
)
from src.llm.catalogue import ModelCatalogue
from src.llm.model_registry import CatalogueSnapshot, ProviderDiscovery
from src.llm.tasks import ProjectAISettings
from src.llm.usage import LLMCallRecord
from src.screening.agreement import screening_status
from src.state.db import (
    AIVerificationORM,
    AnalysisPlanORM,
    AuditReportORM,
    ClaimReviewORM,
    CohortContributionDecisionORM,
    CohortOverlapWarningORM,
    DuplicateCandidateORM,
    EvidenceExcerptORM,
    ExtractedArmORM,
    ExtractedValueORM,
    ExtractionItemORM,
    ExtractionReviewEventORM,
    FindingAcknowledgementORM,
    FullTextDocumentORM,
    FullTextItemORM,
    GateDecisionORM,
    GovernanceRecordORM,
    LLMCallORM,
    LLMCallOutcomeORM,
    ManuscriptApprovalORM,
    ManuscriptVersionORM,
    ModelCatalogueActivationORM,
    ModelCatalogueSnapshotORM,
    ProjectAISettingsORM,
    ProjectORM,
    ProtocolEventORM,
    ProtocolORM,
    QuantitativeUnavailabilityORM,
    ReferenceEntryORM,
    ReferenceVerificationORM,
    ReportingRecordORM,
    ReviewerORM,
    RiskOfBiasORM,
    ScreeningConflictORM,
    ScreeningDecisionORM,
    SearchRawRecordORM,
    SearchRunORM,
    SourceEvidenceORM,
    StateTransitionORM,
    StatsOutputORM,
    StudyRecordORM,
    StudyReportLinkORM,
)
from src.state.machine import GuardContext, PipelineStateMachine
from src.stats_engine.engine import run_plan
from src.stats_engine.inputs import AnalysisInputs

_REVIEW_STAMPS = {"review_status", "reviewed_by", "reviewer_id", "reviewed_at", "review_note"}


class NotFoundError(SysRevError, LookupError):
    pass


class Repository:
    def __init__(self, session: Session) -> None:
        self.session = session

    # ------------------------------------------------------------------ projects
    def create_project(self, project_id: str, title: str) -> ProjectORM:
        project = ProjectORM(id=project_id, title=title, phase=Phase.STAGE1_PROTOCOL.value)
        self.session.add(project)
        self.session.flush()
        return project

    def get_project(self, project_id: str) -> ProjectORM:
        project = self.session.get(ProjectORM, project_id)
        if project is None:
            raise NotFoundError(f"project {project_id} not found")
        return project

    def load_state_machine(self, project_id: str) -> PipelineStateMachine:
        project = self.get_project(project_id)
        return PipelineStateMachine(
            project_id, Phase(project.phase), history=self.list_transitions(project_id)
        )

    def record_transition(self, transition: StateTransition) -> None:
        """Persist a transition produced by the state machine and update the project phase."""
        project = self.get_project(transition.project_id)
        if project.phase != transition.from_phase.value:
            raise SysRevError(
                f"stale state: project is at {project.phase}, transition starts at "
                f"{transition.from_phase}"
            )
        gate_id = None
        if transition.gate_decision is not None:
            self._require_enrolled_reviewer(transition.gate_decision)
            gate_id = self.save_gate_decision(transition.project_id, transition.gate_decision)
        self.session.add(
            StateTransitionORM(
                id=transition.transition_id,
                project_id=transition.project_id,
                from_phase=transition.from_phase.value,
                to_phase=transition.to_phase.value,
                actor_type=transition.actor_type.value,
                actor=transition.actor,
                reason=transition.reason,
                gate_decision_id=gate_id,
                occurred_at=transition.occurred_at,
            )
        )
        project.phase = transition.to_phase.value
        self.session.flush()

    def list_transitions(self, project_id: str) -> list[StateTransition]:
        rows = self.session.scalars(
            select(StateTransitionORM)
            .where(StateTransitionORM.project_id == project_id)
            .order_by(StateTransitionORM.seq)
        ).all()
        result: list[StateTransition] = []
        for row in rows:
            gate = None
            if row.gate_decision_id:
                g = self.session.get(GateDecisionORM, row.gate_decision_id)
                assert g is not None  # FK guarantees existence
                gate = GateDecision(
                    decision_id=g.id,
                    gate=g.gate,  # type: ignore[arg-type]
                    verdict=g.verdict,  # type: ignore[arg-type]
                    reviewer_name=g.reviewer_name,
                    reviewer_id=g.reviewer_id,
                    comments=g.comments,
                    decided_at=g.decided_at,
                )
            result.append(
                StateTransition(
                    transition_id=row.id,
                    project_id=row.project_id,
                    from_phase=Phase(row.from_phase),
                    to_phase=Phase(row.to_phase),
                    actor_type=row.actor_type,  # type: ignore[arg-type]
                    actor=row.actor,
                    reason=row.reason,
                    gate_decision=gate,
                    occurred_at=row.occurred_at,
                )
            )
        return result

    # ------------------------------------------------------------------ reviewers
    def enroll_reviewer(self, display_name: str) -> tuple[str, str]:
        """Create a reviewer and return ``(reviewer_id, token)``. The token is not stored."""
        name = display_name.strip()
        if not name:
            raise ValueError("display_name must not be empty")
        reviewer_id = uuid.uuid4().hex
        secret, salt = generate_secret(), new_salt()
        self.session.add(
            ReviewerORM(
                id=reviewer_id,
                display_name=name,
                salt=salt,
                token_hash=hash_secret(secret, salt),
            )
        )
        self.session.flush()
        return reviewer_id, format_token(reviewer_id, secret)

    def authenticate_reviewer(self, token: str) -> ReviewerORM | None:
        """Return the active reviewer owning ``token``, or ``None``."""
        parsed = parse_token(token)
        if parsed is None:
            return None
        reviewer_id, secret = parsed
        row = self.session.get(ReviewerORM, reviewer_id)
        if row is None or row.revoked_at is not None:
            return None
        return row if verify_secret(secret, row.salt, row.token_hash) else None

    def revoke_reviewer(self, reviewer_id: str) -> None:
        row = self.session.get(ReviewerORM, reviewer_id)
        if row is None:
            raise NotFoundError(f"reviewer {reviewer_id} not found")
        if row.revoked_at is None:
            row.revoked_at = utc_now()
            self.session.flush()

    def list_reviewers(self) -> list[ReviewerORM]:
        return list(self.session.scalars(select(ReviewerORM).order_by(ReviewerORM.created_at)))

    def require_active_reviewer(self, reviewer_id: str | None) -> ReviewerORM:
        """The enrolled, non-revoked reviewer ``reviewer_id`` — or raise. Every human
        approval in Stages 2-4 is persisted only through this check."""
        if reviewer_id is None:
            raise SysRevError("this action requires an authenticated, enrolled human reviewer")
        row = self.session.get(ReviewerORM, reviewer_id)
        if row is None or row.revoked_at is not None:
            raise SysRevError(f"reviewer {reviewer_id} is not enrolled or was revoked")
        return row

    def _require_enrolled_reviewer(self, decision: GateDecision) -> None:
        """A gate decision may only be persisted for an active, enrolled reviewer."""
        if decision.reviewer_id is None:
            raise SysRevError("gate decisions must be authenticated by an enrolled reviewer")
        row = self.session.get(ReviewerORM, decision.reviewer_id)
        if row is None or row.revoked_at is not None:
            raise SysRevError(f"reviewer {decision.reviewer_id} is not enrolled or was revoked")
        if row.display_name != decision.reviewer_name:
            raise SysRevError("reviewer_name does not match the enrolled reviewer")

    def save_gate_decision(self, project_id: str, decision: GateDecision) -> str:
        if decision.reviewer_id is None:
            raise SysRevError("gate decisions must be authenticated by an enrolled reviewer")
        self.session.add(
            GateDecisionORM(
                id=decision.decision_id,
                project_id=project_id,
                gate=decision.gate.value,
                verdict=decision.verdict.value,
                reviewer_id=decision.reviewer_id,
                reviewer_name=decision.reviewer_name,
                comments=decision.comments,
                decided_at=decision.decided_at,
            )
        )
        return decision.decision_id

    def save_prisma(self, project_id: str, counts: PRISMACounts) -> None:
        self.get_project(project_id).prisma_counts = counts.model_dump(mode="json")
        self.session.flush()

    def load_prisma(self, project_id: str) -> PRISMACounts | None:
        data = self.get_project(project_id).prisma_counts
        return None if data is None else PRISMACounts.model_validate(data)

    # ------------------------------------------------------------------ protocol
    def save_protocol(self, project_id: str, protocol: PICOProtocol) -> None:
        if protocol.approved_reviewer_id is not None:
            reviewer = self.require_active_reviewer(protocol.approved_reviewer_id)
            if protocol.approved_by != reviewer.display_name:
                raise SysRevError("approved_by does not match the enrolled reviewer")
        self.session.merge(
            ProtocolORM(
                id=protocol.protocol_id,
                project_id=project_id,
                version=protocol.version,
                status=protocol.status,
                approved_by=protocol.approved_by,
                approved_at=protocol.approved_at,
                payload=protocol.model_dump(mode="json"),
            )
        )
        self.session.flush()

    def latest_protocol(self, project_id: str) -> PICOProtocol | None:
        row = self.session.scalars(
            select(ProtocolORM)
            .where(ProtocolORM.project_id == project_id)
            .order_by(ProtocolORM.version.desc())
            .limit(1)
        ).first()
        return None if row is None else PICOProtocol.model_validate(row.payload)

    def list_protocols(self, project_id: str) -> list[PICOProtocol]:
        rows = self.session.scalars(
            select(ProtocolORM)
            .where(ProtocolORM.project_id == project_id)
            .order_by(ProtocolORM.version)
        ).all()
        return [PICOProtocol.model_validate(r.payload) for r in rows]

    def supersede_older_protocols(self, project_id: str, version: int) -> None:
        """Mark approved versions older than ``version`` as superseded."""
        rows = self.session.scalars(
            select(ProtocolORM).where(
                ProtocolORM.project_id == project_id,
                ProtocolORM.version < version,
                ProtocolORM.status == "approved",
            )
        ).all()
        for row in rows:
            row.status = "superseded"
            row.payload = PICOProtocol.model_validate(
                row.payload | {"status": "superseded"}
            ).model_dump(mode="json")
        self.session.flush()

    def add_protocol_event(
        self, project_id: str, event: ProtocolEvent, protocol: PICOProtocol
    ) -> None:
        if event.actor_type == ActorType.HUMAN:
            self.require_active_reviewer(event.reviewer_id)
        self.session.add(
            ProtocolEventORM(
                id=event.event_id,
                project_id=project_id,
                protocol_version=event.protocol_version,
                action=event.action,
                reviewer_id=event.reviewer_id,
                event=event.model_dump(mode="json"),
                protocol_payload=protocol.model_dump(mode="json"),
                created_at=event.occurred_at,
            )
        )
        self.session.flush()

    def list_protocol_events(self, project_id: str) -> list[tuple[ProtocolEvent, PICOProtocol]]:
        rows = self.session.scalars(
            select(ProtocolEventORM)
            .where(ProtocolEventORM.project_id == project_id)
            .order_by(ProtocolEventORM.seq)
        ).all()
        return [
            (ProtocolEvent.model_validate(r.event), PICOProtocol.model_validate(r.protocol_payload))
            for r in rows
        ]

    # ------------------------------------------------------------------ search log (Session 5)
    def add_search_run(self, project_id: str, run: SearchRun) -> None:
        if run.reviewer_id is not None:
            reviewer = self.require_active_reviewer(run.reviewer_id)
            if run.actor != reviewer.display_name:
                raise SysRevError("search-log actor does not match the enrolled reviewer")
        self.session.add(
            SearchRunORM(
                id=run.run_id,
                project_id=project_id,
                kind=run.kind.value,
                status=run.status.value,
                database=run.database.value if run.database else None,
                strategy_sha256=run.strategy_sha256,
                reviewer_id=run.reviewer_id,
                payload=run.model_dump(mode="json"),
                created_at=run.executed_at,
            )
        )
        self.session.flush()

    def load_search_runs(
        self, project_id: str, kind: SearchRunKind | None = None
    ) -> list[SearchRun]:
        """Search-log entries in the order they were written."""
        stmt = select(SearchRunORM).where(SearchRunORM.project_id == project_id)
        if kind is not None:
            stmt = stmt.where(SearchRunORM.kind == kind.value)
        rows = self.session.scalars(stmt.order_by(SearchRunORM.seq)).all()
        return [SearchRun.model_validate(r.payload) for r in rows]

    def add_raw_records(
        self, project_id: str, run_id: str, rows: Sequence[tuple[str, int, dict[str, object]]]
    ) -> None:
        for record_id, position, raw in rows:
            self.session.add(
                SearchRawRecordORM(
                    project_id=project_id, run_id=run_id, record_id=record_id,
                    position=position, raw=raw,
                )
            )  # fmt: skip
        self.session.flush()

    def raw_record(self, project_id: str, record_id: str) -> dict[str, object] | None:
        row = self.session.scalars(
            select(SearchRawRecordORM).where(
                SearchRawRecordORM.project_id == project_id,
                SearchRawRecordORM.record_id == record_id,
            )
        ).first()
        return None if row is None else dict(row.raw)

    def search_identification_problems(self, project_id: str) -> list[str]:
        """Why the identified records are not yet a complete, prepared screening pool.

        Applies once the project uses the search log (Session 5); projects whose records were
        stored directly through the library API are not affected.
        """
        from src.search.identification import identification_problems

        payloads = self.session.scalars(
            select(StudyRecordORM.payload).where(StudyRecordORM.project_id == project_id)
        ).all()
        unattributed = sum(
            1 for p in payloads if not ((p.get("provenance") or {}).get("search_run_id"))
        )
        return identification_problems(
            self.load_search_runs(project_id), self.latest_protocol(project_id), unattributed
        )

    def delete_unresolved_duplicate_candidates(self, project_id: str) -> int:
        rows = self.session.scalars(
            select(DuplicateCandidateORM).where(
                DuplicateCandidateORM.project_id == project_id,
                DuplicateCandidateORM.resolution == "unresolved",
            )
        ).all()
        for row in rows:
            self.session.delete(row)
        self.session.flush()
        return len(rows)

    # ------------------------------------------------------------------ records
    def save_records(self, project_id: str, records: list[StudyRecord]) -> None:
        # Insert representatives first so self-referencing duplicate_of FKs resolve.
        ordered = sorted(records, key=lambda r: r.duplicate_of is not None)
        for record in ordered:
            self.session.merge(
                StudyRecordORM(
                    id=record.record_id,
                    project_id=project_id,
                    source_database=record.source_database.value,
                    pmid=record.pmid,
                    pmcid=record.pmcid,
                    doi=record.doi,
                    title=record.title,
                    normalized_title=record.normalized_title,
                    publication_year=record.publication_year,
                    first_author_key=record.first_author_key or None,
                    duplicate_of=record.duplicate_of,
                    payload=record.model_dump(mode="json"),
                )
            )
            self.session.flush()

    def load_records(self, project_id: str, *, unique_only: bool = False) -> list[StudyRecord]:
        stmt = select(StudyRecordORM).where(StudyRecordORM.project_id == project_id)
        if unique_only:
            stmt = stmt.where(StudyRecordORM.duplicate_of.is_(None))
        rows = self.session.scalars(stmt.order_by(StudyRecordORM.id)).all()
        return [StudyRecord.model_validate(r.payload) for r in rows]

    # ------------------------------------------------------------------ screening
    def save_screening_decision(self, project_id: str, decision: ScreeningDecision) -> None:
        if decision.is_human:
            reviewer = self.require_active_reviewer(decision.reviewer_id)
            if decision.reviewer_identity != reviewer.display_name:
                raise SysRevError("reviewer_identity does not match the enrolled reviewer")
        self.session.add(
            ScreeningDecisionORM(
                id=decision.decision_id,
                project_id=project_id,
                record_id=decision.record_id,
                reviewer=decision.reviewer.value,
                reviewer_identity=decision.reviewer_identity,
                is_human=decision.is_human,
                stage=decision.stage.value,
                verdict=decision.verdict.value,
                exclusion_reason=decision.exclusion_reason,
                rationale=decision.rationale,
                decided_at=decision.decided_at,
                payload=decision.model_dump(mode="json"),
            )
        )
        self.session.flush()

    def load_screening_decisions(self, project_id: str) -> list[ScreeningDecision]:
        """Stored decisions, followed by any verdict a human revised when re-reviewing a record
        after a protocol amendment (Session 8). A revision is appended as a later human
        ADJUDICATOR decision, so it is the effective one; the earlier rows are unchanged."""
        rows = self.session.scalars(
            select(ScreeningDecisionORM).where(ScreeningDecisionORM.project_id == project_id)
        ).all()
        stored = [ScreeningDecision.model_validate(r.payload) for r in rows]
        return stored + self._amendment_revisions(project_id)

    def _amendment_revisions(self, project_id: str) -> list[ScreeningDecision]:
        out = []
        for res in self.governance_records(project_id, "amendment_resolution"):
            if res.get("resolution") != "revised" or res.get("verdict") is None:
                continue
            out.append(
                ScreeningDecision(
                    decision_id=str(res["_record_id"]),
                    record_id=str(res["subject"]),
                    reviewer=ReviewerRole.ADJUDICATOR,
                    reviewer_identity=str(res["resolved_by"]),
                    reviewer_id=str(res["reviewer_id"]),
                    is_human=True,
                    stage=ScreeningStage(str(res["stage"])),
                    verdict=ScreeningVerdict(str(res["verdict"])),
                    exclusion_reason=(
                        str(res["exclusion_reason"]) if res.get("exclusion_reason") else None
                    ),
                    rationale=(
                        f"re-review after protocol amendment {str(res['amendment_id'])[:12]}: "
                        f"{res['note']}"
                    ),
                    decided_at=res["resolved_at"],  # type: ignore[arg-type]
                )
            )
        return out

    def save_conflict(self, project_id: str, conflict: ScreeningConflict) -> None:
        """Persist a conflict that matches the stored reviewer decisions.

        A *resolved* conflict is accepted only as a mirror of an existing human
        ADJUDICATOR decision by an active, enrolled reviewer with the same verdict; a
        stored resolution can never be changed or withdrawn.
        """
        self._validate_conflict(project_id, conflict)
        self.session.merge(
            ScreeningConflictORM(
                id=conflict.conflict_id,
                project_id=project_id,
                record_id=conflict.record_id,
                stage=conflict.stage.value,
                resolved_verdict=conflict.resolved_verdict.value
                if conflict.resolved_verdict
                else None,
                payload=conflict.model_dump(mode="json"),
            )
        )
        self.session.flush()

    def _validate_conflict(self, project_id: str, conflict: ScreeningConflict) -> None:
        stage = conflict.stage
        existing = self.session.scalars(
            select(ScreeningConflictORM).where(
                ScreeningConflictORM.project_id == project_id,
                ScreeningConflictORM.record_id == conflict.record_id,
                ScreeningConflictORM.stage == stage.value,
            )
        ).first()
        if existing is not None:
            if existing.id != conflict.conflict_id:
                raise SysRevError(f"a conflict for {conflict.record_id} at {stage} already exists")
            if existing.resolved_verdict is not None:
                raise SysRevError(f"conflict {conflict.conflict_id} is already resolved")
        for role, claimed in (
            (ReviewerRole.REVIEWER_1, conflict.reviewer_1_verdict),
            (ReviewerRole.REVIEWER_2, conflict.reviewer_2_verdict),
        ):
            source = self.find_screening_decision(project_id, conflict.record_id, role, stage)
            if source is None or source.verdict != claimed:
                raise SysRevError(
                    f"conflict for {conflict.record_id} does not match the stored {role} "
                    f"decision at {stage}"
                )
        if conflict.resolved_verdict is None:
            return
        reviewer = self.require_active_reviewer(conflict.resolved_by_reviewer_id)
        if conflict.resolved_by != reviewer.display_name:
            raise SysRevError("resolved_by does not match the enrolled reviewer")
        adjudication = self.find_screening_decision(
            project_id, conflict.record_id, ReviewerRole.ADJUDICATOR, stage
        )
        if (
            adjudication is None
            or not adjudication.is_human
            or adjudication.reviewer_id != reviewer.id
            or adjudication.verdict != conflict.resolved_verdict
        ):
            raise SysRevError(
                f"resolution of {conflict.record_id} is not backed by a matching human "
                "adjudication decision"
            )

    def load_conflicts(self, project_id: str) -> list[ScreeningConflict]:
        rows = self.session.scalars(
            select(ScreeningConflictORM).where(ScreeningConflictORM.project_id == project_id)
        ).all()
        return [ScreeningConflict.model_validate(r.payload) for r in rows]

    # ------------------------------------------------------------------ duplicate candidates
    def save_duplicate_candidate(self, project_id: str, candidate: DuplicateCandidate) -> None:
        if candidate.is_resolved:
            reviewer = self.require_active_reviewer(candidate.resolved_by_reviewer_id)
            if candidate.resolved_by != reviewer.display_name:
                raise SysRevError("resolved_by does not match the enrolled reviewer")
        self.session.merge(
            DuplicateCandidateORM(
                id=candidate.candidate_id,
                project_id=project_id,
                record_id_a=candidate.record_id_a,
                record_id_b=candidate.record_id_b,
                resolution=candidate.resolution,
                payload=candidate.model_dump(mode="json"),
            )
        )
        self.session.flush()

    def load_duplicate_candidates(self, project_id: str) -> list[DuplicateCandidate]:
        rows = self.session.scalars(
            select(DuplicateCandidateORM)
            .where(DuplicateCandidateORM.project_id == project_id)
            .order_by(DuplicateCandidateORM.record_id_a, DuplicateCandidateORM.record_id_b)
        ).all()
        return [DuplicateCandidate.model_validate(r.payload) for r in rows]

    def resolve_duplicate_candidate(
        self,
        project_id: str,
        candidate_id: str,
        *,
        resolution: str,
        reviewer_id: str,
        note: str | None = None,
    ) -> DuplicateCandidate:
        """Record a human adjudication; ``duplicate_keep_*`` re-points the dropped cluster.

        The caller remains responsible for updating PRISMA ``duplicates_removed``; the guard
        cross-check rejects the PRISMA snapshot until it matches the stored records.
        """
        reviewer = self.require_active_reviewer(reviewer_id)
        row = self.session.get(DuplicateCandidateORM, candidate_id)
        if row is None or row.project_id != project_id:
            raise NotFoundError(f"duplicate candidate {candidate_id} not found")
        if row.resolution != "unresolved":
            raise SysRevError(f"duplicate candidate {candidate_id} is already resolved")
        resolved = DuplicateCandidate.model_validate(
            row.payload
            | {
                "resolution": resolution,
                "resolved_by": reviewer.display_name,
                "resolved_by_reviewer_id": reviewer.id,
                "resolved_at": utc_now(),
                "resolution_note": note,
            }
        )
        if resolution in ("duplicate_keep_a", "duplicate_keep_b"):
            keep, drop = (
                (resolved.record_id_a, resolved.record_id_b)
                if resolution == "duplicate_keep_a"
                else (resolved.record_id_b, resolved.record_id_a)
            )
            self._merge_clusters(project_id, keep=keep, drop=drop)
        self.save_duplicate_candidate(project_id, resolved)
        return resolved

    def reapply_duplicate_resolution(self, project_id: str, candidate: DuplicateCandidate) -> None:
        """Re-apply a stored human decision after the deduplicator was re-run.

        ``distinct`` needs no change (the pair stays separate); ``duplicate_keep_*`` re-links
        the dropped cluster to the kept one. The decision row itself is not modified.
        """
        if candidate.resolution == "duplicate_keep_a":
            self._merge_clusters(project_id, keep=candidate.record_id_a, drop=candidate.record_id_b)
        elif candidate.resolution == "duplicate_keep_b":
            self._merge_clusters(project_id, keep=candidate.record_id_b, drop=candidate.record_id_a)

    def _merge_clusters(self, project_id: str, *, keep: str, drop: str) -> None:
        def representative(record_id: str) -> str:
            row = self.session.get(StudyRecordORM, record_id)
            if row is None or row.project_id != project_id:
                raise NotFoundError(f"record {record_id} not found")
            return row.duplicate_of or row.id

        keep_rep, drop_rep = representative(keep), representative(drop)
        if keep_rep == drop_rep:
            return
        rows = self.session.scalars(
            select(StudyRecordORM).where(
                StudyRecordORM.project_id == project_id,
                (StudyRecordORM.id == drop_rep) | (StudyRecordORM.duplicate_of == drop_rep),
            )
        ).all()
        for row in rows:
            record = StudyRecord.model_validate(row.payload | {"duplicate_of": keep_rep})
            row.duplicate_of = keep_rep
            row.payload = record.model_dump(mode="json")
        self.session.flush()

    # ------------------------------------------------------------------ cohort overlap
    def save_overlap_warning(self, project_id: str, warning: CohortOverlapWarning) -> None:
        if warning.resolution != OverlapResolution.UNRESOLVED:
            reviewer = self.require_active_reviewer(warning.resolved_by_reviewer_id)
            if warning.resolved_by != reviewer.display_name:
                raise SysRevError("resolved_by does not match the enrolled reviewer")
        self.session.merge(
            CohortOverlapWarningORM(
                id=warning.warning_id,
                project_id=project_id,
                record_id_a=warning.record_id_a,
                record_id_b=warning.record_id_b,
                overlap_score=warning.overlap_score,
                severity=warning.severity.value,
                resolution=warning.resolution.value,
                payload=warning.model_dump(mode="json"),
            )
        )
        self.session.flush()

    def resolve_overlap_warning(
        self,
        project_id: str,
        warning_id: str,
        *,
        resolution: OverlapResolution,
        reviewer_id: str,
        note: str | None = None,
    ) -> CohortOverlapWarning:
        """Record a human cohort-overlap decision on behalf of an enrolled reviewer."""
        reviewer = self.require_active_reviewer(reviewer_id)
        row = self.session.get(CohortOverlapWarningORM, warning_id)
        if row is None or row.project_id != project_id:
            raise NotFoundError(f"cohort-overlap warning {warning_id} not found")
        if row.resolution != OverlapResolution.UNRESOLVED.value:
            raise SysRevError(f"cohort-overlap warning {warning_id} is already resolved")
        resolved = CohortOverlapWarning.model_validate(
            row.payload
            | {
                "resolution": resolution,
                "resolved_by": reviewer.display_name,
                "resolved_by_reviewer_id": reviewer.id,
                "resolution_note": note,
            }
        )
        self.save_overlap_warning(project_id, resolved)
        return resolved

    def load_overlap_warnings(self, project_id: str) -> list[CohortOverlapWarning]:
        rows = self.session.scalars(
            select(CohortOverlapWarningORM).where(CohortOverlapWarningORM.project_id == project_id)
        ).all()
        return [CohortOverlapWarning.model_validate(r.payload) for r in rows]

    # ------------------------------------------------------------------ full text
    def save_fulltext_item(self, project_id: str, item: FullTextItem) -> None:
        self._require_fulltext_humans(item)
        existing = self.session.scalars(
            select(FullTextItemORM).where(
                FullTextItemORM.project_id == project_id,
                FullTextItemORM.record_id == item.record_id,
            )
        ).first()
        row = existing or FullTextItemORM(project_id=project_id, record_id=item.record_id)
        row.status = item.status.value
        row.pdf_filename = item.pdf_filename
        row.pdf_sha256 = item.pdf_sha256
        row.payload = item.model_dump(mode="json")
        self.session.add(row)
        self.session.flush()

    def _require_fulltext_humans(self, item: FullTextItem) -> None:
        """Human manifest decisions must name an active, enrolled reviewer."""
        checks: list[tuple[str | None, str | None]] = []
        if item.match_status == UploadMatchStatus.HUMAN_CONFIRMED:
            checks.append((item.confirmed_by_reviewer_id, item.confirmed_by))
        if item.status in (FullTextStatus.UNOBTAINABLE, FullTextStatus.USER_UPLOADED):
            checks.append((item.recorded_by_reviewer_id, item.recorded_by))
        if item.xml_only_acknowledged_by is not None:
            checks.append((item.xml_only_acknowledged_reviewer_id, item.xml_only_acknowledged_by))
        for reviewer_id, name in checks:
            reviewer = self.require_active_reviewer(reviewer_id)
            if name != reviewer.display_name:
                raise SysRevError("recorded reviewer name does not match the enrolled reviewer")

    def missing_fulltext_manifest(self, project_id: str) -> list[FullTextItem]:
        rows = self.session.scalars(
            select(FullTextItemORM).where(
                FullTextItemORM.project_id == project_id,
                FullTextItemORM.status.in_(
                    [FullTextStatus.PENDING.value, FullTextStatus.MISSING.value]
                ),
            )
        ).all()
        return [FullTextItem.model_validate(r.payload) for r in rows]

    # ------------------------------------------------------------------ extraction
    def save_extracted_arm(self, project_id: str, arm: ExtractedStudyArm) -> None:
        if self.session.get(ExtractedArmORM, arm.arm_id) is not None:
            raise SysRevError(
                f"arm {arm.arm_id} already exists; corrections create a new arm version"
            )
        row = ExtractedArmORM(
            id=arm.arm_id,
            project_id=project_id,
            record_id=arm.record_id,
            outcome_id=arm.outcome_id,
            arm_label=arm.arm_label,
            arm_role=arm.arm_role.value,
            timepoint=arm.timepoint,
            estimate_type=arm.estimate_type.value if arm.estimate_type else None,
            estimate_scale=arm.estimate_scale,
            ci_level=arm.ci_level,
            follow_up=arm.follow_up.model_dump(mode="json") if arm.follow_up else None,
            approved_by_human=arm.approved_by_human,
            approved_by=arm.approved_by,
            approved_at=arm.approved_at,
        )
        for field_name, value in arm.extracted_values().items():
            evidence = self._get_or_add_evidence(value.evidence)
            row.values.append(
                ExtractedValueORM(
                    field_name=field_name,
                    value=value.value,
                    reported_text=value.reported_text,
                    unit=value.unit,
                    evidence=evidence,
                )
            )
        self.session.add(row)
        self.session.flush()

    def approve_extracted_arm(self, arm_id: str, *, reviewer_id: str) -> ExtractedStudyArm:
        """Approve an arm on behalf of an authenticated, enrolled reviewer."""
        reviewer = self.require_active_reviewer(reviewer_id)
        row = self.session.get(ExtractedArmORM, arm_id)
        if row is None:
            raise NotFoundError(f"arm {arm_id} not found")
        arm = self._arm_from_row(row)
        approved = arm.model_copy(
            update={
                "approved_by_human": True,
                "approved_by": reviewer.display_name,
                "approved_at": utc_now(),
            }
        )
        approved = ExtractedStudyArm.model_validate(approved.model_dump())
        row.approved_by_human = True
        row.approved_by = approved.approved_by
        row.approved_at = approved.approved_at
        self.session.flush()
        return approved

    def invalidated_arm_ids(self, project_id: str) -> set[str]:
        """Arms withdrawn after a correction of an accepted value (Session 8). The rows are
        kept (write-once history); every reader ignores them and a rebuilt arm gets a new id."""
        rows = self.session.scalars(
            select(GovernanceRecordORM.subject).where(
                GovernanceRecordORM.project_id == project_id,
                GovernanceRecordORM.kind == "arm_invalidation",
            )
        ).all()
        return set(rows)

    def invalidate_arm(
        self, project_id: str, arm_id: str, *, reviewer_id: str, item_id: str, reason: str
    ) -> None:
        row = self.session.get(ExtractedArmORM, arm_id)
        if row is None or row.project_id != project_id:
            raise NotFoundError(f"arm {arm_id} not found")
        if arm_id in self.invalidated_arm_ids(project_id):
            return
        self.add_governance_record(
            project_id, "arm_invalidation", arm_id,
            {"arm_id": arm_id, "record_id": row.record_id, "outcome_id": row.outcome_id,
             "item_id": item_id, "reason": reason, "reviewer_id": reviewer_id},
            reviewer_id=reviewer_id,
        )  # fmt: skip

    def load_extracted_arms(
        self, project_id: str, *, include_invalidated: bool = False
    ) -> list[ExtractedStudyArm]:
        stmt = select(ExtractedArmORM).where(ExtractedArmORM.project_id == project_id)
        if not include_invalidated:
            invalid = self.invalidated_arm_ids(project_id)
            if invalid:
                stmt = stmt.where(ExtractedArmORM.id.notin_(invalid))
        rows = self.session.scalars(stmt.order_by(ExtractedArmORM.id)).all()
        return [self._arm_from_row(r) for r in rows]

    def _get_or_add_evidence(self, evidence: SourceEvidence) -> SourceEvidenceORM:
        existing = self.session.get(SourceEvidenceORM, evidence.evidence_id)
        if existing is not None:
            if SourceEvidence.model_validate(self._evidence_dict(existing)) != evidence:
                raise SysRevError(
                    f"evidence {evidence.evidence_id} already stored with different content"
                )
            return existing
        row = SourceEvidenceORM(
            id=evidence.evidence_id,
            pdf_filename=evidence.pdf_filename,
            pdf_sha256=evidence.pdf_sha256,
            page_number=evidence.page_number,
            location_type=evidence.location_type.value,
            location_identifier=evidence.location_identifier,
            verbatim_quote=evidence.verbatim_quote,
            extracted_by=evidence.extracted_by.value,
            extractor_identity=evidence.extractor_identity,
            extracted_at=evidence.extracted_at,
        )
        self.session.add(row)
        return row

    @staticmethod
    def _evidence_dict(row: SourceEvidenceORM) -> dict[str, object]:
        return {
            "evidence_id": row.id,
            "pdf_filename": row.pdf_filename,
            "pdf_sha256": row.pdf_sha256,
            "page_number": row.page_number,
            "location_type": row.location_type,
            "location_identifier": row.location_identifier,
            "verbatim_quote": row.verbatim_quote,
            "extracted_by": row.extracted_by,
            "extractor_identity": row.extractor_identity,
            "extracted_at": row.extracted_at,
        }

    def _arm_from_row(self, row: ExtractedArmORM) -> ExtractedStudyArm:
        data: dict[str, object] = {
            "arm_id": row.id,
            "record_id": row.record_id,
            "outcome_id": row.outcome_id,
            "arm_label": row.arm_label,
            "arm_role": row.arm_role,
            "timepoint": row.timepoint,
            "estimate_type": row.estimate_type,
            "estimate_scale": row.estimate_scale,
            "ci_level": row.ci_level,
            "follow_up": row.follow_up,
            "approved_by_human": row.approved_by_human,
            "approved_by": row.approved_by,
            "approved_at": row.approved_at,
        }
        for v in row.values:
            data[v.field_name] = ExtractedValue(
                value=v.value,
                reported_text=v.reported_text,
                unit=v.unit,
                evidence=SourceEvidence.model_validate(self._evidence_dict(v.evidence)),
            )
        return ExtractedStudyArm.model_validate(data)

    def save_rob(self, project_id: str, assessment: RiskOfBiasAssessment) -> None:
        if assessment.approved_by_human:
            reviewer = self.require_active_reviewer(assessment.approved_reviewer_id)
            if assessment.approved_by != reviewer.display_name:
                raise SysRevError("approved_by does not match the enrolled reviewer")
        self.session.merge(
            RiskOfBiasORM(
                id=assessment.assessment_id,
                project_id=project_id,
                record_id=assessment.record_id,
                tool=assessment.tool.value,
                overall=assessment.overall,
                payload=assessment.model_dump(mode="json"),
            )
        )
        self.session.flush()

    def load_rob(self, project_id: str) -> list[RiskOfBiasAssessment]:
        rows = self.session.scalars(
            select(RiskOfBiasORM)
            .where(RiskOfBiasORM.project_id == project_id)
            .order_by(RiskOfBiasORM.id)
        ).all()
        return [RiskOfBiasAssessment.model_validate(r.payload) for r in rows]

    def get_rob(self, project_id: str, assessment_id: str) -> RiskOfBiasAssessment:
        row = self.session.get(RiskOfBiasORM, assessment_id)
        if row is None or row.project_id != project_id:
            raise NotFoundError(f"risk-of-bias assessment {assessment_id} not found")
        return RiskOfBiasAssessment.model_validate(row.payload)

    # ------------------------------------------------------------------ Session 2 helpers
    def get_record(self, project_id: str, record_id: str) -> StudyRecord:
        row = self.session.get(StudyRecordORM, record_id)
        if row is None or row.project_id != project_id:
            raise NotFoundError(f"record {record_id} not found")
        return StudyRecord.model_validate(row.payload)

    def find_screening_decision(
        self, project_id: str, record_id: str, role: ReviewerRole, stage: ScreeningStage
    ) -> ScreeningDecision | None:
        row = self.session.scalars(
            select(ScreeningDecisionORM).where(
                ScreeningDecisionORM.project_id == project_id,
                ScreeningDecisionORM.record_id == record_id,
                ScreeningDecisionORM.reviewer == role.value,
                ScreeningDecisionORM.stage == stage.value,
            )
        ).first()
        return None if row is None else ScreeningDecision.model_validate(row.payload)

    def find_conflict(
        self, project_id: str, record_id: str, stage: ScreeningStage
    ) -> ScreeningConflict | None:
        row = self.session.scalars(
            select(ScreeningConflictORM).where(
                ScreeningConflictORM.project_id == project_id,
                ScreeningConflictORM.record_id == record_id,
                ScreeningConflictORM.stage == stage.value,
            )
        ).first()
        return None if row is None else ScreeningConflict.model_validate(row.payload)

    def included_record_ids(self, project_id: str, stage: ScreeningStage) -> set[str]:
        """Records whose final (consensus or adjudicated) verdict at ``stage`` is include."""
        unique = {r.record_id for r in self.load_records(project_id, unique_only=True)}
        decisions = self.load_screening_decisions(project_id)
        conflicts = self.load_conflicts(project_id)
        ta = screening_status(unique, decisions, conflicts, ScreeningStage.TITLE_ABSTRACT)
        ta_inc = ta.ids_with(ScreeningVerdict.INCLUDE)
        if stage == ScreeningStage.TITLE_ABSTRACT:
            return ta_inc
        retrieved = {i.record_id for i in self.load_fulltext_items(project_id) if i.is_retrieved}
        ft = screening_status(ta_inc & retrieved, decisions, conflicts, ScreeningStage.FULL_TEXT)
        return ft.ids_with(ScreeningVerdict.INCLUDE)

    # full-text items and stored documents
    def load_fulltext_items(self, project_id: str) -> list[FullTextItem]:
        rows = self.session.scalars(
            select(FullTextItemORM)
            .where(FullTextItemORM.project_id == project_id)
            .order_by(FullTextItemORM.record_id)
        ).all()
        return [FullTextItem.model_validate(r.payload) for r in rows]

    def get_fulltext_item(self, project_id: str, record_id: str) -> FullTextItem | None:
        row = self.session.scalars(
            select(FullTextItemORM).where(
                FullTextItemORM.project_id == project_id,
                FullTextItemORM.record_id == record_id,
            )
        ).first()
        return None if row is None else FullTextItem.model_validate(row.payload)

    def add_fulltext_document(
        self,
        project_id: str,
        record_id: str,
        *,
        kind: str,
        filename: str,
        sha256: str,
        size_bytes: int,
        source_kind: str,
        source_url: str | None = None,
        original_filename: str | None = None,
        recorded_by: str | None = None,
    ) -> None:
        exists = self.session.scalars(
            select(FullTextDocumentORM).where(
                FullTextDocumentORM.project_id == project_id,
                FullTextDocumentORM.record_id == record_id,
                FullTextDocumentORM.sha256 == sha256,
            )
        ).first()
        if exists is not None:
            return
        self.session.add(
            FullTextDocumentORM(
                project_id=project_id, record_id=record_id, kind=kind, filename=filename,
                original_filename=original_filename, sha256=sha256, size_bytes=size_bytes,
                source_kind=source_kind, source_url=source_url, recorded_by=recorded_by,
            )
        )  # fmt: skip
        self.session.flush()

    def records_with_document(self, project_id: str, sha256: str) -> set[str]:
        return set(
            self.session.scalars(
                select(FullTextDocumentORM.record_id).where(
                    FullTextDocumentORM.project_id == project_id,
                    FullTextDocumentORM.sha256 == sha256,
                )
            )
        )

    def document_original_name(self, project_id: str, sha256: str) -> str | None:
        return self.session.scalars(
            select(FullTextDocumentORM.original_filename).where(
                FullTextDocumentORM.project_id == project_id,
                FullTextDocumentORM.sha256 == sha256,
            )
        ).first()

    # master extraction table
    def save_extraction_item(self, project_id: str, item: ExtractionItem) -> None:
        row = self.session.get(ExtractionItemORM, item.item_id)
        if row is None:
            row = ExtractionItemORM(id=item.item_id, project_id=project_id)
            self.session.add(row)
        elif row.project_id != project_id:
            raise SysRevError("extraction item belongs to a different project")
        elif row.review_status != ReviewStatus.PENDING.value:
            raise SysRevError(
                f"item {item.item_id} was already {row.review_status}; corrections create a "
                "new item version"
            )
        if item.review_status != ReviewStatus.PENDING:
            self.require_active_reviewer(item.reviewer_id)
        row.record_id = item.record_id
        row.review_status = item.review_status.value
        row.data_status = item.data_status.value
        row.verification_status = item.verification.status.value if item.verification else None
        row.payload = item.model_dump(mode="json")
        self.session.flush()

    def supersede_reviewed_item(self, project_id: str, superseded: ExtractionItem) -> None:
        """Mark an accepted/rejected item superseded by a correction (Session 8). Only the
        review status and its stamps change; the earlier payload is kept in the append-only
        review event written by the caller."""
        row = self.session.get(ExtractionItemORM, superseded.item_id)
        if row is None or row.project_id != project_id:
            raise NotFoundError(f"extraction item {superseded.item_id} not found")
        if row.review_status not in (ReviewStatus.ACCEPTED.value, ReviewStatus.REJECTED.value):
            raise SysRevError(f"item {superseded.item_id} is {row.review_status}")
        if superseded.review_status != ReviewStatus.SUPERSEDED:
            raise SysRevError("a reviewed item can only become superseded")
        self.require_active_reviewer(superseded.reviewer_id)
        before = ExtractionItem.model_validate(row.payload)
        if superseded.model_dump(exclude=_REVIEW_STAMPS) != before.model_dump(
            exclude=_REVIEW_STAMPS
        ):
            raise SysRevError("superseding may not change the item's content")
        row.review_status = ReviewStatus.SUPERSEDED.value
        row.payload = superseded.model_dump(mode="json")
        self.session.flush()

    def get_extraction_item(self, project_id: str, item_id: str) -> ExtractionItem:
        row = self.session.get(ExtractionItemORM, item_id)
        if row is None or row.project_id != project_id:
            raise NotFoundError(f"extraction item {item_id} not found")
        return ExtractionItem.model_validate(row.payload)

    def load_extraction_items(
        self, project_id: str, *, include_superseded: bool = False
    ) -> list[ExtractionItem]:
        stmt = select(ExtractionItemORM).where(ExtractionItemORM.project_id == project_id)
        if not include_superseded:
            stmt = stmt.where(ExtractionItemORM.review_status != ReviewStatus.SUPERSEDED.value)
        rows = self.session.scalars(stmt.order_by(ExtractionItemORM.record_id)).all()
        return [ExtractionItem.model_validate(r.payload) for r in rows]

    def add_review_event(self, project_id: str, event: ExtractionReviewEvent) -> None:
        if event.reviewer_id is not None:
            self.require_active_reviewer(event.reviewer_id)
        self.session.add(
            ExtractionReviewEventORM(
                id=event.event_id, project_id=project_id, item_id=event.item_id,
                action=event.action, actor=event.actor, reviewer_id=event.reviewer_id,
                occurred_at=event.occurred_at, payload=event.model_dump(mode="json"),
            )
        )  # fmt: skip
        self.session.flush()

    def list_review_events(
        self, project_id: str, item_id: str | None = None
    ) -> list[ExtractionReviewEvent]:
        stmt = select(ExtractionReviewEventORM).where(
            ExtractionReviewEventORM.project_id == project_id
        )
        if item_id is not None:
            stmt = stmt.where(ExtractionReviewEventORM.item_id == item_id)
        rows = self.session.scalars(stmt.order_by(ExtractionReviewEventORM.seq)).all()
        return [ExtractionReviewEvent.model_validate(r.payload) for r in rows]

    # report → study mapping and qualitative-only inclusion
    def save_study_link(self, project_id: str, link: StudyReportLink) -> None:
        self.require_active_reviewer(link.reviewer_id)
        self.get_record(project_id, link.record_id)
        row = self.session.scalars(
            select(StudyReportLinkORM).where(
                StudyReportLinkORM.project_id == project_id,
                StudyReportLinkORM.record_id == link.record_id,
            )
        ).first() or StudyReportLinkORM(project_id=project_id, record_id=link.record_id)
        row.study_id = link.study_id
        row.payload = link.model_dump(mode="json")
        self.session.add(row)
        self.session.flush()

    def load_study_links(self, project_id: str) -> list[StudyReportLink]:
        rows = self.session.scalars(
            select(StudyReportLinkORM)
            .where(StudyReportLinkORM.project_id == project_id)
            .order_by(StudyReportLinkORM.record_id)
        ).all()
        return [StudyReportLink.model_validate(r.payload) for r in rows]

    def remove_study_link(self, project_id: str, record_id: str, *, reviewer_id: str) -> None:
        """The report becomes its own study again (Session 8). History is kept by the caller
        in the append-only governance log."""
        self.require_active_reviewer(reviewer_id)
        row = self.session.scalars(
            select(StudyReportLinkORM).where(
                StudyReportLinkORM.project_id == project_id,
                StudyReportLinkORM.record_id == record_id,
            )
        ).first()
        if row is None:
            raise NotFoundError(f"report {record_id} is not linked to a study")
        self.session.delete(row)
        self.session.flush()

    def study_ids(self, project_id: str, record_ids: set[str] | None = None) -> dict[str, str]:
        """``record_id -> study_id``; an unlinked report is its own study."""
        links = dict(
            self.session.execute(
                select(StudyReportLinkORM.record_id, StudyReportLinkORM.study_id).where(
                    StudyReportLinkORM.project_id == project_id
                )
            ).all()
        )
        ids = record_ids if record_ids is not None else set(links)
        return {rid: links.get(rid, rid) for rid in ids}

    def save_unavailability(self, project_id: str, entry: QuantitativeUnavailability) -> None:
        self.require_active_reviewer(entry.reviewer_id)
        self.get_record(project_id, entry.record_id)
        self.session.merge(
            QuantitativeUnavailabilityORM(
                id=entry.unavailability_id, project_id=project_id, record_id=entry.record_id,
                outcome_id=entry.outcome_id, payload=entry.model_dump(mode="json"),
            )
        )  # fmt: skip
        self.session.flush()

    def save_contribution_decision(
        self, project_id: str, decision: CohortContributionDecision
    ) -> None:
        reviewer = self.require_active_reviewer(decision.reviewer_id)
        if decision.decided_by != reviewer.display_name:
            raise SysRevError("decided_by does not match the enrolled reviewer")
        for rid in decision.record_ids:
            self.get_record(project_id, rid)
        self.session.add(
            CohortContributionDecisionORM(
                id=decision.decision_id, project_id=project_id, study_id=decision.study_id,
                outcome_id=decision.outcome_id, payload=decision.model_dump(mode="json"),
            )
        )  # fmt: skip
        self.session.flush()

    def load_contribution_decisions(self, project_id: str) -> list[CohortContributionDecision]:
        rows = self.session.scalars(
            select(CohortContributionDecisionORM).where(
                CohortContributionDecisionORM.project_id == project_id
            )
        ).all()
        return [CohortContributionDecision.model_validate(r.payload) for r in rows]

    def cohort_ids(self, project_id: str, record_ids: set[str]) -> dict[str, str]:
        """``record_id -> cohort label``: reports linked to one study, or judged by a human
        to share participants (cohort-overlap warnings resolved as overlapping), share a
        label. Used to prevent the same patients from being pooled twice."""
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

        studies = self.study_ids(project_id)
        by_study: dict[str, list[str]] = {}
        for rid, sid in studies.items():
            by_study.setdefault(sid, []).append(rid)
        for members in by_study.values():
            for other in members[1:]:
                union(members[0], other)
        overlapping = {
            OverlapResolution.OVERLAPPING_KEEP_A,
            OverlapResolution.OVERLAPPING_KEEP_B,
            OverlapResolution.OVERLAPPING_DIFFERENT_OUTCOMES,
        }
        for w in self.load_overlap_warnings(project_id):
            if w.resolution in overlapping:
                union(w.record_id_a, w.record_id_b)
        labels: dict[str, str] = {}
        for rid in record_ids:
            root = find(rid)
            members = sorted(r for r in parent if find(r) == root) or [rid]
            labels[rid] = next((studies[m] for m in members if m in studies), min(members))
        return labels

    def double_counting_conflicts(self, project_id: str) -> list[str]:
        """Outcome data from more than one report of the same cohort, unresolved by a human.

        Arms are grouped by cohort, outcome, arm role, timepoint and estimator. A group fed
        by several reports is a conflict unless the redundant reports are recorded as
        unavailable for that outcome (``overlapping_cohort``) or a human documented that
        they contribute non-overlapping samples. Different outcomes, timepoints and
        estimators never collide.
        """
        arms = self.load_extracted_arms(project_id)
        excluded = {(u.record_id, u.outcome_id) for u in self.load_unavailability(project_id)}
        arms = [
            a
            for a in arms
            if (a.record_id, a.outcome_id) not in excluded and (a.record_id, None) not in excluded
        ]
        cohorts = self.cohort_ids(project_id, {a.record_id for a in arms})
        groups: dict[tuple[str, str, str, str, str], set[str]] = {}
        for a in arms:
            key = (cohorts[a.record_id], a.outcome_id, a.arm_role.value, a.timepoint or "",
                   a.estimate_type.value if a.estimate_type else "raw counts/summary")  # fmt: skip
            groups.setdefault(key, set()).add(a.record_id)
        declared = [
            (d.outcome_id, set(d.record_ids)) for d in self.load_contribution_decisions(project_id)
        ]
        problems: list[str] = []
        for key, records in sorted(groups.items()):
            if len(records) < 2:
                continue
            if any(o == key[1] and records <= ids for o, ids in declared):
                continue
            problems.append(
                f"study/cohort {key[0]}: outcome {key[1]} ({key[2]}, {key[3] or 'no timepoint'},"
                f" {key[4]}) has arms from {len(records)} reports {sorted(records)} — keep one "
                "report or document non-overlapping samples"
            )
        return problems

    def load_unavailability(self, project_id: str) -> list[QuantitativeUnavailability]:
        rows = self.session.scalars(
            select(QuantitativeUnavailabilityORM).where(
                QuantitativeUnavailabilityORM.project_id == project_id
            )
        ).all()
        return [QuantitativeUnavailability.model_validate(r.payload) for r in rows]

    # ------------------------------------------------------------------ Stage 5 statistics
    def list_projects(self) -> list[ProjectORM]:
        return list(self.session.scalars(select(ProjectORM).order_by(ProjectORM.created_at)))

    # ------------------------------------------------------------------ AI settings & usage
    def save_ai_settings(self, project_id: str, settings: ProjectAISettings) -> int:
        """Append a new settings version (no credentials are part of the payload)."""
        self.get_project(project_id)
        latest = self.session.scalar(
            select(func.max(ProjectAISettingsORM.version)).where(
                ProjectAISettingsORM.project_id == project_id
            )
        )
        version = (latest or 0) + 1
        self.session.add(
            ProjectAISettingsORM(
                project_id=project_id, version=version, payload=settings.model_dump(mode="json")
            )
        )
        self.session.flush()
        return version

    def latest_ai_settings(self, project_id: str) -> ProjectAISettings | None:
        row = self.session.scalars(
            select(ProjectAISettingsORM)
            .where(ProjectAISettingsORM.project_id == project_id)
            .order_by(ProjectAISettingsORM.seq.desc())
            .limit(1)
        ).first()
        return ProjectAISettings.model_validate(row.payload) if row else None

    def record_llm_call(self, record: LLMCallRecord) -> None:
        self.get_project(record.project_id)
        self.session.add(
            LLMCallORM(
                id=record.call_id, project_id=record.project_id, task=record.task,
                provider=record.provider, requested_model=record.requested_model,
                status=record.status, call_key=record.call_key,
                total_tokens=record.total_tokens, cost_usd_estimate=record.cost_usd_estimate,
                payload=record.model_dump(mode="json"),
            )
        )  # fmt: skip
        self.session.flush()

    def load_llm_calls(self, project_id: str) -> list[LLMCallRecord]:
        rows = self.session.scalars(
            select(LLMCallORM).where(LLMCallORM.project_id == project_id).order_by(LLMCallORM.seq)
        )
        return [LLMCallRecord.model_validate(r.payload) for r in rows]

    def record_persisted_result(
        self, project_id: str, records: Sequence[LLMCallRecord], result_ref: str
    ) -> None:
        """Link successful calls to the validated result they produced (after it was saved).

        Only these links suppress a repeat of an identical request; a provider reply that
        failed validation leaves no link, so the request can simply be retried.
        """
        for record in records:
            if record.status != "ok" or record.project_id != project_id:
                continue
            self.session.add(
                LLMCallOutcomeORM(
                    project_id=project_id, call_id=record.call_id, call_key=record.call_key,
                    result_ref=result_ref,
                )
            )  # fmt: skip
        self.session.flush()

    def has_persisted_result(self, project_id: str, call_key: str) -> bool:
        return (
            self.session.scalars(
                select(LLMCallOutcomeORM.seq).where(
                    LLMCallOutcomeORM.project_id == project_id,
                    LLMCallOutcomeORM.call_key == call_key,
                )
            ).first()
            is not None
        )

    def llm_usage_totals(self, project_id: str) -> dict[str, float | int | str]:
        """Project totals over successful and failed calls (estimates, not invoices)."""
        calls = self.load_llm_calls(project_id)
        ok = [c for c in calls if c.status == "ok"]
        return {
            "calls": len(calls),
            "successful_calls": len(ok),
            "failed_calls": sum(c.status == "error" for c in calls),
            "tokens": sum(c.total_tokens for c in calls),
            "input_tokens": sum(c.input_tokens or 0 for c in calls),
            "output_tokens": sum(c.output_tokens or 0 for c in calls),
            "reasoning_tokens": sum(c.reasoning_tokens or 0 for c in calls),
            "cache_read_tokens": sum(c.cache_read_tokens or 0 for c in calls),
            "cache_write_tokens": sum(c.cache_write_tokens or 0 for c in calls),
            "cost_usd_estimate": round(sum(c.cost_usd_estimate or 0.0 for c in calls), 9),
            "unpriced_successful_calls": sum(c.cost_usd_estimate is None for c in ok),
            # Session 8: never present an unknown price as zero
            "cost_status": (
                "no successful calls"
                if not ok
                else "unavailable (no call had a verified list price)"
                if all(c.cost_usd_estimate is None for c in ok)
                else "partial (lower bound: some calls had no list price)"
                if any(c.cost_usd_estimate is None for c in ok)
                else "complete estimate"
            ),
        }

    def save_ai_verification(self, project_id: str, verification: AIVerification) -> None:
        item = self.get_extraction_item(project_id, verification.item_id)
        if item.record_id != verification.record_id:
            raise SysRevError("verification record_id does not match the item")
        self.session.add(
            AIVerificationORM(
                id=verification.verification_id, project_id=project_id,
                item_id=verification.item_id, verdict=verification.verdict,
                payload=verification.model_dump(mode="json"),
            )
        )  # fmt: skip
        self.session.flush()

    def load_ai_verifications(self, project_id: str) -> list[AIVerification]:
        rows = self.session.scalars(
            select(AIVerificationORM)
            .where(AIVerificationORM.project_id == project_id)
            .order_by(AIVerificationORM.seq)
        )
        return [AIVerification.model_validate(r.payload) for r in rows]

    def save_analysis_plan(self, project_id: str, plan: AnalysisPlan) -> None:
        """Append a plan version. Approval must come from an active enrolled reviewer, and an
        approved plan must be consistent with the protocol's outcomes."""
        project = self.get_project(project_id)
        if plan.status == "approved":
            if project.phase in (
                Phase.STAGE6_MANUSCRIPT.value,
                Phase.STAGE6_AUDIT.value,
                Phase.COMPLETED.value,
            ):
                raise SysRevError(
                    f"a new analysis plan cannot be approved in {project.phase}: it would "
                    "invalidate the statistics output, and the pipeline has no route back to "
                    "Stage 5 to re-run them (Session 8 safeguard)"
                )
            reviewer = self.require_active_reviewer(plan.approved_reviewer_id)
            if plan.approved_by != reviewer.display_name:
                raise SysRevError("approved_by does not match the enrolled reviewer")
            protocol = self.latest_protocol(project_id)
            if protocol is None:
                raise SysRevError("no protocol: an analysis plan needs protocol outcomes")
            outcomes = {o.outcome_id: o for o in protocol.outcomes}
            problems: list[str] = []
            for spec in plan.analyses:
                outcome = outcomes.get(spec.outcome_id)
                if outcome is None:
                    problems.append(f"{spec.analysis_id}: unknown outcome {spec.outcome_id}")
                else:
                    problems.extend(
                        f"{spec.analysis_id}: {p}" for p in spec.check_against_outcome(outcome)
                    )
            if problems:
                raise SysRevError("analysis plan contradicts the protocol: " + "; ".join(problems))
        exists = self.session.scalars(
            select(AnalysisPlanORM).where(
                AnalysisPlanORM.project_id == project_id,
                AnalysisPlanORM.id == plan.plan_id,
                AnalysisPlanORM.version == plan.version,
                AnalysisPlanORM.status == plan.status,
            )
        ).first()
        if exists is not None:
            raise SysRevError("this plan version is already stored; plans are append-only")
        self.session.add(
            AnalysisPlanORM(
                id=plan.plan_id, project_id=project_id, version=plan.version,
                status=plan.status, approved_reviewer_id=plan.approved_reviewer_id,
                sha256=plan.sha256(), payload=plan.model_dump(mode="json"),
            )
        )  # fmt: skip
        self.session.flush()

    def latest_analysis_plan(
        self, project_id: str, *, approved_only: bool = True
    ) -> AnalysisPlan | None:
        stmt = select(AnalysisPlanORM).where(AnalysisPlanORM.project_id == project_id)
        if approved_only:
            stmt = stmt.where(AnalysisPlanORM.status == "approved")
        row = self.session.scalars(stmt.order_by(AnalysisPlanORM.seq.desc()).limit(1)).first()
        if row is None:
            return None
        plan = AnalysisPlan.model_validate(row.payload)
        if plan.sha256() != row.sha256:
            raise SysRevError(f"stored analysis plan {plan.plan_id} fails its integrity hash")
        return plan

    def load_stage5_inputs(self, project_id: str) -> AnalysisInputs:
        """Every persisted fact the statistics engine may read (all human-reviewed)."""
        protocol = self.latest_protocol(project_id)
        arms = self.load_extracted_arms(project_id)
        eligible = self.included_record_ids(project_id, ScreeningStage.FULL_TEXT)
        record_ids = eligible | {a.record_id for a in arms}
        labels: dict[str, str] = {}
        for rid in sorted(record_ids):
            rec = self.get_record(project_id, rid)
            author = rec.authors[0].last_name if rec.authors and rec.authors[0].last_name else None
            year = rec.publication_year
            labels[rid] = " ".join(str(x) for x in (author or rid[:12], year) if x)
        return AnalysisInputs(
            outcomes={} if protocol is None else {o.outcome_id: o for o in protocol.outcomes},
            arms=arms,
            eligible_record_ids=frozenset(eligible),
            study_of=self.study_ids(project_id, record_ids),
            cohort_of=self.cohort_ids(project_id, record_ids),
            labels=labels,
            unavailability=self.load_unavailability(project_id),
            contributions=self.load_contribution_decisions(project_id),
        )

    def record_stats_output(self, project_id: str, output: StatsOutput) -> str:
        """Store a ``stats_output.json`` and make it the project's statistics artifact.

        Refused unless the project is in Stage 5 and the artifact verifies against the
        current approved plan and the current approved extraction data, **and** its analyses
        are exactly what the deterministic engine computes from them (numerical provenance:
        the plan is re-run here once; status checks later only re-verify hashes).
        """
        project = self.get_project(project_id)
        if project.phase != Phase.STAGE5_META_ANALYSIS.value:
            raise SysRevError(f"statistics can only be recorded in Stage 5 (phase={project.phase})")
        content = output.to_canonical_json()
        problems = self._verify_stats(project_id, output, content)
        if not problems:
            problems = self._verify_provenance(project_id, output)
        if problems:
            raise SysRevError("statistics artifact rejected: " + "; ".join(problems))
        sha = output.sha256()
        self.session.add(
            StatsOutputORM(
                id=output.output_id, project_id=project_id, plan_id=output.plan_id,
                plan_sha256=output.plan_sha256, input_sha256=output.input_sha256, sha256=sha,
                content=content,
            )
        )  # fmt: skip
        project.stats_output_sha256 = sha
        self.session.flush()
        return sha

    def _verify_provenance(self, project_id: str, output: StatsOutput) -> list[str]:
        """Recompute every analysis from the approved plan and current data; any difference
        in any field (bit-for-bit, via the results fingerprint) rejects the artifact."""
        plan = self.latest_analysis_plan(project_id)
        if plan is None:  # pragma: no cover - _verify_stats reports this first
            return ["no approved analysis plan"]
        try:
            expected = run_plan(
                plan,
                self.load_stage5_inputs(project_id),
                project_id=project_id,
                generated_at=output.generated_at,
            )
        except SysRevError as exc:
            return [f"the approved plan cannot be recomputed from current data: {exc}"]
        if expected.results_sha256 == output.results_sha256:
            return []
        diffs = _first_differences(
            [a.model_dump(mode="json") for a in expected.analyses],
            [a.model_dump(mode="json") for a in output.analyses],
            "analyses",
        )
        return [
            "numerical results do not match a deterministic recomputation by the statistics "
            "engine (first differences: " + "; ".join(diffs[:5]) + ")"
        ]

    def current_stats_output(self, project_id: str) -> StatsOutput | None:
        project = self.get_project(project_id)
        if project.stats_output_sha256 is None:
            return None
        row = self._stats_row(project_id, project.stats_output_sha256)
        return None if row is None else StatsOutput.model_validate_json(row.content)

    def _stats_row(self, project_id: str, sha: str) -> StatsOutputORM | None:
        return self.session.scalars(
            select(StatsOutputORM)
            .where(StatsOutputORM.project_id == project_id, StatsOutputORM.sha256 == sha)
            .order_by(StatsOutputORM.seq.desc())
            .limit(1)
        ).first()

    def stats_output_problems(self, project_id: str) -> list[str]:
        """Why the project's statistics artifact is not a valid basis for Stage 6."""
        project = self.get_project(project_id)
        if project.stats_output_sha256 is None:
            return []
        row = self._stats_row(project_id, project.stats_output_sha256)
        if row is None:
            return ["the recorded stats_output.json hash has no stored artifact"]
        if hashlib.sha256(row.content.encode("utf-8")).hexdigest() != row.sha256:
            return ["stored stats_output.json content does not match its SHA-256"]
        try:
            output = StatsOutput.model_validate_json(row.content)
        except ValueError as exc:
            return [f"stored stats_output.json fails validation: {exc}"]
        return self._verify_stats(project_id, output, row.content)

    def _verify_stats(self, project_id: str, output: StatsOutput, content: str) -> list[str]:
        problems: list[str] = []
        if output.to_canonical_json() != content:
            problems.append("stats_output.json is not in canonical form")
        if output.project_id != project_id:
            problems.append("stats_output.json belongs to another project")
        plan = self.latest_analysis_plan(project_id)
        if plan is None:
            problems.append("no approved analysis plan")
        else:
            if (output.plan_id, output.plan_sha256) != (plan.plan_id, plan.sha256()):
                problems.append("computed from a plan that is not the latest approved plan")
            if [a.analysis_id for a in output.analyses] != [a.analysis_id for a in plan.analyses]:
                problems.append("analyses do not match the approved plan")
        if output.input_sha256 != self.load_stage5_inputs(project_id).sha256():
            problems.append(
                "approved extraction data changed after the statistics were computed; re-run "
                "Stage 5"
            )
        return problems

    # ------------------------------------------------------------------ manuscript (Stage 6)
    STAGE6_EDITABLE = (Phase.STAGE6_MANUSCRIPT.value, Phase.STAGE6_AUDIT.value)

    def save_manuscript_version(self, version: ManuscriptVersion) -> None:
        """Append a version. Versions are numbered consecutively, only in Stage 6, and a human
        edit must come from an active enrolled reviewer."""
        project = self.get_project(version.project_id)
        if project.phase not in self.STAGE6_EDITABLE:
            raise SysRevError(f"manuscripts can only be written in Stage 6 (phase={project.phase})")
        latest = self.latest_manuscript_version(version.project_id)
        expected = 1 if latest is None else latest.version + 1
        if version.version != expected:
            raise SysRevError(f"next manuscript version must be {expected}")
        if latest is not None and version.parent_version is None:
            raise SysRevError("a new version names its parent version")
        if version.actor_type == ActorType.HUMAN or version.reviewer_id is not None:
            reviewer = self.require_active_reviewer(version.reviewer_id)
            if version.created_by != reviewer.display_name:
                raise SysRevError("created_by does not match the enrolled reviewer")
        self.session.add(
            ManuscriptVersionORM(
                project_id=version.project_id, version=version.version,
                content_sha256=version.content_sha256, actor_type=version.actor_type.value,
                reviewer_id=version.reviewer_id, payload=version.model_dump(mode="json"),
            )
        )  # fmt: skip
        project.manuscript_version = version.version
        self.session.flush()

    def _version_from_row(self, row: ManuscriptVersionORM) -> ManuscriptVersion:
        version = ManuscriptVersion.model_validate(
            {k: v for k, v in row.payload.items() if k != "content_sha256"}
        )
        if version.content_sha256 != row.content_sha256:
            raise SysRevError(f"stored manuscript version {row.version} fails its content hash")
        return version

    def latest_manuscript_version(self, project_id: str) -> ManuscriptVersion | None:
        row = self.session.scalars(
            select(ManuscriptVersionORM)
            .where(ManuscriptVersionORM.project_id == project_id)
            .order_by(ManuscriptVersionORM.version.desc())
            .limit(1)
        ).first()
        return None if row is None else self._version_from_row(row)

    def get_manuscript_version(self, project_id: str, version: int) -> ManuscriptVersion:
        row = self.session.scalars(
            select(ManuscriptVersionORM).where(
                ManuscriptVersionORM.project_id == project_id,
                ManuscriptVersionORM.version == version,
            )
        ).first()
        if row is None:
            raise NotFoundError(f"manuscript version {version} not found")
        return self._version_from_row(row)

    def list_manuscript_versions(self, project_id: str) -> list[ManuscriptVersion]:
        rows = self.session.scalars(
            select(ManuscriptVersionORM)
            .where(ManuscriptVersionORM.project_id == project_id)
            .order_by(ManuscriptVersionORM.version)
        )
        return [self._version_from_row(r) for r in rows]

    def save_reference_entry(self, project_id: str, entry: ReferenceEntry) -> None:
        """Register imported literature (fetched by identifier). Included-study references are
        derived from stored records and are never written here."""
        self.get_project(project_id)
        if entry.origin != "imported_literature":
            raise SysRevError("only imported literature is stored; study references are derived")
        reviewer = self.require_active_reviewer(entry.reviewer_id)
        if entry.added_by != reviewer.display_name:
            raise SysRevError("added_by does not match the enrolled reviewer")
        self.session.add(ReferenceEntryORM(project_id=project_id, ref_key=entry.ref_key,
                                           reviewer_id=entry.reviewer_id,
                                           payload=entry.model_dump(mode="json")))  # fmt: skip
        self.session.flush()

    def load_reference_entries(self, project_id: str) -> list[ReferenceEntry]:
        rows = self.session.scalars(
            select(ReferenceEntryORM)
            .where(ReferenceEntryORM.project_id == project_id)
            .order_by(ReferenceEntryORM.seq)
        )
        latest = {r.ref_key: ReferenceEntry.model_validate(r.payload) for r in rows}
        return [latest[k] for k in sorted(latest)]

    def save_reference_verification(self, project_id: str, v: ReferenceVerification) -> None:
        self.get_project(project_id)
        if v.source == "human_attestation":
            self._check_attestation(project_id, v)
        self.session.add(ReferenceVerificationORM(
            id=v.verification_id, project_id=project_id, ref_key=v.ref_key, status=v.status,
            payload=v.model_dump(mode="json"),
        ))  # fmt: skip
        self.session.flush()

    def _check_attestation(self, project_id: str, v: ReferenceVerification) -> None:
        """Human identity attestation: only for an included report with no DOI/PMID, whose
        linked PDF identity was confirmed, against the reference's current metadata."""
        from src.manuscript.evidence import record_reference

        reviewer = self.require_active_reviewer(v.reviewer_id)
        if v.checked_by != reviewer.display_name:
            raise SysRevError("checked_by does not match the enrolled reviewer")
        included = self.included_record_ids(project_id, ScreeningStage.FULL_TEXT)
        rid = next((r for r in included if f"study-{r}"[:64] == v.ref_key), None)
        if rid is None:
            raise SysRevError("attestation is only for references of included reports")
        record = self.get_record(project_id, rid)
        if record.doi or record.pmid:
            raise SysRevError("this reference has a DOI/PMID: verify it live (PubMed/Crossref)")
        if v.metadata_sha256 != record_reference(record).metadata_sha256():
            raise SysRevError("attestation does not match the reference's current metadata")
        item = self.get_fulltext_item(project_id, rid)
        if item is None or not item.has_usable_pdf:
            raise SysRevError(
                "the report has no PDF with a confirmed identity to attest the reference against"
            )
        if v.pdf_sha256 != item.pdf_sha256:
            raise SysRevError("the attestation does not name the report's current confirmed PDF")

    def load_reference_verifications(self, project_id: str) -> list[ReferenceVerification]:
        rows = self.session.scalars(
            select(ReferenceVerificationORM)
            .where(ReferenceVerificationORM.project_id == project_id)
            .order_by(ReferenceVerificationORM.seq)
        )
        return [ReferenceVerification.model_validate(r.payload) for r in rows]

    def save_evidence_excerpt(self, project_id: str, excerpt: EvidenceExcerpt) -> None:
        reviewer = self.require_active_reviewer(excerpt.reviewer_id)
        if excerpt.added_by != reviewer.display_name:
            raise SysRevError("added_by does not match the enrolled reviewer")
        self.session.add(EvidenceExcerptORM(
            id=excerpt.excerpt_id, project_id=project_id, ref_key=excerpt.ref_key,
            reviewer_id=excerpt.reviewer_id, payload=excerpt.model_dump(mode="json"),
        ))  # fmt: skip
        self.session.flush()

    def load_evidence_excerpts(self, project_id: str) -> list[EvidenceExcerpt]:
        rows = self.session.scalars(
            select(EvidenceExcerptORM)
            .where(EvidenceExcerptORM.project_id == project_id)
            .order_by(EvidenceExcerptORM.seq)
        )
        return [EvidenceExcerpt.model_validate(r.payload) for r in rows]

    # ------------------------------------------------------------------ audit
    def save_audit_report(self, report: AuditReport) -> None:
        """Store an audit (append-only). Refused unless it is exactly what the Auditor
        computes for the stored manuscript version and the current evidence — a hand-built or
        selectively populated report cannot be recorded."""
        from src.manuscript.service import verify_audit_report

        problems = verify_audit_report(self, report)
        if problems:
            raise SysRevError("audit report rejected: " + "; ".join(problems))
        self.session.add(
            AuditReportORM(
                id=report.report_id,
                project_id=report.project_id,
                manuscript_version=report.manuscript_version,
                passed=report.passed,
                generated_at=report.generated_at,
                payload=report.model_dump(mode="json"),
            )
        )
        self.session.flush()

    def latest_audit_report(self, project_id: str) -> AuditReport | None:
        row = self.session.scalars(
            select(AuditReportORM)
            .where(AuditReportORM.project_id == project_id)
            .order_by(AuditReportORM.generated_at.desc(), AuditReportORM.id)
            .limit(1)
        ).first()
        return None if row is None else AuditReport.model_validate(row.payload)

    def list_audit_reports(self, project_id: str) -> list[AuditReport]:
        rows = self.session.scalars(
            select(AuditReportORM)
            .where(AuditReportORM.project_id == project_id)
            .order_by(AuditReportORM.generated_at)
        )
        return [AuditReport.model_validate(r.payload) for r in rows]

    def get_audit_report(self, project_id: str, report_id: str) -> AuditReport:
        row = self.session.get(AuditReportORM, report_id)
        if row is None or row.project_id != project_id:
            raise NotFoundError(f"audit report {report_id} not found")
        return AuditReport.model_validate(row.payload)

    def save_acknowledgement(self, ack: FindingAcknowledgement) -> None:
        """A human's assessment of a REQUIRES_HUMAN_REVIEW finding (blocking ones refused)."""
        reviewer = self.require_active_reviewer(ack.reviewer_id)
        if ack.reviewer_name != reviewer.display_name:
            raise SysRevError("reviewer_name does not match the enrolled reviewer")
        report = self.get_audit_report(ack.project_id, ack.audit_report_id)
        finding = next((f for f in report.findings if f.finding_id == ack.finding_id), None)
        if finding is None:
            raise NotFoundError(f"finding {ack.finding_id} is not in audit {report.report_id}")
        if finding.status != CheckStatus.REQUIRES_HUMAN_REVIEW:
            raise SysRevError(
                f"finding {ack.finding_id} is {finding.status}: only findings that require human "
                "review can be acknowledged; blocking findings must be fixed"
            )
        if ack.manuscript_sha256 != report.manuscript_sha256 or ack.evidence_sha256 != (
            fingerprint_sha256(report.evidence_fingerprint)
        ):
            raise SysRevError("acknowledgement does not match the audited manuscript/evidence")
        self.session.add(FindingAcknowledgementORM(
            id=ack.ack_id, project_id=ack.project_id, audit_report_id=ack.audit_report_id,
            finding_id=ack.finding_id, reviewer_id=ack.reviewer_id,
            payload=ack.model_dump(mode="json"),
        ))  # fmt: skip
        self.session.flush()

    def load_acknowledgements(self, project_id: str) -> list[FindingAcknowledgement]:
        rows = self.session.scalars(
            select(FindingAcknowledgementORM)
            .where(FindingAcknowledgementORM.project_id == project_id)
            .order_by(FindingAcknowledgementORM.seq)
        )
        return [FindingAcknowledgement.model_validate(r.payload) for r in rows]

    def save_manuscript_approval(self, approval: ManuscriptApproval) -> None:
        """Human sign-off. Refused unless the approval matches the current manuscript, a
        current passing audit, and every human-review finding is acknowledged."""
        from src.manuscript.service import approval_blockers

        reviewer = self.require_active_reviewer(approval.reviewer_id)
        if approval.reviewer_name != reviewer.display_name:
            raise SysRevError("reviewer_name does not match the enrolled reviewer")
        problems = approval_blockers(self, approval.project_id, approval)
        if problems:
            raise SysRevError("manuscript cannot be approved: " + "; ".join(problems))
        self.session.add(ManuscriptApprovalORM(
            id=approval.approval_id, project_id=approval.project_id,
            manuscript_version=approval.manuscript_version,
            audit_report_id=approval.audit_report_id, reviewer_id=approval.reviewer_id,
            payload=approval.model_dump(mode="json"),
        ))  # fmt: skip
        self.session.flush()

    def load_manuscript_approvals(self, project_id: str) -> list[ManuscriptApproval]:
        rows = self.session.scalars(
            select(ManuscriptApprovalORM)
            .where(ManuscriptApprovalORM.project_id == project_id)
            .order_by(ManuscriptApprovalORM.seq)
        )
        return [ManuscriptApproval.model_validate(r.payload) for r in rows]

    def save_claim_review(self, review: ClaimReview) -> None:
        self.session.add(ClaimReviewORM(id=review.review_id, project_id=review.project_id,
                                        audit_report_id=review.audit_report_id,
                                        payload=review.model_dump(mode="json")))  # fmt: skip
        self.session.flush()

    def load_claim_reviews(self, project_id: str) -> list[ClaimReview]:
        rows = self.session.scalars(
            select(ClaimReviewORM)
            .where(ClaimReviewORM.project_id == project_id)
            .order_by(ClaimReviewORM.seq)
        )
        return [ClaimReview.model_validate(r.payload) for r in rows]

    # ------------------------------------------------------------------ reporting (Session 7)
    def _add_reporting(
        self,
        project_id: str,
        kind: str,
        record_id: str,
        subject: str,
        reviewer_id: str | None,
        payload: dict[str, object],
    ) -> None:
        import json

        text = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        self.session.add(ReportingRecordORM(
            id=record_id, project_id=project_id, kind=kind, subject=subject[:128],
            reviewer_id=reviewer_id, payload=payload,
            payload_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        ))  # fmt: skip
        self.session.flush()

    def _load_reporting(self, project_id: str, kind: str) -> list[dict[str, object]]:
        rows = self.session.scalars(
            select(ReportingRecordORM)
            .where(ReportingRecordORM.project_id == project_id, ReportingRecordORM.kind == kind)
            .order_by(ReportingRecordORM.seq)
        )
        return [r.payload for r in rows]

    def _lock_for_reporting(self, project_id: str) -> None:
        """Take the database write lock for this transaction *before* the preconditions of a
        reporting decision are checked, so the state checked is the state the decision is
        stored against: no other writer can change records, decisions or evidence between
        the check and the commit (SQLite allows one writer; on other databases this locks the
        project row, which serialises reporting decisions)."""
        self.session.execute(
            text("UPDATE projects SET title = title WHERE id = :pid"), {"pid": project_id}
        )

    def _require_reporting_phase(self, project_id: str) -> None:
        phase = self.get_project(project_id).phase
        if phase not in self.STAGE6_EDITABLE:
            raise SysRevError(
                f"reporting decisions are recorded in Stage 6 (phase={phase}); they depend on "
                "the verified statistics output and the manuscript"
            )

    def _require_named_reviewer(self, reviewer_id: str | None, name: str) -> None:
        reviewer = self.require_active_reviewer(reviewer_id)
        if reviewer.display_name != name:
            raise SysRevError("the recorded name does not match the enrolled reviewer")

    def save_checklist_decision(self, decision: ChecklistDecision) -> None:
        self.save_checklist_decisions([decision])

    def save_checklist_decisions(self, decisions: list[ChecklistDecision]) -> None:
        """Reviewer decisions on checklist items, each re-verified against the latest
        manuscript version and the current evidence (basis hash recomputed). All or none are
        stored; a batch is verified against one evidence state."""
        from src.manuscript.evidence import build_evidence_package
        from src.reporting.service import checklist_decision_problems

        if not decisions:
            return
        pid = decisions[0].project_id
        if any(d.project_id != pid for d in decisions):
            raise SysRevError("a batch of checklist decisions belongs to one project")
        self._require_reporting_phase(pid)
        for d in decisions:
            self._require_named_reviewer(d.reviewer_id, d.reviewer_name)
        self._lock_for_reporting(pid)
        try:
            package = build_evidence_package(self, pid)
        except SysRevError as exc:
            raise SysRevError(f"checklist decision refused: evidence unavailable: {exc}") from exc
        problems = [
            f"{d.checklist} {d.item_id}: {p}"
            for d in decisions
            for p in checklist_decision_problems(self, d, package)
        ]
        if problems:
            raise SysRevError("checklist decision refused: " + "; ".join(problems))
        for d in decisions:
            self._add_reporting(pid, "checklist_decision", d.decision_id,
                                f"{d.checklist}:{d.item_id}", d.reviewer_id,
                                d.model_dump(mode="json"))  # fmt: skip

    def load_checklist_decisions(self, project_id: str) -> list[ChecklistDecision]:
        return [ChecklistDecision.model_validate(p)
                for p in self._load_reporting(project_id, "checklist_decision")]  # fmt: skip

    def save_grade_assessment(self, assessment: GradeAssessment) -> None:
        """A GRADE record. Human statuses need an active enrolled reviewer; every record is
        checked against the current analysis (input hash, applicability, baselines)."""
        from src.reporting.service import grade_problems

        self._require_reporting_phase(assessment.project_id)
        if assessment.reviewer_id is not None or assessment.actor_type == ActorType.HUMAN:
            self._require_named_reviewer(assessment.reviewer_id, assessment.created_by)
        self._lock_for_reporting(assessment.project_id)
        problems = grade_problems(self, assessment)
        if problems:
            raise SysRevError("GRADE record refused: " + "; ".join(problems))
        self._add_reporting(assessment.project_id, "grade_assessment", assessment.assessment_id,
                            assessment.analysis_id, assessment.reviewer_id,
                            assessment.model_dump(mode="json"))  # fmt: skip

    def load_grade_assessments(self, project_id: str) -> list[GradeAssessment]:
        return [GradeAssessment.model_validate(p)
                for p in self._load_reporting(project_id, "grade_assessment")]  # fmt: skip

    def save_synthesis_group(self, group: SynthesisGroup) -> None:
        from src.reporting.service import synthesis_problems

        self._require_reporting_phase(group.project_id)
        if group.reviewer_id is not None or group.actor_type == ActorType.HUMAN:
            self._require_named_reviewer(group.reviewer_id, group.created_by)
        self._lock_for_reporting(group.project_id)
        problems = synthesis_problems(self, group)
        if problems:
            raise SysRevError("synthesis refused: " + "; ".join(problems))
        self._add_reporting(group.project_id, "synthesis_group", uuid.uuid4().hex,
                            group.group_id, group.reviewer_id,
                            group.model_dump(mode="json"))  # fmt: skip

    def load_synthesis_groups(self, project_id: str) -> list[SynthesisGroup]:
        return [SynthesisGroup.model_validate(p)
                for p in self._load_reporting(project_id, "synthesis_group")]  # fmt: skip

    def save_swim_applicability(self, decision: SwimApplicability) -> None:
        from src.reporting.service import swim_applicability_problems

        self._require_reporting_phase(decision.project_id)
        self._require_named_reviewer(decision.reviewer_id, decision.reviewer_name)
        self._lock_for_reporting(decision.project_id)
        problems = swim_applicability_problems(self, decision)
        if problems:
            raise SysRevError("SWiM applicability decision refused: " + "; ".join(problems))
        self._add_reporting(decision.project_id, "swim_applicability", decision.decision_id,
                            "swim", decision.reviewer_id,
                            decision.model_dump(mode="json"))  # fmt: skip

    def load_swim_applicability(self, project_id: str) -> list[SwimApplicability]:
        return [SwimApplicability.model_validate(p)
                for p in self._load_reporting(project_id, "swim_applicability")]  # fmt: skip

    # ------------------------------------------------------------------ guards
    # ------------------------------------------------------------------ Session 8: models
    def save_catalogue_snapshot(self, snapshot: CatalogueSnapshot) -> None:
        if snapshot.reviewer_id is not None:
            reviewer = self.require_active_reviewer(snapshot.reviewer_id)
            if reviewer.display_name != snapshot.created_by:
                raise SysRevError("created_by does not match the enrolled reviewer")
        self.session.add(
            ModelCatalogueSnapshotORM(
                id=snapshot.snapshot_id, kind=snapshot.kind, sha256=snapshot.sha256,
                parent_sha256=snapshot.parent_sha256, created_by=snapshot.created_by,
                reviewer_id=snapshot.reviewer_id, payload=snapshot.model_dump(mode="json"),
            )
        )  # fmt: skip
        self.session.flush()

    def load_catalogue_snapshots(self) -> list[CatalogueSnapshot]:
        rows = self.session.scalars(
            select(ModelCatalogueSnapshotORM).order_by(ModelCatalogueSnapshotORM.seq)
        ).all()
        return [self._snapshot(r) for r in rows]

    def get_catalogue_snapshot(self, snapshot_id: str) -> CatalogueSnapshot:
        row = self.session.scalars(
            select(ModelCatalogueSnapshotORM).where(ModelCatalogueSnapshotORM.id == snapshot_id)
        ).first()
        if row is None:
            raise NotFoundError(f"catalogue snapshot {snapshot_id} not found")
        return self._snapshot(row)

    @staticmethod
    def _snapshot(row: ModelCatalogueSnapshotORM) -> CatalogueSnapshot:
        snapshot = CatalogueSnapshot.model_validate(row.payload)
        if snapshot.sha256 != row.sha256:
            raise SysRevError(f"catalogue snapshot {row.id} fails its integrity hash")
        return snapshot

    def catalogue_activations(self) -> list[ModelCatalogueActivationORM]:
        return list(
            self.session.scalars(
                select(ModelCatalogueActivationORM).order_by(ModelCatalogueActivationORM.seq)
            )
        )

    def active_catalogue(self, base: ModelCatalogue) -> ModelCatalogue:
        """The latest activated snapshot, or ``base`` (built-in + local override) if none."""
        latest = self.session.scalars(
            select(ModelCatalogueActivationORM)
            .order_by(ModelCatalogueActivationORM.seq.desc())
            .limit(1)
        ).first()
        if latest is None:
            return base
        snapshot = self.get_catalogue_snapshot(latest.snapshot_id)
        if snapshot.sha256 != latest.sha256:
            raise SysRevError("the active catalogue snapshot does not match its activation")
        return snapshot.catalogue()

    def latest_discovery(self) -> dict[str, ProviderDiscovery]:
        """The most recent listing attempt per provider (successful or not)."""
        out: dict[str, ProviderDiscovery] = {}
        for snapshot in self.load_catalogue_snapshots():
            for d in snapshot.discovery:
                out[d.provider] = d
        return out

    def activate_catalogue_snapshot(
        self,
        snapshot_id: str,
        *,
        base: ModelCatalogue,
        expected_sha256: str,
        expected_active_sha256: str,
        reviewer_id: str,
        note: str | None = None,
    ) -> ModelCatalogueActivationORM:
        """Activate a previewed snapshot. Refused (nothing saved) if the snapshot is not the
        one previewed, or if the active catalogue changed since the preview, or if the
        snapshot was built on another catalogue (it would silently drop those changes)."""
        reviewer = self.require_active_reviewer(reviewer_id)
        # Write lock before the checks (SQLite: one writer), as for reporting decisions.
        self.session.execute(text("UPDATE model_catalogue_activations SET note = note WHERE 1 = 0"))
        snapshot = self.get_catalogue_snapshot(snapshot_id)
        active = self.active_catalogue(base)
        if snapshot.sha256 != expected_sha256:
            raise SysRevError("the catalogue snapshot has changed since it was previewed")
        if active.sha256 != expected_active_sha256:
            raise SysRevError(
                "the active catalogue has changed since this preview was shown; reload the "
                "preview before activating"
            )
        if snapshot.parent_sha256 != active.sha256:
            raise SysRevError(
                "this snapshot was built on a different catalogue than the active one; refresh "
                "or verify again so that no intervening change is lost"
            )
        if snapshot.sha256 == active.sha256:
            raise SysRevError("nothing to activate: the snapshot equals the active catalogue")
        row = ModelCatalogueActivationORM(
            id=uuid.uuid4().hex, snapshot_id=snapshot.snapshot_id, sha256=snapshot.sha256,
            previous_sha256=active.sha256, reviewer_id=reviewer.id, note=note,
        )  # fmt: skip
        self.session.add(row)
        self.session.flush()
        return row

    # ------------------------------------------------------------------ Session 8: governance
    def add_governance_record(
        self,
        project_id: str,
        kind: str,
        subject: str,
        payload: dict[str, object],
        *,
        reviewer_id: str | None,
        record_id: str | None = None,
    ) -> str:
        import json

        if reviewer_id is not None:
            self.require_active_reviewer(reviewer_id)
        text_ = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                           default=str)  # fmt: skip
        rid = record_id or uuid.uuid4().hex
        self.session.add(GovernanceRecordORM(
            id=rid, project_id=project_id, kind=kind, subject=subject[:128],
            reviewer_id=reviewer_id, payload=json.loads(text_),
            payload_sha256=hashlib.sha256(text_.encode("utf-8")).hexdigest(),
        ))  # fmt: skip
        self.session.flush()
        return rid

    def governance_records(
        self, project_id: str, kind: str, subject: str | None = None
    ) -> list[dict[str, object]]:
        stmt = select(GovernanceRecordORM).where(
            GovernanceRecordORM.project_id == project_id, GovernanceRecordORM.kind == kind
        )
        if subject is not None:
            stmt = stmt.where(GovernanceRecordORM.subject == subject[:128])
        rows = self.session.scalars(stmt.order_by(GovernanceRecordORM.seq)).all()
        return [r.payload | {"_record_id": r.id, "_recorded_at": r.created_at.isoformat()}
                for r in rows]  # fmt: skip

    def lock_project(self, project_id: str) -> None:
        """Take the database write lock for this transaction before checking that the state a
        reviewer saw is still current (see ``_lock_for_reporting``; SQLite only)."""
        self._lock_for_reporting(project_id)

    def build_guard_context(self, project_id: str) -> GuardContext:
        """Derive every guard fact from persisted rows. Missing data counts as missing work."""
        project = self.get_project(project_id)
        protocol = self.latest_protocol(project_id)
        protocol_blockers = (
            ("protocol not drafted",) if protocol is None else tuple(protocol.approval_blockers())
        )
        if protocol is not None and not self._authenticated_approval(protocol):
            protocol_blockers += (
                "the current protocol version has not been approved by an authenticated, "
                "enrolled reviewer",
            )

        def count(stmt: Select[int]) -> int:
            return int(self.session.scalar(stmt) or 0)

        # ---- Stage 2: records, duplicates, title/abstract screening
        record_rows = self.session.execute(
            select(StudyRecordORM.id, StudyRecordORM.duplicate_of).where(
                StudyRecordORM.project_id == project_id
            )
        ).all()
        unique_ids = {rid for rid, dup in record_rows if dup is None}
        decisions = self.load_screening_decisions(project_id)
        conflicts = self.load_conflicts(project_id)
        ta = screening_status(unique_ids, decisions, conflicts, ScreeningStage.TITLE_ABSTRACT)
        ta_included = ta.ids_with(ScreeningVerdict.INCLUDE)

        unresolved_duplicates = count(
            select(func.count())
            .select_from(DuplicateCandidateORM)
            .where(
                DuplicateCandidateORM.project_id == project_id,
                DuplicateCandidateORM.resolution == "unresolved",
            )
        )
        warnings = count(
            select(func.count())
            .select_from(CohortOverlapWarningORM)
            .where(
                CohortOverlapWarningORM.project_id == project_id,
                CohortOverlapWarningORM.resolution == OverlapResolution.UNRESOLVED.value,
            )
        )

        # ---- Stage 3: full-text manifest (only title/abstract-included records matter)
        items = {i.record_id: i for i in self.load_fulltext_items(project_id)}
        fulltext_missing = len(ta_included - set(items))
        fulltext_blocking = sum(1 for i in items.values() if i.blocks_gate2)
        retrieved = {rid for rid in ta_included if rid in items and items[rid].is_retrieved}
        unobtainable = {
            rid
            for rid in ta_included
            if rid in items and items[rid].status == FullTextStatus.UNOBTAINABLE
        }

        # ---- Stage 4: full-text eligibility and extraction
        ft = screening_status(retrieved, decisions, conflicts, ScreeningStage.FULL_TEXT)
        ft_included = ft.ids_with(ScreeningVerdict.INCLUDE)
        invalid_arms = self.invalidated_arm_ids(project_id)
        arm_rows = self.session.execute(
            select(ExtractedArmORM.record_id, ExtractedArmORM.approved_by_human).where(
                ExtractedArmORM.project_id == project_id,
                ExtractedArmORM.id.notin_(invalid_arms),
            )
        ).all()
        arm_records = {rid for rid, _ in arm_rows}
        ft_unconfirmed = self._automated_only(ft.final, decisions, conflicts)
        studies = self.study_ids(project_id, ft_included | arm_records)
        studies_with_arms = {studies[rid] for rid in arm_records if rid in studies}
        qualitative_only = {
            u.record_id for u in self.load_unavailability(project_id) if u.outcome_id is None
        }
        without_extraction = {
            rid
            for rid in ft_included
            if rid not in arm_records
            and rid not in qualitative_only
            and studies[rid] not in studies_with_arms
        }
        approved_rob_studies = {
            studies.get(a.record_id, a.record_id)
            for a in self.load_rob(project_id)
            if a.approved_by_human
        }
        studies_without_rob = {studies[rid] for rid in ft_included} - approved_rob_studies
        without_usable_pdf = {
            rid for rid in without_extraction if rid not in items or not items[rid].has_usable_pdf
        }
        pending_items = count(
            select(func.count())
            .select_from(ExtractionItemORM)
            .where(
                ExtractionItemORM.project_id == project_id,
                ExtractionItemORM.review_status == ReviewStatus.PENDING.value,
            )
        )
        # Defence in depth: the NOT NULL FK already forbids this; count anyway.
        no_evidence = count(
            select(func.count())
            .select_from(ExtractedValueORM)
            .join(ExtractedArmORM)
            .where(
                ExtractedArmORM.project_id == project_id,
                ExtractedArmORM.id.notin_(invalid_arms),
                ExtractedValueORM.evidence_id.is_(None),
            )
        )

        # ---- PRISMA snapshot cross-checked against the persisted facts above
        prisma = self.load_prisma(project_id)
        prisma_violations: list[str] = []
        if prisma is not None:
            prisma_violations = prisma.invariant_violations()
            expected = {
                "total_identified": len(record_rows),
                "duplicates_removed": len(record_rows) - len(unique_ids),
                "records_screened": len(ta.final),
                "records_excluded": len(ta.ids_with(ScreeningVerdict.EXCLUDE)),
                "records_awaiting_screening": len(ta.missing) + len(ta.unresolved),
                "reports_not_retrieved": len(unobtainable),
                "reports_assessed": len(retrieved),
                "reports_included": len(ft_included),
            }
            for name, value in expected.items():
                actual = getattr(prisma, name)
                if actual != value:
                    prisma_violations.append(
                        f"{name} is {actual} but persisted records imply {value}"
                    )
            ft_excluded = len(ft.ids_with(ScreeningVerdict.EXCLUDE))
            if sum(prisma.reports_excluded_by_reason.values()) != ft_excluded:
                prisma_violations.append(
                    f"reports excluded with reasons sum to "
                    f"{sum(prisma.reports_excluded_by_reason.values())} but {ft_excluded} "
                    "full-text exclusions are recorded"
                )
            if prisma.marked_ineligible_by_automation or prisma.removed_other_reasons:
                prisma_violations.append(
                    "removals other than duplicates cannot yet be verified against persisted "
                    "records"
                )

        from src.search.amendments import amendment_problems

        audit = self.latest_audit_report(project_id)
        audit_problems: tuple[str, ...] = ()
        if project.phase in (Phase.STAGE6_MANUSCRIPT.value, Phase.STAGE6_AUDIT.value):
            from src.manuscript.service import completion_problems

            audit_problems = tuple(completion_problems(self, project_id))
        return GuardContext(
            protocol_blockers=protocol_blockers,
            search_problems=tuple(self.search_identification_problems(project_id)),
            records_identified=len(record_rows),
            records_unique=len(unique_ids),
            unresolved_duplicate_candidates=unresolved_duplicates,
            # mandatory dual title/abstract screening: an adjudication does not replace it
            ta_missing_decisions=len(ta.incomplete_passes),
            ta_unresolved_conflicts=len(ta.unresolved),
            ta_included=len(ta_included),
            unresolved_cohort_warnings=warnings,
            prisma_recorded=prisma is not None,
            prisma_violations=tuple(prisma_violations),
            prisma_awaiting_screening=prisma.records_awaiting_screening if prisma else 0,
            prisma_pending_reports=(
                prisma.reports_awaiting_retrieval + prisma.reports_awaiting_assessment
                if prisma
                else 0
            ),
            fulltext_items_missing=fulltext_missing,
            fulltext_items_blocking=fulltext_blocking,
            ft_missing_decisions=len(ft.missing),
            ft_unresolved_conflicts=len(ft.unresolved),
            ft_included=len(ft_included),
            ft_awaiting_human_confirmation=len(ft_unconfirmed),
            included_reports_without_extraction=len(without_extraction),
            included_reports_without_usable_pdf=len(without_usable_pdf),
            arms_for_non_included_reports=sum(1 for rid, _ in arm_rows if rid not in ft_included),
            extracted_values_without_evidence=no_evidence,
            extraction_arms=len(arm_rows),
            unapproved_extraction_arms=sum(1 for _, approved in arm_rows if not approved),
            extraction_items_pending=pending_items,
            included_studies_without_approved_rob=len(studies_without_rob),
            unresolved_double_counting=len(self.double_counting_conflicts(project_id)),
            stats_output_present=project.stats_output_sha256 is not None,
            stats_output_problems=tuple(self.stats_output_problems(project_id)),
            manuscript_present=project.manuscript_version is not None,
            audit_passed=None if audit is None else audit.passed,
            audit_problems=audit_problems,
            amendment_problems=tuple(amendment_problems(self, project_id)),
        )

    def _authenticated_approval(self, protocol: PICOProtocol) -> bool:
        if protocol.status != "approved" or protocol.approved_reviewer_id is None:
            return False
        row = self.session.get(ReviewerORM, protocol.approved_reviewer_id)
        return (
            row is not None and row.revoked_at is None and row.display_name == protocol.approved_by
        )

    @staticmethod
    def _automated_only(
        final: dict[str, ScreeningVerdict],
        decisions: list[ScreeningDecision],
        conflicts: list[ScreeningConflict],
    ) -> set[str]:
        """Full-text outcomes not confirmed by any human (adjudication or human reviewers).

        Conflict rows are ignored: only persisted, authenticated human decisions count.
        """
        del conflicts
        human: set[str] = set()
        by_record: dict[str, list[ScreeningDecision]] = {}
        for d in decisions:
            if d.stage == ScreeningStage.FULL_TEXT:
                by_record.setdefault(d.record_id, []).append(d)
        for rid, ds in by_record.items():
            roles = {d.reviewer for d in ds if d.is_human}
            if (
                ReviewerRole.ADJUDICATOR in roles
                or {
                    ReviewerRole.REVIEWER_1,
                    ReviewerRole.REVIEWER_2,
                }
                <= roles
            ):
                human.add(rid)
        return set(final) - human


def _first_differences(expected: object, actual: object, path: str) -> list[str]:
    """Paths where two JSON-like values differ (used to explain a provenance rejection)."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        out: list[str] = []
        for key in sorted(set(expected) | set(actual)):
            out += _first_differences(expected.get(key), actual.get(key), f"{path}.{key}")
        return out
    if isinstance(expected, list) and isinstance(actual, list) and len(expected) == len(actual):
        out = []
        for i, (x, y) in enumerate(zip(expected, actual, strict=True)):
            out += _first_differences(x, y, f"{path}[{i}]")
        return out
    return [] if expected == actual else [f"{path}: engine {expected!r} != artifact {actual!r}"]
