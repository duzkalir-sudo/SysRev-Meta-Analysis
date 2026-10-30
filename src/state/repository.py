"""Persistence gateway: converts validated Pydantic contracts to ORM rows and back.

Every load re-validates the stored payload through Pydantic, so corrupted or
hand-edited rows fail loudly instead of propagating into the analysis.
"""

from __future__ import annotations

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from src.core.enums import FullTextStatus, OverlapResolution, Phase
from src.core.exceptions import SysRevError
from src.core.schemas import (
    AuditReport,
    CohortOverlapWarning,
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
from src.core.schemas.base import utc_now
from src.state.db import (
    AuditReportORM,
    CohortOverlapWarningORM,
    ExtractedArmORM,
    ExtractedValueORM,
    FullTextItemORM,
    GateDecisionORM,
    ProjectORM,
    ProtocolORM,
    RiskOfBiasORM,
    ScreeningConflictORM,
    ScreeningDecisionORM,
    SourceEvidenceORM,
    StateTransitionORM,
    StudyRecordORM,
)
from src.state.machine import GuardContext, PipelineStateMachine


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

    def save_gate_decision(self, project_id: str, decision: GateDecision) -> str:
        self.session.add(
            GateDecisionORM(
                id=decision.decision_id,
                project_id=project_id,
                gate=decision.gate.value,
                verdict=decision.verdict.value,
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
        rows = self.session.scalars(
            select(ScreeningDecisionORM).where(ScreeningDecisionORM.project_id == project_id)
        ).all()
        return [ScreeningDecision.model_validate(r.payload) for r in rows]

    def save_conflict(self, project_id: str, conflict: ScreeningConflict) -> None:
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

    def load_conflicts(self, project_id: str) -> list[ScreeningConflict]:
        rows = self.session.scalars(
            select(ScreeningConflictORM).where(ScreeningConflictORM.project_id == project_id)
        ).all()
        return [ScreeningConflict.model_validate(r.payload) for r in rows]

    # ------------------------------------------------------------------ cohort overlap
    def save_overlap_warning(self, project_id: str, warning: CohortOverlapWarning) -> None:
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

    def load_overlap_warnings(self, project_id: str) -> list[CohortOverlapWarning]:
        rows = self.session.scalars(
            select(CohortOverlapWarningORM).where(CohortOverlapWarningORM.project_id == project_id)
        ).all()
        return [CohortOverlapWarning.model_validate(r.payload) for r in rows]

    # ------------------------------------------------------------------ full text
    def save_fulltext_item(self, project_id: str, item: FullTextItem) -> None:
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

    def approve_extracted_arm(self, arm_id: str, approved_by: str) -> ExtractedStudyArm:
        row = self.session.get(ExtractedArmORM, arm_id)
        if row is None:
            raise NotFoundError(f"arm {arm_id} not found")
        arm = self._arm_from_row(row)
        approved = arm.model_copy(
            update={"approved_by_human": True, "approved_by": approved_by, "approved_at": utc_now()}
        )
        approved = ExtractedStudyArm.model_validate(approved.model_dump())
        row.approved_by_human = True
        row.approved_by = approved.approved_by
        row.approved_at = approved.approved_at
        self.session.flush()
        return approved

    def load_extracted_arms(self, project_id: str) -> list[ExtractedStudyArm]:
        rows = self.session.scalars(
            select(ExtractedArmORM)
            .where(ExtractedArmORM.project_id == project_id)
            .order_by(ExtractedArmORM.id)
        ).all()
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

    # ------------------------------------------------------------------ audit
    def save_audit_report(self, report: AuditReport) -> None:
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
            .order_by(AuditReportORM.generated_at.desc())
            .limit(1)
        ).first()
        return None if row is None else AuditReport.model_validate(row.payload)

    # ------------------------------------------------------------------ guards
    def build_guard_context(self, project_id: str) -> GuardContext:
        """Compute the facts that state-machine guards evaluate, directly from the database."""
        project = self.get_project(project_id)
        protocol = self.latest_protocol(project_id)
        protocol_blockers = (
            ("protocol not drafted",) if protocol is None else tuple(protocol.approval_blockers())
        )

        def count(stmt: Select[int]) -> int:
            return int(self.session.scalar(stmt) or 0)

        records = count(
            select(func.count())
            .select_from(StudyRecordORM)
            .where(StudyRecordORM.project_id == project_id)
        )
        conflicts = count(
            select(func.count())
            .select_from(ScreeningConflictORM)
            .where(
                ScreeningConflictORM.project_id == project_id,
                ScreeningConflictORM.resolved_verdict.is_(None),
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
        fulltext_blocking = len(self.missing_fulltext_manifest(project_id))
        arms = count(
            select(func.count())
            .select_from(ExtractedArmORM)
            .where(ExtractedArmORM.project_id == project_id)
        )
        unapproved = count(
            select(func.count())
            .select_from(ExtractedArmORM)
            .where(
                ExtractedArmORM.project_id == project_id,
                ExtractedArmORM.approved_by_human.is_(False),
            )
        )
        # Defence in depth: the NOT NULL FK already forbids this; count anyway.
        no_evidence = count(
            select(func.count())
            .select_from(ExtractedValueORM)
            .join(ExtractedArmORM)
            .where(
                ExtractedArmORM.project_id == project_id,
                ExtractedValueORM.evidence_id.is_(None),
            )
        )
        prisma = self.load_prisma(project_id)
        audit = self.latest_audit_report(project_id)
        return GuardContext(
            protocol_blockers=protocol_blockers,
            records_identified=records,
            unresolved_screening_conflicts=conflicts,
            unresolved_cohort_warnings=warnings,
            prisma_violations=tuple(prisma.invariant_violations()) if prisma else (),
            prisma_pending_reports=(
                prisma.reports_awaiting_retrieval + prisma.reports_awaiting_assessment
                if prisma
                else 0
            ),
            fulltext_items_blocking=fulltext_blocking,
            extracted_values_without_evidence=no_evidence,
            extraction_arms=arms,
            unapproved_extraction_arms=unapproved,
            stats_output_present=project.stats_output_sha256 is not None,
            manuscript_present=project.manuscript_version is not None,
            audit_passed=None if audit is None else audit.passed,
        )
