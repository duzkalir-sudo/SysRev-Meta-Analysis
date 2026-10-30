"""Stage 6 workflow on top of the repository: versions, audits, acknowledgements, sign-off.

Rules enforced here (and re-checked by the repository and the state-machine guard):

* a version records the evidence fingerprint it was rendered from; any later change to the
  statistics output, analysis plan, extraction inputs, protocol, PRISMA snapshot, search log,
  risk-of-bias assessments or reference registry makes its audit stale;
* an audit is recorded only if it equals the Auditor's own recomputation for the stored
  version and current evidence; an audit of an older version never authorises a newer one;
* ``REQUIRES_HUMAN_REVIEW`` findings are acknowledged by authenticated humans, per exact
  manuscript content and evidence state; blocking findings cannot be acknowledged;
* completion needs a human sign-off of the exact version and audit. No model output, and no
  automated actor, can approve a manuscript.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Any, cast

from src.auditor.auditor import audit_manuscript
from src.core.enums import ActorType, CheckStatus, Phase
from src.core.exceptions import SysRevError
from src.core.schemas.audit import AuditReport
from src.core.schemas.manuscript import (
    FindingAcknowledgement,
    GenerationProvenance,
    JournalSettings,
    ManuscriptApproval,
    ManuscriptSection,
    ManuscriptVersion,
    ReferenceVerification,
    StyleSettings,
    fingerprint_sha256,
)
from src.manuscript.document import render
from src.manuscript.drafting import HEADINGS, template_sections
from src.manuscript.evidence import EvidencePackage, build_evidence_package
from src.state.repository import Repository


def verify_audit_report(repo: Repository, report: AuditReport) -> list[str]:
    """Problems that prevent storing ``report``: it must be the Auditor's own result for a
    stored version and the current evidence (recomputed here)."""
    if report.manuscript_sha256 is None or not report.complete:
        return [
            "the report is incomplete (no manuscript hash, evidence fingerprint, coverage "
            "or required scopes); only the Stage 6 Auditor's complete reports are recorded"
        ]
    try:
        version = repo.get_manuscript_version(report.project_id, report.manuscript_version)
    except SysRevError as exc:
        return [str(exc)]
    if version.content_sha256 != report.manuscript_sha256:
        return ["the report does not describe the stored manuscript version"]
    try:
        package = build_evidence_package(repo, report.project_id)
    except SysRevError as exc:
        return [f"evidence package unavailable: {exc}"]
    expected = audit_manuscript(version, package)
    if expected.results_sha256() != report.results_sha256():
        return [
            "the report differs from the Auditor's recomputation for this manuscript and "
            "the current evidence (tampered, selective or stale)"
        ]
    return []


def _pending_reviews(repo: Repository, report: AuditReport) -> list[str]:
    ev = fingerprint_sha256(report.evidence_fingerprint)
    acked = {
        a.finding_id
        for a in repo.load_acknowledgements(report.project_id)
        if a.manuscript_sha256 == report.manuscript_sha256
        and a.evidence_sha256 == ev
        and _active(repo, a.reviewer_id)
    }
    return [
        f.finding_id
        for f in report.findings
        if f.status == CheckStatus.REQUIRES_HUMAN_REVIEW and f.finding_id not in acked
    ]


def _active(repo: Repository, reviewer_id: str) -> bool:
    try:
        repo.require_active_reviewer(reviewer_id)
    except SysRevError:
        return False
    return True


def audit_currency_problems(
    repo: Repository, project_id: str
) -> tuple[list[str], AuditReport | None]:
    """Is the latest audit about the current manuscript and the current evidence?"""
    version = repo.latest_manuscript_version(project_id)
    if version is None:
        return ["no manuscript version"], None
    report = repo.latest_audit_report(project_id)
    if report is None:
        return ["the current manuscript version has not been audited"], None
    problems: list[str] = []
    if report.manuscript_version != version.version or report.manuscript_sha256 != (
        version.content_sha256
    ):
        problems.append(
            f"the latest audit examined version {report.manuscript_version}, not "
            f"the current version {version.version}"
        )
    try:
        package = build_evidence_package(repo, project_id)
    except SysRevError as exc:
        return [*problems, f"evidence package unavailable: {exc}"], report
    if package.fingerprint != report.evidence_fingerprint:
        changed = sorted(
            k
            for k in package.fingerprint
            if package.fingerprint[k] != report.evidence_fingerprint.get(k)
        )
        problems.append(
            "the evidence changed after the audit (" + ", ".join(changed) + "); "
            "re-render and re-audit"
        )
    return problems, report


def completion_problems(repo: Repository, project_id: str) -> list[str]:
    problems, report = audit_currency_problems(repo, project_id)
    if report is None:
        return problems
    if not report.complete:
        problems.append("the latest audit is incomplete")
    if not report.passed:
        problems.append(f"the latest audit has {report.failures} blocking finding(s)")
    pending = _pending_reviews(repo, report)
    if pending:
        problems.append(f"{len(pending)} finding(s) requiring human review are not acknowledged")
    version = repo.latest_manuscript_version(project_id)
    signed = [
        a
        for a in repo.load_manuscript_approvals(project_id)
        if version is not None
        and a.manuscript_sha256 == version.content_sha256
        and a.audit_report_id == report.report_id
        and a.evidence_sha256 == fingerprint_sha256(report.evidence_fingerprint)
        and _active(repo, a.reviewer_id)
    ]
    if not signed:
        problems.append("no human sign-off of this manuscript version and audit")
    return problems


def approval_blockers(repo: Repository, project_id: str, approval: ManuscriptApproval) -> list[str]:
    project = repo.get_project(project_id)
    if project.phase != Phase.STAGE6_AUDIT.value:
        return [f"sign-off happens in the audit phase (phase={project.phase})"]
    problems, report = audit_currency_problems(repo, project_id)
    if report is None:
        return problems
    version = repo.latest_manuscript_version(project_id)
    assert version is not None
    if (
        (approval.manuscript_version, approval.manuscript_sha256)
        != (version.version, version.content_sha256)
        or approval.audit_report_id != report.report_id
        or approval.evidence_sha256 != (fingerprint_sha256(report.evidence_fingerprint))
    ):
        problems.append("the sign-off does not name the current version, audit and evidence")
    if not report.passed:
        problems.append(f"the audit has {report.failures} blocking finding(s)")
    pending = _pending_reviews(repo, report)
    if pending:
        problems.append(f"{len(pending)} finding(s) requiring human review are not acknowledged")
    return problems


@dataclass(frozen=True)
class Actor:
    """Who creates a version: an authenticated human, or the system/an agent."""

    name: str
    actor_type: ActorType
    reviewer_id: str | None = None


class ManuscriptService:
    def __init__(self, repo: Repository, project_id: str) -> None:
        self.repo = repo
        self.project_id = project_id

    def package(self) -> EvidencePackage:
        return build_evidence_package(self.repo, self.project_id)

    def _require_phase(self, *phases: Phase) -> None:
        phase = self.repo.get_project(self.project_id).phase
        if phase not in {p.value for p in phases}:
            raise SysRevError(f"not allowed in phase {phase}")

    def _new_version(
        self,
        sections: list[ManuscriptSection],
        actor: Actor,
        summary: str,
        provenance: GenerationProvenance,
        *,
        style: StyleSettings | None = None,
        journal: JournalSettings | None = None,
        package: EvidencePackage | None = None,
    ) -> ManuscriptVersion:
        self._require_phase(Phase.STAGE6_MANUSCRIPT, Phase.STAGE6_AUDIT)
        package = package or self.package()
        latest = self.repo.latest_manuscript_version(self.project_id)
        rendered = render(sections, package)
        version = ManuscriptVersion(
            project_id=self.project_id,
            version=1 if latest is None else latest.version + 1,
            parent_version=None if latest is None else latest.version,
            sections=sections,
            style=style or (latest.style if latest else StyleSettings()),
            journal=journal or (latest.journal if latest else JournalSettings()),
            bindings=rendered.bindings,
            rendered_markdown=rendered.markdown,
            evidence_fingerprint=dict(package.fingerprint),
            created_by=actor.name,
            actor_type=actor.actor_type,
            reviewer_id=actor.reviewer_id,
            change_summary=summary,
            provenance=[provenance],
        )
        self.repo.save_manuscript_version(version)
        return version

    def create_template_draft(
        self,
        actor: Actor,
        *,
        style: StyleSettings | None = None,
        journal: JournalSettings | None = None,
    ) -> ManuscriptVersion:
        package = self.package()
        return self._new_version(
            template_sections(package),
            actor,
            "deterministic template draft from the evidence package",
            GenerationProvenance(method="deterministic_template", sections=list(HEADINGS)),
            style=style,
            journal=journal,
            package=package,
        )

    def edit(
        self,
        actor: Actor,
        changes: dict[str, str],
        summary: str,
        *,
        style: StyleSettings | None = None,
        journal: JournalSettings | None = None,
    ) -> ManuscriptVersion:
        """A human edit of section sources (a new version; earlier ones are preserved)."""
        if actor.actor_type != ActorType.HUMAN:
            raise SysRevError("manual edits are made by an authenticated human")
        latest = self._latest()
        sections = []
        for sec in latest.sections:
            if sec.section_id in changes and changes[sec.section_id] != sec.source:
                sec = sec.model_copy(update={"source": changes[sec.section_id], "origin": "human"})
            sections.append(sec)
        unknown = set(changes) - {s.section_id for s in latest.sections}
        for sid in sorted(unknown):
            if sid not in HEADINGS:
                raise SysRevError(f"unknown section {sid}")
            sections.append(
                ManuscriptSection.model_validate(
                    {
                        "section_id": sid,
                        "heading": HEADINGS[sid],
                        "source": changes[sid],
                        "origin": "human",
                    }
                )
            )
        return self._new_version(
            sections,
            actor,
            summary,
            GenerationProvenance(method="human_edit", sections=sorted(changes)),
            style=style,
            journal=journal,
        )

    def rerender(
        self, actor: Actor, summary: str = "re-rendered against the current evidence"
    ) -> ManuscriptVersion:
        latest = self._latest()
        return self._new_version(
            list(latest.sections), actor, summary, GenerationProvenance(method="rerender")
        )

    def add_model_sections(
        self, sections: list[ManuscriptSection], actor: Actor, provenance: GenerationProvenance
    ) -> ManuscriptVersion:
        latest = self._latest()
        by_id = {s.section_id: s for s in sections}
        merged = [by_id.pop(s.section_id, s) for s in latest.sections] + list(by_id.values())
        return self._new_version(
            merged,
            actor,
            f"model-drafted section(s): {', '.join(sorted(s.section_id for s in sections))}",
            provenance,
        )

    def _latest(self) -> ManuscriptVersion:
        latest = self.repo.latest_manuscript_version(self.project_id)
        if latest is None:
            raise SysRevError("no manuscript version yet: create a draft first")
        return latest

    def audit(self, version: int | None = None) -> AuditReport:
        """Run the deterministic Auditor on a stored version and record the report."""
        self._require_phase(Phase.STAGE6_MANUSCRIPT, Phase.STAGE6_AUDIT)
        v = (
            self._latest()
            if version is None
            else self.repo.get_manuscript_version(self.project_id, version)
        )
        report = audit_manuscript(v, self.package())
        self.repo.save_audit_report(report)
        return report

    def acknowledge(
        self,
        report_id: str,
        finding_id: str,
        *,
        decision: str,
        note: str,
        reviewer_id: str,
        reviewer_name: str,
    ) -> FindingAcknowledgement:
        report = self.repo.get_audit_report(self.project_id, report_id)
        assert report.manuscript_sha256 is not None
        ack = FindingAcknowledgement(
            project_id=self.project_id,
            audit_report_id=report_id,
            finding_id=finding_id,
            manuscript_sha256=report.manuscript_sha256,
            evidence_sha256=fingerprint_sha256(report.evidence_fingerprint),
            decision=cast(Any, decision),
            note=note,
            reviewer_id=reviewer_id,
            reviewer_name=reviewer_name,
        )
        self.repo.save_acknowledgement(ack)
        return ack

    def attest_reference(
        self, ref_key: str, *, confirmed: bool, basis: str, reviewer_id: str, reviewer_name: str
    ) -> ReferenceVerification:
        """Human identity check of an included report that has no DOI/PMID (see
        ``Repository._check_attestation`` for the preconditions)."""
        state = self.package().references.get(ref_key)
        if state is None:
            raise SysRevError(f"reference {ref_key} is not in the registry")
        item = (
            self.repo.get_fulltext_item(self.project_id, state.entry.record_id)
            if state.entry.record_id
            else None
        )
        v = ReferenceVerification(
            ref_key=ref_key,
            metadata_sha256=state.entry.metadata_sha256(),
            source="human_attestation",
            status="verified" if confirmed else "mismatch",
            message=basis,
            checked_by=reviewer_name,
            reviewer_id=reviewer_id,
            pdf_sha256=item.pdf_sha256 if item is not None else None,
        )
        self.repo.save_reference_verification(self.project_id, v)
        return v

    def approve(
        self, *, reviewer_id: str, reviewer_name: str, statement: str
    ) -> ManuscriptApproval:
        version = self._latest()
        report = self.repo.latest_audit_report(self.project_id)
        if report is None:
            raise SysRevError("no audit to approve on")
        approval = ManuscriptApproval(
            project_id=self.project_id,
            manuscript_version=version.version,
            manuscript_sha256=version.content_sha256,
            audit_report_id=report.report_id,
            evidence_sha256=fingerprint_sha256(report.evidence_fingerprint),
            reviewer_id=reviewer_id,
            reviewer_name=reviewer_name,
            statement=statement,
        )
        self.repo.save_manuscript_approval(approval)
        return approval

    def status(self) -> dict[str, Any]:
        latest = self.repo.latest_manuscript_version(self.project_id)
        out: dict[str, Any] = {
            "phase": self.repo.get_project(self.project_id).phase,
            "latest_version": latest.version if latest else None,
            "manuscript_sha256": latest.content_sha256 if latest else None,
        }
        try:
            package = self.package()
            out["evidence_fingerprint"] = package.fingerprint
            out["evidence_current_for_latest_version"] = (
                latest is not None and latest.evidence_fingerprint == package.fingerprint
            )
            out["evidence_problems"] = []
        except SysRevError as exc:
            out["evidence_problems"] = [str(exc)]
        report = self.repo.latest_audit_report(self.project_id)
        if report is not None:
            out["latest_audit"] = {
                "report_id": report.report_id,
                "version": report.manuscript_version,
                "passed": report.passed,
                "complete": report.complete,
                "blocking": report.failures,
                "requires_human_review": report.requires_human_review,
                "pending_acknowledgements": len(_pending_reviews(self.repo, report)),
                "generated_at": report.generated_at.isoformat(),
            }
        out["completion_problems"] = (
            completion_problems(self.repo, self.project_id) if latest else ["no manuscript version"]
        )
        return out

    def diff(self, a: int, b: int) -> str:
        va = self.repo.get_manuscript_version(self.project_id, a)
        vb = self.repo.get_manuscript_version(self.project_id, b)
        return "".join(
            difflib.unified_diff(
                va.rendered_markdown.splitlines(keepends=True),
                vb.rendered_markdown.splitlines(keepends=True),
                fromfile=f"version {a}",
                tofile=f"version {b}",
            )
        )
