"""Session 7 reporting workflow on top of the repository.

The repository calls the ``*_problems`` functions before storing any reporting record, so a
record is stored only if it matches what the system recomputes *now*: the latest manuscript
version and the current evidence for a checklist decision; the current analysis inputs and
an applicable GRADE approach for a certainty assessment; the current, accepted extraction
evidence for a synthesis; the current approved syntheses for a SWiM applicability decision.
Every human decision comes from an active enrolled reviewer (checked by the repository).

:class:`ReportingService` gives the API and the local UI one place to read the state, record
decisions and export the outputs.
"""

from __future__ import annotations

import csv
import io
from typing import Any

from src.core.enums import ActorType, EffectMeasure
from src.core.exceptions import SysRevError
from src.core.schemas.manuscript import ManuscriptVersion, fingerprint_sha256
from src.core.schemas.reporting import (
    ChecklistDecision,
    GradeAssessment,
    SwimApplicability,
    SynthesisFinding,
    SynthesisGroup,
)
from src.reporting import checklist as cl
from src.reporting import grade as gr
from src.reporting import prisma_flow as flow
from src.reporting.absolute_effects import AbsoluteEffectError, absolute_effect
from src.reporting.review import reporting_checks
from src.reporting.standards import CATALOGUES, STANDARD_VERSION, item
from src.reporting.state import grade_inputs, group_problems, synthesis_entries, synthesis_sha256
from src.state.repository import Repository


def _package(repo: Repository, project_id: str) -> Any:
    from src.manuscript.evidence import build_evidence_package

    return build_evidence_package(repo, project_id)


# --------------------------------------------------------------------------- verification


def checklist_decision_problems(
    repo: Repository, d: ChecklistDecision, package: Any = None
) -> list[str]:
    try:
        it = item(d.checklist, d.item_id)
    except KeyError as exc:
        return [str(exc)]
    version = repo.latest_manuscript_version(d.project_id)
    if version is None:
        return ["no manuscript version: checklist items are located in a manuscript"]
    problems: list[str] = []
    if (d.manuscript_version, d.manuscript_sha256) != (version.version, version.content_sha256):
        problems.append(
            f"the decision names version {d.manuscript_version}, but the current manuscript is "
            f"version {version.version}"
        )
    for sid in d.section_ids:
        sec = version.section(sid)
        if sec is None or not sec.source.strip():
            problems.append(f"section {sid} does not exist or is empty in the manuscript")
    if package is None:
        try:
            package = _package(repo, d.project_id)
        except SysRevError as exc:
            return [*problems, f"evidence package unavailable: {exc}"]
    if d.evidence_sha256 != fingerprint_sha256(package.fingerprint):
        problems.append("the decision does not name the current evidence state")
    for k in d.evidence_keys:
        if k not in package.facts and k not in package.templates:
            problems.append(f"unknown evidence key {k}")
    if d.checklist == "swim" and not package.reporting.swim_checklist_required:
        problems.append(
            "the SWiM checklist is not required: no approved synthesis without meta-analysis, "
            "no current applicability decision, or SWiM was judged not applicable"
        )
    basis = cl.decision_basis(it, version, list(d.section_ids), package.fingerprint, d.status)
    if basis != d.basis_sha256:
        problems.append("the basis hash does not match the located sections and evidence")
    return problems


def grade_problems(repo: Repository, g: GradeAssessment) -> list[str]:
    stats = repo.current_stats_output(g.project_id)
    if stats is None or repo.stats_output_problems(g.project_id):
        return ["no verified statistics output: GRADE rates the current analyses"]
    a = next((x for x in stats.analyses if x.analysis_id == g.analysis_id), None)
    previous = [
        r for r in repo.load_grade_assessments(g.project_id) if r.analysis_id == g.analysis_id
    ]
    if g.status == "withdrawn":
        if not previous or previous[-1].status == "withdrawn":
            return ["there is no GRADE record to withdraw for this analysis"]
        return []
    if a is None:
        return [f"analysis {g.analysis_id} is not in the current statistics output"]
    protocol = repo.latest_protocol(g.project_id)
    assert protocol is not None
    problems: list[str] = []
    if g.outcome_id != a.outcome_id:
        problems.append(f"analysis {a.analysis_id} concerns outcome {a.outcome_id}")
    if g.question_type != gr.question_type(a):
        problems.append(
            f"analysis {a.analysis_id} answers a {gr.question_type(a).replace('_', ' ')} question"
        )
    approved_rob = [x for x in repo.load_rob(g.project_id) if x.approved_by_human]
    sha, rob = grade_inputs(repo, g.project_id, a, protocol, approved_rob)
    if g.inputs_sha256 != sha:
        problems.append("the record does not describe the current analysis inputs")
    if g.status == "approved":
        problems += gr.applicability_problems(g.approach, a, rob)
        if g.baseline_risks and a.effect_measure not in (EffectMeasure.RR, EffectMeasure.OR):
            problems.append(
                "baseline risks are used only to derive absolute effects from risk or odds "
                f"ratios, not from {a.effect_measure.value}"
            )
        est = a.primary or None
        for br in g.baseline_risks:
            if br.ref_key is not None:
                refs = {e.ref_key for e in repo.load_reference_entries(g.project_id)}
                refs |= {f"study-{r}"[:64] for r in gr.contributing_records(a)}
                if br.ref_key not in refs:
                    problems.append(
                        f"baseline source {br.ref_key} is not in the reference registry"
                    )
            if est is not None:
                try:
                    absolute_effect(
                        a.effect_measure,
                        br.risk,
                        est.estimate_natural,
                        est.ci_lower_natural,
                        est.ci_upper_natural,
                    )
                except AbsoluteEffectError as exc:
                    problems.append(str(exc))
    return problems


def synthesis_problems(repo: Repository, group: SynthesisGroup) -> list[str]:
    protocol = repo.latest_protocol(group.project_id)
    if protocol is None or protocol.status != "approved":
        return ["an approved protocol is required"]
    previous = [
        r for r in repo.load_synthesis_groups(group.project_id) if r.group_id == group.group_id
    ]
    if group.status == "withdrawn":
        if not previous or previous[-1].status == "withdrawn":
            return ["there is no synthesis group to withdraw"]
        return []
    sha, problems = group_problems(repo, group.project_id, group, protocol)
    out = (
        []
        if group.inputs_sha256 == sha
        else ["the record does not describe the current study-level evidence"]
    )
    if group.status == "approved":
        out += problems
    else:  # drafts may hold text problems, never references to missing or unusable data
        out += [p for p in problems if "does not exist" in p or "not included" in p]
    return out


def swim_applicability_problems(repo: Repository, d: SwimApplicability) -> list[str]:
    protocol = repo.latest_protocol(d.project_id)
    if protocol is None:
        return ["no protocol"]
    entries = synthesis_entries(repo, d.project_id, protocol)
    if not any(e.status == "approved" for e in entries.values()):
        return [
            "no synthesis without meta-analysis is approved, so SWiM reporting is not triggered"
        ]
    if d.synthesis_sha256 != synthesis_sha256(entries):
        return ["the decision does not describe the current approved syntheses"]
    return []


# --------------------------------------------------------------------------- service


class ReportingService:
    def __init__(self, repo: Repository, project_id: str) -> None:
        self.repo = repo
        self.project_id = project_id

    def package(self) -> Any:
        return _package(self.repo, self.project_id)

    def _version(self) -> ManuscriptVersion:
        v = self.repo.latest_manuscript_version(self.project_id)
        if v is None:
            raise SysRevError("no manuscript version: create a draft first")
        return v

    # ---- PRISMA flow diagram
    def flow(self, variant: str = flow.SUPPORTED_VARIANT) -> dict[str, Any]:
        """The diagram source (with the SVG hash). Raises UnsupportedVariantError for template
        variants this project cannot fill."""
        from src.manuscript.evidence import prisma_consistency_problems
        from src.reporting.state import exclusion_breakdown

        prisma = self.repo.load_prisma(self.project_id)
        if prisma is None:
            raise SysRevError("no PRISMA snapshot recorded")
        runs = self.repo.load_search_runs(self.project_id)
        from src.core.enums import SearchRunKind, SearchRunStatus

        searched = frozenset(
            r.database.value
            for r in runs
            if r.database
            and r.kind in (SearchRunKind.EXECUTION, SearchRunKind.IMPORT)
            and r.status != SearchRunStatus.FAILED
        )
        spec = flow.build_flow_spec(
            prisma,
            searched_sources=searched,
            problems=prisma_consistency_problems(self.repo, self.project_id, prisma),
            breakdown=exclusion_breakdown(self.repo, self.project_id),
            variant=variant,
        )
        svg = flow.svg_from_spec(spec)
        return {
            "spec": spec,
            "svg": svg,
            "svg_sha256": flow.sha256_text(svg),
            "spec_sha256": flow.spec_sha256(spec),
        }

    # ---- checklists
    def checklist(self, name: str) -> dict[str, Any]:
        if name not in CATALOGUES:
            raise SysRevError(f"unknown checklist {name}")
        package = self.package()
        version = self._version()
        states = cl.item_states(
            name, version, package.fingerprint, package.reporting.checklist_decisions
        )
        return {
            "checklist": name,
            "standard": STANDARD_VERSION[name],
            "manuscript_version": version.version,
            "manuscript_sha256": version.content_sha256,
            "evidence_sha256": fingerprint_sha256(package.fingerprint),
            "complete": cl.complete(states),
            "summary": cl.summary(states),
            "note": "A complete checklist record means each item has a current human decision; "
            "it is not a claim that the review is PRISMA-compliant.",
            "items": [_item_row(st) for st in states],
        }

    def decide_item(
        self,
        checklist: str,
        item_id: str,
        *,
        status: str,
        section_ids: list[str],
        location: str,
        justification: str,
        evidence_keys: list[str],
        reviewer_id: str,
        reviewer_name: str,
    ) -> ChecklistDecision:
        row = {
            "checklist": checklist,
            "item_id": item_id,
            "status": status,
            "section_ids": section_ids,
            "location": location,
            "justification": justification,
            "evidence_keys": evidence_keys,
        }
        return self.decide_items([row], reviewer_id=reviewer_id, reviewer_name=reviewer_name)[0]

    def decide_items(
        self, rows: list[dict[str, Any]], *, reviewer_id: str, reviewer_name: str
    ) -> list[ChecklistDecision]:
        """Several item decisions by one reviewer, made on the same manuscript version and
        evidence state (each still states its own status, location and justification)."""
        package = self.package()
        version = self._version()
        decisions = []
        for row in rows:
            it = item(row["checklist"], row["item_id"])
            sections = list(row.get("section_ids") or [])
            decisions.append(
                ChecklistDecision.model_validate(
                    {
                        "project_id": self.project_id,
                        "checklist": row["checklist"],
                        "item_id": row["item_id"],
                        "status": row["status"],
                        "section_ids": sections,
                        "location": row.get("location") or "",
                        "justification": row["justification"],
                        "evidence_keys": list(row.get("evidence_keys") or []),
                        "manuscript_version": version.version,
                        "manuscript_sha256": version.content_sha256,
                        "evidence_sha256": fingerprint_sha256(package.fingerprint),
                        "basis_sha256": cl.decision_basis(
                            it, version, sections, package.fingerprint, row["status"]
                        ),
                        "reviewer_id": reviewer_id,
                        "reviewer_name": reviewer_name,
                    }
                )
            )
        self.repo.save_checklist_decisions(decisions)
        return decisions

    # ---- GRADE and Summary of Findings
    def grade(self) -> list[dict[str, Any]]:
        package = self.package()
        out = []
        for aid, e in package.reporting.grade.items():
            out.append(
                {
                    "analysis_id": aid,
                    "outcome": e.analysis.outcome_name,
                    "question_type": e.question_type,
                    "status": e.status,
                    "stale": e.stale,
                    "inputs_sha256": e.inputs_sha256,
                    "problems": e.problems,
                    "record": e.record.model_dump(mode="json") if e.record else None,
                    "decision_support": e.support,
                    "applicable_approaches": [
                        k
                        for k in (
                            "interventions_randomised",
                            "interventions_nrsi_robins_i",
                            "interventions_nonrandomised_other",
                            "overall_prognosis",
                        )
                        if not gr.applicability_problems(
                            k, e.analysis, _rob_of(self.repo, self.project_id, e)
                        )
                    ],
                }
            )
        return out

    def save_grade(
        self, analysis_id: str, body: dict[str, Any], *, reviewer_id: str, reviewer_name: str
    ) -> GradeAssessment:
        """Record a draft, an approval or a 'not assessed' declaration (all authenticated)."""
        package = self.package()
        e = package.reporting.grade.get(analysis_id)
        if e is None:
            raise SysRevError(f"analysis {analysis_id} is not in the statistics output")
        data = {
            k: body[k]
            for k in (
                "status",
                "approach",
                "framework_rationale",
                "initial_certainty",
                "initial_rationale",
                "domains",
                "upgrades",
                "final_certainty",
                "departure_rationale",
                "summary",
                "baseline_risks",
                "notes",
                "not_assessed_reason",
                "status_note",
            )
            if k in body and body[k] is not None
        }
        if "initial_certainty" not in data and data.get("approach") in G_FIXED:
            data["initial_certainty"] = G_FIXED[data["approach"]]
        g = GradeAssessment.model_validate(
            data
            | {
                "project_id": self.project_id,
                "analysis_id": analysis_id,
                "outcome_id": e.analysis.outcome_id,
                "question_type": e.question_type,
                "inputs_sha256": e.inputs_sha256,
                "created_by": reviewer_name,
                "actor_type": ActorType.HUMAN,
                "reviewer_id": reviewer_id,
            }
        )
        self.repo.save_grade_assessment(g)
        return g

    def withdraw_grade(
        self, analysis_id: str, reason: str, *, reviewer_id: str, reviewer_name: str
    ) -> GradeAssessment:
        previous = [
            r
            for r in self.repo.load_grade_assessments(self.project_id)
            if r.analysis_id == analysis_id
        ]
        if not previous:
            raise SysRevError("there is no GRADE record to withdraw for this analysis")
        last = previous[-1]
        g = GradeAssessment(
            project_id=self.project_id,
            analysis_id=analysis_id,
            outcome_id=last.outcome_id,
            question_type=last.question_type,
            status="withdrawn",
            status_note=reason,
            inputs_sha256=last.inputs_sha256,
            created_by=reviewer_name,
            actor_type=ActorType.HUMAN,
            reviewer_id=reviewer_id,
        )
        self.repo.save_grade_assessment(g)
        return g

    def sof_markdown(self) -> str:
        from src.reporting.export import sof_markdown

        return sof_markdown(self.package())

    # ---- synthesis without meta-analysis and SWiM
    def syntheses(self) -> list[dict[str, Any]]:
        package = self.package()
        return [
            {
                "group_id": gid,
                "status": g.status,
                "stale": g.stale,
                "problems": g.problems,
                "inputs_sha256": g.inputs_sha256,
                "record": g.record.model_dump(mode="json"),
            }
            for gid, g in sorted(package.reporting.groups.items())
        ]

    def save_group(
        self, body: dict[str, Any], *, reviewer_id: str, reviewer_name: str
    ) -> SynthesisGroup:
        protocol = self.repo.latest_protocol(self.project_id)
        if protocol is None:
            raise SysRevError("no protocol")
        data = dict(body)
        data["findings"] = [SynthesisFinding.model_validate(f) for f in body["findings"]]
        draft = SynthesisGroup.model_validate(
            data
            | {
                "project_id": self.project_id,
                "inputs_sha256": "0" * 64,
                "created_by": reviewer_name,
                "actor_type": ActorType.HUMAN,
                "reviewer_id": reviewer_id,
            }
        )
        sha, _ = group_problems(self.repo, self.project_id, draft, protocol)
        group = draft.model_copy(update={"inputs_sha256": sha})
        group = SynthesisGroup.model_validate(group.model_dump())
        self.repo.save_synthesis_group(group)
        return group

    def withdraw_group(
        self, group_id: str, reason: str, *, reviewer_id: str, reviewer_name: str
    ) -> SynthesisGroup:
        previous = [
            g for g in self.repo.load_synthesis_groups(self.project_id) if g.group_id == group_id
        ]
        if not previous:
            raise SysRevError(f"no synthesis group {group_id}")
        g = previous[-1].model_copy(
            update={
                "status": "withdrawn",
                "status_note": reason,
                "created_by": reviewer_name,
                "reviewer_id": reviewer_id,
                "actor_type": ActorType.HUMAN,
            }
        )
        g = SynthesisGroup.model_validate(g.model_dump())
        self.repo.save_synthesis_group(g)
        return g

    def decide_swim(
        self, applicability: str, rationale: str, *, reviewer_id: str, reviewer_name: str
    ) -> SwimApplicability:
        package = self.package()
        d = SwimApplicability.model_validate(
            {
                "project_id": self.project_id,
                "applicability": applicability,
                "rationale": rationale,
                "synthesis_sha256": package.reporting.synthesis_sha256,
                "reviewer_id": reviewer_id,
                "reviewer_name": reviewer_name,
            }
        )
        self.repo.save_swim_applicability(d)
        return d

    def swim(self) -> dict[str, Any]:
        package = self.package()
        rs = package.reporting
        suggestion = (
            "applicable: SWiM addresses quantitative intervention-effect data synthesised "
            "without meta-analysis"
            if package.review_design.value == "comparative"
            else "not directly applicable or partially applicable: SWiM was developed for "
            "intervention effects; descriptive single-group, prevalence or prognostic syntheses "
            "differ (a reviewer decides)"
        )
        return {
            "required": rs.swim_required,
            "decision": rs.swim.model_dump(mode="json") if rs.swim else None,
            "stale": rs.swim_stale,
            "checklist_required": rs.swim_checklist_required,
            "suggestion": suggestion,
            "synthesis_sha256": rs.synthesis_sha256,
        }

    # ---- overview
    def overview(self) -> dict[str, Any]:
        package = self.package()
        version = self.repo.latest_manuscript_version(self.project_id)
        checks = reporting_checks(version, package) if version else []
        blocking = [
            c
            for c in checks
            if c.status.value in ("missing", "mismatch", "unsupported", "not_found", "not_checked")
        ]
        review = [c for c in checks if c.status.value == "requires_human_review"]
        return {
            "flow": {
                "status": package.reporting.flow_spec["status"],
                "warnings": package.reporting.flow_spec["warnings"],
                "counts": package.reporting.flow_spec["counts"],
            },
            "manuscript_version": version.version if version else None,
            "blockers": [_check_row(c) for c in blocking],
            "requires_human_review": [_check_row(c) for c in review],
            "grade": {k: e.status for k, e in package.reporting.grade.items()},
            "syntheses": {k: g.status for k, g in package.reporting.groups.items()},
            "swim_required": package.reporting.swim_required,
            "note": "These checks are part of the Stage 6 audit; nothing here certifies "
            "compliance with a reporting guideline.",
        }

    # ---- exports
    def checklist_csv(self, name: str) -> str:
        data = self.checklist(name)
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(
            [
                "checklist",
                "standard",
                "section",
                "topic",
                "item",
                "requirement",
                "status",
                "location",
                "sections",
                "justification",
                "reviewer",
                "decided_at",
                "manuscript_version",
                "manuscript_sha256",
                "evidence_sha256",
                "stale",
                "candidate_locations",
            ]
        )
        for row in data["items"]:
            d = row["decision"] or {}
            w.writerow(
                [
                    name,
                    data["standard"],
                    row["section"],
                    row["topic"],
                    row["item_id"],
                    row["requirement"],
                    row["status"],
                    d.get("location", ""),
                    ";".join(d.get("section_ids", [])),
                    d.get("justification", ""),
                    d.get("reviewer_name", ""),
                    d.get("decided_at", ""),
                    d.get("manuscript_version", ""),
                    d.get("manuscript_sha256", ""),
                    d.get("evidence_sha256", ""),
                    row["stale"],
                    " | ".join(row["candidates"]),
                ]
            )
        return buf.getvalue()


G_FIXED = {
    "interventions_randomised": "high",
    "interventions_nrsi_robins_i": "high",
    "interventions_nonrandomised_other": "low",
}


def _rob_of(repo: Repository, project_id: str, e: Any) -> dict[str, Any]:
    approved = [x for x in repo.load_rob(project_id) if x.approved_by_human]
    return gr.latest_approved_rob(approved, gr.contributing_records(e.analysis))


def _item_row(st: cl.ItemState) -> dict[str, Any]:
    return {
        "item_id": st.item.item_id,
        "section": st.item.section,
        "topic": st.item.topic,
        "requirement": st.item.text,
        "status": st.status,
        "decided": st.decided,
        "suggested_status": st.suggested_status,
        "candidates": st.candidates,
        "stale": st.stale,
        "stale_reason": st.stale_reason,
        "decision": st.decision.model_dump(mode="json") if st.decision else None,
    }


def _check_row(c: Any) -> dict[str, Any]:
    return {
        "status": c.status.value,
        "location": c.location,
        "message": c.message,
        "action": c.action,
        "key": c.key,
    }
