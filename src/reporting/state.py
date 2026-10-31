"""Current reporting state of a project, and its contribution to the evidence package.

:func:`build_reporting_state` reads the append-only reporting records and decides, for each,
whether it is *current* (its input hash equals the hash of the inputs it depends on now) or
*stale*. Only current, approved records produce manuscript facts; stale and draft records are
listed so the Auditor can block on them. :func:`add_reporting_outputs` adds the facts,
table/figure templates and expected outputs (coverage) to the evidence package.

Fingerprint components (see ``src.manuscript.evidence``):

* ``reporting_sha256`` — the current GRADE records, approved syntheses and the SWiM
  applicability decision (content: it changes what the manuscript says);
* ``reporting_decisions_sha256`` — checklist decisions (audit-only: they never change the
  rendered text, but an audit or sign-off made before a decision changed is no longer valid).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.core.enums import (
    ComparisonType,
    EffectMeasure,
    ReviewerRole,
    ReviewStatus,
    ScreeningStage,
    ScreeningVerdict,
)
from src.core.exceptions import SysRevError
from src.core.schemas.extraction import ExtractedStudyArm, ExtractedValue, RiskOfBiasAssessment
from src.core.schemas.pipeline import PRISMACounts
from src.core.schemas.protocol import PICOProtocol
from src.core.schemas.reporting import (
    LEVEL_LABEL,
    ChecklistDecision,
    GradeAssessment,
    SwimApplicability,
    SynthesisGroup,
)
from src.core.schemas.stats import AnalysisResult, StatsOutput, canonical_sha256
from src.reporting import grade as gr
from src.reporting import synthesis as syn
from src.reporting.absolute_effects import (
    AbsoluteEffectError,
    absolute_effect,
    baseline_text,
    difference_text,
    risk_text,
)
from src.reporting.prisma_flow import (
    ExclusionBreakdown,
    build_flow_spec,
)

if TYPE_CHECKING:
    from src.manuscript.evidence import ExpectedOutput, _Builder
    from src.state.repository import Repository

NOT_ASSESSED_TEXT = {
    "framework_not_applicable": "no certainty framework with a clear methodological basis "
    "applies to this question",
    "not_performed": "a certainty assessment was not performed",
    "insufficient_information": "the information needed for a certainty assessment was "
    "insufficient",
}


@dataclass
class GradeEntry:
    analysis: AnalysisResult
    question_type: str
    inputs_sha256: str
    record: GradeAssessment | None = None  # latest record (any status)
    stale: bool = False
    problems: list[str] = field(default_factory=list)
    support: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        """approved / not_assessed / draft / none (no record or withdrawn) / stale."""
        if self.record is None or self.record.status == "withdrawn":
            return "none"
        if self.stale:
            return "stale"
        return self.record.status

    @property
    def current(self) -> GradeAssessment | None:
        return self.record if self.status in ("approved", "not_assessed") else None


@dataclass
class GroupEntry:
    record: SynthesisGroup
    inputs_sha256: str
    stale: bool = False
    problems: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.record.status == "withdrawn":
            return "withdrawn"
        return "stale" if self.stale else self.record.status


@dataclass
class ReportingState:
    flow_spec: dict[str, Any]
    breakdown: ExclusionBreakdown | None
    grade: dict[str, GradeEntry]
    groups: dict[str, GroupEntry]
    swim: SwimApplicability | None
    swim_stale: bool
    synthesis_sha256: str
    checklist_decisions: list[ChecklistDecision]
    content_sha256: str = ""
    decisions_sha256: str = ""
    # GRADE records whose analysis is no longer in the statistics output (not withdrawn)
    orphans: list[str] = field(default_factory=list)

    @property
    def approved_groups(self) -> list[SynthesisGroup]:
        return [g.record for g in self.groups.values() if g.status == "approved"]

    @property
    def swim_required(self) -> bool:
        """SWiM reporting is triggered once any synthesis without meta-analysis is approved;
        whether it applies is then a recorded human judgement."""
        return bool(self.approved_groups)

    @property
    def swim_checklist_required(self) -> bool:
        return (
            self.swim_required
            and self.swim is not None
            and not self.swim_stale
            and self.swim.applicability != "not_applicable"
        )


# --------------------------------------------------------------------------- helpers


def exclusion_breakdown(repo: Repository, project_id: str) -> ExclusionBreakdown:
    """Title/abstract exclusions decided with a human involved (an adjudication, or at least
    one human of the two screeners) versus by automated screening passes alone."""
    from src.screening.agreement import screening_status

    unique = {r.record_id for r in repo.load_records(project_id) if r.duplicate_of is None}
    decisions = repo.load_screening_decisions(project_id)
    ta = screening_status(unique, decisions, [], ScreeningStage.TITLE_ABSTRACT)
    roles: dict[str, dict[ReviewerRole, bool]] = {}
    for d in decisions:
        if d.stage == ScreeningStage.TITLE_ABSTRACT:
            roles.setdefault(d.record_id, {})[d.reviewer] = d.is_human
    human = automation = 0
    for rid in ta.ids_with(ScreeningVerdict.EXCLUDE):
        if any(roles.get(rid, {}).values()):
            human += 1
        else:
            automation += 1
    return ExclusionBreakdown(human, automation)


def _value_texts(v: ExtractedValue | None) -> list[str]:
    return [] if v is None else [v.reported_text, v.evidence.verbatim_quote]


def arm_texts(arm: ExtractedStudyArm) -> list[str]:
    out: list[str] = []
    for name in type(arm).model_fields:
        value = getattr(arm, name)
        if isinstance(value, ExtractedValue):
            out += _value_texts(value)
    return out


def group_inputs(
    repo: Repository, project_id: str, group: SynthesisGroup, protocol: PICOProtocol
) -> tuple[str, dict[str, list[str]], list[str]]:
    """(inputs hash, evidence texts per record, problems) of a synthesis group."""
    problems: list[str] = []
    included = repo.included_record_ids(project_id, ScreeningStage.FULL_TEXT)
    arms = {a.arm_id: a for a in repo.load_extracted_arms(project_id)}
    outcome = next((o for o in protocol.outcomes if o.outcome_id == group.outcome_id), None)
    if outcome is None:
        problems.append(f"outcome {group.outcome_id} is not in the approved protocol")
    texts: dict[str, list[str]] = {}
    payload: dict[str, Any] = {"items": {}, "arms": {}}
    for f in group.findings:
        if f.record_id not in included:
            problems.append(f"report {f.record_id} is not included at full text")
        bucket = texts.setdefault(f.record_id, [])
        for iid in f.item_ids:
            try:
                item = repo.get_extraction_item(project_id, iid)
            except SysRevError:
                problems.append(f"extraction item {iid} does not exist")
                continue
            payload["items"][iid] = item.model_dump(mode="json")
            if item.record_id != f.record_id:
                problems.append(f"item {iid} belongs to report {item.record_id}, not {f.record_id}")
            if item.review_status != ReviewStatus.ACCEPTED:
                problems.append(
                    f"item {iid} is {item.review_status.value}: only items accepted by a human "
                    "reviewer can support a finding"
                )
            if item.outcome_id not in (None, group.outcome_id):
                problems.append(f"item {iid} concerns outcome {item.outcome_id}")
            bucket += [t for t in (item.reported_text, item.text_value) if t]
            if item.evidence is not None:
                bucket.append(item.evidence.verbatim_quote)
        for aid in f.arm_ids:
            arm = arms.get(aid)
            if arm is None:
                problems.append(f"arm {aid} does not exist")
                continue
            payload["arms"][aid] = arm.model_dump(mode="json")
            if arm.record_id != f.record_id:
                problems.append(f"arm {aid} belongs to report {arm.record_id}, not {f.record_id}")
            if not arm.approved_by_human:
                problems.append(f"arm {aid} is not approved by a human reviewer")
            if arm.outcome_id != group.outcome_id:
                problems.append(f"arm {aid} concerns outcome {arm.outcome_id}")
            bucket += arm_texts(arm)
    rids = sorted({f.record_id for f in group.findings})
    payload |= {
        "outcome": outcome.model_dump(mode="json") if outcome else None,
        "review_design": protocol.review_design.value,
        "included": {r: r in included for r in rids},
        "study_links": repo.study_ids(project_id, set(rids)),
    }
    return canonical_sha256(payload), texts, problems


def group_problems(
    repo: Repository, project_id: str, group: SynthesisGroup, protocol: PICOProtocol
) -> tuple[str, list[str]]:
    sha, texts, problems = group_inputs(repo, project_id, group, protocol)
    outcome = next((o for o in protocol.outcomes if o.outcome_id == group.outcome_id), None)
    comparative = outcome is not None and outcome.comparison == ComparisonType.COMPARATIVE
    problems += syn.finding_problems(group, texts, comparative)
    problems += syn.summary_problems(group)
    if not comparative and "comparator" in group.grouping_basis:
        problems.append("a single-group outcome has no comparator to group by")
    return sha, problems


def _latest(records: list[Any], key: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for r in records:  # chronological
        out[getattr(r, key)] = r
    return out


def grade_inputs(
    repo: Repository,
    project_id: str,
    a: AnalysisResult,
    protocol: PICOProtocol,
    approved_rob: list[RiskOfBiasAssessment],
) -> tuple[str, dict[str, RiskOfBiasAssessment]]:
    """(inputs hash, contributing studies' approved risk of bias) of analysis ``a``."""
    rids = gr.contributing_records(a)
    rob = gr.latest_approved_rob(approved_rob, rids)
    outcome = next((o for o in protocol.outcomes if o.outcome_id == a.outcome_id), None)
    study_of = repo.study_ids(project_id, set(rids))
    approved = [x for x in approved_rob if x.approved_by_human]
    return gr.inputs_sha256(a, outcome, protocol.review_design.value, approved, study_of), rob


def synthesis_entries(
    repo: Repository, project_id: str, protocol: PICOProtocol
) -> dict[str, GroupEntry]:
    groups: dict[str, GroupEntry] = {}
    for gid, rec in _latest(repo.load_synthesis_groups(project_id), "group_id").items():
        if rec.status == "withdrawn":
            groups[gid] = GroupEntry(rec, "", False)
            continue
        sha, problems = group_problems(repo, project_id, rec, protocol)
        groups[gid] = GroupEntry(rec, sha, rec.inputs_sha256 != sha, problems)
    return groups


def synthesis_sha256(groups: dict[str, GroupEntry]) -> str:
    """Hash of the current, approved syntheses (what a SWiM applicability decision covers)."""
    approved = sorted(
        (g.record for g in groups.values() if g.status == "approved"), key=lambda r: r.group_id
    )
    return canonical_sha256([g.model_dump(mode="json") for g in approved])


# --------------------------------------------------------------------------- state


def build_reporting_state(
    repo: Repository,
    project_id: str,
    *,
    protocol: PICOProtocol,
    stats: StatsOutput,
    prisma: PRISMACounts,
    prisma_problems: list[str],
    searched_sources: frozenset[str],
    approved_rob: list[RiskOfBiasAssessment],
) -> ReportingState:
    breakdown = exclusion_breakdown(repo, project_id)
    spec = build_flow_spec(
        prisma,
        searched_sources=searched_sources,
        problems=[p for p in prisma_problems if p.startswith("PRISMA")],
        breakdown=breakdown,
    )
    grade_records = _latest(repo.load_grade_assessments(project_id), "analysis_id")
    grade: dict[str, GradeEntry] = {}
    for a in stats.analyses:
        sha, rob = grade_inputs(repo, project_id, a, protocol, approved_rob)
        entry = GradeEntry(a, gr.question_type(a), sha, support=gr.decision_support(a, rob))
        rec = grade_records.get(a.analysis_id)
        if rec is not None:
            entry.record = rec
            entry.stale = rec.inputs_sha256 != sha and rec.status != "withdrawn"
            if rec.status == "approved":
                entry.problems = gr.applicability_problems(rec.approach, a, rob)
        grade[a.analysis_id] = entry
    groups = synthesis_entries(repo, project_id, protocol)
    synthesis_sha = synthesis_sha256(groups)
    swim_records = repo.load_swim_applicability(project_id)
    swim = swim_records[-1] if swim_records else None
    state = ReportingState(
        flow_spec=spec,
        breakdown=breakdown,
        grade=grade,
        groups=groups,
        swim=swim,
        swim_stale=swim is not None and swim.synthesis_sha256 != synthesis_sha,
        synthesis_sha256=synthesis_sha,
        checklist_decisions=repo.load_checklist_decisions(project_id),
        orphans=sorted(
            aid
            for aid, rec in grade_records.items()
            if aid not in grade and rec.status != "withdrawn"
        ),
    )
    state.content_sha256 = canonical_sha256(
        {
            "grade": {
                k: e.current.model_dump(mode="json") if e.current else e.status
                for k, e in sorted(grade.items())
            },
            "synthesis": synthesis_sha,
            "swim": None if swim is None or state.swim_stale else swim.model_dump(mode="json"),
        }
    )
    state.decisions_sha256 = canonical_sha256([d.decision_id for d in state.checklist_decisions])
    return state


# --------------------------------------------------------------------------- outputs


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def add_reporting_outputs(
    b: _Builder,
    state: ReportingState,
    *,
    protocol: PICOProtocol,
    stats: StatsOutput,
    expected: list[ExpectedOutput],
    templates: dict[str, str],
    included_ref_keys: dict[str, str],
    study_of: dict[str, str],
    performed: set[str],
) -> None:
    from src.manuscript.evidence import ExpectedOutput

    rsrc = "approved reporting records (GRADE / synthesis), human-authored"
    # ---- PRISMA flow diagram (legend; the figure file is generated from the same snapshot)
    templates["figure:prisma"] = (
        "PRISMA 2020 flow diagram of study identification, screening and inclusion. "
        "{{prisma.identified}} records were identified and {{prisma.duplicates_removed}} "
        "duplicates were removed; {{prisma.records_screened}} records were screened and "
        "{{prisma.reports_assessed}} reports were assessed for eligibility; "
        "{{prisma.studies_included}} studies, reported in {{prisma.reports_included}} reports, "
        "were included. Counts are those of the stored PRISMA snapshot. File: prisma_flow.svg "
        "(regenerable from prisma_flow.json)."
    )
    expected.append(
        ExpectedOutput("figure:prisma", "PRISMA 2020 flow diagram legend", ("figures",))
    )

    # ---- GRADE
    approved = {k: e for k, e in state.grade.items() if e.status == "approved"}
    declared = {k: e for k, e in state.grade.items() if e.status == "not_assessed"}
    if approved:
        performed.add("grade")
    by_approach: dict[str, list[str]] = {}
    for e in approved.values():
        assert e.record is not None
        assert e.record.approach is not None
        by_approach.setdefault(e.record.approach, []).append(e.analysis.outcome_name)
    from src.core.schemas.reporting import APPROACH_LABEL

    if approved:
        parts = [
            f"{APPROACH_LABEL[k]} for {_join(sorted(set(v)))}"
            for k, v in sorted(by_approach.items())
        ]
        text = (
            "The certainty of the evidence was rated by authenticated human reviewers with "
            + "; ".join(parts)
            + ". For each outcome the reviewers recorded the initial level, a judgement with a "
            "rationale for each of the domains risk of bias, inconsistency, indirectness, "
            "imprecision and publication bias, any reasons for rating up, and the final "
            "certainty (high, moderate, low or very low). Statistical results were used as "
            "decision support only, and no statistical test for publication bias was performed."
        )
        others = [e for k, e in state.grade.items() if k not in approved]
        if others:
            text += (
                " Certainty was not rated for "
                + _join(sorted({e.analysis.outcome_name for e in others}))
                + "."
            )
    else:
        text = (
            "The certainty of the evidence was not assessed with GRADE or another formal framework."
        )
    for e in declared.values():
        assert e.record is not None
        assert e.record.not_assessed_reason is not None
        text += (
            f" For {e.analysis.outcome_name}, certainty was not assessed: "
            f"{NOT_ASSESSED_TEXT[e.record.not_assessed_reason]}"
            f" ({(e.record.status_note or '').rstrip('.')})."
        )
    b.add(
        "grade.methods", text, "certainty-assessment methods (or that none was made)", rsrc, "grade"
    )
    expected.append(ExpectedOutput("grade.methods", "certainty assessment methods", ("methods",)))
    limits: list[str] = []
    for aid, e in state.grade.items():
        a = e.analysis
        common: dict[str, Any] = {"analysis_id": aid, "measure": a.effect_measure.value}
        if e.status == "approved":
            g = e.record
            assert g is not None
            assert g.final_certainty is not None
            src = f"GRADE assessment {g.assessment_id} approved by {g.created_by}"
            b.add(
                f"grade.{aid}.statement",
                gr.certainty_statement(g, a.outcome_name),
                f"certainty of evidence for {a.outcome_name}",
                src,
                "grade",
                **common,
            )
            b.add(
                f"grade.{aid}.certainty",
                LEVEL_LABEL[g.final_certainty],
                f"GRADE certainty for {a.outcome_name}",
                src,
                "grade",
                **common,
            )
            b.add(
                f"grade.{aid}.reasons",
                gr.reasons_text(g),
                f"reasons for the certainty rating of {a.outcome_name}",
                src,
                "grade",
                **common,
            )
            cell = LEVEL_LABEL[g.final_certainty] + (
                " (prognosis framework)" if g.approach == "overall_prognosis" else ""
            )
            expected.append(
                ExpectedOutput(
                    f"grade.{aid}.statement",
                    f"certainty of evidence for {a.outcome_name}",
                    ("results",),
                )
            )
            if g.rated_down:
                limits.append(
                    f"for {a.outcome_name}, "
                    + _join(
                        [
                            f"{d.domain.replace('_', ' ')} ({d.concern.replace('_', ' ')})"
                            for d in g.rated_down
                        ]
                    )
                )
        elif e.status == "not_assessed":
            assert e.record is not None
            assert e.record.not_assessed_reason is not None
            cell = "not assessed: " + NOT_ASSESSED_TEXT[e.record.not_assessed_reason]
        else:
            cell = "not assessed (no current certainty assessment recorded)"
        b.add(
            f"grade.{aid}.cell",
            cell,
            f"certainty column entry for {a.outcome_name}",
            rsrc,
            "grade",
            **common,
        )
    if limits:
        b.add(
            "grade.limitations",
            "The certainty of the evidence was rated down " + "; ".join(limits) + ".",
            "domains for which the certainty of evidence was rated down",
            rsrc,
            "grade",
        )

    # ---- Summary of Findings (comparative, approved) and single-group evidence summary
    sof_rows: list[str] = []
    for e in approved.values():
        a = e.analysis
        if e.question_type != "comparative_intervention_effect":
            continue
        sof_rows.append(_sof_row(b, a, e, stats))
    if sof_rows:
        templates["table:sof"] = "\n".join(
            [
                "| Outcome (measure and follow-up) | Relative effect | Risk with comparator "
                "(assumed) | Risk with intervention | Risk difference | Participants (studies) "
                "| Certainty of the evidence (GRADE) | Reasons |",
                "|---|---|---|---|---|---|---|---|",
                *sof_rows,
            ]
        )
        expected.append(ExpectedOutput("table:sof", "GRADE Summary of Findings table", ("tables",)))
    single = [a for a in stats.analyses if gr.question_type(a) == "single_group_estimate"]
    if single:
        rows = [
            "| Outcome | Measure and timepoint | Studies | Participants | Events | Estimate | "
            "Certainty of the evidence |",
            "|---|---|---|---|---|---|---|",
        ]
        for a in single:
            p = f"stats.{a.analysis_id}"
            if a.status == "pooled":
                est = f"{{{{{p}.pooled_estimate}}}} (pooled)"
            elif a.status == "single_study" and f"{p}.single_study_estimate" in b.facts:
                est = f"{{{{{p}.single_study_estimate}}}} (single study; not pooled)"
            else:
                est = "not estimated"
            part = f"{{{{{p}.participants}}}}" if f"{p}.participants" in b.facts else "not reported"
            ev = f"{{{{{p}.events}}}}" if f"{p}.events" in b.facts else "not reported"
            rows.append(
                f"| {{{{{p}.outcome}}}} | {{{{{p}.measure_label}}}} | {{{{{p}.k}}}} | {part} "
                f"| {ev} | {est} | {{{{grade.{a.analysis_id}.cell}}}} |"
            )
        templates["table:evidence_summary"] = "\n".join(rows)
        expected.append(
            ExpectedOutput(
                "table:evidence_summary", "evidence summary for single-group outcomes", ("tables",)
            )
        )

    # ---- structured synthesis without meta-analysis
    groups = state.approved_groups
    if groups:
        performed.add("synthesis_without_meta_analysis")
        outcome_name = {o.outcome_id: o.name for o in protocol.outcomes}
        b.add(
            "synthesis.methods",
            " ".join(
                syn.methods_text(g, outcome_name.get(g.outcome_id, g.outcome_id)) for g in groups
            ),
            "methods of synthesis without meta-analysis (SWiM items 1-7)",
            rsrc,
            "synthesis",
        )
        expected.append(
            ExpectedOutput(
                "synthesis.methods", "methods of synthesis without meta-analysis", ("methods",)
            )
        )
        rows = [
            "| Synthesis group | Study | Study-level finding (source-linked) | Direction of "
            "effect |",
            "|---|---|---|---|",
        ]
        for sg in groups:
            p = f"synthesis.{sg.group_id}"
            src = f"synthesis group {sg.group_id} approved by {sg.created_by}"
            b.add(f"{p}.title", sg.title, "synthesis group", src, "synthesis")
            b.add(
                f"{p}.summary",
                sg.summary.rstrip(".") + ".",
                f"approved narrative summary of {sg.title} (human-authored; numbers checked "
                "against the study-level findings)",
                src,
                "synthesis",
                record_ids=tuple(sg.record_ids),
            )
            b.add(
                f"{p}.contribution",
                _contribution(sg, stats, study_of, outcome_name),
                "which studies contributed, and their relation to any meta-analysis",
                src,
                "synthesis",
                record_ids=tuple(sg.record_ids),
            )
            b.add(
                f"{p}.limitations",
                sg.limitations.rstrip(".") + ".",
                f"limitations of the synthesis {sg.title} (SWiM item 9)",
                src,
                "synthesis",
            )
            if sg.synthesis_method == "vote_counting_direction_of_effect":
                b.add(
                    f"{p}.direction",
                    syn.direction_text(sg),
                    "tabulated directions of effect (no statistical test)",
                    src,
                    "synthesis",
                )
            expected.append(
                ExpectedOutput(f"{p}.summary", f"synthesis findings: {sg.title}", ("results",))
            )
            expected.append(
                ExpectedOutput(
                    f"{p}.limitations",
                    f"limitations of the synthesis: {sg.title}",
                    ("limitations", "discussion"),
                )
            )
            for f in sg.findings:
                fk = f"{p}.finding.{f.record_id}"
                b.add(
                    fk,
                    f.finding.rstrip(".") + ".",
                    "study-level finding (human-written, "
                    "numbers checked against the linked extraction evidence)",
                    src,
                    "synthesis",
                    record_ids=(f.record_id,),
                )
                cite = (
                    f"[@{included_ref_keys[f.record_id]}]"
                    if f.record_id in (included_ref_keys)
                    else ""
                )
                rows.append(
                    f"| {{{{{p}.title}}}} | {{{{study.{study_of.get(f.record_id, f.record_id)}"
                    f".label}}}} {cite} | {{{{{fk}}}}} | {f.direction.replace('_', ' ')} |"
                )
        templates["table:synthesis"] = "\n".join(rows)
        expected.append(
            ExpectedOutput(
                "table:synthesis",
                "table of study-level findings synthesised without meta-analysis",
                ("tables",),
            )
        )


def _contribution(
    g: SynthesisGroup,
    stats: StatsOutput,
    study_of: dict[str, str],
    outcome_name: dict[str, str],
) -> str:
    studies = sorted({study_of.get(r, r) for r in g.record_ids})
    pooled = set()
    for a in stats.analyses:
        if a.outcome_id == g.outcome_id and a.status in ("pooled", "single_study"):
            pooled |= {study_of.get(r, r) for s in a.studies for r in s.record_ids}
    both = [s for s in studies if s in pooled]
    name = outcome_name.get(g.outcome_id, g.outcome_id)
    noun = "study" if len(studies) == 1 else "studies"
    text = f"{len(studies)} {noun} contributed to this synthesis without meta-analysis"
    if both:
        text += (
            f"; {len(both)} of them also contributed to a statistical analysis of {name}, "
            "so the two syntheses are not independent"
        )
    else:
        text += f"; none of them contributed to a statistical analysis of {name}"
    return text + "."


def _sof_row(b: _Builder, a: AnalysisResult, e: GradeEntry, stats: StatsOutput) -> str:
    del stats
    g = e.record
    assert g is not None
    aid = a.analysis_id
    p = f"stats.{aid}"
    rel_key = f"{p}.pooled_estimate" if a.status == "pooled" else f"{p}.single_study_estimate"
    rel = f"{{{{{rel_key}}}}}" + ("" if a.status == "pooled" else " (single study)")
    level = float(a.settings.get("confidence_level", 0.95)) * 100
    comp_cells: list[str] = []
    int_cells: list[str] = []
    diff_cells: list[str] = []
    est: float | None
    lo: float | None
    hi: float | None
    r = a.primary if a.status == "pooled" else None
    if r is not None:
        est, lo, hi = r.estimate_natural, r.ci_lower_natural, r.ci_upper_natural
    elif a.studies and a.studies[0].estimate_natural is not None:
        s = a.studies[0]
        est, lo, hi = s.estimate_natural, s.ci_lower_natural, s.ci_upper_natural
    else:  # pragma: no cover - applicability requires an estimate
        est = lo = hi = None
    if a.effect_measure in (EffectMeasure.RR, EffectMeasure.OR):
        if not g.baseline_risks:
            comp_cells = int_cells = diff_cells = [
                "not calculated: no assumed comparator risk with a stated source was recorded"
            ]
        for i, br in enumerate(g.baseline_risks, start=1):
            k = f"sof.{aid}.{i}"
            src = f"GRADE assessment {g.assessment_id}: baseline '{br.label}' ({br.source})"
            b.add(f"{k}.label", br.label, "assumed comparator risk label", src, "sof")
            t, n = baseline_text(br.risk)
            b.add(f"{k}.comparator", t, f"assumed comparator risk ({br.label})", src, "sof", n)
            if est is None or lo is None or hi is None:
                continue
            try:
                eff = absolute_effect(a.effect_measure, br.risk, est, lo, hi)
            except AbsoluteEffectError as exc:
                b.add(f"{k}.intervention", f"not calculable ({exc})", "absolute effect", src, "sof")
                int_cells.append(f"{{{{{k}.intervention}}}}")
                continue
            tr, nr = risk_text(eff, level)
            td, nd = difference_text(eff, level)
            b.add(
                f"{k}.intervention",
                tr,
                f"risk with intervention ({br.label}); "
                f"{a.effect_measure.value} from {rel_key} applied to the assumed comparator "
                "risk",
                src + f"; {rel_key}",
                "sof",
                nr,
                role="absolute_risk",
                analysis_id=aid,
            )
            b.add(
                f"{k}.difference",
                td,
                f"risk difference ({br.label})",
                src + f"; {rel_key}",
                "sof",
                nd,
                role="risk_difference",
                analysis_id=aid,
            )
            comp_cells.append(f"{{{{{k}.label}}}}: {{{{{k}.comparator}}}}")
            int_cells.append(f"{{{{{k}.intervention}}}}")
            diff_cells.append(f"{{{{{k}.difference}}}}")
    elif a.effect_measure == EffectMeasure.MD:
        rel = "not applicable (continuous outcome)"
        comp_cells = ["not recorded (no sourced comparator mean)"]
        int_cells = ["see risk difference column"]
        diff_cells = [f"mean difference {{{{{rel_key}}}}}"]
    else:
        rel = f"{{{{{rel_key}}}}} (standardised mean difference)"
        comp_cells = int_cells = diff_cells = [
            "not re-expressed in natural units (no sourced standard deviation)"
        ]
    part = f"{{{{{p}.participants}}}}" if f"{p}.participants" in b.facts else "not reported"
    return (
        f"| {{{{{p}.outcome}}}} ({{{{{p}.measure_label}}}}) | {rel} | {'<br>'.join(comp_cells)} "
        f"| {'<br>'.join(int_cells)} | {'<br>'.join(diff_cells)} | {part} ({{{{{p}.k}}}} "
        f"studies) | {{{{grade.{aid}.certainty}}}} | {{{{grade.{aid}.reasons}}}} |"
    )
