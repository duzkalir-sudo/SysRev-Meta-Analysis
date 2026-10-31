"""Deterministic reporting checks shared by the Stage 6 Auditor and the reporting overview.

Each check returns a :class:`ReportingCheck` with a status from the Auditor's vocabulary.
Blocking statuses (missing, mismatch, unsupported) mean work is required; ``requires human
review`` needs an authenticated acknowledgement; ``verified`` is recorded for coverage.
Genuine inapplicability is not missing work: an outcome whose certainty was declared *not
assessed* by a reviewer, a review without any synthesis without meta-analysis, or a SWiM
judgement of *not applicable* produces no blocking finding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.core.enums import CheckStatus
from src.reporting import checklist as cl
from src.reporting.absolute_effects import AbsoluteEffectError, absolute_effect
from src.reporting.prisma_flow import flow_counts

if TYPE_CHECKING:
    from src.core.schemas.manuscript import ManuscriptVersion
    from src.core.schemas.pipeline import PRISMACounts
    from src.manuscript.evidence import EvidencePackage
    from src.reporting.state import GradeEntry

# Diagram box -> evidence fact rendered in the manuscript.
FLOW_FACTS = {
    "records_screened": "prisma.records_screened",
    "records_excluded": "prisma.records_excluded",
    "duplicates_removed": "prisma.duplicates_removed",
    "reports_sought": "prisma.reports_sought",
    "reports_not_retrieved": "prisma.reports_not_retrieved",
    "reports_assessed": "prisma.reports_assessed",
    "reports_excluded": "prisma.reports_excluded",
    "studies_included": "prisma.studies_included",
    "reports_included": "prisma.reports_included",
}


@dataclass(frozen=True, slots=True)
class ReportingCheck:
    status: CheckStatus
    location: str
    message: str
    action: str
    key: str | None = None
    value: str | None = None


def _checklist_checks(
    name: str, label: str, version: ManuscriptVersion, package: EvidencePackage
) -> list[ReportingCheck]:
    assert package.reporting is not None
    out: list[ReportingCheck] = []
    states = cl.item_states(
        name, version, package.fingerprint, package.reporting.checklist_decisions
    )
    for st in states:
        loc = f"{label} item {st.item.item_id}"
        key = f"{name}:{st.item.item_id}"
        if st.stale:
            out.append(
                ReportingCheck(
                    CheckStatus.MISSING,
                    loc,
                    f"the decision on {label} item {st.item.item_id} is stale ({st.stale_reason})",
                    "Re-confirm the item against the current manuscript and evidence.",
                    key,
                )
            )
        elif not st.decided:
            hint = (
                ("; candidate location(s): " + "; ".join(st.candidates))
                if st.candidates
                else ("; no candidate location was found")
            )
            out.append(
                ReportingCheck(
                    CheckStatus.MISSING,
                    loc,
                    f"{label} item {st.item.item_id} ({st.item.topic}) has no human decision{hint}",
                    "Record whether the item is reported (with its location), partially reported, "
                    "missing or not applicable (with a justification).",
                    key,
                )
            )
        elif st.status == "missing":
            out.append(
                ReportingCheck(
                    CheckStatus.MISSING,
                    loc,
                    f"a reviewer recorded {label} item {st.item.item_id} ({st.item.topic}) as not "
                    "reported",
                    "Report the item, or record it as not applicable with a reason.",
                    key,
                )
            )
        elif st.status == "partially_reported":
            assert st.decision is not None
            out.append(
                ReportingCheck(
                    CheckStatus.REQUIRES_HUMAN_REVIEW,
                    loc,
                    f"{label} item {st.item.item_id} is only partially reported "
                    f"({st.decision.justification[:200]})",
                    "Complete the reporting, or acknowledge the gap.",
                    key,
                )
            )
        else:
            assert st.decision is not None
            out.append(
                ReportingCheck(
                    CheckStatus.VERIFIED,
                    loc,
                    f"{st.decision.status.replace('_', ' ')}: decided by "
                    f"{st.decision.reviewer_name} on version {st.decision.manuscript_version}",
                    "None.",
                    key,
                )
            )
    return out


def checklist_complete(name: str, version: ManuscriptVersion, package: EvidencePackage) -> bool:
    assert package.reporting is not None
    return cl.complete(
        cl.item_states(name, version, package.fingerprint, package.reporting.checklist_decisions)
    )


def reporting_checks(version: ManuscriptVersion, package: EvidencePackage) -> list[ReportingCheck]:
    rs = package.reporting
    if rs is None:  # pragma: no cover - always built by build_evidence_package
        return [
            ReportingCheck(
                CheckStatus.NOT_CHECKED,
                "whole manuscript",
                "no reporting state",
                "Rebuild the evidence package.",
            )
        ]
    out: list[ReportingCheck] = []
    # ---- PRISMA flow diagram vs the persisted PRISMA facts
    counts = rs.flow_spec["counts"]
    for box, key in FLOW_FACTS.items():
        fact = package.facts.get(key)
        shown = fact.numbers[0].value if fact is not None and fact.numbers else None
        ok = shown == counts[box]
        out.append(
            ReportingCheck(
                CheckStatus.VERIFIED if ok else CheckStatus.MISMATCH,
                "figure: PRISMA flow diagram",
                f"diagram box {box} shows {counts[box]}; the manuscript fact {key} is {shown}",
                "None." if ok else "Regenerate the diagram and re-render from the same snapshot.",
                key,
            )
        )
    ident = package.facts.get("prisma.identified")
    total = counts["identified_databases"] + counts["identified_registers"]
    if ident is None or ident.numbers[0].value != total:
        out.append(
            ReportingCheck(
                CheckStatus.MISMATCH,
                "figure: PRISMA flow diagram",
                f"records identified from databases and registers ({total}) differ from the "
                "manuscript total",
                "Regenerate the diagram and re-render.",
                "prisma.identified",
            )
        )
    if flow_counts(package_prisma(package)) != counts:
        out.append(
            ReportingCheck(
                CheckStatus.MISMATCH,
                "figure: PRISMA flow diagram",
                "the diagram counts differ from the stored PRISMA snapshot",
                "Regenerate the diagram.",
                "prisma",
            )
        )
    for w in rs.flow_spec["warnings"]:
        unclassified = "source type is not recorded" in w
        out.append(
            ReportingCheck(
                CheckStatus.REQUIRES_HUMAN_REVIEW if unclassified else CheckStatus.MISMATCH,
                "figure: PRISMA flow diagram",
                f"flow diagram warning: {w}",
                "Confirm how these records were identified."
                if unclassified
                else "Resolve the PRISMA inconsistency; the diagram shows this warning visibly.",
                "prisma",
            )
        )
    # ---- PRISMA 2020 checklists
    out += _checklist_checks("prisma_2020_main", "PRISMA 2020", version, package)
    out += _checklist_checks("prisma_2020_abstract", "PRISMA 2020 for Abstracts", version, package)
    # ---- GRADE
    for aid, e in sorted(rs.grade.items()):
        loc = f"GRADE {aid} ({e.analysis.outcome_name})"
        if e.status == "stale":
            out.append(
                ReportingCheck(
                    CheckStatus.MISMATCH,
                    loc,
                    f"the {e.record.status if e.record else ''} GRADE record was made on inputs "
                    "that "
                    "have changed (analysis result, outcome, contributing studies' risk of bias or "
                    "study links)",
                    "Re-assess and approve again, or withdraw the record.",
                    f"grade.{aid}",
                )
            )
        elif e.status == "draft":
            out.append(
                ReportingCheck(
                    CheckStatus.MISSING,
                    loc,
                    "a GRADE assessment was drafted but not approved",
                    "Approve it (authenticated), record the outcome as not assessed, or withdraw "
                    "the draft.",
                    f"grade.{aid}",
                )
            )
        elif e.status == "approved":
            for prob in e.problems:
                out.append(
                    ReportingCheck(
                        CheckStatus.MISMATCH,
                        loc,
                        prob,
                        "Re-assess with an applicable approach.",
                        f"grade.{aid}",
                    )
                )
            out += _sof_checks(aid, e, package)
            if not e.problems:
                out.append(
                    ReportingCheck(
                        CheckStatus.VERIFIED,
                        loc,
                        f"approved by {e.record.created_by if e.record else ''}; certainty "
                        "consistent with the recorded domain judgements",
                        "None.",
                        f"grade.{aid}",
                    )
                )
    for aid in rs.orphans:
        out.append(
            ReportingCheck(
                CheckStatus.MISMATCH,
                f"GRADE {aid}",
                "a GRADE record refers to an analysis that is no longer in the statistics output",
                "Withdraw the record (authenticated).",
                f"grade.{aid}",
            )
        )
    # ---- synthesis without meta-analysis
    for gid, g in sorted(rs.groups.items()):
        loc = f"synthesis {gid}"
        if g.status == "stale":
            out.append(
                ReportingCheck(
                    CheckStatus.MISMATCH,
                    loc,
                    "the synthesis was approved on study-level evidence that has changed",
                    "Review the findings and approve again, or withdraw.",
                    f"synthesis.{gid}",
                )
            )
        elif g.status == "draft":
            out.append(
                ReportingCheck(
                    CheckStatus.MISSING,
                    loc,
                    "a synthesis group was drafted but not approved",
                    "Approve it (authenticated) or withdraw it.",
                    f"synthesis.{gid}",
                )
            )
        elif g.status == "approved":
            for prob in g.problems:
                out.append(
                    ReportingCheck(
                        CheckStatus.UNSUPPORTED,
                        loc,
                        prob,
                        "Correct the synthesis and approve it again.",
                        f"synthesis.{gid}",
                    )
                )
            for sentence in g.unverifiable:
                out.append(
                    ReportingCheck(
                        CheckStatus.REQUIRES_HUMAN_REVIEW,
                        loc,
                        "the narrative summary makes a directional statement that the recorded "
                        f"study-level directions can neither confirm nor contradict: {sentence!r}",
                        "Confirm that the statement follows from the tabulated findings, or "
                        "reword it using the recorded directions.",
                        f"synthesis.{gid}.summary",
                    )
                )
    # ---- SWiM
    if rs.swim_required:
        if rs.swim is None or rs.swim_stale:
            out.append(
                ReportingCheck(
                    CheckStatus.MISSING,
                    "SWiM",
                    "syntheses without meta-analysis are reported but there is no current decision "
                    "on whether the SWiM guideline applies"
                    + (" (the syntheses changed after the decision)" if rs.swim_stale else ""),
                    "Record whether SWiM is applicable, partially applicable or not applicable, "
                    "with a rationale.",
                    "swim",
                )
            )
        elif rs.swim_checklist_required:
            out += _checklist_checks("swim", "SWiM", version, package)
    return out


def package_prisma(package: EvidencePackage) -> PRISMACounts:
    assert package.prisma is not None
    return package.prisma


def _sof_checks(aid: str, e: GradeEntry, package: EvidencePackage) -> list[ReportingCheck]:
    """Recompute every absolute effect from the *current* statistics output and the human
    baseline and compare with the Summary of Findings facts."""
    out: list[ReportingCheck] = []
    g = e.record
    a = package.analysis(aid)
    if g is None or a is None or not g.baseline_risks:
        return out
    est: float | None
    lo: float | None
    hi: float | None
    if a.primary is not None:
        r = a.primary
        est, lo, hi = r.estimate_natural, r.ci_lower_natural, r.ci_upper_natural
    elif a.studies and a.studies[0].estimate_natural is not None:
        st = a.studies[0]
        est, lo, hi = st.estimate_natural, st.ci_lower_natural, st.ci_upper_natural
    else:
        return out
    for i, br in enumerate(g.baseline_risks, start=1):
        key = f"sof.{aid}.{i}.intervention"
        fact = package.facts.get(key)
        loc = f"Summary of Findings {aid} ({br.label})"
        if est is None or lo is None or hi is None:
            continue
        try:
            eff = absolute_effect(a.effect_measure, br.risk, est, lo, hi)
        except AbsoluteEffectError as exc:
            out.append(
                ReportingCheck(
                    CheckStatus.MISMATCH, loc, str(exc), "Correct the baseline risk.", key
                )
            )
            continue
        expect = [eff.risk * 1000, *sorted((eff.risk_lower * 1000, eff.risk_upper * 1000))]
        shown = [n.value for n in fact.numbers if n.role == "absolute_risk"] if fact else []
        ok = len(shown) == 3 and all(abs(x - y) < 1e-9 for x, y in zip(shown, expect, strict=True))
        out.append(
            ReportingCheck(
                CheckStatus.VERIFIED if ok else CheckStatus.MISMATCH,
                loc,
                "absolute effect recomputed from the current statistics output and the recorded "
                f"baseline risk ({br.risk:g}): {'consistent' if ok else 'differs from the table'}",
                "None." if ok else "Re-render the manuscript (Summary of Findings).",
                key,
            )
        )
    return out
