"""Session 7: human-supervised GRADE, Summary of Findings, synthesis without meta-analysis
and SWiM. All projects, studies and numbers are SYNTHETIC.

Expected absolute effects are recomputed in the tests from first principles (risk with the
intervention = assumed comparator risk × risk ratio), independently of the module under test.
"""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy.orm import Session

from src.core.enums import ActorType, CheckStatus, EffectMeasure, RoBTool
from src.core.exceptions import SysRevError
from src.core.schemas.reporting import GradeAssessment
from src.extraction import link_reports_to_study
from src.manuscript.evidence import build_evidence_package
from src.manuscript.service import ManuscriptService
from src.reporting.absolute_effects import AbsoluteEffectError, absolute_effect
from src.reporting.export import sof_docx, sof_markdown
from src.reporting.service import ReportingService
from src.reporting.state import synthesis_entries
from tests.conftest import approved_rob
from tests.manuscript_helpers import (
    REVIEWER,
    comparative_project,
    complete_manuscript,
    confirm_checklists,
    decide_swim,
    human,
    out_of_band,
    save_grade,
    save_group,
    single_arm_project,
)

RAT = "synthetic rationale for this judgement"


def domains(**concern: str) -> list[dict[str, str]]:
    names = ("risk_of_bias", "inconsistency", "indirectness", "imprecision", "publication_bias")
    return [
        {
            "domain": d,
            "concern": concern.get(d, "not_serious"),
            "rationale": "registers and grey literature were searched; no unpublished trials "
            "identified"
            if d == "publication_bias"
            else RAT,
        }
        for d in names
    ]


def approval(**kw: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "status": "approved",
        "approach": "interventions_randomised",
        "framework_rationale": "all contributing studies are randomised trials (RoB 2)",
        "domains": domains(imprecision="serious"),
        "final_certainty": "moderate",
        "summary": "rated down once for imprecision",
    }
    body.update(kw)
    return body


def _grade(**kw: Any) -> GradeAssessment:
    data: dict[str, Any] = {
        "project_id": "p",
        "analysis_id": "a",
        "outcome_id": "o",
        "question_type": "comparative_intervention_effect",
        "inputs_sha256": "0" * 64,
        "created_by": REVIEWER,
        "actor_type": ActorType.HUMAN,
        "reviewer_id": "r1",
        "initial_certainty": "high",
    } | approval()
    data.update(kw)
    return GradeAssessment.model_validate(data)


# --------------------------------------------------------------------------- schema rules


def test_certainty_follows_explicit_human_inputs() -> None:
    assert _grade().implied_certainty == "moderate"
    g = _grade(
        domains=domains(risk_of_bias="very_serious", imprecision="serious"),
        final_certainty="very_low",
    )
    assert g.implied_certainty == "very_low"
    floor = _grade(
        domains=domains(
            risk_of_bias="very_serious", imprecision="very_serious", inconsistency="serious"
        ),
        final_certainty="very_low",
    )
    assert floor.implied_certainty == "very_low"  # never below very low
    nrs = _grade(
        approach="interventions_nonrandomised_other",
        initial_certainty="low",
        domains=domains(),
        final_certainty="high",
        upgrades=[{"factor": "large_effect", "levels": 2, "rationale": RAT}],
    )
    assert nrs.implied_certainty == "high"


def test_starting_level_depends_on_the_approach() -> None:
    with pytest.raises(ValueError, match="starts at high"):
        _grade(initial_certainty="low", final_certainty="very_low")
    with pytest.raises(ValueError, match="starts at low"):
        _grade(approach="interventions_nonrandomised_other", initial_certainty="high")
    _grade(approach="interventions_nrsi_robins_i", initial_certainty="high")


def test_significance_never_raises_certainty() -> None:
    with pytest.raises(ValueError, match="rating up applies"):  # randomised: no upgrading
        _grade(
            domains=domains(),
            final_certainty="high",
            upgrades=[{"factor": "large_effect", "levels": 1, "rationale": RAT}],
        )
    with pytest.raises(ValueError, match="cannot exceed"):
        _grade(
            approach="interventions_nonrandomised_other",
            initial_certainty="low",
            domains=domains(),
            final_certainty="moderate",
        )
    with pytest.raises(ValueError, match="two levels"):
        _grade(
            approach="interventions_nonrandomised_other",
            initial_certainty="low",
            upgrades=[{"factor": "dose_response", "levels": 2, "rationale": RAT}],
        )


def test_missing_or_unjustified_judgements_are_refused() -> None:
    with pytest.raises(ValueError, match="all five"):
        _grade(domains=domains()[:4])
    with pytest.raises(ValueError, match="at least 10"):
        _grade(domains=[d | {"rationale": "ok"} for d in domains()])
    with pytest.raises(ValueError, match="explain the departure"):
        _grade(final_certainty="low")  # implied moderate
    _grade(final_certainty="low", departure_rationale="two borderline concerns, rated once more")
    with pytest.raises(ValueError, match="authenticated human"):
        _grade(actor_type=ActorType.AGENT, reviewer_id=None)
    forged = _grade().model_dump(mode="json") | {"implied_certainty": "high"}
    with pytest.raises(ValueError, match="contradicts"):
        GradeAssessment.model_validate(forged)


@pytest.mark.parametrize(
    "basis",
    [
        "Egger's test p = 0.40",
        "funnel plot symmetric",
        "test was non-significant",
        "not statistically significant asymmetry",
    ],
)
def test_publication_bias_absence_not_inferred_from_tests(basis: str) -> None:
    pub = [
        d | {"rationale": basis} if d["domain"] == "publication_bias" else d
        for d in domains(imprecision="serious")
    ]
    with pytest.raises(ValueError, match="absence of publication bias"):
        _grade(domains=pub)


def test_single_group_estimates_are_not_intervention_effects() -> None:
    with pytest.raises(ValueError, match="does not apply"):
        _grade(question_type="single_group_estimate")
    with pytest.raises(ValueError, match="explicit methodological rationale"):
        _grade(
            question_type="single_group_estimate",
            approach="overall_prognosis",
            initial_certainty="high",
            initial_rationale="reviewer's documented choice",
            framework_rationale="event-rate question (short)",
        )
    with pytest.raises(ValueError, match="does not assume a starting level"):
        _grade(
            question_type="single_group_estimate",
            approach="overall_prognosis",
            initial_certainty=None,
            framework_rationale="x" * 60,
        )


def test_absolute_effect_arithmetic() -> None:
    eff = absolute_effect(EffectMeasure.RR, 0.2, 0.6, 0.4, 0.9)
    assert (eff.risk, eff.risk_lower, eff.risk_upper) == pytest.approx((0.12, 0.08, 0.18))
    assert eff.difference == pytest.approx((-80.0, -120.0, -20.0))
    odds = absolute_effect(EffectMeasure.OR, 0.2, 2.0, 1.0, 4.0)
    assert odds.risk == pytest.approx(0.2 * 2 / (1 - 0.2 + 0.2 * 2))
    with pytest.raises(AbsoluteEffectError, match="impossible risk"):
        absolute_effect(EffectMeasure.RR, 0.6, 2.5, 2.0, 3.0)
    with pytest.raises(AbsoluteEffectError, match="only from risk ratios and odds ratios"):
        absolute_effect(EffectMeasure.SMD, 0.2, 0.5, 0.1, 0.9)


# --------------------------------------------------------------------------- comparative


def _comp(session: Session) -> tuple[Any, str, ReportingService]:
    repo, rid, _ = comparative_project(session)
    return repo, rid, ReportingService(repo, "mc1")


def test_sof_absolute_effects_require_and_use_a_sourced_baseline(session: Session) -> None:
    repo, rid, rs = _comp(session)
    base = {
        "label": "Trial control groups",
        "risk": 0.15,
        "source": "median control-group risk reported by the three synthetic trials",
    }
    save_grade(rs, "comp", approval(baseline_risks=[base]), rid)
    package = build_evidence_package(repo, "mc1")
    a = package.analysis("comp")
    assert a is not None and a.primary is not None
    rr, lo, hi = a.primary.estimate_natural, a.primary.ci_lower_natural, a.primary.ci_upper_natural
    fact = package.facts["sof.comp.1.intervention"]
    values = [n.value for n in fact.numbers if n.role == "absolute_risk"]
    assert values == pytest.approx([150 * rr, 150 * lo, 150 * hi])  # 0.15 × RR × 1000
    assert package.facts["sof.comp.1.comparator"].text == "150 per 1000"
    assert package.facts["grade.comp.certainty"].text == "moderate"
    assert "rated down for imprecision (serious)" in package.facts["grade.comp.statement"].text
    md = sof_markdown(package)
    assert "Summary of Findings" in md and "`sof.comp.1.intervention`" in md
    assert sof_docx(package)[:2] == b"PK"
    svc = ManuscriptService(repo, "mc1")
    v = complete_manuscript(svc, human(repo, rid))
    assert "{{table:sof}}" in (v.section("tables").source if v.section("tables") else "")
    r = svc.audit()
    assert r.passed, [f.message for f in r.findings if f.blocking]
    assert r.scopes_checked.get("reporting", 0) > 0


def test_without_baseline_no_absolute_effect_is_calculated(session: Session) -> None:
    repo, rid, rs = _comp(session)
    save_grade(rs, "comp", approval(), rid)
    package = build_evidence_package(repo, "mc1")
    assert not any(k.startswith("sof.comp.") for k in package.facts)
    assert "not calculated: no assumed comparator risk" in package.templates["table:sof"]
    with pytest.raises(SysRevError, match="not in the reference registry"):
        save_grade(
            rs,
            "comp",
            approval(
                baseline_risks=[
                    {
                        "label": "external",
                        "risk": 0.1,
                        "source": "synthetic external cohort estimate",
                        "ref_key": "no-such-reference",
                    }
                ]
            ),
            rid,
        )


def test_approach_must_match_the_evidence(session: Session) -> None:
    _, rid, rs = _comp(session)
    with pytest.raises(SysRevError, match="robins_i"):
        save_grade(rs, "comp", approval(approach="interventions_nrsi_robins_i"), rid)
    with pytest.raises((SysRevError, ValueError), match="does not apply"):
        save_grade(
            rs,
            "comp",
            approval(
                approach="overall_prognosis",
                initial_certainty="high",
                initial_rationale="x" * 30,
                framework_rationale="y" * 60,
            ),
            rid,
        )
    with pytest.raises(SysRevError):  # an unknown (automated) identity cannot record GRADE
        save_grade(rs, "comp", approval(), "agent")
    view = {g["analysis_id"]: g for g in rs.grade()}["comp"]
    assert view["applicable_approaches"] == ["interventions_randomised"]
    assert "no value here determines" in view["decision_support"]["disclaimer"]
    assert view["decision_support"]["domains"]["publication_bias"]["note"].startswith(
        "no test for small-study effects"
    )


def test_changed_risk_of_bias_makes_the_assessment_stale(session: Session) -> None:
    repo, rid, rs = _comp(session)
    save_grade(rs, "comp", approval(), rid)
    svc = ManuscriptService(repo, "mc1")
    complete_manuscript(svc, human(repo, rid))
    assert svc.audit().passed
    high = approved_rob("t1", rid, name=REVIEWER).model_copy(update={"overall": "high"})
    assert high.tool == RoBTool.ROB2
    repo.save_rob("mc1", high)
    package = build_evidence_package(repo, "mc1")
    assert package.reporting is not None and package.reporting.grade["comp"].status == "stale"
    assert "grade.comp.statement" not in package.facts  # a stale rating is never reported
    svc.rerender(human(repo, rid))
    confirm_checklists(repo, "mc1", rid)
    r = svc.audit()
    assert any(
        f.scope == "reporting" and f.status == CheckStatus.MISMATCH and "inputs that" in f.message
        for f in r.findings
    )
    assert not r.passed
    shown = next(r for r in rs.grade() if r["analysis_id"] == "comp")["based_on_assessment_id"]
    rs.withdraw_grade(
        "comp",
        "risk of bias re-assessed; GRADE to be redone",
        based_on_assessment_id=shown,
        reviewer_id=rid,
        reviewer_name=REVIEWER,
    )
    package = build_evidence_package(repo, "mc1")
    assert package.reporting is not None and package.reporting.grade["comp"].status == "none"


def test_draft_blocks_but_declared_not_assessed_does_not(session: Session) -> None:
    repo, rid, rs = _comp(session)
    save_grade(
        rs,
        "comp",
        {"status": "draft", "approach": "interventions_randomised", "domains": domains()},
        rid,
    )
    svc = ManuscriptService(repo, "mc1")
    complete_manuscript(svc, human(repo, rid))
    r = svc.audit()
    assert any("drafted but not approved" in f.message for f in r.findings if f.blocking)
    save_grade(
        rs,
        "comp",
        {
            "status": "not_assessed",
            "not_assessed_reason": "not_performed",
            "status_note": "certainty assessment outside the scope of this synthetic review",
        },
        rid,
    )
    package = build_evidence_package(repo, "mc1")
    assert "certainty was not assessed" in package.facts["grade.methods"].text
    assert "table:sof" not in package.templates
    svc.rerender(human(repo, rid))
    confirm_checklists(repo, "mc1", rid)
    r = svc.audit()
    assert r.passed, [f.message for f in r.findings if f.blocking]


def test_typed_certainty_claims_are_checked(session: Session) -> None:
    repo, rid, rs = _comp(session)
    svc = ManuscriptService(repo, "mc1")
    complete_manuscript(svc, human(repo, rid))
    latest = repo.latest_manuscript_version("mc1")
    assert latest is not None
    disc = latest.section("discussion")
    assert disc is not None
    claim = " For any complication, the evidence was of high certainty."
    svc.edit(human(repo, rid), {"discussion": disc.source + claim}, "claim")
    r = svc.audit()
    assert any(
        f.status == CheckStatus.UNSUPPORTED and "no current, approved GRADE" in f.message
        for f in r.findings
    )
    save_grade(rs, "comp", approval(), rid)
    svc.rerender(human(repo, rid))
    r = svc.audit()
    assert any(
        f.status == CheckStatus.MISMATCH and "approved certainty" in f.message for f in r.findings
    )


# --------------------------------------------------------------------------- single arm


def test_single_arm_evidence_summary_is_not_a_comparative_sof(session: Session) -> None:
    repo, rid, _ = single_arm_project(session)
    rs = ReportingService(repo, "ms1")
    with pytest.raises((SysRevError, ValueError), match="does not apply"):
        save_grade(rs, "nec", approval(), rid)
    save_grade(
        rs,
        "nec",
        approval(
            approach="overall_prognosis",
            initial_certainty="high",
            initial_rationale="reviewer-recorded starting level for an event-rate question",
            framework_rationale="the question is the incidence of necrosis after treatment in a "
            "broad category of patients, an event-rate (prognosis) question",
            domains=domains(imprecision="serious", risk_of_bias="serious"),
            final_certainty="low",
        ),
        rid,
    )
    package = build_evidence_package(repo, "ms1")
    assert "table:sof" not in package.templates
    summary = package.templates["table:evidence_summary"]
    assert "{{grade.nec.cell}}" in summary and "{{grade.km5.cell}}" in summary
    assert package.facts["grade.nec.cell"].text == "low (prognosis framework)"
    assert package.facts["grade.km5.cell"].text.startswith("not assessed")
    assert "not as an intervention effect" in package.facts["grade.nec.statement"].text
    md = sof_markdown(package)
    assert "not a comparative Summary of Findings table" in md


# --------------------------------------------------------------------------- synthesis


def _group(repo: Any, rid: str, **kw: Any) -> dict[str, Any]:
    arms = [a.arm_id for a in repo.load_extracted_arms("mc1") if a.record_id == "t1"]
    body: dict[str, Any] = {
        "group_id": "g1",
        "outcome_id": "complication",
        "status": "approved",
        "title": "Trials reporting complications by severity",
        "grouping_basis": ["outcome", "study_design"],
        "description": "trials that reported complications only by severity grade",
        "rationale": "severity-graded complication counts are not compatible with the "
        "pre-specified any-complication outcome",
        "reason_not_pooled": "incompatible_measures",
        "reason_note": "complications reported by severity grade only",
        "standardised_metric": "direction of effect",
        "synthesis_method": "structured_tabulation",
        "method_justification": "a single trial cannot be summarised quantitatively; its "
        "result is tabulated",
        "certainty_method": "not assessed with GRADE for this synthesis",
        "presentation": "table ordered by risk of bias",
        "limitations": "only one trial; the direction of effect is uncertain",
        "findings": [
            {
                "record_id": "t1",
                "arm_ids": arms,
                "finding": "Complications occurred in 12 of 100 participants receiving X and in 20 "
                "of 100 receiving placebo.",
                "direction": "favours_intervention",
                "direction_basis": "fewer complications in the X arm",
            }
        ],
        "summary": "The single trial in this group favoured X: 12 of 100 versus 20 of 100.",
    }
    body.update(kw)
    return body


def test_mixed_review_synthesis_is_source_linked_and_not_pooled(session: Session) -> None:
    repo, rid, rs = _comp(session)
    save_group(rs, _group(repo, rid), rid)
    package = build_evidence_package(repo, "mc1")
    contribution = package.facts["synthesis.g1.contribution"].text
    assert contribution.startswith("1 study contributed to this synthesis without meta-analysis")
    assert "also contributed to a statistical analysis" in contribution
    assert "{{synthesis.g1.finding.t1}}" in package.templates["table:synthesis"]
    assert package.reporting is not None and package.reporting.swim_required
    svc = ManuscriptService(repo, "mc1")
    complete_manuscript(svc, human(repo, rid))
    r = svc.audit()
    swim = [f for f in r.findings if f.blocking and "SWiM" in f.message]
    assert swim and "no current decision" in swim[0].message
    decide_swim(
        rs, "not_applicable", "synthetic: a single tabulated trial, no synthesis of effects", rid
    )
    r = svc.audit()
    assert not [f for f in r.findings if f.blocking and "SWiM" in f.location]
    decide_swim(
        rs, "applicable", "quantitative intervention effects synthesised without meta-analysis", rid
    )
    svc.rerender(human(repo, rid))  # the supplementary list now names the SWiM checklist
    confirm_checklists(repo, "mc1", rid)
    r = svc.audit()
    assert len([f for f in r.findings if f.blocking and f.location.startswith("SWiM item")]) == 10
    confirm_checklists(repo, "mc1", rid, ("swim",))
    r = svc.audit()
    assert r.passed, [f.message for f in r.findings if f.blocking]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"summary": "The pooled estimate favoured X: 12 of 100 versus 20 of 100."}, "pooled"),
        (
            {"summary": "One of the trials showed a statistically significant reduction."},
            "statistical significance",
        ),
        (
            {"summary": "The single trial favoured X, with 15 of 100 complications."},
            "does not appear in the approved study-level findings",
        ),
    ],
)
def test_unsupported_synthesis_claims_are_refused(
    session: Session, change: dict[str, str], message: str
) -> None:
    repo, rid, rs = _comp(session)
    with pytest.raises(SysRevError, match=message):
        save_group(rs, _group(repo, rid, **change), rid)
    draft = save_group(rs, _group(repo, rid, status="draft", **change), rid)
    assert draft.status == "draft"
    assert any(message.split()[0] in p for p in rs.syntheses()[0]["problems"])


def test_finding_numbers_must_come_from_linked_evidence(session: Session) -> None:
    repo, rid, rs = _comp(session)
    body = _group(repo, rid)
    body["findings"][0]["finding"] = "Complications occurred in 13 of 100 participants (X)."
    body["summary"] = "The single trial favoured X."
    with pytest.raises(SysRevError, match="does not appear in its linked evidence"):
        save_group(rs, body, rid)


def test_directions_need_a_comparison_and_consistent_claims(session: Session) -> None:
    repo, rid, _ = single_arm_project(session)
    arms = [
        a.arm_id
        for a in repo.load_extracted_arms("ms1")
        if a.record_id == "s2" and a.outcome_id == "necrosis"
    ]
    body = _group(
        repo,
        rid,
        outcome_id="necrosis",
        grouping_basis=["outcome"],
        summary="Necrosis was reported in 3 of 40 patients in one series.",
    )
    body["findings"] = [
        {
            "record_id": "s2",
            "arm_ids": arms,
            "finding": "Necrosis occurred in 3 of 40 patients.",
            "direction": "favours_intervention",
            "direction_basis": "fewer events than expected",
        }
    ]
    rs = ReportingService(repo, "ms1")
    with pytest.raises(SysRevError, match="needs a comparison"):
        save_group(rs, body, rid)
    body["findings"][0] |= {"direction": "not_applicable", "direction_basis": None}
    body["summary"] = "Necrosis was consistently lower than in other series."
    with pytest.raises(SysRevError, match="directional statement"):
        save_group(rs, body, rid)


def test_changed_study_links_invalidate_an_approved_synthesis(session: Session) -> None:
    repo, rid, rs = _comp(session)
    save_group(rs, _group(repo, rid), rid)
    with out_of_band(repo, "mc1"):
        link_reports_to_study(
            repo,
            "mc1",
            ["t1", "t2"],
            study_id="study-t1",
            primary_record_id="t1",
            reviewer=repo.require_active_reviewer(rid),
            note="synthetic",
        )
    protocol = repo.latest_protocol("mc1")
    assert protocol is not None
    assert synthesis_entries(repo, "mc1", protocol)["g1"].status == "stale"
    # the links are also Stage 5 inputs: the statistics must be re-run before any reporting
    with pytest.raises(SysRevError, match="not a valid basis"):
        build_evidence_package(repo, "mc1")


def test_vote_counting_by_significance_in_prose_is_unsupported(session: Session) -> None:
    repo, rid, _ = _comp(session)
    svc = ManuscriptService(repo, "mc1")
    complete_manuscript(svc, human(repo, rid))
    latest = repo.latest_manuscript_version("mc1")
    assert latest is not None
    disc = latest.section("discussion")
    assert disc is not None
    svc.edit(
        human(repo, rid),
        {
            "discussion": disc.source + " Two of the three trials showed "
            "a significant reduction in complications."
        },
        "vote",
    )
    r = svc.audit()
    assert any(
        "by statistical significance" in f.message
        for f in r.findings
        if f.status == CheckStatus.UNSUPPORTED
    )
