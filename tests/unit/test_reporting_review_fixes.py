"""Regression tests for the independent review of PR #12 (all projects and data SYNTHETIC).

1. Human decisions are bound to the state actually displayed (all four decision families),
   and the precondition check runs under the database write lock.
2. Reports of one study, and studies sharing participants, are one voting unit.
3. Significance-based classification cannot be presented as direction-based vote counting.
4. "Not applicable" checklist decisions go stale when the sections they rely on change.
5. ROBINS-I bodies of evidence can be rated down three levels for risk of bias.
6. Narrative summaries that contradict the recorded directions are refused.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from src.api_server.app import create_app
from src.core.enums import ActorType, CheckStatus, OverlapResolution, OverlapSeverity, RoBTool
from src.core.exceptions import SysRevError
from src.core.schemas.cohort import CohortOverlapWarning, OverlapSignal
from src.core.schemas.extraction import ROBINS_I_DOMAINS, DomainJudgement, RiskOfBiasAssessment
from src.core.schemas.reporting import GradeAssessment, SynthesisFinding
from src.extraction import link_reports_to_study
from src.manuscript.evidence import build_evidence_package
from src.manuscript.service import ManuscriptService
from src.reporting.service import ReportingService
from src.reporting.state import unit_problems
from src.state.db import init_db, make_engine, make_session_factory
from src.state.repository import Repository
from tests.conftest import NOW, approved_rob
from tests.manuscript_helpers import (
    REVIEWER,
    comparative_project,
    complete_manuscript,
    confirm_checklists,
    decide_item,
    decide_swim,
    human,
    out_of_band,
    save_grade,
    save_group,
    seen,
    single_arm_project,
)
from tests.unit.test_reporting_grade import _grade, _group, approval, domains

STALE = "the state you reviewed has changed"


def _arms(repo: Repository, rid: str) -> list[str]:
    return [a.arm_id for a in repo.load_extracted_arms("mc1") if a.record_id == rid]


def _second_finding(repo: Repository, record: str = "t2", **kw: Any) -> dict[str, Any]:
    events = {"t2": ("8", "80", "15", "82"), "t3": ("5", "60", "9", "58")}[record]
    return {
        "record_id": record,
        "arm_ids": _arms(repo, record),
        "finding": f"Complications occurred in {events[0]} of {events[1]} participants "
        f"receiving X and in {events[2]} of {events[3]} receiving placebo.",
        "direction": "favours_intervention",
        "direction_basis": "fewer complications observed in the X arm",
    } | kw


# =========================================================================== 1. stale screens


def test_checklist_decision_on_an_outdated_screen_is_refused(session: Session) -> None:
    repo, rid, _ = single_arm_project(session)
    svc = ManuscriptService(repo, "ms1")
    complete_manuscript(svc, human(repo, rid))
    rs = ReportingService(repo, "ms1")
    screen = rs.checklist("prisma_2020_main")  # GET: what the reviewer sees
    row = next(r for r in screen["items"] if r["item_id"] == "6")
    before = len(repo.load_checklist_decisions("ms1"))
    latest = repo.latest_manuscript_version("ms1")
    assert latest is not None and latest.section("methods") is not None
    methods = latest.section("methods").source  # type: ignore[union-attr]
    svc.edit(human(repo, rid), {"methods": methods + " Another edit."}, "another operation")
    with pytest.raises(SysRevError, match=STALE):
        rs.decide_item(
            "prisma_2020_main",
            "6",
            status="reported",
            section_ids=["methods"],
            location="Methods",
            justification="decided on the old screen",
            evidence_keys=[],
            based_on_decision_id=row["decision"]["decision_id"],
            seen=seen(screen),
            reviewer_id=rid,
            reviewer_name=REVIEWER,
        )
    assert len(repo.load_checklist_decisions("ms1")) == before


def test_checklist_decision_after_an_evidence_change_is_refused(session: Session) -> None:
    """Same manuscript version, but the evidence changed (a GRADE approval)."""
    repo, rid, _ = comparative_project(session)
    complete_manuscript(ManuscriptService(repo, "mc1"), human(repo, rid))
    rs = ReportingService(repo, "mc1")
    screen = rs.checklist("prisma_2020_main")
    row = next(r for r in screen["items"] if r["item_id"] == "15")
    save_grade(rs, "comp", approval(), rid)  # another operation changes the evidence
    with pytest.raises(SysRevError, match="the evidence changed"):
        rs.decide_item(
            "prisma_2020_main",
            "15",
            status="not_applicable",
            section_ids=[],
            location="",
            justification="no certainty assessment on the old screen",
            evidence_keys=[],
            based_on_decision_id=row["decision"]["decision_id"],
            seen=seen(screen),
            reviewer_id=rid,
            reviewer_name=REVIEWER,
        )


def test_concurrent_decision_on_the_same_item_is_refused(session: Session) -> None:
    repo, rid, _ = single_arm_project(session)
    complete_manuscript(ManuscriptService(repo, "ms1"), human(repo, rid))
    rs = ReportingService(repo, "ms1")
    screen = rs.checklist("prisma_2020_main")  # two reviewers open the same screen
    shown = next(r for r in screen["items"] if r["item_id"] == "25")["decision"]["decision_id"]
    common = {
        "status": "not_applicable",
        "section_ids": [],
        "location": "",
        "evidence_keys": [],
        "based_on_decision_id": shown,
        "seen": seen(screen),
        "reviewer_id": rid,
        "reviewer_name": REVIEWER,
    }
    rs.decide_item("prisma_2020_main", "25", justification="first decision on it", **common)
    with pytest.raises(SysRevError, match="decision on item 25 changed"):
        rs.decide_item("prisma_2020_main", "25", justification="second, unaware", **common)


def test_grade_decision_on_outdated_inputs_is_refused(session: Session) -> None:
    repo, rid, _ = comparative_project(session)
    rs = ReportingService(repo, "mc1")
    shown = next(r for r in rs.grade() if r["analysis_id"] == "comp")  # GET
    later = approved_rob("t1", rid, name=REVIEWER).model_copy(
        update={"overall": "high", "approved_at": NOW + timedelta(days=1)}
    )
    with out_of_band(repo, "mc1"):  # risk of bias re-assessed (outside the app since Session 9)
        repo.save_rob("mc1", later)
    body = approval() | {
        "inputs_sha256": shown["inputs_sha256"],
        "based_on_assessment_id": shown["based_on_assessment_id"],
    }
    with pytest.raises(SysRevError, match=STALE):
        rs.save_grade("comp", body, reviewer_id=rid, reviewer_name=REVIEWER)
    assert repo.load_grade_assessments("mc1") == []
    # a record saved by someone else after the screen was read is not overwritten silently
    current = next(r for r in rs.grade() if r["analysis_id"] == "comp")
    save_grade(rs, "comp", approval(final_certainty="moderate"), rid)
    stale_body = approval() | {
        "inputs_sha256": current["inputs_sha256"],
        "based_on_assessment_id": None,
    }
    with pytest.raises(SysRevError, match="GRADE record for comp changed"):
        rs.save_grade("comp", stale_body, reviewer_id=rid, reviewer_name=REVIEWER)
    assert len(repo.load_grade_assessments("mc1")) == 1


def test_synthesis_saved_on_outdated_evidence_is_refused(session: Session) -> None:
    repo, rid, _ = comparative_project(session)
    rs = ReportingService(repo, "mc1")
    body = _group(repo, rid)
    preview = rs.preview_group(body)  # what the reviewer saw
    with out_of_band(repo, "mc1"):
        link_reports_to_study(
            repo,
            "mc1",
            ["t1", "t2"],
            study_id="study-t1",
            primary_record_id="t1",
            reviewer=repo.require_active_reviewer(rid),
            note="synthetic: changed after the preview",
        )
    shown = {k: preview[k] for k in ("inputs_sha256", "based_on_revision_id")}
    with pytest.raises(SysRevError, match=STALE):
        rs.save_group(shown | body, reviewer_id=rid, reviewer_name=REVIEWER)
    assert repo.load_synthesis_groups("mc1") == []


def test_synthesis_revision_race_is_refused(session: Session) -> None:
    repo, rid, _ = comparative_project(session)
    rs = ReportingService(repo, "mc1")
    body = _group(repo, rid)
    preview = rs.preview_group(body)
    save_group(rs, body, rid)  # another reviewer saves the group first
    shown = {k: preview[k] for k in ("inputs_sha256", "based_on_revision_id")}
    with pytest.raises(SysRevError, match="synthesis group g1 changed"):
        rs.save_group(shown | body, reviewer_id=rid, reviewer_name=REVIEWER)
    assert len(repo.load_synthesis_groups("mc1")) == 1


def test_swim_decision_on_outdated_syntheses_is_refused(session: Session) -> None:
    repo, rid, _ = comparative_project(session)
    rs = ReportingService(repo, "mc1")
    save_group(rs, _group(repo, rid), rid)
    screen = rs.swim()  # GET
    second = _group(repo, rid, group_id="g2", title="A second group (synthetic)")
    second["findings"] = [_second_finding(repo, "t3")]
    second["summary"] = "The single trial in this group favoured the intervention."
    save_group(rs, second, rid)  # another operation changes the approved syntheses
    with pytest.raises(SysRevError, match=STALE):
        rs.decide_swim(
            "applicable",
            "decided on the old list of syntheses (synthetic)",
            synthesis_sha256=screen["synthesis_sha256"],
            based_on_decision_id=screen["based_on_decision_id"],
            reviewer_id=rid,
            reviewer_name=REVIEWER,
        )
    assert repo.load_swim_applicability("mc1") == []


@pytest.fixture
def http(tmp_path: Path) -> Iterator[tuple[TestClient, dict[str, str]]]:
    url = f"sqlite:///{tmp_path / 'api.db'}"
    engine = make_engine(url)
    init_db(engine)
    with make_session_factory(engine)() as s:
        repo, rid, token = single_arm_project(s)
        complete_manuscript(ManuscriptService(repo, "ms1"), human(repo, rid))
        s.commit()
    yield TestClient(create_app(url, tmp_path / "data")), {"Authorization": f"Bearer {token}"}
    engine.dispose()


def test_http_old_screen_is_rejected_and_nothing_is_stored(
    http: tuple[TestClient, dict[str, str]],
) -> None:
    client, auth = http
    base = "/projects/ms1/reporting/checklists/prisma_2020_main"
    screen = client.get(base).json()
    row = next(r for r in screen["items"] if r["item_id"] == "26")
    body = {
        "status": "missing",
        "justification": "competing interests not declared",
        "based_on_decision_id": row["decision"]["decision_id"],
        "seen": {
            k: screen[k] for k in ("manuscript_version", "manuscript_sha256", "evidence_sha256")
        },
    }
    v = client.get("/projects/ms1/manuscript/versions/2").json()
    disc = next(s for s in v["sections"] if s["section_id"] == "discussion")
    edit = client.post(
        "/projects/ms1/manuscript/versions",
        headers=auth,
        json={
            "changes": {"discussion": disc["source"] + " Edited."},
            "summary": "another operation",
        },
    )
    assert edit.status_code == 200
    r = client.post(f"{base}/items/26", json=body, headers=auth)
    assert r.status_code == 409 and STALE in r.text
    after = client.get(base).json()
    assert (
        next(x for x in after["items"] if x["item_id"] == "26")["decision"]["decision_id"]
        == (row["decision"]["decision_id"])
    )


def test_precondition_check_holds_the_write_lock(tmp_path: Path) -> None:
    """Between the check and the commit no other writer can change the database: the
    decision transaction takes the write lock before checking (SQLite: one writer)."""
    url = f"sqlite:///{tmp_path / 'lock.db'}"
    engine = make_engine(url)
    init_db(engine)
    with make_session_factory(engine)() as s:
        Repository(s).create_project("p1", "Synthetic")
        s.commit()
    other = create_engine(url, connect_args={"timeout": 0.2})
    with make_session_factory(engine)() as s:
        Repository(s)._lock_for_reporting("p1")  # the decision transaction begins
        with (
            make_session_factory(other)() as s2,
            pytest.raises(OperationalError, match="locked"),
        ):
            Repository(s2).create_project("p2", "concurrent writer")
        s.rollback()
    other.dispose()
    engine.dispose()


# =========================================================================== 2. voting units


def test_two_reports_of_one_study_are_one_vote(session: Session) -> None:
    repo, rid, _ = comparative_project(session, study_links=[["t1", "t2"]])
    rs = ReportingService(repo, "mc1")
    body = _group(
        repo,
        rid,
        synthesis_method="vote_counting_direction_of_effect",
        summary="The trial favoured the intervention.",
    )
    body["findings"].append(_second_finding(repo))
    with pytest.raises(SysRevError, match="exactly one must represent it"):
        save_group(rs, body, rid)
    body["findings"][1]["selected_for_unit"] = False
    with pytest.raises(ValueError, match="documented selection rule"):
        save_group(rs, body, rid)
    body["selection_rule"] = "the primary report (largest follow-up) represents the study"
    save_group(rs, body, rid)
    package = build_evidence_package(repo, "mc1")
    contribution = package.facts["synthesis.g1.contribution"].text
    assert contribution.startswith("1 study contributed")
    assert "2 reports; 1 independent study unit" in contribution
    direction = package.facts["synthesis.g1.direction"].text
    assert direction.startswith("Of 1 independent study units")
    assert "1 favoured the intervention" in direction and "2 favoured" not in direction


def test_participant_overlap_makes_studies_one_voting_unit(session: Session) -> None:
    repo, rid, _ = comparative_project(session)
    warning = CohortOverlapWarning(
        record_id_a="t1",
        record_id_b="t3",
        overlap_score=0.8,
        severity=OverlapSeverity.HIGH,
        signals=[OverlapSignal(kind="coauthor_overlap", score=0.8, detail="synthetic")],
    )
    repo.save_overlap_warning("mc1", warning)
    with out_of_band(repo, "mc1"):
        repo.resolve_overlap_warning(
            "mc1",
            warning.warning_id,
            resolution=OverlapResolution.OVERLAPPING_DIFFERENT_OUTCOMES,
            reviewer_id=rid,
            note="synthetic: shared participants",
        )
    rs = ReportingService(repo, "mc1")
    body = _group(repo, rid, summary="The trials favoured the intervention.")
    body["findings"].append(_second_finding(repo, "t3"))
    from src.core.schemas.reporting import SynthesisGroup

    group = SynthesisGroup.model_validate(
        body
        | {
            "project_id": "mc1",
            "inputs_sha256": "0" * 64,
            "created_by": REVIEWER,
            "actor_type": ActorType.HUMAN,
            "reviewer_id": rid,
            "findings": [SynthesisFinding.model_validate(f) for f in body["findings"]],
        }
    )
    assert any("exactly one must represent it" in p for p in unit_problems(repo, "mc1", group))
    with pytest.raises(SysRevError, match="exactly one must represent it"):
        save_group(rs, body, rid)


# =========================================================================== 3. significance


@pytest.mark.parametrize(
    "basis",
    [
        "p>0.05; classified this way because it was not statistically significant",
        "the 95% confidence interval includes the null",
        "P value 0.30, consistent with chance",
    ],
)
def test_significance_based_classification_is_refused(session: Session, basis: str) -> None:
    repo, rid, _ = comparative_project(session)
    rs = ReportingService(repo, "mc1")
    body = _group(
        repo,
        rid,
        synthesis_method="vote_counting_direction_of_effect",
        summary="The trial was tabulated without a quantitative summary.",
    )
    body["findings"][0] |= {"direction": "point_estimate_at_null", "direction_basis": basis}
    with pytest.raises(ValueError, match="relies on statistical significance"):
        save_group(rs, body, rid)
    assert repo.load_synthesis_groups("mc1") == []
    old = dict(
        body["findings"][0],
        direction="no_clear_difference",
        direction_basis="observed no difference between arms",
    )
    with pytest.raises(ValueError, match="direction"):  # the significance-like category is gone
        SynthesisFinding.model_validate(old)


def test_undeterminable_and_mixed_directions_are_tabulated_not_counted(
    session: Session,
) -> None:
    repo, rid, _ = comparative_project(session)
    rs = ReportingService(repo, "mc1")
    body = _group(
        repo,
        rid,
        synthesis_method="vote_counting_direction_of_effect",
        summary="The results were tabulated by observed direction.",
    )
    body["findings"].append(
        _second_finding(
            repo,
            direction="direction_not_determinable",
            direction_basis="no arm-level estimate reported for the severity grades",
        )
    )
    save_group(rs, body, rid)
    text = build_evidence_package(repo, "mc1").facts["synthesis.g1.direction"].text
    assert "Of 2 independent study units" in text
    assert "1 favoured the intervention and 0 favoured the comparator" in text
    assert "1 did not report enough to establish a direction (not counted)" in text
    assert "irrespective of statistical significance" in text


# =========================================================================== 4. N/A basis


def test_not_applicable_decision_goes_stale_when_its_section_changes(session: Session) -> None:
    repo, rid, _ = single_arm_project(session)
    svc = ManuscriptService(repo, "ms1")
    complete_manuscript(svc, human(repo, rid))
    rs = ReportingService(repo, "ms1")
    decide_item(
        rs,
        "prisma_2020_main",
        "25",
        rid,
        status="not_applicable",
        section_ids=["supplementary"],
        justification="no external support",
    )
    latest = repo.latest_manuscript_version("ms1")
    assert latest is not None
    sup = latest.section("supplementary")
    assert sup is not None
    svc.edit(
        human(repo, rid),
        {
            "supplementary": sup.source + " Funding: Sponsor Z funded the "
            "review and designed the analysis."
        },
        "funding",
    )
    item = next(r for r in rs.checklist("prisma_2020_main")["items"] if r["item_id"] == "25")
    assert item["stale"] and not item["decided"]
    # re-confirming another item does not revalidate the old decision
    decide_item(
        rs,
        "prisma_2020_main",
        "26",
        rid,
        status="not_applicable",
        justification="re-confirmed after the edit",
    )
    item = next(r for r in rs.checklist("prisma_2020_main")["items"] if r["item_id"] == "25")
    assert item["stale"]
    report = svc.audit()
    assert any("item 25 is stale" in f.message for f in report.findings if f.blocking)


# =========================================================================== 5. ROBINS-I


def test_three_level_downgrade_only_for_robins_i_risk_of_bias() -> None:
    g = _grade(
        approach="interventions_nrsi_robins_i",
        domains=domains(risk_of_bias="extremely_serious"),
        final_certainty="very_low",
        framework_rationale="all studies non-randomised, assessed with ROBINS-I",
    )
    assert g.implied_certainty == "very_low" and g.departure_rationale is None
    with pytest.raises(ValueError, match="applies only to risk of bias with the ROBINS-I"):
        _grade(
            approach="interventions_nrsi_robins_i",
            domains=domains(imprecision="extremely_serious"),
            final_certainty="very_low",
        )
    for approach, start in (
        ("interventions_randomised", "high"),
        ("interventions_nonrandomised_other", "low"),
    ):
        with pytest.raises(ValueError, match="applies only to risk of bias with the ROBINS-I"):
            _grade(
                approach=approach,
                initial_certainty=start,
                domains=domains(risk_of_bias="extremely_serious"),
                final_certainty="very_low",
            )


def test_robins_i_assessment_reports_the_three_level_judgement(session: Session) -> None:
    repo, rid, _ = comparative_project(session)
    # set-up of a different body of evidence, written outside the app (frozen since Session 9)
    with out_of_band(repo, "mc1"):
        for t in ("t1", "t2", "t3"):
            repo.save_rob(
                "mc1",
                RiskOfBiasAssessment(
                    record_id=t,
                    tool=RoBTool.ROBINS_I,
                    domains=[
                        DomainJudgement(
                            domain=d,
                            judgement="critical" if d == "confounding" else "low",
                            support="synthetic",
                        )
                        for d in ROBINS_I_DOMAINS
                    ],
                    overall="critical",
                    assessor=REVIEWER,
                    assessed_by=ActorType.HUMAN,
                    approved_by_human=True,
                    approved_by=REVIEWER,
                    approved_reviewer_id=rid,
                    approved_at=NOW + timedelta(days=1),
                ),
            )
    rs = ReportingService(repo, "mc1")
    view = next(r for r in rs.grade() if r["analysis_id"] == "comp")
    assert "interventions_nrsi_robins_i" in view["applicable_approaches"]
    save_grade(
        rs,
        "comp",
        approval(
            approach="interventions_nrsi_robins_i",
            framework_rationale="non-randomised studies assessed with ROBINS-I (synthetic)",
            domains=domains(risk_of_bias="extremely_serious"),
            final_certainty="very_low",
            summary="rated down three levels for critical risk of bias (confounding)",
        ),
        rid,
    )
    package = build_evidence_package(repo, "mc1")
    statement = package.facts["grade.comp.statement"].text
    assert "very low" in statement and "risk of bias (extremely serious)" in statement
    assert package.facts["grade.comp.certainty"].text == "very low"
    assert "extremely serious" in package.facts["grade.limitations"].text
    assert "{{grade.comp.reasons}}" in package.templates["table:sof"]


# =========================================================================== 6. contradictions


def test_summary_contradicting_the_recorded_directions_is_refused(session: Session) -> None:
    repo, rid, _ = comparative_project(session)
    rs = ReportingService(repo, "mc1")
    for summary in (
        "All studies consistently favoured the comparator.",
        "The trial favoured placebo.",
    ):
        with pytest.raises(SysRevError, match="contradicts the recorded directions"):
            save_group(rs, _group(repo, rid, summary=summary), rid)
    assert repo.load_synthesis_groups("mc1") == []
    save_group(rs, _group(repo, rid, summary="The trial favoured the intervention."), rid)


def test_unverifiable_directional_wording_needs_explicit_review(session: Session) -> None:
    repo, rid, _ = comparative_project(session)
    rs = ReportingService(repo, "mc1")
    save_group(rs, _group(repo, rid, summary="Intervention X reduced complications."), rid)
    svc = ManuscriptService(repo, "mc1")
    complete_manuscript(svc, human(repo, rid))
    decide_swim(rs, "not_applicable", "synthetic: a single tabulated trial", rid)
    svc.rerender(human(repo, rid))
    confirm_checklists(repo, "mc1", rid)
    report = svc.audit()
    review = [
        f
        for f in report.findings
        if f.scope == "reporting"
        and f.status == CheckStatus.REQUIRES_HUMAN_REVIEW
        and "neither confirm" in f.message
    ]
    assert review and "reduced complications" in review[0].message


def test_old_grade_records_cannot_claim_extremely_serious_elsewhere() -> None:
    data = _grade().model_dump(mode="json")
    data["domains"][0] |= {"domain": "inconsistency", "concern": "extremely_serious"}
    data["domains"][2] |= {"domain": "risk_of_bias"}
    data.pop("implied_certainty")
    with pytest.raises(ValueError, match="applies only to risk of bias with the ROBINS-I"):
        GradeAssessment.model_validate(data)


def _audit_with_summary(session_factory: Any, summary: str) -> tuple[list[str], Any]:
    """A complete synthetic manuscript whose approved synthesis has ``summary``; returns the
    unverifiable statements and the audit report."""
    from src.reporting import synthesis as syn
    from src.reporting.state import arms_of

    with session_factory() as session:
        repo, rid, _ = comparative_project(session)
        rs = ReportingService(repo, "mc1")
        group = save_group(rs, _group(repo, rid, summary=summary), rid)
        protocol = repo.latest_protocol("mc1")
        assert protocol is not None
        assert syn.summary_problems(group, arms_of(protocol)) == []
        unverifiable = syn.unverifiable_statements(group, arms_of(protocol))
        svc = ManuscriptService(repo, "mc1")
        complete_manuscript(svc, human(repo, rid))
        decide_swim(rs, "not_applicable", "synthetic: a single tabulated trial", rid)
        svc.rerender(human(repo, rid))
        confirm_checklists(repo, "mc1", rid)
        return unverifiable, svc.audit()


@pytest.mark.parametrize(
    "compound",
    [
        "The trial favoured the intervention, and placebo reduced complications.",
        "The trial favoured the intervention and reduced complications.",
    ],
)
def test_checked_clause_does_not_hide_an_unverifiable_interpretation(compound: str) -> None:
    def factory() -> Session:
        engine = make_engine("sqlite:///:memory:")
        init_db(engine)
        return make_session_factory(engine)()

    standalone = "Placebo reduced complications."
    for summary in (standalone, compound):
        unverifiable, report = _audit_with_summary(factory, summary)
        assert unverifiable == [summary]  # the interpretation is never exempted
        review = [
            f
            for f in report.findings
            if f.scope == "reporting"
            and f.status == CheckStatus.REQUIRES_HUMAN_REVIEW
            and "neither confirm" in f.message
        ]
        assert len(review) == 1 and summary in review[0].message
        assert report.passed  # it passes only once a human acknowledges the review item
        assert report.requires_human_review >= 1
    # a sentence whose only directional claim was checked is not over-flagged
    only_checked, report = _audit_with_summary(factory, "The trial favoured the intervention.")
    assert only_checked == []
    assert not [f for f in report.findings if "neither confirm" in f.message]
