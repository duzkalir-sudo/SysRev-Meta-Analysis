"""Session 9 remediation: regressions for the audit findings H1-H3 and M1-M5.

Every scenario uses SYNTHETIC data (invented PDFs, records and numbers; 10.5555 test DOIs).
Refusals are checked together with "nothing was written"; each refusal has a successful
control showing the supported route.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.api_server.app import create_app
from src.core.enums import (
    ActorType,
    GateId,
    GateVerdict,
    Phase,
    ReviewStatus,
    ScreeningStage,
    ScreeningVerdict,
)
from src.core.exceptions import SysRevError
from src.core.schemas.pipeline import GateDecision
from src.core.schemas.screening import ScreeningDecision
from src.extraction import (
    ReviewError,
    arm_basis,
    build_arms,
    correction_state_sha256,
    link_reports_to_study,
    links_state_sha256,
    record_unavailability,
    reject_item,
    revise_reviewed_item,
    submit_proposals,
)
from src.fulltext import StaleSourceError, confirm_pdf_match, register_upload
from src.screening.llm_screening import adjudicate
from src.state.db import (
    ExtractedArmORM,
    ExtractionItemORM,
    ExtractionReviewEventORM,
    FullTextDocumentORM,
    GovernanceRecordORM,
    ScreeningDecisionORM,
)
from src.state.machine import PipelineStateMachine
from src.state.repository import Repository
from src.stats_engine.engine import run_plan
from tests.conftest import NOW, approved_rob, make_evidence
from tests.manuscript_helpers import comparative_project, out_of_band
from tests.unit.test_master_table import OTHER_PDF, PDF, PID, Project, crude, km
from tests.unit.test_master_table import project as master_project  # noqa: F401 (fixture)

REVISED_PDF = PDF + b"\n% synthetic revision of the same report\n"
P13 = "p13"  # project id of the tests.unit.test_pr13_review_fixes builder


@pytest.fixture
def project(master_project: Project) -> Project:  # noqa: F811
    master_project.repo.get_project(PID).phase = Phase.STAGE4_EXTRACTION_ROB.value
    master_project.repo.session.flush()
    return master_project


def rows(repo: Repository) -> tuple[int, ...]:
    """Row counts of every table a refused operation must leave untouched."""
    s = repo.session
    return tuple(
        int(s.scalar(select(func.count()).select_from(t)) or 0)
        for t in (
            ExtractionItemORM,
            ExtractionReviewEventORM,
            ExtractedArmORM,
            GovernanceRecordORM,
            ScreeningDecisionORM,
            FullTextDocumentORM,
        )
    )


def arms_of(repo: Repository, rid: str = "r1") -> list[Any]:
    return [a for a in repo.load_extracted_arms(PID) if a.record_id == rid]


def integrity(repo: Repository) -> tuple[str, ...]:
    return repo.build_guard_context(PID).extraction_integrity_problems


# =========================================================================== H1
def test_h1_value_accepted_after_build_invalidates_the_arm_and_blocks_until_rebuilt(
    project: Project,
) -> None:
    repo = project.repo
    (est,) = project.submit(km("reported_estimate", "62.1", 1, 1, 3))
    project.accept(est)
    (old,) = build_arms(repo, PID, "r1", reviewer=project.reviewer)
    assert set(arm_basis(repo, PID, old.arm_id) or {}) == {est.item_id}
    assert not any(p.startswith("r1 /") for p in integrity(repo))
    (risk,) = project.submit(km("n_at_risk", "40", 2, 1, 3))
    project.accept(risk)
    # the stale arm is withdrawn from every reader, kept as history, and Gate 3 is blocked
    assert old.arm_id in repo.invalidated_arm_ids(PID)
    assert repo.session.get(ExtractedArmORM, old.arm_id) is not None
    assert arms_of(repo) == []
    assert any("not represented by any approved arm" in p for p in integrity(repo))
    history = [e.action for e in repo.list_review_events(PID, risk.item_id)]
    assert history == ["proposed", "accepted", "arms_invalidated"]
    # rebuilding from the current accepted values restores consistency
    (new,) = build_arms(repo, PID, "r1", reviewer=project.reviewer)
    assert new.arm_id != old.arm_id and new.n_at_risk is not None
    assert new.n_at_risk.value == 40
    assert set(arm_basis(repo, PID, new.arm_id) or {}) == {est.item_id, risk.item_id}
    assert not any(p.startswith("r1 /") for p in integrity(repo))


def test_h1_rejected_value_corrected_and_accepted_invalidates_the_arm(project: Project) -> None:
    repo = project.repo
    est, risk = project.submit(
        km("reported_estimate", "62.1", 1, 1, 3), km("n_at_risk", "22", 2, 1, 3)
    )
    project.accept(est)
    reject_item(repo, PID, risk.item_id, reviewer=project.reviewer, note="wrong column")
    (old,) = build_arms(repo, PID, "r1", reviewer=project.reviewer)
    assert old.n_at_risk is None
    shown = correction_state_sha256(repo, PID, repo.get_extraction_item(PID, risk.item_id))
    new, invalidated = revise_reviewed_item(
        repo, PID, risk.item_id, km("n_at_risk", "40", 2, 1, 3), project.doc("r1"),
        reviewer=project.reviewer, note="3-year column", expected_state_sha256=shown,
    )  # fmt: skip
    assert invalidated == [old.arm_id] and new.review_status == ReviewStatus.ACCEPTED
    assert repo.get_extraction_item(PID, risk.item_id).review_status == ReviewStatus.SUPERSEDED
    (rebuilt,) = build_arms(repo, PID, "r1", reviewer=project.reviewer)
    assert rebuilt.n_at_risk is not None and rebuilt.n_at_risk.value == 40


def test_h1_accepting_a_value_of_another_arm_leaves_a_valid_arm_alone(project: Project) -> None:
    repo = project.repo
    (est,) = project.submit(km("reported_estimate", "62.1", 1, 1, 3))
    project.accept(est)
    (arm,) = build_arms(repo, PID, "r1", reviewer=project.reviewer)
    events, n = project.submit(crude("events", "45"), crude("n_analyzed", "57"))
    project.accept(events)  # a different arm key (crude, last follow-up): no arm exists yet
    project.accept(n)
    assert arm.arm_id not in repo.invalidated_arm_ids(PID)
    assert any("not represented by any approved arm" in p for p in integrity(repo))
    built = build_arms(repo, PID, "r1", reviewer=project.reviewer)
    assert len(built) == 1 and built[0].events is not None
    assert {a.arm_id for a in arms_of(repo)} == {arm.arm_id, built[0].arm_id}
    assert not any(p.startswith("r1 /") for p in integrity(repo))


def test_h1_arm_basis_mismatch_written_outside_the_app_is_detected(project: Project) -> None:
    repo = project.repo
    (est,) = project.submit(km("reported_estimate", "62.1", 1, 1, 3))
    project.accept(est)
    (arm,) = build_arms(repo, PID, "r1", reviewer=project.reviewer)
    (risk,) = project.submit(km("n_at_risk", "40", 2, 1, 3))
    me = project.reviewer
    accepted = risk.model_copy(  # an acceptance written directly to the database
        update={"review_status": ReviewStatus.ACCEPTED, "reviewed_by": me.display_name,
                "reviewer_id": me.id, "reviewed_at": NOW}
    )  # fmt: skip
    row = repo.session.get(ExtractionItemORM, risk.item_id)
    assert row is not None
    row.review_status, row.payload = "accepted", accepted.model_dump(mode="json")
    repo.session.flush()
    assert arms_of(repo)[0].arm_id == arm.arm_id
    assert any("no longer matches the accepted values" in p for p in integrity(repo))


# =========================================================================== H2
def _gate3_valid_stage5(session: Session) -> tuple[Repository, str]:
    """A complete synthetic comparative project taken through a *real* Gate 3 approval."""
    repo, rid, _ = comparative_project(session)
    repo.get_project("mc1").phase = Phase.GATE3_EXTRACTION_APPROVAL.value
    session.flush()
    reviewer = repo.require_active_reviewer(rid)
    machine = repo.load_state_machine("mc1")
    decision = GateDecision(
        gate=GateId.GATE_3, verdict=GateVerdict.APPROVED,
        reviewer_name=reviewer.display_name, reviewer_id=rid,
    )  # fmt: skip
    repo.record_transition(machine.submit_gate_decision(decision, repo.build_guard_context("mc1")))
    assert repo.current_phase("mc1") == Phase.STAGE5_META_ANALYSIS.value
    assert repo.governance_records("mc1", "gate3_basis")
    return repo, rid


def _rerun_stats(repo: Repository) -> None:
    plan = repo.latest_analysis_plan("mc1")
    assert plan is not None
    output = run_plan(plan, repo.load_stage5_inputs("mc1"), project_id="mc1", generated_at=NOW)
    repo.record_stats_output("mc1", output)


def _stage6_violations(repo: Repository) -> list[str]:
    return PipelineStateMachine("mc1", Phase.STAGE5_META_ANALYSIS).check(
        Phase.STAGE6_MANUSCRIPT, repo.build_guard_context("mc1")
    )


def test_h2_gate3_valid_project_opens_stage6(session: Session) -> None:
    repo, _ = _gate3_valid_stage5(session)
    assert repo.stats_output_problems("mc1") == []
    assert _stage6_violations(repo) == []  # successful control


def test_h2_gate3_approved_data_are_refused_in_stage5_before_writing(session: Session) -> None:
    repo, rid = _gate3_valid_stage5(session)
    reviewer = repo.require_active_reviewer(rid)
    before = rows(repo)
    attempts = [
        lambda: adjudicate(
            repo, "mc1", "t1", ScreeningStage.TITLE_ABSTRACT, verdict=ScreeningVerdict.EXCLUDE,
            rationale="late", exclusion_reason="E1", reviewer=reviewer,
        ),
        lambda: repo.save_screening_decision(
            "mc1",
            ScreeningDecision(record_id="t1", reviewer="reviewer_1", reviewer_identity="agent",
                              stage=ScreeningStage.TITLE_ABSTRACT,
                              verdict=ScreeningVerdict.EXCLUDE, rationale="late",
                              exclusion_reason="E1"),
        ),
        lambda: repo.save_rob("mc1", approved_rob("t1", rid, name=reviewer.display_name)),
        lambda: repo.save_extracted_arm("mc1", repo.load_extracted_arms("mc1")[0].model_copy(
            update={"arm_id": "f" * 32})),
        lambda: repo.invalidate_arm("mc1", repo.load_extracted_arms("mc1")[0].arm_id,
                                    reviewer_id=rid, item_id=None, reason="late"),
    ]  # fmt: skip
    for attempt in attempts:
        with pytest.raises(SysRevError, match="approved at Gate 3"):
            attempt()
        assert rows(repo) == before
    assert _stage6_violations(repo) == []


def test_h2_supported_stage5_operations_still_work(session: Session) -> None:
    repo, rid = _gate3_valid_stage5(session)
    reviewer = repo.require_active_reviewer(rid)
    link_reports_to_study(
        repo, "mc1", ["t1", "t2"], study_id="S", primary_record_id="t1", reviewer=reviewer,
        note="synthetic: one trial, two reports",
        expected_state_sha256=links_state_sha256(repo, "mc1"),
    )  # fmt: skip
    # the link creates double counting: the Gate 3 re-check blocks Stage 6 until it is resolved
    assert any("more than one report of the same cohort" in v for v in _stage6_violations(repo))
    record_unavailability(
        repo, "mc1", "t2", reviewer=reviewer, reason_code="overlapping_cohort",
        reason="same trial as t1", outcome_id="complication",
    )  # fmt: skip
    assert any("re-run Stage 5" in v for v in _stage6_violations(repo))
    _rerun_stats(repo)
    assert _stage6_violations(repo) == []


def test_h2_pending_item_blocks_stage6_even_after_stats_are_recomputed(session: Session) -> None:
    repo, _ = _gate3_valid_stage5(session)
    from src.core.enums import DataStatus, ExtractionDomain
    from src.core.schemas.master import ExtractionItem

    pending = ExtractionItem(
        record_id="t1", domain=ExtractionDomain.STUDY_CHARACTERISTICS, field_name="setting",
        value_kind="text", data_status=DataStatus.NOT_REPORTED, proposed_by=ActorType.AGENT,
        proposer_identity="llm:test",
    )  # fmt: skip
    with pytest.raises(SysRevError, match="approved at Gate 3"):
        repo.save_extraction_item("mc1", pending)
    with out_of_band(repo, "mc1"):  # written outside the application
        repo.save_extraction_item("mc1", pending)
    _rerun_stats(repo)  # a pending item does not change the statistics inputs
    assert repo.stats_output_problems("mc1") == []
    violations = _stage6_violations(repo)
    assert any("1 master-table extraction item(s) await human review" in v for v in violations)
    assert any("Gate 3 approval" in v for v in violations)  # the approved data changed


@pytest.mark.parametrize("change", ["rob", "eligibility", "prisma"])
def test_h2_incomplete_rob_changed_eligibility_or_stale_prisma_block_stage6(
    session: Session, change: str
) -> None:
    repo, rid = _gate3_valid_stage5(session)
    from src.core.schemas.pipeline import PRISMACounts

    with out_of_band(repo, "mc1"):
        if change == "rob":
            for a in repo.load_rob("mc1"):
                if a.record_id == "t1":
                    repo.save_rob("mc1", a.model_copy(update={
                        "approved_by_human": False, "approved_by": None,
                        "approved_reviewer_id": None, "approved_at": None}))  # fmt: skip
        elif change == "eligibility":
            reviewer = repo.require_active_reviewer(rid)
            repo.session.execute(  # the human adjudication of t3 is replaced by an exclusion
                ScreeningDecisionORM.__table__.delete().where(
                    ScreeningDecisionORM.record_id == "t3",
                    ScreeningDecisionORM.stage == ScreeningStage.FULL_TEXT.value,
                    ScreeningDecisionORM.reviewer == "adjudicator",
                )
            )
            adjudicate(repo, "mc1", "t3", ScreeningStage.FULL_TEXT,
                       verdict=ScreeningVerdict.EXCLUDE, rationale="late", exclusion_reason="E1",
                       reviewer=reviewer)  # fmt: skip
        else:
            stale = repo.load_prisma("mc1")
            assert stale is not None
            data = stale.model_dump()
            db = next(iter(data["identified_by_database"]))
            data["identified_by_database"][db] += 1  # internally consistent, but stale
            for key in ("total_identified", "records_screened", "records_excluded"):
                if key in data:
                    data[key] += 1
            repo.save_prisma("mc1", PRISMACounts.model_validate(data))
    violations = _stage6_violations(repo)
    expected = {
        "rob": "lack a human-approved risk-of-bias assessment",
        "eligibility": "belong to reports not included",
        "prisma": "PRISMA",
    }[change]
    assert any(expected in v for v in violations), violations


def test_h2_extraction_operations_are_refused_after_gate3(project: Project) -> None:
    repo = project.repo
    (est,) = project.submit(km("reported_estimate", "62.1", 1, 1, 3))
    project.accept(est)
    build_arms(repo, PID, "r1", reviewer=project.reviewer)
    (pending,) = project.submit(km("n_at_risk", "40", 2, 1, 3))
    repo.get_project(PID).phase = Phase.STAGE5_META_ANALYSIS.value
    repo.session.flush()
    before = rows(repo)
    for attempt in (
        lambda: project.submit(crude("events", "45")),
        lambda: project.accept(pending),
        lambda: reject_item(repo, PID, pending.item_id, reviewer=project.reviewer, note="x"),
        lambda: build_arms(repo, PID, "r2", reviewer=project.reviewer),
    ):
        with pytest.raises((SysRevError, ReviewError), match="approved at Gate 3"):
            attempt()
        assert rows(repo) == before


# =========================================================================== H3
def test_h3_pending_item_verified_against_a_replaced_pdf_cannot_be_accepted(
    project: Project,
) -> None:
    repo = project.repo
    (risk,) = project.submit(km("n_at_risk", "40", 2, 1, 3))
    old = repo.get_fulltext_item(PID, "r1")
    assert old is not None and old.pdf_sha256
    register_upload(repo, project.store, PID, "r1", REVISED_PDF, original_filename="v2.pdf",
                    reviewer=project.reviewer, replaces_sha256=old.pdf_sha256)  # fmt: skip
    before = rows(repo)
    with pytest.raises(ReviewError, match="source was replaced"):
        project.accept(risk)
    assert rows(repo) == before
    assert repo.get_extraction_item(PID, risk.item_id).review_status == ReviewStatus.PENDING


def test_h3_replacement_invalidates_arms_and_requires_reverification(project: Project) -> None:
    repo = project.repo
    (est,) = project.submit(km("reported_estimate", "62.1", 1, 1, 3))
    project.accept(est)
    (arm,) = build_arms(repo, PID, "r1", reviewer=project.reviewer)
    old = repo.get_fulltext_item(PID, "r1")
    assert old is not None and old.pdf_sha256
    item = register_upload(
        repo, project.store, PID, "r1", REVISED_PDF, original_filename="v2.pdf",
        reviewer=project.reviewer, replaces_sha256=old.pdf_sha256,
    )  # fmt: skip
    if item.match_unconfirmed:
        confirm_pdf_match(repo, PID, "r1", reviewer=project.reviewer, note="same report",
                          expected_pdf_sha256=item.pdf_sha256)  # fmt: skip
    assert arm.arm_id in repo.invalidated_arm_ids(PID)
    assert repo.session.get(ExtractedArmORM, arm.arm_id) is not None  # history kept
    replacement = repo.governance_records(PID, "source_replacement", "r1")
    assert replacement and replacement[-1]["invalidated_arms"] == [arm.arm_id]
    assert any("cite a PDF that is no longer the linked one" in p for p in integrity(repo))
    with pytest.raises(ReviewError, match="source was replaced"):  # old values cannot be built
        build_arms(repo, PID, "r1", reviewer=project.reviewer)
    # re-verification against the new PDF (a human correction with the same printed number)
    shown = correction_state_sha256(repo, PID, repo.get_extraction_item(PID, est.item_id))
    new, _ = revise_reviewed_item(
        repo, PID, est.item_id, km("reported_estimate", "62.1", 1, 1, 3), project.doc("r1"),
        reviewer=project.reviewer, note="re-verified against the revised PDF",
        expected_state_sha256=shown,
    )  # fmt: skip
    assert new.evidence is not None and new.evidence.pdf_sha256 == item.pdf_sha256
    (rebuilt,) = build_arms(repo, PID, "r1", reviewer=project.reviewer)
    assert rebuilt.reported_estimate is not None
    assert rebuilt.reported_estimate.evidence.pdf_sha256 == item.pdf_sha256
    assert not any(p.startswith(("r1:", "r1 /")) for p in integrity(repo))


def test_h3_unnamed_or_stale_replacement_and_stale_confirmation_store_nothing(
    project: Project,
) -> None:
    repo = project.repo
    before_files = sorted(p.name for p in project.store.root.rglob("*") if p.is_file())
    before = rows(repo)
    with pytest.raises(StaleSourceError, match="must name that PDF"):
        register_upload(repo, project.store, PID, "r1", REVISED_PDF, original_filename="x.pdf",
                        reviewer=project.reviewer)  # fmt: skip
    with pytest.raises(StaleSourceError, match="must name that PDF"):
        register_upload(repo, project.store, PID, "r1", REVISED_PDF, original_filename="x.pdf",
                        reviewer=project.reviewer, replaces_sha256="0" * 64)  # fmt: skip
    with pytest.raises(StaleSourceError, match="not the"):
        confirm_pdf_match(repo, PID, "r1", reviewer=project.reviewer, note="old screen",
                          expected_pdf_sha256="0" * 64)  # fmt: skip
    assert rows(repo) == before
    assert sorted(p.name for p in project.store.root.rglob("*") if p.is_file()) == before_files


def test_h3_late_phase_replacement_is_refused_before_anything_is_stored(
    project: Project,
) -> None:
    repo = project.repo
    old = repo.get_fulltext_item(PID, "r1")
    assert old is not None and old.pdf_sha256
    repo.get_project(PID).phase = Phase.STAGE6_MANUSCRIPT.value
    repo.session.flush()
    before_files = sorted(p.name for p in project.store.root.rglob("*") if p.is_file())
    before = rows(repo)
    with pytest.raises(SysRevError, match="approved at Gate 3"):
        register_upload(repo, project.store, PID, "r1", OTHER_PDF, original_filename="o.pdf",
                        reviewer=project.reviewer, replaces_sha256=old.pdf_sha256)  # fmt: skip
    with pytest.raises(SysRevError, match="approved at Gate 3"):
        confirm_pdf_match(repo, PID, "r1", reviewer=project.reviewer, note="late",
                          expected_pdf_sha256=old.pdf_sha256)  # fmt: skip
    assert rows(repo) == before
    assert sorted(p.name for p in project.store.root.rglob("*") if p.is_file()) == before_files
    current = repo.get_fulltext_item(PID, "r1")
    assert current is not None and current.pdf_sha256 == old.pdf_sha256


def test_h3_http_replacement_confirmation_and_acceptance(tmp_path: Path) -> None:
    from tests.unit.test_pr13_review_fixes import accepted_estimate, build, stage

    with TestClient(create_app("sqlite:///:memory:", data_dir=tmp_path)) as c:
        with c.app.state.session_factory() as s:  # type: ignore[attr-defined]
            p = build(s, tmp_path / "fulltext", {"r1": PDF})
            stage(p.repo, Phase.STAGE4_EXTRACTION_ROB)
            accepted_estimate(p, arms=True)
            (pending,) = submit_proposals(
                p.repo, P13, "r1", p.doc("r1"), [km("n_at_risk", "40", 2, 1, 3)],
                proposed_by=ActorType.AGENT, proposer_identity="llm:test",
            )  # fmt: skip
            _, token = p.repo.enroll_reviewer("Dr HTTP")
            s.commit()
        auth = {"Authorization": f"Bearer {token}"}
        base = f"/projects/{P13}/fulltext/r1"
        manifest = c.get(f"/projects/{P13}/fulltext/manifest", params={"only_open": False})
        row = next(r for r in manifest.json() if r["record_id"] == "r1")
        shown = row["pdf_sha256"]
        # an unnamed replacement and a confirmation of another PDF are refused (409)
        r = c.put(f"{base}/pdf", content=REVISED_PDF, headers=auth | {"X-Filename": "v2.pdf"})
        assert r.status_code == 409 and "must name that PDF" in r.json()["detail"]
        r = c.post(f"{base}/confirm-match", headers=auth,
                   json={"note": "x", "expected_pdf_sha256": "0" * 64})  # fmt: skip
        assert r.status_code == 409
        r = c.post(f"{base}/confirm-match", headers=auth, json={"note": "x"})
        assert r.status_code == 422  # the examined PDF must be named
        # control: a replacement naming the shown PDF is linked and invalidates the arm
        r = c.put(f"{base}/pdf", content=REVISED_PDF,
                  headers=auth | {"X-Filename": "v2.pdf", "X-Replaces-SHA256": shown})  # fmt: skip
        assert r.status_code == 201
        new_sha = r.json()["pdf_sha256"]
        if r.json()["match_status"] not in ("confirmed", "human_confirmed"):
            r = c.post(f"{base}/confirm-match", headers=auth,
                       json={"note": "same report", "expected_pdf_sha256": new_sha})  # fmt: skip
            assert r.status_code == 200
        # the pending value verified against the old PDF is not acceptable
        r = c.post(f"/projects/{P13}/extraction/items/{pending.item_id}/accept", headers=auth,
                   json={"note": None})  # fmt: skip
        assert r.status_code == 409 and "source was replaced" in r.json()["detail"]
        # late phase: refused before anything is stored
        with c.app.state.session_factory() as s:  # type: ignore[attr-defined]
            stage(Repository(s), Phase.STAGE5_META_ANALYSIS, P13)
            s.commit()
        r = c.put(f"{base}/pdf", content=OTHER_PDF,
                  headers=auth | {"X-Filename": "o.pdf", "X-Replaces-SHA256": new_sha})  # fmt: skip
        assert r.status_code == 409 and "approved at Gate 3" in r.json()["detail"]
        with c.app.state.session_factory() as s:  # type: ignore[attr-defined]
            ft = Repository(s).get_fulltext_item(P13, "r1")
            assert ft is not None and ft.pdf_sha256 == new_sha


def test_h3_approval_of_rob_citing_a_replaced_pdf_is_reported(project: Project) -> None:
    repo = project.repo
    old = repo.get_fulltext_item(PID, "r1")
    assert old is not None and old.pdf_sha256
    rob = approved_rob("r1", project.reviewer.id, name=project.reviewer.display_name)
    domains = [
        d.model_copy(
            update={"evidence": [make_evidence("Complete obliteration", pdf_sha256=old.pdf_sha256)]}
        )
        for d in rob.domains
    ]
    repo.save_rob(PID, rob.model_copy(update={"domains": domains}))
    assert not any("risk-of-bias" in p for p in integrity(repo))
    register_upload(repo, project.store, PID, "r1", REVISED_PDF, original_filename="v2.pdf",
                    reviewer=project.reviewer, replaces_sha256=old.pdf_sha256)  # fmt: skip
    # the remedy is a new assessment approved against the linked PDF (documented)
    assert any("risk-of-bias assessment cites a PDF" in p for p in integrity(repo))


# =========================================================================== M1 / M2
def _km_spec(scale: str) -> Any:
    from src.core.schemas.stats import AnalysisSpec
    from tests.stats_helpers import YEARS_5

    return AnalysisSpec(
        analysis_id="k", outcome_id="control_km", effect_measure="KM_PROBABILITY",
        timepoints=[YEARS_5], reported_ci_scale=scale,
    )  # fmt: skip


def _km(arms: list[Any], scale: str) -> Any:
    from src.stats_engine.engine import run_analysis
    from tests.stats_helpers import OUTCOMES, inputs

    return run_analysis(_km_spec(scale), OUTCOMES["control_km"], inputs(arms))


def test_m1_identity_scale_keeps_a_valid_zero_confidence_limit() -> None:
    from tests.stats_helpers import km_arm

    arms = [km_arm("s1", "40.0", "30.0", "50.0"), km_arm("s2", "50.0", "42.0", "58.0"),
            km_arm("s3", "6.0", "0.0", "12.0")]  # fmt: skip
    res = _km(arms, "plain")
    assert res.status == "pooled" and res.k == 3 and not res.excluded
    s3 = next(s for s in res.studies if s.study_id == "s3")
    z = 1.959963984540054
    assert s3.yi == pytest.approx(0.06) and s3.vi == pytest.approx(((0.12 - 0.0) / (2 * z)) ** 2)
    assert s3.data["ci_lower"] == 0.0  # the printed source value is unchanged


def test_m1_transformed_scale_excludes_only_the_untransformable_study() -> None:
    from tests.stats_helpers import km_arm

    arms = [km_arm("a", "92.3", "87.7", "95.2"), km_arm("b", "85.0", "78.6", "89.6"),
            km_arm("zero", "6.0", "0.0", "12.0")]  # fmt: skip
    res = _km(arms, "log_log")
    assert res.status == "pooled" and res.k == 2
    (excluded,) = res.excluded
    assert excluded.study_id == "zero" and excluded.reason_code == "ci_bound_not_transformable"
    assert "outside the domain (0, 1) of the declared log_log transformation" in excluded.reason


def test_m2_each_printed_number_has_its_own_rounding_tolerance() -> None:
    from tests.stats_helpers import km_arm

    # true 62.34 % with a log(-log) CI (SE 0.10): estimate printed to 0, limits to 1 decimal
    mixed = km_arm("mixed", "62", "56.3", "67.8")
    res = _km([mixed, km_arm("b", "85.0", "78.6", "89.6")], "log_log")
    study = next(s for s in res.studies if s.study_id == "mixed")
    assert study.data["rounding_half_unit_estimate"] == pytest.approx(0.005)
    assert study.data["rounding_half_unit_ci_lower"] == pytest.approx(0.0005)
    assert study.data["rounding_half_unit_ci_upper"] == pytest.approx(0.0005)
    assert study.data["estimate"] == 0.62  # printed value used, not the reconstructed one
    # the same limits around an estimate printed as "50.0" are genuinely inconsistent
    wrong = km_arm("wrong", "58.0", "56.3", "67.8")
    plain_sym = km_arm("plain_sym", "50.0", "42.0", "58.0")
    res = _km([mixed, wrong, plain_sym, km_arm("b", "85.0", "78.6", "89.6")], "log_log")
    codes = {e.study_id: e.reason_code for e in res.excluded}
    assert codes == {"wrong": "ci_inconsistent", "plain_sym": "ci_inconsistent"}
    assert "estimate ±0.0005, lower ±0.0005, upper ±0.0005" in next(
        e.reason for e in res.excluded if e.study_id == "wrong"
    )
    # supported scales: a CI symmetric on the identity scale passes there
    assert _km([plain_sym, km_arm("c", "40.0", "30.0", "50.0")], "plain").k == 2


# =========================================================================== M3
METAFOR_MH_WEIGHTS_RR = [45.48829, 33.69503, 20.81668]  # metafor 4.4-0 weights.rma.mh
METAFOR_MH_WEIGHTS_OR = [44.75733, 33.90707, 21.33559]


@pytest.mark.parametrize("measure", ["RR", "OR"])
def test_m3_common_effect_weights_are_the_mantel_haenszel_weights(
    session: Session, measure: str
) -> None:
    from src.core.schemas.stats import AnalysisSpec
    from src.stats_engine.engine import IV_COMMON_LABEL, run_analysis
    from src.stats_engine.report import analysis_csv, analysis_markdown

    repo, _, _ = comparative_project(session)
    inputs = repo.load_stage5_inputs("mc1")
    spec = AnalysisSpec(analysis_id="c", outcome_id="complication", effect_measure=measure)
    a = run_analysis(spec, inputs.outcomes["complication"], inputs)
    assert a.common_effect is not None and a.common_effect.method.startswith("Mantel-Haenszel")
    expected = METAFOR_MH_WEIGHTS_RR if measure == "RR" else METAFOR_MH_WEIGHTS_OR
    assert [s.weight_common_pct for s in a.studies] == pytest.approx(expected, abs=1e-5)
    iv = [s.weight_sensitivity_pct[IV_COMMON_LABEL] for s in a.studies]
    if measure == "RR":  # the IV weights the audit found in the CE column (now sensitivity)
        assert iv == pytest.approx([47.902, 32.514, 19.584], abs=1e-3)
    assert sum(iv) == pytest.approx(100.0)
    assert all(s.weight_random_pct is not None for s in a.studies)
    md = analysis_markdown(a)
    assert "Weight CE (Mantel-Haenszel) %" in md and f"Weight {IV_COMMON_LABEL} %" in md
    header = analysis_csv(a).splitlines()[0].split(",")
    assert {"weight_common_model", "weight_random_model", "weight_sensitivity_pct"} <= set(header)
    # pooled estimates are unchanged by the change of weights (validated against metafor)
    assert a.common_effect.estimate_natural > 0


def test_m3_glmm_primary_shows_no_study_weights_and_labels_iv_logit_ones(tmp_path: Path) -> None:
    from src.core.schemas.stats import AnalysisSpec
    from src.stats_engine.engine import IV_LOGIT_RE_LABEL, run_analysis
    from src.stats_engine.report import analysis_markdown, forest_plot
    from tests.stats_helpers import LAST_FU, OUTCOMES, inputs, necrosis_arms

    spec = AnalysisSpec(analysis_id="n", outcome_id="necrosis", effect_measure="PROPORTION",
                        timepoints=[LAST_FU])  # fmt: skip
    a = run_analysis(spec, OUTCOMES["necrosis"], inputs(necrosis_arms()))
    assert a.status == "pooled" and a.primary is not None
    assert a.primary.method.startswith("binomial-normal GLMM")
    assert all(s.weight_random_pct is None and s.weight_common_pct is None for s in a.studies)
    iv = [s.weight_sensitivity_pct[IV_LOGIT_RE_LABEL] for s in a.studies]
    assert sum(iv) == pytest.approx(100.0)
    md = analysis_markdown(a)
    assert "Weight RE (GLMM: no study weights) %" in md and IV_LOGIT_RE_LABEL in md
    svg = forest_plot(a, tmp_path / "n.svg")
    assert svg is not None and "equal box sizes (GLMM: no study weights)" in svg.read_text()
    # logit-IV primary: the reported random-effects model has (inverse-variance) weights
    spec_iv = spec.model_copy(update={"proportion_method": "logit_inverse_variance"})
    b = run_analysis(spec_iv, OUTCOMES["necrosis"], inputs(necrosis_arms()))
    assert [s.weight_random_pct for s in b.studies] == pytest.approx(iv)
    assert all(not s.weight_sensitivity_pct for s in b.studies)
    svg_b = forest_plot(b, tmp_path / "b.svg")
    assert svg_b is not None and "weight % (inverse-variance logit)" in svg_b.read_text()


# =========================================================================== M4
def _stage5_project(session: Session) -> tuple[Repository, str]:
    repo, rid, _ = comparative_project(session)
    repo.get_project("mc1").phase = Phase.STAGE5_META_ANALYSIS.value
    session.flush()
    return repo, rid


def test_m4_real_rerun_retires_the_plot_of_an_analysis_without_data(
    session: Session, tmp_path: Path
) -> None:
    from src.stats_engine.stage5 import run_stage5, verified_plots

    repo, rid = _stage5_project(session)
    out = tmp_path / "stats"
    first = run_stage5(repo, "mc1", output_dir=out, generated_at=NOW)
    assert first.output.analyses[0].status == "pooled"
    assert [n for n, _ in verified_plots(out, first.sha256)[0]] == ["comp"]
    (out / "notes.txt").write_text("a reviewer's own file")  # unrelated: never removed
    reviewer = repo.require_active_reviewer(rid)
    for t in ("t1", "t2", "t3"):  # a supported Stage 5 operation that removes all data
        record_unavailability(repo, "mc1", t, reviewer=reviewer, reason_code="other",
                              reason="synthetic: outcome definition differs",
                              outcome_id="complication")  # fmt: skip
    second = run_stage5(repo, "mc1", output_dir=out, generated_at=NOW)
    assert second.output.analyses[0].status == "no_data"
    assert not (out / "plots" / "comp.svg").exists()
    assert verified_plots(out, second.sha256) == ([], [])
    assert (out / "notes.txt").read_text() == "a reviewer's own file"
    # the previous output's plots are never shown for the current output
    plots, withheld = verified_plots(out, first.sha256)
    assert plots == [] and withheld and "another statistics output" in withheld[0]


def test_m4_edited_or_unlisted_plots_are_never_shown(session: Session, tmp_path: Path) -> None:
    from src.stats_engine.stage5 import run_stage5, verified_plots

    repo, rid = _stage5_project(session)
    out = tmp_path / "stats"
    art = run_stage5(repo, "mc1", output_dir=out, generated_at=NOW)
    (out / "plots" / "stray.svg").write_text("<svg/>")  # not in the manifest
    (out / "plots" / "comp.svg").write_text("<svg>edited</svg>")
    plots, withheld = verified_plots(out, art.sha256)
    assert plots == [] and any("modified after it was generated" in w for w in withheld)
    # a rerun keeps the edited file (hash differs from the manifest) only if it is not
    # regenerated; here it is regenerated, and the stray file is left alone
    reviewer = repo.require_active_reviewer(rid)
    for t in ("t1", "t2", "t3"):
        record_unavailability(repo, "mc1", t, reviewer=reviewer, reason_code="other",
                              reason="synthetic", outcome_id="complication")  # fmt: skip
    again = run_stage5(repo, "mc1", output_dir=out, generated_at=NOW)
    assert (out / "plots" / "comp.svg").read_text() == "<svg>edited</svg>"  # kept: edited
    assert (out / "plots" / "stray.svg").exists()
    assert verified_plots(out, again.sha256) == ([], [])


def test_m4_ui_shows_only_plots_of_the_current_valid_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    from src.core.config import get_settings
    from src.state.db import init_db, make_engine, make_session_factory
    from src.stats_engine.stage5 import run_stage5

    url = f"sqlite:///{tmp_path / 'ui.db'}"
    data_dir = tmp_path / "data"
    monkeypatch.setenv("SYSREV_DATABASE_URL", url)
    monkeypatch.setenv("SYSREV_DATA_DIR", str(data_dir))
    monkeypatch.delenv("SYSREV_API_URL", raising=False)
    get_settings.cache_clear()
    st.cache_resource.clear()
    engine = make_engine(url)
    init_db(engine)
    stats_dir = data_dir / "output" / "mc1" / "stats"
    with make_session_factory(engine)() as s:
        repo, _ = _stage5_project(s)
        run_stage5(repo, "mc1", output_dir=stats_dir, generated_at=NOW)
        s.commit()
    (stats_dir / "plots" / "stray.svg").write_text("<svg/>")
    app = str(Path(__file__).resolve().parents[2] / "src" / "ui" / "app.py")

    def page() -> AppTest:
        at = AppTest.from_file(app, default_timeout=60)
        at.session_state["project_id"] = "mc1"
        at.session_state["page"] = "Statistics"
        return at.run()

    def plots_shown(at: AppTest) -> list[str]:
        return [m.value for m in at.markdown if str(m.value).startswith("**Forest plot:")]

    try:
        at = page()
        assert not at.exception
        assert plots_shown(at) == ["**Forest plot: comp**"]  # the stray file is not shown
        with make_session_factory(engine)() as s:  # statistics become stale (data changed)
            repo = Repository(s)
            with out_of_band(repo, "mc1"):
                arm = repo.load_extracted_arms("mc1")[0]
                repo.invalidate_arm("mc1", arm.arm_id, reviewer_id=repo.list_reviewers()[0].id,
                                    item_id=None, reason="synthetic change")  # fmt: skip
            s.commit()
        at = page()
        assert not at.exception and plots_shown(at) == []
        assert any("not shown" in w.value for w in at.warning)
    finally:
        engine.dispose()
        get_settings.cache_clear()
        st.cache_resource.clear()


# =========================================================================== M5
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0,500", 0.5),  # a leading 0 group is never a thousands group
        ("12,5", 12.5),
        ("1234,567", 1234.567),  # a 4-digit group cannot precede a thousands separator
        ("1,234,567", 1234567.0),
        ("1,234.5", 1234.5),
        ("1.234,5", 1234.5),  # thousands dot, decimal comma
        ("−0,25", -0.25),  # Unicode minus
        ("12·5", 12.5),  # Lancet mid-dot
        ("45,5%", 45.5),
        ("1 000", 1000.0),  # thin-space grouping
    ],
)
def test_m5_unambiguous_forms(text: str, expected: float) -> None:
    from src.core.schemas.extraction import parse_reported_number

    assert parse_reported_number(text) == expected


def test_m5_ambiguous_forms_need_a_recorded_interpretation() -> None:
    from src.core.schemas.extraction import AmbiguousNumberError, parse_reported_number

    with pytest.raises(AmbiguousNumberError) as info:
        parse_reported_number("1,250")
    assert info.value.readings == {"decimal_comma": 1.25, "thousands_comma": 1250.0}
    assert parse_reported_number("1,250", "thousands_comma") == 1250.0
    assert parse_reported_number("1,250", "decimal_comma") == 1.25
    with pytest.raises(ValueError, match="cannot be read as thousands_comma"):
        parse_reported_number("12,5", "thousands_comma")
    # the project convention reads "1.234" with a decimal point unless recorded otherwise
    assert parse_reported_number("1.234") == 1.234
    assert parse_reported_number("1.234", "thousands_point") == 1234.0


def test_m5_historical_values_are_flagged_not_reinterpreted(project: Project) -> None:
    from src.core.schemas.extraction import ExtractedStudyArm, ExtractedValue

    ev = make_evidence("Mean score 0,500; 1,250 patients")
    legacy = ExtractedValue(value=500.0, reported_text="0,500", evidence=ev)  # old reading
    assert legacy.value == 500.0 and legacy.format_status == "legacy_reading_differs"
    unread = ExtractedValue(value=1250.0, reported_text="1,250", evidence=ev)
    assert unread.format_status == "ambiguous_uninterpreted"
    with pytest.raises(ValueError, match="differs from reported_text"):
        ExtractedValue(value=1.25, reported_text="1,250", evidence=ev)  # no recorded reading
    read = ExtractedValue(value=1.25, reported_text="1,250", evidence=ev,
                          number_format="decimal_comma")  # fmt: skip
    assert read.format_status == "interpreted"
    # a stored arm with the old reading loads unchanged and blocks Gate 3 explicitly
    repo = project.repo
    item = repo.get_fulltext_item(PID, "r1")
    assert item is not None
    ev = ev.model_copy(update={"pdf_sha256": item.pdf_sha256, "pdf_filename": item.pdf_filename})
    arm = ExtractedStudyArm(
        record_id="r1", outcome_id="obliteration", arm_label="legacy", arm_role="single_group",
        mean=ExtractedValue(value=500.0, reported_text="0,500", evidence=ev),
    )  # fmt: skip
    repo.save_extracted_arm(PID, arm)
    (stored,) = [a for a in repo.load_extracted_arms(PID) if a.arm_label == "legacy"]
    assert stored.mean is not None and stored.mean.value == 500.0
    assert any("ambiguous printed format" in p for p in integrity(repo))


def test_m5_extraction_correction_review_and_arm(session: Session, tmp_path: Path) -> None:
    from src.extraction import correct_item
    from src.fulltext import DocumentStore
    from tests.fakes import approved_protocol, enroll, include_at_title_abstract
    from tests.fixtures.pdf_factory import Page, build_pdf
    from tests.unit.test_master_table import arm_value

    pdf = build_pdf([
        Page().text(50, 60, "SYNTHETIC TEST DOCUMENT - NOT A REAL STUDY")
        .text(50, 90, "Of 1,250 treated patients, 45 had complete obliteration."),
    ])  # fmt: skip
    repo = Repository(session)
    repo.create_project(PID, "Synthetic number formats")
    repo.save_protocol(PID, approved_protocol())
    from tests.conftest import make_record

    rec = make_record("n1", "Synthetic number-format report")
    repo.save_records(PID, [rec])
    include_at_title_abstract(repo, PID, [rec])
    store = DocumentStore(tmp_path)
    me = enroll(repo)
    ft = register_upload(repo, store, PID, "n1", pdf, original_filename="n1.pdf", reviewer=me)
    if ft.match_unconfirmed:
        confirm_pdf_match(repo, PID, "n1", reviewer=me, note="synthetic",
                          expected_pdf_sha256=ft.pdf_sha256)  # fmt: skip
    adjudicate(repo, PID, "n1", ScreeningStage.FULL_TEXT, verdict=ScreeningVerdict.INCLUDE,
               rationale="eligible", reviewer=me)  # fmt: skip
    repo.get_project(PID).phase = Phase.STAGE4_EXTRACTION_ROB.value
    quote = "Of 1,250 treated patients, 45 had complete obliteration."
    from src.core.schemas.protocol import FollowUpTimepoint

    common = {"page": 1, "timepoint": "last follow-up",
              "follow_up": FollowUpTimepoint(kind="last_follow_up")}  # fmt: skip
    doc = Project(repo, store, me).doc("n1")
    n, events = submit_proposals(
        repo, PID, "n1", doc,
        [arm_value("n_analyzed", "1,250", quote, **common),
         arm_value("events", "45", quote, **common)],
        proposed_by=ActorType.AGENT, proposer_identity="llm:test",
    )  # fmt: skip
    assert n.data_status == "ambiguous" and n.value is None
    assert any("ambiguous number format" in f for f in n.flags)
    shown = correction_state_sha256(repo, PID, n)
    fixed = correct_item(
        repo, PID, n.item_id,
        arm_value("n_analyzed", "1,250", quote, number_format="thousands_comma", **common),
        doc, reviewer=me, note="the paper groups thousands with commas (Table 1: 12,345 ...)",
        expected_state_sha256=shown,
    )  # fmt: skip
    assert fixed.value == 1250.0 and fixed.number_format == "thousands_comma"
    assert fixed.format_status == "interpreted" and fixed.reported_text == "1,250"
    assert any("read as thousands_comma" in f for f in fixed.flags)
    from src.extraction import accept_item

    accept_item(repo, PID, events.item_id, reviewer=me)
    (arm,) = build_arms(repo, PID, "n1", reviewer=me)
    (stored,) = repo.load_extracted_arms(PID)
    assert stored.n_analyzed is not None and stored.n_analyzed.value == 1250.0
    assert stored.n_analyzed.number_format == "thousands_comma"  # persisted with the arm
    assert stored.n_analyzed.reported_text == "1,250"  # verbatim source text preserved
    assert arm.arm_id == stored.arm_id


def test_m5_number_format_column_is_added_to_an_existing_database(tmp_path: Path) -> None:
    from sqlalchemy import inspect, text

    from src.state.db import init_db, make_engine

    engine = make_engine(f"sqlite:///{tmp_path / 'old.db'}")
    init_db(engine)
    with engine.begin() as conn:  # simulate a database created before Session 9
        conn.execute(text("ALTER TABLE extracted_values DROP COLUMN number_format"))

    def columns() -> set[str]:
        return {c["name"] for c in inspect(engine).get_columns("extracted_values")}

    assert "number_format" not in columns()
    init_db(engine)  # existing rows keep NULL: no recorded interpretation
    assert "number_format" in columns()
    engine.dispose()


def test_m5_api_lists_verbatim_text_and_interpreted_value(tmp_path: Path) -> None:
    from tests.unit.test_pr13_review_fixes import accepted_estimate, build, stage

    with TestClient(create_app("sqlite:///:memory:", data_dir=tmp_path)) as c:
        with c.app.state.session_factory() as s:  # type: ignore[attr-defined]
            p = build(s, tmp_path / "fulltext", {"r1": PDF})
            stage(p.repo, Phase.STAGE4_EXTRACTION_ROB)
            accepted_estimate(p, arms=False)
            s.commit()
        (item,) = c.get(f"/projects/{P13}/extraction/items").json()
        assert item["reported_text"] == "62.1" and item["value"] == 62.1
        assert item["format_status"] == "ok" and item["number_format"] is None
