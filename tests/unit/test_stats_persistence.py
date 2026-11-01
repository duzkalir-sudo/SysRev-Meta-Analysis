"""Stage 5 persistence and the Stage 5 → Stage 6 guard. All data are SYNTHETIC."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import update
from sqlalchemy.orm import Session

from src.core.enums import (
    FullTextStatus,
    Phase,
    ReviewDesign,
    ReviewerRole,
    ScreeningStage,
    ScreeningVerdict,
    StudyDesign,
)
from src.core.exceptions import StatisticsInputError, SysRevError
from src.core.schemas import FullTextItem, ScreeningDecision, StudyRecord
from src.core.schemas.protocol import FollowUpTimepoint
from src.core.schemas.stats import AnalysisResult, AnalysisSpec, StatsOutput, results_fingerprint
from src.state import repository as repository_module
from src.state.db import AppendOnlyViolationError, StatsOutputORM
from src.state.machine import PipelineStateMachine
from src.state.repository import Repository
from src.stats_engine.engine import run_plan
from src.stats_engine.stage5 import run_stage5, write_artifacts
from tests.conftest import NOW, make_protocol, make_record
from tests.fakes import include_at_title_abstract
from tests.manuscript_helpers import out_of_band
from tests.stats_helpers import LAST_FU, NECROSIS, OUTCOMES, REVIEWER, plan, stats_arm

PID = "p5"
NEC = AnalysisSpec(
    analysis_id="nec", outcome_id="necrosis", effect_measure="PROPORTION", timepoints=[LAST_FU]
)


@pytest.fixture
def repo(session: Session) -> Repository:
    r = Repository(session)
    r.create_project(PID, "Synthetic SRS review")
    r.save_protocol(
        PID,
        make_protocol(
            review_design=ReviewDesign.SINGLE_ARM,
            comparator=None,
            study_designs=[StudyDesign.CASE_SERIES],
            outcomes=[OUTCOMES["control_crude"], OUTCOMES["necrosis"]],
        ),
    )
    records = [make_record(rid, f"Synthetic case series {rid}") for rid, _, _ in NECROSIS]
    r.save_records(PID, records)
    include_at_full_text(r, records)
    for rid, x, n in NECROSIS:
        r.save_extracted_arm(
            PID, stats_arm(rid, "necrosis", follow_up=LAST_FU, events=x, n_analyzed=n)
        )
    return r


def include_at_full_text(repo: Repository, records: list[StudyRecord]) -> None:
    """Synthetic dual include decisions at both screening stages, with a retrieved PDF."""
    include_at_title_abstract(repo, PID, records)
    for rec in records:
        repo.save_fulltext_item(
            PID,
            FullTextItem(record_id=rec.record_id, title=rec.title,
                         status=FullTextStatus.OA_DOWNLOADED, pdf_filename=f"{rec.record_id}.pdf",
                         pdf_sha256="a" * 64),  # cited by make_evidence (Session 9, H3)
        )  # fmt: skip
        for role in (ReviewerRole.REVIEWER_1, ReviewerRole.REVIEWER_2):
            repo.save_screening_decision(
                PID,
                ScreeningDecision(record_id=rec.record_id, reviewer=role,
                                  reviewer_identity=f"agent-{role}", stage=ScreeningStage.FULL_TEXT,
                                  verdict=ScreeningVerdict.INCLUDE, rationale="synthetic include"),
            )  # fmt: skip


def _reviewer(repo: Repository) -> str:
    rid, _ = repo.enroll_reviewer(REVIEWER)
    return rid


def _to_stage5(repo: Repository) -> None:
    repo.get_project(PID).phase = Phase.STAGE5_META_ANALYSIS.value


def _record(repo: Repository, reviewer_id: str) -> str:
    approved = plan(NEC, reviewer_id=reviewer_id)
    repo.save_analysis_plan(PID, approved)
    output = run_plan(approved, repo.load_stage5_inputs(PID), project_id=PID, generated_at=NOW)
    return repo.record_stats_output(PID, output)


def test_plan_approval_requires_enrolled_reviewer_and_protocol_consistency(
    repo: Repository,
) -> None:
    with pytest.raises(SysRevError):
        repo.save_analysis_plan(PID, plan(NEC, reviewer_id="unknown"))
    rid = _reviewer(repo)
    with pytest.raises(SysRevError, match="does not match"):
        repo.save_analysis_plan(PID, plan(NEC, reviewer_id=rid, name="Someone Else"))
    bad = AnalysisSpec(
        analysis_id="b",
        outcome_id="necrosis",
        effect_measure="PROPORTION",
        timepoints=[FollowUpTimepoint(kind="fixed", value=3, unit="years")],
    )
    with pytest.raises(SysRevError, match="contradicts the protocol"):
        repo.save_analysis_plan(PID, plan(bad, reviewer_id=rid))
    draft = plan(NEC, approved=False)
    repo.save_analysis_plan(PID, draft)
    assert repo.latest_analysis_plan(PID) is None
    assert repo.latest_analysis_plan(PID, approved_only=False) == draft
    ok = plan(NEC, reviewer_id=rid)
    repo.save_analysis_plan(PID, ok)
    assert repo.latest_analysis_plan(PID) == ok
    with pytest.raises(SysRevError, match="append-only"):
        repo.save_analysis_plan(PID, ok)


def test_stats_recorded_only_in_stage5(repo: Repository) -> None:
    rid = _reviewer(repo)
    with pytest.raises(SysRevError, match="only be recorded in Stage 5"):
        _record(repo, rid)


def test_valid_artifact_opens_stage6(repo: Repository) -> None:
    rid = _reviewer(repo)
    _to_stage5(repo)
    sha = _record(repo, rid)
    ctx = repo.build_guard_context(PID)
    assert ctx.stats_output_present and ctx.stats_output_problems == ()
    m = PipelineStateMachine(PID, Phase.STAGE5_META_ANALYSIS)
    # The artifact itself is valid. Since Session 9 (H2) Stage 5 -> 6 also re-checks the Gate 3
    # conditions, which this minimal fixture does not meet (it sets the phase directly);
    # tests/unit/test_session9_remediation.py covers a complete Gate 3-valid project.
    violations = m.check(Phase.STAGE6_MANUSCRIPT, ctx)
    assert violations and all(v.startswith("Gate 3 condition no longer met") for v in violations)
    current = repo.current_stats_output(PID)
    assert current is not None and current.sha256() == sha
    assert current.analyses[0].status == "pooled"


def test_changed_data_after_stats_blocks_stage6(repo: Repository) -> None:
    rid = _reviewer(repo)
    _to_stage5(repo)
    _record(repo, rid)
    repo.save_records(PID, [make_record("s7", "Synthetic case series s7")])
    arm = stats_arm("s7", "necrosis", follow_up=LAST_FU, events="1", n_analyzed="9")
    with pytest.raises(SysRevError, match="approved at Gate 3"):  # Session 9 (H2)
        repo.save_extracted_arm(PID, arm)
    with out_of_band(repo, PID):  # a write outside the application is still detected
        repo.save_extracted_arm(PID, arm)
    ctx = repo.build_guard_context(PID)
    assert any("changed after the statistics" in p for p in ctx.stats_output_problems)
    violations = PipelineStateMachine(PID, Phase.STAGE5_META_ANALYSIS).check(
        Phase.STAGE6_MANUSCRIPT, ctx
    )
    assert any(v.startswith("stats_output.json:") for v in violations)


def test_new_plan_invalidates_old_stats(repo: Repository) -> None:
    rid = _reviewer(repo)
    _to_stage5(repo)
    _record(repo, rid)
    repo.save_analysis_plan(
        PID, plan(NEC.model_copy(update={"random_effects_ci": "wald"}), reviewer_id=rid)
    )
    assert any("latest approved plan" in p for p in repo.stats_output_problems(PID))


def test_tampered_artifact_detected_and_rows_append_only(
    repo: Repository, session: Session
) -> None:
    rid = _reviewer(repo)
    _to_stage5(repo)
    sha = _record(repo, rid)
    session.commit()
    row = session.query(StatsOutputORM).one()
    content = row.content
    row.content = content.replace('"pooled"', '"single_study"')
    with pytest.raises(AppendOnlyViolationError):
        session.flush()
    session.rollback()
    assert repo.stats_output_problems(PID) == []
    # a raw SQL edit bypasses the ORM listener but not the integrity check
    session.execute(
        update(StatsOutputORM)
        .where(StatsOutputORM.sha256 == sha)
        .values(content=content.replace('"pooled"', '"single_study"'))
    )
    session.expire_all()
    assert any("does not match its SHA-256" in p for p in repo.stats_output_problems(PID))


def test_foreign_project_output_rejected(repo: Repository) -> None:
    rid = _reviewer(repo)
    _to_stage5(repo)
    approved = plan(NEC, reviewer_id=rid)
    repo.save_analysis_plan(PID, approved)
    output = run_plan(approved, repo.load_stage5_inputs(PID), project_id="other", generated_at=NOW)
    with pytest.raises(SysRevError, match="another project"):
        repo.record_stats_output(PID, output)


def test_run_stage5_refusals(repo: Repository, tmp_path: Path) -> None:
    with pytest.raises(StatisticsInputError, match="cannot run in phase"):
        run_stage5(repo, PID, output_dir=tmp_path)
    _to_stage5(repo)
    with pytest.raises(StatisticsInputError, match="analysis plan"):
        run_stage5(repo, PID, output_dir=tmp_path)
    repo.save_analysis_plan(PID, plan(NEC, reviewer_id=_reviewer(repo)))
    # an arm for a report that was never included at full-text assessment
    repo.save_records(PID, [make_record("stray", "Synthetic excluded report")])
    with out_of_band(repo, PID):  # refused by the application after Gate 3 (Session 9)
        repo.save_extracted_arm(
            PID, stats_arm("stray", "necrosis", follow_up=LAST_FU, events="1", n_analyzed="9")
        )
    with pytest.raises(StatisticsInputError, match="not included at full text"):
        run_stage5(repo, PID, output_dir=tmp_path)


def test_artifacts_written_with_manifest(repo: Repository, tmp_path: Path) -> None:
    rid = _reviewer(repo)
    _to_stage5(repo)
    _record(repo, rid)
    output = repo.current_stats_output(PID)
    assert output is not None
    files = write_artifacts(output, tmp_path, render_plots=True)
    assert {
        "stats_output.json",
        "report.md",
        "tables/nec.md",
        "tables/nec.csv",
        "plots/nec.svg",
    } <= set(files)
    assert (tmp_path / "stats_output.json").read_text() == output.to_canonical_json()
    assert output.sha256() in (tmp_path / "manifest.json").read_text()


def _forged(output: StatsOutput, path: tuple[str, ...], delta: float) -> StatsOutput:
    """Alter one number and rebuild every hash so the artifact is self-consistent."""
    payload = output.model_dump(mode="json")
    target = payload["analyses"][0]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] += delta
    analyses = [AnalysisResult.model_validate(a) for a in payload["analyses"]]
    payload["results_sha256"] = results_fingerprint(
        analyses, output.plan_sha256, output.input_sha256
    )
    return StatsOutput.model_validate(payload)  # passes schema and fingerprint checks


@pytest.mark.parametrize(
    "path",
    [
        ("random_effects", "estimate"),
        ("primary", "ci_upper_natural"),
        ("studies", 0, "yi"),
        ("heterogeneity", "q"),
    ],
)
def test_numerically_altered_self_consistent_output_rejected(
    repo: Repository, path: tuple[object, ...]
) -> None:
    rid = _reviewer(repo)
    _to_stage5(repo)
    approved = plan(NEC, reviewer_id=rid)
    repo.save_analysis_plan(PID, approved)
    genuine = run_plan(approved, repo.load_stage5_inputs(PID), project_id=PID, generated_at=NOW)
    forged = _forged(genuine, path, 1e-9)  # type: ignore[arg-type]
    assert forged.results_sha256 != genuine.results_sha256
    with pytest.raises(SysRevError, match="deterministic recomputation") as err:
        repo.record_stats_output(PID, forged)
    assert str(path[0]) in str(err.value)
    assert repo.get_project(PID).stats_output_sha256 is None
    assert repo.record_stats_output(PID, genuine) == genuine.sha256()


def test_status_checks_do_not_recompute_statistics(
    repo: Repository, monkeypatch: pytest.MonkeyPatch
) -> None:
    rid = _reviewer(repo)
    _to_stage5(repo)
    _record(repo, rid)

    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("the engine must not run on a status request")

    monkeypatch.setattr(repository_module, "run_plan", boom)
    assert repo.build_guard_context(PID).stats_output_problems == ()
