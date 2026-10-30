"""Session 2: provider interface, dual LLM screening, conservative policy, adjudication.

Synthetic records only; scripted providers stand in for real models.
"""

from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from src.core.enums import Phase, ReviewerRole, ScreeningStage, ScreeningVerdict
from src.core.exceptions import SysRevError
from src.core.schemas import ScreeningDecision
from src.llm import (
    LLMConfig,
    LLMResponseError,
    ProviderNotConfiguredError,
    ProviderRegistry,
    parse_json_object,
)
from src.screening.llm_screening import (
    DualScreeningRunner,
    LLMScreener,
    LLMScreeningResponse,
    adjudicate,
    agreement_summary,
    apply_conservative_policy,
    criteria_catalogue,
    full_text_source,
    title_abstract_source,
)
from src.screening.reporting import (
    check_reviewer_claims,
    methods_statement,
    screening_reporting_facts,
)
from src.state.machine import PipelineStateMachine
from src.state.repository import Repository
from tests.conftest import make_record
from tests.fakes import ScriptedProvider, approved_protocol, config, enroll
from tests.fixtures.pdf_factory import synthetic_article

PID = "p1"
TA = ScreeningStage.TITLE_ABSTRACT
ABSTRACT = (
    "SYNTHETIC. We report 57 adults with cerebral arteriovenous malformations treated "
    "with stereotactic radiosurgery. This is a rat model study."
)


def include(**kw: object) -> dict[str, object]:
    return {"verdict": "include", "information_sufficient": True, "rationale": "fits"} | kw


def exclude(**kw: object) -> dict[str, object]:
    return {
        "verdict": "exclude",
        "met_exclusion_criteria": ["E1"],
        "supporting_quotes": ["This is a rat model study."],
        "information_sufficient": True,
        "rationale": "animal study",
    } | kw


@pytest.fixture
def repo(session: Session) -> Repository:
    r = Repository(session)
    r.create_project(PID, "Synthetic review")
    r.save_protocol(PID, approved_protocol())
    r.save_records(
        PID,
        [make_record(f"r{i}", f"Synthetic record {i}", abstract=ABSTRACT) for i in range(1, 5)],
    )
    return r


def runner(repo: Repository, a: object, b: object, model_b: str = "model-b") -> DualScreeningRunner:
    s1 = LLMScreener(ScriptedProvider("model-a", a), config("pass-1"), ReviewerRole.REVIEWER_1)  # type: ignore[arg-type]
    s2 = LLMScreener(
        ScriptedProvider(model_b, b),  # type: ignore[arg-type]
        config("pass-2", model_b),
        ReviewerRole.REVIEWER_2,
    )
    return DualScreeningRunner(repo, PID, approved_protocol(), [s1, s2])


def sources(repo: Repository) -> list[object]:
    return [title_abstract_source(r) for r in repo.load_records(PID)]


# --------------------------------------------------------------------------- provider


def test_registry_fails_explicitly_without_provider() -> None:
    reg = ProviderRegistry()
    with pytest.raises(ProviderNotConfiguredError, match="no LLM configuration"):
        reg.build(None)
    with pytest.raises(ProviderNotConfiguredError, match="not registered"):
        reg.build(LLMConfig(provider="anthropic", model_id="x", label="l"))
    reg.register("scripted", lambda c: ScriptedProvider(c.model_id, []))
    assert reg.build(config("l")).model_id == "model-a"
    reg.register("liar", lambda c: ScriptedProvider("other", []))
    with pytest.raises(ProviderNotConfiguredError, match="asked for"):
        reg.build(LLMConfig(provider="liar", model_id="x", label="l"))


def test_parse_json_object() -> None:
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    for bad in ("not json", "[1, 2]"):
        with pytest.raises(LLMResponseError):
            parse_json_object(bad)


# --------------------------------------------------------------------------- policy


def _policy(resp: dict[str, object]) -> tuple[ScreeningVerdict, str | None, tuple[str, ...]]:
    proto = approved_protocol(exclusion_criteria=["Animal studies"])
    src = title_abstract_source(make_record("r1", "T", abstract=ABSTRACT))
    out = apply_conservative_policy(
        LLMScreeningResponse.model_validate(resp), criteria_catalogue(proto), src
    )
    return out.verdict, out.exclusion_reason, out.notes


def test_supported_exclusion_is_linked_to_protocol_criterion() -> None:
    verdict, reason, _ = _policy(exclude())
    assert verdict == ScreeningVerdict.EXCLUDE
    assert reason == "E1: Animal studies"


@pytest.mark.parametrize(
    ("override", "why"),
    [
        ({"met_exclusion_criteria": ["E9"]}, "not linked to a predefined"),
        ({"information_sufficient": False}, "insufficient information"),
        ({"supporting_quotes": []}, "without a supporting quote"),
        ({"supporting_quotes": ["Patients were mice."]}, "absent from the source"),
    ],
)
def test_unsupported_exclusion_becomes_uncertain(override: dict[str, object], why: str) -> None:
    verdict, reason, notes = _policy(exclude(**override))
    assert verdict == ScreeningVerdict.UNCERTAIN
    assert reason is None
    assert any(why in n for n in notes)


def test_criteria_catalogue_ids() -> None:
    cat = criteria_catalogue(approved_protocol())
    assert list(cat) == ["I1", "E1"]


# --------------------------------------------------------------------------- dual screening


def test_two_independent_passes_disagreement_conflicts_and_kappa(repo: Repository) -> None:
    a = [include(), include(), exclude(), include()]
    b = [
        include(),
        exclude(),
        exclude(),
        {"verdict": "uncertain", "information_sufficient": False, "rationale": "?"},
    ]
    r = runner(repo, a, b)
    summary = r.run(sources(repo))
    assert summary.decided == 8 and summary.conflicts_created == 2
    decisions = repo.load_screening_decisions(PID)
    assert all(not d.is_human and d.automation is not None for d in decisions)
    assert {d.automation.model_id for d in decisions if d.automation} == {"model-a", "model-b"}
    # independence: neither prompt contains the other pass's output
    p1 = r.screeners[0].provider.requests  # type: ignore[attr-defined]
    assert all("pass-2" not in req.prompt for req in p1)
    stats = agreement_summary(repo, PID, TA)
    assert stats.paired_records == 4 and stats.unresolved_conflicts == 2
    assert stats.statistics is not None
    assert stats.statistics.observed_agreement == pytest.approx(0.5)
    conflicts = {c.record_id for c in repo.load_conflicts(PID)}
    assert conflicts == {"r2", "r4"}


def test_malformed_reply_becomes_uncertain_and_failures_resume(
    session: Session, repo: Repository
) -> None:
    a = ["{not json", include(), include(), include()]
    b = [include(), RuntimeError("network down"), include(), include()]
    r = runner(repo, a, b)
    first = r.run(sources(repo))
    assert len(first.failures) == 1 and first.decided == 7
    malformed = repo.find_screening_decision(PID, "r1", ReviewerRole.REVIEWER_1, TA)
    assert malformed is not None and malformed.verdict == ScreeningVerdict.UNCERTAIN
    session.commit()
    # "restart": a fresh runner only performs the missing decision
    r2 = runner(repo, [], [include()])
    second = r2.run(sources(repo))
    assert second.decided == 1 and second.resumed_skips == 7 and not second.failures


def test_same_model_twice_is_reported_honestly(repo: Repository) -> None:
    r = runner(repo, [include()] * 4, [include()] * 4, model_b="model-a")
    summary = r.run(sources(repo))
    assert summary.independence_note and "same model" in summary.independence_note
    facts = screening_reporting_facts(repo, PID, TA)
    assert facts.same_model_both_passes
    assert "two runs of the same model" in methods_statement(facts)


def test_runner_rejects_shared_provider_and_unapproved_protocol(repo: Repository) -> None:
    prov = ScriptedProvider("model-a", [])
    s1 = LLMScreener(prov, config("a"), ReviewerRole.REVIEWER_1)
    s2 = LLMScreener(prov, config("b"), ReviewerRole.REVIEWER_2)
    with pytest.raises(ValueError, match="separate provider"):
        DualScreeningRunner(repo, PID, approved_protocol(), [s1, s2])
    s2 = LLMScreener(ScriptedProvider("model-a", []), config("b"), ReviewerRole.REVIEWER_2)
    with pytest.raises(SysRevError, match="approved protocol"):
        DualScreeningRunner(
            repo, PID, approved_protocol().model_copy(update={"status": "draft"}), [s1, s2]
        )
    with pytest.raises(ValueError, match="adjudicator"):
        LLMScreener(prov, config("c"), ReviewerRole.ADJUDICATOR)


# --------------------------------------------------------------------------- adjudication


def test_adjudication_requires_enrolled_human_and_resolves_conflict(repo: Repository) -> None:
    runner(repo, [include(), exclude(), include(), include()], [include()] * 4).run(sources(repo))
    assert repo.find_conflict(PID, "r2", TA) is not None
    reviewer = enroll(repo)
    with pytest.raises(ValueError, match="include or exclude"):
        adjudicate(
            repo,
            PID,
            "r2",
            TA,
            verdict=ScreeningVerdict.UNCERTAIN,
            rationale="x",
            reviewer=reviewer,
        )
    adjudicate(
        repo,
        PID,
        "r2",
        TA,
        verdict=ScreeningVerdict.INCLUDE,
        rationale="eligible",
        reviewer=reviewer,
    )
    conflict = repo.find_conflict(PID, "r2", TA)
    assert conflict is not None and conflict.resolved_by == "Dr Human"
    with pytest.raises(SysRevError, match="already has a human adjudication"):
        adjudicate(
            repo,
            PID,
            "r2",
            TA,
            verdict=ScreeningVerdict.EXCLUDE,
            rationale="x",
            exclusion_reason="E1",
            reviewer=reviewer,
        )
    repo.revoke_reviewer(reviewer.id)
    with pytest.raises(SysRevError, match="revoked"):
        adjudicate(
            repo, PID, "r3", TA, verdict=ScreeningVerdict.INCLUDE, rationale="x", reviewer=reviewer
        )
    ctx = repo.build_guard_context(PID)
    assert ctx.ta_unresolved_conflicts == 0


def test_unauthenticated_human_decision_cannot_be_persisted(repo: Repository) -> None:
    forged = ScreeningDecision(
        record_id="r1",
        reviewer=ReviewerRole.ADJUDICATOR,
        reviewer_identity="Dr Nobody",
        is_human=True,
        stage=TA,
        verdict=ScreeningVerdict.INCLUDE,
        rationale="trust me",
    )
    with pytest.raises(SysRevError, match="authenticated"):
        repo.save_screening_decision(PID, forged)
    reviewer = enroll(repo, "Dr Real")
    impostor = forged.model_copy(update={"reviewer_id": reviewer.id})
    with pytest.raises(SysRevError, match="does not match"):
        repo.save_screening_decision(PID, impostor)


def test_full_text_llm_outcomes_need_human_confirmation(repo: Repository, tmp_path: object) -> None:
    from src.core.enums import FullTextStatus
    from src.core.schemas import FullTextItem
    from src.pdf_extractor import parse_pdf

    runner(repo, [include()] * 4, [include()] * 4).run(sources(repo))
    doc = parse_pdf(synthetic_article(), filename="a.pdf")
    repo.save_fulltext_item(
        PID,
        FullTextItem(
            record_id="r1",
            title="T",
            status=FullTextStatus.OA_DOWNLOADED,
            pdf_filename="a.pdf",
            pdf_sha256=doc.pdf_sha256,
        ),
    )
    src = full_text_source("r1", doc)
    assert "Kaplan-Meier" in src.text and "<<page" in src.context
    quote = "We retrospectively reviewed 57 patients"
    r = runner(repo, [include(supporting_quotes=[quote])], [include()])
    r.run([src])
    ctx = repo.build_guard_context(PID)
    assert ctx.ft_included == 1 and ctx.ft_awaiting_human_confirmation == 1
    v = PipelineStateMachine(PID, Phase.STAGE4_EXTRACTION_ROB).check(
        Phase.GATE3_EXTRACTION_APPROVAL, ctx
    )
    assert any("await confirmation" in x for x in v)
    adjudicate(
        repo,
        PID,
        "r1",
        ScreeningStage.FULL_TEXT,
        verdict=ScreeningVerdict.INCLUDE,
        rationale="confirmed",
        reviewer=enroll(repo),
    )
    assert repo.build_guard_context(PID).ft_awaiting_human_confirmation == 0


# --------------------------------------------------------------------------- reporting


def test_reporting_never_presents_llm_passes_as_two_humans(repo: Repository) -> None:
    runner(repo, [include(), exclude(), include(), include()], [include()] * 4).run(sources(repo))
    adjudicate(
        repo, PID, "r2", TA, verdict=ScreeningVerdict.INCLUDE, rationale="ok", reviewer=enroll(repo)
    )
    facts = screening_reporting_facts(repo, PID, TA)
    assert facts.human_independent_reviewers == 0
    assert facts.human_adjudicators == ("Dr Human",)
    text = methods_statement(facts)
    assert "not equivalent to independent human reviewers" in text
    assert "kappa" in text
    assert not check_reviewer_claims(text, facts)
    bad = "Two independent reviewers screened all titles and abstracts."
    assert check_reviewer_claims(bad, facts)
