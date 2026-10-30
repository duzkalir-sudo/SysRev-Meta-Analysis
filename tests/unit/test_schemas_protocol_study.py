from __future__ import annotations

from datetime import date, datetime

import pytest
from pydantic import ValidationError

from src.core.enums import ReviewerRole, ScreeningStage, ScreeningVerdict, SourceDatabase
from src.core.schemas import (
    Author,
    CohortOverlapWarning,
    DuplicateCandidate,
    MeSHTerm,
    OutcomeDefinition,
    OverlapSignal,
    RecruitmentPeriod,
    ScreeningConflict,
    ScreeningDecision,
    SearchStrategy,
    StudyRecord,
)
from tests.conftest import NOW, make_protocol, make_record

# --------------------------------------------------------------------------- PICO


def test_valid_protocol_has_no_blockers() -> None:
    protocol = make_protocol()
    assert protocol.approval_blockers() == []
    assert protocol.status == "draft"


def test_protocol_requires_primary_outcome() -> None:
    secondary = OutcomeDefinition(
        outcome_id="qol",
        name="Quality of life",
        kind="secondary",
        data_type="continuous",
        preferred_effect_measure="SMD",
    )
    with pytest.raises(ValidationError, match="primary outcome"):
        make_protocol(outcomes=[secondary])


def test_protocol_rejects_duplicate_outcome_ids() -> None:
    o = make_protocol().outcomes[0]
    with pytest.raises(ValidationError, match="unique"):
        make_protocol(outcomes=[o, o])


def test_protocol_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        make_protocol(hallucinated_field="x")


def test_protocol_requires_exclusion_criteria() -> None:
    with pytest.raises(ValidationError):
        make_protocol(exclusion_criteria=[])


def test_outcome_effect_measure_must_match_data_type() -> None:
    with pytest.raises(ValidationError, match="invalid for comparative dichotomous"):
        OutcomeDefinition(
            outcome_id="mort",
            name="Mortality",
            kind="primary",
            data_type="dichotomous",
            preferred_effect_measure="MD",
        )


def test_approval_blocked_by_unverified_mesh_and_unvalidated_search() -> None:
    protocol = make_protocol(
        mesh_terms=[MeSHTerm(descriptor="Invented Heading")],
        search_strategies=[SearchStrategy(database=SourceDatabase.PUBMED, query="x AND y")],
    )
    blockers = protocol.approval_blockers()
    assert any("MeSH" in b for b in blockers)
    assert any("syntax-validated" in b for b in blockers)
    with pytest.raises(ValidationError, match="cannot be approved"):
        protocol.model_validate(
            protocol.model_dump()
            | {"status": "approved", "approved_by": "Dr Reviewer", "approved_at": NOW}
        )


def test_approved_protocol_requires_approver() -> None:
    with pytest.raises(ValidationError, match="approved_by"):
        make_protocol(status="approved")
    ok = make_protocol(status="approved", approved_by="Dr Reviewer", approved_at=NOW)
    assert ok.status == "approved"


def test_prospero_id_format() -> None:
    assert make_protocol(prospero_id="CRD42024123456").prospero_id == "CRD42024123456"
    with pytest.raises(ValidationError):
        make_protocol(prospero_id="CRD123")


def test_verified_mesh_term_needs_timestamp() -> None:
    with pytest.raises(ValidationError, match="verified_at"):
        MeSHTerm(descriptor="Diabetes Mellitus", verified=True)


def test_mesh_descriptor_ui_pattern() -> None:
    assert MeSHTerm(descriptor="X", descriptor_ui="D003920").descriptor_ui == "D003920"
    with pytest.raises(ValidationError):
        MeSHTerm(descriptor="X", descriptor_ui="Q000000")


@pytest.mark.parametrize("query", ["(a AND b", 'a AND "b'])
def test_search_strategy_rejects_unbalanced_syntax(query: str) -> None:
    with pytest.raises(ValidationError, match="unbalanced"):
        SearchStrategy(database=SourceDatabase.PUBMED, query=query)


def test_search_strategy_date_order() -> None:
    with pytest.raises(ValidationError, match="date_from"):
        SearchStrategy(
            database=SourceDatabase.PUBMED,
            query="x",
            date_from=date(2024, 1, 1),
            date_to=date(2020, 1, 1),
        )


# --------------------------------------------------------------------------- StudyRecord


def test_study_record_normalises_identifiers() -> None:
    r = make_record("r1", "A Trial", doi="https://doi.org/10.5555/ABC", pmid="007", pmcid="99")
    assert (r.doi, r.pmid, r.pmcid) == ("10.5555/abc", "7", "PMC99")


def test_study_record_rejects_invalid_doi() -> None:
    with pytest.raises(ValidationError, match="Invalid DOI"):
        make_record("r1", "A Trial", doi="not-a-doi")


def test_study_record_rejects_self_duplicate_and_bad_year() -> None:
    with pytest.raises(ValidationError, match="itself"):
        make_record("r1", "A Trial", duplicate_of="r1")
    with pytest.raises(ValidationError):
        make_record("r1", "A Trial", publication_year=1700)


def test_study_record_requires_aware_timestamps() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        make_record("r1", "A Trial", retrieved_at=datetime(2024, 1, 1))


def test_study_record_validate_assignment() -> None:
    r = make_record("r1", "A Trial")
    with pytest.raises(ValidationError):
        r.doi = "garbage"


def test_author_requires_a_name_and_valid_orcid() -> None:
    with pytest.raises(ValidationError, match="requires"):
        Author(fore_name="Jane")
    with pytest.raises(ValidationError):
        Author(last_name="Smith", orcid="1234")
    assert Author(last_name="Smith", orcid="0000-0002-1825-009X").orcid


def test_author_match_key() -> None:
    assert Author(last_name="Smith", fore_name="Jane").match_key == "smith j"
    assert Author(collective_name="The STUDY Group").match_key == "the study group"
    assert Author(full_name="Jane Smith").match_key == "jane smith"


def test_completeness_score_orders_metadata_richness() -> None:
    sparse = make_record("a", "T", authors=[])
    rich = make_record("b", "T", pmid="1", doi="10.5555/x", abstract="abc")
    assert rich.completeness_score() > sparse.completeness_score()


def test_recruitment_period_order() -> None:
    with pytest.raises(ValidationError):
        RecruitmentPeriod(start_year=2020, end_year=2019)


def test_study_record_json_round_trip() -> None:
    r = make_record(
        "r1", "Trial", doi="10.5555/x", recruitment_period={"start_year": 2015, "end_year": 2017}
    )
    assert StudyRecord.model_validate(r.model_dump(mode="json")) == r


# --------------------------------------------------------------------------- Screening


def _decision(**kw: object) -> ScreeningDecision:
    data: dict[str, object] = {
        "record_id": "r1",
        "reviewer": ReviewerRole.REVIEWER_1,
        "reviewer_identity": "screener-agent-a",
        "stage": ScreeningStage.TITLE_ABSTRACT,
        "verdict": ScreeningVerdict.INCLUDE,
        "rationale": "Population and design match criteria 1-2.",
    }
    data.update(kw)
    return ScreeningDecision(**data)  # type: ignore[arg-type]


def test_exclusion_requires_reason() -> None:
    with pytest.raises(ValidationError, match="exclusion_reason"):
        _decision(verdict=ScreeningVerdict.EXCLUDE)
    assert _decision(verdict=ScreeningVerdict.EXCLUDE, exclusion_reason="wrong population")


def test_reason_only_for_exclusions() -> None:
    with pytest.raises(ValidationError, match="only allowed"):
        _decision(exclusion_reason="x")


def test_adjudicator_must_be_human() -> None:
    with pytest.raises(ValidationError, match="human"):
        _decision(reviewer=ReviewerRole.ADJUDICATOR)
    assert _decision(reviewer=ReviewerRole.ADJUDICATOR, is_human=True)


def test_human_full_text_decision_cannot_be_uncertain() -> None:
    with pytest.raises(ValidationError, match="full-text"):
        _decision(stage=ScreeningStage.FULL_TEXT, verdict=ScreeningVerdict.UNCERTAIN, is_human=True)
    # an automated pass may defer to a human rather than guess (conservative policy)
    assert _decision(stage=ScreeningStage.FULL_TEXT, verdict=ScreeningVerdict.UNCERTAIN)


def test_conflict_rules() -> None:
    with pytest.raises(ValidationError, match="not a conflict"):
        ScreeningConflict(
            record_id="r1",
            stage=ScreeningStage.TITLE_ABSTRACT,
            reviewer_1_verdict=ScreeningVerdict.INCLUDE,
            reviewer_2_verdict=ScreeningVerdict.INCLUDE,
        )
    c = ScreeningConflict(
        record_id="r1",
        stage=ScreeningStage.TITLE_ABSTRACT,
        reviewer_1_verdict=ScreeningVerdict.INCLUDE,
        reviewer_2_verdict=ScreeningVerdict.EXCLUDE,
    )
    assert not c.is_resolved
    with pytest.raises(ValidationError, match="together"):
        c.resolved_verdict = ScreeningVerdict.INCLUDE
    with pytest.raises(ValidationError, match="include or exclude"):
        ScreeningConflict.model_validate(
            c.model_dump()
            | {"resolved_verdict": "uncertain", "resolved_by": "Dr R", "resolved_at": NOW}
        )


# --------------------------------------------------------------------------- Cohort warnings


def test_cohort_warning_canonical_order_and_resolution() -> None:
    sig = [OverlapSignal(kind="coauthor_overlap", score=0.9, detail="x")]
    with pytest.raises(ValidationError, match="canonical"):
        CohortOverlapWarning(
            record_id_a="b", record_id_b="a", overlap_score=0.9, severity="high", signals=sig
        )
    with pytest.raises(ValidationError, match="itself"):
        CohortOverlapWarning(
            record_id_a="a", record_id_b="a", overlap_score=0.9, severity="high", signals=sig
        )
    with pytest.raises(ValidationError, match="resolved_by"):
        CohortOverlapWarning(
            record_id_a="a",
            record_id_b="b",
            overlap_score=0.9,
            severity="high",
            signals=sig,
            resolution="distinct_cohorts",
        )


def test_duplicate_candidate_rules() -> None:
    base = {"match_basis": "exact doi (10.5555/x)", "conflict": "different PMIDs"}
    with pytest.raises(ValidationError, match="canonical"):
        DuplicateCandidate(record_id_a="b", record_id_b="a", **base)
    with pytest.raises(ValidationError, match="itself"):
        DuplicateCandidate(record_id_a="a", record_id_b="a", **base)
    with pytest.raises(ValidationError, match="resolved_by"):
        DuplicateCandidate(record_id_a="a", record_id_b="b", resolution="distinct", **base)
    with pytest.raises(ValidationError, match="only allowed once resolved"):
        DuplicateCandidate(record_id_a="a", record_id_b="b", resolved_by="Dr R", **base)
    ok = DuplicateCandidate(
        record_id_a="a",
        record_id_b="b",
        resolution="distinct",
        resolved_by="Dr R",
        resolved_at=NOW,
        **base,
    )
    assert ok.is_resolved
