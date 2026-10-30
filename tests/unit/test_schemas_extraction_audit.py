from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.core.enums import ActorType, CheckStatus, FullTextStatus, RoBTool, SourceDatabase
from src.core.schemas import (
    AuditReport,
    CitationCheck,
    DomainJudgement,
    ExtractedValue,
    FullTextItem,
    GateDecision,
    NumericClaimCheck,
    PRISMACounts,
    RiskOfBiasAssessment,
    parse_reported_number,
)
from src.core.schemas.extraction import ROB2_DOMAINS, ROBINS_I_DOMAINS
from tests.conftest import make_arm, make_evidence, make_value

# --------------------------------------------------------------------------- numbers


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("12.5", 12.5),
        ("12·5", 12.5),
        ("−3.2", -3.2),
        ("1,234", 1234.0),
        ("1,234.5", 1234.5),
        ("12,5", 12.5),
        ("45%", 45.0),
        ("1 000", 1000.0),
    ],
)
def test_parse_reported_number(text: str, expected: float) -> None:
    assert parse_reported_number(text) == expected


@pytest.mark.parametrize("text", ["abc", "1.2.3", "nan", "inf"])
def test_parse_reported_number_rejects(text: str) -> None:
    with pytest.raises(ValueError, match=r"."):
        parse_reported_number(text)


# --------------------------------------------------------------------------- SourceEvidence


def test_source_evidence_is_immutable() -> None:
    ev = make_evidence("mean 12.5")
    with pytest.raises(ValidationError):
        ev.page_number = 9  # type: ignore[misc]


@pytest.mark.parametrize(
    "override",
    [{"page_number": 0}, {"verbatim_quote": "   "}, {"pdf_filename": ""}, {"pdf_sha256": "xyz"}],
)
def test_source_evidence_validation(override: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        make_evidence("mean 12.5", **override)


def test_extracted_value_must_occur_in_quote() -> None:
    with pytest.raises(ValidationError, match="does not occur"):
        ExtractedValue(value=13.0, reported_text="13.0", evidence=make_evidence("mean 12.5"))


def test_extracted_value_must_equal_reported_text() -> None:
    """A computed value (e.g. a mean derived from a median) cannot masquerade as extracted."""
    with pytest.raises(ValidationError, match="derived values"):
        ExtractedValue(value=12.6, reported_text="12.5", evidence=make_evidence("mean 12.5"))


def test_extracted_value_whitespace_tolerant_match() -> None:
    v = ExtractedValue(
        value=1234, reported_text="1,234", evidence=make_evidence("total of\n  1,234 patients")
    )
    assert v.value == 1234


# --------------------------------------------------------------------------- ExtractedStudyArm


def test_valid_arm() -> None:
    arm = make_arm()
    assert set(arm.extracted_values()) == {"n_randomized", "n_analyzed", "mean", "sd"}


def test_arm_requires_outcome_data() -> None:
    arm = make_arm()
    with pytest.raises(ValidationError, match="outcome datum"):
        make_arm(mean=None, sd=None, n_randomized=arm.n_randomized)


def test_arm_counts_must_be_integers() -> None:
    with pytest.raises(ValidationError, match="integer"):
        make_arm(n_randomized=make_value("120.5", 120.5))


def test_arm_events_cannot_exceed_analysed() -> None:
    with pytest.raises(ValidationError, match="events cannot exceed"):
        make_arm(events=make_value("130", 130))


def test_arm_analysed_cannot_exceed_randomised() -> None:
    with pytest.raises(ValidationError, match="n_analyzed"):
        make_arm(n_analyzed=make_value("121", 121))


def test_arm_quantile_order() -> None:
    with pytest.raises(ValidationError, match="must not exceed"):
        make_arm(median=make_value("10", 10), q1=make_value("12", 12))


def test_arm_negative_sd() -> None:
    with pytest.raises(ValidationError, match="non-negative"):
        make_arm(sd=make_value("-1", -1))


def test_arm_approval_requires_identity() -> None:
    with pytest.raises(ValidationError, match="approved_by"):
        make_arm(approved_by_human=True)


# --------------------------------------------------------------------------- Risk of bias


def _rob2(judgements: list[str], overall: str | None) -> RiskOfBiasAssessment:
    return RiskOfBiasAssessment(
        record_id="r1",
        tool=RoBTool.ROB2,
        domains=[
            DomainJudgement(domain=d, judgement=j, support="justification")
            for d, j in zip(ROB2_DOMAINS, judgements, strict=True)
        ],
        overall=overall,
        assessor="Dr Reviewer",
        assessed_by=ActorType.HUMAN,
    )


def test_rob2_valid() -> None:
    assert _rob2(["low"] * 5, "low").overall == "low"
    assert _rob2(["low", "low", "some_concerns", "low", "low"], "some_concerns")


def test_rob2_overall_cannot_be_milder_than_worst_domain() -> None:
    with pytest.raises(ValidationError, match="less severe"):
        _rob2(["low", "high", "low", "low", "low"], "some_concerns")


def test_rob2_requires_all_domains_and_valid_levels() -> None:
    with pytest.raises(ValidationError, match="exactly the domains"):
        RiskOfBiasAssessment(
            record_id="r1", tool=RoBTool.ROB2, assessor="x", assessed_by=ActorType.AGENT
        )
    with pytest.raises(ValidationError, match="invalid"):
        _rob2(["low", "moderate", "low", "low", "low"], None)


def test_robins_i_no_information_rule() -> None:
    def make(judgements: list[str], overall: str) -> RiskOfBiasAssessment:
        return RiskOfBiasAssessment(
            record_id="r1",
            tool=RoBTool.ROBINS_I,
            domains=[
                DomainJudgement(domain=d, judgement=j, support="s")
                for d, j in zip(ROBINS_I_DOMAINS, judgements, strict=True)
            ],
            overall=overall,
            assessor="x",
            assessed_by=ActorType.HUMAN,
        )

    assert make(["low"] * 6 + ["no_information"], "no_information")
    assert make(["low"] * 6 + ["critical"], "critical")
    with pytest.raises(ValidationError):
        make(["low"] * 6 + ["serious"], "no_information")
    with pytest.raises(ValidationError, match="less severe"):
        make(["low"] * 6 + ["critical"], "serious")


def test_nos_star_limits() -> None:
    ok = RiskOfBiasAssessment(
        record_id="r1",
        tool=RoBTool.NOS,
        nos_stars={"selection": 4, "comparability": 2, "outcome_or_exposure": 3},
        assessor="x",
        assessed_by=ActorType.HUMAN,
    )
    assert ok.nos_stars is not None
    with pytest.raises(ValidationError, match="comparability"):
        RiskOfBiasAssessment(
            record_id="r1",
            tool=RoBTool.NOS,
            nos_stars={"selection": 4, "comparability": 3, "outcome_or_exposure": 3},
            assessor="x",
            assessed_by=ActorType.HUMAN,
        )


# --------------------------------------------------------------------------- Full text


def test_fulltext_item_rules() -> None:
    item = FullTextItem(record_id="r1", title="T", doi="10.5555/x")
    assert item.blocks_gate2
    with pytest.raises(ValidationError, match="pdf_filename"):
        FullTextItem(record_id="r1", title="T", status=FullTextStatus.USER_UPLOADED)
    with pytest.raises(ValidationError, match="unobtainable_reason"):
        FullTextItem(record_id="r1", title="T", status=FullTextStatus.UNOBTAINABLE)
    done = FullTextItem(
        record_id="r1",
        title="T",
        status=FullTextStatus.USER_UPLOADED,
        pdf_filename="r1.pdf",
        pdf_sha256="b" * 64,
    )
    assert not done.blocks_gate2


# --------------------------------------------------------------------------- Audit


def _numeric(
    status: CheckStatus, reported: float = 0.82, expected: float = 0.82
) -> NumericClaimCheck:
    return NumericClaimCheck(
        claim_text="RR 0.82",
        manuscript_location="Results ¶2",
        reported_value=reported,
        stats_key="primary.pooled_rr",
        expected_value=expected,
        tolerance=0.005,
        status=status,
    )


def test_numeric_check_status_must_match_values() -> None:
    assert _numeric(CheckStatus.VERIFIED)
    assert _numeric(CheckStatus.MISMATCH, reported=0.79)
    with pytest.raises(ValidationError, match="contradicts"):
        _numeric(CheckStatus.VERIFIED, reported=0.79)
    with pytest.raises(ValidationError, match="contradicts"):
        _numeric(CheckStatus.MISMATCH)


def test_citation_check_verified_needs_record() -> None:
    with pytest.raises(ValidationError, match="reference a record"):
        CitationCheck(
            citation_text="Doe 2020",
            manuscript_location="ref 1",
            doi="10.5555/x",
            status=CheckStatus.VERIFIED,
        )


def test_audit_report_passed_is_derived() -> None:
    cite = CitationCheck(
        citation_text="Doe 2020",
        manuscript_location="ref 1",
        doi="10.5555/x",
        matched_record_id="r1",
        status=CheckStatus.VERIFIED,
    )
    good = AuditReport(
        project_id="p1",
        manuscript_version=1,
        stats_output_sha256="c" * 64,
        numeric_checks=[_numeric(CheckStatus.VERIFIED)],
        citation_checks=[cite],
    )
    assert good.passed and good.failures == 0
    bad = good.model_copy(update={"numeric_checks": [_numeric(CheckStatus.MISMATCH, 0.7)]})
    assert not bad.passed
    empty = AuditReport(project_id="p1", manuscript_version=1, stats_output_sha256="c" * 64)
    assert not empty.passed  # nothing checked is not a pass

    # Round-trip works; a forged verdict is rejected.
    dumped = good.model_dump(mode="json")
    assert AuditReport.model_validate(dumped).passed
    forged = bad.model_dump(mode="json") | {"passed": True}
    with pytest.raises(ValidationError, match="contradicts"):
        AuditReport.model_validate(forged)


# --------------------------------------------------------------------------- Gate / PRISMA


def test_gate_rejection_requires_comments() -> None:
    with pytest.raises(ValidationError, match="comments"):
        GateDecision(gate="gate_1", verdict="rejected", reviewer_name="Dr R")
    assert GateDecision(gate="gate_1", verdict="approved", reviewer_name="Dr R")


def _prisma(**kw: object) -> PRISMACounts:
    data: dict[str, object] = {
        "identified_by_database": {SourceDatabase.PUBMED: 60, SourceDatabase.OPENALEX: 40},
        "duplicates_removed": 20,
        "records_screened": 80,
        "records_excluded": 60,
        "reports_sought": 20,
        "reports_not_retrieved": 2,
        "reports_assessed": 18,
        "reports_excluded_by_reason": {"wrong population": 5, "wrong design": 3},
        "reports_included": 10,
        "studies_included": 9,
    }
    data.update(kw)
    return PRISMACounts(**data)  # type: ignore[arg-type]


def test_prisma_valid_and_round_trip() -> None:
    p = _prisma()
    assert p.total_identified == 100
    assert PRISMACounts.model_validate(p.model_dump(mode="json")) == p


@pytest.mark.parametrize(
    ("override", "fragment"),
    [
        ({"records_screened": 81}, "records_screened"),
        ({"reports_sought": 21}, "reports_sought"),
        ({"reports_assessed": 17}, "reports_assessed"),
        ({"reports_included": 11}, "reports_included"),
        ({"studies_included": 11}, "studies_included"),
    ],
)
def test_prisma_invariants(override: dict[str, object], fragment: str) -> None:
    with pytest.raises(ValidationError, match=fragment):
        _prisma(**override)


def test_prisma_rejects_forged_total() -> None:
    with pytest.raises(ValidationError, match="total_identified"):
        PRISMACounts.model_validate(_prisma().model_dump(mode="json") | {"total_identified": 5})
