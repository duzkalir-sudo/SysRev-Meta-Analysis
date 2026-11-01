"""Regression tests for the independent review of PR #11 (Session 6). All data SYNTHETIC.

1. numbers in biomedical formats (leading decimals, scientific notation, signed values,
   spelled-out counts, title) cannot bypass the audit, without false matches in identifiers;
2. unbound narrative assertions never pass silently: they require human review;
3. follow-up timepoints in the characteristics table are source-bound facts;
4. a PRISMA snapshot that contradicts the stored records blocks the audit;
5. included reports without DOI/PMID have an evidence-based human identity attestation;
6. the PRISMA study count equals the documented report-to-study links exactly;
7. an attestation is bound to the exact identity-confirmed PDF it relied on.
"""

from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from src.core.enums import (
    CheckStatus,
    OverlapResolution,
    OverlapSeverity,
    Phase,
    SourceDatabase,
    TimepointType,
)
from src.core.exceptions import SysRevError
from src.core.schemas.audit import REQUIRED_SCOPES, AuditReport
from src.core.schemas.cohort import CohortOverlapWarning, OverlapSignal
from src.core.schemas.manuscript import ReferenceVerification
from src.core.schemas.pipeline import PRISMACounts
from src.core.schemas.protocol import FollowUpTimepoint
from src.extraction import link_reports_to_study
from src.manuscript.evidence import build_evidence_package, prisma_consistency_problems
from src.manuscript.formatting import DisplayNumber, find_numbers, mask_identifiers
from src.manuscript.service import ManuscriptService, completion_problems
from src.search.prisma_sync import derive_prisma
from src.state.repository import Repository
from tests.manuscript_helpers import (
    REVIEWER,
    complete_manuscript,
    confirm_checklists,
    human,
    out_of_band,
    single_arm_project,
)
from tests.stats_helpers import LAST_FU, MONTHS_24, stats_arm
from tests.unit.test_manuscript_audit import Ctx, _ctx, blocking, findings

WINDOW = FollowUpTimepoint(kind=TimepointType.WINDOW, window_min=12, window_max=36, unit="months")


@pytest.fixture
def ms(session: Session) -> Ctx:
    return _ctx(session)


# --------------------------------------------------------------------------- 1. formats


@pytest.mark.parametrize(
    ("text", "values"),
    [
        ("p = .032 and p < .001", [0.032, 0.001]),
        ("p = 3.2e-5 or 3.2E−05", [3.2e-5, 3.2e-5]),
        ("4.1 × 10⁻³ and 2.5 x 10^-4", [4.1e-3, 2.5e-4]),
        ("MD −0.52 and -1.2; 1,234; 12·5", [-0.52, -1.2, 1234.0, 12.5]),
        ("range 3-5", [3.0, 5.0]),
    ],
)
def test_numeric_formats_are_parsed(text: str, values: list[float]) -> None:
    assert [f.value for f in find_numbers(text)] == pytest.approx(values)


@pytest.mark.parametrize(
    "text",
    [
        "ICD-10, COVID-19, SF-36, EQ-5D, IL-6, HbA1c, T2-weighted, BRAF V600E",
        "doi:10.5555/synthetic.12 and https://doi.org/10.5555/x.3",
        "PMID: 90000001, PMC1234567, NCT01234567, CRD42020123456",
        "as reported [3] and [1–3]",
    ],
)
def test_identifiers_and_terms_are_not_numbers(text: str) -> None:
    assert find_numbers(mask_identifiers(text)) == []


def test_scientific_notation_precision() -> None:
    (f,) = find_numbers("3.2e-5")
    assert f.decimals == 6  # tolerance 5e-7: 3.2e-5 is not a rounding of 3.3e-5
    assert DisplayNumber("p_value", 3.24e-5, 6).accepts_typed(f.value, f.decimals)
    assert not DisplayNumber("p_value", 3.3e-5, 6).accepts_typed(f.value, f.decimals)
    # a typed value never matches an integer count by rounding
    assert not DisplayNumber("count", 0.0, 0).accepts_typed(f.value, f.decimals)


def test_leading_decimal_p_value_mismatch(ms: Ctx) -> None:
    r = ms.edit("results", "{{stats.nec.heterogeneity}}", "Q = 1.34 (df = 5; p = .002); I² = 0%")
    assert any("p value .002" in m for m in findings(r, CheckStatus.MISMATCH)) and not r.passed


def test_scientific_notation_p_value_is_untraceable(ms: Ctx) -> None:
    r = ms.append("results", "The test of heterogeneity gave p = 3.2e-5.")
    assert any("3.2e-5" in m for m in findings(r, CheckStatus.UNSUPPORTED)) and not r.passed


def test_superscript_notation_is_checked(ms: Ctx) -> None:
    r = ms.append("results", "The pooled necrosis proportion was 7.1 × 10⁻² overall.")
    assert not r.passed and any("7.1 × 10⁻²" in m for m in blocking(r))


def test_negative_typed_value(mc_ctx: Ctx) -> None:
    r = mc_ctx.append("results", "The log risk ratio was −0.80 for complications.")
    assert not r.passed and any("−0.80" in m for m in blocking(r))


@pytest.fixture
def mc_ctx(session: Session) -> Ctx:
    return _ctx(session, comparative=True)


def test_spelled_out_total_is_checked(ms: Ctx) -> None:
    r = ms.append("results", "In total, seven studies were included.")
    assert any("7" in m or "seven" in m for m in findings(r, CheckStatus.MISMATCH))
    ok = ms.edit(
        "results", "In total, seven studies were included.", "In total, six studies were included."
    )
    assert not findings(ok, CheckStatus.MISMATCH, "literal_numbers")
    assert findings(ok, CheckStatus.REQUIRES_HUMAN_REVIEW, "literal_numbers")  # unbound


def test_subset_count_requires_review_not_silence(ms: Ctx) -> None:
    r = ms.append("discussion", "Two studies used frameless radiosurgery.")
    flagged = [f for f in r.findings if f.excerpt.startswith("Two studies")]
    assert any(
        f.scope == "literal_numbers" and f.status == CheckStatus.REQUIRES_HUMAN_REVIEW
        for f in flagged
    )
    r2 = ms.edit("discussion", "Two studies used", "Three studies used")
    assert any(
        "matches no approved total" in m
        for m in findings(r2, CheckStatus.REQUIRES_HUMAN_REVIEW, "literal_numbers")
    )


def test_number_in_title_is_audited(ms: Ctx) -> None:
    r = ms.edit("title", "{{protocol.title}}", "Radiation necrosis in 9 studies (synthetic)")
    assert any(
        f.location.startswith("title")
        and f.status in (CheckStatus.MISMATCH, CheckStatus.UNSUPPORTED)
        for f in r.findings
    )
    assert not r.passed


def test_technical_terms_do_not_create_false_findings(ms: Ctx) -> None:
    r = ms.append(
        "methods",
        "Toxicity was graded with CTCAE criteria; scans were "
        "T2-weighted MRI and comorbidity was coded with ICD-10.",
    )
    assert not findings(r, CheckStatus.UNSUPPORTED, "literal_numbers")
    assert not findings(r, CheckStatus.MISMATCH, "literal_numbers")


# --------------------------------------------------------------------------- 2. narrative


@pytest.mark.parametrize(
    ("section", "sentence"),
    [
        ("discussion", "Radiosurgery is safe in elderly patients with brain metastases."),
        ("introduction", "Brain metastases occur frequently in lung cancer."),
        ("conclusions", "Radiosurgery should replace whole-brain radiotherapy."),
        ("results", "Necrosis was rare across the included series."),
        ("abstract", "Radiosurgery is well tolerated."),
    ],
)
def test_uncited_assertion_requires_human_review(ms: Ctx, section: str, sentence: str) -> None:
    r = ms.append(section, sentence)
    flagged = [f for f in r.findings if f.excerpt == sentence and f.status != CheckStatus.VERIFIED]
    assert flagged, f"{sentence!r} passed without any finding"
    assert all(
        f.status in (CheckStatus.REQUIRES_HUMAN_REVIEW, CheckStatus.UNSUPPORTED) for f in flagged
    )
    # it is not certified: completion requires a human acknowledgement of it
    assert r.requires_human_review >= 1


def test_assertion_beside_verified_values_still_needs_review(ms: Ctx) -> None:
    sentence = "Necrosis was {{stats.nec.pooled_estimate}}, so radiosurgery is safe."
    r = ms.append("discussion", sentence)
    assert any(
        "wording around them" in m
        for m in findings(r, CheckStatus.REQUIRES_HUMAN_REVIEW, "narrative")
    )


def test_system_text_headings_and_pointers_are_not_flagged(ms: Ctx) -> None:
    r = ms.append(
        "results",
        "**Additional analyses.**\n\nCharacteristics of the included "
        "studies are shown in Table 1.\n\n(Figure 1)",
    )
    narrative = [f for f in r.findings if f.scope == "narrative"]
    assert not any("Table 1" in f.excerpt or "Figure 1" in f.excerpt for f in narrative)
    assert not any(f.excerpt.startswith("**Additional analyses.**") for f in narrative)
    # deterministic text derived from the evidence package is not flagged
    assert not any(f.location.startswith("methods") and "Synthesis" in f.excerpt for f in narrative)


def test_narrative_scope_is_required() -> None:
    assert "narrative" in REQUIRED_SCOPES
    report = AuditReport(project_id="p", manuscript_version=1, stats_output_sha256="0" * 64)
    assert not report.complete


# --------------------------------------------------------------------------- 3. follow-up


def test_follow_up_values_are_bound_and_verified(session: Session) -> None:
    repo, rid, _ = single_arm_project(
        session,
        extra_arms=[
            stats_arm("s1", "control_crude", follow_up=MONTHS_24, events="20", n_analyzed="25"),
            stats_arm("s2", "control_crude", follow_up=WINDOW, events="30", n_analyzed="40"),
            stats_arm("s3", "control_crude", follow_up=LAST_FU, events="12", n_analyzed="18"),
        ],
    )
    package = build_evidence_package(repo, "ms1")
    fu = {f.text: f for f in package.facts.values() if f.role == "follow_up"}
    assert {"24 months", "12 to 36 months"} <= set(fu)
    assert [n.value for n in fu["24 months"].numbers] == [24.0]
    assert [n.value for n in fu["12 to 36 months"].numbers] == [12.0, 36.0]
    last = [
        f
        for f in package.facts.values()
        if f.key.endswith(".follow_up") and f.text == "last follow-up"
    ]
    assert last and all(f.numbers == () for f in last)
    svc = ManuscriptService(repo, "ms1")
    complete_manuscript(svc, human(repo, rid))
    report = svc.audit()
    assert report.passed, blocking(report)
    table = [
        f
        for f in report.findings
        if f.location.startswith("tables") and f.status != CheckStatus.VERIFIED
    ]
    assert table == []  # no untraceable numbers, no review deadlock in the table
    assert any(
        c.stats_key
        and c.stats_key.endswith(".follow_up#follow_up")
        and c.status == CheckStatus.VERIFIED
        for c in report.numeric_checks
    )


# --------------------------------------------------------------------------- 4. PRISMA


def test_inconsistent_prisma_snapshot_blocks(ms: Ctx) -> None:
    wrong = PRISMACounts(  # internally consistent, but contradicts the stored records
        identified_by_database={SourceDatabase.PUBMED: 9},
        duplicates_removed=3,
        records_screened=6,
        reports_sought=6,
        reports_assessed=6,
        studies_included=6,
        reports_included=6,
    )
    assert wrong.invariant_violations() == []
    ms.repo.save_prisma(ms.pid, wrong)
    ms.svc.rerender(ms.actor)
    confirm_checklists(ms.svc.repo, ms.svc.project_id, ms.rid)
    r = ms.svc.audit()
    problems = findings(r, CheckStatus.MISMATCH, "provenance")
    assert any("identified_by_database" in m for m in problems)
    assert any("duplicates_removed" in m for m in problems)
    assert not r.passed
    # the stored values are reported as they are, never silently repaired
    assert build_evidence_package(ms.repo, ms.pid).facts["prisma.identified"].text == "9"


def test_consistent_prisma_has_no_problem(ms: Ctx) -> None:
    assert build_evidence_package(ms.repo, ms.pid).problems == []


# --------------------------------------------------------------------------- 5. identity


def _no_ids(session: Session) -> tuple[ManuscriptService, str]:
    repo, rid, _ = single_arm_project(
        session, record_overrides={"s1": {"pmid": None, "doi": None, "provenance": None}}
    )
    svc = ManuscriptService(repo, "ms1")
    complete_manuscript(svc, human(repo, rid))
    return svc, rid


def test_reference_without_identifiers_can_be_attested(session: Session) -> None:
    svc, rid = _no_ids(session)
    first = svc.audit()
    assert any("study-s1" in m for m in findings(first, CheckStatus.NOT_CHECKED, "citations"))
    svc.attest_reference(
        "study-s1",
        confirmed=True,
        basis="title, authors and year match the identity-confirmed PDF title page",
        reviewer_id=rid,
        reviewer_name=REVIEWER,
    )
    svc.rerender(human(svc.repo, rid))
    confirm_checklists(svc.repo, svc.project_id, rid)
    second = svc.audit()
    assert second.passed, blocking(second)
    assert any("attested by" in m for m in findings(second, CheckStatus.VERIFIED, "citations"))


def test_attestation_preconditions(session: Session) -> None:
    svc, rid = _no_ids(session)
    with pytest.raises(SysRevError, match="verify it live"):  # identifiers: live lookup only
        svc.attest_reference(
            "study-s2",
            confirmed=True,
            basis="looks right to me",
            reviewer_id=rid,
            reviewer_name=REVIEWER,
        )
    with pytest.raises(SysRevError):  # an unknown/automated identity cannot attest
        svc.attest_reference(
            "study-s1",
            confirmed=True,
            basis="automated attestation",
            reviewer_id="agent",
            reviewer_name="agent",
        )
    item = svc.repo.get_fulltext_item("ms1", "s1")
    assert item is not None
    with out_of_band(svc.repo, "ms1"):  # Session 9: refused by the application after Gate 3
        svc.repo.save_fulltext_item("ms1", item.model_copy(update={"match_status": None}))
    with pytest.raises(SysRevError, match="confirmed identity"):
        svc.attest_reference(
            "study-s1",
            confirmed=True,
            basis="compared with the PDF",
            reviewer_id=rid,
            reviewer_name=REVIEWER,
        )
    with pytest.raises(ValueError, match="basis"):
        ReferenceVerification(
            ref_key="k",
            metadata_sha256="0" * 64,
            source="human_attestation",
            status="verified",
            checked_by="x",
            reviewer_id="r",
        )


def test_attested_mismatch_blocks(session: Session) -> None:
    svc, rid = _no_ids(session)
    svc.attest_reference(
        "study-s1",
        confirmed=False,
        basis="the PDF title page names another first author",
        reviewer_id=rid,
        reviewer_name=REVIEWER,
    )
    svc.rerender(human(svc.repo, rid))
    confirm_checklists(svc.repo, svc.project_id, rid)
    r = svc.audit()
    assert findings(r, CheckStatus.MISMATCH, "citations") and not r.passed


# --------------------------------------------------------------------------- 6. study count
# (second review of PR #11: the exact study count, and PDF-bound attestations)


def _overlap(repo: Repository, rid: str, a: str, b: str) -> None:
    w = CohortOverlapWarning(
        record_id_a=a,
        record_id_b=b,
        overlap_score=0.8,
        severity=OverlapSeverity.HIGH,
        signals=[OverlapSignal(kind="coauthor_overlap", score=0.8, detail="synthetic")],
    )
    repo.save_overlap_warning("ms1", w)
    with out_of_band(repo, "ms1"):
        repo.resolve_overlap_warning(
            "ms1",
            w.warning_id,
            resolution=OverlapResolution.OVERLAPPING_DIFFERENT_OUTCOMES,
            reviewer_id=rid,
            note="overlapping recruitment at one centre",
        )


def _study_count_problems(repo: Repository, n: int) -> list[str]:
    snapshot = derive_prisma(repo, "ms1").model_copy(update={"studies_included": n})
    return prisma_consistency_problems(repo, "ms1", snapshot)


def test_overlapping_cohorts_are_not_one_study(session: Session) -> None:
    repo, rid, _ = single_arm_project(session)
    _overlap(repo, rid, "s1", "s2")
    _overlap(repo, rid, "s3", "s4")
    # six reports, no report-to-study link: six studies, whatever the participant overlap
    assert _study_count_problems(repo, 6) == []
    for wrong in (4, 5):  # the cohort count and an intermediate count are both refused
        problems = _study_count_problems(repo, wrong)
        assert any("exactly 6 distinct studies" in p for p in problems), problems


def test_multiple_reports_of_one_study_need_a_documented_link(session: Session) -> None:
    repo, rid, _ = single_arm_project(session)
    reviewer = repo.require_active_reviewer(rid)
    assert _study_count_problems(repo, 5)  # not yet documented
    with out_of_band(repo, "ms1"):
        link_reports_to_study(
            repo,
            "ms1",
            ["s1", "s2"],
            study_id="study-a",
            primary_record_id="s1",
            reviewer=reviewer,
            note="s2 is the long-term follow-up of s1",
        )
    assert _study_count_problems(repo, 5) == []  # exact, reproducible from the links
    assert _study_count_problems(repo, 6) and _study_count_problems(repo, 4)


def test_intermediate_study_count_blocks_the_audit(session: Session) -> None:
    repo, rid, _ = single_arm_project(
        session,
        study_links=[["s1", "s2"]],
        before_stats=lambda r, reviewer_id: _overlap(r, reviewer_id, "s3", "s4"),
    )
    # the documented count is 5; 4 would equal the number of distinct cohorts
    repo.save_prisma("ms1", derive_prisma(repo, "ms1").model_copy(update={"studies_included": 4}))
    svc = ManuscriptService(repo, "ms1")
    complete_manuscript(svc, human(repo, rid))
    report = svc.audit()
    assert not report.passed
    assert any(
        "exactly 5 distinct studies" in m
        for m in findings(report, CheckStatus.MISMATCH, "provenance")
    )
    repo.save_prisma("ms1", derive_prisma(repo, "ms1"))
    svc.rerender(human(repo, rid))
    confirm_checklists(svc.repo, svc.project_id, rid)
    assert svc.audit().passed


# --------------------------------------------------------------------------- 7. PDF-bound


def _attested(session: Session) -> tuple[ManuscriptService, str]:
    svc, rid = _no_ids(session)
    svc.attest_reference(
        "study-s1",
        confirmed=True,
        basis="title, authors and year match the identity-confirmed PDF title page",
        reviewer_id=rid,
        reviewer_name=REVIEWER,
    )
    svc.rerender(human(svc.repo, rid))
    confirm_checklists(svc.repo, svc.project_id, rid)
    return svc, rid


def _sign_off(svc: ManuscriptService, rid: str) -> None:
    report = svc.audit()
    assert report.passed, blocking(report)
    for f in report.findings:
        if f.needs_human:
            svc.acknowledge(
                report.report_id,
                f.finding_id,
                decision="confirmed_supported",
                note="checked against the sources",
                reviewer_id=rid,
                reviewer_name=REVIEWER,
            )
    svc.repo.get_project("ms1").phase = Phase.STAGE6_AUDIT.value
    svc.approve(reviewer_id=rid, reviewer_name=REVIEWER, statement="Approved after review.")
    assert completion_problems(svc.repo, "ms1") == []


def test_attestation_records_the_exact_pdf(session: Session) -> None:
    svc, _ = _attested(session)
    attestation = next(
        v for v in svc.repo.load_reference_verifications("ms1") if v.source == "human_attestation"
    )
    item = svc.repo.get_fulltext_item("ms1", "s1")
    assert item is not None and attestation.pdf_sha256 == item.pdf_sha256
    with pytest.raises(ValueError, match="names the PDF"):
        ReferenceVerification(
            ref_key="k",
            metadata_sha256="0" * 64,
            source="human_attestation",
            status="verified",
            message="compared with the PDF",
            checked_by="x",
            reviewer_id="r",
        )


def test_unchanged_pdf_evidence_remains_valid(session: Session) -> None:
    svc, rid = _attested(session)
    _sign_off(svc, rid)
    item = svc.repo.get_fulltext_item("ms1", "s1")
    assert item is not None
    with out_of_band(svc.repo, "ms1"):  # re-saving identical evidence changes nothing
        svc.repo.save_fulltext_item("ms1", item)
    assert completion_problems(svc.repo, "ms1") == []


def test_replaced_pdf_invalidates_attestation_audit_and_approval(session: Session) -> None:
    svc, rid = _attested(session)
    _sign_off(svc, rid)
    item = svc.repo.get_fulltext_item("ms1", "s1")
    assert item is not None
    replaced = item.model_copy(update={"pdf_sha256": "c" * 64, "pdf_filename": "s1_v2.pdf"})
    with pytest.raises(SysRevError, match="approved at Gate 3"):  # Session 9 (H3)
        svc.repo.save_fulltext_item("ms1", replaced)
    with out_of_band(svc.repo, "ms1"):  # a new PDF written outside the application
        svc.repo.save_fulltext_item("ms1", replaced)
    problems = completion_problems(svc.repo, "ms1")
    assert any("evidence changed" in p and "references_sha256" in p for p in problems)
    svc.rerender(human(svc.repo, rid))
    confirm_checklists(svc.repo, svc.project_id, rid)
    r = svc.audit()
    stale = findings(r, CheckStatus.NOT_CHECKED, "citations")
    assert any("no longer valid" in m and "attest again" in m for m in stale) and not r.passed
    # re-verification against the replacement PDF restores it
    with pytest.raises(SysRevError, match="current confirmed PDF"):
        svc.repo.save_reference_verification(
            "ms1",
            ReferenceVerification(
                ref_key="study-s1",
                metadata_sha256=svc.package().references["study-s1"].entry.metadata_sha256(),
                source="human_attestation",
                status="verified",
                message="relying on the old PDF",
                checked_by=REVIEWER,
                reviewer_id=rid,
                pdf_sha256=item.pdf_sha256,
            ),
        )
    svc.attest_reference(
        "study-s1",
        confirmed=True,
        basis="title page of the replacement PDF matches the stored metadata",
        reviewer_id=rid,
        reviewer_name=REVIEWER,
    )
    svc.rerender(human(svc.repo, rid))
    confirm_checklists(svc.repo, svc.project_id, rid)
    assert svc.audit().passed


def test_revoked_pdf_identity_invalidates_attestation(session: Session) -> None:
    svc, rid = _attested(session)
    _sign_off(svc, rid)
    item = svc.repo.get_fulltext_item("ms1", "s1")
    assert item is not None
    with out_of_band(svc.repo, "ms1"):  # Session 9: refused by the application after Gate 3
        svc.repo.save_fulltext_item("ms1", item.model_copy(update={"match_status": None}))
    assert any("evidence changed" in p for p in completion_problems(svc.repo, "ms1"))
    svc.rerender(human(svc.repo, rid))
    confirm_checklists(svc.repo, svc.project_id, rid)
    r = svc.audit()
    assert any(
        "confirmed PDF is now none" in m for m in findings(r, CheckStatus.NOT_CHECKED, "citations")
    )
    with pytest.raises(SysRevError, match="confirmed identity"):
        svc.attest_reference(
            "study-s1",
            confirmed=True,
            basis="compared with the PDF again",
            reviewer_id=rid,
            reviewer_name=REVIEWER,
        )
