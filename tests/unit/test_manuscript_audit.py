"""Session 6: the independent Stage 6 Auditor against deliberately corrupted manuscripts.

Every study, number and reference is SYNTHETIC. The baseline is the deterministic template
draft of a synthetic single-arm review (or a comparative one) with the author-input markers
filled by neutral text; it passes the deterministic checks. Each test introduces one error
and asserts that the Auditor reports it with a blocking (or human-review) status.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from sqlalchemy.orm import Session

from src.core.enums import CheckStatus, Phase
from src.core.exceptions import GuardViolationError, SysRevError
from src.core.schemas.audit import BLOCKING_STATUSES, AuditReport, NumericClaimCheck
from src.core.schemas.manuscript import (
    EvidenceExcerpt,
    ManuscriptVersion,
    ReferenceEntry,
    ReferenceVerification,
)
from src.manuscript.evidence import build_evidence_package
from src.manuscript.formatting import DisplayNumber, find_numbers, fmt, fmt_p
from src.manuscript.service import Actor, ManuscriptService, completion_problems
from src.state.repository import Repository
from tests.conftest import NOW
from tests.manuscript_helpers import (
    REVIEWER,
    SYSTEM,
    comparative_project,
    complete_manuscript,
    human,
    single_arm_project,
)


@dataclass
class Ctx:
    repo: Repository
    svc: ManuscriptService
    actor: Actor
    rid: str
    pid: str

    def edit(self, section: str, old: str, new: str) -> AuditReport:
        latest = self.repo.latest_manuscript_version(self.pid)
        assert latest is not None
        sec = latest.section(section)
        assert sec is not None and old in sec.source, f"{old!r} not in {section}"
        self.svc.edit(self.actor, {section: sec.source.replace(old, new, 1)}, "test edit")
        return self.svc.audit()

    def append(self, section: str, text: str) -> AuditReport:
        latest = self.repo.latest_manuscript_version(self.pid)
        assert latest is not None
        sec = latest.section(section)
        assert sec is not None
        self.svc.edit(self.actor, {section: sec.source + "\n\n" + text}, "test edit")
        return self.svc.audit()


def _ctx(session: Session, comparative: bool = False) -> Ctx:
    repo, rid, _ = (comparative_project if comparative else single_arm_project)(session)
    pid = "mc1" if comparative else "ms1"
    svc = ManuscriptService(repo, pid)
    actor = human(repo, rid)
    complete_manuscript(svc, actor)
    return Ctx(repo, svc, actor, rid, pid)


@pytest.fixture
def ms(session: Session) -> Ctx:
    return _ctx(session)


@pytest.fixture
def mc(session: Session) -> Ctx:
    return _ctx(session, comparative=True)


def findings(report: AuditReport, status: CheckStatus, scope: str | None = None) -> list[str]:
    return [
        f.message
        for f in report.findings
        if f.status == status and (scope is None or f.scope == scope)
    ]


def blocking(report: AuditReport) -> list[str]:
    return [f"{f.scope}: {f.message}" for f in report.findings if f.status in BLOCKING_STATUSES]


# --------------------------------------------------------------------------- baseline


@pytest.mark.parametrize("comparative", [False, True])
def test_grounded_manuscript_passes_deterministic_checks(
    session: Session, comparative: bool
) -> None:
    ctx = _ctx(session, comparative)
    report = ctx.svc.audit()
    assert report.complete and report.passed, blocking(report)
    assert report.numeric_checks and all(
        c.status == CheckStatus.VERIFIED for c in report.numeric_checks
    )
    assert report.citation_checks and all(
        c.status == CheckStatus.VERIFIED for c in report.citation_checks
    )
    assert all(c.status == CheckStatus.VERIFIED for c in report.coverage)
    assert set(report.scopes_checked) >= {"numeric_bindings", "coverage", "citations"}
    assert report.limitations  # the audit states what it cannot verify
    # Passing the deterministic checks is not approval: a human sign-off is still required.
    assert "no human sign-off of this manuscript version and audit" in completion_problems(
        ctx.repo, ctx.pid
    )


def test_template_draft_without_author_input_is_blocked(session: Session) -> None:
    repo, _, _ = single_arm_project(session)
    svc = ManuscriptService(repo, "ms1")
    svc.create_template_draft(SYSTEM)
    report = svc.audit()
    missing = findings(report, CheckStatus.MISSING, "structure")
    assert not report.passed and any("author input required" in m for m in missing)
    # never invents a registration: it asks the author
    assert any("registration status" in m for m in missing)


def test_evidence_package_distinguishes_result_types(ms: Ctx) -> None:
    p = build_evidence_package(ms.repo, ms.pid)
    nec, km = p.facts["stats.nec.pooled_estimate"], p.facts["stats.km5.pooled_estimate"]
    assert (nec.estimator, km.estimator) == ("crude_proportion", "kaplan_meier")
    assert nec.pooled and nec.model == "random_effects" and nec.null_value is None
    assert p.facts["stats.nec.study.s1"].pooled is False
    # no data is a statement, never a zero or an estimate
    assert "stats.ctl.pooled_estimate" not in p.facts
    assert "no estimate was calculated" in p.facts["stats.ctl.status"].text
    # records / reports / studies are separate counts with units
    assert p.facts["prisma.identified"].count_unit == "records"
    assert p.facts["prisma.reports_included"].count_unit == "reports"
    assert p.facts["prisma.studies_included"].count_unit == "studies"
    # single-group p-values (tests of logit = 0) are not offered as findings
    assert "stats.nec.p_value" not in p.facts
    # every number displayed in a fact has its full-precision source value
    for f in p.facts.values():
        assert len(find_numbers(f.text)) == len(f.numbers), f.key


def test_display_tolerance_rules() -> None:
    dn = DisplayNumber("pooled_estimate", 5.94, 1)
    assert dn.accepts(5.9) and not dn.accepts(6.0) and not dn.accepts(5.8)
    assert DisplayNumber("count", 6.0, 0).accepts(6) and not DisplayNumber("count", 6.0, 0).accepts(
        7
    )
    text, p = fmt_p(0.0004)
    assert text == "p < 0.001" and p.accepts(0.001)
    # a p-value of 0.0015 may never be displayed as "p < 0.001"
    assert not DisplayNumber("p_value", 0.0015, 3, relation="lt").accepts(0.001)
    assert fmt_p(0.0123)[0] == "p = 0.012" and not fmt_p(0.0123)[1].accepts(0.013)
    assert fmt(-0.004, 2) == "0.00" and fmt(-1.5, 1) == "−1.5" and fmt(12345.0, 0) == "12,345"


# --------------------------------------------------------------------------- numbers


def test_incorrect_pooled_estimate(ms: Ctx) -> None:
    r = ms.edit("results", "{{stats.nec.pooled_estimate}}", "6.9% (95% CI 3.3% to 10.3%)")
    assert any("does not match" in m for m in findings(r, CheckStatus.MISMATCH, "literal_numbers"))
    assert any("main result for Radiation necrosis" in m for m in findings(r, CheckStatus.MISSING))
    assert not r.passed


def test_altered_confidence_interval(ms: Ctx) -> None:
    r = ms.edit("results", "{{stats.nec.pooled_estimate}}", "5.9% (95% CI 2.0% to 10.3%)")
    mism = findings(r, CheckStatus.MISMATCH, "literal_numbers")
    assert any("ci bound 2.0" in m for m in mism) and not r.passed
    # the correct estimate typed by hand is not verified either: it needs human review
    assert any("5.9" in m for m in findings(r, CheckStatus.REQUIRES_HUMAN_REVIEW))


def test_altered_heterogeneity(ms: Ctx) -> None:
    r = ms.edit("results", "{{stats.nec.heterogeneity}}", "Q = 1.34 (df = 5; p = 0.930); I² = 35%")
    assert any("i2 35" in m for m in findings(r, CheckStatus.MISMATCH)) and not r.passed


def test_altered_p_value_comparative(mc: Ctx) -> None:
    r = mc.edit("results", "{{stats.comp.p_value}}", "p = 0.04")
    assert any("p value 0.04" in m for m in findings(r, CheckStatus.MISMATCH)) and not r.passed


def test_wrong_study_count(ms: Ctx) -> None:
    r = ms.edit(
        "results", "{{prisma.studies_included}} studies, described in", "7 studies, described in"
    )
    assert any("7 studies" in m for m in findings(r, CheckStatus.MISMATCH)) and not r.passed


def test_inconsistent_percentage(ms: Ctx) -> None:
    r = ms.append(
        "results", "Necrosis occurred in 3 of 40 patients (9.5%) in one series [@study-s2]."
    )
    assert any("9.5% is not 3/40" in m for m in findings(r, CheckStatus.MISMATCH))


def test_wrong_follow_up_period(ms: Ctx) -> None:
    r = ms.append("discussion", "At 3 years, local control was {{stats.km5.pooled_estimate}}.")
    assert any("follow-up '3 year'" in m for m in findings(r, CheckStatus.MISMATCH))


def test_unavailable_result_reported_as_zero(ms: Ctx) -> None:
    r = ms.append("results", "Local tumour control (crude) was 0% in the included studies.")
    assert any("unavailable result is not zero" in m for m in findings(r, CheckStatus.MISMATCH))


def test_untraceable_number(ms: Ctx) -> None:
    r = ms.append("results", "The median lesion volume was 4.37 cm³.")
    assert any("not traceable" in m for m in findings(r, CheckStatus.UNSUPPORTED))


def test_tampered_table_value(ms: Ctx) -> None:
    """A table cell whose frozen value no longer equals the approved statistics output."""
    v = ms.repo.latest_manuscript_version(ms.pid)
    assert v is not None
    cell = next(
        b for b in v.bindings if b.section_id == "tables" and b.key == "stats.nec.pooled_estimate"
    )
    forged = cell.model_copy(update={"rendered_text": "6.9% (95% CI 3.3% to 10.3%)"})
    tampered = ManuscriptVersion.model_validate(
        v.model_dump(exclude={"content_sha256"})
        | {
            "version": v.version + 1,
            "parent_version": v.version,
            "bindings": [forged if b == cell else b for b in v.bindings],
            "change_summary": "tampered table",
        }
    )
    ms.repo.save_manuscript_version(tampered)
    r = ms.svc.audit()
    assert any(
        f.location.startswith("tables")
        and f.status == CheckStatus.MISMATCH
        and f.scope == "numeric_bindings"
        for f in r.findings
    )
    assert any(c.status == CheckStatus.MISMATCH for c in r.numeric_checks)
    assert findings(r, CheckStatus.MISMATCH, "provenance")


# --------------------------------------------------------------------------- labels


def test_kaplan_meier_mislabelled_as_crude(ms: Ctx) -> None:
    r = ms.append(
        "discussion", "The crude proportion of local control was {{stats.km5.pooled_estimate}}."
    )
    assert any(
        "kaplan meier estimate but is labelled as crude proportion" in m
        for m in findings(r, CheckStatus.MISMATCH, "semantic_labels")
    )


def test_crude_mislabelled_as_kaplan_meier(ms: Ctx) -> None:
    r = ms.append(
        "discussion", "The Kaplan-Meier estimate of necrosis was {{stats.nec.pooled_estimate}}."
    )
    assert findings(r, CheckStatus.MISMATCH, "semantic_labels")


def test_single_study_presented_as_pooled(ms: Ctx) -> None:
    r = ms.append("discussion", "The pooled estimate was {{stats.nec.study.s1}}.")
    assert any(
        "single-study result but is described as pooled" in m
        for m in findings(r, CheckStatus.MISMATCH)
    )


def test_model_mislabelled(ms: Ctx) -> None:
    r = ms.append("discussion", "The common-effect estimate was {{stats.nec.random_effects}}.")
    assert any("described as common-effect" in m for m in findings(r, CheckStatus.MISMATCH))


def test_significance_contradicting_interval(mc: Ctx) -> None:
    r = mc.append(
        "discussion",
        "There was no significant difference in complications ({{stats.comp.pooled_estimate}}).",
    )
    assert any("excludes the null" in m for m in findings(r, CheckStatus.MISMATCH))


def test_relative_effect_described_as_absolute(mc: Ctx) -> None:
    r = mc.append("discussion", "The absolute risk reduction was {{stats.comp.pooled_estimate}}.")
    assert any("relative effect" in m for m in findings(r, CheckStatus.MISMATCH))


def test_significance_claim_for_single_group_estimate(ms: Ctx) -> None:
    r = ms.append("discussion", "Necrosis was significant ({{stats.nec.pooled_estimate}}).")
    assert any("single-group" in m for m in findings(r, CheckStatus.UNSUPPORTED))


def test_clinical_importance_requires_review(mc: Ctx) -> None:
    r = mc.append(
        "discussion", "This reduction is clinically important ({{stats.comp.pooled_estimate}})."
    )
    assert any("clinical importance" in m for m in findings(r, CheckStatus.REQUIRES_HUMAN_REVIEW))


# --------------------------------------------------------------------------- methods


@pytest.mark.parametrize(
    ("sentence", "status", "fragment"),
    [
        (
            "Two reviewers independently screened all records.",
            CheckStatus.MISMATCH,
            "independent human reviewers",
        ),
        (
            "Data were extracted independently by two reviewers.",
            CheckStatus.MISMATCH,
            "human data extraction",
        ),
        (
            "The protocol was registered in PROSPERO (CRD42020123456).",
            CheckStatus.MISMATCH,
            "registration",
        ),
        ("The protocol was prospectively registered.", CheckStatus.UNSUPPORTED, "registration"),
        ("We also searched Embase and Google Scholar.", CheckStatus.UNSUPPORTED, "search log"),
        (
            "Publication bias was assessed with Egger's test.",
            CheckStatus.UNSUPPORTED,
            "publication bias test",
        ),
        ("Subgroup analyses by histology were performed.", CheckStatus.UNSUPPORTED, "subgroup"),
        (
            "Means were estimated from medians using Hozo's method.",
            CheckStatus.UNSUPPORTED,
            "mean sd conversion",
        ),
        ("Quality was assessed with the ROBINS-I tool.", CheckStatus.UNSUPPORTED, "ROBINS-I"),
        ("Certainty of evidence was rated with GRADE.", CheckStatus.UNSUPPORTED, "grade"),
    ],
)
def test_misrepresented_methods(ms: Ctx, sentence: str, status: CheckStatus, fragment: str) -> None:
    r = ms.append("methods", sentence)
    assert any(fragment in m for m in findings(r, status, "methods")), r.findings
    assert not r.passed


def test_negated_methods_are_not_flagged(ms: Ctx) -> None:
    r = ms.append(
        "methods",
        "Egger's test was not performed because too few studies were "
        "available, and no subgroup analyses were conducted.",
    )
    assert not findings(r, CheckStatus.UNSUPPORTED, "methods") and r.passed, blocking(r)


def test_missing_methodological_detail(ms: Ctx) -> None:
    r = ms.edit("methods", "{{screening.title_abstract}}", "")
    assert any("title_abstract screening procedure" in m for m in findings(r, CheckStatus.MISSING))


# --------------------------------------------------------------------------- claims


@pytest.mark.parametrize(
    ("sentence", "status"),
    [
        (
            "Radiosurgery causes radiation necrosis in susceptible patients.",
            CheckStatus.REQUIRES_HUMAN_REVIEW,
        ),
        (
            "Clinicians should offer radiosurgery to all such patients.",
            CheckStatus.REQUIRES_HUMAN_REVIEW,
        ),
        ("The underlying mechanism is endothelial injury.", CheckStatus.REQUIRES_HUMAN_REVIEW),
        ("This review proves that radiosurgery is safe.", CheckStatus.UNSUPPORTED),
        ("Radiosurgery is superior to surgery for these lesions.", CheckStatus.UNSUPPORTED),
    ],
)
def test_unsupported_clinical_claims(ms: Ctx, sentence: str, status: CheckStatus) -> None:
    r = ms.append("discussion", sentence)
    assert findings(r, status, "claims")


def test_abstract_result_absent_from_results(ms: Ctx) -> None:
    r = ms.append("abstract", "The common-effect estimate was {{stats.nec.common_effect}}.")
    assert any("not in the Results" in m for m in findings(r, CheckStatus.MISSING, "consistency"))


# --------------------------------------------------------------------------- citations


def _import_reference(ctx: Ctx, pmid: str = "91111111") -> str:
    name = REVIEWER
    ctx.repo.save_reference_entry(
        ctx.pid,
        ReferenceEntry(
            ref_key=f"pmid-{pmid}",
            origin="imported_literature",
            pmid=pmid,
            title="Synthetic cohort study of necrosis after radiosurgery",
            authors=["Hazel C"],
            journal="Synthetic Neuro-Oncology",
            year=2019,
            metadata_source="pubmed_efetch",
            retrieved_at=NOW,
            added_by=name,
            reviewer_id=ctx.rid,
        ),
    )
    ctx.repo.save_evidence_excerpt(
        ctx.pid,
        EvidenceExcerpt(
            ref_key=f"pmid-{pmid}",
            text="In this cohort, the risk of radiation necrosis increased "
            "with larger lesion volume and with higher prescription dose.",
            location="Results, p 3",
            added_by=name,
            reviewer_id=ctx.rid,
        ),
    )
    return f"pmid-{pmid}"


def test_invented_reference_and_typed_numbers(ms: Ctx) -> None:
    r = ms.append("introduction", "Necrosis is common [@pmid-99999999]. It is rare [7].")
    assert findings(r, CheckStatus.NOT_FOUND, "citations")
    assert any("typed citation number" in m for m in findings(r, CheckStatus.UNSUPPORTED))


def test_author_attribution_without_registry_reference(ms: Ctx) -> None:
    r = ms.append("discussion", "Smith et al. reported similar findings.")
    assert any("named authors" in m for m in findings(r, CheckStatus.UNSUPPORTED, "citations"))


def test_existing_reference_that_does_not_support_claim(ms: Ctx) -> None:
    key = _import_reference(ms)
    r = ms.append("introduction", f"Whole-brain treatment in children doubles mortality [@{key}].")
    assert any(
        "do not appear to support" in m
        for m in findings(r, CheckStatus.UNSUPPORTED, "claim_support")
    )
    # existence was never verified live: not checked, blocking
    assert findings(r, CheckStatus.NOT_CHECKED, "citations")


def test_consistent_excerpt_still_needs_human_and_identity_check(ms: Ctx) -> None:
    key = _import_reference(ms)
    r = ms.append(
        "introduction",
        f"Radiation necrosis risk increased with larger lesion volume "
        f"and higher prescription dose [@{key}].",
    )
    assert any(
        "semantic support needs human confirmation" in m
        for m in findings(r, CheckStatus.REQUIRES_HUMAN_REVIEW, "claim_support")
    )
    assert not r.passed  # identity not verified
    entry = next(e for e in ms.repo.load_reference_entries(ms.pid) if e.ref_key == key)
    for status in ("unavailable", "verified"):
        ms.repo.save_reference_verification(
            ms.pid,
            ReferenceVerification(
                ref_key=key,
                metadata_sha256=entry.metadata_sha256(),
                source="pubmed",
                status=status,
                checked_by="test",  # type: ignore[arg-type]
            ),
        )
    # the evidence changed (verification recorded): the previous audit is stale
    assert any("evidence changed" in p for p in completion_problems(ms.repo, ms.pid))
    ms.svc.rerender(ms.actor)
    r2 = ms.svc.audit()
    assert r2.passed, blocking(r2)
    assert r2.requires_human_review >= 1


def test_wrong_number_attributed_to_reference(ms: Ctx) -> None:
    key = _import_reference(ms)
    r = ms.append(
        "introduction",
        f"Radiation necrosis risk increased by 40% with larger lesion volume [@{key}].",
    )
    assert any(
        "do not appear in the recorded excerpt" in m
        for m in findings(r, CheckStatus.UNSUPPORTED, "claim_support")
    )


def test_misattributed_study_result(ms: Ctx) -> None:
    r = ms.append("results", "Birch 2016 reported {{stats.nec.study.s2}} [@study-s1].")
    assert any("different study" in m for m in findings(r, CheckStatus.MISMATCH, "claim_support"))
    ok = ms.edit("results", "[@study-s1].", "[@study-s2].")
    assert any("come from the cited report" in m for m in findings(ok, CheckStatus.VERIFIED))


def test_duplicate_reference(ms: Ctx) -> None:
    ms.repo.save_reference_entry(
        ms.pid,
        ReferenceEntry(
            ref_key="dup-1",
            origin="imported_literature",
            pmid="90000001",
            title="Duplicate",
            metadata_source="pubmed_efetch",
            retrieved_at=NOW,
            added_by=REVIEWER,
            reviewer_id=ms.rid,
        ),
    )
    r = ms.append("discussion", "As described previously [@dup-1].")
    assert any("duplicate reference" in m for m in findings(r, CheckStatus.MISMATCH, "citations"))


def test_uncited_included_study(ms: Ctx) -> None:
    r = ms.edit("results", "[@study-s1; ", "[")
    r = ms.edit("tables", "{{table:study_characteristics}}", "(table removed)")
    assert any("never cited" in m for m in findings(r, CheckStatus.MISSING, "coverage"))


# --------------------------------------------------------------------------- versions/audits


def test_partial_and_tampered_audits_are_refused(ms: Ctx) -> None:
    partial = AuditReport(
        project_id=ms.pid,
        manuscript_version=1,
        stats_output_sha256="0" * 64,
        numeric_checks=[
            NumericClaimCheck(
                claim_text="5.9%",
                manuscript_location="Results",
                reported_value=5.9,
                stats_key="k",
                expected_value=5.9,
                status=CheckStatus.VERIFIED,
            )
        ],
    )
    assert not partial.passed
    with pytest.raises(SysRevError, match="incomplete"):
        ms.repo.save_audit_report(partial)
    real = ms.edit("results", "{{stats.nec.pooled_estimate}}", "6.9% (95% CI 3.3% to 10.3%)")
    assert not real.passed
    cleaned = real.model_copy(
        update={
            "report_id": "forged",
            "findings": [f for f in real.findings if not f.blocking],
            "coverage": [
                c.model_copy(update={"status": CheckStatus.VERIFIED}) for c in real.coverage
            ],
            "numeric_checks": [],
            "citation_checks": [],
        }
    )
    assert cleaned.passed  # it *looks* like a pass ...
    with pytest.raises(SysRevError, match="recomputation"):
        ms.repo.save_audit_report(cleaned)  # ... but it is not what the Auditor computes


def test_stale_audits_never_authorise_newer_versions(ms: Ctx) -> None:
    report = ms.svc.audit()
    assert report.passed
    ms.svc.rerender(ms.actor)  # new version: identical text, but a different version
    problems = completion_problems(ms.repo, ms.pid)
    assert any("examined version" in p for p in problems)


def test_changed_extraction_data_invalidates(ms: Ctx) -> None:
    from tests.stats_helpers import LAST_FU, stats_arm

    ms.svc.audit()
    ms.repo.save_extracted_arm(
        ms.pid, stats_arm("s1", "control_crude", follow_up=LAST_FU, events="20", n_analyzed="25")
    )
    problems = completion_problems(ms.repo, ms.pid)
    assert any("evidence package unavailable" in p for p in problems)
    with pytest.raises(SysRevError, match="re-run"):
        ms.svc.audit()


def test_acknowledgement_and_sign_off_rules(ms: Ctx) -> None:
    bad = ms.append(
        "discussion",
        "Radiosurgery causes necrosis. The pooled estimate was {{stats.nec.study.s1}}.",
    )
    block = next(f for f in bad.findings if f.blocking)
    with pytest.raises(SysRevError, match="must be fixed"):
        ms.svc.acknowledge(
            bad.report_id,
            block.finding_id,
            decision="confirmed_supported",
            note="trying to wave it through",
            reviewer_id=ms.rid,
            reviewer_name=REVIEWER,
        )
    ok = ms.edit("discussion", " The pooled estimate was {{stats.nec.study.s1}}.", "")
    assert ok.passed and ok.requires_human_review == 1
    project = ms.repo.get_project(ms.pid)
    with pytest.raises(SysRevError, match="audit phase"):
        ms.svc.approve(reviewer_id=ms.rid, reviewer_name=REVIEWER, statement="approve it now")
    project.phase = Phase.STAGE6_AUDIT.value
    with pytest.raises(SysRevError, match="not acknowledged"):
        ms.svc.approve(reviewer_id=ms.rid, reviewer_name=REVIEWER, statement="approve it now")
    review = next(f for f in ok.findings if f.needs_human)
    with pytest.raises(SysRevError):  # an unknown (e.g. automated) identity cannot act
        ms.svc.acknowledge(
            ok.report_id,
            review.finding_id,
            decision="confirmed_supported",
            note="automated agent approval",
            reviewer_id="agent",
            reviewer_name="agent",
        )
    ms.svc.acknowledge(
        ok.report_id,
        review.finding_id,
        decision="accepted_as_author_statement",
        note="causal wording reviewed: supported by RCT evidence cited",
        reviewer_id=ms.rid,
        reviewer_name=REVIEWER,
    )
    with pytest.raises(SysRevError):
        ms.svc.approve(reviewer_id="agent", reviewer_name="agent", statement="self-approval")
    ms.svc.approve(reviewer_id=ms.rid, reviewer_name=REVIEWER, statement="Approved version.")
    assert completion_problems(ms.repo, ms.pid) == []
    machine = ms.repo.load_state_machine(ms.pid)
    ctx = ms.repo.build_guard_context(ms.pid)
    assert machine.check(Phase.COMPLETED, ctx) == []
    # any later edit invalidates the audit and the sign-off
    ms.svc.edit(ms.actor, {"conclusions": "Revised conclusions without numbers."}, "edit")
    ctx = ms.repo.build_guard_context(ms.pid)
    with pytest.raises(GuardViolationError):
        machine.transition(Phase.COMPLETED, actor="x", actor_type=ms.actor.actor_type, context=ctx)


def test_versions_are_append_only_and_need_stage6(ms: Ctx) -> None:
    from sqlalchemy import select

    from src.state.db import AppendOnlyViolationError, ManuscriptVersionORM

    row = ms.repo.session.scalars(select(ManuscriptVersionORM)).first()
    assert row is not None
    row.change_summary = "x"  # type: ignore[attr-defined]
    row.payload = {}
    with pytest.raises(AppendOnlyViolationError):
        ms.repo.session.flush()
    ms.repo.session.rollback()


def test_versions_refused_outside_stage6(session: Session) -> None:
    repo, _, _ = single_arm_project(session)
    svc = ManuscriptService(repo, "ms1")
    repo.get_project("ms1").phase = Phase.STAGE5_META_ANALYSIS.value
    with pytest.raises(SysRevError, match="not allowed in phase"):
        svc.create_template_draft(SYSTEM)
