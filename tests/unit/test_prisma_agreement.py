from __future__ import annotations

import pytest

from src.core.enums import ReviewerRole, ScreeningStage, ScreeningVerdict, SourceDatabase
from src.core.exceptions import PRISMAInconsistencyError
from src.core.schemas import ScreeningDecision
from src.screening.agreement import (
    agreement_for_stage,
    cohens_kappa,
    detect_conflicts,
    final_verdicts,
)
from src.screening.prisma import PRISMATracker
from tests.conftest import NOW

INC, EXC, UNC = ScreeningVerdict.INCLUDE, ScreeningVerdict.EXCLUDE, ScreeningVerdict.UNCERTAIN
TA = ScreeningStage.TITLE_ABSTRACT

# --------------------------------------------------------------------------- Cohen's kappa


def test_kappa_textbook_example() -> None:
    """2x2 table a=20 (yes/yes), b=5, c=10, d=15 (n=50).

    p_o = 35/50 = 0.70; rater A yes = 25/50, rater B yes = 30/50;
    p_e = 0.5*0.6 + 0.5*0.4 = 0.50; κ = (0.70-0.50)/(1-0.50) = 0.40.
    """
    a = ["y"] * 20 + ["y"] * 5 + ["n"] * 10 + ["n"] * 15
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    stats = cohens_kappa(a, b)
    assert stats.observed_agreement == pytest.approx(0.70)
    assert stats.expected_agreement == pytest.approx(0.50)
    assert stats.kappa == pytest.approx(0.40)


def test_kappa_perfect_and_chance_and_undefined() -> None:
    assert cohens_kappa(["y", "n", "y"], ["y", "n", "y"]).kappa == pytest.approx(1.0)
    assert cohens_kappa(["y", "n"], ["n", "y"]).kappa == pytest.approx(-1.0)
    undefined = cohens_kappa(["y", "y"], ["y", "y"])
    assert undefined.kappa is None and undefined.observed_agreement == 1.0


def test_kappa_three_categories() -> None:
    # p_o = 4/6; marginals A: i=2,e=3,u=1; B: i=3,e=3,u=0 -> p_e = (2*3+3*3+1*0)/36 = 15/36
    a = ["i", "i", "e", "e", "e", "u"]
    b = ["i", "i", "e", "e", "i", "e"]
    s = cohens_kappa(a, b)
    p_e = 15 / 36
    assert s.expected_agreement == pytest.approx(p_e)
    assert s.kappa == pytest.approx((4 / 6 - p_e) / (1 - p_e))


def test_kappa_input_validation() -> None:
    with pytest.raises(ValueError, match="same number"):
        cohens_kappa(["y"], ["y", "n"])
    with pytest.raises(ValueError, match="at least one"):
        cohens_kappa([], [])


# --------------------------------------------------------------------------- conflicts


def _d(rid: str, reviewer: ReviewerRole, verdict: ScreeningVerdict) -> ScreeningDecision:
    return ScreeningDecision(
        record_id=rid,
        reviewer=reviewer,
        reviewer_identity=f"agent-{reviewer.value}",
        stage=TA,
        verdict=verdict,
        exclusion_reason="wrong population" if verdict == EXC else None,
        rationale="r",
    )


R1, R2 = ReviewerRole.REVIEWER_1, ReviewerRole.REVIEWER_2


def _decisions() -> list[ScreeningDecision]:
    pairs = {"a": (INC, INC), "b": (EXC, EXC), "c": (INC, EXC), "d": (UNC, UNC), "e": (UNC, INC)}
    return [d for rid, (v1, v2) in pairs.items() for d in (_d(rid, R1, v1), _d(rid, R2, v2))]


def test_detect_conflicts() -> None:
    conflicts = detect_conflicts(_decisions(), TA)
    assert [c.record_id for c in conflicts] == ["c", "d", "e"]


def test_agreement_for_stage() -> None:
    stats = agreement_for_stage(_decisions(), TA)
    assert stats.n_items == 5
    assert stats.observed_agreement == pytest.approx(3 / 5)


def test_pairing_requires_both_reviewers_and_no_duplicates() -> None:
    with pytest.raises(ValueError, match="both reviewers"):
        detect_conflicts([_d("a", R1, INC)], TA)
    with pytest.raises(ValueError, match="duplicate"):
        detect_conflicts([_d("a", R1, INC), _d("a", R1, EXC), _d("a", R2, INC)], TA)


def test_final_verdicts_require_resolution() -> None:
    decisions = _decisions()
    conflicts = detect_conflicts(decisions, TA)
    with pytest.raises(ValueError, match="unresolved"):
        final_verdicts(decisions, conflicts, TA)
    resolved = [
        c.model_copy(update={"resolved_verdict": INC, "resolved_by": "Dr R", "resolved_at": NOW})
        for c in conflicts
    ]
    verdicts = final_verdicts(decisions, resolved, TA)
    assert verdicts == {"a": INC, "b": EXC, "c": INC, "d": INC, "e": INC}


# --------------------------------------------------------------------------- PRISMA tracker


def _populated_tracker() -> PRISMATracker:
    t = PRISMATracker()
    t.record_identified(SourceDatabase.PUBMED, 6)
    t.record_identified(SourceDatabase.OPENALEX, 4)
    t.record_duplicates_removed(3)
    t.record_automation_ineligible(1)
    for rid in ["r1", "r2", "r3", "r4"]:
        t.record_title_abstract_decision(rid, include=True)
    for rid in ["r5", "r6"]:
        t.record_title_abstract_decision(rid, include=False)
    t.record_retrieval("r1", retrieved=True)
    t.record_retrieval("r2", retrieved=True)
    t.record_retrieval("r3", retrieved=True)
    t.record_retrieval("r4", retrieved=False)
    t.record_fulltext_decision("r1", include=True)
    t.record_fulltext_decision("r2", include=True)
    t.record_fulltext_decision("r3", include=False, reason="wrong comparator")
    return t


def test_prisma_tracker_snapshot() -> None:
    t = _populated_tracker()
    s = t.snapshot()
    assert s.total_identified == 10
    assert s.records_screened == 6
    assert s.records_excluded == 2
    assert s.reports_sought == 4
    assert s.reports_not_retrieved == 1
    assert s.reports_assessed == 3
    assert s.reports_excluded_by_reason == {"wrong comparator": 1}
    assert s.reports_included == 2
    assert s.studies_included == 2
    assert t.pending() == {"awaiting_retrieval_decision": 0, "awaiting_fulltext_decision": 0}


def test_prisma_tracker_studies_vs_reports() -> None:
    t = _populated_tracker()
    t.set_studies_included(1)  # two reports of one cohort
    assert t.snapshot().studies_included == 1
    t.set_studies_included(3)
    with pytest.raises(PRISMAInconsistencyError, match="studies_included"):
        t.snapshot()


def test_prisma_tracker_detects_count_mismatch() -> None:
    t = _populated_tracker()
    t.record_identified(SourceDatabase.PUBMED, 1)  # an identified record never screened
    with pytest.raises(PRISMAInconsistencyError, match="records_screened"):
        t.snapshot()


def test_prisma_tracker_decisions_are_idempotent() -> None:
    t = _populated_tracker()
    t.record_title_abstract_decision("r5", include=False)
    t.record_fulltext_decision("r3", include=False, reason="wrong comparator")
    assert t.snapshot().records_screened == 6


def test_prisma_tracker_order_and_reason_rules() -> None:
    t = PRISMATracker()
    with pytest.raises(PRISMAInconsistencyError, match="not included"):
        t.record_retrieval("x", retrieved=True)
    t.record_title_abstract_decision("x", include=True)
    with pytest.raises(PRISMAInconsistencyError, match="no retrieved"):
        t.record_fulltext_decision("x", include=True)
    t.record_retrieval("x", retrieved=True)
    assert t.pending()["awaiting_fulltext_decision"] == 1
    with pytest.raises(ValueError, match="reason"):
        t.record_fulltext_decision("x", include=False)
    for method in (t.record_identified.__get__(t),):
        with pytest.raises(ValueError, match="non-negative"):
            method(SourceDatabase.PUBMED, -1)
    for fn in (
        t.record_register_hits,
        t.record_duplicates_removed,
        t.record_automation_ineligible,
        t.record_removed_other,
        t.set_studies_included,
    ):
        with pytest.raises(ValueError, match="non-negative"):
            fn(-1)


def test_prisma_snapshot_mid_review_reports_pending_work() -> None:
    t = PRISMATracker()
    t.record_identified(SourceDatabase.PUBMED, 3)
    for rid in ["a", "b", "c"]:
        t.record_title_abstract_decision(rid, include=True)
    t.record_retrieval("a", retrieved=True)
    s = t.snapshot()
    assert (s.reports_awaiting_retrieval, s.reports_awaiting_assessment) == (2, 1)
    assert not s.is_final
    t.record_retrieval("b", retrieved=False)
    t.record_retrieval("c", retrieved=True)
    t.record_fulltext_decision("a", include=True)
    t.record_fulltext_decision("c", include=False, reason="wrong outcome")
    final = t.snapshot()
    assert final.is_final
    assert (final.reports_assessed, final.reports_included) == (2, 1)


def test_prisma_redeciding_clears_downstream_state() -> None:
    t = _populated_tracker()
    t.record_title_abstract_decision("r1", include=False)  # was included and assessed
    s = t.snapshot()
    assert s.records_excluded == 3
    assert s.reports_included == 1
    t.record_retrieval("r2", retrieved=False)  # previously included report now unretrievable
    s = t.snapshot()
    assert s.reports_not_retrieved == 2 and s.reports_included == 0
    assert s.is_final
