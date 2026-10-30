from __future__ import annotations

import pytest

from src.core.enums import OverlapSeverity
from src.core.schemas import RecruitmentPeriod
from src.screening.cohort_overlap import (
    CohortOverlapConfig,
    CohortOverlapDetector,
    institution_keys,
)
from tests.conftest import make_author, make_record

AFF_A = "Department of Cardiology, Fictional University Hospital, Exampleton, Testland."
AFF_B = "Institute of Imaginary Research, Sampletown 12345, Testland. author@example.org"
AFF_OTHER = "Division of Surgery, Placeholder Medical Centre, Otherville, Elsewhere."


def team(*names: tuple[str, str], aff: str = AFF_A) -> list:  # type: ignore[type-arg]
    return [make_author(last, fore, aff) for last, fore in names]


GROUP = [("Synthetica", "Ana"), ("Placebo", "Bruno"), ("Control", "Carla"), ("Blinded", "Dan")]


def test_institution_keys_strip_departments_cities_and_generic_words() -> None:
    assert institution_keys(AFF_A) == ["fictional"]
    assert institution_keys(AFF_B) == ["imaginary"]
    assert institution_keys("Dept of X, Mayo Clinic, Rochester, MN 55905, USA") == ["mayo"]
    assert institution_keys("Private practice, Exampleton") == []


def test_same_group_same_institution_overlapping_recruitment_is_high() -> None:
    a = make_record(
        "a",
        "Primary results",
        authors=team(*GROUP),
        recruitment_period=RecruitmentPeriod(start_year=2012, end_year=2016),
    )
    b = make_record(
        "b",
        "Five-year follow-up",
        authors=team(*GROUP[:3]),
        publication_year=2023,
        recruitment_period=RecruitmentPeriod(start_year=2013, end_year=2016),
    )
    [w] = CohortOverlapDetector().detect([a, b])
    assert w.severity == OverlapSeverity.HIGH
    assert w.overlap_score == pytest.approx(1.0)
    assert w.shared_authors == ["control c", "placebo b", "synthetica a"]
    assert w.shared_institutions == ["fictional"]
    assert w.recruitment_overlap_years == 4
    assert {s.kind for s in w.signals} == {
        "coauthor_overlap",
        "affiliation_similarity",
        "recruitment_overlap",
    }


def test_shared_registry_id_is_decisive() -> None:
    a = make_record(
        "a", "Trial report", registry_ids=["NCT01234567"], authors=team(("Alpha", "A"), aff=AFF_A)
    )
    b = make_record(
        "b",
        "Secondary analysis",
        registry_ids=["NCT01234567"],
        authors=team(("Omega", "Z"), aff=AFF_OTHER),
    )
    [w] = CohortOverlapDetector().detect([a, b])
    assert w.overlap_score == 1.0
    assert w.severity == OverlapSeverity.HIGH
    assert w.shared_registry_ids == ["NCT01234567"]


def test_unrelated_studies_not_flagged() -> None:
    a = make_record("a", "Trial one", authors=team(("Alpha", "A"), ("Beta", "B"), aff=AFF_A))
    b = make_record("b", "Trial two", authors=team(("Gamma", "G"), ("Delta", "D"), aff=AFF_OTHER))
    assert CohortOverlapDetector().detect([a, b]) == []


def test_same_institution_without_shared_authors_not_flagged() -> None:
    """Institutions alone (no shared authors, unknown recruitment) are too weak a signal."""
    a = make_record("a", "Trial one", authors=team(("Alpha", "A"), aff=AFF_A))
    b = make_record("b", "Trial two", authors=team(("Gamma", "G"), aff=AFF_A))
    assert CohortOverlapDetector().detect([a, b]) == []


def test_same_institution_with_overlapping_recruitment_flagged() -> None:
    period = RecruitmentPeriod(start_year=2015, end_year=2018)
    a = make_record("a", "Trial one", authors=team(("Alpha", "A")), recruitment_period=period)
    b = make_record("b", "Trial two", authors=team(("Gamma", "G")), recruitment_period=period)
    [w] = CohortOverlapDetector().detect([a, b])
    assert w.overlap_score == pytest.approx(0.6)
    assert w.severity == OverlapSeverity.MODERATE


def test_disjoint_recruitment_lowers_score() -> None:
    a = make_record(
        "a",
        "Cohort 1",
        authors=team(*GROUP),
        recruitment_period=RecruitmentPeriod(start_year=2005, end_year=2008),
    )
    b = make_record(
        "b",
        "Cohort 2",
        authors=team(*GROUP),
        recruitment_period=RecruitmentPeriod(start_year=2015, end_year=2018),
    )
    [w] = CohortOverlapDetector().detect([a, b])
    assert w.recruitment_overlap_years == 0
    assert w.overlap_score == pytest.approx(0.7)  # 0.4*1 + 0.3*1 + 0.3*0


def test_unknown_recruitment_uses_publication_proximity_at_half_weight() -> None:
    a = make_record("a", "Report 1", authors=team(*GROUP), publication_year=2020)
    b = make_record("b", "Report 2", authors=team(*GROUP), publication_year=2021)
    [w] = CohortOverlapDetector().detect([a, b])
    proximity = next(s for s in w.signals if s.kind == "publication_proximity")
    assert proximity.score == pytest.approx(0.8)
    assert w.overlap_score == pytest.approx(0.4 + 0.3 + 0.5 * 0.3 * 0.8)


def test_single_common_surname_in_large_teams_is_downweighted() -> None:
    big_a = [("Wang", "Li"), ("Alpha", "A"), ("Beta", "B"), ("Gamma", "G")]
    big_b = [("Wang", "Lei"), ("Delta", "D"), ("Eps", "E"), ("Zeta", "Z")]
    a = make_record("a", "T1", authors=team(*big_a, aff=AFF_A))
    b = make_record("b", "T2", authors=team(*big_b, aff=AFF_OTHER))
    det = CohortOverlapDetector()
    score, shared = det.coauthor_signal(a, b)
    assert shared == ["wang l"]
    assert score == pytest.approx(0.125)  # (1/4) * 0.5
    assert det.detect([a, b]) == []


def test_affiliation_fuzzy_matching() -> None:
    a = make_record("a", "T1", authors=team(("X", "Y"), aff="Fictional University Hospital, Town"))
    b = make_record("b", "T2", authors=team(("Q", "R"), aff="The Fictional University, Town"))
    c = make_record("c", "T3", authors=team(("Q", "R"), aff="Imaginary University, Town"))
    det = CohortOverlapDetector()
    assert det.affiliation_signal(a, b)[0] == 1.0
    assert det.affiliation_signal(a, c)[0] == 0.0


def test_detect_is_sorted_canonical_and_order_independent() -> None:
    recs = [
        make_record("c", "R3", authors=team(*GROUP)),
        make_record("a", "R1", authors=team(*GROUP)),
        make_record("b", "R2", authors=team(*GROUP[:2])),
        make_record("z", "Unrelated", authors=team(("Nobody", "N"), aff=AFF_OTHER)),
    ]
    det = CohortOverlapDetector()
    w1 = det.detect(recs)
    w2 = det.detect(list(reversed(recs)))
    key = [(w.record_id_a, w.record_id_b, w.overlap_score) for w in w1]
    assert key == [(w.record_id_a, w.record_id_b, w.overlap_score) for w in w2]
    assert all(w.record_id_a < w.record_id_b for w in w1)
    assert [w.overlap_score for w in w1] == sorted((w.overlap_score for w in w1), reverse=True)
    assert all("z" not in (w.record_id_a, w.record_id_b) for w in w1)


def test_detect_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        CohortOverlapDetector().detect([make_record("a", "T"), make_record("a", "T")])


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        CohortOverlapConfig(w_coauthor=0.9)
    with pytest.raises(ValueError, match="thresholds"):
        CohortOverlapConfig(flag_threshold=0.8, moderate_threshold=0.5)


def test_custom_threshold() -> None:
    a = make_record("a", "R1", authors=team(*GROUP), publication_year=2010)
    b = make_record(
        "b", "R2", authors=team(GROUP[0], ("New", "N"), aff=AFF_OTHER), publication_year=2010
    )
    strict = CohortOverlapDetector(
        CohortOverlapConfig(flag_threshold=0.9, moderate_threshold=0.95, high_threshold=0.99)
    )
    lenient = CohortOverlapDetector(CohortOverlapConfig(flag_threshold=0.2))
    assert strict.detect([a, b]) == []
    assert lenient.detect([a, b])
