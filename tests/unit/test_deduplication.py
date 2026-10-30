from __future__ import annotations

import itertools
import random

import pytest

from src.core.enums import SourceDatabase
from src.screening.deduplication import Deduplicator, deduplicate
from tests.conftest import make_author, make_record

TITLE = "Effect of drug X on glycaemic control in adults with type 2 diabetes: a randomised trial"


def test_exact_doi_match_merges() -> None:
    a = make_record("a", TITLE, doi="10.5555/x")
    b = make_record(
        "b",
        "Completely different title text",
        doi="https://doi.org/10.5555/X",
        source_database=SourceDatabase.OPENALEX,
    )
    result = deduplicate([a, b])
    assert result.duplicates_removed == 1
    assert result.duplicate_of == {"b": "a"}
    assert result.match_reasons["b"] == "exact doi"


def test_exact_pmid_and_pmcid_match() -> None:
    recs = [
        make_record("a", "T one", pmid="1"),
        make_record("b", "T two", pmid="1"),
        make_record("c", "T three", pmcid="PMC5"),
        make_record("d", "T four", pmcid="5"),
    ]
    result = deduplicate(recs)
    assert result.duplicate_of == {"b": "a", "d": "c"}


def test_transitive_identifier_chain() -> None:
    # a~b by DOI, b~c by PMID  =>  a, b, c are one cluster.
    a = make_record("a", "Title A", doi="10.5555/1")
    b = make_record("b", "Title B", doi="10.5555/1", pmid="9")
    c = make_record("c", "Title C", pmid="9")
    result = deduplicate([a, b, c])
    assert len(result.unique_records) == 1
    assert (
        set(result.duplicate_of) == {"a", "c"}
        or set(result.duplicate_of) == {"b", "c"}
        or (len(result.duplicate_of) == 2)
    )


def test_fuzzy_title_match_with_formatting_differences() -> None:
    a = make_record("a", TITLE, pmid="1", abstract="text")
    b = make_record(
        "b",
        "EFFECT OF DRUG X ON GLYCAEMIC CONTROL IN ADULTS WITH TYPE-2 DIABETES - "
        "A RANDOMISED TRIAL.",
        source_database=SourceDatabase.EUROPE_PMC,
        publication_year=2021,
    )
    result = deduplicate([a, b])
    assert result.duplicate_of == {"b": "a"}
    assert result.match_reasons["b"].startswith("fuzzy title")


def test_fuzzy_match_blocked_by_year_gap() -> None:
    a = make_record("a", TITLE, publication_year=2015)
    b = make_record("b", TITLE, publication_year=2020)
    assert deduplicate([a, b]).duplicates_removed == 0


def test_missing_year_compared_against_all() -> None:
    a = make_record("a", TITLE, publication_year=2015)
    b = make_record("b", TITLE, publication_year=None)
    assert deduplicate([a, b]).duplicate_of == {"b": "a"}


@pytest.mark.parametrize(
    ("extra_a", "extra_b", "reason"),
    [
        ({"doi": "10.5555/a"}, {"doi": "10.5555/b"}, "different DOIs"),
        ({"pmid": "1"}, {"pmid": "2"}, "different PMIDs"),
        ({"authors": [make_author("Smith", "J")]}, {"authors": [make_author("Jones", "K")]},
         "different first-author surnames"),
    ],
)  # fmt: skip
def test_contradicting_evidence_blocks_fuzzy_merge(
    extra_a: dict[str, object], extra_b: dict[str, object], reason: str
) -> None:
    a = make_record("a", TITLE, **extra_a)
    b = make_record("b", TITLE, **extra_b)
    result = deduplicate([a, b])
    assert result.duplicates_removed == 0
    assert len(result.possible_duplicates) == 1
    pd = result.possible_duplicates[0]
    assert (pd.record_id_a, pd.record_id_b, pd.blocking_reason) == ("a", "b", reason)
    assert pd.title_similarity == pytest.approx(100.0)


def test_cluster_level_contradiction_prevents_transitive_false_merge() -> None:
    # a and c have different DOIs; b (no DOI) is title-identical to both.
    a = make_record("a", TITLE, doi="10.5555/a")
    b = make_record("b", TITLE)
    c = make_record("c", TITLE, doi="10.5555/c")
    result = deduplicate([a, b, c])
    clusters = result.clusters()
    assert len(result.unique_records) == 2
    assert not any({"a", "c"} <= set(members) for members in clusters.values())
    assert result.possible_duplicates


def test_different_titles_not_merged() -> None:
    a = make_record("a", "Drug X in type 2 diabetes: a randomised trial")
    b = make_record("b", "Drug Y in type 1 diabetes: an observational cohort")
    assert deduplicate([a, b]).duplicates_removed == 0


def test_representative_is_most_complete_and_identifiers_filled() -> None:
    sparse = make_record(
        "a",
        TITLE,
        source_database=SourceDatabase.OPENALEX,
        authors=[],
        doi="10.5555/z",
        registry_ids=["NCT01234567"],
    )
    rich = make_record(
        "b",
        TITLE,
        pmid="42",
        abstract="abstract",
        doi="10.5555/z",
        journal="J",
        mesh_headings=["Diabetes Mellitus"],
    )
    result = deduplicate([sparse, rich])
    assert result.duplicate_of == {"a": "b"}
    rep = result.unique_records[0]
    assert rep.record_id == "b"
    assert rep.registry_ids == ["NCT01234567"]
    flagged = {r.record_id: r for r in result.all_records}
    assert flagged["a"].duplicate_of == "b"
    assert flagged["b"].duplicate_of is None


def test_identifier_fill_never_overwrites() -> None:
    a = make_record("a", TITLE, pmid="1", doi="10.5555/a", abstract="A")
    b = make_record("b", TITLE, pmid="1", pmcid="PMC7", abstract="B")
    rep = deduplicate([a, b]).unique_records[0]
    assert rep.record_id == "a"
    assert (rep.doi, rep.pmcid, rep.abstract) == ("10.5555/a", "PMC7", "A")


def test_deterministic_regardless_of_input_order() -> None:
    recs = [
        make_record("r1", TITLE, pmid="1"),
        make_record("r2", TITLE.upper(), doi="10.5555/q"),
        make_record("r3", TITLE + ".", source_database=SourceDatabase.OPENALEX),
        make_record("r4", "An unrelated title about statins and myopathy"),
        make_record("r5", "An unrelated title about statins and myopathy.", pmid="5"),
        make_record("r6", "Something else entirely", doi="10.5555/q"),
    ]
    reference = deduplicate(recs)
    rng = random.Random(1234)
    for _ in range(10):
        shuffled = recs[:]
        rng.shuffle(shuffled)
        result = deduplicate(shuffled)
        assert result.duplicate_of == reference.duplicate_of
        assert [r.record_id for r in result.unique_records] == [
            r.record_id for r in reference.unique_records
        ]


def test_counts_and_clusters_are_consistent() -> None:
    recs = [make_record(f"r{i}", f"{TITLE} {'variant ' * (i % 3)}".strip()) for i in range(9)]
    result = deduplicate(recs)
    assert len(result.unique_records) + result.duplicates_removed == len(recs)
    members = list(itertools.chain.from_iterable(result.clusters().values()))
    assert sorted(members) == sorted(r.record_id for r in recs)


def test_input_validation() -> None:
    with pytest.raises(ValueError, match="unique"):
        deduplicate([make_record("a", "T"), make_record("a", "T")])
    with pytest.raises(ValueError, match="title_threshold"):
        Deduplicator(title_threshold=10)
    with pytest.raises(ValueError, match="year_tolerance"):
        Deduplicator(year_tolerance=-1)
    assert deduplicate([]).unique_records == []
    assert len(deduplicate([make_record("a", "T")]).unique_records) == 1


def test_threshold_controls_sensitivity() -> None:
    a = make_record("a", "Drug X lowers HbA1c in adults: randomised trial")
    b = make_record("b", "Drug X lowers HbA1c in older adults: randomised trial")
    assert deduplicate([a, b], title_threshold=99.0).duplicates_removed == 0
    assert deduplicate([a, b], title_threshold=85.0).duplicates_removed == 1
