from __future__ import annotations

import pytest

from src.core.exceptions import InvalidIdentifierError
from src.core.identifiers import (
    extract_registry_ids,
    normalize_doi,
    normalize_person_name,
    normalize_pmcid,
    normalize_pmid,
    normalize_title,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("10.5555/ABC.123", "10.5555/abc.123"),
        ("https://doi.org/10.5555/abc.123", "10.5555/abc.123"),
        ("http://dx.doi.org/10.5555/abc.123", "10.5555/abc.123"),
        ("doi: 10.5555/abc.123.", "10.5555/abc.123"),
        ("  DOI:10.5555/(SICI)1097-0258  ", "10.5555/(sici)1097-0258"),
        (None, None),
        ("   ", None),
    ],
)
def test_normalize_doi(raw: str | None, expected: str | None) -> None:
    assert normalize_doi(raw) == expected


@pytest.mark.parametrize("raw", ["11.1234/x", "10.12/x", "10.5555/", "not a doi", "10.5555/a b"])
def test_normalize_doi_rejects_invalid(raw: str) -> None:
    with pytest.raises(InvalidIdentifierError):
        normalize_doi(raw)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("12345", "12345"),
        (12345, "12345"),
        ("00012345", "12345"),
        ("https://pubmed.ncbi.nlm.nih.gov/12345/", "12345"),
        ("PMID: 12345", "12345"),
        (None, None),
    ],
)
def test_normalize_pmid(raw: str | int | None, expected: str | None) -> None:
    assert normalize_pmid(raw) == expected


@pytest.mark.parametrize("raw", ["0", "abc", "1234567890", "12.5"])
def test_normalize_pmid_rejects_invalid(raw: str) -> None:
    with pytest.raises(InvalidIdentifierError):
        normalize_pmid(raw)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("PMC123456", "PMC123456"),
        ("pmc123456", "PMC123456"),
        ("123456", "PMC123456"),
        ("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123456/", "PMC123456"),
    ],
)
def test_normalize_pmcid(raw: str, expected: str) -> None:
    assert normalize_pmcid(raw) == expected


def test_normalize_pmcid_rejects_invalid() -> None:
    with pytest.raises(InvalidIdentifierError):
        normalize_pmcid("PMCX1")


def test_normalize_title_is_accent_case_and_markup_insensitive() -> None:
    a = normalize_title("Effect of <i>Café</i>-based Intervention: A Randomised Trial.")
    b = normalize_title("effect of cafe based intervention a randomised trial")
    assert a == b


def test_normalize_title_empty() -> None:
    assert normalize_title(None) == ""
    assert normalize_title("") == ""


def test_normalize_person_name() -> None:
    assert normalize_person_name("Müller", "Hans Peter") == "muller h"
    assert normalize_person_name("van der Berg", "Anna") == "vanderberg a"
    assert normalize_person_name(None, "x") == ""
    assert normalize_person_name("Smith") == "smith"


def test_extract_registry_ids() -> None:
    text = (
        "Registered at ClinicalTrials.gov (NCT01234567) and ISRCTN12345678; "
        "protocol CRD42020123456. Repeat NCT01234567."
    )
    assert extract_registry_ids(text) == ["CRD42020123456", "ISRCTN12345678", "NCT01234567"]
    assert extract_registry_ids(None) == []
    assert extract_registry_ids("NCT123") == []
