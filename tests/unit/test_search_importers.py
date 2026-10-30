"""Session 5: RIS / BibTeX / CSV import parsers (synthetic exports only)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.search.importers import (
    ImportFileError,
    decode,
    detect_columns,
    guess_format,
    parse_bibtex,
    parse_file,
    parse_ris,
)
from src.ui import protocol_form as pf

FIX = Path(__file__).parents[1] / "fixtures" / "search"


def test_ris_maps_present_fields_and_keeps_raw_tags() -> None:
    parsed = parse_file((FIX / "embase_synthetic.ris").read_bytes(), "ris")
    assert len(parsed.entries) == 3 and not parsed.errors
    first = parsed.entries[0]
    assert first.title == "Effect of drug X on HbA1c: a randomised controlled trial"
    assert first.doi == "10.5555/synth.0001"
    assert [a.last_name for a in first.authors] == ["Synthetica", "Placebo"]
    assert first.year == 2021 and first.source_id == "L2000000001"
    assert "(ISRCTN12345678)" in (first.abstract or "")  # continuation line joined
    assert first.registry_ids == ["ISRCTN12345678"]
    assert first.raw["AN"] == ["L2000000001"]
    assert first.pmid is None  # never guessed from the accession number
    assert parsed.entries[1].publication_types == ["CONF"]


def test_ris_structure_errors_refuse_the_file() -> None:
    with pytest.raises(ImportFileError, match="not terminated"):
        parse_ris("TY  - JOUR\nTI  - Unfinished\n")
    with pytest.raises(ImportFileError, match="outside an entry"):
        parse_ris("TI  - No type\nER  - \n")
    with pytest.raises(ImportFileError, match="'TY' before 'ER'"):
        parse_ris("TY  - JOUR\nTY  - JOUR\nER  - \n")


def test_untitled_entries_are_reported_and_invalid_ids_left_out() -> None:
    text = "TY  - JOUR\nAU  - Nobody, N\nER  - \nTY  - JOUR\nTI  - T\nDO  - not-a-doi\nER  - \n"
    entries = parse_ris(text)
    parsed = parse_file(text.encode(), "ris")
    assert parsed.errors == ["entry 1: no title"]
    assert entries[1].doi is None
    assert any("invalid doi" in w for w in entries[1].warnings)


def test_bibtex_nested_braces_latex_accents_and_identifiers() -> None:
    entries = parse_bibtex((FIX / "scopus_synthetic.bib").read_text("utf-8"))
    assert len(entries) == 2
    a, b = entries
    assert a.title == "Effect of drug X on HbA1c: a randomised controlled trial"
    assert a.doi == "10.5555/synth.0001" and a.pmid == "90000001"
    assert a.keywords == ["drug x", "condition y"]
    assert b.year == 2023
    assert b.authors[0].last_name == "Müller" and b.authors[0].fore_name == "Jörg"
    assert b.raw["note"] == "Synthetic entry \\& test"  # raw kept verbatim


def test_bibtex_unbalanced_braces_refused() -> None:
    with pytest.raises(ImportFileError, match="unbalanced"):
        parse_bibtex("@article{x, title = {open")


def test_csv_column_detection_registry_ids_and_override() -> None:
    data = (FIX / "ctgov_synthetic.csv").read_bytes()
    parsed = parse_file(data, "csv")
    assert parsed.column_map["title"] == "Study Title"
    assert parsed.column_map["registry_id"] == "NCT Number"
    assert parsed.column_map["abstract"] == "Brief Summary"
    e = parsed.entries[0]
    assert e.registry_ids == ["NCT09999991"] and e.source_id == "NCT09999991"
    assert e.raw["Study Status"] == "COMPLETED"
    overridden = parse_file(data, "csv", column_map={"abstract": "Conditions"})
    assert overridden.entries[0].abstract == "Condition Y"
    with pytest.raises(ImportFileError, match="not in the file"):
        parse_file(data, "csv", column_map={"title": "Nope"})
    with pytest.raises(ImportFileError, match="no title column"):
        parse_file(b"a,b\n1,2\n", "csv")


def test_scopus_like_headers_detected() -> None:
    mapping = detect_columns(
        ["Authors", "Title", "Year", "Source title", "DOI", "PubMed ID", "EID", "Abstract"]
    )
    assert mapping == {
        "title": "Title",
        "abstract": "Abstract",
        "authors": "Authors",
        "year": "Year",
        "journal": "Source title",
        "doi": "DOI",
        "pmid": "PubMed ID",
        "source_id": "EID",
    }


def test_decoding_and_format_guess() -> None:
    assert decode("é".encode())[1] == "utf-8-sig"
    assert decode("é".encode("cp1252")) == ("é", "cp1252")
    with pytest.raises(ImportFileError, match="neither UTF-8"):
        decode(b"\x81\x8d\x8f\x90\x9d\xff\xfe\x00\x81")
    assert guess_format("export.RIS") == "ris" and guess_format("x.bib") == "bibtex"
    assert guess_format("x.csv") == "csv" and guess_format("x.txt") is None


def test_protocol_form_helpers() -> None:
    tps = pf.parse_timepoints("5 years; 24-36 months; last follow-up; 1 year")
    assert tps == [
        {"kind": "fixed", "value": 5.0, "unit": "years"},
        {"kind": "window", "window_min": 24.0, "window_max": 36.0, "unit": "months"},
        {"kind": "last_follow_up"},
        {"kind": "fixed", "value": 1.0, "unit": "years"},
    ]
    assert pf.format_timepoints(tps) == "5 years; 24-36 months; last follow-up; 1 years"
    with pytest.raises(ValueError, match="cannot read timepoint"):
        pf.parse_timepoints("soon")
    mesh = pf.parse_mesh("Diabetes Mellitus\nRadiosurgery [noexp] [majr]")
    assert mesh[1] == {"descriptor": "Radiosurgery", "explode": False, "major_topic_only": True}
    assert pf.format_mesh(mesh) == "Diabetes Mellitus\nRadiosurgery [noexp] [majr]"
    assert pf.parse_date(" 2020-01-31 ") == "2020-01-31" and pf.parse_date("") is None
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        pf.parse_date("31/01/2020")
    out = pf.upsert_outcome([{"outcome_id": "a", "n": 1}], {"outcome_id": "a", "n": 2})
    assert out == [{"outcome_id": "a", "n": 2}]
    s = pf.set_strategy([{"database": "pubmed", "q": 1}], "pubmed", None)
    assert s == []


# ---------------------------------------------------------------------- export edge cases
# Synthetic files imitating common export quirks; NOT actual vendor exports.
def test_ris_with_bom_crlf_and_blank_lines_inside_entries() -> None:
    text = (
        "﻿TY  - JOUR\r\nTI  - First synthetic title\r\n\r\nAU  - Smith, J. M.\r\n"
        "AU  - O'Brien, K.\r\nPY  - 2020///\r\nER  - \r\n\r\nTY  - JOUR\r\nT1  - Second\r\n"
        "ER  -\r\n"
    )
    parsed = parse_file(text.encode("utf-8"), "ris")
    assert parsed.encoding == "utf-8-sig" and not parsed.errors
    assert [e.title for e in parsed.entries] == ["First synthetic title", "Second"]
    assert [a.last_name for a in parsed.entries[0].authors] == ["Smith", "O'Brien"]
    assert parsed.entries[0].year == 2020


def test_semicolon_csv_with_multiline_quoted_abstract() -> None:
    text = (
        "Title;Authors;Year;DOI;Abstract\n"
        '"Synthetic; a title with a semicolon";"Smith J.; Doe A.";2019;10.5555/csv.1;'
        '"Line one\nline two"\n'
        '"Second synthetic title";"Roe R.";2020;;"Short"\n'
    )
    parsed = parse_file(text.encode(), "csv")
    assert len(parsed.entries) == 2 and not parsed.errors
    first = parsed.entries[0]
    assert first.title == "Synthetic; a title with a semicolon"
    assert first.abstract == "Line one line two" and first.doi == "10.5555/csv.1"
    assert [a.full_name for a in first.authors] == ["Smith J.", "Doe A."]
    assert parsed.entries[1].doi is None


def test_bibtex_comment_string_and_unquoted_fields_are_skipped_or_kept() -> None:
    text = (
        '@comment{exported by a synthetic tool}\n@string{jsm = "J Synth Med"}\n'
        '@ARTICLE{Key2021,\n  title = "Quoted {Synthetic} title",\n  year = 2021,\n'
        "  month = mar,\n  doi = {https://doi.org/10.5555/BIB.9}\n}\n"
    )
    entries = parse_bibtex(text)
    assert len(entries) == 1
    e = entries[0]
    assert e.title == "Quoted Synthetic title" and e.year == 2021
    assert e.doi == "10.5555/bib.9" and e.raw["month"] == "mar"
