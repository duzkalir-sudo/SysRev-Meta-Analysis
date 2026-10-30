"""Session 2: PDF parsing and verification of evidence against the parsed document.

The PDFs are generated SYNTHETIC fixtures (tests/fixtures/pdf_factory.py).
"""

from __future__ import annotations

import pytest

from src.core.enums import ActorType, EvidenceCheckStatus, EvidenceLocationType
from src.core.schemas import SourceEvidence
from src.pdf_extractor import (
    HashMismatchError,
    PDFParseError,
    match_key,
    number_in_text,
    parse_location_id,
    parse_pdf,
    select_context,
    sha256_bytes,
    verify_evidence,
)
from src.pdf_extractor.models import ParsedDocument
from tests.fixtures.pdf_factory import Page, build_pdf, synthetic_article

FILE = "synthetic.pdf"


@pytest.fixture(scope="module")
def doc() -> ParsedDocument:
    return parse_pdf(synthetic_article(), filename=FILE)


def ev(doc: ParsedDocument, quote: str, page: int, **kw: object) -> SourceEvidence:
    data: dict[str, object] = {
        "pdf_filename": FILE,
        "pdf_sha256": doc.pdf_sha256,
        "page_number": page,
        "location_type": EvidenceLocationType.PARAGRAPH,
        "location_identifier": f"page {page}",
        "verbatim_quote": quote,
        "extracted_by": ActorType.AGENT,
        "extractor_identity": "test",
    } | kw
    return SourceEvidence(**data)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- parsing


def test_page_indexed_text_labels_and_printed_numbers(doc: ParsedDocument) -> None:
    assert doc.n_pages == 4 and doc.has_page_labels
    assert [p.page_label for p in doc.pages] == ["101", "102", "103", "104"]
    p1 = doc.pages[0]
    assert p1.physical_page == 1 and p1.printed_page_text == "101"
    assert "57 patients" in p1.text
    assert doc.pdf_sha256 == sha256_bytes(synthetic_article())


def test_block_classification(doc: ParsedDocument) -> None:
    kinds = {b.kind for p in doc.pages for b in p.blocks}
    assert {
        "paragraph",
        "heading",
        "table_caption",
        "figure_legend",
        "supplementary",
        "page_number",
    } <= kinds
    caption = next(b for b in doc.pages[1].blocks if b.kind == "table_caption")
    assert caption.block_id == "p2.b3" and doc.find_block("p2.b3") == caption


def test_table_rows_columns_headers_and_ids(doc: ParsedDocument) -> None:
    table = doc.pages[1].tables[0]
    assert table.table_id == "p2.t1" and table.caption == "Table 2. Outcomes by timepoint"
    assert table.header == ("Outcome", "3 years", "5 years")
    assert table.n_rows == 4 and table.n_cols == 3
    assert table.cell(1, 2) == "81.4" and table.cell_id(1, 2) == "p2.t1.r1c2"
    assert not table.unreadable
    assert parse_location_id("p2.t1.r1c2") == (2, "t", 1, 1, 2)
    assert parse_location_id("Table 2") is None


def test_image_only_page_is_flagged_not_invented(doc: ParsedDocument) -> None:
    p3 = doc.pages[2]
    assert p3.is_image_only and p3.text == "" and p3.needs_manual_review
    assert "OCR not performed" in p3.issues[0]
    assert doc.pages_needing_review == [3]


def test_caption_without_ruled_table_and_unreadable_table() -> None:
    pdf = build_pdf(
        [
            Page()
            .text(50, 60, "Table 1. Baseline characteristics")
            .text(50, 80, "Age 54 Sex 30 male"),
            Page().table(50, 60, [80, 80, 80], [["A", "", ""], ["", "", ""], ["", "", ""]]),
        ]
    )
    d = parse_pdf(pdf, filename="x.pdf")
    assert any("no ruled table" in i for i in d.pages[0].issues)
    assert d.pages[1].tables[0].unreadable
    assert not d.has_page_labels and d.pages[0].page_label is None


def test_malformed_pdf_and_hash_mismatch() -> None:
    with pytest.raises(PDFParseError):
        parse_pdf(b"%PDF-1.4 garbage", filename="bad.pdf")
    with pytest.raises(HashMismatchError):
        parse_pdf(synthetic_article(), filename=FILE, expected_sha256="0" * 64)


def test_context_selection_is_limited_and_labelled(doc: ParsedDocument) -> None:
    snippets = select_context(doc, ["obliteration"], max_chars=400)
    assert snippets and sum(len(s.text) for s in snippets) <= 400
    assert all(s.location_id.startswith(f"p{s.page}.") for s in snippets)


# --------------------------------------------------------------------------- matching


def test_match_key_tolerates_artifacts_but_never_digits() -> None:
    assert match_key("ﬁxed  dose− 12·5 %") == match_key("fixed dose- 12.5%")
    assert match_key("treat-\nment") == match_key("treatment")
    assert match_key("12 34") != match_key("1234")
    assert match_key("10-\n20") != match_key("1020")


def test_number_token_boundaries() -> None:
    assert number_in_text("45", "achieved in 45 of 57 patients")
    assert not number_in_text("12", "a value of 125 mm")
    assert not number_in_text("12", "a value of 12.5 mm")
    assert number_in_text("12·5", "mean 12.5 (SD 3)")
    assert number_in_text("−2.1", "change -2.1 points")


# --------------------------------------------------------------------------- verification


def test_verified_quote_on_correct_page(doc: ParsedDocument) -> None:
    check = verify_evidence(
        ev(doc, "Complete obliteration was achieved in 45 of 57 patients (78.9%).", 2),
        doc,
        reported_text="45",
    )
    assert check.status == EvidenceCheckStatus.VERIFIED and check.matched_pages == (2,)


def test_incorrect_page_reference(doc: ParsedDocument) -> None:
    check = verify_evidence(ev(doc, "45 of 57 patients", 1), doc, reported_text="45")
    assert check.status == EvidenceCheckStatus.WRONG_PAGE and check.matched_pages == (2,)


def test_fabricated_quotation(doc: ParsedDocument) -> None:
    check = verify_evidence(ev(doc, "Obliteration was achieved in 52 of 57 patients", 2), doc)
    assert check.status == EvidenceCheckStatus.QUOTE_NOT_FOUND


def test_wrong_hash_and_wrong_file(doc: ParsedDocument) -> None:
    check = verify_evidence(ev(doc, "45 of 57 patients", 2, pdf_sha256="b" * 64), doc)
    assert check.status == EvidenceCheckStatus.HASH_MISMATCH
    check = verify_evidence(ev(doc, "45 of 57 patients", 2, pdf_sha256=None), doc)
    assert check.status == EvidenceCheckStatus.HASH_MISMATCH
    check = verify_evidence(ev(doc, "45 of 57 patients", 2, pdf_filename="other.pdf"), doc)
    assert check.status == EvidenceCheckStatus.FILE_MISMATCH


def test_page_out_of_range_and_unreadable_page(doc: ParsedDocument) -> None:
    assert verify_evidence(ev(doc, "x", 9), doc).status == EvidenceCheckStatus.PAGE_OUT_OF_RANGE
    assert verify_evidence(ev(doc, "x", 3), doc).status == EvidenceCheckStatus.PAGE_UNREADABLE


def test_number_must_be_in_quote(doc: ParsedDocument) -> None:
    check = verify_evidence(ev(doc, "45 of 57 patients", 2), doc, reported_text="4")
    assert check.status == EvidenceCheckStatus.NUMBER_NOT_IN_QUOTE


def test_table_cell_location_is_checked(doc: ParsedDocument) -> None:
    table_ev = {"location_type": EvidenceLocationType.TABLE}
    ok = ev(doc, "Obliteration (KM, %) 62.1 81.4", 2, location_identifier="p2.t1.r1c2", **table_ev)
    assert verify_evidence(ok, doc, reported_text="81.4").status == EvidenceCheckStatus.VERIFIED
    wrong_cell = ok.model_copy(update={"location_identifier": "p2.t1.r1c1"})
    check = verify_evidence(wrong_cell, doc, reported_text="81.4")
    assert check.status == EvidenceCheckStatus.LOCATION_MISMATCH and "62.1" in check.detail
    wrong_page_id = ok.model_copy(update={"location_identifier": "p1.t1"})
    assert verify_evidence(wrong_page_id, doc).status == EvidenceCheckStatus.LOCATION_MISMATCH
    block = ev(doc, "45 of 57 patients", 2, location_identifier="p2.b3")
    assert verify_evidence(block, doc).status == EvidenceCheckStatus.LOCATION_MISMATCH
