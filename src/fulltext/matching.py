"""Does a PDF belong to the study record it is being linked to?

Deterministic signals from the first pages of the parsed PDF: the record's DOI / PMID
printed on page 1, DOIs of *other* articles on page 1, fuzzy title similarity, and the
first author's surname. Only strong evidence yields ``CONFIRMED``; contradictory evidence
yields ``MISMATCH``; everything else is ``UNCERTAIN`` and must be confirmed by a human
before Gate 2 can be released. Nothing is ever auto-accepted on weak evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from src.core.enums import UploadMatchStatus
from src.core.identifiers import normalize_doi, normalize_title
from src.core.schemas.study import StudyRecord
from src.pdf_extractor.models import ParsedDocument

_DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s\"<>]+", re.IGNORECASE)
TITLE_STRONG = 95.0
TITLE_WEAK = 60.0


@dataclass(frozen=True, slots=True)
class MatchAssessment:
    status: UploadMatchStatus
    title_similarity: float
    doi_found: bool
    other_dois: tuple[str, ...]
    author_found: bool
    pmid_found: bool
    notes: tuple[str, ...] = field(default_factory=tuple)


def _dois(text: str) -> set[str]:
    out: set[str] = set()
    for raw in _DOI_RE.findall(text):
        try:
            doi = normalize_doi(raw.rstrip(".,;)]"))
        except ValueError:
            continue
        if doi:
            out.add(doi)
    return out


def assess_match(record: StudyRecord, doc: ParsedDocument) -> MatchAssessment:
    first = doc.pages[0].text if doc.pages else ""
    head = "\n".join(p.text for p in doc.pages[:2])
    notes: list[str] = []
    if not head.strip():
        return MatchAssessment(
            UploadMatchStatus.UNCERTAIN, 0.0, False, (), False, False,
            ("no text layer on the first pages; identity cannot be checked automatically",),
        )  # fmt: skip
    page1_dois = _dois(first)
    doi_found = bool(record.doi and record.doi in _dois(head))
    other = tuple(sorted(page1_dois - {record.doi} if record.doi else page1_dois))
    norm_head = normalize_title(head)
    title_sim = float(fuzz.partial_ratio(record.normalized_title, norm_head))
    surname = record.authors[0].last_name if record.authors else None
    author_found = (
        bool(surname)
        and re.search(rf"\b{re.escape(surname or '')}\b", head, re.IGNORECASE) is not None
    )
    pmid_found = bool(record.pmid) and re.search(rf"\b{record.pmid}\b", head) is not None

    if doi_found:
        status = UploadMatchStatus.CONFIRMED
        notes.append("record DOI printed in the PDF")
        if title_sim < TITLE_WEAK:
            status = UploadMatchStatus.UNCERTAIN
            notes.append(f"but title similarity is low ({title_sim:.0f})")
    elif other and title_sim < TITLE_STRONG:
        status = UploadMatchStatus.MISMATCH
        notes.append(f"page 1 shows a different DOI {other[0]}")
    elif title_sim >= TITLE_STRONG and (author_found or pmid_found):
        status = UploadMatchStatus.CONFIRMED
        notes.append("title and first author/PMID match")
    elif title_sim < TITLE_WEAK and not author_found:
        status = UploadMatchStatus.MISMATCH
        notes.append(f"title similarity {title_sim:.0f} and first author not found")
    else:
        status = UploadMatchStatus.UNCERTAIN
        notes.append(f"title similarity {title_sim:.0f}; author found: {author_found}")
    return MatchAssessment(
        status, title_sim, doi_found, other, author_found, pmid_found, tuple(notes)
    )
