"""Reference registry: import by identifier and live identity verification.

References enter the registry only from bibliographic APIs (PubMed E-utilities by PMID,
Crossref by DOI) or from stored study records — never from a language model. Verification
compares the stored entry with a *live* lookup:

* the identifier must resolve (else ``not_found``);
* PubMed and Crossref must agree with each other on DOI/PMID where both are known;
* title (token-sort similarity ≥ 90 after normalisation), year and first-author surname
  must match (journal names are compared but only reported, as abbreviations differ).

A network or API failure is recorded as ``unavailable`` — never as a verification — so the
Auditor reports the reference as *not checked*. Verifying that a paper exists says nothing
about whether it supports a sentence; claim-level support is assessed separately.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Any, Protocol, cast

from rapidfuzz import fuzz

from src.apis.crossref import CrossrefWork
from src.core.exceptions import SysRevError
from src.core.identifiers import normalize_title
from src.core.schemas.base import utc_now
from src.core.schemas.manuscript import ReferenceEntry, ReferenceVerification
from src.core.schemas.study import StudyRecord
from src.manuscript.evidence import vancouver_author

TITLE_THRESHOLD = 90


class PubMedLookup(Protocol):
    async def efetch_records(
        self, pmids: Any, *, query: str = "", total_hits: int | None = None
    ) -> list[StudyRecord]: ...


class CrossrefLookup(Protocol):
    async def get_work(self, doi: str) -> CrossrefWork | None: ...


def _fold(text: str | None) -> str:
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).casefold().strip()


def entry_from_pubmed(
    rec: StudyRecord, *, added_by: str, reviewer_id: str | None
) -> ReferenceEntry:
    if rec.pmid is None:
        raise SysRevError("PubMed returned a record without a PMID")
    return ReferenceEntry(
        ref_key=f"pmid-{rec.pmid}",
        origin="imported_literature",
        pmid=rec.pmid,
        doi=rec.doi,
        title=rec.title,
        authors=[x for x in (vancouver_author(a) for a in rec.authors) if x],
        journal=rec.journal,
        year=rec.publication_year,
        metadata_source="pubmed_efetch",
        retrieved_at=utc_now(),
        added_by=added_by,
        reviewer_id=reviewer_id,
    )


def entry_from_crossref(
    w: CrossrefWork, *, added_by: str, reviewer_id: str | None
) -> ReferenceEntry:
    if not w.title:
        raise SysRevError(f"Crossref record {w.doi} has no title; it cannot be cited")
    key = "doi-" + "".join(c if c.isalnum() else "_" for c in w.doi)
    return ReferenceEntry(
        ref_key=key[:64],
        origin="imported_literature",
        doi=w.doi,
        title=w.title,
        authors=[w.first_author_family] if w.first_author_family else [],
        journal=w.container_title,
        year=w.year,
        metadata_source="crossref",
        retrieved_at=utc_now(),
        added_by=added_by,
        reviewer_id=reviewer_id,
    )


async def import_reference(
    *,
    pmid: str | None,
    doi: str | None,
    pubmed: PubMedLookup | None,
    crossref: CrossrefLookup | None,
    added_by: str,
    reviewer_id: str | None,
) -> ReferenceEntry:
    """Fetch a background reference by identifier. Raises if the lookup fails."""
    if pmid:
        if pubmed is None:
            raise SysRevError("PubMed client not configured (set SYSREV_NCBI_EMAIL)")
        recs = await pubmed.efetch_records([pmid])
        if not recs:
            raise SysRevError(f"PMID {pmid} was not found in PubMed")
        return entry_from_pubmed(recs[0], added_by=added_by, reviewer_id=reviewer_id)
    if doi:
        if crossref is None:
            raise SysRevError("Crossref client not configured")
        work = await crossref.get_work(doi)
        if work is None:
            raise SysRevError(f"DOI {doi} was not found in Crossref")
        return entry_from_crossref(work, added_by=added_by, reviewer_id=reviewer_id)
    raise SysRevError("a PMID or DOI is required; references are never entered from memory")


@dataclass(frozen=True, slots=True)
class _Observed:
    title: str | None
    year: int | None
    first_author: str | None
    journal: str | None
    doi: str | None
    pmid: str | None


def _compare(entry: ReferenceEntry, obs: _Observed) -> tuple[str, dict[str, str]]:
    results: dict[str, str] = {}
    ok = True
    if obs.title:
        score = fuzz.token_sort_ratio(normalize_title(entry.title), normalize_title(obs.title))
        good = score >= TITLE_THRESHOLD
        results["title"] = f"{'match' if good else 'MISMATCH'} (similarity {score:.0f})"
        ok &= good
    else:
        results["title"] = "not available from source"
    if entry.year is not None and obs.year is not None:
        good = entry.year == obs.year
        results["year"] = "match" if good else f"MISMATCH (source {obs.year})"
        ok &= good
    if entry.authors and obs.first_author:
        good = _fold(entry.authors[0]).split(" ")[0] == _fold(obs.first_author).split(" ")[0]
        results["first_author"] = "match" if good else f"MISMATCH (source {obs.first_author})"
        ok &= good
    if entry.journal and obs.journal:
        same = fuzz.token_set_ratio(_fold(entry.journal), _fold(obs.journal)) >= 85
        results["journal"] = (
            "match"
            if same
            else f"differs (source {obs.journal!r}; abbreviations may differ — not decisive)"
        )
    for name in ("doi", "pmid"):
        mine, theirs = getattr(entry, name), getattr(obs, name)
        if mine and theirs:
            good = mine.lower() == theirs.lower()
            results[name] = "match" if good else f"MISMATCH (source {theirs})"
            ok &= good
    return ("verified" if ok else "mismatch"), results


async def verify_reference(
    entry: ReferenceEntry,
    *,
    pubmed: PubMedLookup | None,
    crossref: CrossrefLookup | None,
    checked_by: str,
) -> list[ReferenceVerification]:
    """One live verification per available identifier/source."""
    out: list[ReferenceVerification] = []
    meta = entry.metadata_sha256()

    def rec(source: str, status: str, fields: dict[str, str], message: str) -> None:
        out.append(
            ReferenceVerification(
                ref_key=entry.ref_key,
                metadata_sha256=meta,
                source=cast(Any, source),
                status=cast(Any, status),
                field_results=fields,
                message=message,
                checked_by=checked_by,
            )
        )

    if entry.pmid:
        if pubmed is None:
            rec("pubmed", "unavailable", {}, "PubMed client not configured")
        else:
            try:
                recs = await pubmed.efetch_records([entry.pmid])
            except Exception as exc:  # network/API failure: never a verification
                rec(
                    "pubmed", "unavailable", {}, f"lookup failed: {type(exc).__name__}: {exc}"[:300]
                )
            else:
                if not recs:
                    rec("pubmed", "not_found", {}, f"PMID {entry.pmid} not found in PubMed")
                else:
                    r = recs[0]
                    first = r.authors[0].last_name if r.authors else None
                    status, fields = _compare(
                        entry,
                        _Observed(r.title, r.publication_year, first, r.journal, r.doi, r.pmid),
                    )
                    rec("pubmed", status, fields, "live PubMed E-utilities lookup")
    if entry.doi:
        if crossref is None:
            rec("crossref", "unavailable", {}, "Crossref client not configured")
        else:
            try:
                work = await crossref.get_work(entry.doi)
            except Exception as exc:
                rec(
                    "crossref",
                    "unavailable",
                    {},
                    f"lookup failed: {type(exc).__name__}: {exc}"[:300],
                )
            else:
                if work is None:
                    rec("crossref", "not_found", {}, f"DOI {entry.doi} not found in Crossref")
                else:
                    status, fields = _compare(
                        entry,
                        _Observed(
                            work.title,
                            work.year,
                            work.first_author_family,
                            work.container_title,
                            work.doi,
                            None,
                        ),
                    )
                    rec("crossref", status, fields, "live Crossref lookup")
    if not out:
        rec("crossref", "unavailable", {}, "no PMID or DOI: identity cannot be verified")
    return out


def combined_status(verifications: list[ReferenceVerification]) -> str:
    """Worst decisive result wins; 'unavailable' only if nothing decisive was obtained."""
    statuses = {v.status for v in verifications}
    for s in ("mismatch", "not_found", "verified"):
        if s in statuses:
            return s
    return "unavailable"
