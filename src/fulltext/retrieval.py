"""Stage 3: resumable open-access retrieval and human-authenticated manual steps.

``FullTextRetriever.run`` walks the title/abstract-included records and, for each one
without a finished manifest entry, tries (1) Europe PMC open-access JATS XML (via the
record's PMCID, or a PMID→PMCID lookup) and (2) Unpaywall open-access PDF locations. Each
record is committed as soon as it is processed, so the run can be interrupted and resumed
after an application restart; finished records are skipped. Failures are recorded as
attempts and leave the report ``MISSING`` — never excluded.

Manual steps (upload, confirm a PDF↔record link, mark unobtainable) require an enrolled,
non-revoked reviewer; the reviewer's identity is recorded on the manifest entry.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx
from lxml import etree

from src.apis.europepmc import EuropePMCClient
from src.apis.unpaywall import UnpaywallClient
from src.core.enums import FullTextStatus, ScreeningStage, UploadMatchStatus
from src.core.exceptions import APIClientError, SysRevError
from src.core.schemas.extraction import AttemptOutcome, FullTextItem, RetrievalAttempt
from src.core.schemas.study import StudyRecord
from src.fulltext.download import PDFDownloader, validate_pdf_bytes
from src.fulltext.matching import assess_match
from src.fulltext.storage import DocumentStore, StoredFile
from src.pdf_extractor.parser import PDFParseError, parse_pdf
from src.state.db import ReviewerORM
from src.state.repository import NotFoundError, Repository

logger = logging.getLogger(__name__)

FINISHED = frozenset(
    {FullTextStatus.OA_DOWNLOADED, FullTextStatus.USER_UPLOADED, FullTextStatus.UNOBTAINABLE}
)
# Retried when ``retry_missing``: MISSING, and XML-only items (page-grounded extraction
# needs a PDF, so the PDF is sought again; the stored XML is reused, not re-fetched).
RETRYABLE = frozenset({FullTextStatus.MISSING, FullTextStatus.OA_STRUCTURED_TEXT})
MATCH_PAGES = 3


class UploadRejectedError(SysRevError, ValueError):
    """An uploaded file is not a valid PDF."""


@dataclass
class RetrievalSummary:
    processed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    pdf: list[str] = field(default_factory=list)
    xml_only: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    needs_confirmation: list[str] = field(default_factory=list)


def _base_item(record: StudyRecord, previous: FullTextItem | None) -> dict[str, object]:
    first = record.authors[0] if record.authors else None
    return {
        "record_id": record.record_id,
        "first_author": first.last_name or first.collective_name if first else None,
        "year": record.publication_year,
        "journal": record.journal,
        "title": record.title,
        "pmid": record.pmid,
        "doi": record.doi,
        "pmcid": record.pmcid,
        "attempts": list(previous.attempts) if previous else [],
    }


def _check_pdf(
    repo: Repository, project_id: str, record: StudyRecord, stored: StoredFile, data: bytes
) -> tuple[UploadMatchStatus, list[str]]:
    """Identity check of a PDF against its record, plus duplicate detection."""
    notes: list[str] = []
    try:
        doc = parse_pdf(data, filename=stored.filename, max_pages=MATCH_PAGES)
        assessment = assess_match(record, doc)
        status, notes = assessment.status, list(assessment.notes)
    except PDFParseError as exc:
        status, notes = UploadMatchStatus.UNCERTAIN, [f"text could not be parsed: {exc}"]
    others = repo.records_with_document(project_id, stored.sha256) - {record.record_id}
    if others:
        notes.append(f"identical PDF already linked to record(s) {sorted(others)}")
        if status == UploadMatchStatus.CONFIRMED:
            status = UploadMatchStatus.UNCERTAIN
    return status, notes


class FullTextRetriever:
    def __init__(
        self,
        repo: Repository,
        project_id: str,
        store: DocumentStore,
        *,
        downloader: PDFDownloader | None = None,
        europepmc: EuropePMCClient | None = None,
        unpaywall: UnpaywallClient | None = None,
        commit: Callable[[], None] | None = None,
    ) -> None:
        self.repo = repo
        self.project_id = project_id
        self.store = store
        self.downloader = downloader
        self.europepmc = europepmc
        self.unpaywall = unpaywall
        self._commit = commit or repo.session.commit

    async def run(
        self, *, retry_missing: bool = True, limit: int | None = None
    ) -> RetrievalSummary:
        self.store.cleanup_partials()
        summary = RetrievalSummary()
        included = sorted(
            self.repo.included_record_ids(self.project_id, ScreeningStage.TITLE_ABSTRACT)
        )
        for record_id in included:
            if limit is not None and len(summary.processed) >= limit:
                break
            existing = self.repo.get_fulltext_item(self.project_id, record_id)
            if existing and (
                existing.status in FINISHED or (existing.status in RETRYABLE and not retry_missing)
            ):
                summary.skipped.append(record_id)
                continue
            record = self.repo.get_record(self.project_id, record_id)
            item = await self.retrieve_one(record, existing)
            self.repo.save_fulltext_item(self.project_id, item)
            self._commit()  # checkpoint: survives interruption / restart
            summary.processed.append(record_id)
            if item.status == FullTextStatus.OA_DOWNLOADED:
                summary.pdf.append(record_id)
            elif item.status == FullTextStatus.OA_STRUCTURED_TEXT:
                summary.xml_only.append(record_id)
            else:
                summary.missing.append(record_id)
            if item.match_unconfirmed:
                summary.needs_confirmation.append(record_id)
        return summary

    async def retrieve_one(
        self, record: StudyRecord, previous: FullTextItem | None
    ) -> FullTextItem:
        data = _base_item(record, previous)
        attempts: list[RetrievalAttempt] = data["attempts"]  # type: ignore[assignment]
        reuse_xml = False
        xml: StoredFile | None
        if previous is not None and previous.xml_filename and previous.xml_sha256:
            reuse_xml = True
            pmcid = previous.pmcid
            xml = StoredFile(self.store.root / previous.xml_filename, previous.xml_sha256, 0)
        else:
            pmcid, xml = await self._xml(record, attempts)
        data["pmcid"] = pmcid
        if xml is not None:
            data |= {"xml_filename": xml.filename, "xml_sha256": xml.sha256,
                     "source_kind": "pmc_xml"}  # fmt: skip
        if xml is not None and not reuse_xml:
            self.repo.add_fulltext_document(
                self.project_id, record.record_id, kind="xml", filename=xml.filename,
                sha256=xml.sha256, size_bytes=xml.size_bytes, source_kind="pmc_xml",
            )  # fmt: skip
        pdf, url, license_ = await self._pdf(record, attempts)
        if pdf is not None:
            content = self.store.read_verified(pdf.filename, pdf.sha256)
            status, notes = _check_pdf(self.repo, self.project_id, record, pdf, content)
            self.repo.add_fulltext_document(
                self.project_id, record.record_id, kind="pdf", filename=pdf.filename,
                sha256=pdf.sha256, size_bytes=pdf.size_bytes, source_kind="oa_pdf",
                source_url=url,
            )  # fmt: skip
            data |= {
                "status": FullTextStatus.OA_DOWNLOADED, "pdf_filename": pdf.filename,
                "pdf_sha256": pdf.sha256, "source_url": url, "source_kind": "oa_pdf",
                "license": license_, "match_status": status, "match_notes": notes,
            }  # fmt: skip
        elif xml is not None:
            data["status"] = FullTextStatus.OA_STRUCTURED_TEXT
        else:
            data["status"] = FullTextStatus.MISSING
        return FullTextItem.model_validate(data)

    async def _xml(
        self, record: StudyRecord, attempts: list[RetrievalAttempt]
    ) -> tuple[str | None, StoredFile | None]:
        pmcid = record.pmcid
        if self.europepmc is None:
            return pmcid, None
        service = "europe_pmc_fulltext_xml"
        try:
            if pmcid is None and record.pmid:
                pmcid = await self.europepmc.find_pmcid(record.pmid)
            if pmcid is None:
                attempts.append(RetrievalAttempt(service=service, outcome="skipped",
                                                 detail="no PMCID"))  # fmt: skip
                return None, None
            xml = await self.europepmc.fetch_fulltext_xml(pmcid)
        except APIClientError as exc:
            kind: AttemptOutcome = "http_error" if exc.status_code else "network_error"
            attempts.append(RetrievalAttempt(service=service, outcome=kind, detail=str(exc)[:300]))
            return pmcid, None
        if xml is None:
            attempts.append(RetrievalAttempt(service=service, outcome="not_available",
                                             detail=pmcid))  # fmt: skip
            return pmcid, None
        try:
            etree.fromstring(xml, etree.XMLParser(resolve_entities=False, no_network=True))
        except etree.XMLSyntaxError as exc:
            attempts.append(RetrievalAttempt(service=service, outcome="malformed", detail=str(exc)))
            return pmcid, None
        stored = self.store.put_bytes(xml, ".xml")
        attempts.append(RetrievalAttempt(service=service, outcome="retrieved", detail=pmcid))
        return pmcid, stored

    async def _pdf(
        self, record: StudyRecord, attempts: list[RetrievalAttempt]
    ) -> tuple[StoredFile | None, str | None, str | None]:
        if self.unpaywall is None or self.downloader is None:
            return None, None, None
        if not record.doi:
            attempts.append(RetrievalAttempt(service="unpaywall", outcome="skipped",
                                             detail="no DOI"))  # fmt: skip
            return None, None, None
        try:
            result = await self.unpaywall.lookup(record.doi)
        except APIClientError as exc:
            kind: AttemptOutcome = "http_error" if exc.status_code else "network_error"
            attempts.append(RetrievalAttempt(service="unpaywall", outcome=kind,
                                             detail=str(exc)[:300]))  # fmt: skip
            return None, None, None
        if result is None or not result.pdf_candidates:
            attempts.append(RetrievalAttempt(service="unpaywall", outcome="not_available",
                                             detail="no open-access PDF location"))  # fmt: skip
            return None, None, None
        licenses = {
            loc.url_for_pdf: loc.license
            for loc in [result.best_oa_location, *result.oa_locations]
            if loc is not None and loc.url_for_pdf
        }
        for url in result.pdf_candidates:
            try:
                fetched = await self.downloader.fetch(url)
            except (httpx.HTTPError, OSError) as exc:
                attempts.append(
                    RetrievalAttempt(
                        service="oa_pdf", url=url, outcome="network_error", detail=str(exc)[:300]
                    )
                )
                continue  # fmt: skip
            if fetched.ok:
                assert fetched.stored is not None
                attempts.append(RetrievalAttempt(service="oa_pdf", url=url, outcome="retrieved"))
                return fetched.stored, url, licenses.get(url)
            attempts.append(
                RetrievalAttempt(
                    service="oa_pdf", url=url, outcome=fetched.failure or "network_error",
                    detail=fetched.detail,
                )
            )  # fmt: skip
        return None, None, None


# --------------------------------------------------------------------------- manual steps


def register_upload(
    repo: Repository,
    store: DocumentStore,
    project_id: str,
    record_id: str,
    data: bytes,
    *,
    original_filename: str,
    reviewer: ReviewerORM,
) -> FullTextItem:
    """Link a user-supplied PDF to ``record_id`` after validation and an identity check."""
    reviewer = repo.require_active_reviewer(reviewer.id)
    record = repo.get_record(project_id, record_id)
    stored, problem = validate_pdf_bytes(data, store)
    if stored is None:
        raise UploadRejectedError(f"{original_filename}: {problem}")
    previous = repo.get_fulltext_item(project_id, record_id)
    status, notes = _check_pdf(repo, project_id, record, stored, data)
    repo.add_fulltext_document(
        project_id, record_id, kind="pdf", filename=stored.filename, sha256=stored.sha256,
        size_bytes=stored.size_bytes, source_kind="user_upload",
        original_filename=original_filename, recorded_by=reviewer.display_name,
    )  # fmt: skip
    base = _base_item(record, previous)
    base["attempts"] = [
        *(previous.attempts if previous else []),
        RetrievalAttempt(service="user_upload", outcome="retrieved", detail=original_filename),
    ]
    item = FullTextItem.model_validate(
        base
        | {
            "pmcid": record.pmcid or (previous.pmcid if previous else None),
            "status": FullTextStatus.USER_UPLOADED,
            "pdf_filename": stored.filename,
            "pdf_sha256": stored.sha256,
            "source_kind": "user_upload",
            "match_status": status,
            "match_notes": notes,
            "recorded_by": reviewer.display_name,
            "xml_filename": previous.xml_filename if previous else None,
            "xml_sha256": previous.xml_sha256 if previous else None,
        }
    )
    repo.save_fulltext_item(project_id, item)
    return item


def confirm_pdf_match(
    repo: Repository, project_id: str, record_id: str, *, reviewer: ReviewerORM, note: str
) -> FullTextItem:
    """A human confirms that the linked PDF really is this report (uncertain/mismatch)."""
    reviewer = repo.require_active_reviewer(reviewer.id)
    if not note.strip():
        raise ValueError("a confirmation note is required (what was checked)")
    item = repo.get_fulltext_item(project_id, record_id)
    if item is None or item.pdf_sha256 is None:
        raise NotFoundError(f"record {record_id} has no linked PDF to confirm")
    confirmed = FullTextItem.model_validate(
        item.model_dump()
        | {
            "match_status": UploadMatchStatus.HUMAN_CONFIRMED,
            "confirmed_by": reviewer.display_name,
            "match_notes": [*item.match_notes, f"confirmed by {reviewer.display_name}: {note}"],
        }
    )
    repo.save_fulltext_item(project_id, confirmed)
    return confirmed


def mark_unobtainable(
    repo: Repository, project_id: str, record_id: str, *, reviewer: ReviewerORM, reason: str
) -> FullTextItem:
    """Record that a report could not be obtained (PRISMA "not retrieved", not excluded)."""
    reviewer = repo.require_active_reviewer(reviewer.id)
    record = repo.get_record(project_id, record_id)
    previous = repo.get_fulltext_item(project_id, record_id)
    if previous is not None and previous.is_retrieved:
        raise SysRevError(f"record {record_id} already has a retrieved full text")
    item = FullTextItem.model_validate(
        _base_item(record, previous)
        | {
            "status": FullTextStatus.UNOBTAINABLE,
            "unobtainable_reason": reason,
            "recorded_by": reviewer.display_name,
        }
    )
    repo.save_fulltext_item(project_id, item)
    return item
