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


class StaleSourceError(SysRevError):
    """A full-text decision was made on a screen showing a PDF that is no longer linked."""


def register_upload(
    repo: Repository,
    store: DocumentStore,
    project_id: str,
    record_id: str,
    data: bytes,
    *,
    original_filename: str,
    reviewer: ReviewerORM,
    replaces_sha256: str | None = None,
) -> FullTextItem:
    """Link a user-supplied PDF to ``record_id`` after validation and an identity check.

    Session 9 (H3): replacing an already linked PDF must name the PDF being replaced
    (``replaces_sha256``, the hash shown to the reviewer), so a screen loaded before another
    replacement cannot overwrite it. After Gate 3 the linked PDF is frozen and the upload is
    refused before anything is stored. In earlier phases a replacement invalidates the active
    arms of the report (kept as history); accepted values that cite the old PDF must then be
    re-verified against the new one before Gate 3 (``source_replacement_problems``).
    """
    reviewer = repo.require_active_reviewer(reviewer.id)
    record = repo.get_record(project_id, record_id)
    repo.require_gate3_open(project_id, f"uploading a PDF for {record_id}")
    previous = repo.get_fulltext_item(project_id, record_id)
    old_sha = previous.pdf_sha256 if previous is not None else None
    if old_sha is not None and replaces_sha256 != old_sha:
        raise StaleSourceError(
            f"{record_id} already has a linked PDF ({old_sha[:12]}…); a replacement must name "
            "that PDF as the one it replaces (it was "
            + ("not named" if replaces_sha256 is None else f"{replaces_sha256[:12]}…")
            + "): the linked PDF has changed since it was displayed, or the replacement was not "
            "bound to it - reload, then upload again; nothing was stored"
        )
    stored, problem = validate_pdf_bytes(data, store)
    if stored is None:
        raise UploadRejectedError(f"{original_filename}: {problem}")
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
            "recorded_by_reviewer_id": reviewer.id,
            "xml_filename": previous.xml_filename if previous else None,
            "xml_sha256": previous.xml_sha256 if previous else None,
        }
    )
    repo.save_fulltext_item(project_id, item)
    if old_sha is not None and old_sha != stored.sha256:
        invalidate_after_replacement(repo, project_id, record_id, old_sha, stored.sha256,
                                     reviewer=reviewer)  # fmt: skip
    return item


def invalidate_after_replacement(
    repo: Repository, project_id: str, record_id: str, old_sha: str, new_sha: str, *,
    reviewer: ReviewerORM,
) -> list[str]:  # fmt: skip
    """Session 9 (H3): the arms of a report rest on its linked PDF. When that PDF is replaced,
    every active arm of the report is invalidated (the rows stay stored as history) and the
    replacement is recorded; the arms must be rebuilt from values verified against the new
    PDF and approved again."""
    arms = [a.arm_id for a in repo.load_extracted_arms(project_id) if a.record_id == record_id]
    for arm_id in arms:
        repo.invalidate_arm(
            project_id, arm_id, reviewer_id=reviewer.id, item_id=None,
            reason=f"source PDF replaced ({old_sha[:12]}… → {new_sha[:12]}…)",
            kind="source_replaced",
        )  # fmt: skip
    repo.add_governance_record(
        project_id, "source_replacement", record_id,
        {"record_id": record_id, "old_sha256": old_sha, "new_sha256": new_sha,
         "invalidated_arms": arms, "reviewer_id": reviewer.id},
        reviewer_id=reviewer.id,
    )  # fmt: skip
    return arms


def confirm_pdf_match(
    repo: Repository,
    project_id: str,
    record_id: str,
    *,
    reviewer: ReviewerORM,
    note: str,
    expected_pdf_sha256: str | None = None,
) -> FullTextItem:
    """A human confirms that the linked PDF really is this report (uncertain/mismatch).

    Session 9 (H3): ``expected_pdf_sha256`` binds the confirmation to the PDF the reviewer
    examined (required over HTTP); a confirmation from a screen showing a PDF that has since
    been replaced is refused with nothing written."""
    reviewer = repo.require_active_reviewer(reviewer.id)
    if not note.strip():
        raise ValueError("a confirmation note is required (what was checked)")
    repo.require_gate3_open(project_id, f"confirming the identity of the PDF of {record_id}")
    item = repo.get_fulltext_item(project_id, record_id)
    if item is None or item.pdf_sha256 is None:
        raise NotFoundError(f"record {record_id} has no linked PDF to confirm")
    if expected_pdf_sha256 is not None and expected_pdf_sha256 != item.pdf_sha256:
        raise StaleSourceError(
            f"the PDF linked to {record_id} is now {item.pdf_sha256[:12]}…, not the "
            f"{expected_pdf_sha256[:12]}… that was examined: the linked PDF has changed since it "
            "was displayed - reload and check the current PDF (nothing was saved)"
        )
    confirmed = FullTextItem.model_validate(
        item.model_dump()
        | {
            "match_status": UploadMatchStatus.HUMAN_CONFIRMED,
            "confirmed_by": reviewer.display_name,
            "confirmed_by_reviewer_id": reviewer.id,
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
    repo.require_gate3_open(project_id, f"marking {record_id} unobtainable")
    previous = repo.get_fulltext_item(project_id, record_id)
    if previous is not None and previous.is_retrieved:
        raise SysRevError(f"record {record_id} already has a retrieved full text")
    item = FullTextItem.model_validate(
        _base_item(record, previous)
        | {
            "status": FullTextStatus.UNOBTAINABLE,
            "unobtainable_reason": reason,
            "recorded_by": reviewer.display_name,
            "recorded_by_reviewer_id": reviewer.id,
        }
    )
    repo.save_fulltext_item(project_id, item)
    return item


def acknowledge_xml_only(
    repo: Repository, project_id: str, record_id: str, *, reviewer: ReviewerORM, note: str
) -> FullTextItem:
    """A human records that only structured XML is available (no usable PDF).

    This releases Gate 2 for the report (it was retrieved and can be assessed for
    eligibility), but it does not make the report extraction-ready: numeric evidence must
    cite a physical PDF page, so until a PDF is uploaded the report blocks Gate 3 unless
    it is recorded as unavailable for quantitative synthesis.
    """
    reviewer = repo.require_active_reviewer(reviewer.id)
    if not note.strip():
        raise ValueError("an acknowledgement note is required (e.g. which sources were tried)")
    repo.require_gate3_open(project_id, f"acknowledging XML-only full text for {record_id}")
    item = repo.get_fulltext_item(project_id, record_id)
    if item is None or item.status != FullTextStatus.OA_STRUCTURED_TEXT:
        raise SysRevError(f"record {record_id} is not an XML-only full-text item")
    acknowledged = FullTextItem.model_validate(
        item.model_dump()
        | {
            "xml_only_acknowledged_by": reviewer.display_name,
            "xml_only_acknowledged_reviewer_id": reviewer.id,
            "xml_only_note": note,
        }
    )
    repo.save_fulltext_item(project_id, acknowledged)
    return acknowledged
