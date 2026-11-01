"""Stage 3: lawful open-access full-text retrieval, uploads and the missing-PDF manifest."""

from src.fulltext.download import DownloadLimits, DownloadResult, PDFDownloader
from src.fulltext.manifest import (
    ManifestRow,
    ReportFlow,
    build_manifest,
    manifest_csv,
    manifest_json,
    report_flow,
)
from src.fulltext.matching import MatchAssessment, assess_match
from src.fulltext.retrieval import (
    FullTextRetriever,
    RetrievalSummary,
    StaleSourceError,
    UploadRejectedError,
    acknowledge_xml_only,
    confirm_pdf_match,
    mark_unobtainable,
    register_upload,
)
from src.fulltext.storage import DocumentStore, StoredFile

__all__ = [
    "DocumentStore",
    "DownloadLimits",
    "DownloadResult",
    "FullTextRetriever",
    "ManifestRow",
    "MatchAssessment",
    "PDFDownloader",
    "ReportFlow",
    "RetrievalSummary",
    "StaleSourceError",
    "StoredFile",
    "UploadRejectedError",
    "acknowledge_xml_only",
    "assess_match",
    "build_manifest",
    "confirm_pdf_match",
    "manifest_csv",
    "manifest_json",
    "mark_unobtainable",
    "register_upload",
    "report_flow",
]
