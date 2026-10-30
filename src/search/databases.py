"""Which literature sources can be selected, and how each one is searched.

Only PubMed, Europe PMC and OpenAlex are executed directly, through the existing
deterministic clients. The other sources need a subscription, an institutional API
entitlement or their own interface; for them the reviewer runs the protocol strategy in the
vendor interface, exports the results and imports the file (RIS, BibTeX or CSV). The
system never reports such a database as searched unless an import was recorded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.core.enums import SourceDatabase


@dataclass(frozen=True, slots=True)
class DatabaseInfo:
    database: SourceDatabase
    label: str
    access: Literal["api", "manual_import"]
    is_register: bool
    platform: str
    export_formats: tuple[str, ...]
    note: str


DATABASES: dict[SourceDatabase, DatabaseInfo] = {
    d.database: d
    for d in (
        DatabaseInfo(
            SourceDatabase.PUBMED, "PubMed", "api", False, "NCBI E-utilities",
            ("ris", "csv"),
            "Executed directly (ESearch + EFetch). Searches matching > 10,000 records are "
            "partitioned by publication date and verified against PubMed's total count.",
        ),
        DatabaseInfo(
            SourceDatabase.EUROPE_PMC, "Europe PMC", "api", False, "Europe PMC REST API",
            ("ris", "csv"),
            "Executed directly (/search, cursorMark paging).",
        ),
        DatabaseInfo(
            SourceDatabase.OPENALEX, "OpenAlex", "api", False, "OpenAlex Works API",
            ("ris", "csv"),
            "Executed directly (/works, cursor paging). The query is OpenAlex's full-text "
            "'search' parameter; filters are OpenAlex filter expressions.",
        ),
        DatabaseInfo(
            SourceDatabase.EMBASE, "Embase", "manual_import", False, "Embase.com or Ovid",
            ("ris", "csv"),
            "Subscription database; no API access is implemented. Run the strategy in "
            "Embase.com or Ovid, export all results (RIS recommended) and import the file.",
        ),
        DatabaseInfo(
            SourceDatabase.SCOPUS, "Scopus", "manual_import", False, "Scopus (Elsevier)",
            ("ris", "bibtex", "csv"),
            "Subscription database. Elsevier's Scopus Search API needs an API key and an "
            "institutional entitlement and is not implemented; export from the Scopus "
            "interface and import the file.",
        ),
        DatabaseInfo(
            SourceDatabase.WEB_OF_SCIENCE, "Web of Science", "manual_import", False,
            "Web of Science Core Collection (Clarivate)", ("ris", "bibtex"),
            "Subscription database; Clarivate APIs need separate keys and are not "
            "implemented. Export from the Web of Science interface and import the file.",
        ),
        DatabaseInfo(
            SourceDatabase.COCHRANE_CENTRAL, "Cochrane CENTRAL", "manual_import", False,
            "Cochrane Library", ("ris",),
            "No public search API is used. Run the strategy in the Cochrane Library, export "
            "the CENTRAL trials (RIS) and import the file.",
        ),
        DatabaseInfo(
            SourceDatabase.CLINICALTRIALS_GOV, "ClinicalTrials.gov", "manual_import", True,
            "ClinicalTrials.gov", ("csv", "ris"),
            "Trial register (counted in the PRISMA 'registers' box). Direct API search is "
            "not implemented in this version: download the search results as CSV and "
            "import the file.",
        ),
    )
}  # fmt: skip

SELECTABLE = tuple(DATABASES)


def info(database: SourceDatabase) -> DatabaseInfo:
    try:
        return DATABASES[database]
    except KeyError:
        raise ValueError(f"{database} is not a selectable search source") from None


def is_api(database: SourceDatabase) -> bool:
    return info(database).access == "api"
