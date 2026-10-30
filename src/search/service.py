"""Stage 2 identification: execute approved searches, import exports, prepare for screening.

Every action is attributed to an authenticated reviewer and written to the append-only
search log *after* it happened. Searches run only in Stage 2 (after Gate 1 approved the
protocol), only for strategies of the approved protocol, and only before screening starts:
adding records after screening has begun would silently change the screening pool.

* **API execution** (PubMed, Europe PMC, OpenAlex) sends the strategy exactly as approved
  (filters appended as ``AND (...)`` clauses, date limits as documented API parameters),
  preserves every raw response body with its SHA-256, and compares the records retrieved
  with the count the database reported. PubMed searches above 10,000 records are
  partitioned by publication date; an incomplete union aborts the run.
* **Import** (any selectable source, required for subscription databases) preserves the
  original file with its SHA-256 and each raw entry, refuses a file already imported, and
  checks the records imported against the count the reviewer read in the interface.
* **Prepare for screening** runs the deterministic deduplicator over all records, stores
  ambiguous pairs for human adjudication and records the PRISMA snapshot.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, Literal

import httpx

from src.apis.europepmc import EUROPEPMC_BASE, EuropePMCClient, parse_europepmc_result
from src.apis.openalex import OPENALEX_BASE, OpenAlexClient, parse_openalex_work
from src.apis.pubmed import EUTILS_BASE, PubMedClient
from src.core.config import Settings, get_settings
from src.core.enums import ActorType, Phase, SearchRunKind, SearchRunStatus, SourceDatabase
from src.core.exceptions import APIClientError, SysRevError
from src.core.schemas.base import new_id
from src.core.schemas.protocol import PICOProtocol, SearchStrategy
from src.core.schemas.search import DatePartition, SearchFileRef, SearchRun
from src.core.schemas.study import SearchProvenance, StudyRecord
from src.fulltext.storage import DocumentStore
from src.screening.deduplication import Deduplicator
from src.search.databases import DATABASES, info, is_api
from src.search.importers import ImportFileError, ImportFormat, ParsedEntry, parse_file
from src.search.prisma_sync import derive_prisma, identification_by_source
from src.search.protocol_editor import Reviewer, record_strategy_validation
from src.state.repository import Repository

MAX_DIRECT_RECORDS = 50_000  # larger API result sets must be narrowed or split
# Shared DOI/PMID but titles below this token-set similarity: flagged for a human instead of
# merged (a wrong identifier in an export must not silently delete a different report).
EXACT_MATCH_TITLE_FLOOR = 60.0  # engineering heuristic; not validated on a labelled corpus


class SearchError(SysRevError):
    """The search action is not allowed or not possible in the current state."""


@dataclass(slots=True)
class SearchClients:
    """Factories for the existing literature clients (injectable for tests)."""

    pubmed: Callable[[], PubMedClient]
    europepmc: Callable[[], EuropePMCClient]
    openalex: Callable[[], OpenAlexClient]

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> SearchClients:
        s = settings or get_settings()
        common: dict[str, Any] = {
            "timeout": s.http_timeout_seconds,
            "max_retries": s.http_max_retries,
        }

        def pubmed() -> PubMedClient:
            if not s.ncbi_email:
                raise SearchError("set SYSREV_NCBI_EMAIL (NCBI requires a contact e-mail)")
            key = s.ncbi_api_key.get_secret_value() if s.ncbi_api_key else None
            return PubMedClient(email=s.ncbi_email, tool=s.ncbi_tool, api_key=key, **common)

        def europepmc() -> EuropePMCClient:
            return EuropePMCClient(requests_per_second=s.europepmc_requests_per_second, **common)

        def openalex() -> OpenAlexClient:
            key = s.openalex_api_key.get_secret_value() if s.openalex_api_key else None
            return OpenAlexClient(
                mailto=s.openalex_mailto,
                api_key=key,
                requests_per_second=s.openalex_requests_per_second,
                **common,
            )

        return cls(pubmed, europepmc, openalex)


@dataclass(frozen=True, slots=True)
class ExecutableQuery:
    """The exact request derived from a strategy (stored verbatim in the search log)."""

    query: str
    parameters: dict[str, str]
    endpoint: str


def executable_query(strategy: SearchStrategy) -> ExecutableQuery:
    """Translate a strategy into the documented request of its API (no rewriting)."""
    db = strategy.database
    if not is_api(db):
        raise SearchError(f"{info(db).label} is not searched through an API; import an export")
    if strategy.languages:
        raise SearchError(
            "language limits are not translated automatically: write them into the query "
            "(PubMed/Europe PMC) or as an OpenAlex filter, and clear the language list"
        )
    if db == SourceDatabase.PUBMED:
        query = " AND ".join(
            [
                f"({strategy.query})" if strategy.filters else strategy.query,
                *(f"({f})" for f in strategy.filters),
            ]
        )
        params: dict[str, str] = {}
        if strategy.date_from or strategy.date_to:
            params = {
                "datetype": "pdat",
                "mindate": strategy.date_from.strftime("%Y/%m/%d")
                if strategy.date_from
                else "1800",
                "maxdate": strategy.date_to.strftime("%Y/%m/%d") if strategy.date_to else "3000",
            }
        return ExecutableQuery(query, params, f"{EUTILS_BASE}esearch.fcgi")
    if db == SourceDatabase.EUROPE_PMC:
        clauses = [*(f"({f})" for f in strategy.filters)]
        if strategy.date_from or strategy.date_to:
            lo = strategy.date_from.isoformat() if strategy.date_from else "1800-01-01"
            hi = strategy.date_to.isoformat() if strategy.date_to else "3000-12-31"
            clauses.append(f"(FIRST_PDATE:[{lo} TO {hi}])")
        query = " AND ".join([f"({strategy.query})" if clauses else strategy.query, *clauses])
        return ExecutableQuery(query, {}, f"{EUROPEPMC_BASE}search")
    filters = list(strategy.filters)
    if strategy.date_from:
        filters.append(f"from_publication_date:{strategy.date_from.isoformat()}")
    if strategy.date_to:
        filters.append(f"to_publication_date:{strategy.date_to.isoformat()}")
    params = {"filter": ",".join(filters)} if filters else {}
    return ExecutableQuery(strategy.query, params, f"{OPENALEX_BASE}works")


def protocol_restrictions(strategy: SearchStrategy) -> dict[str, Any]:
    """The restrictions a protocol strategy defines besides its query (as written)."""
    return {
        "date_from": strategy.date_from.isoformat() if strategy.date_from else None,
        "date_to": strategy.date_to.isoformat() if strategy.date_to else None,
        "languages": list(strategy.languages),
        "filters": list(strategy.filters),
    }


def _has_restrictions(r: dict[str, Any]) -> bool:
    return bool(r["date_from"] or r["date_to"] or r["languages"] or r["filters"])


class _ResponseCapture:
    """Collects raw response bodies (``api_key`` removed from the recorded URL)."""

    def __init__(self) -> None:
        self.bodies: list[tuple[str, bytes, str]] = []

    def __call__(self, response: httpx.Response) -> None:
        url = response.request.url
        clean = url.copy_remove_param("api_key") if "api_key" in url.params else url
        suffix = ".xml" if "xml" in response.headers.get("content-type", "") else ".json"
        self.bodies.append((f"{response.request.method} {clean}", response.content, suffix))


def _with_run(record: StudyRecord, run_id: str, file_sha: str | None = None) -> StudyRecord:
    assert record.provenance is not None
    prov = record.provenance.model_copy(
        update={"search_run_id": run_id, "source_file_sha256": file_sha}
    )
    return record.model_copy(update={"provenance": prov})


class SearchService:
    def __init__(
        self,
        repo: Repository,
        project_id: str,
        *,
        store: DocumentStore,
        clients: SearchClients | None = None,
        dedup_title_threshold: float = 95.0,
    ) -> None:
        self.repo = repo
        self.project_id = project_id
        self.store = store
        self._clients = clients
        self.dedup_title_threshold = dedup_title_threshold

    @property
    def clients(self) -> SearchClients:
        if self._clients is None:
            self._clients = SearchClients.from_settings()
        return self._clients

    # ------------------------------------------------------------------ state checks
    def _phase(self) -> Phase:
        return Phase(self.repo.get_project(self.project_id).phase)

    def _require_identification_open(self) -> None:
        phase = self._phase()
        if phase != Phase.STAGE2_SEARCH_SCREENING:
            raise SearchError(
                f"searches are executed and imported in {Phase.STAGE2_SEARCH_SCREENING}, after "
                f"Gate 1 approved the protocol (project is in {phase})"
            )
        if self.repo.load_screening_decisions(self.project_id):
            raise SearchError(
                "screening has started: adding or re-deduplicating records now would change "
                "the screening pool (update searches are not supported yet)"
            )

    def _approved_strategy(self, sha: str) -> tuple[PICOProtocol, SearchStrategy]:
        protocol = self.repo.latest_protocol(self.project_id)
        if protocol is None or protocol.status != "approved":
            raise SearchError("the current protocol version is not approved")
        strategy = protocol.strategy(sha)
        if strategy is None:
            raise SearchError(f"strategy {sha[:12]} is not part of the approved protocol")
        return protocol, strategy

    def _store_responses(self, capture: _ResponseCapture) -> list[SearchFileRef]:
        refs = []
        for request, body, suffix in capture.bodies:
            stored = self.store.put_bytes(body, suffix)
            refs.append(
                SearchFileRef(
                    role="api_response",
                    filename=stored.filename,
                    sha256=stored.sha256,
                    size_bytes=stored.size_bytes,
                    request=request,
                )
            )
        return refs

    def _base(self, reviewer: Reviewer, **kw: Any) -> dict[str, Any]:
        active = self.repo.require_active_reviewer(reviewer.id)
        return {
            "actor": active.display_name,
            "actor_type": ActorType.HUMAN,
            "reviewer_id": active.id,
            **kw,
        }

    # ------------------------------------------------------------------ preview
    async def preview(self, sha: str, *, reviewer: Reviewer) -> SearchRun:
        """Count-only search. In Stage 1 a clean preview marks the strategy syntax-validated."""
        phase = self._phase()
        protocol = self.repo.latest_protocol(self.project_id)
        strategy = protocol.strategy(sha) if protocol else None
        if protocol is None or strategy is None:
            raise SearchError(f"strategy {sha[:12]} is not in the current protocol")
        if phase not in (Phase.STAGE1_PROTOCOL, Phase.STAGE2_SEARCH_SCREENING):
            raise SearchError("previews are available in Stage 1 and Stage 2")
        plan = executable_query(strategy)
        capture = _ResponseCapture()
        base = self._base(
            reviewer,
            kind=SearchRunKind.PREVIEW,
            database=strategy.database,
            strategy_sha256=sha,
            protocol_version=protocol.version,
            platform=strategy.platform,
            query=plan.query,
            parameters=plan.parameters,
            endpoint=plan.endpoint,
            protocol_restrictions=protocol_restrictions(strategy),
            restrictions_applied=(
                "applied_by_system"
                if _has_restrictions(protocol_restrictions(strategy))
                else "none_defined"
            ),
        )
        errors: list[str] = []
        warnings: list[str] = []
        translation = None
        try:
            if strategy.database == SourceDatabase.PUBMED:
                async with self.clients.pubmed() as client:
                    client.response_hook = capture
                    c = await client.count(
                        plan.query,
                        mindate=plan.parameters.get("mindate"),
                        maxdate=plan.parameters.get("maxdate"),
                    )
                count, translation, errors, warnings = (
                    c.count,
                    c.query_translation,
                    c.errors,
                    c.warnings,
                )
            elif strategy.database == SourceDatabase.EUROPE_PMC:
                async with self.clients.europepmc() as client:
                    client.response_hook = capture
                    count = await client.count(plan.query)
            else:
                async with self.clients.openalex() as client:
                    client.response_hook = capture
                    count = await client.count(
                        search=plan.query, filter_=plan.parameters.get("filter")
                    )
        except APIClientError as exc:
            run = SearchRun(**base, status=SearchRunStatus.FAILED, problems=[str(exc)[:500]])
            self.repo.add_search_run(self.project_id, run)
            return run
        run = SearchRun(
            **base,
            hits_reported=count,
            query_translation=translation,
            warnings=[*warnings, *(f"error: {e}" for e in errors)],
            files=self._store_responses(capture),
        )
        self.repo.add_search_run(self.project_id, run)
        if phase == Phase.STAGE1_PROTOCOL and protocol.status == "draft" and not errors:
            method: Literal["api_preview", "api_accepted"] = (
                "api_preview" if strategy.database == SourceDatabase.PUBMED else "api_accepted"
            )
            note = f"preview {run.run_id}: {count} records"
            record_strategy_validation(
                self.repo, self.project_id, sha, method=method, reviewer=reviewer, note=note
            )
        return run

    # ------------------------------------------------------------------ execution
    async def execute(self, sha: str, *, reviewer: Reviewer) -> SearchRun:
        self._require_identification_open()
        protocol, strategy = self._approved_strategy(sha)
        if any(
            r.strategy_sha256 == sha and r.kind == SearchRunKind.EXECUTION and r.identifies_records
            for r in self.repo.load_search_runs(self.project_id)
        ):
            raise SearchError("this strategy was already executed; re-running would duplicate it")
        plan = executable_query(strategy)
        run_id = new_id()
        capture = _ResponseCapture()
        base = self._base(
            reviewer,
            run_id=run_id,
            kind=SearchRunKind.EXECUTION,
            database=strategy.database,
            strategy_sha256=sha,
            protocol_version=protocol.version,
            platform=strategy.platform,
            query=plan.query,
            parameters=plan.parameters,
            endpoint=plan.endpoint,
            protocol_restrictions=protocol_restrictions(strategy),
            restrictions_applied=(
                "applied_by_system"
                if _has_restrictions(protocol_restrictions(strategy))
                else "none_defined"
            ),
        )
        try:
            if strategy.database == SourceDatabase.PUBMED:
                records, extra = await self._run_pubmed(plan, strategy, capture)
            elif strategy.database == SourceDatabase.EUROPE_PMC:
                records, extra = await self._run_europepmc(plan, capture)
            else:
                records, extra = await self._run_openalex(plan, capture)
        except (APIClientError, SearchError) as exc:
            run = SearchRun(
                **base,
                status=SearchRunStatus.FAILED,
                problems=[str(exc)[:1000]],
                files=self._store_responses(capture),
            )
            self.repo.add_search_run(self.project_id, run)
            return run
        records = [_with_run(r, run_id) for r in records]
        problems = list(extra.pop("problems"))
        status = SearchRunStatus.COUNT_MISMATCH if problems else SearchRunStatus.COMPLETE
        run = SearchRun(
            **base,
            **extra,
            status=status,
            problems=problems,
            records_saved=len(records),
            files=self._store_responses(capture),
        )
        self.repo.add_search_run(self.project_id, run)
        self.repo.save_records(self.project_id, records)
        return run

    async def _run_pubmed(
        self, plan: ExecutableQuery, strategy: SearchStrategy, capture: _ResponseCapture
    ) -> tuple[list[StudyRecord], dict[str, Any]]:
        async with self.clients.pubmed() as client:
            client.response_hook = capture
            result = await client.esearch_all(
                plan.query, date_from=strategy.date_from, date_to=strategy.date_to
            )
            if result.count > MAX_DIRECT_RECORDS:
                raise SearchError(
                    f"{result.count} records exceed the direct-retrieval limit "
                    f"of {MAX_DIRECT_RECORDS}; narrow or split the strategy"
                )
            records = await client.efetch_records(
                result.pmids, query=plan.query, total_hits=result.count
            )
        problems = []
        if len(records) != result.count:
            missing = sorted(set(result.pmids) - {r.pmid for r in records if r.pmid})
            problems.append(
                f"PubMed reported {result.count} records; {len(records)} were parsed from "
                f"EFetch (not returned or untitled PMIDs: {', '.join(missing[:20])}"
                f"{' …' if len(missing) > 20 else ''})"
            )
        extra: dict[str, Any] = {
            "hits_reported": result.count,
            "records_received": len(result.pmids),
            "query_translation": result.query_translation,
            "problems": problems,
            "partitions": [
                DatePartition(mindate=a, maxdate=b, count=n)
                for a, b, n in result.partitions
                if len(result.partitions) > 1
            ],
        }
        return records, extra

    async def _run_europepmc(
        self, plan: ExecutableQuery, capture: _ResponseCapture
    ) -> tuple[list[StudyRecord], dict[str, Any]]:
        executed_at = datetime.now(UTC)
        async with self.clients.europepmc() as client:
            client.response_hook = capture
            total = await client.count(plan.query)
            if total > MAX_DIRECT_RECORDS:
                raise SearchError(
                    f"{total} records exceed the direct-retrieval limit of "
                    f"{MAX_DIRECT_RECORDS}; narrow or split the strategy"
                )
            received, records, untitled = 0, [], []
            async for hits, raw in client.iter_results(plan.query):
                received += 1
                total = hits
                rec = parse_europepmc_result(
                    raw, query=plan.query, executed_at=executed_at, total_hits=hits
                )
                if rec is None:
                    untitled.append(f"{raw.get('source', '?')}:{raw.get('id', '?')}")
                else:
                    records.append(rec)
        return records, self._api_counts(total, received, records, untitled)

    async def _run_openalex(
        self, plan: ExecutableQuery, capture: _ResponseCapture
    ) -> tuple[list[StudyRecord], dict[str, Any]]:
        executed_at = datetime.now(UTC)
        filter_ = plan.parameters.get("filter")
        label = " | ".join(
            p for p in (f"search={plan.query}", f"filter={filter_}" if filter_ else "") if p
        )
        async with self.clients.openalex() as client:
            client.response_hook = capture
            total = await client.count(search=plan.query, filter_=filter_)
            if total > MAX_DIRECT_RECORDS:
                raise SearchError(
                    f"{total} records exceed the direct-retrieval limit of "
                    f"{MAX_DIRECT_RECORDS}; narrow or split the strategy"
                )
            received, records, untitled = 0, [], []
            async for hits, raw in client.iter_works(search=plan.query, filter_=filter_):
                received += 1
                total = hits
                rec = parse_openalex_work(
                    raw, query=label, executed_at=executed_at, total_hits=hits
                )
                if rec is None:
                    untitled.append(str(raw.get("id", "?")))
                else:
                    records.append(rec)
        return records, self._api_counts(total, received, records, untitled)

    @staticmethod
    def _api_counts(
        total: int, received: int, records: list[StudyRecord], untitled: list[str]
    ) -> dict[str, Any]:
        problems = []
        if received != total:
            problems.append(
                f"the database reported {total} records but {received} were returned while paging"
            )
        if untitled:
            problems.append(
                f"{len(untitled)} result(s) without a title could not be stored: "
                + ", ".join(untitled[:20])
            )
        return {"hits_reported": total, "records_received": received, "problems": problems}

    # ------------------------------------------------------------------ import
    def import_file(
        self,
        sha: str,
        data: bytes,
        *,
        filename: str,
        fmt: ImportFormat,
        searched_on: date,
        reviewer: Reviewer,
        hits_reported: int | None = None,
        column_map: dict[str, str] | None = None,
        note: str | None = None,
        restrictions_confirmed: bool = False,
        restrictions_note: str | None = None,
    ) -> SearchRun:
        """Import one export file. ``hits_reported`` is the total the vendor interface showed
        for the whole search (``None`` = not recorded; then completeness stays unverified
        until a reviewer acknowledges it). ``restrictions_confirmed`` is the reviewer's
        authenticated statement that the protocol's date/language limits and filters were
        applied in the vendor interface."""
        self._require_identification_open()
        protocol, strategy = self._approved_strategy(sha)
        if searched_on > datetime.now(UTC).date():
            raise ImportFileError("the search date cannot be in the future")
        file_sha = hashlib.sha256(data).hexdigest()
        for r in self.repo.load_search_runs(self.project_id, SearchRunKind.IMPORT):
            if any(f.sha256 == file_sha for f in r.files):
                raise ImportFileError(f"this file was already imported (run {r.run_id[:12]})")
        parsed = parse_file(data, fmt, column_map=column_map)
        if not parsed.entries:
            raise ImportFileError("the file contains no entries")
        if parsed.errors:
            raise ImportFileError(
                "every entry needs a title; nothing was imported: " + "; ".join(parsed.errors[:20])
            )
        run_id = new_id()
        searched_at = datetime(searched_on.year, searched_on.month, searched_on.day, tzinfo=UTC)
        label = info(strategy.database).label
        stored = self.store.put_bytes(data, f".{fmt if fmt != 'bibtex' else 'bib'}")
        records: list[StudyRecord] = []
        for e in parsed.entries:
            try:
                records.append(
                    self._record(
                        e,
                        strategy,
                        run_id,
                        file_sha,
                        searched_at,
                        hits_reported,
                        f"manual export from {label} ({fmt}): {filename}",
                    )
                )
            except ValueError as exc:
                raise ImportFileError(f"entry {e.position}: {exc}") from exc
        run = SearchRun(
            **self._base(reviewer, run_id=run_id, kind=SearchRunKind.IMPORT),
            database=strategy.database,
            strategy_sha256=sha,
            protocol_version=protocol.version,
            platform=strategy.platform,
            query=strategy.query,
            endpoint=f"manual export ({label})",
            searched_on=searched_on,
            hits_reported=hits_reported,
            records_received=len(parsed.entries),
            records_saved=len(records),
            import_format=fmt,
            column_map=parsed.column_map,
            encoding=parsed.encoding,
            warnings=[
                *self._import_warnings(strategy, records, hits_reported, restrictions_confirmed),
                *parsed.warnings[:200],
            ],
            **self._restriction_fields(strategy, restrictions_confirmed, restrictions_note),
            files=[
                SearchFileRef(
                    role="import_file",
                    filename=stored.filename,
                    original_name=filename,
                    sha256=stored.sha256,
                    size_bytes=stored.size_bytes,
                )
            ],
            note=note,
        )
        self.repo.add_search_run(self.project_id, run)
        self.repo.save_records(self.project_id, records)
        self.repo.add_raw_records(
            self.project_id,
            run_id,
            [
                (r.record_id, e.position, e.raw)
                for r, e in zip(records, parsed.entries, strict=True)
            ],
        )
        return run

    @staticmethod
    def _record(
        e: ParsedEntry,
        strategy: SearchStrategy,
        run_id: str,
        file_sha: str,
        searched_at: datetime,
        hits: int | None,
        endpoint: str,
    ) -> StudyRecord:
        assert e.title is not None
        return StudyRecord(
            source_database=strategy.database,
            provenance=SearchProvenance(
                database=strategy.database,
                query=strategy.query,
                endpoint=endpoint,
                executed_at=searched_at,
                total_hits_reported=hits,
                source_record_id=e.source_id or f"entry-{e.position}",
                search_run_id=run_id,
                source_file_sha256=file_sha,
            ),
            pmid=e.pmid,
            pmcid=e.pmcid,
            doi=e.doi,
            title=e.title,
            abstract=e.abstract,
            authors=e.authors,
            journal=e.journal,
            publication_year=e.year if e.year and 1800 <= e.year <= 2100 else None,
            publication_types=e.publication_types,
            keywords=e.keywords,
            language=e.language,
            registry_ids=e.registry_ids,
        )

    @staticmethod
    def _restriction_fields(
        strategy: SearchStrategy, confirmed: bool, note: str | None
    ) -> dict[str, Any]:
        restrictions = protocol_restrictions(strategy)
        applied = (
            "none_defined"
            if not _has_restrictions(restrictions)
            else "confirmed_by_reviewer"
            if confirmed
            else "not_confirmed"
        )
        return {
            "protocol_restrictions": restrictions,
            "restrictions_applied": applied,
            "restrictions_note": note,
        }

    @staticmethod
    def _import_warnings(
        strategy: SearchStrategy, records: list[StudyRecord], hits: int | None, confirmed: bool
    ) -> list[str]:
        out = []
        if hits is None:
            out.append(
                "the total reported by the vendor interface was not recorded: "
                "completeness of this export cannot be verified"
            )
        if _has_restrictions(protocol_restrictions(strategy)) and not confirmed:
            out.append(
                "the protocol's restrictions (dates/languages/filters) were not "
                "confirmed as applied in the vendor interface"
            )
        lo = strategy.date_from.year if strategy.date_from else None
        hi = strategy.date_to.year if strategy.date_to else None
        outside = sum(
            1
            for r in records
            if r.publication_year is not None
            and (
                (lo is not None and r.publication_year < lo)
                or (hi is not None and r.publication_year > hi)
            )
        )
        if outside:
            out.append(
                f"{outside} imported record(s) have a publication year outside the "
                "protocol date limits (the limit may not have been applied, or the "
                "database limits on another date field)"
            )
        return out

    def record_zero_results(
        self,
        sha: str,
        *,
        searched_on: date,
        note: str,
        reviewer: Reviewer,
        restrictions_confirmed: bool = False,
        restrictions_note: str | None = None,
    ) -> SearchRun:
        """A manual search that was run and returned no records (nothing to export).

        Recorded as an import of zero records with an explicit vendor total of 0 and the
        reviewer's authenticated attestation, so the database is reported as searched with
        0 results rather than falsely as "not searched".
        """
        self._require_identification_open()
        protocol, strategy = self._approved_strategy(sha)
        if searched_on > datetime.now(UTC).date():
            raise ImportFileError("the search date cannot be in the future")
        label = info(strategy.database).label
        run = SearchRun(
            **self._base(reviewer, kind=SearchRunKind.IMPORT),
            database=strategy.database,
            strategy_sha256=sha,
            protocol_version=protocol.version,
            platform=strategy.platform,
            query=strategy.query,
            endpoint=f"manual search ({label}): zero results attested",
            searched_on=searched_on,
            hits_reported=0,
            records_received=0,
            records_saved=0,
            note=note,
            **self._restriction_fields(strategy, restrictions_confirmed, restrictions_note),
        )
        self.repo.add_search_run(self.project_id, run)
        return run

    # ------------------------------------------------------------------ human declarations
    def record_not_searched(self, sha: str, *, reason: str, reviewer: Reviewer) -> SearchRun:
        self._require_identification_open()
        protocol, strategy = self._approved_strategy(sha)
        from src.search.identification import strategy_state

        if strategy_state(self.repo.load_search_runs(self.project_id), sha).status == "searched":
            raise SearchError(
                "this strategy was already searched; it cannot be declared not searched"
            )
        run = SearchRun(
            **self._base(reviewer, kind=SearchRunKind.NOT_SEARCHED),
            database=strategy.database,
            strategy_sha256=sha,
            protocol_version=protocol.version,
            platform=strategy.platform,
            query=strategy.query,
            note=reason,
        )
        self.repo.add_search_run(self.project_id, run)
        return run

    def acknowledge(
        self,
        *,
        note: str,
        reviewer: Reviewer,
        run_id: str | None = None,
        strategy_sha256: str | None = None,
    ) -> SearchRun:
        """Acknowledge an execution's count mismatch, or a strategy's unreconciled imports
        (shortfall, unknown vendor total or conflicting totals). An import acknowledgement is
        bound to the current counts and becomes invalid when further files are imported."""
        from src.search.identification import import_totals

        phase = self._phase()
        if phase != Phase.STAGE2_SEARCH_SCREENING:
            raise SearchError(f"acknowledgements are recorded in Stage 2 (project is in {phase})")
        runs = self.repo.load_search_runs(self.project_id)
        if run_id is not None:
            target = next((r for r in runs if r.run_id == run_id), None)
            if target is None or target.status != SearchRunStatus.COUNT_MISMATCH:
                raise SearchError("no execution with a count mismatch has this id")
            ack = SearchRun(
                **self._base(reviewer, kind=SearchRunKind.ACKNOWLEDGEMENT),
                refers_to_run_id=run_id,
                database=target.database,
                strategy_sha256=target.strategy_sha256,
                note=note,
                summary={"problems": target.problems},
            )
        elif strategy_sha256 is not None:
            totals = import_totals(runs, strategy_sha256)
            if not totals.needs_acknowledgement:
                raise SearchError(
                    "the imports of this strategy are reconciled with the vendor "
                    "total (or there are none); nothing to acknowledge"
                )
            last = [
                r
                for r in runs
                if r.kind == SearchRunKind.IMPORT and r.strategy_sha256 == strategy_sha256
            ][-1]
            ack = SearchRun(
                **self._base(reviewer, kind=SearchRunKind.ACKNOWLEDGEMENT),
                refers_to_run_id=last.run_id,
                database=last.database,
                strategy_sha256=strategy_sha256,
                note=note,
                summary=totals.ack_summary(),
            )
        else:
            raise SearchError("name the run or the strategy to acknowledge")
        self.repo.add_search_run(self.project_id, ack)
        return ack

    # ------------------------------------------------------------------ preparation
    def prepare_for_screening(self, *, reviewer: Reviewer) -> SearchRun:
        """Deduplicate all identified records and record the PRISMA snapshot."""
        self._require_identification_open()
        if not any(r.identifies_records for r in self.repo.load_search_runs(self.project_id)):
            raise SearchError("no executed or imported search yet")
        # Human adjudications survive re-preparation (e.g. after a further import): they are
        # kept and re-applied, and the same pair is never proposed again.
        resolved = [
            c for c in self.repo.load_duplicate_candidates(self.project_id) if c.is_resolved
        ]
        records = [
            r.model_copy(update={"duplicate_of": None})
            for r in self.repo.load_records(self.project_id)
        ]
        dedup = Deduplicator(
            title_threshold=self.dedup_title_threshold,
            exact_match_title_floor=EXACT_MATCH_TITLE_FLOOR,
        )
        result = dedup.deduplicate(records)
        self.repo.save_records(self.project_id, result.all_records)
        self.repo.delete_unresolved_duplicate_candidates(self.project_id)
        decided = {frozenset((c.record_id_a, c.record_id_b)) for c in resolved}
        for c in resolved:
            self.repo.reapply_duplicate_resolution(self.project_id, c)
        proposed = [
            pd
            for pd in result.possible_duplicates
            if frozenset((pd.record_id_a, pd.record_id_b)) not in decided
        ]
        for pd in proposed:
            self.repo.save_duplicate_candidate(self.project_id, pd.to_candidate())
        by_source = identification_by_source(self.repo, self.project_id)
        run = SearchRun(
            **self._base(reviewer, kind=SearchRunKind.DEDUPLICATION),
            records_saved=0,
            summary={
                "records": len(records),
                "unique": len(self.repo.load_records(self.project_id, unique_only=True)),
                "duplicates_removed": sum(
                    1 for r in self.repo.load_records(self.project_id) if r.duplicate_of
                ),
                "automatic_duplicates": result.duplicates_removed,
                "human_decisions_reapplied": len(resolved),
                "possible_duplicates_for_review": len(proposed),
                "by_source": by_source,
                "match_reasons": dict(sorted(result.match_reasons.items())),
                "settings": {
                    "title_threshold": dedup.title_threshold,
                    "year_tolerance": dedup.year_tolerance,
                    "exact_match_title_floor": dedup.exact_match_title_floor,
                },
            },
        )
        self.repo.add_search_run(self.project_id, run)
        self.repo.save_prisma(self.project_id, derive_prisma(self.repo, self.project_id))
        return run

    def resolve_duplicate(
        self, candidate_id: str, *, resolution: str, reviewer: Reviewer, note: str | None = None
    ) -> Any:
        phase = self._phase()
        if phase not in (Phase.STAGE2_SEARCH_SCREENING, Phase.GATE1_5_SCREENING_CONFLICTS):
            raise SearchError(f"duplicates are adjudicated in Stage 2 / Gate 1.5 (now {phase})")
        resolved = self.repo.resolve_duplicate_candidate(
            self.project_id,
            candidate_id,
            resolution=resolution,
            reviewer_id=reviewer.id,
            note=note,
        )
        self.repo.save_prisma(self.project_id, derive_prisma(self.repo, self.project_id))
        return resolved


__all__ = [
    "DATABASES",
    "MAX_DIRECT_RECORDS",
    "ExecutableQuery",
    "SearchClients",
    "SearchError",
    "SearchService",
    "executable_query",
]
