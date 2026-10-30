"""Is identification complete and prepared for screening? (pure functions over the log)

Used by :meth:`Repository.build_guard_context` (Stage 2 → Gate 1.5), the AI screening
step and the search report. Applies only once a project uses the search log.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.core.enums import SearchRunKind, SearchRunStatus
from src.core.schemas.protocol import PICOProtocol, SearchStrategy
from src.core.schemas.search import SearchRun

LOG_KINDS = frozenset(
    {SearchRunKind.EXECUTION, SearchRunKind.IMPORT, SearchRunKind.NOT_SEARCHED,
     SearchRunKind.DEDUPLICATION}
)  # fmt: skip


@dataclass(frozen=True, slots=True)
class ImportTotals:
    """All imports of one strategy (a large export may be split over several files)."""

    files: int
    records: int
    reported: int | None  # latest count declared from the vendor interface, if any

    @property
    def mismatch(self) -> bool:
        return self.reported is not None and self.reported != self.records


def import_totals(runs: Sequence[SearchRun], strategy_sha256: str) -> ImportTotals:
    imports = [r for r in runs if r.kind == SearchRunKind.IMPORT
               and r.strategy_sha256 == strategy_sha256 and r.identifies_records]  # fmt: skip
    declared = [r.hits_reported for r in imports if r.hits_reported is not None]
    return ImportTotals(
        files=len(imports),
        records=sum(r.records_saved for r in imports),
        reported=declared[-1] if declared else None,
    )


def _acknowledged_import(runs: Sequence[SearchRun], sha: str, totals: ImportTotals) -> bool:
    return any(
        r.kind == SearchRunKind.ACKNOWLEDGEMENT
        and r.strategy_sha256 == sha
        and r.summary.get("records") == totals.records
        and r.summary.get("reported") == totals.reported
        for r in runs
    )


def strategy_status(runs: Sequence[SearchRun], strategy: SearchStrategy) -> str:
    sha = strategy.sha256
    if any(r.kind == SearchRunKind.NOT_SEARCHED and r.strategy_sha256 == sha for r in runs):
        return "not searched (declared by a reviewer)"
    executed = [r for r in runs if r.kind == SearchRunKind.EXECUTION
                and r.strategy_sha256 == sha and r.identifies_records]  # fmt: skip
    totals = import_totals(runs, sha)
    parts = []
    if executed:
        parts.append(f"executed ({sum(r.records_saved for r in executed)} records)")
    if totals.files:
        parts.append(f"imported ({totals.records} records in {totals.files} file(s))")
    return "; ".join(parts) or "not yet searched"


def identification_problems(
    runs: Sequence[SearchRun], protocol: PICOProtocol | None, unattributed_records: int
) -> list[str]:
    if not any(r.kind in LOG_KINDS for r in runs):
        return []
    problems: list[str] = []
    covered = {r.strategy_sha256 for r in runs
               if r.identifies_records or r.kind == SearchRunKind.NOT_SEARCHED}  # fmt: skip
    for s in protocol.search_strategies if protocol else []:
        if s.sha256 not in covered:
            problems.append(
                f"no execution or import is recorded for the {s.database} strategy "
                f"{s.sha256[:12]} (or record it as not searched, with a reason)"
            )
        totals = import_totals(runs, s.sha256)
        if totals.mismatch and not _acknowledged_import(runs, s.sha256, totals):
            problems.append(
                f"{s.database} strategy {s.sha256[:12]}: the interface reported "
                f"{totals.reported} records but {totals.records} were imported (import the "
                "remaining export files, or a reviewer must acknowledge the discrepancy)"
            )
    acknowledged = {r.refers_to_run_id for r in runs if r.kind == SearchRunKind.ACKNOWLEDGEMENT}
    for r in runs:
        if (r.kind == SearchRunKind.EXECUTION and r.status == SearchRunStatus.COUNT_MISMATCH
                and r.run_id not in acknowledged):  # fmt: skip
            problems.append(
                f"{r.database} execution {r.run_id[:12]}: "
                + "; ".join(r.problems)
                + " (a reviewer must acknowledge the discrepancy)"
            )
    identifying = [i for i, r in enumerate(runs) if r.identifies_records]
    dedup = [i for i, r in enumerate(runs) if r.kind == SearchRunKind.DEDUPLICATION]
    if identifying and (not dedup or dedup[-1] < identifying[-1]):
        problems.append(
            "records have not been prepared for screening (deduplicated) since the last "
            "search or import"
        )
    if unattributed_records:
        problems.append(
            f"{unattributed_records} stored record(s) are not attributable to a recorded "
            "search execution or import"
        )
    return problems
