"""Is identification complete and prepared for screening? (pure functions over the log)

The search log is append-only, so the *current* state of a strategy is derived from the
chronological sequence of its entries:

* a successful execution or import (records identified, including a recorded zero-result
  search) makes the strategy **searched** and supersedes any earlier "not searched"
  declaration, which stays in the log;
* a "not searched" declaration makes it **not searched** until superseded; a later
  declaration replaces an earlier one (latest reason applies);
* a failed attempt identifies nothing and never changes the status.

Imports of one strategy (a large export may be split over several files) are reconciled
against the total the reviewer read in the vendor interface: *matched*, *mismatch*,
*unknown* (no total recorded) or *conflicting* (different totals declared). Anything but a
match needs an authenticated acknowledgement recorded **after the last import** and bound
to the exact counts; a further import therefore invalidates an earlier acknowledgement.

Used by :meth:`Repository.build_guard_context` (Stage 2 → Gate 1.5), the AI screening step,
the UI and the PRISMA-S report, so all of them agree.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from src.core.enums import SearchRunKind, SearchRunStatus
from src.core.schemas.protocol import PICOProtocol
from src.core.schemas.search import SearchRun

LOG_KINDS = frozenset(
    {
        SearchRunKind.EXECUTION,
        SearchRunKind.IMPORT,
        SearchRunKind.NOT_SEARCHED,
        SearchRunKind.DEDUPLICATION,
    }
)
StrategyStatus = Literal["searched", "not_searched", "not_yet_searched"]
ReconciliationState = Literal["no_imports", "matched", "mismatch", "unknown", "conflicting"]


@dataclass(frozen=True, slots=True)
class ImportTotals:
    """Reconciliation of all imports of one strategy with the vendor-reported total."""

    files: int
    records: int
    declared: tuple[int, ...]  # distinct totals declared across the imports, in order
    last_import_index: int | None

    @property
    def reported(self) -> int | None:
        """The vendor total, only when exactly one value was declared (never inferred)."""
        return self.declared[0] if len(self.declared) == 1 else None

    @property
    def state(self) -> ReconciliationState:
        if self.last_import_index is None:
            return "no_imports"
        if not self.declared:
            return "unknown"
        if len(self.declared) > 1:
            return "conflicting"
        return "matched" if self.declared[0] == self.records else "mismatch"

    @property
    def needs_acknowledgement(self) -> bool:
        return self.state in ("mismatch", "unknown", "conflicting")

    def ack_summary(self) -> dict[str, object]:
        """What an acknowledgement must record to stay valid for these totals."""
        return {
            "records": self.records,
            "reported": self.reported,
            "declared": list(self.declared),
            "state": self.state,
        }


def import_totals(runs: Sequence[SearchRun], strategy_sha256: str) -> ImportTotals:
    idx = [
        i
        for i, r in enumerate(runs)
        if r.kind == SearchRunKind.IMPORT
        and r.strategy_sha256 == strategy_sha256
        and r.identifies_records
    ]
    declared: list[int] = []
    for i in idx:
        value = runs[i].hits_reported
        if value is not None and value not in declared:
            declared.append(value)
    return (
        ImportTotals(
            files=sum(1 for i in idx if any(f.role == "import_file" for f in runs[i].files)),
            records=sum(runs[i].records_saved for i in idx),
            declared=tuple(declared),
            last_import_index=idx[-1] if idx else None,
        )
        if idx
        else ImportTotals(0, 0, (), None)
    )


def valid_import_acknowledgement(
    runs: Sequence[SearchRun], sha: str, totals: ImportTotals
) -> SearchRun | None:
    """The latest acknowledgement recorded after the last import and matching its counts."""
    if totals.last_import_index is None:
        return None
    expected = totals.ack_summary()
    for r in reversed(runs[totals.last_import_index + 1 :]):
        if (
            r.kind == SearchRunKind.ACKNOWLEDGEMENT
            and r.strategy_sha256 == sha
            and r.refers_to_run_id is not None
            and all(r.summary.get(k) == v for k, v in expected.items())
        ):
            return r
    return None


@dataclass(frozen=True, slots=True)
class StrategyState:
    strategy_sha256: str
    status: StrategyStatus
    executions: int
    executed_records: int
    imports: ImportTotals
    failed_attempts: int
    declarations: tuple[SearchRun, ...]
    effective_declaration: SearchRun | None
    contradictory_declarations: int  # declared "not searched" after records were identified
    import_acknowledgement: SearchRun | None

    @property
    def detail(self) -> str:
        parts: list[str] = []
        if self.status == "not_searched" and self.effective_declaration is not None:
            parts.append(
                f"not searched (declared by {self.effective_declaration.actor}: "
                f"{self.effective_declaration.note})"
            )
        if self.executions:
            parts.append(f"executed ({self.executed_records} records)")
        if self.imports.files or self.imports.last_import_index is not None:
            parts.append(
                f"imported ({self.imports.records} records in {self.imports.files} "
                f"file(s); vendor total: {_reconciliation_text(self.imports)})"
            )
        if self.status == "searched" and self.declarations:
            parts.append(
                f"{len(self.declarations)} earlier 'not searched' declaration(s) superseded"
            )
        if self.failed_attempts:
            parts.append(f"{self.failed_attempts} failed attempt(s), nothing identified")
        if self.status == "not_yet_searched" and not parts:
            parts.append("not yet searched")
        elif self.status == "not_yet_searched":
            parts.insert(0, "not yet searched")
        return "; ".join(parts)


def _reconciliation_text(t: ImportTotals) -> str:
    return {
        "no_imports": "no imports",
        "matched": f"{t.reported} reported, matched",
        "mismatch": f"{t.reported} reported but {t.records} imported",
        "unknown": "not recorded (unknown)",
        "conflicting": "conflicting totals declared: " + ", ".join(map(str, t.declared)),
    }[t.state]


def strategy_state(runs: Sequence[SearchRun], strategy_sha256: str) -> StrategyState:
    """Current state of one strategy from the chronological sequence of its log entries."""
    status: StrategyStatus = "not_yet_searched"
    declarations: list[SearchRun] = []
    effective: SearchRun | None = None
    contradictory = executions = executed_records = failed = 0
    for r in runs:
        if r.strategy_sha256 != strategy_sha256:
            continue
        if r.kind == SearchRunKind.NOT_SEARCHED:
            declarations.append(r)
            if status == "searched":
                contradictory += 1  # records exist; the declaration cannot undo that
            else:
                status, effective = "not_searched", r
        elif r.kind in (SearchRunKind.EXECUTION, SearchRunKind.IMPORT):
            if not r.identifies_records:
                failed += 1
                continue
            status, effective = "searched", None
            if r.kind == SearchRunKind.EXECUTION:
                executions += 1
                executed_records += r.records_saved
    totals = import_totals(runs, strategy_sha256)
    return StrategyState(
        strategy_sha256=strategy_sha256,
        status=status,
        executions=executions,
        executed_records=executed_records,
        imports=totals,
        failed_attempts=failed,
        declarations=tuple(declarations),
        effective_declaration=effective,
        contradictory_declarations=contradictory,
        import_acknowledgement=valid_import_acknowledgement(runs, strategy_sha256, totals),
    )


def run_effects(runs: Sequence[SearchRun]) -> dict[str, str]:
    """Current effect of each log entry, for the audit-trail views and exports."""
    effects: dict[str, str] = {}
    by_sha: dict[str | None, list[tuple[int, SearchRun]]] = {}
    for i, r in enumerate(runs):
        by_sha.setdefault(r.strategy_sha256, []).append((i, r))
    for sha, entries in by_sha.items():
        totals = import_totals(runs, sha) if sha else None
        valid_ack = valid_import_acknowledgement(runs, sha, totals) if sha and totals else None
        for pos, (_, r) in enumerate(entries):
            later = [x for _, x in entries[pos + 1 :]]
            if r.kind == SearchRunKind.NOT_SEARCHED:
                earlier_success = any(x.identifies_records for _, x in entries[:pos])
                if earlier_success:
                    effects[r.run_id] = "contradicts earlier identified records (not in effect)"
                elif any(x.identifies_records for x in later):
                    effects[r.run_id] = "superseded by a later successful search/import"
                elif any(x.kind == SearchRunKind.NOT_SEARCHED for x in later):
                    effects[r.run_id] = "replaced by a later declaration"
                else:
                    effects[r.run_id] = "in effect"
            elif r.kind in (SearchRunKind.EXECUTION, SearchRunKind.IMPORT):
                effects[r.run_id] = (
                    "records identified"
                    if r.identifies_records
                    else "failed attempt: nothing identified, not a search"
                )
            elif r.kind == SearchRunKind.ACKNOWLEDGEMENT:
                if r.summary.get("state") is None and "records" not in r.summary:
                    effects[r.run_id] = "acknowledges an execution count mismatch"
                else:
                    effects[r.run_id] = (
                        "valid for the current imports"
                        if r is valid_ack
                        else "no longer applies (imports changed after it)"
                    )
    return effects


def identification_problems(
    runs: Sequence[SearchRun], protocol: PICOProtocol | None, unattributed_records: int
) -> list[str]:
    if not any(r.kind in LOG_KINDS for r in runs):
        return []
    problems: list[str] = []
    for s in protocol.search_strategies if protocol else []:
        state = strategy_state(runs, s.sha256)
        label = f"{s.database} strategy {s.sha256[:12]}"
        if state.status == "not_yet_searched":
            failed = (
                f" ({state.failed_attempts} failed attempt(s) identified nothing)"
                if state.failed_attempts
                else ""
            )
            problems.append(
                f"no execution or import is recorded for the {label}{failed} (or record it "
                "as not searched, with a reason)"
            )
        if state.contradictory_declarations:
            problems.append(
                f"{label}: a 'not searched' declaration was recorded after records "
                "had been identified; the log is contradictory"
            )
        t = state.imports
        if t.needs_acknowledgement and state.import_acknowledgement is None:
            reason = {
                "mismatch": f"the interface reported {t.reported} records but {t.records} "
                "were imported (import the remaining export files, or",
                "unknown": f"the database-reported total was not recorded for the {t.records} "
                "imported records, so completeness cannot be verified (record "
                "the total with the import, or",
                "conflicting": "different totals were declared for the imports ("
                + ", ".join(map(str, t.declared))
                + ") (",
            }[t.state]
            problems.append(f"{label}: {reason} a reviewer must acknowledge it with a reason)")
    acknowledged = {r.refers_to_run_id for r in runs if r.kind == SearchRunKind.ACKNOWLEDGEMENT}
    for r in runs:
        if (
            r.kind == SearchRunKind.EXECUTION
            and r.status == SearchRunStatus.COUNT_MISMATCH
            and r.run_id not in acknowledged
        ):
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
