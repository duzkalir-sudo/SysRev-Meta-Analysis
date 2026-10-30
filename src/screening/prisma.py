"""PRISMA 2020 flow tracker.

Counts are *derived from recorded events* (records identified, deduplication
result, screening outcomes, retrieval outcomes) rather than typed in, and a
validated :class:`PRISMACounts` snapshot is produced on demand. Reference:
Page MJ, McKenzie JE, Bossuyt PM, et al. The PRISMA 2020 statement. BMJ
2021;372:n71. doi:10.1136/bmj.n71
"""

from __future__ import annotations

from collections import Counter

from src.core.enums import SourceDatabase
from src.core.exceptions import PRISMAInconsistencyError
from src.core.schemas.pipeline import PRISMACounts


class PRISMATracker:
    def __init__(self) -> None:
        self._identified: Counter[SourceDatabase] = Counter()
        self._registers = 0
        self._duplicates = 0
        self._automation_ineligible = 0
        self._other_removed = 0
        self._screened_excluded: set[str] = set()
        self._screened_included: set[str] = set()
        self._not_retrieved: set[str] = set()
        self._retrieved: set[str] = set()
        self._fulltext_excluded: dict[str, str] = {}
        self._included_reports: set[str] = set()
        self._studies_included: int | None = None

    # -------------------------------------------------------------- identification
    def record_identified(self, database: SourceDatabase, n: int) -> None:
        if n < 0:
            raise ValueError("n must be non-negative")
        self._identified[database] += n

    def record_register_hits(self, n: int) -> None:
        if n < 0:
            raise ValueError("n must be non-negative")
        self._registers += n

    def record_duplicates_removed(self, n: int) -> None:
        if n < 0:
            raise ValueError("n must be non-negative")
        self._duplicates += n

    def record_automation_ineligible(self, n: int) -> None:
        if n < 0:
            raise ValueError("n must be non-negative")
        self._automation_ineligible += n

    def record_removed_other(self, n: int) -> None:
        if n < 0:
            raise ValueError("n must be non-negative")
        self._other_removed += n

    # -------------------------------------------------------------- screening
    def record_title_abstract_decision(self, record_id: str, *, include: bool) -> None:
        """Final (post-adjudication) title/abstract outcome for one unique record.

        Re-deciding a record clears its downstream retrieval and eligibility state.
        """
        self._screened_included.discard(record_id)
        self._screened_excluded.discard(record_id)
        if not include:
            self._retrieved.discard(record_id)
            self._not_retrieved.discard(record_id)
            self._included_reports.discard(record_id)
            self._fulltext_excluded.pop(record_id, None)
        (self._screened_included if include else self._screened_excluded).add(record_id)

    def record_retrieval(self, record_id: str, *, retrieved: bool) -> None:
        if record_id not in self._screened_included:
            raise PRISMAInconsistencyError(
                f"{record_id} was not included at title/abstract screening"
            )
        self._retrieved.discard(record_id)
        self._not_retrieved.discard(record_id)
        if not retrieved:
            self._included_reports.discard(record_id)
            self._fulltext_excluded.pop(record_id, None)
        (self._retrieved if retrieved else self._not_retrieved).add(record_id)

    def record_fulltext_decision(
        self, record_id: str, *, include: bool, reason: str | None = None
    ) -> None:
        if record_id not in self._retrieved:
            raise PRISMAInconsistencyError(f"{record_id} has no retrieved full text")
        self._included_reports.discard(record_id)
        self._fulltext_excluded.pop(record_id, None)
        if include:
            self._included_reports.add(record_id)
        else:
            if not reason:
                raise ValueError("full-text exclusions require a reason (PRISMA item 16b)")
            self._fulltext_excluded[record_id] = reason

    def set_studies_included(self, n: int) -> None:
        """Number of distinct studies (after linking multiple reports of one cohort)."""
        if n < 0:
            raise ValueError("n must be non-negative")
        self._studies_included = n

    # -------------------------------------------------------------- output
    def snapshot(self) -> PRISMACounts:
        screened = len(self._screened_included) + len(self._screened_excluded)
        reports_included = len(self._included_reports)
        pending = self.pending()
        try:
            return PRISMACounts(
                identified_by_database=dict(self._identified),
                identified_from_registers=self._registers,
                duplicates_removed=self._duplicates,
                marked_ineligible_by_automation=self._automation_ineligible,
                removed_other_reasons=self._other_removed,
                records_screened=screened,
                records_excluded=len(self._screened_excluded),
                reports_sought=len(self._screened_included),
                reports_not_retrieved=len(self._not_retrieved),
                reports_assessed=len(self._retrieved),  # retrieved reports are assessed
                reports_excluded_by_reason=dict(
                    sorted(Counter(self._fulltext_excluded.values()).items())
                ),
                reports_included=reports_included,
                reports_awaiting_retrieval=pending["awaiting_retrieval_decision"],
                reports_awaiting_assessment=pending["awaiting_fulltext_decision"],
                studies_included=(
                    self._studies_included
                    if self._studies_included is not None
                    else reports_included
                ),
            )
        except ValueError as exc:
            raise PRISMAInconsistencyError(str(exc)) from exc

    def pending(self) -> dict[str, int]:
        """Work items not yet accounted for, useful for progress display."""
        return {
            "awaiting_retrieval_decision": len(
                self._screened_included - self._retrieved - self._not_retrieved
            ),
            "awaiting_fulltext_decision": len(
                self._retrieved - self._included_reports - set(self._fulltext_excluded)
            ),
        }
