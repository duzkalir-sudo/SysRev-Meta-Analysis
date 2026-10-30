"""Deterministic multi-pass record deduplication.

Pass 1 — exact identifiers: records sharing a DOI, PMID, PMCID or OpenAlex ID
are merged (union-find), *unless* the merge would put two different values of
any identifier into one cluster. Conflicts are checked across every member of
both clusters, so a transitive chain (a~b by DOI, b~c by PMID, but a and c carry
different DOIs) cannot produce a mixed cluster. Blocked identifier matches are
returned as ``PossibleDuplicate`` for human adjudication.

Pass 2 — fuzzy titles: within publication-year blocks (±``year_tolerance``),
titles are compared with RapidFuzz ``token_sort_ratio`` on normalised text.
A pair scoring ≥ ``title_threshold`` is merged *only* if no hard evidence
contradicts it: differing DOIs, differing PMIDs, or differing first-author
surnames. Contradicted high-similarity pairs are reported as
``PossibleDuplicate`` for human review instead of being merged — this keeps
the engine conservative (a false merge silently deletes a study; a missed
merge is caught downstream by screening).

The output is independent of input order: clusters are deterministic and the
representative is the most complete record (ties broken by source priority,
then ``record_id``).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
from rapidfuzz import fuzz, process

from src.core.enums import SourceDatabase
from src.core.schemas.screening import DuplicateCandidate
from src.core.schemas.study import StudyRecord

SOURCE_PRIORITY: dict[SourceDatabase, int] = {
    SourceDatabase.PUBMED: 0,
    SourceDatabase.EUROPE_PMC: 1,
    SourceDatabase.CROSSREF: 2,
    SourceDatabase.OPENALEX: 3,
    SourceDatabase.EMBASE: 4,
    SourceDatabase.SCOPUS: 5,
    SourceDatabase.WEB_OF_SCIENCE: 6,
    SourceDatabase.COCHRANE_CENTRAL: 7,
    SourceDatabase.CLINICALTRIALS_GOV: 8,
    SourceDatabase.MANUAL: 9,
}


@dataclass(frozen=True, slots=True)
class PossibleDuplicate:
    """A pair that matched but was *not* merged because evidence conflicts.

    ``match_basis`` is ``"exact <identifier>"`` or ``"fuzzy title"``;
    ``title_similarity`` is only set for fuzzy matches.
    """

    record_id_a: str
    record_id_b: str
    match_basis: str
    blocking_reason: str
    title_similarity: float | None = None

    def to_candidate(self) -> DuplicateCandidate:
        a, b = sorted((self.record_id_a, self.record_id_b))
        return DuplicateCandidate(
            record_id_a=a,
            record_id_b=b,
            match_basis=self.match_basis,
            conflict=self.blocking_reason,
            title_similarity=self.title_similarity,
        )


@dataclass(slots=True)
class DeduplicationResult:
    unique_records: list[StudyRecord]
    all_records: list[StudyRecord]
    duplicate_of: dict[str, str]
    match_reasons: dict[str, str]
    possible_duplicates: list[PossibleDuplicate] = field(default_factory=list)

    @property
    def duplicates_removed(self) -> int:
        return len(self.duplicate_of)

    @property
    def requires_adjudication(self) -> bool:
        return bool(self.possible_duplicates)

    def clusters(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = defaultdict(list)
        for rec in self.unique_records:
            groups[rec.record_id].append(rec.record_id)
        for dup, rep in sorted(self.duplicate_of.items()):
            groups[rep].append(dup)
        return dict(groups)


class _ClusterSet:
    """Union-find that remembers cluster members and refuses contradictory merges."""

    def __init__(self, records: Iterable[StudyRecord]) -> None:
        ordered = sorted(records, key=lambda r: r.record_id)
        self.uf = _UnionFind(r.record_id for r in ordered)
        self.members: dict[str, list[StudyRecord]] = {r.record_id: [r] for r in ordered}

    def find(self, record_id: str) -> str:
        return self.uf.find(record_id)

    def try_merge(
        self, a: str, b: str, conflict: Callable[[StudyRecord, StudyRecord], str | None]
    ) -> str | None:
        """Merge the clusters of ``a`` and ``b``; return a blocking reason instead if any
        member of one cluster conflicts with any member of the other."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return None
        for x in self.members[ra]:
            for y in self.members[rb]:
                reason = conflict(x, y)
                if reason:
                    return reason
        self.uf.union(ra, rb)
        root = self.find(ra)
        other = rb if root == ra else ra
        self.members[root].extend(self.members.pop(other))
        return None


class _UnionFind:
    def __init__(self, items: Iterable[str]) -> None:
        self.parent = {i: i for i in items}

    def find(self, x: str) -> str:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        lo, hi = sorted((ra, rb))
        self.parent[hi] = lo
        return True


IDENTIFIER_FIELDS = ("doi", "pmid", "pmcid", "openalex_id")
_IDENTIFIER_LABELS = {
    "doi": "DOIs",
    "pmid": "PMIDs",
    "pmcid": "PMCIDs",
    "openalex_id": "OpenAlex IDs",
}


def _identifier_conflict(a: StudyRecord, b: StudyRecord) -> str | None:
    """Two records carry different values for the same identifier."""
    for attr in IDENTIFIER_FIELDS:
        va, vb = getattr(a, attr), getattr(b, attr)
        if va and vb and va != vb:
            return f"different {_IDENTIFIER_LABELS[attr]} ({va} vs {vb})"
    return None


def _exact_conflict_factory(
    title_floor: float | None,
) -> Callable[[StudyRecord, StudyRecord], str | None]:
    """Identifier conflict, or a shared identifier on clearly unrelated titles.

    A wrong DOI/PMID in an export (or an erratum sharing a DOI) would otherwise merge two
    different reports; such pairs go to a human instead. Translated titles may also fall
    below the floor; they are then adjudicated rather than merged (conservative).
    """

    def conflict(a: StudyRecord, b: StudyRecord) -> str | None:
        reason = _identifier_conflict(a, b)
        if reason or title_floor is None:
            return reason
        ta, tb = a.normalized_title, b.normalized_title
        if ta and tb:
            score = fuzz.token_sort_ratio(ta, tb)
            if score < title_floor:
                return f"shared identifier but dissimilar titles (similarity {score:.0f})"
        return None

    return conflict


def _contradiction(a: StudyRecord, b: StudyRecord) -> str | None:
    """Evidence against a fuzzy-title merge: identifier conflict or first-author mismatch."""
    conflict = _identifier_conflict(a, b)
    if conflict:
        return conflict
    sa, sb = a.first_author_key.split(" ")[0], b.first_author_key.split(" ")[0]
    if sa and sb and sa != sb:
        return "different first-author surnames"
    return None


def _representative_key(r: StudyRecord) -> tuple[int, int, str]:
    return (-r.completeness_score(), SOURCE_PRIORITY[r.source_database], r.record_id)


class Deduplicator:
    def __init__(
        self,
        *,
        title_threshold: float = 95.0,
        year_tolerance: int = 1,
        exact_match_title_floor: float | None = None,
    ) -> None:
        if not 50.0 <= title_threshold <= 100.0:
            raise ValueError("title_threshold must be within [50, 100]")
        if year_tolerance < 0:
            raise ValueError("year_tolerance must be >= 0")
        if exact_match_title_floor is not None and not 0 <= exact_match_title_floor <= 100:
            raise ValueError("exact_match_title_floor must be within [0, 100]")
        self.title_threshold = title_threshold
        self.year_tolerance = year_tolerance
        self.exact_match_title_floor = exact_match_title_floor

    def deduplicate(self, records: Sequence[StudyRecord]) -> DeduplicationResult:
        ids = [r.record_id for r in records]
        if len(ids) != len(set(ids)):
            raise ValueError("record_id values must be unique")
        by_id = {r.record_id: r for r in records}
        cs = _ClusterSet(records)
        reasons: dict[frozenset[str], str] = {}

        possible = self._exact_pass(records, cs, reasons)
        possible += self._fuzzy_pass(records, cs, reasons)

        clusters: dict[str, list[StudyRecord]] = defaultdict(list)
        for rid in sorted(by_id):
            clusters[cs.find(rid)].append(by_id[rid])

        duplicate_of: dict[str, str] = {}
        match_reasons: dict[str, str] = {}
        unique: list[StudyRecord] = []
        updated: dict[str, StudyRecord] = {}
        for members in clusters.values():
            rep = min(members, key=_representative_key)
            merged = _fill_identifiers(rep, members)
            unique.append(merged)
            updated[rep.record_id] = merged
            for m in members:
                if m.record_id == rep.record_id:
                    continue
                duplicate_of[m.record_id] = rep.record_id
                updated[m.record_id] = m.model_copy(update={"duplicate_of": rep.record_id})
                match_reasons[m.record_id] = _cluster_reason(m, members, reasons)

        unique.sort(key=lambda r: r.record_id)
        # One entry per unordered pair (first reason wins); merged pairs cannot appear here
        # because a blocked pair's clusters can never be joined later (blocks are permanent).
        seen: set[frozenset[str]] = set()
        unique_possible: list[PossibleDuplicate] = []
        for pd in possible:
            key = frozenset((pd.record_id_a, pd.record_id_b))
            if key not in seen and cs.find(pd.record_id_a) != cs.find(pd.record_id_b):
                seen.add(key)
                unique_possible.append(pd)
        possible = unique_possible
        return DeduplicationResult(
            unique_records=unique,
            all_records=[updated[r.record_id] for r in records],
            duplicate_of=duplicate_of,
            match_reasons=match_reasons,
            possible_duplicates=possible,
        )

    def _exact_pass(
        self,
        records: Sequence[StudyRecord],
        cs: _ClusterSet,
        reasons: dict[frozenset[str], str],
    ) -> list[PossibleDuplicate]:
        conflict = _exact_conflict_factory(self.exact_match_title_floor)
        possible: list[PossibleDuplicate] = []
        ordered = sorted(records, key=lambda x: x.record_id)
        for attr in IDENTIFIER_FIELDS:
            groups: dict[str, list[str]] = defaultdict(list)
            for r in ordered:
                key = getattr(r, attr)
                if key:
                    groups[key].append(r.record_id)
            for key in sorted(groups):
                ids = groups[key]
                for idx, other in enumerate(ids[1:], start=1):
                    # Join the first compatible earlier record; flag every conflicting one.
                    for prev in ids[:idx]:
                        if cs.find(prev) == cs.find(other):
                            continue
                        blocker = cs.try_merge(prev, other, conflict)
                        if blocker:
                            possible.append(
                                PossibleDuplicate(prev, other, f"exact {attr} ({key})", blocker)
                            )
                        else:
                            reasons.setdefault(frozenset((prev, other)), f"exact {attr}")
        return possible

    def _fuzzy_pass(
        self,
        records: Sequence[StudyRecord],
        cs: _ClusterSet,
        reasons: dict[frozenset[str], str],
    ) -> list[PossibleDuplicate]:
        ordered = sorted(records, key=lambda x: x.record_id)
        titles = [r.normalized_title for r in ordered]
        candidates = self._candidate_pairs(ordered, titles)

        possible: list[PossibleDuplicate] = []
        # Most similar pairs first; ties broken by ids -> deterministic.
        for score, i, j in sorted(candidates, key=lambda t: (-t[0], t[1], t[2])):
            a, b = ordered[i], ordered[j]
            if cs.find(a.record_id) == cs.find(b.record_id):
                continue
            # Conservative: any contradiction between the two clusters blocks the merge.
            blocker = cs.try_merge(a.record_id, b.record_id, _contradiction)
            if blocker:
                possible.append(
                    PossibleDuplicate(a.record_id, b.record_id, "fuzzy title", blocker, score)
                )
                continue
            reasons.setdefault(frozenset((a.record_id, b.record_id)), f"fuzzy title ({score:.1f})")
        return possible

    def _candidate_pairs(
        self, ordered: list[StudyRecord], titles: list[str]
    ) -> set[tuple[float, int, int]]:
        """Title pairs above threshold, blocked by publication year (±tolerance).

        Records without a year are compared against every record.
        """
        blocks: dict[int | None, list[int]] = defaultdict(list)
        for idx, r in enumerate(ordered):
            if titles[idx]:
                blocks[r.publication_year].append(idx)
        years = sorted(y for y in blocks if y is not None)
        comparisons: list[tuple[list[int], list[int]]] = []
        for k, y1 in enumerate(years):
            for y2 in years[k:]:
                if y2 - y1 > self.year_tolerance:
                    break
                comparisons.append((blocks[y1], blocks[y2]))
        if None in blocks:
            everything = [i for idxs in blocks.values() for i in idxs]
            comparisons.append((blocks[None], everything))

        pairs: set[tuple[float, int, int]] = set()
        for left, right in comparisons:
            matrix = process.cdist(
                [titles[i] for i in left],
                [titles[j] for j in right],
                scorer=fuzz.token_sort_ratio,
                score_cutoff=self.title_threshold,
                dtype="float32",
            )
            for li, rj in np.argwhere(matrix >= self.title_threshold):
                i, j = left[int(li)], right[int(rj)]
                if i != j:
                    lo, hi = (i, j) if i < j else (j, i)
                    pairs.add((round(float(matrix[li, rj]), 4), lo, hi))
        return pairs


def _cluster_reason(
    member: StudyRecord, members: list[StudyRecord], reasons: dict[frozenset[str], str]
) -> str:
    found = sorted(
        {
            reason
            for other in members
            if (reason := reasons.get(frozenset((member.record_id, other.record_id))))
        }
    )
    return "; ".join(found) if found else "transitive match within cluster"


def _fill_identifiers(rep: StudyRecord, members: list[StudyRecord]) -> StudyRecord:
    """Copy identifiers missing from the representative from its duplicates (never overwrite)."""
    update: dict[str, object] = {}
    for attr in ("pmid", "doi", "pmcid", "openalex_id", "abstract"):
        if getattr(rep, attr):
            continue
        for m in sorted(members, key=_representative_key):
            value = getattr(m, attr)
            if value:
                update[attr] = value
                break
    registry = sorted({rid for m in members for rid in m.registry_ids})
    if registry != rep.registry_ids:
        update["registry_ids"] = registry
    return rep.model_copy(update=update) if update else rep


def deduplicate(records: Sequence[StudyRecord], **kwargs: float) -> DeduplicationResult:
    return Deduplicator(**kwargs).deduplicate(records)  # type: ignore[arg-type]
