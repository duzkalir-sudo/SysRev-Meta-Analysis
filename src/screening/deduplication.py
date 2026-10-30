"""Deterministic multi-pass record deduplication.

Pass 1 — exact identifiers: records sharing a DOI, PMID, PMCID or OpenAlex ID
are merged (union-find, so matches are transitive).

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
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
from rapidfuzz import fuzz, process

from src.core.enums import SourceDatabase
from src.core.schemas.study import StudyRecord

SOURCE_PRIORITY: dict[SourceDatabase, int] = {
    SourceDatabase.PUBMED: 0,
    SourceDatabase.EUROPE_PMC: 1,
    SourceDatabase.CROSSREF: 2,
    SourceDatabase.OPENALEX: 3,
    SourceDatabase.MANUAL: 4,
}


@dataclass(frozen=True, slots=True)
class PossibleDuplicate:
    record_id_a: str
    record_id_b: str
    title_similarity: float
    blocking_reason: str


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

    def clusters(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = defaultdict(list)
        for rec in self.unique_records:
            groups[rec.record_id].append(rec.record_id)
        for dup, rep in sorted(self.duplicate_of.items()):
            groups[rep].append(dup)
        return dict(groups)


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


def _contradiction(a: StudyRecord, b: StudyRecord) -> str | None:
    if a.doi and b.doi and a.doi != b.doi:
        return "different DOIs"
    if a.pmid and b.pmid and a.pmid != b.pmid:
        return "different PMIDs"
    sa, sb = a.first_author_key.split(" ")[0], b.first_author_key.split(" ")[0]
    if sa and sb and sa != sb:
        return "different first-author surnames"
    return None


def _representative_key(r: StudyRecord) -> tuple[int, int, str]:
    return (-r.completeness_score(), SOURCE_PRIORITY[r.source_database], r.record_id)


class Deduplicator:
    def __init__(self, *, title_threshold: float = 95.0, year_tolerance: int = 1) -> None:
        if not 50.0 <= title_threshold <= 100.0:
            raise ValueError("title_threshold must be within [50, 100]")
        if year_tolerance < 0:
            raise ValueError("year_tolerance must be >= 0")
        self.title_threshold = title_threshold
        self.year_tolerance = year_tolerance

    def deduplicate(self, records: Sequence[StudyRecord]) -> DeduplicationResult:
        ids = [r.record_id for r in records]
        if len(ids) != len(set(ids)):
            raise ValueError("record_id values must be unique")
        by_id = {r.record_id: r for r in records}
        uf = _UnionFind(sorted(by_id))
        reasons: dict[frozenset[str], str] = {}

        self._exact_pass(records, uf, reasons)
        possible = self._fuzzy_pass(records, uf, reasons)

        clusters: dict[str, list[StudyRecord]] = defaultdict(list)
        for rid in sorted(by_id):
            clusters[uf.find(rid)].append(by_id[rid])

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
        # Hide pairs that ended up merged transitively through other members.
        possible = [p for p in possible if uf.find(p.record_id_a) != uf.find(p.record_id_b)]
        return DeduplicationResult(
            unique_records=unique,
            all_records=[updated[r.record_id] for r in records],
            duplicate_of=duplicate_of,
            match_reasons=match_reasons,
            possible_duplicates=possible,
        )

    @staticmethod
    def _exact_pass(
        records: Sequence[StudyRecord], uf: _UnionFind, reasons: dict[frozenset[str], str]
    ) -> None:
        for attr in ("doi", "pmid", "pmcid", "openalex_id"):
            first_seen: dict[str, str] = {}
            for r in sorted(records, key=lambda x: x.record_id):
                key = getattr(r, attr)
                if not key:
                    continue
                if key in first_seen:
                    other = first_seen[key]
                    uf.union(other, r.record_id)
                    reasons.setdefault(frozenset((other, r.record_id)), f"exact {attr}")
                else:
                    first_seen[key] = r.record_id

    def _fuzzy_pass(
        self,
        records: Sequence[StudyRecord],
        uf: _UnionFind,
        reasons: dict[frozenset[str], str],
    ) -> list[PossibleDuplicate]:
        ordered = sorted(records, key=lambda x: x.record_id)
        titles = [r.normalized_title for r in ordered]
        candidates = self._candidate_pairs(ordered, titles)

        members: dict[str, list[StudyRecord]] = defaultdict(list)
        for r in ordered:
            members[uf.find(r.record_id)].append(r)

        possible: list[PossibleDuplicate] = []
        # Most similar pairs first; ties broken by ids -> deterministic.
        for score, i, j in sorted(candidates, key=lambda t: (-t[0], t[1], t[2])):
            a, b = ordered[i], ordered[j]
            ra, rb = uf.find(a.record_id), uf.find(b.record_id)
            if ra == rb:
                continue
            # Conservative: any contradiction between the two clusters blocks the merge.
            blocker = next(
                (
                    reason
                    for x in members[ra]
                    for y in members[rb]
                    if (reason := _contradiction(x, y))
                ),
                None,
            )
            if blocker:
                possible.append(PossibleDuplicate(a.record_id, b.record_id, score, blocker))
                continue
            uf.union(a.record_id, b.record_id)
            root = uf.find(a.record_id)
            other = rb if root == ra else ra
            members[root].extend(members.pop(other))
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
