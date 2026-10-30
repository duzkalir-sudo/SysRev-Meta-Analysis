"""Overlapping Cohort Detector.

Multiple publications from one cohort (interim/final reports, secondary
analyses, extension studies) inflate precision if pooled as independent
studies. The Cochrane Handbook (Higgins et al., version 6, Chapter 5,
"Collecting data") recommends linking multiple reports of the same study, with
the study rather than the report as the unit of analysis. This
module *flags* candidate pairs; it never excludes or merges anything.

Signals (each scored in [0, 1]):

* ``shared_registry_id`` — identical trial-registry IDs (e.g. NCT numbers).
  Decisive: score 1.0, severity HIGH.
* ``coauthor_overlap`` — overlap coefficient |A∩B| / min(|A|, |B|) of
  normalised ``surname + first initial`` keys. One shared name in teams of ≥4
  is down-weighted ×0.5 because surname/initial collisions are common.
* ``affiliation_similarity`` — fraction of the smaller study's institutions
  matched (RapidFuzz ``token_set_ratio`` ≥ threshold) after stripping
  department-level segments and generic words (university, hospital, ...).
* ``recruitment_overlap`` — overlapping recruitment years / shorter period.
  When either period is unknown, ``publication_proximity`` substitutes with at
  most half the weight, because publication dates are a weak proxy.

Composite = w_coauthor·co + w_affiliation·aff + w_recruitment·rec, with
default weights 0.40 / 0.30 / 0.30. These weights and thresholds are
engineering heuristics, not validated estimates; they are configurable and
every warning lists its individual signals so a human can judge them.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations

from rapidfuzz import fuzz

from src.core.enums import OverlapSeverity
from src.core.identifiers import normalize_title
from src.core.schemas.cohort import CohortOverlapWarning, OverlapSignal
from src.core.schemas.study import StudyRecord

INSTITUTION_KEYWORDS = (
    "universit",
    "hospital",
    "clinic",
    "institut",
    "college",
    "centre",
    "center",
    "school",
    "faculty",
    "foundation",
    "infirmary",
    "klinik",
    "ospedale",
    "hopital",
    "academy",
    "trust",
    "council",
    "ministry",
    "laborator",
)
DEPARTMENT_PREFIXES = (
    "department",
    "dept",
    "division",
    "section",
    "unit",
    "service",
    "laboratory of",
    "lab of",
    "program",
    "programme",
    "graduate",
)
GENERIC_WORDS = frozenset(
    {
        "university",
        "universite",
        "universitat",
        "universidad",
        "universita",
        "universidade",
        "univ",
        "hospital",
        "hospitals",
        "hopital",
        "ospedale",
        "klinik",
        "klinikum",
        "clinic",
        "clinics",
        "clinical",
        "medical",
        "medicine",
        "center",
        "centre",
        "centro",
        "institute",
        "institut",
        "instituto",
        "istituto",
        "college",
        "school",
        "faculty",
        "department",
        "dept",
        "division",
        "unit",
        "of",
        "the",
        "and",
        "for",
        "de",
        "la",
        "le",
        "del",
        "der",
        "di",
        "des",
        "du",
        "et",
        "y",
        "e",
        "research",
        "sciences",
        "science",
        "health",
        "healthcare",
        "general",
        "national",
        "foundation",
        "trust",
        "nhs",
        "affiliated",
        "teaching",
        "academic",
        "memorial",
    }
)
_EMAIL_RE = re.compile(r"\S+@\S+")
_DIGITS_RE = re.compile(r"\d+")


@dataclass(frozen=True, slots=True)
class CohortOverlapConfig:
    w_coauthor: float = 0.40
    w_affiliation: float = 0.30
    w_recruitment: float = 0.30
    flag_threshold: float = 0.45
    moderate_threshold: float = 0.55
    high_threshold: float = 0.70
    affiliation_match_threshold: float = 85.0
    publication_window_years: int = 5

    def __post_init__(self) -> None:
        total = self.w_coauthor + self.w_affiliation + self.w_recruitment
        if abs(total - 1.0) > 1e-9:
            raise ValueError("signal weights must sum to 1")
        if not (0 <= self.flag_threshold <= self.moderate_threshold <= self.high_threshold <= 1):
            raise ValueError("thresholds must satisfy 0 <= flag <= moderate <= high <= 1")


def institution_keys(affiliation: str) -> list[str]:
    """Distinctive institution keys from one affiliation string.

    ``"Department of Cardiology, Mayo Clinic, Rochester, MN 55905, USA"`` →
    ``["mayo"]``.
    """
    text = _DIGITS_RE.sub(" ", _EMAIL_RE.sub(" ", affiliation))
    segments = [s.strip() for s in re.split(r"[,;]", text) if s.strip()]
    keys: list[str] = []
    for seg in segments:
        low = seg.casefold()
        if low.startswith(DEPARTMENT_PREFIXES):
            continue
        if not any(k in low for k in INSTITUTION_KEYWORDS):
            continue
        tokens = [t for t in normalize_title(seg).split() if t not in GENERIC_WORDS]
        if tokens:
            keys.append(" ".join(tokens))
    return list(dict.fromkeys(keys))


def _institutions(record: StudyRecord) -> list[str]:
    keys: list[str] = []
    for aff in record.all_affiliations:
        keys.extend(institution_keys(aff))
    return sorted(set(keys))


def _author_keys(record: StudyRecord) -> set[str]:
    return {k for a in record.authors if (k := a.match_key)}


class CohortOverlapDetector:
    def __init__(self, config: CohortOverlapConfig | None = None) -> None:
        self.config = config or CohortOverlapConfig()

    # ------------------------------------------------------------------ signals
    def coauthor_signal(self, a: StudyRecord, b: StudyRecord) -> tuple[float, list[str]]:
        ka, kb = _author_keys(a), _author_keys(b)
        if not ka or not kb:
            return 0.0, []
        shared = sorted(ka & kb)
        if not shared:
            return 0.0, []
        smaller = min(len(ka), len(kb))
        score = len(shared) / smaller
        if len(shared) == 1 and smaller >= 4:
            score *= 0.5
        return min(1.0, score), shared

    def affiliation_signal(self, a: StudyRecord, b: StudyRecord) -> tuple[float, list[str]]:
        ia, ib = _institutions(a), _institutions(b)
        if not ia or not ib:
            return 0.0, []
        small, large = (ia, ib) if len(ia) <= len(ib) else (ib, ia)
        matched = [
            inst
            for inst in small
            if max(fuzz.token_set_ratio(inst, other) for other in large)
            >= self.config.affiliation_match_threshold
        ]
        return len(matched) / len(small), matched

    def recruitment_signal(self, a: StudyRecord, b: StudyRecord) -> tuple[float | None, int | None]:
        pa, pb = a.recruitment_period, b.recruitment_period
        if pa is None or pb is None:
            return None, None
        overlap = max(0, min(pa.end_year, pb.end_year) - max(pa.start_year, pb.start_year) + 1)
        shorter = min(pa.end_year - pa.start_year + 1, pb.end_year - pb.start_year + 1)
        return overlap / shorter, overlap

    def publication_signal(self, a: StudyRecord, b: StudyRecord) -> float | None:
        if a.publication_year is None or b.publication_year is None:
            return None
        gap = abs(a.publication_year - b.publication_year)
        return max(0.0, 1.0 - gap / self.config.publication_window_years)

    # ------------------------------------------------------------------ scoring
    def compare(self, a: StudyRecord, b: StudyRecord) -> CohortOverlapWarning | None:
        if a.record_id > b.record_id:
            a, b = b, a
        cfg = self.config
        signals: list[OverlapSignal] = []

        shared_registry = sorted(set(a.registry_ids) & set(b.registry_ids))
        co, shared_authors = self.coauthor_signal(a, b)
        aff, shared_inst = self.affiliation_signal(a, b)
        rec, overlap_years = self.recruitment_signal(a, b)

        if shared_registry:
            signals.append(
                OverlapSignal(
                    kind="shared_registry_id",
                    score=1.0,
                    detail=f"identical registry ID(s): {', '.join(shared_registry)}",
                )
            )
        if co:
            signals.append(
                OverlapSignal(
                    kind="coauthor_overlap",
                    score=co,
                    detail=(
                        f"{len(shared_authors)} shared author key(s): " + ", ".join(shared_authors)
                    ),
                )
            )
        if aff:
            signals.append(
                OverlapSignal(
                    kind="affiliation_similarity",
                    score=aff,
                    detail=f"matched institution(s): {', '.join(shared_inst)}",
                )
            )

        if rec is not None:
            signals.append(
                OverlapSignal(
                    kind="recruitment_overlap",
                    score=rec,
                    detail=f"{overlap_years} overlapping recruitment year(s)",
                )
            )
            time_term = cfg.w_recruitment * rec
        else:
            pub = self.publication_signal(a, b)
            if pub:
                signals.append(
                    OverlapSignal(
                        kind="publication_proximity",
                        score=pub,
                        detail=(
                            f"published {a.publication_year} and {b.publication_year}; "
                            "recruitment periods unknown"
                        ),
                    )
                )
            time_term = 0.5 * cfg.w_recruitment * (pub or 0.0)

        if shared_registry:
            score = 1.0
        else:
            # Gate: institutions or dates alone are too weak to imply a shared cohort.
            if not (co > 0 or (aff >= 0.5 and rec is not None and rec > 0)):
                return None
            score = cfg.w_coauthor * co + cfg.w_affiliation * aff + time_term
        score = round(min(1.0, score), 4)
        if score < cfg.flag_threshold:
            return None

        if score >= cfg.high_threshold:
            severity = OverlapSeverity.HIGH
        elif score >= cfg.moderate_threshold:
            severity = OverlapSeverity.MODERATE
        else:
            severity = OverlapSeverity.LOW

        return CohortOverlapWarning(
            record_id_a=a.record_id,
            record_id_b=b.record_id,
            overlap_score=score,
            severity=severity,
            signals=signals,
            shared_authors=shared_authors,
            shared_institutions=shared_inst,
            shared_registry_ids=shared_registry,
            recruitment_overlap_years=overlap_years,
        )

    def _candidate_pairs(self, records: Sequence[StudyRecord]) -> set[tuple[int, int]]:
        """Pairs sharing at least one author key, institution token or registry ID."""
        index: dict[str, set[int]] = defaultdict(set)
        for i, r in enumerate(records):
            for key in _author_keys(r):
                index[f"a:{key}"].add(i)
            for rid in r.registry_ids:
                index[f"r:{rid}"].add(i)
            for inst in _institutions(r):
                for token in inst.split():
                    index[f"i:{token}"].add(i)
        pairs: set[tuple[int, int]] = set()
        for members in index.values():
            if len(members) > 1:
                pairs.update(combinations(sorted(members), 2))
        return pairs

    def detect(self, records: Iterable[StudyRecord]) -> list[CohortOverlapWarning]:
        """All flagged pairs, most suspicious first (ties by record ids)."""
        ordered = sorted(records, key=lambda r: r.record_id)
        if len({r.record_id for r in ordered}) != len(ordered):
            raise ValueError("record_id values must be unique")
        warnings = [
            w
            for i, j in sorted(self._candidate_pairs(ordered))
            if (w := self.compare(ordered[i], ordered[j])) is not None
        ]
        warnings.sort(key=lambda w: (-w.overlap_score, w.record_id_a, w.record_id_b))
        return warnings
