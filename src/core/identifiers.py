"""Deterministic normalisation and validation of bibliographic identifiers."""

from __future__ import annotations

import html
import re
import unicodedata

from src.core.exceptions import InvalidIdentifierError

# Crossref recommends ^10.\d{4,9}/[-._;()/:A-Z0-9]+$ for modern DOIs; legacy
# DOIs contain other characters, so the suffix only forbids whitespace here.
_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")
_DOI_PREFIXES = re.compile(
    r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*|https?://doi\.org/)", re.IGNORECASE
)
_PMID_RE = re.compile(r"^\d{1,9}$")
_PMCID_RE = re.compile(r"^PMC\d{1,9}$")

# Trial / review registry identifiers.
REGISTRY_PATTERNS: dict[str, re.Pattern[str]] = {
    "clinicaltrials.gov": re.compile(r"\bNCT\d{8}\b"),
    "isrctn": re.compile(r"\bISRCTN\d{8}\b"),
    "euctr": re.compile(r"\b\d{4}-\d{6}-\d{2}\b"),
    "chictr": re.compile(r"\bChiCTR[-A-Z0-9]{6,20}\b"),
    "anzctr": re.compile(r"\bACTRN\d{14}\b"),
    "drks": re.compile(r"\bDRKS\d{8}\b"),
    "ctri": re.compile(r"\bCTRI/\d{4}/\d{2,3}/\d{6}\b"),
    "jrct": re.compile(r"\bjRCT[a-z]?\d{10}\b"),
    "umin": re.compile(r"\bUMIN\d{9}\b"),
    "prospero": re.compile(r"\bCRD\d{11}\b"),
}

_TAG_RE = re.compile(r"<[^>]+>")
_NON_ALNUM_RE = re.compile(r"[^0-9a-z]+")


def normalize_doi(raw: str | None) -> str | None:
    """Return the canonical lower-case DOI or ``None`` for empty input.

    DOIs are case-insensitive (DOI Handbook §2.4), so lower-casing is lossless
    for matching purposes. Resolver prefixes and trailing punctuation are removed.

    Raises:
        InvalidIdentifierError: if the cleaned value is not a syntactically valid DOI.
    """
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    value = _DOI_PREFIXES.sub("", value).strip().rstrip(".,;")
    value = value.lower()
    if not _DOI_RE.match(value):
        raise InvalidIdentifierError(f"Invalid DOI: {raw!r}")
    return value


def normalize_pmid(raw: str | int | None) -> str | None:
    if raw is None:
        return None
    value = str(raw).strip()
    if not value:
        return None
    value = re.sub(r"^(?:https?://pubmed\.ncbi\.nlm\.nih\.gov/|pmid:\s*)", "", value, flags=re.I)
    value = value.rstrip("/")
    if not _PMID_RE.match(value) or int(value) == 0:
        raise InvalidIdentifierError(f"Invalid PMID: {raw!r}")
    return str(int(value))


def normalize_pmcid(raw: str | None) -> str | None:
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    value = re.sub(r"^https?://(?:www\.)?ncbi\.nlm\.nih\.gov/pmc/articles/", "", value, flags=re.I)
    value = value.rstrip("/").upper()
    if value.isdigit():
        value = f"PMC{value}"
    if not _PMCID_RE.match(value):
        raise InvalidIdentifierError(f"Invalid PMCID: {raw!r}")
    return value


def extract_registry_ids(text: str | None) -> list[str]:
    """Find trial-registry identifiers in free text, sorted and de-duplicated."""
    if not text:
        return []
    found: set[str] = set()
    for pattern in REGISTRY_PATTERNS.values():
        found.update(pattern.findall(text))
    return sorted(found)


def strip_markup(text: str) -> str:
    """Remove HTML/XML tags and unescape entities."""
    return html.unescape(_TAG_RE.sub(" ", text))


def normalize_title(title: str | None) -> str:
    """Canonical title key for fuzzy comparison.

    Steps: strip markup, NFKD-decompose and drop combining marks (diacritics),
    case-fold, replace every non-alphanumeric run with one space, trim.
    """
    if not title:
        return ""
    text = strip_markup(title)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold()
    return _NON_ALNUM_RE.sub(" ", text).strip()


def normalize_person_name(last_name: str | None, first_names: str | None = None) -> str:
    """``"surname f"`` key (surname + first initial), accent- and case-insensitive."""
    if not last_name:
        return ""
    surname = normalize_title(last_name).replace(" ", "")
    initial = ""
    if first_names:
        cleaned = normalize_title(first_names)
        initial = cleaned[:1]
    return f"{surname} {initial}".strip()
