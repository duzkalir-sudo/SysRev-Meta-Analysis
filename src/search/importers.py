"""Deterministic parsers for exported search results: RIS, BibTeX and CSV.

Each parser returns one :class:`ParsedEntry` per entry in the file with the raw fields
exactly as read, and maps only fields that are present to :class:`StudyRecord` fields.
Nothing is completed, corrected or looked up. Identifiers that fail validation are left
out of the record (they remain visible in the raw entry) and reported as warnings.

An entry without a title cannot become a record; the whole import is then refused, so the
number of stored records always equals the number of entries in the file.

Field mappings follow the RIS tag list (Research Information Systems format, as used by
EndNote/Reference Manager exports) and common BibTeX field names. Vendor exports differ;
CSV columns are detected from common header names and the mapping is shown to the
reviewer, who can override it.
"""

from __future__ import annotations

import csv
import io
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from src.core.exceptions import InvalidIdentifierError, SysRevError
from src.core.identifiers import (
    extract_registry_ids,
    normalize_doi,
    normalize_pmcid,
    normalize_pmid,
)
from src.core.schemas.study import Author

ImportFormat = Literal["ris", "bibtex", "csv"]
MAX_IMPORT_BYTES = 200 * 1024 * 1024
csv.field_size_limit(16 * 1024 * 1024)  # long abstracts in CSV exports


class ImportFileError(SysRevError, ValueError):
    """The file cannot be imported as a whole (nothing is stored)."""


@dataclass(slots=True)
class ParsedEntry:
    position: int  # 1-based position in the file
    raw: dict[str, Any]
    title: str | None = None
    abstract: str | None = None
    authors: list[Author] = field(default_factory=list)
    journal: str | None = None
    year: int | None = None
    doi: str | None = None
    pmid: str | None = None
    pmcid: str | None = None
    source_id: str | None = None
    keywords: list[str] = field(default_factory=list)
    language: str | None = None
    publication_types: list[str] = field(default_factory=list)
    registry_ids: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ParsedFile:
    format: ImportFormat
    encoding: str
    entries: list[ParsedEntry]
    column_map: dict[str, str] = field(default_factory=dict)

    @property
    def errors(self) -> list[str]:
        return [f"entry {e.position}: no title" for e in self.entries if not e.title]

    @property
    def warnings(self) -> list[str]:
        return [f"entry {e.position}: {w}" for e in self.entries for w in e.warnings]


# ---------------------------------------------------------------------- helpers
def decode(data: bytes) -> tuple[str, str]:
    """Decode UTF-8 (with or without BOM), else Windows-1252; refuse anything else."""
    if len(data) > MAX_IMPORT_BYTES:
        raise ImportFileError("file too large")
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise ImportFileError("the file is neither UTF-8 nor Windows-1252 text; re-export as UTF-8")


_YEAR = re.compile(r"(?<!\d)(1[89]\d{2}|20\d{2}|2100)(?!\d)")


def _year(value: str | None) -> int | None:
    m = _YEAR.search(value or "")
    return int(m.group(1)) if m else None


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    text = " ".join(value.split())
    return text or None


def _identifier(
    entry: ParsedEntry, attr: str, value: str | None, fn: Callable[[str | None], str | None]
) -> None:
    if not value or getattr(entry, attr):
        return
    try:
        setattr(entry, attr, fn(value))
    except InvalidIdentifierError:
        entry.warnings.append(f"invalid {attr} {value!r} left out (kept in the raw entry)")


def _author(name: str) -> Author | None:
    """``"Last, First"`` is split at the comma; any other form is kept whole as full_name."""
    name = " ".join(name.split()).strip(" ;")
    if not name:
        return None
    if "," in name:
        last, _, fore = name.partition(",")
        if last.strip():
            return Author(last_name=last.strip(), fore_name=fore.strip() or None)
    return Author(full_name=name, name_parsed_heuristically=True)


def _finish(entry: ParsedEntry) -> ParsedEntry:
    entry.title = _clean(entry.title)
    entry.abstract = _clean(entry.abstract)
    found = set(entry.registry_ids)
    for text in (entry.abstract, entry.title, entry.source_id):
        found.update(extract_registry_ids(text))
    entry.registry_ids = sorted(found)
    return entry


# ---------------------------------------------------------------------- RIS
_RIS_LINE = re.compile(r"^([A-Z][A-Z0-9])  -(?: (.*))?$")


def parse_ris(text: str) -> list[ParsedEntry]:
    entries: list[ParsedEntry] = []
    current: dict[str, list[str]] | None = None
    last_tag: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        m = _RIS_LINE.match(line)
        if m:
            tag, value = m.group(1), (m.group(2) or "").strip()
            if tag == "TY":
                if current is not None:
                    raise ImportFileError(f"entry {len(entries) + 1}: 'TY' before 'ER'")
                current = {"TY": [value]}
            elif current is None:
                raise ImportFileError(f"tag {tag} outside an entry (missing 'TY')")
            elif tag == "ER":
                entries.append(_ris_entry(len(entries) + 1, current))
                current = None
            else:
                current.setdefault(tag, []).append(value)
            last_tag = tag
        elif line.strip() and current is not None and last_tag and last_tag in current:
            current[last_tag][-1] = f"{current[last_tag][-1]} {line.strip()}"  # continuation
        elif line.strip() and current is None:
            raise ImportFileError(f"unexpected text outside an entry: {line[:60]!r}")
    if current is not None:
        raise ImportFileError("the last entry is not terminated by 'ER  -'")
    return entries


def _first(tags: dict[str, list[str]], *names: str) -> str | None:
    for name in names:
        for value in tags.get(name, []):
            if value.strip():
                return value
    return None


def _ris_entry(position: int, tags: dict[str, list[str]]) -> ParsedEntry:
    e = ParsedEntry(position=position, raw={k: list(v) for k, v in tags.items()})
    e.title = _first(tags, "TI", "T1", "CT")
    e.abstract = _first(tags, "AB", "N2")
    for name in tags.get("AU", []) + tags.get("A1", []):
        if author := _author(name):
            e.authors.append(author)
    e.journal = _first(tags, "T2", "JO", "JF", "JA", "J2", "T3")
    e.year = _year(_first(tags, "PY", "Y1", "DA"))
    _identifier(e, "doi", _first(tags, "DO"), normalize_doi)
    e.source_id = _first(tags, "AN", "ID")
    e.keywords = [k for k in tags.get("KW", []) if k]
    e.language = _first(tags, "LA")
    e.publication_types = [t for t in tags.get("TY", []) if t]
    return _finish(e)


# ---------------------------------------------------------------------- BibTeX
_ACCENTS = {'"': "\u0308", "'": "\u0301", "`": "\u0300", "^": "\u0302", "~": "\u0303",
            "c": "\u0327", "v": "\u030c", "=": "\u0304", "u": "\u0306", ".": "\u0307"}  # fmt: skip
_ACCENT_RE = re.compile(r"\\([\"'`^~cv=u.])\s*\{?\s*([A-Za-z])\s*\}?")


def _latex(text: str) -> str:
    text = _ACCENT_RE.sub(lambda m: unicodedata.normalize("NFC", m.group(2) + _ACCENTS[m.group(1)]),
                          text)  # fmt: skip
    text = re.sub(r"\\([&%$#_{}])", r"\1", text)
    text = text.replace("{", "").replace("}", "").replace("~", " ")
    return " ".join(text.split())


def parse_bibtex(text: str) -> list[ParsedEntry]:
    entries: list[ParsedEntry] = []
    i, n = 0, len(text)
    while True:
        at = text.find("@", i)
        if at < 0:
            break
        m = re.match(r"@\s*([A-Za-z]+)\s*([{(])", text[at:])
        if not m:
            i = at + 1
            continue
        kind = m.group(1).lower()
        open_ch = m.group(2)
        close_ch = "}" if open_ch == "{" else ")"
        start = at + m.end()
        depth, j = 1, start
        while j < n and depth:
            ch = text[j]
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
            j += 1
        if depth:
            raise ImportFileError(f"unbalanced braces in BibTeX entry starting at offset {at}")
        body = text[start : j - 1]
        i = j
        if kind in ("comment", "preamble", "string"):
            continue
        entries.append(_bib_entry(len(entries) + 1, kind, body))
    return entries


def _bib_fields(body: str) -> tuple[str, dict[str, str]]:
    key, _, rest = body.partition(",")
    fields: dict[str, str] = {}
    i, n = 0, len(rest)
    while i < n:
        m = re.compile(r"\s*([A-Za-z][\w:-]*)\s*=\s*").match(rest, i)
        if not m:
            i += 1
            continue
        name = m.group(1).lower()
        i = m.end()
        if i < n and rest[i] in '{"':
            close = "}" if rest[i] == "{" else '"'
            depth, j = 1, i + 1
            while j < n and depth:
                if rest[j] == "{" and close == "}":
                    depth += 1
                elif rest[j] == close and (close == "}" or rest[j - 1] != "\\"):
                    depth -= 1
                j += 1
            value = rest[i + 1 : j - 1]
            i = j
        else:
            m2 = re.compile(r"([^,]*)").match(rest, i)
            assert m2 is not None
            value = m2.group(1).strip()
            i = m2.end()
        fields[name] = value
    return key.strip(), fields


def _bib_entry(position: int, kind: str, body: str) -> ParsedEntry:
    key, f = _bib_fields(body)
    e = ParsedEntry(position=position, raw={"entry_type": kind, "key": key, **f})

    def get(name: str) -> str | None:
        return _latex(f[name]) if name in f else None

    e.title = get("title")
    e.abstract = get("abstract")
    authors = f.get("author", "")
    for name in re.split(r"\s+and\s+", authors) if authors.strip() else []:
        if author := _author(_latex(name)):
            e.authors.append(author)
    e.journal = get("journal") or get("journaltitle") or get("booktitle")
    e.year = _year(get("year") or get("date"))
    _identifier(e, "doi", get("doi"), normalize_doi)
    _identifier(e, "pmid", get("pmid"), normalize_pmid)
    _identifier(e, "pmcid", get("pmcid"), normalize_pmcid)
    e.source_id = key or None
    kw = get("keywords") or ""
    e.keywords = [k.strip() for k in re.split(r"[;,]", kw) if k.strip()]
    e.language = get("language")
    e.publication_types = [kind]
    return _finish(e)


# ---------------------------------------------------------------------- CSV
CSV_FIELDS = ("title", "abstract", "authors", "year", "journal", "doi", "pmid", "pmcid",
              "source_id", "keywords", "registry_id", "language")  # fmt: skip
# Header names seen in common exports (case-insensitive). Detected columns are shown to the
# reviewer before import; vendor formats change, so the mapping can be overridden.
CSV_ALIASES: dict[str, tuple[str, ...]] = {
    "title": ("title", "article title", "study title", "document title", "ti"),
    "abstract": ("abstract", "brief summary", "ab"),
    "authors": ("authors", "author", "author full names", "au"),
    "year": ("year", "publication year", "py", "pubyear"),
    "journal": ("source title", "journal", "journal title", "source", "so"),
    "doi": ("doi", "do"),
    "pmid": ("pmid", "pubmed id", "pubmed_id"),
    "pmcid": ("pmcid", "pmc id"),
    "source_id": ("eid", "accession number", "ut (unique wos id)", "id", "an"),
    "keywords": ("author keywords", "keywords", "kw"),
    "registry_id": ("nct number", "trial id", "registration number"),
    "language": ("language", "language of original document", "la"),
}


def detect_columns(headers: list[str]) -> dict[str, str]:
    lookup = {h.strip().lower(): h for h in headers}
    mapping: dict[str, str] = {}
    for target, aliases in CSV_ALIASES.items():
        for alias in aliases:
            if alias in lookup:
                mapping[target] = lookup[alias]
                break
    return mapping


def _getter(row: dict[str, str], mapping: dict[str, str]) -> Callable[[str], str | None]:
    def get(name: str) -> str | None:
        return (row.get(mapping[name]) or None) if name in mapping else None

    return get


def parse_csv(
    text: str, column_map: dict[str, str] | None = None
) -> tuple[list[ParsedEntry], dict[str, str]]:
    sample = text[:4096]
    try:
        dialect: type[csv.Dialect] | csv.Dialect = csv.Sniffer().sniff(sample, ",;\t")
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    headers = list(reader.fieldnames or [])
    if not headers:
        raise ImportFileError("the CSV file has no header row")
    mapping = detect_columns(headers)
    for target, column in (column_map or {}).items():
        if target not in CSV_FIELDS:
            raise ImportFileError(f"unknown target field {target!r}")
        if column and column not in headers:
            raise ImportFileError(f"column {column!r} is not in the file")
        if column:
            mapping[target] = column
        else:
            mapping.pop(target, None)
    if "title" not in mapping:
        raise ImportFileError("no title column detected; map one explicitly")
    entries: list[ParsedEntry] = []
    for row in reader:
        clean_row = {k: (v or "") for k, v in row.items() if k is not None}
        if not any(v.strip() for v in clean_row.values()):
            continue  # blank line
        get = _getter(clean_row, mapping)
        e = ParsedEntry(position=len(entries) + 1, raw=clean_row)
        e.title = get("title")
        e.abstract = get("abstract")
        for name in re.split(r";", get("authors") or ""):
            if author := _author(name):
                e.authors.append(author)
        e.journal = _clean(get("journal"))
        e.year = _year(get("year"))
        _identifier(e, "doi", get("doi"), normalize_doi)
        _identifier(e, "pmid", get("pmid"), normalize_pmid)
        _identifier(e, "pmcid", get("pmcid"), normalize_pmcid)
        e.source_id = _clean(get("source_id")) or _clean(get("registry_id"))
        e.keywords = [k.strip() for k in re.split(r";", get("keywords") or "") if k.strip()]
        e.language = _clean(get("language"))
        e.registry_ids = extract_registry_ids(get("registry_id"))
        entries.append(_finish(e))
    return entries, mapping


def parse_file(
    data: bytes, fmt: ImportFormat, *, column_map: dict[str, str] | None = None
) -> ParsedFile:
    text, encoding = decode(data)
    if fmt == "ris":
        return ParsedFile("ris", encoding, parse_ris(text))
    if fmt == "bibtex":
        return ParsedFile("bibtex", encoding, parse_bibtex(text))
    if fmt == "csv":
        entries, mapping = parse_csv(text, column_map)
        return ParsedFile("csv", encoding, entries, mapping)
    raise ImportFileError(f"unsupported format {fmt!r}")


def guess_format(filename: str) -> ImportFormat | None:
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    mapping: dict[str, ImportFormat] = {"ris": "ris", "bib": "bibtex", "bibtex": "bibtex",
                                        "csv": "csv"}  # fmt: skip
    return mapping.get(suffix)
