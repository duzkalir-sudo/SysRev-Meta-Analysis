"""Journal-ready display of numbers that the statistics engine already computed.

Nothing here computes a result: values are only rounded for display. Every displayed number
is recorded as a :class:`DisplayNumber` holding the full-precision source value and the
display precision, so the Auditor can check the printed text with an explicit tolerance:

* a number shown with *d* decimals must lie within ``0.5 × 10⁻ᵈ`` of the source value
  (expressed in the displayed unit, e.g. percent), i.e. it is a correct rounding;
* counts are integers and must match exactly (tolerance 0);
* ``p < 0.001`` is correct only if the source p-value is below 0.001.

Display rules (fixed, documented in ARCHITECTURE.md §9.8):

* probabilities (crude proportions, Kaplan–Meier, cumulative incidence) are shown as
  percentages with one decimal place;
* ratios, differences, τ², Q use the analysis plan's ``display_decimals`` (τ² uses 4);
* I² is shown as a whole-number percentage;
* p-values are shown with three decimals, or as ``p < 0.001``;
* negative numbers use the Unicode minus sign; intervals are written "X to Y" so a hyphen
  is never confused with a minus sign.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

MINUS = "−"

SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")

# A number as it can appear in biomedical manuscript text:
# * optional minus (Unicode minus, or an ASCII hyphen not preceded by a word character, so a
#   range "3-5" is not read as 3 and -5);
# * grouped thousands, decimal point or Lancet mid-dot, or a leading decimal (".032");
# * an optional exponent: "3.2e-5", "3.2E−05", "3.2 × 10^-5", "3.2 × 10⁻⁵".
# Digits that are part of a word or a hyphenated term ("HbA1c", "ICD-10", "COVID-19",
# "SF-36", "IL-6") are not numbers.
NUMBER_RE = re.compile(
    r"(?<![\w.,·/])(?<![A-Za-z][-‐])(?P<sign>[\u2212-])?"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:[.·]\d+)?|\d+(?:[.·]\d+)?|[.·]\d+)"
    r"(?P<exp>[eE][\u2212+-]?\d+|\s?[×x]\s?10(?:\^[\u2212+-]?\d+|[⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+))?"
    r"(?![\w])"
)
# Identifiers that contain digits but are not numerical claims: DOIs, PMIDs/PMCIDs, URLs,
# registration numbers and bracketed citation numbers (reported separately as citations).
IDENTIFIER_RE = re.compile(
    r"\b10\.\d{4,9}/\S+|\bdoi:\s*\S+|https?://\S+|www\.\S+|\bPMID:?\s*\d+|\bPMC\d+|"
    r"\bCRD\d+|\bNCT\d+|\[\s*\d+(?:\s*[,–-]\s*\d+)*\s*\]",
    re.IGNORECASE,
)


def mask_identifiers(text: str) -> str:
    """Blank out identifiers (same length, so offsets are preserved)."""
    return IDENTIFIER_RE.sub(lambda m: " " * len(m.group(0)), text)


@dataclass(frozen=True, slots=True)
class DisplayNumber:
    """One number as displayed, with the full-precision value it represents."""

    role: str
    value: float  # full precision, already in the displayed unit (e.g. percent)
    decimals: int
    relation: Literal["eq", "lt", "gt"] = "eq"

    @property
    def tolerance(self) -> float:
        if self.relation != "eq":
            return 0.0
        return 0.5 * 10.0 ** (-self.decimals) + 1e-9 * max(1.0, abs(self.value))

    def displayed(self) -> str:
        return fmt(self.value, self.decimals)

    def accepts(self, shown: float) -> bool:
        """Whether ``shown`` (as printed) correctly represents this value."""
        if self.relation == "lt":
            return self.value < shown
        if self.relation == "gt":
            return self.value > shown
        if self.role == "count":
            return shown == self.value  # counts are integers: no rounding tolerance
        return abs(shown - self.value) <= self.tolerance

    def accepts_typed(self, typed: float, typed_decimals: int) -> bool:
        """Whether a number typed by a person (with its own precision) is a correct rounding
        of this source value — e.g. "6%" or "5.9%" for 5.94%, but not "3.2e-5" for 0."""
        if self.relation != "eq" or self.role == "count":
            return self.accepts(typed)
        tol = 0.5 * 10.0 ** (-typed_decimals) + 1e-9 * max(1.0, abs(self.value))
        return abs(typed - self.value) <= tol


def fmt(x: float, decimals: int) -> str:
    """Round for display only; thousands grouped from 10,000; Unicode minus."""
    if not math.isfinite(x):
        raise ValueError("non-finite values are never displayed as numbers")
    text = f"{x:,.{decimals}f}" if abs(x) >= 10000 else f"{x:.{decimals}f}"
    if text.startswith("-"):
        text = MINUS + text[1:]
        if float(text[1:].replace(",", "")) == 0:  # "-0.0" -> "0.0"
            text = text[1:]
    return text


def fmt_count(n: int) -> tuple[str, DisplayNumber]:
    return fmt(float(n), 0), DisplayNumber("count", float(n), 0)


def fmt_p(p: float) -> tuple[str, DisplayNumber]:
    if p < 0.001:
        return "p < 0.001", DisplayNumber("p_value", p, 3, relation="lt")
    return f"p = {fmt(p, 3)}", DisplayNumber("p_value", p, 3)


def parse_number(text: str) -> float:
    """Parse a displayed number (Unicode minus, grouping commas, mid-dot decimals)."""
    cleaned = text.replace(MINUS, "-").replace(",", "").replace("·", ".").strip()
    return float(cleaned)


@dataclass(frozen=True, slots=True)
class FoundNumber:
    start: int
    end: int
    text: str
    value: float
    decimals: int


def _exponent(exp: str | None) -> int:
    if not exp:
        return 0
    if exp[0] in "eE":
        return int(exp[1:].replace(MINUS, "-"))
    tail = exp.split("10", 1)[1].lstrip("^").translate(SUPERSCRIPTS).replace(MINUS, "-")
    return int(tail)


def find_numbers(text: str) -> list[FoundNumber]:
    """Every number in ``text``. ``decimals`` is the precision of the represented value, so
    "3.2e-5" has 6 decimals (tolerance 5e-7) and "4 × 10³" has −3."""
    out: list[FoundNumber] = []
    for m in NUMBER_RE.finditer(text):
        num = m.group("num")
        mantissa = parse_number((m.group("sign") or "") + ("0" + num if num[0] in ".·" else num))
        frac = re.split(r"[.·]", num)
        decimals = len(frac[1]) if len(frac) == 2 else 0
        exp = _exponent(m.group("exp"))
        value = float(f"{mantissa!r}e{exp}")  # exact decimal scaling, no binary drift
        out.append(FoundNumber(m.start(), m.end(), m.group(0), value, decimals - exp))
    return out


def numbers_in_text(text: str, role: str = "text") -> tuple[DisplayNumber, ...]:
    """Numbers inside a text fact (exact as written; checked by text identity)."""
    return tuple(DisplayNumber(role, f.value, f.decimals) for f in find_numbers(text))
