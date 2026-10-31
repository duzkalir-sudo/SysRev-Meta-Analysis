"""Deterministic PRISMA 2020 flow diagrams from the stored, validated PRISMA snapshot.

The diagram follows the PRISMA 2020 template for *new systematic reviews which included
searches of databases and registers only* (Page MJ, et al. BMJ 2021;372:n71, Fig 1; CC BY
4.0). Every count is read from the stored :class:`PRISMACounts` snapshot (itself derived
from persisted records, decisions, the full-text manifest and report-to-study links); the
diagram never computes, adjusts or repairs a count.

Other template variants (identification via other methods such as websites, organisations
and citation searching; updated reviews with studies from a previous version) need data this
project does not record. Requesting them raises :class:`UnsupportedVariantError` listing the
fields that cannot be filled — nothing is invented to populate a template.

The *source* of a diagram is a JSON specification (counts, provenance, layout and every
line of text). The SVG (and, where matplotlib is installed, a PNG) is regenerated from that
source deterministically, so the figure is both editable (plain SVG text elements) and
reproducible from the stored snapshot.
"""

from __future__ import annotations

import hashlib
import io
import json
import textwrap
from dataclasses import dataclass
from typing import Any
from xml.sax.saxutils import escape

from src.core.enums import SourceDatabase
from src.core.schemas.pipeline import PRISMACounts
from src.core.schemas.stats import canonical_sha256
from src.search.databases import DATABASES

GENERATOR = "src.reporting.prisma_flow 1.0.0"
SPEC_SCHEMA = "prisma-flow-spec-1"
SUPPORTED_VARIANT = "new_review_databases_registers"
TEMPLATE_NAME = (
    "PRISMA 2020 flow diagram for new systematic reviews which included searches of "
    "databases and registers only"
)
ATTRIBUTION = (
    "Template: Page MJ, McKenzie JE, Bossuyt PM, et al. BMJ 2021;372:n71 (CC BY 4.0). "
    "Counts derived from the project's stored PRISMA snapshot."
)
FOOTNOTE_SOURCES = (
    "*Consider, if feasible to do so, reporting the number of records identified from each "
    "database or register searched (rather than the total number across all "
    "databases/registers)."
)
FOOTNOTE_AUTOMATION = (
    "**If automation tools were used, indicate how many records were excluded by a human and "
    "how many were excluded by automation tools."
)
# Fields of the other official variants that this project cannot fill from its records.
UNSUPPORTED_FIELDS: dict[str, tuple[str, ...]] = {
    "new_review_databases_registers_other_methods": (
        "Records identified from: Websites (n = )",
        "Records identified from: Organisations (n = )",
        "Records identified from: Citation searching (n = )",
        "Reports sought for retrieval (other methods) (n = )",
        "Reports not retrieved (other methods) (n = )",
        "Reports assessed for eligibility (other methods) (n = )",
        "Reports excluded (other methods), with reasons (n = )",
    ),
    "updated_review_databases_registers": (
        "Studies included in previous version of review (n = )",
        "Reports of studies included in previous version of review (n = )",
        "Total studies included in review (n = )",
        "Reports of total included studies (n = )",
    ),
}
UNSUPPORTED_FIELDS["updated_review_databases_registers_other_methods"] = (
    UNSUPPORTED_FIELDS["updated_review_databases_registers"]
    + UNSUPPORTED_FIELDS["new_review_databases_registers_other_methods"]
)
VARIANTS = (SUPPORTED_VARIANT, *UNSUPPORTED_FIELDS)
WHY_UNSUPPORTED = {
    "other_methods": "identification via other methods (websites, organisations, citation "
    "searching) is not recorded as a separate source type, and those reports are not tracked "
    "through a separate retrieval and eligibility flow",
    "updated": "no previous version of the review (its included studies and reports) is "
    "recorded in this project",
}

# Layout (pixels).
WIDTH = 860
BAR_X, BAR_W = 12, 34
LEFT_X, BOX_W = 64, 340
RIGHT_X = 470
PAD = 10
LINE_H = 16
FONT_SIZE = 12
GAP = 34
WRAP = 50


class UnsupportedVariantError(ValueError):
    def __init__(self, variant: str, fields: tuple[str, ...], reason: str) -> None:
        self.variant = variant
        self.fields = fields
        self.reason = reason
        super().__init__(
            f"the PRISMA 2020 '{variant}' diagram cannot be represented from this project's "
            f"data ({reason}); unsupported fields: " + "; ".join(fields)
        )


class RenderingUnavailableError(RuntimeError):
    """A raster export needs an optional dependency that is not installed."""


@dataclass(frozen=True, slots=True)
class ExclusionBreakdown:
    """Title/abstract exclusions by who decided them (derived from stored decisions)."""

    by_human: int
    by_automation: int


def _n(value: int) -> str:
    return f"(n = {value:,})"


def _label(db: SourceDatabase) -> str:
    info = DATABASES.get(db)
    return info.label if info else db.value


def _wrap(lines: list[str]) -> list[str]:
    out: list[str] = []
    for line in lines:
        indent = len(line) - len(line.lstrip(" "))
        wrapped = textwrap.wrap(line.strip(), WRAP - indent) or [""]
        out += [" " * indent + w for w in wrapped]
    return out


def flow_counts(prisma: PRISMACounts) -> dict[str, int]:
    """Every count shown in the diagram, by box field (used by the Auditor)."""
    return {
        "identified_databases": sum(prisma.identified_by_database.values()),
        "identified_registers": prisma.identified_from_registers,
        "duplicates_removed": prisma.duplicates_removed,
        "marked_ineligible_by_automation": prisma.marked_ineligible_by_automation,
        "removed_other_reasons": prisma.removed_other_reasons,
        "records_screened": prisma.records_screened,
        "records_excluded": prisma.records_excluded,
        "reports_sought": prisma.reports_sought,
        "reports_not_retrieved": prisma.reports_not_retrieved,
        "reports_assessed": prisma.reports_assessed,
        "reports_excluded": sum(prisma.reports_excluded_by_reason.values()),
        "studies_included": prisma.studies_included,
        "reports_included": prisma.reports_included,
    }


def build_flow_spec(
    prisma: PRISMACounts,
    *,
    searched_sources: frozenset[str],
    problems: list[str],
    breakdown: ExclusionBreakdown | None = None,
    variant: str = SUPPORTED_VARIANT,
) -> dict[str, Any]:
    """The diagram source. ``searched_sources`` are the source-database values with a logged
    search or import; ``problems`` are the snapshot's differences from the stored records
    (shown as a visible warning, never repaired)."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown PRISMA 2020 diagram variant {variant!r}; known: {VARIANTS}")
    if variant != SUPPORTED_VARIANT:
        why = []
        if variant.startswith("updated"):
            why.append(WHY_UNSUPPORTED["updated"])
        if variant.endswith("other_methods"):
            why.append(WHY_UNSUPPORTED["other_methods"])
        reason = "; ".join(why)
        raise UnsupportedVariantError(variant, UNSUPPORTED_FIELDS[variant], reason)

    warnings = list(problems)
    if not prisma.is_final:
        warnings.append(
            "the PRISMA snapshot is not final: "
            f"{prisma.records_awaiting_screening} record(s) awaiting screening, "
            f"{prisma.reports_awaiting_retrieval} report(s) awaiting retrieval, "
            f"{prisma.reports_awaiting_assessment} report(s) awaiting assessment"
        )
    for v in prisma.invariant_violations():
        warnings.append(f"PRISMA invariant violated: {v}")

    databases = {
        db: n
        for db, n in sorted(prisma.identified_by_database.items(), key=lambda x: x[0].value)
        if not (DATABASES.get(db) and DATABASES[db].is_register)
    }
    unclassified = sorted(db.value for db in databases if db not in DATABASES)
    if unclassified:
        warnings.append(
            "records from source(s) "
            + ", ".join(unclassified)
            + " are counted in the databases box by the PRISMA snapshot; their source type "
            "is not recorded, so identification via other methods cannot be distinguished"
        )
    registers_searched = sorted(
        s
        for s in searched_sources
        if SourceDatabase(s) in DATABASES and DATABASES[SourceDatabase(s)].is_register
    )

    ident = ["Records identified from*:"]
    databases_searched = any(
        not (SourceDatabase(s) in DATABASES and DATABASES[SourceDatabase(s)].is_register)
        for s in searched_sources
    )
    if databases or databases_searched:
        ident.append(f"Databases {_n(sum(databases.values()))}")
        ident += [f"  {_label(db)} {_n(n)}" for db, n in databases.items()]
    else:
        ident.append("Databases: none searched")
    if registers_searched or prisma.identified_from_registers:
        ident.append(f"Registers {_n(prisma.identified_from_registers)}")
        ident += [f"  {DATABASES[SourceDatabase(s)].label}" for s in registers_searched]
    else:
        ident.append("Registers: none searched")
    removed = [
        "Records removed before screening:",
        f"  Duplicate records removed {_n(prisma.duplicates_removed)}",
        "  Records marked as ineligible by automation tools "
        f"{_n(prisma.marked_ineligible_by_automation)}",
        f"  Records removed for other reasons {_n(prisma.removed_other_reasons)}",
    ]
    excluded = ["Records excluded**", _n(prisma.records_excluded)]
    if breakdown is not None and breakdown.by_automation:
        excluded += [
            f"  by a human {_n(breakdown.by_human)}",
            f"  by automation tools {_n(breakdown.by_automation)}",
        ]
        if breakdown.by_human + breakdown.by_automation != prisma.records_excluded:
            warnings.append(
                "the human/automation breakdown of title/abstract exclusions does not add up "
                "to the records excluded in the snapshot"
            )
    reasons = sorted(prisma.reports_excluded_by_reason.items(), key=lambda x: (-x[1], x[0]))
    rep_excl = ["Reports excluded:"]
    rep_excl += [f"  {r} {_n(n)}" for r, n in reasons] or [f"  {_n(0)}"]

    boxes_text: dict[str, list[str]] = {
        "identified": ident,
        "removed": removed,
        "screened": ["Records screened", _n(prisma.records_screened)],
        "excluded": excluded,
        "sought": ["Reports sought for retrieval", _n(prisma.reports_sought)],
        "not_retrieved": ["Reports not retrieved", _n(prisma.reports_not_retrieved)],
        "assessed": ["Reports assessed for eligibility", _n(prisma.reports_assessed)],
        "reports_excluded": rep_excl,
        "included": [
            "Studies included in review",
            _n(prisma.studies_included),
            "Reports of included studies",
            _n(prisma.reports_included),
        ],
    }
    rows = [
        ("Identification", "identified", "removed"),
        ("Screening", "screened", "excluded"),
        ("Screening", "sought", "not_retrieved"),
        ("Screening", "assessed", "reports_excluded"),
        ("Included", "included", None),
    ]
    banner = _wrap([f"WARNING: {w}" for w in warnings]) if warnings else []
    y = 20 + (len(banner) * LINE_H + 2 * PAD + 12 if banner else 0)
    header_y = y
    y += 30 + 16
    boxes: list[dict[str, Any]] = []
    arrows: list[dict[str, Any]] = []
    phases: dict[str, list[float]] = {}
    for phase, left, right in rows:
        left_lines = _wrap(boxes_text[left])
        right_lines = _wrap(boxes_text[right]) if right else []
        h = max(len(left_lines), len(right_lines)) * LINE_H + 2 * PAD
        boxes.append({"id": left, "x": LEFT_X, "y": y, "w": BOX_W, "h": h, "lines": left_lines})
        if right:
            boxes.append(
                {"id": right, "x": RIGHT_X, "y": y, "w": BOX_W, "h": h, "lines": right_lines}
            )
            arrows.append({"x1": LEFT_X + BOX_W, "y1": y + h / 2, "x2": RIGHT_X, "y2": y + h / 2})
        span = phases.setdefault(phase, [y, y + h])
        span[1] = y + h
        y += h + GAP
    for upper, lower in zip(
        [b for b in boxes if b["x"] == LEFT_X],
        [b for b in boxes if b["x"] == LEFT_X][1:],
        strict=False,
    ):
        x = LEFT_X + BOX_W / 2
        arrows.append({"x1": x, "y1": upper["y"] + upper["h"], "x2": x, "y2": lower["y"]})
    notes = _wrap([FOOTNOTE_SOURCES, FOOTNOTE_AUTOMATION, ATTRIBUTION])
    height = y + len(notes) * 14 + 20
    spec: dict[str, Any] = {
        "schema": SPEC_SCHEMA,
        "generator": GENERATOR,
        "standard": "PRISMA 2020 (Page et al., BMJ 2021;372:n71)",
        "template": TEMPLATE_NAME,
        "variant": variant,
        "snapshot_sha256": canonical_sha256(prisma.model_dump(mode="json")),
        "counts": flow_counts(prisma),
        "per_source": {db.value: n for db, n in databases.items()},
        "registers_searched": registers_searched,
        "status": "warning" if warnings else "consistent",
        "warnings": warnings,
        "canvas": {"width": WIDTH, "height": height},
        "header": {
            "y": header_y,
            "text": "Identification of studies via databases and registers",
        },
        "banner": banner,
        "phases": [{"label": k, "y0": v[0], "y1": v[1]} for k, v in phases.items()],
        "boxes": boxes,
        "arrows": arrows,
        "notes": notes,
    }
    return spec


def spec_sha256(spec: dict[str, Any]) -> str:
    return canonical_sha256(spec)


def svg_from_spec(spec: dict[str, Any]) -> str:
    """Regenerate the SVG from its source specification (byte-for-byte deterministic)."""
    if spec.get("schema") != SPEC_SCHEMA:
        raise ValueError("not a PRISMA flow-diagram specification")
    w, h = spec["canvas"]["width"], spec["canvas"]["height"]
    out = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{FONT_SIZE}">',
        f"<title>{escape(spec['template'])}</title>",
        f"<desc>{escape(spec['generator'])}; snapshot sha256 {spec['snapshot_sha256']}; "
        f"status {spec['status']}</desc>",
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        'markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" '
        'fill="#000"/></marker></defs>',
        f'<rect x="0" y="0" width="{w}" height="{h}" fill="#ffffff"/>',
    ]
    if spec["banner"]:
        bh = len(spec["banner"]) * LINE_H + 2 * PAD
        out.append(
            f'<rect x="{LEFT_X}" y="20" width="{RIGHT_X + BOX_W - LEFT_X}" height="{bh}" '
            'fill="#fdecea" stroke="#b00020" stroke-width="1.5"/>'
        )
        for i, line in enumerate(spec["banner"]):
            out.append(
                f'<text x="{LEFT_X + PAD}" y="{20 + PAD + (i + 1) * LINE_H - 4}" '
                f'fill="#b00020" font-weight="bold">{escape(line)}</text>'
            )
    hy = spec["header"]["y"]
    out.append(
        f'<rect x="{LEFT_X}" y="{hy}" width="{RIGHT_X + BOX_W - LEFT_X}" height="30" '
        'fill="#fff2cc" stroke="#000" stroke-width="1"/>'
    )
    out.append(
        f'<text x="{(LEFT_X + RIGHT_X + BOX_W) / 2:g}" y="{hy + 20}" text-anchor="middle" '
        f'font-weight="bold">{escape(spec["header"]["text"])}</text>'
    )
    for p in spec["phases"]:
        y0, y1 = p["y0"], p["y1"]
        cx, cy = BAR_X + BAR_W / 2, (y0 + y1) / 2
        out.append(
            f'<rect x="{BAR_X}" y="{y0:g}" width="{BAR_W}" height="{y1 - y0:g}" rx="6" '
            'fill="#c6dbef" stroke="#000" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{cx:g}" y="{cy:g}" text-anchor="middle" dominant-baseline="middle" '
            f'font-weight="bold" transform="rotate(-90 {cx:g} {cy:g})">'
            f"{escape(p['label'])}</text>"
        )
    for b in spec["boxes"]:
        out.append(
            f'<rect id="{b["id"]}" x="{b["x"]}" y="{b["y"]:g}" width="{b["w"]}" '
            f'height="{b["h"]:g}" fill="#ffffff" stroke="#000" stroke-width="1"/>'
        )
        for i, line in enumerate(b["lines"]):
            indent = len(line) - len(line.lstrip(" "))
            out.append(
                f'<text x="{b["x"] + PAD + indent * 4}" '
                f'y="{b["y"] + PAD + (i + 1) * LINE_H - 4:g}">{escape(line.strip())}</text>'
            )
    for a in spec["arrows"]:
        out.append(
            f'<line x1="{a["x1"]:g}" y1="{a["y1"]:g}" x2="{a["x2"]:g}" y2="{a["y2"]:g}" '
            'stroke="#000" stroke-width="1.2" marker-end="url(#arrow)"/>'
        )
    ny = h - len(spec["notes"]) * 14 - 8
    for i, line in enumerate(spec["notes"]):
        out.append(f'<text x="{LEFT_X}" y="{ny + i * 14:g}" font-size="10">{escape(line)}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def png_from_spec(spec: dict[str, Any], dpi: int = 300) -> bytes:
    """Raster export (requires matplotlib, the optional ``stats`` extra)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
    except ImportError as exc:  # pragma: no cover - depends on the installation
        raise RenderingUnavailableError(
            "PNG export needs matplotlib (pip install -e '.[stats]'); the SVG is available"
        ) from exc
    w, h = spec["canvas"]["width"], spec["canvas"]["height"]
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    fs = FONT_SIZE * 0.75 * 100 / 72 * 0.96
    if spec["banner"]:
        bh = len(spec["banner"]) * LINE_H + 2 * PAD
        ax.add_patch(
            Rectangle(
                (LEFT_X, 20), RIGHT_X + BOX_W - LEFT_X, bh, facecolor="#fdecea", edgecolor="#b00020"
            )
        )
        for i, line in enumerate(spec["banner"]):
            ax.text(
                LEFT_X + PAD,
                20 + PAD + (i + 1) * LINE_H - 4,
                line,
                color="#b00020",
                fontsize=fs,
                weight="bold",
                va="baseline",
            )
    hy = spec["header"]["y"]
    ax.add_patch(
        Rectangle(
            (LEFT_X, hy), RIGHT_X + BOX_W - LEFT_X, 30, facecolor="#fff2cc", edgecolor="black"
        )
    )
    ax.text(
        (LEFT_X + RIGHT_X + BOX_W) / 2,
        hy + 20,
        spec["header"]["text"],
        ha="center",
        fontsize=fs,
        weight="bold",
        va="baseline",
    )
    for p in spec["phases"]:
        ax.add_patch(
            FancyBboxPatch(
                (BAR_X, p["y0"]),
                BAR_W,
                p["y1"] - p["y0"],
                boxstyle="round,pad=0,rounding_size=6",
                facecolor="#c6dbef",
                edgecolor="black",
            )
        )
        ax.text(
            BAR_X + BAR_W / 2,
            (p["y0"] + p["y1"]) / 2,
            p["label"],
            rotation=90,
            ha="center",
            va="center",
            fontsize=fs,
            weight="bold",
        )
    for b in spec["boxes"]:
        ax.add_patch(
            Rectangle((b["x"], b["y"]), b["w"], b["h"], facecolor="white", edgecolor="black")
        )
        for i, line in enumerate(b["lines"]):
            indent = len(line) - len(line.lstrip(" "))
            ax.text(
                b["x"] + PAD + indent * 4,
                b["y"] + PAD + (i + 1) * LINE_H - 4,
                line.strip(),
                fontsize=fs,
                va="baseline",
            )
    for a in spec["arrows"]:
        ax.add_patch(
            FancyArrowPatch(
                (a["x1"], a["y1"]),
                (a["x2"], a["y2"]),
                arrowstyle="-|>",
                mutation_scale=10,
                color="black",
            )
        )
    ny = h - len(spec["notes"]) * 14 - 8
    for i, line in enumerate(spec["notes"]):
        ax.text(LEFT_X, ny + i * 14, line, fontsize=fs * 0.83, va="baseline")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, metadata={"Software": None})
    plt.close(fig)
    return buf.getvalue()


def spec_json(spec: dict[str, Any]) -> str:
    return json.dumps(spec, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def sha256_text(text: str | bytes) -> str:
    data = text.encode("utf-8") if isinstance(text, str) else text
    return hashlib.sha256(data).hexdigest()


__all__ = [
    "SUPPORTED_VARIANT",
    "VARIANTS",
    "ExclusionBreakdown",
    "RenderingUnavailableError",
    "UnsupportedVariantError",
    "build_flow_spec",
    "flow_counts",
    "png_from_spec",
    "svg_from_spec",
]
