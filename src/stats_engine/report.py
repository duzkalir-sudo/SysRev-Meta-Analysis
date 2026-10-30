"""Human-readable renderings of ``stats_output.json``: tables, a report and forest plots.

Everything here formats numbers that the engine already computed; nothing is recomputed.
Rounding happens only for display (``settings.display_decimals``). Forest plots need the
optional ``matplotlib`` dependency (``pip install -e ".[stats]"``) and are written as SVG
with fixed metadata and hash salt so identical results give identical files.
"""

from __future__ import annotations

import csv
import io
import math
from pathlib import Path
from typing import Any

from src.core.enums import EffectMeasure
from src.core.schemas.stats import AnalysisResult, PooledResult, StatsOutput

RATIO_MEASURES = frozenset({EffectMeasure.RR, EffectMeasure.OR})
PROBABILITY_MEASURES = frozenset(
    {EffectMeasure.PROPORTION, EffectMeasure.KM_PROBABILITY, EffectMeasure.CUMULATIVE_INCIDENCE}
)


def _fmt(x: float | None, d: int) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x:.{d}f}"


def _p(p: float) -> str:
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def _pct_scale(a: AnalysisResult) -> float:
    return 100.0 if a.effect_measure in PROBABILITY_MEASURES else 1.0


def _est(a: AnalysisResult, est: float | None, lo: float | None, hi: float | None) -> str:
    d = a.settings.get("display_decimals", 2)
    m = _pct_scale(a)
    unit = "%" if m == 100.0 else ""
    if est is None:
        return "—"
    return (
        f"{_fmt(est * m, d)}{unit} [{_fmt(None if lo is None else lo * m, d)}, "
        f"{_fmt(None if hi is None else hi * m, d)}]"
    )


def _pooled_rows(a: AnalysisResult) -> list[tuple[str, PooledResult]]:
    rows: list[tuple[str, PooledResult]] = []
    if a.common_effect is not None:
        rows.append(("Common effect", a.common_effect))
    if a.random_effects is not None:
        rows.append(("Random effects", a.random_effects))
    rows.extend((f"Sensitivity: {r.model.replace('_', ' ')}", r) for r in a.sensitivity)
    return rows


def analysis_markdown(a: AnalysisResult) -> str:
    d = a.settings.get("display_decimals", 2)
    level = a.settings.get("confidence_level", 0.95) * 100
    lines = [
        f"### {a.analysis_id}: {a.outcome_name} — {a.effect_measure}",
        "",
        f"* Status: **{a.status}** — {a.status_reason}",
        f"* Timepoint(s): {', '.join(a.timepoints) or 'not specified (comparative outcome)'}",
        f"* Studies included: {a.k}; excluded: {len(a.excluded)}",
    ]
    if a.total_events is not None:
        lines.append(
            f"* Events / participants in included studies: "
            f"{a.total_events} / {a.total_participants}"
        )
    lines += [
        "",
        f"| Study | Data | Estimate [{level:g}% CI] | Weight CE % | Weight RE % | Notes |",
        "|---|---|---|---|---|---|",
    ]
    for s in a.studies:
        data = ", ".join(f"{k}={v:g}" for k, v in s.data.items())
        lines.append(
            f"| {s.label} | {data} | "
            f"{_est(a, s.estimate_natural, s.ci_lower_natural, s.ci_upper_natural)} | "
            f"{_fmt(s.weight_common_pct, 1)} | {_fmt(s.weight_random_pct, 1)} | "
            f"{'; '.join(s.notes)} |"
        )
    for name, r in _pooled_rows(a):
        lines.append(
            f"| **{name}** ({r.method}) | k={r.k} | "
            f"**{_est(a, r.estimate_natural, r.ci_lower_natural, r.ci_upper_natural)}** "
            f"p={_p(r.p_value)} | | | |"
        )
    re = a.random_effects
    if re is not None and re.prediction_interval_natural is not None:
        lo, hi = re.prediction_interval_natural
        lines.append(
            f"| Prediction interval (t, df={re.prediction_df}) | | "
            f"{_est(a, (lo + hi) / 2, lo, hi).split(' ', 1)[1]} | | | |"
        )
    if a.heterogeneity is not None:
        h = a.heterogeneity
        lines += [
            "",
            f"Heterogeneity ({h.basis}): Q = {_fmt(h.q, d)} (df = {h.df}, "
            f"p = {_p(h.p_value)}); I² = {_fmt(h.i2, 1)}%.",
        ]
        if re is not None and re.tau2 is not None:
            ci = re.tau2_ci
            ci_txt = f" [{_fmt(ci[0], 4)}, {_fmt(ci[1], 4)}]" if ci else ""
            lines.append(
                f"τ² = {_fmt(re.tau2, 4)}{ci_txt} (Q-profile CI); model I² = "
                f"{_fmt(re.i2_model, 1)}%."
            )
    if a.excluded:
        lines += ["", "Excluded studies:", "", "| Study | Reason code | Reason |", "|---|---|---|"]
        lines += [f"| {e.label} | {e.reason_code} | {_cell(e.reason)} |" for e in a.excluded]
    if a.reports:
        lines += [
            "",
            f"Report ledger (every full-text-eligible report; {len(a.reports)} reports, "
            f"{sum(r.status == 'included' for r in a.reports)} contributing). "
            "`outcome_data_not_extracted_or_unverified` means no approved, verified extraction "
            "exists; it does not mean the paper failed to report the outcome.",
            "",
            "| Report | Record | Study | Cohort | Status | Analysis unit | Reason code | Reason |",
            "|---|---|---|---|---|---|---|---|",
        ]
        lines += [
            f"| {r.label} | {r.record_id} | {r.study_id} | {r.cohort_id} | {r.status} | "
            f"{r.unit_id or ''} | {r.reason_code or ''} | {_cell(r.reason or '')} |"
            for r in a.reports
        ]
    for title, items in (
        ("Methods", a.methods),
        ("Warnings", a.warnings),
        ("Limitations", a.limitations),
    ):
        if items:
            lines += ["", f"{title}:", ""] + [f"* {x}" for x in items]
    return "\n".join(lines) + "\n"


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def analysis_csv(a: AnalysisResult) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(
        [
            "row_type",
            "label",
            "study_id",
            "record_ids",
            "yi",
            "vi",
            "estimate",
            "ci_lower",
            "ci_upper",
            "weight_common_pct",
            "weight_random_pct",
            "method",
            "p_value",
            "reason_code",
            "reason",
            "cohort_id",
            "report_status",
            "unit_id",
        ]
    )
    for s in a.studies:
        w.writerow(
            [
                "study",
                s.label,
                s.study_id,
                ";".join(s.record_ids),
                repr(s.yi),
                repr(s.vi),
                repr(s.estimate_natural),
                repr(s.ci_lower_natural),
                repr(s.ci_upper_natural),
                repr(s.weight_common_pct),
                repr(s.weight_random_pct),
                s.ci_method,
                "",
                "",
                "",
                "",
                "",
                "",
            ]
        )
    for name, r in _pooled_rows(a):
        w.writerow(
            [
                "pooled",
                name,
                "",
                "",
                repr(r.estimate),
                repr(r.se**2),
                repr(r.estimate_natural),
                repr(r.ci_lower_natural),
                repr(r.ci_upper_natural),
                "",
                "",
                r.method,
                repr(r.p_value),
                "",
                "",
                "",
                "",
                "",
            ]
        )
    for e in a.excluded:
        w.writerow(
            [
                "excluded",
                e.label,
                e.study_id,
                ";".join(e.record_ids),
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                e.reason_code,
                e.reason,
                "",
                "",
                "",
            ]
        )
    blank = [""] * 9
    for rr in a.reports:  # the complete per-report ledger
        w.writerow(
            ["report", rr.label, rr.study_id, rr.record_id, *blank,
             rr.reason_code or "", rr.reason or "", rr.cohort_id, rr.status, rr.unit_id or ""]
        )  # fmt: skip
    return buf.getvalue()


def report_markdown(output: StatsOutput) -> str:
    lines = [
        "# Stage 5 statistics report",
        "",
        "Generated deterministically by `src.stats_engine`; no number in this report was "
        "produced by a language model.",
        "",
        f"* Output id: `{output.output_id}`; schema {output.schema_version}; engine "
        f"{output.engine_version}",
        f"* Plan: `{output.plan_id}` (SHA-256 `{output.plan_sha256}`)",
        f"* Input SHA-256: `{output.input_sha256}`",
        f"* Results SHA-256: `{output.results_sha256}`",
        f"* Software: {', '.join(f'{k} {v}' for k, v in sorted(output.software.items()))}",
        f"* Precision: {output.numeric_precision}",
        "",
        "## Summary",
        "",
        "| Analysis | Outcome | Measure | Status | k | Excluded |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {a.analysis_id} | {a.outcome_name} | {a.effect_measure} | {a.status} | {a.k} | "
        f"{len(a.excluded)} |"
        for a in output.analyses
    ]
    lines += ["", "## Analyses", ""]
    lines += [analysis_markdown(a) for a in output.analyses]
    lines += ["## Global limitations", ""] + [f"* {x}" for x in output.limitations]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- forest plots


def forest_plot(a: AnalysisResult, path: Path) -> Path | None:
    """Write an SVG forest plot for a pooled or single-study analysis (else ``None``)."""
    if a.status not in ("pooled", "single_study") or not a.studies:
        return None
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams["svg.hashsalt"] = "sysrev-stats-engine"
    plt.rcParams["svg.fonttype"] = "none"
    d = int(a.settings.get("display_decimals", 2))
    m = _pct_scale(a)
    log_x = a.effect_measure in RATIO_MEASURES
    rows: list[dict[str, Any]] = []
    for s in a.studies:
        w = (
            s.weight_random_pct
            if a.primary and a.primary.model == "random_effects"
            else (s.weight_common_pct)
        )
        rows.append(
            {
                "label": s.label,
                "est": s.estimate_natural,
                "lo": s.ci_lower_natural,
                "hi": s.ci_upper_natural,
                "w": w,
                "kind": "study",
            }
        )
    for name, r in (("Common effect", a.common_effect), ("Random effects", a.random_effects)):
        if r is not None:
            rows.append(
                {
                    "label": f"{name} ({r.method.split(',')[0]})",
                    "est": r.estimate_natural,
                    "lo": r.ci_lower_natural,
                    "hi": r.ci_upper_natural,
                    "kind": "pooled",
                    "pi": r.prediction_interval_natural,
                }
            )
    n = len(rows)
    fig, ax = plt.subplots(figsize=(9.0, 1.2 + 0.32 * n))
    for i, row in enumerate(rows):
        y = n - i
        est, lo, hi = (None if v is None else v * m for v in (row["est"], row["lo"], row["hi"]))
        if est is None or lo is None or hi is None:
            continue
        if row["kind"] == "study":
            ax.plot([lo, hi], [y, y], color="black", linewidth=1)
            size = 4 + 10 * math.sqrt(max(row["w"] or 0.0, 0.0) / 100)
            ax.plot([est], [y], marker="s", color="black", markersize=size, linestyle="none")
        else:
            ax.fill([lo, est, hi, est], [y, y + 0.25, y, y - 0.25], color="black")
            pi = row.get("pi")
            if pi is not None:
                ax.plot(
                    [pi[0] * m, pi[1] * m],
                    [y - 0.35, y - 0.35],
                    color="grey",
                    linewidth=2,
                    solid_capstyle="butt",
                )
        unit = "%" if m == 100.0 else ""
        text = f"{est:.{d}f}{unit} [{lo:.{d}f}, {hi:.{d}f}]"
        if row["kind"] == "study" and row["w"] is not None:
            text += f"  {row['w']:.1f}%"
        ax.annotate(text, xy=(1.01, y), xycoords=("axes fraction", "data"), va="center", fontsize=8)
    ax.set_yticks([n - i for i in range(n)])
    ax.set_yticklabels([r["label"] for r in rows], fontsize=8)
    ax.set_ylim(0.3, n + 0.7)
    if log_x:
        ax.set_xscale("log")
        ax.axvline(1.0, color="grey", linestyle=":", linewidth=1)
    elif a.effect_measure in (EffectMeasure.MD, EffectMeasure.SMD):
        ax.axvline(0.0, color="grey", linestyle=":", linewidth=1)
    ax.set_xlabel(f"{a.effect_measure}" + (" (%)" if m == 100.0 else ""))
    ax.set_title(
        f"{a.outcome_name} — {', '.join(a.timepoints) or 'all timepoints as planned'}", fontsize=9
    )
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.subplots_adjust(left=0.32, right=0.70)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", metadata={"Date": None, "Creator": None})
    plt.close(fig)
    return path
