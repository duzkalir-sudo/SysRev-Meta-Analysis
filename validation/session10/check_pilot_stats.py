"""Session 10: check the statistics of a real pilot against an independent R/metafor run.

Usage (from the repository root; Rscript with metafor, lme4 and jsonlite must be on PATH):

    python -m validation.session10.check_pilot_stats <stats_dir> [--out <dir>] [--rscript PATH]

``<stats_dir>`` is the Stage 5 output directory of the pilot (it contains
``stats_output.json`` and ``manifest.json``). The check is read-only: it never opens the
application database and never writes into ``<stats_dir>``.

What is checked, per analysis with a pooled result (PROPORTION, RR, OR):

* integrity: the SHA-256 of ``stats_output.json`` equals the one in ``manifest.json``;
* every reported model (common effect, random effects, sensitivity) against the metafor
  model with the same method: estimate and confidence limits (analysis and natural scale),
  τ² where reported, and k with the study ids the model used;
* each model's population (k, participants, events) recomputed from the study-level data;
* the study weights shown with the common-effect and random-effects models;
* Cochran's Q, its p-value and I².

Tolerances are those of validation/VALIDATION_REPORT.md (closed form 1e-10, REML 1e-8,
GLMM 2e-4). A model whose method is not recognised is listed as *not compared*, never as
passed; any uncompared value makes the run *inconclusive* (exit 3), a failed comparison or a
missing/incorrect manifest binding *failed* (exit 1). Only exit 0 (*validated*) means the
reported statistics were reproduced. Writes ``pilot_stats_check.json`` and
``pilot_stats_check.md`` to ``--out``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
R_SCRIPT = HERE / "pilot_reference.R"
SUPPORTED = {"PROPORTION", "RR", "OR"}
LOGIT_PREFIX = "logit transform, 0.5 continuity correction for 0 % / 100 % studies; "
TOL = {"closed": (1e-10, 1e-9), "iterative": (1e-8, 1e-7), "glmm": (2e-4, 2e-4)}


def r_environment() -> tuple[dict[str, str], str]:
    """Environment for Rscript. R refuses a session temporary directory whose path contains
    a space ("'R_TempDir' contains space"); on POSIX /tmp is used instead in that case."""
    env = dict(os.environ)
    current = env.get("TMPDIR") or tempfile.gettempdir()
    if " " in current and os.name != "nt" and Path("/tmp").is_dir():
        env["TMPDIR"] = "/tmp"
    return env, env.get("TMPDIR", current)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_key(analysis: dict[str, Any], pooled: dict[str, Any]) -> tuple[str, str] | None:
    """metafor model and tolerance class matching an engine ``PooledResult``."""
    method = pooled["method"].removeprefix(LOGIT_PREFIX)  # logit-IV proportions
    tau2 = analysis["settings"].get("tau2_estimator", "REML")
    if method.startswith("binomial-normal GLMM"):
        return "glmm_random", "glmm"
    if method == "binomial maximum likelihood":
        return "glmm_common", "glmm"
    if method.startswith("Mantel-Haenszel"):
        return "mh_common", "closed"
    if method == "inverse variance":
        return "iv_common", "closed"
    if method.startswith("inverse variance, ") and pooled["model"] == "random_effects":
        if method != _re_label(analysis["settings"]):
            return None  # e.g. a sensitivity model with other settings: not compared
        return "iv_random", "iterative" if tau2 == "REML" else "closed"
    return None


def _re_label(settings: dict[str, Any]) -> str:
    ci = {
        "hksj": "Hartung-Knapp-Sidik-Jonkman t(k−1) CI",
        "hksj_adhoc": "ad hoc (never narrower than Wald) Hartung-Knapp CI",
        "wald": "Wald (normal) CI",
    }[settings.get("random_effects_ci", "hksj")]
    return f"inverse variance, {settings.get('tau2_estimator', 'REML')} τ², {ci}"


def r_input(stats: dict[str, Any]) -> dict[str, Any]:
    analyses = []
    for a in stats["analyses"]:
        if a["status"] != "pooled" or a["effect_measure"] not in SUPPORTED:
            continue
        s = a["settings"]
        analyses.append(
            {
                "analysis_id": a["analysis_id"],
                "measure": a["effect_measure"],
                "tau2_estimator": s.get("tau2_estimator", "REML"),
                "random_effects_ci": s.get("random_effects_ci", "hksj"),
                "confidence_level": s.get("confidence_level", 0.95),
                "studies": [{"study_id": st["study_id"], **st["data"]} for st in a["studies"]],
            }
        )
    return {"analyses": analyses}


def _close(ours: float | None, ref: float | None, cls: str) -> bool:
    if ours is None or ref is None:
        return ours is None and ref is None
    if math.isinf(ours) or math.isinf(ref):
        return ours == ref
    abs_tol, rel_tol = TOL[cls]
    return math.isclose(ours, ref, rel_tol=rel_tol, abs_tol=abs_tol)


def _totals(studies: list[dict[str, Any]]) -> tuple[int | None, int | None]:
    ev = n = 0
    for st in studies:
        d = st["data"]
        if "events_intervention" in d:
            ev += int(d["events_intervention"] + d["events_control"])
            n += int(d["n_intervention"] + d["n_control"])
        elif "events" in d and "n" in d:
            ev += int(d["events"])
            n += int(d["n"])
        else:
            return None, None
    return ev, n


def compare(stats: dict[str, Any], ref: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_id = {r["analysis_id"]: r for r in ref["analyses"]}

    def row(aid: str, what: str, ours: Any, theirs: Any, ok: bool | None, cls: str) -> None:
        status = "not compared" if ok is None else ("pass" if ok else "FAIL")
        rows.append(
            {
                "analysis_id": aid,
                "quantity": what,
                "engine": ours,
                "reference": theirs,
                "tolerance": cls,
                "status": status,
            }
        )

    for a in stats["analyses"]:
        aid = a["analysis_id"]
        if aid not in by_id:
            row(
                aid,
                "analysis",
                a["status"],
                None,
                None,
                f"status {a['status']} not pooled or measure {a['effect_measure']} not covered",
            )
            continue
        r = by_id[aid]
        studies = {s["study_id"]: s for s in a["studies"]}
        models = [
            ("common_effect", a.get("common_effect")),
            ("random_effects", a.get("random_effects")),
        ]
        models += [(f"sensitivity.{i + 1}", m) for i, m in enumerate(a.get("sensitivity") or [])]
        for name, p in models:
            if p is None:
                continue
            key = reference_key(a, p)
            label = f"{name} [{p['method']}]"
            if key is None or key[0] not in r or "error" in r[key[0]]:
                why = "no matching metafor model" if key is None else r.get(key[0], {})
                row(aid, label, p["estimate"], None, None, str(why))
                continue
            rk, cls = key
            m = r[rk]
            for f in (
                "estimate",
                "ci_lower",
                "ci_upper",
                "estimate_natural",
                "ci_lower_natural",
                "ci_upper_natural",
            ):
                row(aid, f"{label} {f}", p[f], m[f], _close(p[f], m[f], cls), cls)
            if p.get("tau2") is not None and m.get("tau2") is not None:
                row(
                    aid,
                    f"{label} tau2",
                    p["tau2"],
                    m["tau2"],
                    _close(p["tau2"], m["tau2"], cls),
                    cls,
                )
            row(aid, f"{label} k", p["k"], m["k"], p["k"] == m["k"], "exact")
            ref_ids = m["study_ids"] if isinstance(m["study_ids"], list) else [m["study_ids"]]
            if p.get("study_ids") is not None:
                row(
                    aid,
                    f"{label} study_ids",
                    p["study_ids"],
                    ref_ids,
                    sorted(p["study_ids"]) == sorted(ref_ids),
                    "exact",
                )
            # population recomputed from the listed study data
            members = [studies[i] for i in ref_ids if i in studies]
            ev, n = _totals(members)
            if p.get("total_participants") is not None:
                row(
                    aid,
                    f"{label} participants",
                    p["total_participants"],
                    n,
                    p["total_participants"] == n,
                    "exact",
                )
            if p.get("total_events") is not None:
                row(aid, f"{label} events", p["total_events"], ev, p["total_events"] == ev, "exact")
            # weights shown with the reported common/random model
            col = {"common_effect": "weight_common_pct", "random_effects": "weight_random_pct"}
            if name in col and "weights_pct" in m:
                w = m["weights_pct"] if isinstance(m["weights_pct"], list) else [m["weights_pct"]]
                ids = (
                    [s["study_id"] for s in a["studies"]]
                    if rk == "mh_common"
                    else [s["study_id"] for s in a["studies"] if s["yi"] is not None]
                )
                for sid, wr in zip(ids, w, strict=True):
                    ours = studies[sid].get(col[name])
                    row(aid, f"{label} weight {sid}", ours, wr, _close(ours, wr, cls), cls)
        h, rh = a.get("heterogeneity"), r.get("heterogeneity")
        if h and rh:
            for ours, theirs, what in (
                (h["q"], rh["Q"], "Q"),
                (h["p_value"], rh["Qp"], "Q p"),
                (h["i2"], rh["I2"], "I2"),
            ):
                row(aid, what, ours, theirs, _close(ours, theirs, "closed"), "closed")
            row(aid, "Q df", h["df"], rh["df"], h["df"] == rh["df"], "exact")
        # analysis-level totals count every listed study (the complete ledger)
        ev, n = _totals(a["studies"])
        if a.get("total_participants") is not None:
            row(
                aid,
                "analysis participants (all listed studies)",
                a["total_participants"],
                n,
                a["total_participants"] == n,
                "exact",
            )
        if a.get("total_events") is not None:
            row(
                aid,
                "analysis events (all listed studies)",
                a["total_events"],
                ev,
                a["total_events"] == ev,
                "exact",
            )
    return rows


HEX64 = re.compile(r"[0-9a-f]{64}")
EXIT = {"validated": 0, "failed": 1, "error": 2, "inconclusive": 3}


def manifest_problems(manifest: Any, digest: str) -> list[str]:
    """Is ``stats_output.json`` (SHA-256 ``digest``) the output this Stage 5 manifest names?

    Stage 5 (``src.stats_engine.stage5.write_artifacts``) records the output's hash twice:
    ``stats_output_sha256`` and ``files["stats_output.json"]``. Both must be present, well
    formed and equal to the file's hash. The digest appearing anywhere else (a note, another
    file's entry) does not bind the file (review of PR #15, P2)."""
    if not isinstance(manifest, dict):
        return ["manifest.json is not a JSON object"]
    problems = []
    named = manifest.get("stats_output_sha256")
    files = manifest.get("files")
    listed = files.get("stats_output.json") if isinstance(files, dict) else None
    for field, value in (("stats_output_sha256", named), ('files["stats_output.json"]', listed)):
        if value is None:
            problems.append(f"manifest field {field} is missing")
        elif not isinstance(value, str) or not HEX64.fullmatch(value):
            problems.append(f"manifest field {field} is not a SHA-256 hex digest")
        elif value != digest:
            problems.append(f"manifest field {field} = {value} does not match the file ({digest})")
    if not isinstance(files, dict):
        problems.append("manifest field files is missing or not an object")
    return problems


def verdict(rows: list[dict[str, Any]], binding: list[str]) -> str:
    """``validated`` only if the output is bound to its manifest, at least one value was
    compared, nothing failed and nothing was left uncompared. Uncompared analyses or models
    (e.g. an unsupported HR outcome) make the run ``inconclusive``: it completed, but did
    not validate everything the output reports."""
    if binding or any(x["status"] == "FAIL" for x in rows):
        return "failed"
    compared = [x for x in rows if x["status"] == "pass"]
    if not compared or any(x["status"] == "not compared" for x in rows):
        return "inconclusive"
    return "validated"


def main(argv: list[str] | None = None) -> int:
    """Exit status: 0 validated, 1 failed, 2 error (no reference could be computed),
    3 inconclusive. Only 0 means that the statistics were validated."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("stats_dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("session10_results"))
    ap.add_argument("--rscript", default=shutil.which("Rscript") or "Rscript")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):  # never fail on a narrow console code page
        sys.stdout.reconfigure(errors="backslashreplace")
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.stats_dir / "stats_output.json"
    digest = sha256_file(out_path)
    try:
        manifest: Any = json.loads((args.stats_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        manifest = None
        binding = [f"manifest.json could not be read ({type(exc).__name__})"]
    else:
        binding = manifest_problems(manifest, digest)
    stats = json.loads(out_path.read_text(encoding="utf-8"))
    rin = args.out / "pilot_r_input.json"
    rout = args.out / "pilot_r_reference.json"
    rin.write_text(json.dumps(r_input(stats), indent=2), encoding="utf-8")
    env, r_tmp = r_environment()
    proc = subprocess.run(
        [args.rscript, str(R_SCRIPT), str(rin), str(rout)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        env=env,
    )
    result: dict[str, Any] = {
        "stats_output_sha256": digest,
        "listed_in_manifest": not binding,
        "manifest_problems": binding,
        "engine_version": stats.get("engine_version"),
        "r_temp_dir": r_tmp,
    }
    if proc.returncode != 0:
        print(proc.stdout + proc.stderr, file=sys.stderr)
        if "R_TempDir" in proc.stdout + proc.stderr:
            print(
                "R cannot use a temporary directory whose path contains a space: set TMPDIR "
                "(Windows: TMP and TEMP) to a folder without spaces and run again.",
                file=sys.stderr,
            )
        result.update(
            verdict="error",
            r_exit_code=proc.returncode,
            r_output=(proc.stdout + proc.stderr)[-2000:],
        )
        (args.out / "pilot_stats_check.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("verdict: error (no reference values were computed)")
        return EXIT["error"]
    ref = json.loads(rout.read_text(encoding="utf-8"))
    rows = compare(stats, ref)
    outcome = verdict(rows, binding)
    validated = sorted(
        {x["analysis_id"] for x in rows if x["status"] == "pass"}
        - {x["analysis_id"] for x in rows if x["status"] != "pass"}
    )
    result.update(
        verdict=outcome,
        reference_software=ref["software"],
        comparisons=len([x for x in rows if x["status"] != "not compared"]),
        failures=len([x for x in rows if x["status"] == "FAIL"]),
        not_compared=len([x for x in rows if x["status"] == "not compared"]),
        validated_analyses=validated,
        not_validated_analyses=sorted({x["analysis_id"] for x in rows} - set(validated)),
        rows=rows,
    )
    (args.out / "pilot_stats_check.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md = [
        "# Pilot statistics vs R metafor (Session 10)",
        "",
        f"* **verdict: {outcome}** (exit status {EXIT[outcome]}; only `validated` means the "
        "reported statistics were reproduced)",
        f"* stats_output.json SHA-256: `{digest}`; bound to its manifest: {not binding}"
        + (f" ({'; '.join(binding)})" if binding else ""),
        f"* engine: {stats.get('engine_version')}; reference: {ref['software']}",
        f"* comparisons: {result['comparisons']}, failures: {result['failures']}, "
        f"not compared: {result['not_compared']}",
        f"* validated analyses: {', '.join(validated) or 'none'}; not validated: "
        f"{', '.join(result['not_validated_analyses']) or 'none'}",
        "",
        "| Analysis | Quantity | Engine | metafor | Tolerance | Status |",
        "|---|---|---|---|---|---|",
    ]
    md += [
        f"| {x['analysis_id']} | {x['quantity']} | {x['engine']} | {x['reference']} "
        f"| {x['tolerance']} | {x['status']} |"
        for x in rows
    ]
    (args.out / "pilot_stats_check.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:8]))
    return EXIT[outcome]


if __name__ == "__main__":
    raise SystemExit(main())
