"""Compare src.stats_engine with the committed metafor reference values.

Usage: python -m validation.compare_metafor [reference.json]

Prints, per quantity, the largest absolute and relative discrepancy. Used by the test
suite (tests/unit/test_stats_metafor_reference.py) and to write VALIDATION_REPORT.md.
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from src.stats_engine import effect_sizes as es
from src.stats_engine.glmm import glmm_logit
from src.stats_engine.mantel_haenszel import mantel_haenszel
from src.stats_engine.numerics import fixed_effect, heterogeneity, random_effects

DEFAULT_REFERENCE = Path(__file__).parent.parent / "tests/fixtures/stats/metafor_reference.json"

# (case id, quantity, python value, R value, tolerance class)
Comparison = tuple[str, str, float, float, str]
TEST_MAP = {"z": "wald", "knha": "hksj", "adhoc": "hksj_adhoc"}


def _iv(
    case_id: str, yi: list[float], vi: list[float], exp: dict[str, Any]
) -> Iterator[Comparison]:
    for i, (a, b) in enumerate(zip(yi, exp["yi"], strict=True)):
        yield case_id, f"yi[{i}]", a, b, "closed"
    for i, (a, b) in enumerate(zip(vi, exp["vi"], strict=True)):
        yield case_id, f"vi[{i}]", a, b, "closed"
    fe = fixed_effect(yi, vi)
    for f in ("estimate", "se", "ci_lower", "ci_upper", "statistic", "p_value"):
        yield case_id, f"common.{f}", getattr(fe, f), exp["common"][f], "closed"
    h = heterogeneity(yi, vi)
    yield case_id, "Q", h.q, exp["Q"], "closed"
    yield case_id, "Q.p", h.p_value, exp["Qp"], "closed"
    yield case_id, "I2(Q)", h.i2, exp["I2_q"], "closed"
    for tm in ("REML", "DL"):
        for test, ours in TEST_MAP.items():
            key = f"{tm}_{test}"
            ref = exp[key]
            tol = "iterative" if tm == "REML" else "closed"
            try:
                re = random_effects(yi, vi, tau2_method=tm, ci_method=ours)  # type: ignore[arg-type]
            except ValueError as exc:  # e.g. HKSJ SE of zero
                yield case_id, f"{key}.error:{exc}", math.nan, ref["se"], "skip"
                continue
            p = re.pooled
            pairs = {
                "estimate": p.estimate,
                "se": p.se,
                "ci_lower": p.ci_lower,
                "ci_upper": p.ci_upper,
                "statistic": p.statistic,
                "p_value": p.p_value,
                "tau2": re.tau2,
                "I2": re.i2_model,
            }
            for f, v in pairs.items():
                yield case_id, f"{key}.{f}", v, ref[f], tol
            if "pi_lower" in ref and re.prediction_interval is not None:
                yield case_id, f"{key}.pi_lower", re.prediction_interval[0], ref["pi_lower"], tol
                yield case_id, f"{key}.pi_upper", re.prediction_interval[1], ref["pi_upper"], tol
            if "tau2_ci" in ref and re.tau2_ci is not None:
                for j, nm in enumerate(("lower", "upper")):
                    yield case_id, f"{key}.tau2_ci.{nm}", re.tau2_ci[j], ref["tau2_ci"][j], "qp"
                    assert re.i2_ci is not None
                    yield case_id, f"{key}.I2_ci.{nm}", re.i2_ci[j], ref["I2_ci"][j], "qp"


def comparisons(reference: dict[str, Any]) -> Iterator[Comparison]:
    for case in reference["cases"]:
        cid, kind, data, exp = case["id"], case["kind"], case["data"], case["expected"]
        if kind == "proportion":
            ests = [es.logit_proportion(x, n) for x, n in zip(data["xi"], data["ni"], strict=True)]
            yield from _iv(cid, [e.yi for e in ests], [e.vi for e in ests], exp)
            g = glmm_logit(data["xi"], data["ni"])
            ref = exp["glmm"]
            for f, v in {
                "estimate": g.mu,
                "se": g.se,
                "ci_lower": g.ci_lower,
                "ci_upper": g.ci_upper,
                "statistic": g.z,
                "tau2": g.tau2,
            }.items():
                yield cid, f"glmm.{f}", v, ref[f], "glmm"
            yield cid, "glmm.p_value", g.p_value, ref["p_value"], "glmm_p"
            if len(data["xi"]) >= 3:
                from scipy import stats

                k = len(data["xi"])
                half = float(stats.t.ppf(0.975, k - 2)) * math.sqrt(g.tau2 + g.se**2)
                yield cid, "glmm.pi_lower", g.mu - half, ref["pi_lower"], "glmm"
                yield cid, "glmm.pi_upper", g.mu + half, ref["pi_upper"], "glmm"
            sx, sn = sum(data["xi"]), sum(data["ni"])
            mu = math.log(sx / (sn - sx))
            se = math.sqrt(1 / sx + 1 / (sn - sx))
            yield cid, "glmm_common.estimate", mu, exp["glmm_common"]["estimate"], "glmm"
            yield cid, "glmm_common.se", se, exp["glmm_common"]["se"], "glmm"
            for i, (x, n) in enumerate(zip(data["xi"], data["ni"], strict=True)):
                lo, hi = es.clopper_pearson(x, n)
                yield cid, f"clopper_pearson[{i}].lower", lo, exp["clopper_pearson"][i][0], "closed"
                yield cid, f"clopper_pearson[{i}].upper", hi, exp["clopper_pearson"][i][1], "closed"
        elif kind == "two_by_two":
            tables = list(zip(data["ai"], data["n1i"], data["ci"], data["n2i"], strict=True))
            for measure in ("RR", "OR"):
                ref = exp[measure]
                kept = [i for i, t in enumerate(tables) if es.two_by_two_informative(*t)]
                kept_ref = ref["kept"] if isinstance(ref["kept"], list) else [ref["kept"]]
                yield f"{cid}.{measure}", "kept", float(kept == kept_ref), 1.0, "closed"
                ests = [es.log_ratio(measure, *tables[i]) for i in kept]
                yield from _iv(f"{cid}.{measure}", [e.yi for e in ests], [e.vi for e in ests], ref)
                mh = mantel_haenszel(measure, [tables[i] for i in kept])
                for f in ("estimate", "se", "ci_lower", "ci_upper", "statistic", "p_value"):
                    yield f"{cid}.{measure}", f"mh.{f}", getattr(mh, f), ref["mh"][f], "closed"
        elif kind == "continuous":
            rows = list(
                zip(
                    data["m1i"],
                    data["sd1i"],
                    data["n1i"],
                    data["m2i"],
                    data["sd2i"],
                    data["n2i"],
                    strict=True,
                )
            )
            for measure, fn in (
                ("MD", es.mean_difference),
                ("SMD", es.standardised_mean_difference),
            ):
                ests = [fn(*r) for r in rows]
                yield from _iv(
                    f"{cid}.{measure}", [e.yi for e in ests], [e.vi for e in ests], exp[measure]
                )
        elif kind == "km_loglog":
            yi, vi = [], []
            for p, lo, hi, lvl in zip(
                data["est"], data["lower"], data["upper"], data["level"], strict=True
            ):
                z = float(__import__("scipy").stats.norm.ppf(1 - (1 - lvl / 100) / 2))
                yi.append(es.to_scale(p / 100, "log_log"))
                se = abs(es.to_scale(hi / 100, "log_log") - es.to_scale(lo / 100, "log_log")) / (
                    2 * z
                )
                vi.append(se**2)
            yield from _iv(cid, yi, vi, exp)
        elif kind == "generic":
            yield from _iv(cid, data["yi"], data["vi"], exp)


TOLERANCES = {  # (absolute, relative)
    "closed": (1e-10, 1e-9),
    "iterative": (1e-8, 1e-7),
    "qp": (1e-6, 1e-6),
    "glmm": (2e-4, 2e-4),
    "glmm_p": (1e-5, 5e-3),
}


def within(a: float, b: float, cls: str) -> bool:
    if cls == "skip":
        return True
    abs_tol, rel_tol = TOLERANCES[cls]
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def main(path: Path = DEFAULT_REFERENCE) -> int:
    ref = json.loads(path.read_text())
    worst: dict[str, tuple[float, float, str]] = {}
    failures = []
    n = 0
    for cid, q, a, b, cls in comparisons(ref):
        n += 1
        if cls == "skip":
            print(f"SKIPPED {cid} {q}")
            continue
        diff = abs(a - b)
        rel = diff / abs(b) if b else diff
        if diff > worst.get(cls, (0.0, 0.0, ""))[0]:
            worst[cls] = (diff, rel, f"{cid} {q}")
        if not within(a, b, cls):
            failures.append(f"{cid} {q}: python={a!r} R={b!r}")
    print(f"metafor {ref['meta']['metafor']}, lme4 {ref['meta']['lme4']}, {ref['meta']['R']}")
    print(f"{n} comparisons")
    for cls, (d, r, where) in sorted(worst.items()):
        print(f"  {cls:10s} max |diff| = {d:.3e} (rel {r:.3e}) at {where}")
    for f in failures:
        print("FAIL", f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])) if len(sys.argv) > 1 else main())
