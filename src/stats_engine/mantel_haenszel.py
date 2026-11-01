"""Mantel-Haenszel common-effect estimates for 2×2 tables (sparse-data robust).

* Odds ratio: Mantel & Haenszel, *J Natl Cancer Inst* 1959;22:719-748; variance of log OR
  from Robins, Breslow & Greenland, *Biometrics* 1986;42:311-323.
* Risk ratio: variance of log RR from Greenland & Robins, *Biometrics* 1985;41:55-68.

Raw counts are used (no continuity correction). Double-zero and double-all-event studies
are removed by the caller (``inputs``/``engine``), following the convention that they carry
no information on a relative effect. Note (Session 9 triage): ``metafor::rma.mh`` applies
``drop00`` only to the study-level ``escalc`` outcomes, not to the Mantel-Haenszel sums. A
double-zero study contributes nothing to those sums for either measure, and a double-all-event
study contributes nothing to the OR, so the estimates agree. A double-all-event study does,
however, enter metafor's MH *risk ratio*, which therefore differs from this engine's when
such a study is present. No validation case contains one; see VALIDATION_REPORT.md.

Study weights (:func:`mantel_haenszel_weights`): wᵢ = bᵢcᵢ/Nᵢ for the odds ratio and
wᵢ = cᵢn₁ᵢ/Nᵢ for the risk ratio — the weights of the Mantel-Haenszel estimators
(OR_MH = Σaᵢdᵢ/Nᵢ ÷ Σbᵢcᵢ/Nᵢ; RR_MH = Σaᵢn₂ᵢ/Nᵢ ÷ Σcᵢn₁ᵢ/Nᵢ), exactly as returned by
``metafor::weights.rma.mh`` (metafor 4.4-0; Mantel & Haenszel 1959; Greenland & Robins 1985).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

from scipy import stats

from src.stats_engine.numerics import EstimationError, Pooled


def mantel_haenszel(
    measure: Literal["RR", "OR"],
    tables: Sequence[tuple[int, int, int, int]],
    *,
    level: float = 0.95,
) -> Pooled:
    """Pooled log RR / log OR. ``tables`` holds (a, n1, c, n2): events/total per arm."""
    if not tables:
        raise EstimationError("no 2x2 tables")
    if measure == "OR":
        r = s = pr = ps_qr = qs = 0.0
        for a, n1, c, n2 in tables:
            b, d, n = n1 - a, n2 - c, n1 + n2
            ri, si = a * d / n, b * c / n
            p, q = (a + d) / n, (b + c) / n
            r += ri
            s += si
            pr += p * ri
            ps_qr += p * si + q * ri
            qs += q * si
        if r == 0 or s == 0:
            raise EstimationError("Mantel-Haenszel OR is undefined (zero numerator/denominator)")
        est = math.log(r / s)
        var = pr / (2 * r**2) + ps_qr / (2 * r * s) + qs / (2 * s**2)
    else:
        r = s = num = 0.0
        for a, n1, c, n2 in tables:
            n = n1 + n2
            r += a * n2 / n
            s += c * n1 / n
            num += (n1 * n2 * (a + c) - a * c * n) / n**2
        if r == 0 or s == 0:
            raise EstimationError("Mantel-Haenszel RR is undefined (zero numerator/denominator)")
        est = math.log(r / s)
        var = num / (r * s)
    se = math.sqrt(var)
    z = float(stats.norm.ppf(1 - (1 - level) / 2))
    stat = est / se
    return Pooled(
        est,
        se,
        est - z * se,
        est + z * se,
        stat,
        float(2 * stats.norm.sf(abs(stat))),
        "normal",
        None,
    )


def mantel_haenszel_weights(
    measure: Literal["RR", "OR"], tables: Sequence[tuple[int, int, int, int]]
) -> list[float]:
    """Mantel-Haenszel study weights in percent (raw counts; ``tables`` as above)."""
    w: list[float] = []
    for a, n1, c, n2 in tables:
        n = n1 + n2
        w.append((n1 - a) * c / n if measure == "OR" else c * n1 / n)
    total = sum(w)
    if total <= 0:
        raise EstimationError("Mantel-Haenszel weights are undefined (all zero)")
    return [100.0 * x / total for x in w]
