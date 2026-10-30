"""Study-level effect sizes and their sampling variances (deterministic, documented).

Conventions follow metafor's ``escalc`` (Viechtbauer, *J Stat Softw* 2010;36(3):1-48) so
that every number can be reproduced independently in R:

* **Log risk ratio / log odds ratio** (2×2 tables, Cochrane Handbook v6, §6.4 and §10.4).
  Studies with no events in either arm, or only events in both arms, carry no
  information about a ratio measure and are *excluded* (``escalc(drop00=TRUE)``). A study
  with a zero cell in one arm receives 0.5 added to all four cells (``add=1/2,
  to="only0"``) for inverse-variance analyses only; Mantel-Haenszel uses the raw counts.
* **Mean difference**: m₁ − m₂, v = s₁²/n₁ + s₂²/n₂.
* **Standardised mean difference** (Hedges' g): d·J with the exact small-sample factor
  J(m) = Γ(m/2) / (√(m/2) Γ((m−1)/2)), m = n₁ + n₂ − 2 (Hedges, *J Educ Stat*
  1981;6:107-128), and the large-sample variance 1/n₁ + 1/n₂ + g²/(2(n₁+n₂)).
* **Logit proportion**: log(x/(n−x)), v = 1/x + 1/(n−x); for x = 0 or x = n, 0.5 is added
  to x and n − x (sensitivity analyses only; the GLMM uses the exact binomial likelihood).
* **Clopper-Pearson** exact CI for a single proportion (*Biometrika* 1934;26:404-413).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from scipy import special, stats

from src.stats_engine.numerics import EstimationError

CONTINUITY_CORRECTION = 0.5
ProbabilityScale = Literal["log_log", "log", "logit", "plain"]


@dataclass(frozen=True, slots=True)
class StudyEstimate:
    yi: float
    vi: float
    continuity_corrected: bool = False


def two_by_two_informative(a: int, n1: int, c: int, n2: int) -> bool:
    """False for double-zero or double-all-event studies (no information on RR/OR)."""
    return not ((a == 0 and c == 0) or (a == n1 and c == n2))


def _check_counts(events: int, n: int) -> None:
    if n <= 0 or events < 0 or events > n:
        raise EstimationError(f"invalid counts: {events} events out of {n}")


def log_ratio(measure: Literal["RR", "OR"], a: int, n1: int, c: int, n2: int) -> StudyEstimate:
    """Log RR or log OR for intervention (a/n1) versus control (c/n2)."""
    _check_counts(a, n1)
    _check_counts(c, n2)
    if not two_by_two_informative(a, n1, c, n2):
        raise EstimationError("double-zero / double-all-event study is not informative")
    b, d = n1 - a, n2 - c
    corrected = min(a, b, c, d) == 0
    if corrected:
        add = CONTINUITY_CORRECTION
        a_, b_, c_, d_ = a + add, b + add, c + add, d + add
    else:
        a_, b_, c_, d_ = float(a), float(b), float(c), float(d)
    if measure == "OR":
        yi = math.log(a_ * d_ / (b_ * c_))
        vi = 1 / a_ + 1 / b_ + 1 / c_ + 1 / d_
    else:
        m1, m2 = a_ + b_, c_ + d_
        yi = math.log((a_ / m1) / (c_ / m2))
        vi = 1 / a_ - 1 / m1 + 1 / c_ - 1 / m2
    return StudyEstimate(yi, vi, corrected)


def mean_difference(
    m1: float, sd1: float, n1: int, m2: float, sd2: float, n2: int
) -> StudyEstimate:
    if n1 < 1 or n2 < 1:
        raise EstimationError("sample sizes must be positive")
    vi = sd1**2 / n1 + sd2**2 / n2
    if vi <= 0:
        raise EstimationError("both standard deviations are zero: variance undefined")
    return StudyEstimate(m1 - m2, vi)


def hedges_j(m: int) -> float:
    """Exact bias-correction factor (Hedges 1981)."""
    if m < 2:
        raise EstimationError("Hedges' correction needs n1 + n2 − 2 ≥ 2")
    return math.exp(special.gammaln(m / 2) - 0.5 * math.log(m / 2) - special.gammaln((m - 1) / 2))


def standardised_mean_difference(
    m1: float, sd1: float, n1: int, m2: float, sd2: float, n2: int
) -> StudyEstimate:
    if n1 < 2 or n2 < 2:
        raise EstimationError("the SMD needs at least two participants per arm")
    m = n1 + n2 - 2
    sp2 = ((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / m
    if sp2 <= 0:
        raise EstimationError("pooled standard deviation is zero: SMD undefined")
    g = hedges_j(m) * (m1 - m2) / math.sqrt(sp2)
    vi = 1 / n1 + 1 / n2 + g**2 / (2 * (n1 + n2))
    return StudyEstimate(g, vi)


def logit_proportion(x: int, n: int) -> StudyEstimate:
    _check_counts(x, n)
    corrected = x in (0, n)
    add = CONTINUITY_CORRECTION if corrected else 0.0
    xe, ne = x + add, n - x + add
    return StudyEstimate(math.log(xe / ne), 1 / xe + 1 / ne, corrected)


def clopper_pearson(x: int, n: int, *, level: float = 0.95) -> tuple[float, float]:
    _check_counts(x, n)
    alpha = 1 - level
    lower = 0.0 if x == 0 else float(stats.beta.ppf(alpha / 2, x, n - x + 1))
    upper = 1.0 if x == n else float(stats.beta.ppf(1 - alpha / 2, x + 1, n - x))
    return lower, upper


def expit(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


# --------------------------------------------------------------------------- probabilities at t


def to_scale(p: float, scale: ProbabilityScale) -> float:
    """Transform a probability in (0, 1) to the declared CI scale."""
    if not 0.0 < p < 1.0:
        raise EstimationError(f"probability {p} is at or outside the boundary (0, 1)")
    if scale == "log_log":
        return math.log(-math.log(p))
    if scale == "log":
        return math.log(p)
    if scale == "logit":
        return math.log(p / (1 - p))
    return p


def from_scale(t: float, scale: ProbabilityScale) -> float:
    if scale == "log_log":
        return math.exp(-math.exp(t))
    if scale == "log":
        return math.exp(t)
    if scale == "logit":
        return expit(t)
    return t
