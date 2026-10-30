"""Inverse-variance meta-analysis on a transformed scale (pure numerics, no I/O).

Every function takes study estimates ``y`` and their sampling variances ``v`` on the
analysis scale (log RR, logit proportion, MD, ...) and returns plain dataclasses. Nothing
here knows about studies, outcomes or the pipeline; input integrity is enforced upstream
(:mod:`src.stats_engine.inputs`).

Methods and references
----------------------
* Common-effect (fixed-effect) inverse-variance pooling and Cochran's Q.
* I² = (Q − df)/Q and the typical within-study variance s² (Higgins & Thompson,
  *Stat Med* 2002;21:1539-1558, eq. 9); model-based I² = τ²/(τ² + s²).
* τ²: DerSimonian & Laird (*Control Clin Trials* 1986;7:177-188) and restricted maximum
  likelihood (REML; Viechtbauer, *J Educ Behav Stat* 2005;30:261-293). REML is solved
  for the global maximum of the restricted log-likelihood on [0, ∞) by grid search plus
  root bracketing of the score, not by a fixed-iteration Fisher scoring.
* Random-effects CIs: Wald (normal), Hartung-Knapp-Sidik-Jonkman (Hartung & Knapp,
  *Stat Med* 2001;20:3875-3889; Sidik & Jonkman, *Stat Med* 2002;21:3153-3159) with
  t(k−1), and its ad hoc variant that never yields a narrower interval than the Wald
  interval (Knapp & Hartung, *Stat Med* 2003;22:2693-2710; Röver, Knapp & Friede,
  *BMC Med Res Methodol* 2015;15:99).
* Q-profile confidence interval for τ² (Viechtbauer, *Stat Med* 2007;26:37-52).
* Prediction interval μ̂ ± t(k−2)·√(τ̂² + SE²) (Higgins, Thompson & Spiegelhalter,
  *J R Stat Soc A* 2009;172:137-159), only for k ≥ 3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy import optimize, stats

FloatArray = npt.NDArray[np.float64]
Tau2Method = Literal["REML", "DL"]
CIMethod = Literal["wald", "hksj", "hksj_adhoc"]


class EstimationError(ValueError):
    """A model could not be estimated from the data (reported, never silently patched)."""


@dataclass(frozen=True, slots=True)
class Pooled:
    estimate: float
    se: float
    ci_lower: float
    ci_upper: float
    statistic: float
    p_value: float
    distribution: Literal["normal", "t"]
    df: int | None


@dataclass(frozen=True, slots=True)
class Heterogeneity:
    q: float
    df: int
    p_value: float
    i2: float  # percent, Q-based (Higgins & Thompson 2002)
    h2: float
    s2_typical: float


@dataclass(frozen=True, slots=True)
class RandomEffects:
    pooled: Pooled
    tau2: float
    tau2_method: Tau2Method
    i2_model: float  # percent, τ²/(τ² + s²)
    h2_model: float
    tau2_ci: tuple[float, float] | None
    i2_ci: tuple[float, float] | None
    prediction_interval: tuple[float, float] | None
    prediction_df: int | None
    weights: FloatArray  # normalised random-effects weights (sum to 1)


def as_arrays(y: npt.ArrayLike, v: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    ya = np.asarray(y, dtype=np.float64)
    va = np.asarray(v, dtype=np.float64)
    if ya.ndim != 1 or ya.shape != va.shape or ya.size == 0:
        raise EstimationError("y and v must be non-empty 1-D arrays of equal length")
    if not (np.all(np.isfinite(ya)) and np.all(np.isfinite(va))):
        raise EstimationError("estimates and variances must be finite")
    if np.any(va <= 0):
        raise EstimationError("sampling variances must be strictly positive")
    return ya, va


def _z(level: float) -> float:
    return float(stats.norm.ppf(1 - (1 - level) / 2))


def _two_sided_p(stat: float, df: int | None) -> float:
    if df is None:
        return float(2 * stats.norm.sf(abs(stat)))
    return float(2 * stats.t.sf(abs(stat), df))


def fixed_effect(y: npt.ArrayLike, v: npt.ArrayLike, *, level: float = 0.95) -> Pooled:
    """Inverse-variance common-effect estimate with a Wald (normal) CI."""
    ya, va = as_arrays(y, v)
    w = 1.0 / va
    mu = float(np.sum(w * ya) / np.sum(w))
    se = math.sqrt(1.0 / float(np.sum(w)))
    z = _z(level)
    stat = mu / se
    return Pooled(mu, se, mu - z * se, mu + z * se, stat, _two_sided_p(stat, None), "normal", None)


def typical_within_variance(v: FloatArray) -> float:
    """s² = (k−1)Σw / ((Σw)² − Σw²) (Higgins & Thompson 2002, eq. 9)."""
    w = 1.0 / v
    k = v.size
    denom = float(np.sum(w) ** 2 - np.sum(w**2))
    if k < 2 or denom <= 0:
        return float("nan")
    return float((k - 1) * np.sum(w) / denom)


def heterogeneity(y: npt.ArrayLike, v: npt.ArrayLike) -> Heterogeneity:
    """Cochran's Q (inverse-variance weights), I², H² and s². Requires k ≥ 2."""
    ya, va = as_arrays(y, v)
    k = ya.size
    if k < 2:
        raise EstimationError("heterogeneity statistics are undefined for a single study")
    w = 1.0 / va
    mu = float(np.sum(w * ya) / np.sum(w))
    q = float(np.sum(w * (ya - mu) ** 2))
    df = k - 1
    i2 = 0.0 if q <= 0 else max(0.0, (q - df) / q) * 100.0
    return Heterogeneity(
        q=q,
        df=df,
        p_value=float(stats.chi2.sf(q, df)),
        i2=i2,
        h2=q / df,
        s2_typical=typical_within_variance(va),
    )


def tau2_dl(y: FloatArray, v: FloatArray) -> float:
    w = 1.0 / v
    mu = float(np.sum(w * y) / np.sum(w))
    q = float(np.sum(w * (y - mu) ** 2))
    c = float(np.sum(w) - np.sum(w**2) / np.sum(w))
    return max(0.0, (q - (y.size - 1)) / c)


def _restricted_loglik(t2: float, y: FloatArray, v: FloatArray) -> float:
    w = 1.0 / (v + t2)
    sw = float(np.sum(w))
    mu = float(np.sum(w * y)) / sw
    return float(-0.5 * (np.sum(np.log(v + t2)) + math.log(sw) + np.sum(w * (y - mu) ** 2)))


def _reml_score(t2: float, y: FloatArray, v: FloatArray) -> float:
    """d l_R / d τ² (up to the factor ½)."""
    w = 1.0 / (v + t2)
    sw = float(np.sum(w))
    mu = float(np.sum(w * y)) / sw
    return float(np.sum(w**2 * (y - mu) ** 2) + np.sum(w**2) / sw - sw)


def tau2_reml(y: FloatArray, v: FloatArray) -> float:
    """Global maximiser of the restricted log-likelihood over τ² ≥ 0.

    A log-spaced grid locates every local maximum (sign change + → − of the score); each
    is refined with Brent's root finder to near machine precision and the candidate with
    the largest restricted log-likelihood (including the boundary τ² = 0) is returned.
    """
    if y.size < 2:
        raise EstimationError("τ² cannot be estimated from a single study")
    spread = float(np.max(y) - np.min(y))
    upper = 10.0 * (spread**2 + float(np.max(v))) + 1e-8
    grid = np.concatenate(([0.0], np.geomspace(upper * 1e-12, upper, 400)))
    scores = np.array([_reml_score(float(t), y, v) for t in grid])
    candidates = [0.0]
    for i in range(grid.size - 1):
        if scores[i] > 0 >= scores[i + 1]:
            root = optimize.brentq(
                _reml_score,
                float(grid[i]),
                float(grid[i + 1]),
                args=(y, v),
                xtol=1e-15,
                rtol=4 * np.finfo(float).eps,
                maxiter=500,
            )
            candidates.append(float(root))
    if scores[-1] > 0:  # pragma: no cover - the bound is far beyond any finite maximiser
        raise EstimationError("REML τ² search did not bracket a maximum")
    return max(candidates, key=lambda t: _restricted_loglik(t, y, v))


def generalised_q(t2: float, y: FloatArray, v: FloatArray) -> float:
    w = 1.0 / (v + t2)
    mu = float(np.sum(w * y) / np.sum(w))
    return float(np.sum(w * (y - mu) ** 2))


def tau2_qprofile_ci(y: FloatArray, v: FloatArray, *, level: float = 0.95) -> tuple[float, float]:
    """Q-profile CI: τ² values where the generalised Q equals χ²(k−1) quantiles."""
    k = y.size
    if k < 2:
        raise EstimationError("a τ² confidence interval needs at least two studies")
    df = k - 1
    alpha = 1 - level
    crit_lo = float(stats.chi2.ppf(1 - alpha / 2, df))  # defines the lower bound
    crit_hi = float(stats.chi2.ppf(alpha / 2, df))  # defines the upper bound
    q0 = generalised_q(0.0, y, v)

    def solve(crit: float) -> float:
        if q0 <= crit:
            return 0.0
        hi = max(1.0, float(np.var(y)) * 10 + float(np.max(v)))
        while generalised_q(hi, y, v) > crit:
            hi *= 10.0
            if hi > 1e12:  # pragma: no cover - Q(τ²) → 0 as τ² → ∞, so this is unreachable
                raise EstimationError("Q-profile bound could not be bracketed")
        return float(
            optimize.brentq(
                lambda t: generalised_q(t, y, v) - crit,
                0.0,
                hi,
                xtol=1e-15,
                rtol=4 * np.finfo(float).eps,
                maxiter=500,
            )
        )

    return solve(crit_lo), solve(crit_hi)


def random_effects(
    y: npt.ArrayLike,
    v: npt.ArrayLike,
    *,
    tau2_method: Tau2Method = "REML",
    ci_method: CIMethod = "hksj",
    level: float = 0.95,
) -> RandomEffects:
    """Random-effects (normal-normal) model. Requires k ≥ 2."""
    ya, va = as_arrays(y, v)
    k = ya.size
    if k < 2:
        raise EstimationError("a random-effects model needs at least two studies")
    t2 = tau2_reml(ya, va) if tau2_method == "REML" else tau2_dl(ya, va)
    w = 1.0 / (va + t2)
    sw = float(np.sum(w))
    mu = float(np.sum(w * ya)) / sw
    se_wald = math.sqrt(1.0 / sw)
    df: int | None
    if ci_method == "wald":
        se, df, crit = se_wald, None, _z(level)
    else:
        q_hk = float(np.sum(w * (ya - mu) ** 2)) / (k - 1)
        factor = max(1.0, q_hk) if ci_method == "hksj_adhoc" else q_hk
        se = math.sqrt(factor / sw)
        df = k - 1
        crit = float(stats.t.ppf(1 - (1 - level) / 2, df))
    if se == 0.0:
        raise EstimationError(
            "the Hartung-Knapp standard error is zero (all estimates identical); "
            "use the Wald or ad hoc Hartung-Knapp interval"
        )
    stat = mu / se
    pooled = Pooled(
        mu,
        se,
        mu - crit * se,
        mu + crit * se,
        stat,
        _two_sided_p(stat, df),
        "normal" if df is None else "t",
        df,
    )
    s2 = typical_within_variance(va)
    i2_model = 100.0 * t2 / (t2 + s2)
    tau2_ci = tau2_qprofile_ci(ya, va, level=level)
    i2_ci = (100.0 * tau2_ci[0] / (tau2_ci[0] + s2), 100.0 * tau2_ci[1] / (tau2_ci[1] + s2))
    pi: tuple[float, float] | None = None
    pi_df: int | None = None
    if k >= 3:
        pi_df = k - 2
        half = float(stats.t.ppf(1 - (1 - level) / 2, pi_df)) * math.sqrt(t2 + se**2)
        pi = (mu - half, mu + half)
    return RandomEffects(
        pooled=pooled,
        tau2=t2,
        tau2_method=tau2_method,
        i2_model=i2_model,
        h2_model=(t2 + s2) / s2,
        tau2_ci=tau2_ci,
        i2_ci=i2_ci,
        prediction_interval=pi,
        prediction_df=pi_df,
        weights=w / sw,
    )


def study_wald_ci(y: float, v: float, *, level: float = 0.95) -> tuple[float, float]:
    half = _z(level) * math.sqrt(v)
    return y - half, y + half
