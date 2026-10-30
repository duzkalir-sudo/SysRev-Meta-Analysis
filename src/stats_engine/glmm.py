"""Random-effects logistic regression (binomial-normal GLMM) for single proportions.

Model: xᵢ ~ Binomial(nᵢ, pᵢ), logit(pᵢ) = μ + uᵢ, uᵢ ~ N(0, τ²). The exact binomial
likelihood is used, so studies with 0 or nᵢ events need no continuity correction
(Stijnen, Hamza & Özdemir, *Stat Med* 2010;29:3046-3067; Hamza, van Houwelingen &
Stijnen, *J Clin Epidemiol* 2008;61:41-51). Use of the Freeman-Tukey double-arcsine
transformation is deliberately avoided (Schwarzer et al., *Res Synth Methods*
2019;10:476-483).

Estimation is by maximum likelihood. The marginal likelihood of each study is integrated
by adaptive Gauss-Hermite quadrature centred on the conditional mode (default 7 nodes,
as in ``metafor::rma.glmm`` → ``lme4::glmer(nAGQ = 7)``). μ is profiled by Brent's method
for each τ; τ is located on a grid and refined by bounded Brent; the τ = 0 boundary is
evaluated explicitly. SE(μ̂) is taken from the inverse observed information matrix in
(τ, μ) (finite differences), mirroring ``lme4``'s ``vcov(use.hessian = TRUE)``. The
confidence interval is a Wald (normal) interval, which can under-cover with few studies.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import optimize, special, stats

from src.stats_engine.numerics import EstimationError, FloatArray

_TAU_MAX = 15.0


@dataclass(frozen=True, slots=True)
class GLMMResult:
    mu: float
    se: float
    ci_lower: float
    ci_upper: float
    z: float
    p_value: float
    tau2: float
    loglik: float
    n_agq: int
    at_boundary: bool


class _Model:
    def __init__(self, x: FloatArray, n: FloatArray, n_agq: int) -> None:
        self.x, self.n = x, n
        self.nodes, weights = np.polynomial.hermite.hermgauss(n_agq)
        self.log_w = np.log(weights) + self.nodes**2
        self.log_binom = (
            special.gammaln(n + 1) - special.gammaln(x + 1) - special.gammaln(n - x + 1)
        )

    def _g(self, z: FloatArray, mu: float, tau: float) -> FloatArray:
        """x·log p + (n−x)·log(1−p) − z²/2 at quadrature points ``z`` (studies × nodes)."""
        eta = mu + tau * z
        x, nx = self.x[:, None], (self.n - self.x)[:, None]
        ll = -x * np.logaddexp(0, -eta) - nx * np.logaddexp(0, eta)
        return np.asarray(ll - 0.5 * z**2, dtype=np.float64)

    def _g1(self, z: FloatArray, mu: float, tau: float) -> FloatArray:
        eta = mu + tau * z
        ll = -self.x * np.logaddexp(0, -eta) - (self.n - self.x) * np.logaddexp(0, eta)
        return np.asarray(ll - 0.5 * z**2, dtype=np.float64)

    def loglik(self, mu: float, tau: float) -> float:
        # Conditional mode of each study's integrand by Newton's method with step halving
        # (the integrand is strictly log-concave, so a safeguarded ascent always converges).
        z = np.zeros_like(self.x)
        gz = self._g1(z, mu, tau)
        for _ in range(200):
            p = special.expit(mu + tau * z)
            grad = tau * (self.x - self.n * p) - z
            hess = -(tau**2) * self.n * p * (1 - p) - 1.0
            step = -grad / hess
            t = np.ones_like(z)
            for _ in range(60):
                cand = z + t * step
                gc = self._g1(cand, mu, tau)
                worse = gc < gz - 1e-12 * np.abs(gz)
                if not np.any(worse):
                    break
                t = np.where(worse, t / 2, t)
            z, gz = cand, gc
            if float(np.max(np.abs(t * step))) < 1e-12:
                break
        p = special.expit(mu + tau * z)
        sigma = 1.0 / np.sqrt(tau**2 * self.n * p * (1 - p) + 1.0)
        pts = z[:, None] + math.sqrt(2.0) * sigma[:, None] * self.nodes[None, :]
        log_int = special.logsumexp(self._g(pts, mu, tau) + self.log_w[None, :], axis=1)
        log_int = log_int + np.log(math.sqrt(2.0) * sigma) - 0.5 * math.log(2 * math.pi)
        return float(np.sum(log_int + self.log_binom))

    def profile_mu(self, tau: float, start: float) -> tuple[float, float]:
        res = optimize.minimize_scalar(
            lambda m: -self.loglik(m, tau),
            bounds=(start - 20.0, start + 20.0),
            method="bounded",
            options={"xatol": 1e-11, "maxiter": 1000},
        )
        return float(res.x), -float(res.fun)


def glmm_logit(
    events: Sequence[int], totals: Sequence[int], *, n_agq: int = 7, level: float = 0.95
) -> GLMMResult:
    x = np.asarray(events, dtype=np.float64)
    n = np.asarray(totals, dtype=np.float64)
    if x.size < 2:
        raise EstimationError("the GLMM needs at least two studies")
    if np.any(n <= 0) or np.any(x < 0) or np.any(x > n):
        raise EstimationError("invalid counts")
    if np.all(x == 0) or np.all(x == n):
        raise EstimationError(
            "every study has 0 events (or every study has only events): the maximum "
            "likelihood estimate lies on the boundary and does not exist"
        )
    model = _Model(x, n, n_agq)
    start = math.log((x.sum() + 0.5) / (n.sum() - x.sum() + 0.5))

    taus = np.concatenate(([0.0], np.geomspace(1e-2, _TAU_MAX, 24)))
    prof = [model.profile_mu(float(t), start) for t in taus]
    best = int(np.argmax([ll for _, ll in prof]))
    if best == taus.size - 1:
        raise EstimationError("τ estimate diverges (no finite maximum likelihood estimate)")
    lo = float(taus[max(best - 1, 0)])
    hi = float(taus[best + 1])
    res = optimize.minimize_scalar(
        lambda t: -model.profile_mu(t, start)[1],
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 1e-10, "maxiter": 1000},
    )
    tau, ll = float(res.x), -float(res.fun)
    ll0 = prof[0][1]
    # τ indistinguishable from 0 (flat profile) is reported as the boundary estimate τ² = 0
    at_boundary = ll0 >= ll - 1e-10 or tau < 1e-5
    if at_boundary:
        tau, ll = 0.0, ll0
    mu, ll = model.profile_mu(tau, start)

    se = _se_mu(model, mu, tau, at_boundary)
    zcrit = float(stats.norm.ppf(1 - (1 - level) / 2))
    zval = mu / se
    return GLMMResult(
        mu=mu,
        se=se,
        ci_lower=mu - zcrit * se,
        ci_upper=mu + zcrit * se,
        z=zval,
        p_value=float(2 * stats.norm.sf(abs(zval))),
        tau2=tau**2,
        loglik=ll,
        n_agq=n_agq,
        at_boundary=at_boundary,
    )


def _se_mu(model: _Model, mu: float, tau: float, at_boundary: bool) -> float:
    hm = 1e-4 * max(1.0, abs(mu))
    f = model.loglik
    d_mm = (f(mu + hm, tau) - 2 * f(mu, tau) + f(mu - hm, tau)) / hm**2
    if at_boundary or tau < 1e-4:
        if d_mm >= 0:
            raise EstimationError("observed information for μ is not positive")
        return math.sqrt(-1.0 / d_mm)
    ht = 1e-4 * max(1.0, tau)
    d_tt = (f(mu, tau + ht) - 2 * f(mu, tau) + f(mu, tau - ht)) / ht**2
    d_mt = (
        f(mu + hm, tau + ht) - f(mu + hm, tau - ht) - f(mu - hm, tau + ht) + f(mu - hm, tau - ht)
    ) / (4 * hm * ht)
    info = -np.array([[d_tt, d_mt], [d_mt, d_mm]])
    if np.linalg.det(info) <= 0 or info[1, 1] <= 0:
        raise EstimationError("observed information matrix is not positive definite")
    cov = np.linalg.inv(info)
    return math.sqrt(float(cov[1, 1]))
