"""Edge cases and invariants of the numerical core (independent of the R references)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.stats_engine import effect_sizes as es
from src.stats_engine.glmm import _Model, glmm_logit
from src.stats_engine.mantel_haenszel import mantel_haenszel
from src.stats_engine.numerics import (
    EstimationError,
    _restricted_loglik,
    fixed_effect,
    generalised_q,
    heterogeneity,
    random_effects,
    tau2_dl,
    tau2_qprofile_ci,
    tau2_reml,
)

PRITZ_X = [16, 10, 4, 43, 10, 25, 13, 12, 22, 4, 5, 18, 58, 6]
PRITZ_N = [17, 12, 8, 58, 10, 42, 14, 12, 41, 5, 6, 23, 68, 10]


def test_single_study_has_no_heterogeneity_or_random_effects() -> None:
    with pytest.raises(EstimationError, match="single study"):
        heterogeneity([0.1], [0.01])
    with pytest.raises(EstimationError):
        random_effects([0.1], [0.01])
    with pytest.raises(EstimationError):
        glmm_logit([3], [10])
    fe = fixed_effect([0.1], [0.04])
    assert fe.estimate == 0.1 and fe.se == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("y", "v"), [([], []), ([1.0], [0.0]), ([1.0, 2.0], [0.1]), ([math.nan, 1.0], [0.1, 0.1])]
)
def test_invalid_inputs_rejected(y: list[float], v: list[float]) -> None:
    with pytest.raises(EstimationError):
        fixed_effect(y, v)


def test_homogeneous_data_give_tau2_zero_and_lower_qp_bound_zero() -> None:
    y = np.array([0.10, 0.12, 0.11, 0.105])
    v = np.array([0.01, 0.02, 0.015, 0.012])
    assert tau2_reml(y, v) == 0.0 and tau2_dl(y, v) == 0.0
    lo, hi = tau2_qprofile_ci(y, v)
    # Q(0) is below even the 2.5 % χ² quantile, so both Q-profile bounds are 0 (as in metafor)
    assert lo == hi == 0.0
    assert tau2_qprofile_ci(np.array([0.1, 0.5, 0.2]), np.array([0.01, 0.01, 0.01]))[1] > 0
    re = random_effects(y, v, ci_method="wald")
    fe = fixed_effect(y, v)
    assert re.pooled.estimate == pytest.approx(fe.estimate) and re.i2_model == 0.0


def test_reml_is_global_maximum() -> None:
    y = np.array([-0.9, 0.4, 0.1, -0.3, 1.2])
    v = np.array([0.04, 0.05, 0.2, 0.01, 0.3])
    t2 = tau2_reml(y, v)
    grid = np.linspace(0, 5, 5001)
    assert _restricted_loglik(t2, y, v) >= max(_restricted_loglik(float(t), y, v) for t in grid)


def test_qprofile_bounds_solve_the_q_equation() -> None:
    from scipy import stats

    y = np.array([-0.9, 0.4, 0.1, -0.3, 1.2])
    v = np.array([0.04, 0.05, 0.2, 0.01, 0.3])
    lo, hi = tau2_qprofile_ci(y, v)
    assert generalised_q(lo, y, v) == pytest.approx(stats.chi2.ppf(0.975, 4), rel=1e-9)
    assert generalised_q(hi, y, v) == pytest.approx(stats.chi2.ppf(0.025, 4), rel=1e-9)


def test_prediction_interval_needs_three_studies() -> None:
    assert random_effects([0.1, 0.5], [0.01, 0.02]).prediction_interval is None
    re = random_effects([0.1, 0.5, 0.3], [0.01, 0.02, 0.015])
    assert re.prediction_interval is not None and re.prediction_df == 1


def test_hksj_zero_se_refused_but_adhoc_works() -> None:
    with pytest.raises(EstimationError, match="Hartung-Knapp standard error is zero"):
        random_effects([0.2, 0.2, 0.2], [0.01, 0.02, 0.03], ci_method="hksj")
    re = random_effects([0.2, 0.2, 0.2], [0.01, 0.02, 0.03], ci_method="hksj_adhoc")
    assert re.pooled.se > 0


def test_hksj_adhoc_never_narrower_than_wald() -> None:
    y, v = [0.10, 0.12, 0.11, 0.105], [0.01, 0.02, 0.015, 0.012]
    adhoc = random_effects(y, v, ci_method="hksj_adhoc").pooled
    wald = random_effects(y, v, ci_method="wald").pooled
    assert adhoc.se >= wald.se


# --------------------------------------------------------------------------- effect sizes


def test_double_zero_and_double_all_event_tables_are_not_informative() -> None:
    assert not es.two_by_two_informative(0, 10, 0, 12)
    assert not es.two_by_two_informative(10, 10, 12, 12)
    assert es.two_by_two_informative(0, 10, 1, 12)
    with pytest.raises(EstimationError):
        es.log_ratio("OR", 0, 10, 0, 12)


def test_continuity_correction_only_for_zero_cells() -> None:
    plain = es.log_ratio("RR", 2, 10, 4, 12)
    assert not plain.continuity_corrected
    assert plain.yi == pytest.approx(math.log((2 / 10) / (4 / 12)))
    zero = es.log_ratio("OR", 0, 10, 3, 12)
    assert zero.continuity_corrected
    assert zero.yi == pytest.approx(math.log(0.5 * 9.5 / (10.5 * 3.5)))


def test_hedges_correction_factor() -> None:
    # J(m) = Γ(m/2)/(√(m/2)Γ((m−1)/2)) (Hedges 1981); ≈ 1 − 3/(4m − 1) for large m
    for m in (2, 5, 10, 40, 200):
        exact = math.gamma(m / 2) / (math.sqrt(m / 2) * math.gamma((m - 1) / 2))
        assert es.hedges_j(m) == pytest.approx(exact, rel=1e-12)
    assert es.hedges_j(200) == pytest.approx(1 - 3 / (4 * 200 - 1), abs=1e-6)
    assert es.hedges_j(10) < es.hedges_j(100) < 1
    with pytest.raises(EstimationError):
        es.standardised_mean_difference(1, 1, 1, 0, 1, 5)
    with pytest.raises(EstimationError):
        es.standardised_mean_difference(1, 0, 5, 0, 0, 5)
    with pytest.raises(EstimationError):
        es.mean_difference(1, 0, 5, 0, 0, 5)


def test_clopper_pearson_boundaries() -> None:
    assert es.clopper_pearson(0, 20)[0] == 0.0
    assert es.clopper_pearson(20, 20)[1] == 1.0
    _, hi = es.clopper_pearson(0, 20)
    assert hi == pytest.approx(1 - 0.025 ** (1 / 20))  # exact for x = 0
    with pytest.raises(EstimationError):
        es.clopper_pearson(5, 3)


@pytest.mark.parametrize("scale", ["log_log", "log", "logit", "plain"])
def test_probability_scales_round_trip(scale: es.ProbabilityScale) -> None:
    for p in (0.01, 0.3, 0.92, 0.999):
        assert es.from_scale(es.to_scale(p, scale), scale) == pytest.approx(p, rel=1e-12)
    # scale-specific domains (Session 9, M1 and review P2): identity [0, 1], log (0, 1],
    # logit and log(-log) (0, 1)
    valid = {"plain": (0.0, 1.0), "log": (1.0,), "logit": (), "log_log": ()}[scale]
    invalid = {"plain": (-0.01, 1.01), "log": (0.0, 1.01), "logit": (0.0, 1.0),
               "log_log": (0.0, 1.0)}[scale]  # fmt: skip
    for p in valid:
        assert es.from_scale(es.to_scale(p, scale), scale) == pytest.approx(p, abs=1e-15)
    for p in invalid:
        with pytest.raises(EstimationError, match="outside the domain"):
            es.to_scale(p, scale)


def test_logit_correction_for_boundary_proportions() -> None:
    assert es.logit_proportion(0, 10).continuity_corrected
    assert es.logit_proportion(10, 10).yi == pytest.approx(math.log(10.5 / 0.5))
    assert not es.logit_proportion(3, 10).continuity_corrected
    assert es.expit(-800.0) == 0.0 and es.expit(800.0) == 1.0


# --------------------------------------------------------------------------- MH and GLMM


def test_mantel_haenszel_undefined_cases() -> None:
    with pytest.raises(EstimationError):
        mantel_haenszel("OR", [])
    with pytest.raises(EstimationError, match="undefined"):
        mantel_haenszel("OR", [(0, 10, 2, 10), (0, 12, 3, 11)])  # no events in intervention
    with pytest.raises(EstimationError, match="undefined"):
        mantel_haenszel("RR", [(0, 10, 2, 10)])


def test_mantel_haenszel_single_table_equals_crude_estimate() -> None:
    mh = mantel_haenszel("OR", [(3, 20, 6, 22)])
    assert mh.estimate == pytest.approx(math.log((3 * 16) / (17 * 6)))
    mh_rr = mantel_haenszel("RR", [(3, 20, 6, 22)])
    assert mh_rr.estimate == pytest.approx(math.log((3 / 20) / (6 / 22)))


def test_glmm_boundary_cases() -> None:
    with pytest.raises(EstimationError, match="boundary"):
        glmm_logit([0, 0, 0], [10, 20, 30])
    with pytest.raises(EstimationError, match="boundary"):
        glmm_logit([10, 20], [10, 20])
    with pytest.raises(EstimationError, match="invalid"):
        glmm_logit([11, 2], [10, 20])
    # homogeneous data: τ² at the boundary, μ equals the pooled logit
    g = glmm_logit([10, 20, 30], [100, 200, 300])
    assert g.at_boundary and g.tau2 == 0.0
    assert g.mu == pytest.approx(math.log(60 / 540), abs=1e-6)
    assert g.se == pytest.approx(math.sqrt(1 / 60 + 1 / 540), rel=1e-4)


def test_glmm_conditional_loglik_is_unimodal_in_mu() -> None:
    """Regression: an undamped Newton step made the likelihood non-monotone at large |μ|."""
    model = _Model(np.array(PRITZ_X, dtype=float), np.array(PRITZ_N, dtype=float), 7)
    mus = np.linspace(-15, 20, 141)
    ll = np.array([model.loglik(float(m), 0.7) for m in mus])
    peak = int(np.argmax(ll))
    assert np.all(np.diff(ll[: peak + 1]) > 0) and np.all(np.diff(ll[peak:]) < 0)


def test_glmm_is_a_local_maximum_and_quadrature_converged() -> None:
    g = glmm_logit(PRITZ_X, PRITZ_N)
    model = _Model(np.array(PRITZ_X, dtype=float), np.array(PRITZ_N, dtype=float), 7)
    tau = math.sqrt(g.tau2)
    for dm, dt in ((1e-3, 0), (-1e-3, 0), (0, 1e-3), (0, -1e-3)):
        assert model.loglik(g.mu + dm, tau + dt) <= g.loglik + 1e-9
    g21 = glmm_logit(PRITZ_X, PRITZ_N, n_agq=21)
    assert g21.mu == pytest.approx(g.mu, abs=1e-4) and g21.tau2 == pytest.approx(g.tau2, abs=1e-3)
