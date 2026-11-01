"""Stage 5 engine: run a human-approved analysis plan over approved extraction data.

Every analysis ends in an explicit status (``pooled``, ``single_study``,
``descriptive_only``, ``not_estimable``, ``unsupported``, ``no_data``) with a reason, and
lists every study it included or excluded. Unsupported requests are reported, never
approximated: no hazard ratio is derived from survival percentages, no SD from an SE,
range or IQR, no events count from a percentage, and no timepoint is converted.
"""

from __future__ import annotations

import math
import platform
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import numpy as np
import scipy
from scipy import stats

from src.core.enums import EffectMeasure
from src.core.exceptions import StatisticsInputError
from src.core.schemas.extraction import ExtractedStudyArm, ExtractedValue
from src.core.schemas.protocol import OutcomeDefinition
from src.core.schemas.stats import (
    POOLABLE_MEASURES,
    AnalysisPlan,
    AnalysisResult,
    AnalysisSpec,
    ExcludedStudy,
    HeterogeneityResult,
    PooledResult,
    StatsOutput,
    StudyResult,
    results_fingerprint,
)
from src.stats_engine import effect_sizes as es
from src.stats_engine.glmm import glmm_logit
from src.stats_engine.inputs import (
    AnalysisInputs,
    Selection,
    Unit,
    account_reports,
    all_reports_excluded,
    select_units,
)
from src.stats_engine.mantel_haenszel import mantel_haenszel, mantel_haenszel_weights
from src.stats_engine.numerics import (
    EstimationError,
    Pooled,
    RandomEffects,
    fixed_effect,
    heterogeneity,
    random_effects,
    study_wald_ci,
)

# 1.1.0 (Session 9): per-number rounding tolerance and identity-scale boundaries for KM/CIF
# (M1, M2); weights attached to the model they belong to (M3).
ENGINE_VERSION = "1.1.0"
GLMM_NODES = 7
# Labels of sensitivity models whose study weights are reported (Session 9, M3)
IV_COMMON_LABEL = "inverse-variance common effect (sensitivity)"
IV_LOGIT_RE_LABEL = "inverse-variance logit random effects (sensitivity)"

NUMERIC_PRECISION = (
    "All quantities are IEEE 754 binary64 floats serialised with the shortest repr that "
    "round-trips exactly; nothing is rounded before storage. Tables and plots round only "
    "for display (settings.display_decimals). Root finding uses Brent's method to near "
    "machine precision; the GLMM uses 7-node adaptive Gauss-Hermite quadrature and "
    "finite-difference second derivatives (agreement with lme4 ~1e-4 relative)."
)

GLOBAL_LIMITATIONS = [
    "Only measures validated against metafor in this release are pooled: crude single-group "
    "proportions, Kaplan-Meier/actuarial probabilities and competing-risks cumulative "
    "incidences at a fixed time (with reported CIs), RR, OR, MD and SMD. HR, RD, IRR, "
    "incidence rates, means, medians and median time-to-event are reported as unsupported.",
    "No value is imputed or converted: SDs are not derived from SEs, CIs, ranges or IQRs "
    "(Wan 2014 / Luo 2018 methods are not implemented); events are not derived from "
    "percentages; hazard ratios are never derived from survival percentages; and individual "
    "participant data are never reconstructed from Kaplan-Meier curves.",
    "Meta-regression, subgroup interaction tests, network meta-analysis, publication-bias "
    "tests (Egger, Begg, trim-and-fill), leave-one-out and Bayesian methods are not "
    "implemented in this release.",
    "Kaplan-Meier and cumulative-incidence pooling assumes the authors' confidence interval "
    "is symmetric on the scale declared in the analysis plan; studies whose interval is "
    "inconsistent with that scale (allowing for rounding of the printed numbers) are "
    "described but not pooled. This cannot detect a different but coincidentally "
    "consistent variance estimator.",
    "Random-effects confidence and prediction intervals can be unreliable with few "
    "studies; τ² and I² are reported with Q-profile confidence intervals to show their "
    "imprecision. Statistical heterogeneity measures do not explain clinical or "
    "methodological diversity.",
]


def software_versions() -> dict[str, str]:
    return {
        "engine": f"src.stats_engine {ENGINE_VERSION}",
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(terse=True),
    }


# --------------------------------------------------------------------------- helpers

Transform = Callable[[float], float]


@dataclass
class _Data:
    """Per-study effect sizes ready for pooling."""

    units: list[Unit] = field(default_factory=list)
    yi: list[float] = field(default_factory=list)
    vi: list[float] = field(default_factory=list)
    studies: list[dict[str, Any]] = field(default_factory=list)


def _num(v: ExtractedValue | None) -> float | None:
    return None if v is None else v.value


def _count(v: ExtractedValue | None) -> int | None:
    return None if v is None else int(v.value)


def _pooled_result(
    p: Pooled,
    *,
    model: Literal["common_effect", "random_effects"],
    method: str,
    scale: str,
    back: Transform,
    k: int,
    re: RandomEffects | None = None,
) -> PooledResult:
    lo, hi = sorted((back(p.ci_lower), back(p.ci_upper)))
    extra: dict[str, Any] = {}
    if re is not None:
        extra = {
            "tau2": re.tau2,
            "tau2_ci": re.tau2_ci,
            "i2_model": re.i2_model,
            "i2_ci": re.i2_ci,
            "prediction_interval": re.prediction_interval,
            "prediction_interval_natural": None
            if re.prediction_interval is None
            else tuple(sorted(back(x) for x in re.prediction_interval)),
            "prediction_df": re.prediction_df,
        }
    return PooledResult(
        model=model,
        method=method,
        scale=scale,
        estimate=p.estimate,
        se=p.se,
        ci_lower=p.ci_lower,
        ci_upper=p.ci_upper,
        estimate_natural=back(p.estimate),
        ci_lower_natural=lo,
        ci_upper_natural=hi,
        statistic=p.statistic,
        p_value=p.p_value,
        distribution=p.distribution,
        df=p.df,
        k=k,
        **extra,
    )


def _re_method_label(spec: AnalysisSpec) -> str:
    ci = {
        "hksj": "Hartung-Knapp-Sidik-Jonkman t(k−1) CI",
        "hksj_adhoc": "ad hoc (never narrower than Wald) Hartung-Knapp CI",
        "wald": "Wald (normal) CI",
    }[spec.random_effects_ci]
    return f"inverse variance, {spec.tau2_estimator} τ², {ci}"


def _heterogeneity(yi: list[float], vi: list[float], basis: str) -> HeterogeneityResult:
    h = heterogeneity(yi, vi)
    return HeterogeneityResult(
        basis=basis, q=h.q, df=h.df, p_value=h.p_value, i2=h.i2, h2=h.h2, s2_typical=h.s2_typical
    )


def _identity(x: float) -> float:
    return x


# --------------------------------------------------------------------------- analysis


class _Analysis:
    def __init__(self, spec: AnalysisSpec, outcome: OutcomeDefinition, inputs: AnalysisInputs):
        self.spec = spec
        self.outcome = outcome
        self.inputs = inputs
        self.level = spec.confidence_level
        self.excluded: list[ExcludedStudy] = []
        self.warnings: list[str] = []
        self.methods: list[str] = []
        self.limitations: list[str] = []
        self.sensitivity: list[PooledResult] = []
        self.selection: Selection | None = None

    # ---- framework
    def run(self) -> AnalysisResult:
        measure = self.spec.effect_measure
        if measure not in POOLABLE_MEASURES:
            reason = _UNSUPPORTED.get(measure, _UNSUPPORTED_DEFAULT)
            self.selection = all_reports_excluded(self.inputs, "unsupported_measure", reason)
            return self._result("unsupported", reason, scale="none")
        selection = select_units(self.spec, self.outcome, self.inputs)
        self.selection = selection
        self.warnings.extend(selection.notes)
        handler = {
            EffectMeasure.PROPORTION: self._proportion,
            EffectMeasure.KM_PROBABILITY: self._time_t,
            EffectMeasure.CUMULATIVE_INCIDENCE: self._time_t,
            EffectMeasure.RR: self._ratio,
            EffectMeasure.OR: self._ratio,
            EffectMeasure.MD: self._continuous,
            EffectMeasure.SMD: self._continuous,
        }[measure]
        return handler(selection.units)

    def _exclude(self, unit: Unit, code: str, reason: str) -> None:
        self.excluded.append(
            ExcludedStudy.model_validate(
                {
                    "study_id": unit.unit_id,
                    "record_ids": list(unit.record_ids),
                    "label": unit.label,
                    "arm_ids": unit.arm_ids,
                    "reason_code": code,
                    "reason": reason,
                }
            )
        )

    def _settings(self) -> dict[str, Any]:
        s = self.spec.model_dump(mode="json")
        s.update(
            continuity_correction=es.CONTINUITY_CORRECTION,
            continuity_correction_rule="added to all cells of studies with a zero cell "
            "(inverse-variance analyses only)",
            glmm_quadrature_nodes=GLMM_NODES,
            prediction_interval_min_k=3,
        )
        return s

    def _result(
        self, status: str, reason: str, *, scale: str, k_data: _Data | None = None, **kw: Any
    ) -> AnalysisResult:
        studies = [StudyResult.model_validate(s) for s in (k_data.studies if k_data else [])]
        selection = self.selection or all_reports_excluded(
            self.inputs, "analysis_failed", f"analysis not completed: {reason}"
        )
        included = {rid: s.study_id for s in studies for rid in s.record_ids}
        excluded, reports = account_reports(self.inputs, selection, included, self.excluded)
        return AnalysisResult.model_validate(
            {
                "analysis_id": self.spec.analysis_id,
                "outcome_id": self.outcome.outcome_id,
                "outcome_name": self.outcome.name,
                "effect_measure": self.spec.effect_measure,
                "scale": scale,
                "timepoints": [t.label for t in self.spec.timepoints],
                "status": status,
                "status_reason": reason,
                "k": len(studies),
                "studies": studies,
                "excluded": excluded,
                "reports": reports,
                "sensitivity": self.sensitivity,
                "settings": self._settings(),
                "methods": self.methods,
                "warnings": self.warnings,
                "limitations": self.limitations,
                **kw,
            }
        )

    def _small_k_warnings(self, k: int) -> None:
        if k == 2:
            self.warnings.append(
                "Only two studies: τ² is very imprecisely estimated, a Hartung-Knapp interval "
                "uses t(1) and is very wide, and no prediction interval is given (needs k ≥ 3)."
            )

    def _pool_iv(self, d: _Data, scale: str, back: Transform) -> dict[str, Any]:
        """Common-effect (IV) and random-effects results + heterogeneity from yi/vi."""
        k = len(d.yi)
        fe = fixed_effect(d.yi, d.vi, level=self.level)
        w_fe = 1 / np.asarray(d.vi)
        w_fe = w_fe / w_fe.sum()
        # The weights are returned, not attached: each handler attaches them to the model
        # they belong to (Session 9, M3).
        iv_common_weights = [float(100 * wf) for wf in w_fe]
        self._small_k_warnings(k)
        common = _pooled_result(
            fe, model="common_effect", method="inverse variance", scale=scale, back=back, k=k
        )
        het = _heterogeneity(d.yi, d.vi, f"inverse-variance weights on the {scale} scale")
        # The random-effects fit can fail on its own (e.g. a zero Hartung-Knapp SE when all
        # estimates are identical); that must not discard the estimable results above.
        try:
            re = random_effects(
                d.yi,
                d.vi,
                tau2_method=self.spec.tau2_estimator,
                ci_method=self.spec.random_effects_ci,
                level=self.level,
            )
        except EstimationError as exc:
            return {
                "common": common,
                "random": None,
                "random_error": str(exc),
                "het": het,
                "w_common": iv_common_weights,
                "w_random": None,
            }
        iv_random_weights = [float(100 * wr) for wr in re.weights]
        if self.spec.random_effects_ci == "hksj":
            wald = random_effects(
                d.yi, d.vi, tau2_method=self.spec.tau2_estimator, ci_method="wald", level=self.level
            )
            if re.pooled.se < wald.pooled.se:
                self.warnings.append(
                    "The Hartung-Knapp interval is narrower than the Wald interval (q < 1); "
                    "the ad hoc variant (random_effects_ci='hksj_adhoc') would not be."
                )
        return {
            "common": common,
            "random_error": None,
            "het": het,
            "w_common": iv_common_weights,
            "w_random": iv_random_weights,
            "random": _pooled_result(
                re.pooled,
                model="random_effects",
                method=_re_method_label(self.spec),
                scale=scale,
                back=back,
                k=k,
                re=re,
            ),
        }

    @staticmethod
    def _attach_weights(
        d: _Data,
        *,
        common: list[float] | None = None,
        random: list[float] | None = None,
        sensitivity: dict[str, list[float] | None] | None = None,
    ) -> None:
        """Attach study weights to the model they belong to (Session 9, M3)."""
        for i, s in enumerate(d.studies):
            if common is not None:
                s["weight_common_pct"] = common[i]
            if random is not None:
                s["weight_random_pct"] = random[i]
            for label, w in (sensitivity or {}).items():
                if w is not None:
                    s.setdefault("weight_sensitivity_pct", {})[label] = w[i]

    def _finish(
        self,
        d: _Data,
        scale: str,
        common: PooledResult | None,
        random: PooledResult | None,
        het: HeterogeneityResult | None,
        *,
        random_error: str | None = None,
        common_error: str | None = None,
        **kw: Any,
    ) -> AnalysisResult:
        """Assemble a pooled result; a failed model is reported, never replaced."""
        is_random = self.spec.primary_model == "random_effects"
        primary = random if is_random else common
        if primary is None:
            error = (random_error if is_random else common_error) or "see warnings"
            # other estimable models stay visible, explicitly labelled as auxiliary
            self.sensitivity.extend(r for r in (common, random) if r is not None)
            return self._result(
                "not_estimable",
                f"the primary {self.spec.primary_model.replace('_', '-')} model is not "
                f"estimable: {error}. No substitute interval is reported as primary.",
                scale=scale,
                k_data=d,
                heterogeneity=het,
                **kw,
            )
        failed: list[tuple[str, PooledResult | None, str | None]] = [
            ("random-effects", random, random_error),
            ("common-effect", common, common_error),
        ]
        for model, result, failure in failed:
            if result is None and failure is not None:
                self.warnings.append(
                    f"The {model} model is not estimable: {failure}. No substitute interval is "
                    "reported."
                )
        return self._result(
            "pooled",
            f"{len(d.yi)} studies pooled ({self.spec.primary_model} primary)",
            scale=scale,
            k_data=d,
            primary=primary,
            common_effect=common,
            random_effects=random,
            heterogeneity=het,
            **kw,
        )

    def _no_pool(self, d: _Data, scale: str, **kw: Any) -> AnalysisResult | None:
        k = len(d.studies)
        if k == 0:
            sel = self.selection
            codes = [e.reason_code for e in self.excluded]
            if sel is not None:
                codes += [e.reason_code for e in sel.excluded]
                codes += [r.code for r in sel.record_reasons.values()]
            if codes:
                return self._result(
                    "descriptive_only" if any(c in _DESCRIBABLE for c in codes) else "no_data",
                    "no study provides data that can be analysed for this plan; see excluded "
                    "studies and their reasons",
                    scale=scale,
                    k_data=d,
                    **kw,
                )
            return self._result(
                "no_data", "no approved data for this outcome", scale=scale, k_data=d, **kw
            )
        if k == 1:
            return self._result(
                "single_study",
                "one eligible study: its own estimate is reported; pooling, "
                "heterogeneity and prediction intervals are undefined",
                scale=scale,
                k_data=d,
                **kw,
            )
        return None

    # ---- single-group crude proportions
    def _proportion(self, units: list[Unit]) -> AnalysisResult:
        d = _Data()
        xs: list[int] = []
        ns: list[int] = []
        for u in units:
            arm = u.arms["single_group"]
            x, n = _count(arm.events), _count(arm.n_analyzed)
            if x is None or n is None:
                self._exclude(
                    u,
                    "missing_data",
                    "needs events and n_analyzed (n_randomized is "
                    "never substituted; percentages are never converted to counts)",
                )
                continue
            if n == 0:
                self._exclude(u, "invalid_data", "n_analyzed is zero")
                continue
            est = es.logit_proportion(x, n)
            lo, hi = es.clopper_pearson(x, n, level=self.level)
            notes = (
                ["0.5 added to events and non-events for the logit sensitivity analysis"]
                if est.continuity_corrected
                else []
            )
            d.units.append(u)
            d.yi.append(est.yi)
            d.vi.append(est.vi)
            xs.append(x)
            ns.append(n)
            d.studies.append(
                {
                    "study_id": u.unit_id,
                    "record_ids": list(u.record_ids),
                    "label": u.label,
                    "arm_ids": u.arm_ids,
                    "data": {"events": x, "n": n},
                    "yi": est.yi,
                    "vi": est.vi,
                    "estimate_natural": x / n,
                    "ci_lower_natural": lo,
                    "ci_upper_natural": hi,
                    "ci_method": "Clopper-Pearson exact",
                    "continuity_corrected": est.continuity_corrected,
                    "notes": notes,
                }
            )
        totals: dict[str, Any] = {"total_events": sum(xs), "total_participants": sum(ns)}
        scale = "logit"
        self.methods.append(
            "Crude proportions (events ÷ n analysed, censoring ignored) on the logit scale. "
            "Study CIs: Clopper-Pearson exact."
        )
        early = self._no_pool(d, scale, **totals)
        if early is not None:
            return early
        k = len(xs)
        if all(x == 0 for x in xs) or all(x == n for x, n in zip(xs, ns, strict=True)):
            self.warnings.append(
                "Every study has 0 events (or every study has only events); a pooled estimate "
                "would be an artefact of continuity corrections. Totals are reported instead."
            )
            return self._result(
                "descriptive_only",
                "all studies at the same boundary (0 % or 100 %): no estimable pooled proportion",
                scale=scale,
                k_data=d,
                **totals,
            )
        iv = self._pool_iv(d, scale, es.expit)
        iv_random: PooledResult | None = None
        if iv["random"] is not None:
            iv_random = iv["random"].model_copy(
                update={
                    "method": "logit transform, 0.5 continuity correction for 0 % / 100 % "
                    "studies; " + iv["random"].method
                }
            )
        glmm_error: str | None = None
        try:
            g = glmm_logit(xs, ns, n_agq=GLMM_NODES, level=self.level)
        except EstimationError as exc:
            g = None
            glmm_error = str(exc)
        glmm_res: PooledResult | None = None
        if g is not None:
            pi = None
            if k >= 3:
                half = float(stats.t.ppf(1 - (1 - self.level) / 2, k - 2)) * math.sqrt(
                    g.tau2 + g.se**2
                )
                pi = (g.mu - half, g.mu + half)
            s2 = iv["het"].s2_typical
            glmm_res = PooledResult(
                model="random_effects",
                method=f"binomial-normal GLMM (ML, {GLMM_NODES}-node adaptive Gauss-Hermite), "
                "Wald CI",
                scale=scale,
                estimate=g.mu,
                se=g.se,
                ci_lower=g.ci_lower,
                ci_upper=g.ci_upper,
                estimate_natural=es.expit(g.mu),
                ci_lower_natural=es.expit(g.ci_lower),
                ci_upper_natural=es.expit(g.ci_upper),
                statistic=g.z,
                p_value=g.p_value,
                distribution="normal",
                df=None,
                tau2=g.tau2,
                i2_model=100 * g.tau2 / (g.tau2 + s2),
                prediction_interval=pi,
                prediction_interval_natural=None
                if pi is None
                else (es.expit(pi[0]), es.expit(pi[1])),
                prediction_df=None if pi is None else k - 2,
                k=k,
            )
            if g.at_boundary:
                self.warnings.append("GLMM: τ² estimated at the boundary 0.")
        common = _binomial_common(xs, ns, self.level, k)
        self.methods.append(
            "Primary random-effects model: "
            + (
                "binomial-normal GLMM (exact binomial likelihood; no continuity correction; "
                "Stijnen et al. 2010)."
                if self.spec.proportion_method == "glmm"
                else "inverse-variance logit model (" + _re_method_label(self.spec) + ")."
            )
        )
        self.methods.append(
            "Common-effect model: binomial maximum likelihood (pooled logistic regression). "
            "Heterogeneity: Cochran's Q / I² from logit-transformed proportions with inverse-"
            "variance weights (0.5 correction for 0 % / 100 % studies)."
        )
        if self.spec.proportion_method == "glmm":
            random, random_error = glmm_res, glmm_error
            aux, aux_error, aux_name = iv_random, iv["random_error"], "inverse-variance logit"
        else:
            random, random_error = iv_random, iv["random_error"]
            aux, aux_error, aux_name = glmm_res, glmm_error, "binomial-normal GLMM"
        # Weights (M3): the binomial ML common-effect model and the binomial-normal GLMM have
        # no inverse-variance study weights, so none are shown for them. The IV-logit
        # random-effects weights are attached to the IV-logit model only — as the reported
        # random-effects model when it is primary, else as a labelled sensitivity model.
        if self.spec.proportion_method == "glmm":
            self._attach_weights(d, sensitivity={IV_LOGIT_RE_LABEL: iv["w_random"]})
            self.methods.append(
                "Study weights: the binomial-normal GLMM and the binomial maximum-likelihood "
                "common-effect model have no inverse-variance study weights, so none are shown "
                "for them (metafor likewise provides no weights() for rma.glmm); forest-plot "
                "boxes are drawn at equal size. Weights of the inverse-variance logit "
                "sensitivity model are reported separately and labelled as such."
            )
        else:
            self._attach_weights(d, random=iv["w_random"])
        if aux is not None:
            self.sensitivity.append(aux)
        else:
            self.warnings.append(
                f"Sensitivity analysis ({aux_name} random-effects model) not estimable: "
                f"{aux_error}. No substitute interval is reported; the primary result is "
                "unaffected."
            )
        self.limitations.append(
            "Crude proportions ignore censoring and differing follow-up durations; they are not "
            "actuarial or Kaplan-Meier probabilities."
        )
        self.limitations.append(
            "The GLMM confidence interval is a Wald (normal) interval from the observed "
            "information; with few studies it may be too narrow. Compare it with the "
            "inverse-variance Hartung-Knapp sensitivity analysis."
        )
        return self._finish(
            d, scale, common, random, iv["het"], random_error=random_error, **totals
        )

    # ---- KM / cumulative incidence at time t
    def _time_t(self, units: list[Unit]) -> AnalysisResult:
        scale = self.spec.reported_ci_scale
        assert scale is not None  # enforced by AnalysisSpec
        d = _Data()
        for u in units:
            arm = u.arms["single_group"]
            problem = _check_time_t(arm, scale)
            if problem is not None:
                self._exclude(u, problem[0], problem[1])
                continue
            p, lo, hi, h, level = _probabilities(arm)
            try:  # a single unusable study is excluded; it never aborts the analysis (M1)
                yi = es.to_scale(p, scale)
                z = _zcrit(level)
                se = abs(es.to_scale(hi, scale) - es.to_scale(lo, scale)) / (2 * z)
            except EstimationError as exc:  # pragma: no cover - _check_time_t screens these
                self._exclude(u, "invalid_data", f"not transformable to the {scale} scale: {exc}")
                continue
            d.units.append(u)
            d.yi.append(yi)
            d.vi.append(se**2)
            ylo, yhi = study_wald_ci(yi, se**2, level=self.level)
            d.studies.append(
                {
                    "study_id": u.unit_id,
                    "record_ids": list(u.record_ids),
                    "label": u.label,
                    "arm_ids": u.arm_ids,
                    "data": {
                        "estimate": p,
                        "ci_lower": lo,
                        "ci_upper": hi,
                        "ci_level": level * 100,
                        # half a unit of the last printed digit of each number (Session 9, M2)
                        "rounding_half_unit_estimate": h[0],
                        "rounding_half_unit_ci_lower": h[1],
                        "rounding_half_unit_ci_upper": h[2],
                    },
                    "yi": yi,
                    "vi": se**2,
                    "ci_lower": ylo,
                    "ci_upper": yhi,
                    "estimate_natural": p,
                    "ci_lower_natural": lo,
                    "ci_upper_natural": hi,
                    "ci_method": f"as reported ({level * 100:g}% CI); SE on the {scale} scale",
                    "notes": [f"estimator: {arm.estimate_type}"],
                }
            )
        self.methods.append(
            f"{self.spec.effect_measure} at a fixed time: each study's SE derived from its "
            f"reported CI assuming symmetry on the {scale} scale, SE = |g(U) − g(L)| / (2 z); "
            "pooled by inverse variance on that scale and back-transformed."
        )
        self.limitations.append(
            "Pooling relies on author-reported CIs; studies without a usable CI are described "
            "but not pooled. Kaplan-Meier, actuarial and cumulative-incidence estimates are "
            "never pooled with each other or with crude proportions"
            + (
                " (Kaplan-Meier and actuarial combined by plan)."
                if self.spec.combine_km_and_actuarial
                else "."
            )
        )
        early = self._no_pool(d, scale)
        if early is not None:
            return early
        back: Transform = lambda t: es.from_scale(t, scale)  # noqa: E731
        iv = self._pool_iv(d, scale, back)
        self._attach_weights(d, common=iv["w_common"], random=iv["w_random"])
        return self._finish(
            d, scale, iv["common"], iv["random"], iv["het"], random_error=iv["random_error"]
        )

    # ---- comparative dichotomous
    def _ratio(self, units: list[Unit]) -> AnalysisResult:
        is_rr = self.spec.effect_measure == EffectMeasure.RR
        measure: Literal["RR", "OR"] = "RR" if is_rr else "OR"
        d = _Data()
        tables: list[tuple[int, int, int, int]] = []
        for u in units:
            ia, ca = u.arms["intervention"], u.arms["control"]
            a, n1, c, n2 = (
                _count(ia.events),
                _count(ia.n_analyzed),
                _count(ca.events),
                _count(ca.n_analyzed),
            )
            if a is None or n1 is None or c is None or n2 is None:
                self._exclude(u, "missing_data", "needs events and n_analyzed in both arms")
                continue
            if n1 == 0 or n2 == 0:
                self._exclude(u, "invalid_data", "an arm has n_analyzed = 0")
                continue
            if not es.two_by_two_informative(a, n1, c, n2):
                self._exclude(
                    u,
                    "non_informative",
                    f"{a}/{n1} vs {c}/{n2}: no events (or only events) in both arms "
                    f"carries no information on the {measure}",
                )
                continue
            est = es.log_ratio(measure, a, n1, c, n2)
            lo, hi = study_wald_ci(est.yi, est.vi, level=self.level)
            tables.append((a, n1, c, n2))
            d.units.append(u)
            d.yi.append(est.yi)
            d.vi.append(est.vi)
            d.studies.append(
                {
                    "study_id": u.unit_id,
                    "record_ids": list(u.record_ids),
                    "label": u.label,
                    "arm_ids": u.arm_ids,
                    "data": {
                        "events_intervention": a,
                        "n_intervention": n1,
                        "events_control": c,
                        "n_control": n2,
                    },
                    "yi": est.yi,
                    "vi": est.vi,
                    "ci_lower": lo,
                    "ci_upper": hi,
                    "estimate_natural": math.exp(est.yi),
                    "ci_lower_natural": math.exp(lo),
                    "ci_upper_natural": math.exp(hi),
                    "ci_method": "Wald on the log scale",
                    "continuity_corrected": est.continuity_corrected,
                    "notes": ["0.5 added to all cells (zero cell)"]
                    if est.continuity_corrected
                    else [],
                }
            )
        totals: dict[str, Any] = {
            "total_events": sum(t[0] + t[2] for t in tables),
            "total_participants": sum(t[1] + t[3] for t in tables),
        }
        scale = "log"
        self.methods.append(
            f"{measure} on the log scale (escalc conventions: double-zero / double-all-event "
            "studies excluded; 0.5 added to all cells of studies with a zero cell for inverse-"
            "variance analyses only)."
        )
        early = self._no_pool(d, scale, **totals)
        if early is not None:
            return early
        iv = self._pool_iv(d, scale, math.exp)
        common: PooledResult | None = iv["common"]
        common_error: str | None = None
        use_mh = self.spec.common_effect_method in ("auto", "mantel_haenszel")
        self._attach_weights(d, random=iv["w_random"])
        if use_mh:
            # The IV common-effect model becomes a sensitivity analysis; its weights are
            # labelled as such and never shown as the Mantel-Haenszel weights (M3).
            self._attach_weights(d, sensitivity={IV_COMMON_LABEL: iv["w_common"]})
            try:
                mh = mantel_haenszel(measure, tables, level=self.level)
                self._attach_weights(d, common=mantel_haenszel_weights(measure, tables))
                common = _pooled_result(
                    mh,
                    model="common_effect",
                    method="Mantel-Haenszel ("
                    + ("Robins-Breslow-Greenland" if measure == "OR" else "Greenland-Robins")
                    + " variance)",
                    scale=scale,
                    back=math.exp,
                    k=len(tables),
                )
                self.sensitivity.append(iv["common"])
                self.methods.append(
                    "Common-effect model: Mantel-Haenszel on raw counts; study weights are the "
                    "Mantel-Haenszel weights (b·c/N for OR, c·n₁/N for RR), as in metafor's "
                    "weights.rma.mh. Inverse-variance common-effect weights belong to the "
                    "sensitivity analysis and are reported separately."
                )
            except EstimationError as exc:
                common = None
                common_error = f"Mantel-Haenszel estimate not estimable: {exc}"
                self.sensitivity.append(iv["common"])
        else:
            self._attach_weights(d, common=iv["w_common"])
            self.methods.append("Common-effect model: inverse variance.")
        self.methods.append(f"Random-effects model: {_re_method_label(self.spec)}.")
        return self._finish(
            d,
            scale,
            common,
            iv["random"],
            iv["het"],
            random_error=iv["random_error"],
            common_error=common_error,
            **totals,
        )

    # ---- comparative continuous
    def _continuous(self, units: list[Unit]) -> AnalysisResult:
        smd = self.spec.effect_measure == EffectMeasure.SMD
        d = _Data()
        units_seen: set[str | None] = set()
        for u in units:
            ia, ca = u.arms["intervention"], u.arms["control"]
            vals = [
                _num(ia.mean),
                _num(ia.sd),
                _count(ia.n_analyzed),
                _num(ca.mean),
                _num(ca.sd),
                _count(ca.n_analyzed),
            ]
            if any(v is None for v in vals):
                self._exclude(
                    u,
                    "missing_data",
                    "needs mean, SD and n_analyzed in both arms (SDs are never derived "
                    "from SEs, CIs, ranges or IQRs; medians are not used)",
                )
                continue
            m1, s1, n1, m2, s2, n2 = (float(v) for v in vals)  # type: ignore[arg-type]
            unit_names = {v.unit for v in (ia.mean, ia.sd, ca.mean, ca.sd) if v is not None}
            if len(unit_names) > 1:
                self._exclude(
                    u,
                    "unit_mismatch",
                    f"arms report different units {sorted(map(str, unit_names))}",
                )
                continue
            try:
                fn = es.standardised_mean_difference if smd else es.mean_difference
                est = fn(m1, s1, int(n1), m2, s2, int(n2))
            except EstimationError as exc:
                self._exclude(u, "invalid_data", str(exc))
                continue
            units_seen.add(next(iter(unit_names)) if unit_names else None)
            lo, hi = study_wald_ci(est.yi, est.vi, level=self.level)
            d.units.append(u)
            d.yi.append(est.yi)
            d.vi.append(est.vi)
            d.studies.append(
                {
                    "study_id": u.unit_id,
                    "record_ids": list(u.record_ids),
                    "label": u.label,
                    "arm_ids": u.arm_ids,
                    "data": {
                        "mean_intervention": m1,
                        "sd_intervention": s1,
                        "n_intervention": n1,
                        "mean_control": m2,
                        "sd_control": s2,
                        "n_control": n2,
                    },
                    "yi": est.yi,
                    "vi": est.vi,
                    "ci_lower": lo,
                    "ci_upper": hi,
                    "estimate_natural": est.yi,
                    "ci_lower_natural": lo,
                    "ci_upper_natural": hi,
                    "ci_method": "Wald",
                }
            )
        scale = "SMD (Hedges' g)" if smd else "mean difference"
        if not smd and len(units_seen) > 1:
            self.warnings.append(
                f"Studies report different units {sorted(map(str, units_seen))}; mean "
                "differences are not pooled across units (no conversion is performed)."
            )
            return self._result(
                "not_estimable",
                "inconsistent measurement units across studies",
                scale=scale,
                k_data=d,
            )
        self.methods.append(
            "Hedges' g with exact small-sample correction J; variance 1/n₁ + 1/n₂ + g²/(2(n₁+n₂))."
            if smd
            else "Mean difference m₁ − m₂, variance s₁²/n₁ + s₂²/n₂."
        )
        early = self._no_pool(d, scale)
        if early is not None:
            return early
        iv = self._pool_iv(d, scale, _identity)
        self._attach_weights(d, common=iv["w_common"], random=iv["w_random"])
        self.methods.append(
            f"Common-effect: inverse variance. Random effects: {_re_method_label(self.spec)}."
        )
        return self._finish(
            d, scale, iv["common"], iv["random"], iv["het"], random_error=iv["random_error"]
        )


_DESCRIBABLE = frozenset(
    {
        "missing_data",
        "boundary_estimate",
        "ci_inconsistent",
        "ci_bound_not_transformable",
        "non_informative",
        "invalid_data",
    }
)
_UNSUPPORTED_DEFAULT = "this summary measure is not supported by the statistics engine"
_UNSUPPORTED: dict[EffectMeasure, str] = {
    EffectMeasure.HR: (
        "Hazard ratios are not pooled: the extraction schema holds no contrast-level HR with "
        "its CI, and an HR is never derived from survival percentages or curves."
    ),
    EffectMeasure.RD: "Risk differences are not implemented/validated in this release.",
    EffectMeasure.IRR: "Incidence-rate ratios are not implemented/validated in this release.",
    EffectMeasure.INCIDENCE_RATE: "Incidence rates (events per person-time) are not "
    "implemented/validated in this release.",
    EffectMeasure.MEAN: "Single-group means are not implemented/validated in this release.",
    EffectMeasure.MEDIAN: "Medians cannot be pooled by inverse variance without unvalidated "
    "conversions; not supported.",
    EffectMeasure.MEDIAN_TIME_TO_EVENT: "Median times to event are not pooled (no validated "
    "method without individual participant data).",
}


def _binomial_common(xs: list[int], ns: list[int], level: float, k: int) -> PooledResult | None:
    """Common-effect binomial ML: logit(Σx / Σn), SE = √(1/Σx + 1/(Σn − Σx))."""
    sx, sn = sum(xs), sum(ns)
    if sx == 0 or sx == sn:
        return None
    mu = math.log(sx / (sn - sx))
    se = math.sqrt(1 / sx + 1 / (sn - sx))
    z = _zcrit(level)
    p = Pooled(
        mu,
        se,
        mu - z * se,
        mu + z * se,
        mu / se,
        float(2 * stats.norm.sf(abs(mu / se))),
        "normal",
        None,
    )
    return _pooled_result(
        p,
        model="common_effect",
        method="binomial maximum likelihood",
        scale="logit",
        back=es.expit,
        k=k,
    )


def _zcrit(level: float) -> float:
    return float(stats.norm.ppf(1 - (1 - level) / 2))


def _decimals(text: str) -> int:
    t = text.strip().rstrip("%")
    for sep in (".", "·", ","):
        if sep in t:
            return len(t.split(sep)[-1])
    return 0


HalfUnits = tuple[float, float, float]


def _probabilities(arm: ExtractedStudyArm) -> tuple[float, float, float, HalfUnits, float]:
    """(p, lower, upper, rounding half-units, ci level) on the proportion scale.

    Each printed number carries its own precision: "62" can stand for any value in
    [61.5, 62.5) while "56.3" stands for [56.25, 56.35). The rounding half-unit is therefore
    derived separately for the estimate, the lower and the upper limit (Session 9, M2)."""
    est, lo, hi = arm.reported_estimate, arm.ci_lower, arm.ci_upper
    if est is None or lo is None or hi is None or arm.ci_level is None:
        raise ValueError("estimate and confidence interval required")
    div = 100.0 if arm.estimate_scale == "percent" else 1.0
    h = tuple(0.5 * 10.0 ** (-_decimals(v.reported_text)) / div for v in (est, lo, hi))
    return est.value / div, lo.value / div, hi.value / div, (h[0], h[1], h[2]), arm.ci_level / 100.0


def _check_time_t(arm: ExtractedStudyArm, scale: es.ProbabilityScale) -> tuple[str, str] | None:
    if arm.reported_estimate is None:
        return "missing_data", "no reported estimate at the analysis timepoint"
    if arm.ci_lower is None or arm.ci_upper is None or arm.ci_level is None:
        return "missing_data", (
            "no reported confidence interval: the estimate is described but not pooled "
            "(an SE is never assumed or derived from n alone)"
        )
    p, lo, hi, h, _ = _probabilities(arm)
    if not 0.0 < p < 1.0:
        return "boundary_estimate", (
            f"estimate {p:g} lies on the boundary; its variance cannot be derived on the "
            f"{scale} scale — described, not pooled"
        )
    if hi <= lo:
        return "invalid_data", "zero-width confidence interval"
    if not (es.scale_domain_contains(lo, scale) and es.scale_domain_contains(hi, scale)):
        # Only the transformed scales are undefined at 0 or 1; on the identity ("plain")
        # scale a limit of exactly 0 % or 100 % is an ordinary value (M1).
        return "ci_bound_not_transformable", (
            f"a confidence limit of {lo:g}–{hi:g} reaches 0 or 1, where the declared {scale} "
            "transformation is undefined, so the study SE cannot be recovered on that scale; "
            "described, not pooled (the source values and the declared scale are unchanged)"
        )
    if not _symmetric_within_rounding(p, lo, hi, h, scale):
        return "ci_inconsistent", (
            f"the reported CI ({lo:g}–{hi:g} around {p:g}) is not symmetric on the declared "
            f"{scale} scale even allowing for the rounding of each printed number "
            f"(estimate ±{h[0]:g}, lower ±{h[1]:g}, upper ±{h[2]:g}); the authors probably "
            "used another method, so the SE cannot be recovered"
        )
    return None


def _symmetric_within_rounding(
    p: float, lo: float, hi: float, h: HalfUnits, scale: es.ProbabilityScale
) -> bool:
    """Is g(U) + g(L) − 2 g(p) = 0 achievable with each printed number moved within its own
    rounding interval (p ± h_p, L ± h_L, U ± h_U)?

    Each interval is intersected with the domain of the scale (the closed [0, 1] for the
    identity scale; just inside the open (0, 1) for the transformed scales). g is monotone,
    so the expression is monotone in each argument and its range over the box is spanned by
    the two opposite corners below (their roles swap for decreasing g, e.g. log(−log p))."""
    eps = 1e-12
    lo_dom, hi_dom = (0.0, 1.0) if scale == "plain" else (eps, 1.0 - eps)

    def g(x: float) -> float:
        return es.to_scale(min(max(x, lo_dom), hi_dom), scale)

    hp, hl, hu = h
    a = g(hi - hu) + g(lo - hl) - 2 * g(p + hp)
    b = g(hi + hu) + g(lo + hl) - 2 * g(p - hp)
    return min(a, b) <= 0.0 <= max(a, b)


# --------------------------------------------------------------------------- public API


def run_analysis(
    spec: AnalysisSpec, outcome: OutcomeDefinition, inputs: AnalysisInputs
) -> AnalysisResult:
    try:
        return _Analysis(spec, outcome, inputs).run()
    except StatisticsInputError:
        raise
    except EstimationError as exc:
        a = _Analysis(spec, outcome, inputs)
        a.warnings.append(str(exc))
        return a._result("not_estimable", f"model estimation failed: {exc}", scale="none")


def run_plan(
    plan: AnalysisPlan,
    inputs: AnalysisInputs,
    *,
    project_id: str,
    generated_at: datetime,
) -> StatsOutput:
    """Run every analysis of an approved plan. Fails closed on unapproved inputs."""
    if plan.status != "approved":
        raise StatisticsInputError("the analysis plan has not been approved by a human reviewer")
    inputs.validate()
    analyses: list[AnalysisResult] = []
    for spec in plan.analyses:
        outcome = inputs.outcomes.get(spec.outcome_id)
        if outcome is None:
            raise StatisticsInputError(
                f"analysis {spec.analysis_id}: outcome {spec.outcome_id} is not in the protocol"
            )
        analyses.append(run_analysis(spec, outcome, inputs))
    plan_sha, input_sha = plan.sha256(), inputs.sha256()
    return StatsOutput(
        engine_version=ENGINE_VERSION,
        project_id=project_id,
        plan_id=plan.plan_id,
        plan_sha256=plan_sha,
        input_sha256=input_sha,
        results_sha256=results_fingerprint(analyses, plan_sha, input_sha),
        generated_at=generated_at,
        software=software_versions(),
        numeric_precision=NUMERIC_PRECISION,
        analyses=analyses,
        limitations=GLOBAL_LIMITATIONS,
    )
