"""Absolute effects for Summary of Findings tables (deterministic arithmetic only).

An absolute effect is obtained by applying the pooled *relative* effect computed by the
statistics engine (risk ratio or odds ratio, with its confidence interval) to an *assumed
comparator risk* that a human entered together with its source (GRADE guidelines 12:
Guyatt GH, et al. J Clin Epidemiol 2013;66:158-72, doi:10.1016/j.jclinepi.2012.01.012).
Without such a baseline risk nothing is calculated.

* risk ratio:  risk with intervention = ACR × RR
* odds ratio:  risk with intervention = ACR × OR / (1 − ACR + ACR × OR)

The confidence interval of the risk with the intervention applies the same formula to the
bounds of the relative effect; as in standard Summary of Findings practice, the uncertainty
of the assumed comparator risk itself is not propagated (it is an assumption, stated with
its source). Results are shown per 1000 people, rounded to whole numbers for display only;
the full-precision values are kept for the Auditor.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.enums import EffectMeasure
from src.manuscript.formatting import DisplayNumber, fmt

PER = 1000
SUPPORTED = frozenset({EffectMeasure.RR, EffectMeasure.OR})


class AbsoluteEffectError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class AbsoluteEffect:
    baseline: float  # assumed comparator risk (0-1)
    risk: float  # risk with the intervention at the point estimate (0-1)
    risk_lower: float
    risk_upper: float

    @property
    def difference(self) -> tuple[float, float, float]:
        """(point, lower, upper) risk differences per PER people."""
        d = [(r - self.baseline) * PER for r in (self.risk, self.risk_lower, self.risk_upper)]
        return d[0], min(d[1], d[2]), max(d[1], d[2])


def _apply(measure: EffectMeasure, baseline: float, relative: float) -> float:
    if measure == EffectMeasure.RR:
        risk = baseline * relative
    elif measure == EffectMeasure.OR:
        risk = baseline * relative / (1 - baseline + baseline * relative)
    else:
        raise AbsoluteEffectError(f"absolute effects are not derived for {measure}")
    if not 0 <= risk <= 1:
        raise AbsoluteEffectError(
            f"the relative effect applied to an assumed comparator risk of {baseline:g} gives an "
            f"impossible risk ({risk:g}); this baseline cannot be combined with this risk ratio"
        )
    return risk


def absolute_effect(
    measure: EffectMeasure, baseline: float, estimate: float, lower: float, upper: float
) -> AbsoluteEffect:
    if not 0 < baseline < 1:
        raise AbsoluteEffectError("the assumed comparator risk must lie strictly between 0 and 1")
    if measure not in SUPPORTED:
        raise AbsoluteEffectError(
            f"absolute effects are derived only from risk ratios and odds ratios, not {measure}"
        )
    return AbsoluteEffect(
        baseline,
        _apply(measure, baseline, estimate),
        _apply(measure, baseline, lower),
        _apply(measure, baseline, upper),
    )


def _per(x: float) -> tuple[str, DisplayNumber]:
    return fmt(x * PER, 0), DisplayNumber("absolute_risk", x * PER, 0)


def _per_n() -> DisplayNumber:
    return DisplayNumber("per_n", float(PER), 0)


def baseline_text(baseline: float) -> tuple[str, tuple[DisplayNumber, ...]]:
    t, n = _per(baseline)
    return f"{t} per {PER}", (n, _per_n())


def risk_text(eff: AbsoluteEffect, level: float) -> tuple[str, tuple[DisplayNumber, ...]]:
    """'85 per 1000 (95% CI 60 to 118)'."""
    lo, hi = sorted((eff.risk_lower, eff.risk_upper))
    t, n = _per(eff.risk)
    tl, nl = _per(lo)
    th, nh = _per(hi)
    lvl = DisplayNumber("ci_level", level, 0)
    return (
        f"{t} per {PER} ({fmt(level, 0)}% CI {tl} to {th})",
        (n, _per_n(), lvl, nl, nh),
    )


def _signed(x: float) -> tuple[str, DisplayNumber]:
    shown = fmt(abs(x), 0)
    word = "" if shown == "0" else (" more" if x > 0 else " fewer")
    return f"{shown}{word}", DisplayNumber("risk_difference", abs(x), 0)


def difference_text(eff: AbsoluteEffect, level: float) -> tuple[str, tuple[DisplayNumber, ...]]:
    """'34 fewer per 1000 (95% CI from 59 fewer to 2 more)'. Magnitudes are shown with a
    'more'/'fewer' word, so the displayed numbers are absolute values."""
    d, lo, hi = eff.difference
    t, n = _signed(d)
    tl, nl = _signed(lo)
    th, nh = _signed(hi)
    lvl = DisplayNumber("ci_level", level, 0)
    return (
        f"{t} per {PER} ({fmt(level, 0)}% CI from {tl} to {th})",
        (n, _per_n(), lvl, nl, nh),
    )
