"""Deterministic statistics engine (Stage 5).

Guardrail: the LLM never performs arithmetic. Effect sizes, pooled estimates and
heterogeneity statistics are computed here from human-approved extraction data according
to a human-approved analysis plan, cross-validated against R ``metafor``
(``validation/``), and written to a versioned, hashed ``stats_output.json``.

* :mod:`.numerics` — inverse-variance common/random effects, REML/DL τ², HKSJ, Q-profile,
  prediction intervals
* :mod:`.effect_sizes` — log RR/OR, MD, Hedges' g, logit proportions, Clopper-Pearson,
  probability-scale transforms
* :mod:`.mantel_haenszel` — MH OR/RR
* :mod:`.glmm` — binomial-normal GLMM for single proportions
* :mod:`.inputs` — input integrity (approval, estimator, timepoint, arms, cohorts)
* :mod:`.engine` — runs a plan, returns :class:`~src.core.schemas.stats.StatsOutput`
* :mod:`.report` — tables, Markdown report, forest plots
* :mod:`.stage5` — pipeline integration (import it directly; it depends on the repository)
"""

from src.stats_engine.engine import ENGINE_VERSION, run_analysis, run_plan
from src.stats_engine.inputs import AnalysisInputs

__all__ = ["ENGINE_VERSION", "AnalysisInputs", "run_analysis", "run_plan"]
