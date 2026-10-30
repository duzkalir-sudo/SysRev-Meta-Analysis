"""Deterministic statistics engine (Stage 5) — interface defined, implementation pending.

Guardrail: the LLM never performs arithmetic. Every effect size, pooled
estimate, heterogeneity statistic, bias test and Median/IQR/Range-to-Mean/SD
conversion (Wan et al. BMC Med Res Methodol 2014;14:135; Luo et al. Stat
Methods Med Res 2018;27:1785-1805) will be implemented here in
scipy/statsmodels (and optionally R ``metafor``) with unit tests against
published reference values, and written to ``stats_output.json``.
"""
