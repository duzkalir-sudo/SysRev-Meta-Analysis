# Stage 5 numerical validation against R `metafor`

This report records how `src/stats_engine` was checked against an independent
implementation. It is reproducible with the commands at the end.

## Reference software

| Component | Version |
|---|---|
| R | 4.3.3 (2024-02-29), Ubuntu 24.04 package `r-base-core` |
| metafor | 4.4-0 (Viechtbauer W. *J Stat Softw* 2010;36(3):1-48) |
| metadat | 1.2-0 (source of the real example datasets) |
| lme4 | 1.1-35.1 (used by `metafor::rma.glmm`) |
| jsonlite | 1.8.8 |

The R package `meta` could not be installed in the development environment (CRAN's
current release requires a newer R than 4.3.3), so **no comparison with `meta` was made**.

## Datasets

| Case | Type | Source |
|---|---|---|
| `pritz1997` | 14 single-group proportions incl. two all-event studies (neurosurgery: hyperdynamic therapy for cerebral vasospasm) | `metadat::dat.pritz1997` |
| `bcg` | 13 two-arm trials (RR, OR) | `metadat::dat.bcg` (Colditz et al., *JAMA* 1994;271:698-702) |
| `hine1989` | 6 sparse two-arm trials (RR, OR) | `metadat::dat.hine1989` |
| `normand1999` | 9 two-arm trials, means/SDs (MD, SMD) | `metadat::dat.normand1999` (Normand, *Stat Med* 1999;18:321-359) |
| `srs_radionecrosis_zero_events` | 6 proportions, two with 0 events | synthetic |
| `srs_tumour_control_all_events` | 5 proportions, two with all events | synthetic |
| `two_studies` | 2 proportions | synthetic |
| `sparse_with_double_zero` | 6 two-arm tables incl. a double-zero study and single-zero cells | synthetic |
| `srs_5yr_local_control` | 4 Kaplan-Meier probabilities with CIs built on the log(−log) scale and printed to 0.1 % | synthetic |
| `homogeneous_tau2_zero` | 4 generic estimates with τ² = 0 | synthetic |
| `two_studies_heterogeneous` | 2 generic estimates, high heterogeneity | synthetic |

Synthetic cases (`validation/r/synthetic_cases.json`) are invented for testing and are not
data from any study. Full bibliographic details of the metadat datasets are in the metadat
documentation (`help(dat.pritz1997, package = "metadat")`, etc.).

## What is compared

For every dataset: study-level yᵢ and vᵢ (escalc conventions: `add = 1/2, to = "only0"`,
`drop00 = TRUE` for RR/OR); common-effect IV estimate, SE, CI, z, p; Cochran's Q, its p and
I²; and for each τ² estimator (REML, DL) × CI method (Wald `test="z"`, HKSJ `test="knha"`,
ad hoc HKSJ `test="adhoc"`): estimate, SE, CI, statistic, p, τ², model I², prediction interval
(`predict(pi.type = "Riley")`, i.e. t with k − 2 df, k ≥ 3) and Q-profile CIs for τ² and I²
(`confint(type = "QP")`). For proportions additionally: the binomial-normal GLMM
(`rma.glmm(measure = "PLO")`, ML, `nAGQ = 7`) estimate, SE, CI, z, p, τ² and prediction
interval; the common-effect binomial GLMM (`method = "EE"`); and Clopper-Pearson intervals
(`binom.test`). For 2×2 tables additionally Mantel-Haenszel RR/OR (`rma.mh`). For KM
probabilities the SE derivation from the reported CI was re-implemented independently in R.

## Results (1,523 comparisons, all within tolerance)

| Class | Tolerance (abs / rel) | Largest absolute difference observed |
|---|---|---|
| Closed-form (escalc, IV, DL, MH, Q, I², Clopper-Pearson) | 1e-10 / 1e-9 | 4.8e-13 |
| REML (metafor run with `threshold = 1e-12`) | 1e-8 / 1e-7 | 7.6e-11 |
| Q-profile τ² / I² CIs | 1e-6 / 1e-6 | 3.7e-11 |
| GLMM (lme4 vs our adaptive Gauss-Hermite ML) | 2e-4 / 2e-4 | 1.3e-5 |
| GLMM p-values | 1e-5 / 5e-3 | 3.7e-9 |

The published BCG example (Viechtbauer 2010: log RR −0.7145, SE 0.1798, τ² 0.3132,
I² 92.22 %, Q = 152.2330) is also reproduced to the printed precision
(`tests/unit/test_stats_metafor_reference.py::test_bcg_matches_published_values`).

### Discrepancy found and resolved

The first comparison flagged Q-profile **upper** τ² bounds in three cases. metafor's
`confint` searches τ² only up to `tau2.max` (default 100 here) and prints `>100`;
re-running R with `tau2.max = 1e6` gives the same root as our implementation (e.g. 860.3832
for `two_studies_heterogeneous`). The reference script now widens the search range; the
Python implementation was not changed.

### Defect found by testing and fixed

A coarser grid in the GLMM τ search exposed a defect: the Newton iteration for each
study's conditional mode had no step control, so at extreme μ the marginal likelihood was
not monotone and the inner optimiser could jump to a wrong mode. The iteration now halves
steps until the (log-concave) integrand increases; a regression test checks unimodality in
μ, and agreement with lme4 is unchanged.

### Session 9 additions

* **Mantel-Haenszel study weights** (`weights.rma.mh`) are now part of the reference
  (`expected.<RR|OR>.mh.weights`, renormalised over the studies the engine keeps): 48
  comparisons, all within the closed-form tolerance. Regenerating the reference with R 4.3.3 /
  metafor 4.4-0 changed no other value.
* **Observation (not changed):** `rma.mh(drop00 = TRUE)` applies `drop00` only to the
  study-level `escalc` outcomes. A double-all-event study (all participants with the event in
  both arms) therefore still enters metafor's Mantel-Haenszel *risk ratio* sums, while this
  engine excludes such studies from every RR/OR analysis as uninformative. In a synthetic check
  (10/10 vs 12/12, 3/30 vs 6/28, 5/25 vs 2/30) metafor gives MH RR 1.017 with and 1.041
  without the double-all-event study. Double-zero studies contribute nothing to either sum,
  and double-all-event studies nothing to the OR, so those estimates agree. No validation case
  contains a double-all-event two-arm study; the engine's convention is documented in
  `src/stats_engine/mantel_haenszel.py`.

## Reproducing

```bash
sudo apt-get install r-base-core r-cran-metafor r-cran-lme4 r-cran-jsonlite   # Ubuntu 24.04
Rscript validation/r/metafor_reference.R validation/r/synthetic_cases.json /tmp/ref.json
python -m validation.compare_metafor /tmp/ref.json
pytest tests/unit/test_stats_metafor_reference.py      # live R test runs when Rscript exists
```

The CI job `r-validation` performs these steps on every pull request.

## Not validated

Anything not listed above, in particular: HR, RD, IRR, incidence rates, single-group means
and medians (all reported as `unsupported`); meta-regression; publication-bias tests; and
the forest-plot rendering (checked for determinism, not against R graphics).
