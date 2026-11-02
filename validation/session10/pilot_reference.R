#!/usr/bin/env Rscript
# Session 10: independent reference for the statistics of a real pilot project.
#
# Usage: Rscript validation/session10/pilot_reference.R <pilot_r_input.json> <output.json>
#
# The input is written by `python -m validation.session10.check_pilot_stats`; it holds, per
# analysis, the study-level counts that the stored stats_output.json says were used, and the
# analysis-plan settings. Nothing is read from the application's code or database.
# Every model is recomputed with metafor (rma, rma.mh, rma.glmm), with the conventions of
# validation/r/metafor_reference.R: escalc(add = 1/2, to = "only0", drop00 = TRUE).
suppressPackageStartupMessages({
  library(metafor)
  library(jsonlite)
})
args <- commandArgs(trailingOnly = TRUE)
inp <- fromJSON(args[1], simplifyVector = FALSE)
ctrl <- list(threshold = 1e-12, maxiter = 10000, stepadj = 0.5)

num <- function(x) if (is.null(x)) NA_real_ else as.numeric(x)
summ <- function(m, back, ids) {
  list(estimate = unname(c(m$beta)), se = m$se, ci_lower = m$ci.lb, ci_upper = m$ci.ub,
       estimate_natural = back(unname(c(m$beta))), ci_lower_natural = back(m$ci.lb),
       ci_upper_natural = back(m$ci.ub), tau2 = if (is.null(m$tau2)) NA_real_ else m$tau2,
       k = m$k, study_ids = ids)
}
safe <- function(expr) tryCatch(expr, error = function(e) list(error = conditionMessage(e)))

out <- list(software = list(R = R.version.string,
                            metafor = as.character(packageVersion("metafor")),
                            lme4 = as.character(packageVersion("lme4"))),
            analyses = list())
for (a in inp$analyses) {
  level <- 100 * num(a$confidence_level)
  ids <- vapply(a$studies, function(s) s$study_id, "")
  res <- list(analysis_id = a$analysis_id, measure = a$measure)
  test_re <- switch(a$random_effects_ci, hksj = "knha", hksj_adhoc = "adhoc", wald = "z")
  if (a$measure == "PROPORTION") {
    xi <- vapply(a$studies, function(s) num(s$events), 0)
    ni <- vapply(a$studies, function(s) num(s$n), 0)
    d <- escalc(measure = "PLO", xi = xi, ni = ni, add = 1/2, to = "only0")
    res$iv_common <- safe(summ(rma(yi, vi, data = d, method = "EE", level = level), plogis, ids))
    res$iv_random <- safe(summ(rma(yi, vi, data = d, method = a$tau2_estimator, test = test_re,
                                   level = level, control = ctrl), plogis, ids))
    res$glmm_random <- safe(summ(rma.glmm(measure = "PLO", xi = xi, ni = ni, method = "ML",
                                          nAGQ = 7, level = level), plogis, ids))
    res$glmm_common <- safe(summ(rma.glmm(measure = "PLO", xi = xi, ni = ni, method = "EE",
                                          level = level), plogis, ids))
    fe <- safe(rma(yi, vi, data = d, method = "EE"))
    if (is.null(fe$error)) {
      res$heterogeneity <- list(Q = fe$QE, Qp = fe$QEp, I2 = fe$I2, df = fe$k - 1)
      res$iv_common$weights_pct <- as.numeric(weights(fe))
    }
    re <- safe(rma(yi, vi, data = d, method = a$tau2_estimator, control = ctrl))
    if (is.null(re$error)) res$iv_random$weights_pct <- as.numeric(weights(re))
  } else if (a$measure %in% c("RR", "OR")) {
    g <- function(f) vapply(a$studies, function(s) num(s[[f]]), 0)
    ai <- g("events_intervention"); n1i <- g("n_intervention")
    ci <- g("events_control"); n2i <- g("n_control")
    d <- escalc(measure = a$measure, ai = ai, n1i = n1i, ci = ci, n2i = n2i,
                add = 1/2, to = "only0", drop00 = TRUE)
    keep <- !is.na(d$yi)
    ivids <- ids[keep]
    res$iv_common <- safe(summ(rma(yi, vi, data = d, method = "EE", level = level), exp, ivids))
    res$iv_random <- safe(summ(rma(yi, vi, data = d, method = a$tau2_estimator, test = test_re,
                                   level = level, control = ctrl), exp, ivids))
    fe <- safe(rma(yi, vi, data = d, method = "EE"))
    if (is.null(fe$error)) {
      res$heterogeneity <- list(Q = fe$QE, Qp = fe$QEp, I2 = fe$I2, df = fe$k - 1)
      res$iv_common$weights_pct <- as.numeric(weights(fe))
    }
    re <- safe(rma(yi, vi, data = d, method = a$tau2_estimator, control = ctrl))
    if (is.null(re$error)) res$iv_random$weights_pct <- as.numeric(weights(re))
    mh <- safe(rma.mh(measure = a$measure, ai = ai, n1i = n1i, ci = ci, n2i = n2i,
                      drop00 = TRUE, level = level))
    if (is.null(mh$error)) {
      w <- as.numeric(weights(mh)); w[is.na(w)] <- 0
      res$mh_common <- c(summ(mh, exp, ids[w > 0]), list(weights_pct = w))
    } else {
      res$mh_common <- mh
    }
  } else {
    res$not_computed <- paste("measure", a$measure, "is not covered by this reference")
  }
  out$analyses[[length(out$analyses) + 1]] <- res
}
write_json(out, args[2], auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null")
