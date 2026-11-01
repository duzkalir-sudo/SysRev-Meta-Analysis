#!/usr/bin/env Rscript
# Independent reference values for src/stats_engine, computed with R metafor (+ lme4).
#
# Usage: Rscript validation/r/metafor_reference.R <synthetic_cases.json> <output.json>
#
# Real datasets come from the 'metadat' package (installed with metafor):
#   dat.pritz1997    Pritz (1997), hyperdynamic therapy for cerebral vasospasm (proportions)
#   dat.bcg          Colditz et al. (1994), BCG vaccine trials (2x2 tables)
#   dat.hine1989     Hine et al. (1989), prophylactic lidocaine; sparse mortality (2x2)
#   dat.normand1999  Normand (1999), length of hospital stay for stroke (means/SDs)
# Synthetic edge cases are read from synthetic_cases.json (invented; not real studies).
suppressPackageStartupMessages({
  library(metafor)
  library(jsonlite)
})
args <- commandArgs(trailingOnly = TRUE)
synthetic <- fromJSON(args[1], simplifyVector = TRUE)
out_path <- args[2]

ctrl <- list(threshold = 1e-12, maxiter = 10000, stepadj = 0.5)

pooled <- function(m) {
  list(estimate = unname(c(m$beta)), se = m$se, ci_lower = m$ci.lb, ci_upper = m$ci.ub,
       statistic = m$zval, p_value = m$pval, tau2 = m$tau2, I2 = m$I2, H2 = m$H2,
       QE = m$QE, QEp = m$QEp)
}

iv_block <- function(yi, vi) {
  fe <- rma(yi, vi, method = "EE")
  res <- list(yi = as.numeric(yi), vi = as.numeric(vi), common = pooled(fe),
              Q = fe$QE, Qp = fe$QEp, I2_q = fe$I2, H2_q = fe$H2)
  for (tm in c("REML", "DL")) {
    for (test in c("z", "knha", "adhoc")) {
      m <- rma(yi, vi, method = tm, test = test, control = ctrl)
      key <- paste(tm, test, sep = "_")
      blk <- pooled(m)
      if (m$k >= 3) {
        p <- predict(m, pi.type = "Riley")
        blk$pi_lower <- p$pi.lb
        blk$pi_upper <- p$pi.ub
      }
      if (test == "z") {
        # tau2.max widened: the default search ceiling (100) truncates wide upper bounds
        ci <- confint(m, type = "QP", control = list(tol = 1e-12, tau2.max = 1e6))$random
        blk$tau2_ci <- unname(ci["tau^2", c("ci.lb", "ci.ub")])
        blk$I2_ci <- unname(ci["I^2(%)", c("ci.lb", "ci.ub")])
      }
      res[[key]] <- blk
    }
  }
  res
}

cases <- list()

# ---- single proportions
prop_case <- function(id, xi, ni, source) {
  es <- escalc(measure = "PLO", xi = xi, ni = ni)  # add = 1/2, to = "only0"
  blk <- iv_block(es$yi, es$vi)
  g <- rma.glmm(measure = "PLO", xi = xi, ni = ni)  # lme4::glmer, nAGQ = 7, ML
  blk$glmm <- list(estimate = unname(c(g$beta)), se = g$se, ci_lower = g$ci.lb,
                   ci_upper = g$ci.ub, statistic = g$zval, p_value = g$pval,
                   tau2 = g$tau2, I2 = g$I2)
  gp <- predict(g, pi.type = "Riley")
  blk$glmm$pi_lower <- gp$pi.lb
  blk$glmm$pi_upper <- gp$pi.ub
  ge <- rma.glmm(measure = "PLO", xi = xi, ni = ni, method = "EE")
  blk$glmm_common <- list(estimate = unname(c(ge$beta)), se = ge$se,
                          ci_lower = ge$ci.lb, ci_upper = ge$ci.ub)
  cp <- t(mapply(function(x, n) binom.test(x, n)$conf.int, xi, ni))
  blk$clopper_pearson <- unname(cp)
  list(id = id, kind = "proportion", source = source,
       data = list(xi = xi, ni = ni), expected = blk)
}
cases[[length(cases) + 1]] <- prop_case("pritz1997", dat.pritz1997$xi, dat.pritz1997$ni,
                                        "metadat::dat.pritz1997")
for (nm in names(synthetic$proportions)) {
  d <- synthetic$proportions[[nm]]
  cases[[length(cases) + 1]] <- prop_case(nm, d$xi, d$ni, "synthetic")
}

# ---- 2x2 tables (RR, OR)
tbl_case <- function(id, ai, n1i, ci, n2i, source) {
  out <- list(id = id, kind = "two_by_two", source = source,
              data = list(ai = ai, n1i = n1i, ci = ci, n2i = n2i), expected = list())
  for (measure in c("RR", "OR")) {
    es <- escalc(measure = measure, ai = ai, n1i = n1i, ci = ci, n2i = n2i, drop00 = TRUE)
    keep <- !is.na(es$yi)
    blk <- iv_block(es$yi[keep], es$vi[keep])
    blk$kept <- which(keep) - 1  # zero-based indices
    mh <- rma.mh(measure = measure, ai = ai, n1i = n1i, ci = ci, n2i = n2i, drop00 = TRUE)
    # Mantel-Haenszel study weights (weights.rma.mh), renormalised over the studies the
    # Python engine keeps (escalc drop00); see VALIDATION_REPORT.md for the difference in
    # double-all-event RR studies (Session 9, M3).
    w <- unname(weights(mh))[keep]
    blk$mh <- list(estimate = unname(c(mh$beta)), se = mh$se, ci_lower = mh$ci.lb,
                   ci_upper = mh$ci.ub, statistic = mh$zval, p_value = mh$pval,
                   weights = as.list(100 * w / sum(w)))
    out$expected[[measure]] <- blk
  }
  out
}
cases[[length(cases) + 1]] <- with(dat.bcg, tbl_case("bcg", tpos, tpos + tneg, cpos, cpos + cneg,
                                                     "metadat::dat.bcg"))
cases[[length(cases) + 1]] <- with(dat.hine1989, tbl_case("hine1989", ai, n1i, ci, n2i,
                                                          "metadat::dat.hine1989"))
for (nm in names(synthetic$two_by_two)) {
  d <- synthetic$two_by_two[[nm]]
  cases[[length(cases) + 1]] <- tbl_case(nm, d$ai, d$n1i, d$ci, d$n2i, "synthetic")
}

# ---- continuous (MD, SMD)
cont <- dat.normand1999
out <- list(id = "normand1999", kind = "continuous", source = "metadat::dat.normand1999",
            data = list(m1i = cont$m1i, sd1i = cont$sd1i, n1i = cont$n1i,
                        m2i = cont$m2i, sd2i = cont$sd2i, n2i = cont$n2i),
            expected = list())
for (measure in c("MD", "SMD")) {
  es <- escalc(measure = measure, m1i = m1i, sd1i = sd1i, n1i = n1i,
               m2i = m2i, sd2i = sd2i, n2i = n2i, data = cont)
  out$expected[[measure]] <- iv_block(es$yi, es$vi)
}
cases[[length(cases) + 1]] <- out

# ---- KM probabilities with CIs (independent R implementation of the SE derivation)
for (nm in names(synthetic$km_probabilities)) {
  if (startsWith(nm, "_")) next
  d <- synthetic$km_probabilities[[nm]]
  g <- function(p) log(-log(p))
  z <- qnorm(1 - (1 - d$level / 100) / 2)
  yi <- g(d$est / 100)
  se <- abs(g(d$upper / 100) - g(d$lower / 100)) / (2 * z)
  cases[[length(cases) + 1]] <- list(id = nm, kind = "km_loglog", source = "synthetic",
                                     data = d, expected = iv_block(yi, se^2))
}

# ---- generic yi/vi edge cases
for (nm in names(synthetic$generic)) {
  d <- synthetic$generic[[nm]]
  cases[[length(cases) + 1]] <- list(id = nm, kind = "generic", source = "synthetic",
                                     data = d, expected = iv_block(d$yi, d$vi))
}

meta <- list(
  R = R.version.string,
  metafor = as.character(packageVersion("metafor")),
  metadat = as.character(packageVersion("metadat")),
  lme4 = as.character(packageVersion("lme4")),
  jsonlite = as.character(packageVersion("jsonlite")),
  rma_control = ctrl,
  notes = paste(
    "escalc defaults (add = 1/2, to = 'only0'); drop00 = TRUE for RR/OR;",
    "rma.glmm(measure = 'PLO') uses lme4::glmer with nAGQ = 7 (ML);",
    "prediction intervals from predict(pi.type = 'Riley'), i.e. t(k-2);",
    "tau^2 CIs from confint(type = 'QP', tau2.max = 1e6)."
  )
)
write(toJSON(list(meta = meta, cases = cases), auto_unbox = TRUE, digits = NA, null = "null",
             na = "null", pretty = TRUE),
      out_path)
