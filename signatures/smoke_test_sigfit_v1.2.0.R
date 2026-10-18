suppressPackageStartupMessages(library(sigfit))

stopifnot(as.character(packageVersion("sigfit")) == "1.2.0")
data("cosmic_signatures")
stopifnot(identical(dim(cosmic_signatures), c(30L, 96L)))

set.seed(120)
signatures <- matrix(
  c(
    0.70, 0.10,
    0.20, 0.20,
    0.10, 0.70
  ),
  nrow = 2,
  ncol = 3,
  byrow = TRUE,
  dimnames = list(c("S1", "S2"), c("A", "B", "C"))
)
counts <- matrix(
  c(70L, 20L, 10L),
  nrow = 1,
  ncol = 3,
  dimnames = list("sample1", c("A", "B", "C"))
)

# A short MCMC run exercises the exact compiled Stan fitting path without
# running a publication-scale fit.
fit <- fit_signatures(
  counts = counts,
  signatures = signatures,
  model = "nmf",
  chains = 1,
  iter = 40,
  warmup = 20,
  seed = 120,
  refresh = 0
)

exposures <- retrieve_pars(fit, par = "exposures")
stopifnot(all(is.finite(as.matrix(exposures$mean))), ncol(exposures$mean) == 2)

dir.create("sigfit_poulos2018_20260726/verification", recursive = TRUE, showWarnings = FALSE)
writeLines(
  c(
    paste0("sigfit=", packageVersion("sigfit")),
    paste0("R=", R.version.string),
    paste0("rstan=", packageVersion("rstan")),
    paste0("StanHeaders=", packageVersion("StanHeaders")),
    paste0("Rcpp=", packageVersion("Rcpp")),
    paste0("bundled_cosmic_signatures=", paste(dim(cosmic_signatures), collapse = "x")),
    paste0("smoke_exposures=", paste(signif(as.numeric(exposures$mean), 6), collapse = ","))
  ),
  "sigfit_poulos2018_20260726/verification/smoke_test_versions.txt"
)

cat("SIGFIT_V1.2.0_SMOKE_TEST_OK\n")
print(exposures)
