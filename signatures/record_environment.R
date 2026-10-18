packages <- c(
  "sigfit", "rstan", "StanHeaders", "Rcpp", "RcppEigen", "BH",
  "coda", "ggplot2", "clue"
)

suppressPackageStartupMessages(library(sigfit))

versions <- data.frame(
  package = packages,
  version = vapply(packages, function(x) as.character(packageVersion(x)), character(1)),
  stringsAsFactors = FALSE
)

out_dir <- "sigfit_poulos2018_20260726/verification"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
write.table(
  versions,
  file.path(out_dir, "package_versions.tsv"),
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)

session <- capture.output(sessionInfo())
writeLines(session, file.path(out_dir, "session_info.txt"))

cat("SIGFIT_ENVIRONMENT_RECORDED\n")
print(versions, row.names = FALSE)
