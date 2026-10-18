#!/usr/bin/env Rscript

cran_packages <- c(
  "deconstructSigs",
  "optparse",
  "data.table",
  "writexl"
)

bioc_packages <- c("BSgenome.Hsapiens.UCSC.hg38")

ensure_installed <- function(pkgs, installer) {
  missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) == 0) {
    message("[OK] already installed: ", paste(pkgs, collapse = ", "))
    return(invisible(NULL))
  }

  message("[INSTALL] ", paste(missing, collapse = ", "))
  installer(missing)
}

options(repos = c(CRAN = "https://cloud.r-project.org"))

if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}

ensure_installed(cran_packages, function(pkgs) {
  install.packages(pkgs, dependencies = TRUE)
})

ensure_installed(bioc_packages, function(pkgs) {
  BiocManager::install(pkgs, ask = FALSE, update = FALSE)
})

message("[DONE] R package bootstrap completed.")
