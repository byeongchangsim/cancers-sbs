#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(optparse)
  library(data.table)
  library(sigfit)
})

parse_bool <- function(x) {
  value <- toupper(as.character(x[[1]]))
  if (value %in% c("TRUE", "T", "1", "YES", "Y")) return(TRUE)
  if (value %in% c("FALSE", "F", "0", "NO", "N")) return(FALSE)
  stop("Could not parse boolean value: ", value)
}

option_list <- list(
  make_option("--input-matrix", type = "character", help = "Prepared sbs96_counts.tsv"),
  make_option("--output-dir", type = "character", help = "Sigfit output directory"),
  make_option("--dataset", type = "character", help = "TCGA or CPTAC"),
  make_option("--cohort", type = "character", help = "Cancer cohort label"),
  make_option("--seed", type = "integer", default = 2018L, help = "Fixed sampling seed [default %default]"),
  make_option("--chains", type = "integer", default = 4L, help = "rstan default [default %default]"),
  make_option("--iter", type = "integer", default = 2000L, help = "rstan default [default %default]"),
  make_option("--warmup", type = "integer", default = 1000L, help = "rstan default [default %default]"),
  make_option("--thin", type = "integer", default = 1L, help = "rstan default [default %default]"),
  make_option("--cores", type = "integer", default = 4L, help = "Parallel chain count [default %default]"),
  make_option("--save-fit-rds", type = "character", default = "FALSE")
)
opt <- parse_args(OptionParser(option_list = option_list))

required <- c("input_matrix", "output_dir", "dataset", "cohort")
missing <- required[vapply(required, function(name) {
  is.null(opt[[name]]) || is.na(opt[[name]]) || !nzchar(opt[[name]])
}, logical(1))]
if (length(missing) > 0) {
  stop("Missing required arguments: ", paste(missing, collapse = ", "))
}
if (!file.exists(opt$input_matrix)) {
  stop("Missing input matrix: ", opt$input_matrix)
}
if (as.character(packageVersion("sigfit")) != "1.2.0") {
  stop(
    "Poulos reproduction requires sigfit 1.2.0; installed version is ",
    as.character(packageVersion("sigfit"))
  )
}
if (opt$warmup >= opt$iter) {
  stop("--warmup must be smaller than --iter")
}

input <- fread(opt$input_matrix, sep = "\t", data.table = FALSE, check.names = FALSE)
metadata_columns <- c("Sample", "PATIENT", "snv_count")
context_columns <- setdiff(names(input), metadata_columns)
if (!identical(names(input)[seq_along(metadata_columns)], metadata_columns)) {
  stop("Input must begin with columns: ", paste(metadata_columns, collapse = ", "))
}
if (length(context_columns) != 96L || anyDuplicated(context_columns)) {
  stop("Input must contain exactly 96 unique context columns and no extra fields")
}
if (anyDuplicated(input$Sample) || anyDuplicated(input$PATIENT)) {
  stop("Input must contain exactly one unique sample per patient")
}

counts <- as.matrix(input[, context_columns, drop = FALSE])
storage.mode(counts) <- "integer"
rownames(counts) <- input$Sample
if (any(counts < 0L) || !identical(as.integer(rowSums(counts)), as.integer(input$snv_count))) {
  stop("Invalid SBS96 counts or row sums")
}
if (any(rowSums(counts) < 30L)) {
  stop("Input contains an ineligible sample with fewer than 30 mutations")
}

data("cosmic_signatures", package = "sigfit")
if (!identical(sort(colnames(cosmic_signatures)), sort(context_columns))) {
  missing_contexts <- setdiff(colnames(cosmic_signatures), context_columns)
  extra_contexts <- setdiff(context_columns, colnames(cosmic_signatures))
  stop(
    "Context labels do not match sigfit COSMIC v2. Missing: ",
    paste(missing_contexts, collapse = ", "),
    "; extra: ", paste(extra_contexts, collapse = ", ")
  )
}
counts <- counts[, colnames(cosmic_signatures), drop = FALSE]

# Poulos et al. fitted exome-normalised COSMIC v2 signatures. In sigfit 1.2.0,
# this is the documented two-step conversion from genome-relative NMF values.
normalised_signatures <- convert_signatures(
  cosmic_signatures,
  ref_opportunities = "human-genome",
  model_to = "emu"
)
exome_signatures <- convert_signatures(
  normalised_signatures,
  ref_opportunities = "human-exome",
  model_to = "nmf"
)
rownames(exome_signatures) <- paste("Signature", seq_len(30L))
colnames(exome_signatures) <- colnames(cosmic_signatures)
if (max(abs(rowSums(exome_signatures) - 1)) > 1e-12) {
  stop("Exome-normalised signature rows do not sum to one")
}

options(mc.cores = min(opt$cores, opt$chains))
rstan::rstan_options(auto_write = TRUE)
message(
  "[INFO] fitting ", nrow(counts), " samples; chains=", opt$chains,
  ", iter=", opt$iter, ", warmup=", opt$warmup, ", seed=", opt$seed
)
fit <- fit_signatures(
  counts = counts,
  signatures = exome_signatures,
  exp_prior = rep(1, 30L),
  model = "nmf",
  chains = opt$chains,
  iter = opt$iter,
  warmup = opt$warmup,
  thin = opt$thin,
  seed = opt$seed
)
exposure_summary <- retrieve_pars(fit, par = "exposures", hpd_prob = 0.95)
means <- as.matrix(exposure_summary[[1]])
lowers <- as.matrix(exposure_summary[[2]])
uppers <- as.matrix(exposure_summary[[3]])

expected_samples <- rownames(counts)
if (!identical(rownames(means), expected_samples)) {
  stop("Sigfit exposure rows do not preserve the input sample order")
}
if (ncol(means) != 30L || any(!is.finite(means)) || any(means < 0) || any(means > 1)) {
  stop("Invalid posterior mean exposure matrix")
}
if (max(abs(rowSums(means) - 1)) > 1e-8) {
  stop("Posterior mean exposures do not sum to one")
}

sbs_names <- paste0("SBS", seq_len(30L))
colnames(means) <- sbs_names
colnames(lowers) <- sbs_names
colnames(uppers) <- sbs_names
sample_metadata <- input[, metadata_columns, drop = FALSE]

sample_means <- cbind(sample_metadata, as.data.frame(means, check.names = FALSE))
patient_means <- cbind(
  data.frame(PATIENT = sample_metadata$PATIENT, stringsAsFactors = FALSE),
  as.data.frame(means, check.names = FALSE)
)
sample_lowers <- cbind(sample_metadata, as.data.frame(lowers, check.names = FALSE))
sample_uppers <- cbind(sample_metadata, as.data.frame(uppers, check.names = FALSE))

stan_summary <- rstan::summary(fit$result, pars = "exposures")$summary
diagnostics <- data.table(
  parameter = rownames(stan_summary),
  mean = stan_summary[, "mean"],
  se_mean = stan_summary[, "se_mean"],
  sd = stan_summary[, "sd"],
  n_eff = stan_summary[, "n_eff"],
  Rhat = stan_summary[, "Rhat"]
)

dir.create(opt$output_dir, recursive = TRUE, showWarnings = FALSE)
fwrite(sample_means, file.path(opt$output_dir, "sample_weights.tsv"), sep = "\t")
fwrite(patient_means, file.path(opt$output_dir, "patient_weights.tsv"), sep = "\t")
fwrite(sample_lowers, file.path(opt$output_dir, "sample_weights_hpd95_lower.tsv"), sep = "\t")
fwrite(sample_uppers, file.path(opt$output_dir, "sample_weights_hpd95_upper.tsv"), sep = "\t")
fwrite(diagnostics, file.path(opt$output_dir, "stan_exposure_diagnostics.tsv"), sep = "\t")
fwrite(
  data.table(
    dataset = toupper(opt$dataset),
    cohort = opt$cohort,
    sample_n = nrow(counts),
    signature_n = ncol(means),
    sigfit_version = as.character(packageVersion("sigfit")),
    model = "nmf",
    exposure_prior = "Dirichlet(1 x 30)",
    signature_reference = "sigfit 1.2.0 cosmic_signatures, genome-to-exome converted",
    chains = opt$chains,
    iter = opt$iter,
    warmup = opt$warmup,
    thin = opt$thin,
    seed = opt$seed,
    seed_note = "Fixed prospectively for reproducibility; Poulos et al. did not report a sampling seed",
    max_Rhat = max(diagnostics$Rhat, na.rm = TRUE),
    min_n_eff = min(diagnostics$n_eff, na.rm = TRUE)
  ),
  file.path(opt$output_dir, "fit_metadata.tsv"),
  sep = "\t"
)
fwrite(
  data.table(Signature = rownames(exome_signatures), as.data.frame(exome_signatures, check.names = FALSE)),
  file.path(opt$output_dir, "cosmic_v2_exome_normalised_reference.tsv"),
  sep = "\t"
)

if (parse_bool(opt$save_fit_rds)) {
  saveRDS(fit, file.path(opt$output_dir, "fit.rds"), compress = FALSE)
}
message("[OK] wrote Sigfit SBS1-SBS30 posterior mean weights to ", opt$output_dir)

