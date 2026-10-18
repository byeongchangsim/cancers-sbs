#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(optparse)
  library(data.table)
  library(writexl)
  library(deconstructSigs)
  library(pheatmap)
})

normalize_patient_id <- function(x) {
  normalize_patient_id_mode(x, "cptac")
}

normalize_patient_id_mode <- function(x, mode) {
  out <- as.character(x)
  if (mode == "tcga") {
    out <- substr(out, 1, 12)
    return(out)
  }
  if (mode == "identity") {
    return(out)
  }
  if (mode == "cptac") {
    out <- sub("_(T|N)$", "", out)
    return(out)
  }
  stop("Unsupported patient mode: ", mode)
}

normalize_seqnames <- function(x) {
  out <- as.character(x)
  out <- sub("^chr", "", out, ignore.case = TRUE)
  out[toupper(out) == "MT"] <- "M"
  paste0("chr", out)
}

normalize_sample_ids <- function(x, mode) {
  out <- as.character(x)
  if (mode == "tcga") {
    out <- sub("-[0-9A-Z]{2}$", "", out)
  }
  out
}

load_ref_genome <- function(pkg_name) {
  suppressPackageStartupMessages(library(pkg_name, character.only = TRUE))
  get(pkg_name, envir = asNamespace(pkg_name))
}

load_signature_reference <- function(cosmic_file) {
  load_signature_reference_with_name(cosmic_file, "legacy30")
}

load_signature_reference_with_name <- function(cosmic_file, ref_name) {
  if (!is.na(cosmic_file) && nzchar(cosmic_file)) {
    cosmic_signatures <- read.table(
      cosmic_file,
      header = TRUE,
      sep = "\t",
      row.names = 1,
      check.names = FALSE
    )
    return(as.data.frame(t(cosmic_signatures), check.names = FALSE))
  }

  if (identical(ref_name, "legacy30")) {
    return(as.data.frame(signatures.cosmic, check.names = FALSE))
  }
  if (identical(ref_name, "nature2013")) {
    return(as.data.frame(signatures.nature2013, check.names = FALSE))
  }
  if (identical(ref_name, "genome67")) {
    return(as.data.frame(signatures.genome.cosmic.v3.may2019, check.names = FALSE))
  }
  if (identical(ref_name, "exome65")) {
    return(as.data.frame(signatures.exome.cosmic.v3.may2019, check.names = FALSE))
  }
  stop("Unsupported signatures reference name: ", ref_name)
}

parse_bool <- function(x) {
  if (is.logical(x)) {
    return(x[[1]])
  }
  x <- toupper(as.character(x[[1]]))
  if (x %in% c("TRUE", "T", "1", "YES", "Y")) {
    return(TRUE)
  }
  if (x %in% c("FALSE", "F", "0", "NO", "N")) {
    return(FALSE)
  }
  stop("Could not parse boolean value: ", x)
}

parse_associated_signatures <- function(x) {
  if (is.null(x) || is.na(x) || !nzchar(x)) {
    return(NULL)
  }
  vals <- trimws(unlist(strsplit(as.character(x), ",")))
  vals <- vals[nzchar(vals)]
  if (length(vals) == 0) {
    return(NULL)
  }
  vals
}

option_list <- list(
  make_option("--input-maf", type = "character", dest = "input_maf", help = "Input mutation table"),
  make_option("--output-xlsx", type = "character", dest = "output_xlsx", help = "Output PATIENT + SBS excel path"),
  make_option("--output-csv", type = "character", dest = "output_csv", help = "Output PATIENT + SBS csv path"),
  make_option("--output-counts", type = "character", dest = "output_counts", default = NA_character_,
              help = "Optional per-sample mutation count csv"),
  make_option("--output-sample-weights", type = "character", dest = "output_sample_weights", default = NA_character_,
              help = "Optional pre-aggregation per-sample signature weight csv"),
  make_option("--output-selected-samples", type = "character", dest = "output_selected_samples", default = NA_character_,
              help = "Optional manifest of samples retained for signature fitting"),
  make_option("--output-sbs96", type = "character", dest = "output_sbs96", default = NA_character_,
              help = "Optional SigProfiler-compatible 96-channel matrix TSV"),
  make_option("--catalogue-only", action = "store_true", dest = "catalogue_only", default = FALSE,
              help = "Build and export SBS96 catalogue without running deconstructSigs"),
  make_option("--plots-pdf", type = "character", dest = "plots_pdf", default = NA_character_,
              help = "Optional PDF for per-sample signature plots"),
  make_option("--heatmap-pdf", type = "character", dest = "heatmap_pdf", default = NA_character_,
              help = "Optional PDF for SBS abundance heatmap"),
  make_option("--cosmic-file", type = "character", dest = "cosmic_file", default = NA_character_,
              help = "Optional COSMIC signature table, e.g. COSMIC_v3.4_SBS_GRCh37/38.txt"),
  make_option("--signatures-ref-name", type = "character", dest = "signatures_ref_name", default = "legacy30",
              help = "Built-in signatures reference to use when cosmic-file is not given: nature2013, legacy30, genome67, exome65 [default %default]"),
  make_option("--tri-counts-method", type = "character", dest = "tri_counts_method", default = "default",
              help = "deconstructSigs tri.counts.method [default %default]"),
  make_option("--associated-signatures", type = "character", dest = "associated_signatures", default = "",
              help = "Comma-separated signatures to pass as associated"),
  make_option("--min-mutations", type = "integer", dest = "min_mutations", default = 0,
              help = "Filter samples with rowSums(sigs_input) <= this threshold [default %default]"),
  make_option("--one-sample-per-patient", type = "character", dest = "one_sample_per_patient", default = "FALSE",
              help = "Randomly retain one eligible sample per patient using a fixed seed [default %default]"),
  make_option("--sample-selection-seed", type = "integer", dest = "sample_selection_seed", default = 2018,
              help = "Seed for one-sample-per-patient selection [default %default]"),
  make_option("--signature-cutoff", type = "double", dest = "signature_cutoff", default = 0.06,
              help = "deconstructSigs signature.cutoff [default %default]"),
  make_option("--signatures-limit", type = "integer", dest = "signatures_limit", default = NA_integer_,
              help = "Optional deconstructSigs signatures.limit"),
  make_option("--contexts-needed", type = "character", dest = "contexts_needed", default = "TRUE",
              help = "Pass TRUE or FALSE to whichSignatures contexts.needed [default %default]"),
  make_option("--sample-col", type = "character", dest = "sample_col", default = "Tumor_Sample_Barcode",
              help = "Sample column name [default %default]"),
  make_option("--chr-col", type = "character", dest = "chr_col", default = "Chromosome",
              help = "Chromosome column name [default %default]"),
  make_option("--pos-col", type = "character", dest = "pos_col", default = "Start_Position",
              help = "Position column name [default %default]"),
  make_option("--ref-col", type = "character", dest = "ref_col", default = "Reference_Allele",
              help = "Reference allele column name [default %default]"),
  make_option("--alt-col", type = "character", dest = "alt_col", default = "Tumor_Seq_Allele2",
              help = "Alternate allele column name [default %default]"),
  make_option("--variant-type-col", type = "character", dest = "variant_type_col", default = "Variant_Type",
              help = "Variant type column name [default %default]"),
  make_option("--patient-mode", type = "character", dest = "patient_mode", default = "cptac",
              help = "How to normalize sample IDs into PATIENT IDs: cptac, tcga, identity [default %default]"),
  make_option("--ref-genome-pkg", type = "character", dest = "ref_genome_pkg",
              default = "BSgenome.Hsapiens.UCSC.hg38",
              help = "BSgenome package name [default %default]")
)

opt <- parse_args(OptionParser(option_list = option_list))

required_args <- if (isTRUE(opt$catalogue_only)) c("input_maf", "output_sbs96") else c("input_maf", "output_xlsx", "output_csv")
missing_args <- required_args[vapply(required_args, function(x) is.null(opt[[x]]) || is.na(opt[[x]]), logical(1))]
if (length(missing_args) > 0) {
  stop("Missing required arguments: ", paste(missing_args, collapse = ", "))
}

mut <- fread(opt$input_maf, sep = "\t", data.table = TRUE)

required_cols <- c(opt$sample_col, opt$chr_col, opt$pos_col, opt$ref_col, opt$alt_col, opt$variant_type_col)
missing_cols <- setdiff(required_cols, names(mut))
if (length(missing_cols) > 0) {
  stop("Missing required input columns: ", paste(missing_cols, collapse = ", "))
}

ref_genome <- load_ref_genome(opt$ref_genome_pkg)
max_chr_sizes <- data.table(
  Chromosome = as.character(seqnames(ref_genome)),
  Max_Position = as.numeric(seqlengths(ref_genome))
)

mut <- copy(mut)
setnames(mut, old = c(opt$sample_col, opt$chr_col, opt$pos_col, opt$ref_col, opt$alt_col, opt$variant_type_col),
         new = c("Sample", "Chromosome", "Start_Position", "Reference_Allele", "Tumor_Seq_Allele2", "Variant_Type"))

mut[, Chromosome := normalize_seqnames(Chromosome)]
mut[, Start_Position := as.numeric(Start_Position)]
mut[, Reference_Allele := toupper(as.character(Reference_Allele))]
mut[, Tumor_Seq_Allele2 := toupper(as.character(Tumor_Seq_Allele2))]
mut[, Sample := normalize_sample_ids(Sample, opt$patient_mode)]

mut <- mut[
  Variant_Type == "SNP" &
    !is.na(Start_Position) &
    nchar(Reference_Allele) == 1 &
    nchar(Tumor_Seq_Allele2) == 1 &
    Reference_Allele %in% c("A", "C", "G", "T") &
    Tumor_Seq_Allele2 %in% c("A", "C", "G", "T") &
    Reference_Allele != Tumor_Seq_Allele2
]

mut <- merge(mut, max_chr_sizes, by = "Chromosome", all.x = FALSE, all.y = FALSE)
mut <- mut[Start_Position <= Max_Position]

if (nrow(mut) == 0) {
  stop("No usable SNP rows remain after filtering.")
}

sigs_input <- mut.to.sigs.input(
  mut.ref = as.data.frame(mut),
  sample.id = "Sample",
  chr = "Chromosome",
  pos = "Start_Position",
  ref = "Reference_Allele",
  alt = "Tumor_Seq_Allele2",
  bsg = ref_genome
)

if (!is.na(opt$min_mutations) && opt$min_mutations > 0) {
  sigs_input <- sigs_input[rowSums(sigs_input) > opt$min_mutations, , drop = FALSE]
}

if (nrow(sigs_input) == 0) {
  stop("No samples remain after the minimum mutation filter.")
}

one_sample_per_patient <- parse_bool(opt$one_sample_per_patient)
eligible_samples <- data.table(
  Sample = rownames(sigs_input),
  PATIENT = normalize_patient_id_mode(rownames(sigs_input), opt$patient_mode),
  snv_count = as.integer(rowSums(sigs_input))
)
setorder(eligible_samples, PATIENT, Sample)
eligible_samples[, selected := TRUE]

if (one_sample_per_patient) {
  set.seed(opt$sample_selection_seed)
  selected_samples <- eligible_samples[, .SD[sample.int(.N, 1L)], by = PATIENT]$Sample
  eligible_samples[, selected := Sample %in% selected_samples]
  sigs_input <- sigs_input[selected_samples, , drop = FALSE]
}

if (!is.na(opt$output_selected_samples)) {
  dir.create(dirname(opt$output_selected_samples), recursive = TRUE, showWarnings = FALSE)
  fwrite(eligible_samples, opt$output_selected_samples)
}

if (!is.na(opt$output_sbs96)) {
  dir.create(dirname(opt$output_sbs96), recursive = TRUE, showWarnings = FALSE)
  sbs96_output <- data.frame(
    MutationType = colnames(sigs_input),
    t(sigs_input),
    row.names = NULL,
    check.names = FALSE
  )
  fwrite(sbs96_output, opt$output_sbs96, sep = "\t")
}

if (isTRUE(opt$catalogue_only)) {
  message("[OK] SigProfiler-compatible SBS96 catalogue: ", opt$output_sbs96)
  message("[OK] samples exported: ", nrow(sigs_input))
  quit(save = "no", status = 0L)
}

message("[INFO] usable SNP rows: ", nrow(mut))
message("[INFO] sigs_input samples: ", nrow(sigs_input))
message("[INFO] one sample per patient: ", one_sample_per_patient)
if (one_sample_per_patient) {
  message("[INFO] sample selection seed: ", opt$sample_selection_seed)
}

signatures_ref <- load_signature_reference_with_name(opt$cosmic_file, opt$signatures_ref_name)
message("[INFO] signature reference rows: ", nrow(signatures_ref))

associated_sigs <- parse_associated_signatures(opt$associated_signatures)
contexts_needed <- parse_bool(opt$contexts_needed)

results <- vector("list", nrow(sigs_input))
names(results) <- rownames(sigs_input)

if (!is.na(opt$plots_pdf)) {
  dir.create(dirname(opt$plots_pdf), recursive = TRUE, showWarnings = FALSE)
  pdf(opt$plots_pdf)
  on.exit(dev.off(), add = TRUE)
}

sample_ids <- rownames(sigs_input)
for (idx in seq_along(sample_ids)) {
  sample_id <- sample_ids[[idx]]
  if (idx == 1 || idx %% 50 == 0 || idx == length(sample_ids)) {
    message("[INFO] fitting sample ", idx, "/", length(sample_ids), ": ", sample_id)
  }
  fit <- whichSignatures(
    tumor.ref = sigs_input,
    signatures.ref = signatures_ref,
    sample.id = sample_id,
    associated = associated_sigs,
    signatures.limit = opt$signatures_limit,
    signature.cutoff = opt$signature_cutoff,
    contexts.needed = contexts_needed,
    tri.counts.method = opt$tri_counts_method
  )

  results[[sample_id]] <- fit
  if (!is.na(opt$plots_pdf)) {
    plotSignatures(fit)
  }
}

weights_list <- lapply(results, function(res) {
  if (!is.null(res$weights)) {
    res$weights
  } else {
    NA
  }
})

weights_df <- do.call(rbind, weights_list)
weights_df <- data.frame(Sample = names(weights_list), weights_df, row.names = NULL, check.names = FALSE)
weights_df$PATIENT <- normalize_patient_id_mode(weights_df$Sample, opt$patient_mode)

if (!is.na(opt$output_sample_weights)) {
  dir.create(dirname(opt$output_sample_weights), recursive = TRUE, showWarnings = FALSE)
  fwrite(weights_df, opt$output_sample_weights)
}

sbs_cols <- setdiff(names(weights_df), c("Sample", "PATIENT"))
patient_df <- aggregate(weights_df[, sbs_cols, drop = FALSE], by = list(PATIENT = weights_df$PATIENT), FUN = mean)
patient_df <- patient_df[order(patient_df$PATIENT), , drop = FALSE]

sample_counts <- mut[, .(snv_count = .N), by = Sample][order(Sample)]
sample_counts[, PATIENT := normalize_patient_id_mode(Sample, opt$patient_mode)]

dir.create(dirname(opt$output_xlsx), recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(opt$output_csv), recursive = TRUE, showWarnings = FALSE)

write_xlsx(patient_df, opt$output_xlsx)
fwrite(patient_df, opt$output_csv)

if (!is.na(opt$output_counts)) {
  dir.create(dirname(opt$output_counts), recursive = TRUE, showWarnings = FALSE)
  fwrite(sample_counts, opt$output_counts)
}

if (!is.na(opt$heatmap_pdf)) {
  dir.create(dirname(opt$heatmap_pdf), recursive = TRUE, showWarnings = FALSE)
  pdf(opt$heatmap_pdf, width = 10, height = 8)
  pheatmap(
    as.matrix(patient_df[, sbs_cols, drop = FALSE]),
    cluster_rows = FALSE,
    cluster_cols = TRUE,
    scale = "row",
    show_rownames = FALSE,
    use_raster = TRUE,
    fontsize_col = 5,
    fontsize_row = 10,
    main = "Heatmap of SBS Signature Abundances",
    color = colorRampPalette(c("blue", "white", "red"))(255),
    border_color = "black"
  )
  dev.off()
}

message("[OK] samples fit: ", nrow(weights_df))
message("[OK] patients exported: ", nrow(patient_df))
message("[OK] SBS columns: ", length(sbs_cols))
message("[OUT] xlsx: ", opt$output_xlsx)
message("[OUT] csv:  ", opt$output_csv)
