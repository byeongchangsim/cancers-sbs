#!/usr/bin/env Rscript

"%||%" <- function(x, y) {
  if (is.null(x) || length(x) == 0 || is.na(x[[1]]) || !nzchar(x[[1]])) y else x
}

suppressPackageStartupMessages({
  library(optparse)
  library(data.table)
  library(deconstructSigs)
})

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

normalize_patient_ids <- function(x, mode) {
  out <- as.character(x)
  if (mode == "tcga") {
    return(substr(out, 1, 12))
  }
  if (mode == "cptac") {
    return(sub("_(T|N)$", "", out))
  }
  if (mode == "identity") {
    return(out)
  }
  stop("Unsupported patient mode: ", mode)
}

load_ref_genome <- function(pkg_name) {
  suppressPackageStartupMessages(library(pkg_name, character.only = TRUE))
  get(pkg_name, envir = asNamespace(pkg_name))
}

context_to_sigfit <- function(context) {
  parsed <- strcapture(
    "^([ACGT])\\[([CT])>([ACGT])\\]([ACGT])$",
    context,
    proto = list(left = character(), ref = character(), alt = character(), right = character())
  )
  if (anyNA(parsed) || any(parsed$ref == parsed$alt)) {
    stop("Unexpected deconstructSigs context label(s): ", paste(context[!complete.cases(parsed)], collapse = ", "))
  }
  paste0(parsed$left, parsed$ref, parsed$right, ">", parsed$left, parsed$alt, parsed$right)
}

assert_identical_manifest <- function(regenerated, frozen, frozen_path) {
  required <- c("Sample", "PATIENT", "snv_count", "selected")
  missing <- setdiff(required, names(frozen))
  if (length(missing) > 0) {
    stop("Frozen selection manifest is missing columns: ", paste(missing, collapse = ", "))
  }

  frozen <- frozen[, ..required]
  frozen[, `:=`(
    Sample = as.character(Sample),
    PATIENT = as.character(PATIENT),
    snv_count = as.integer(snv_count),
    selected = as.logical(selected)
  )]
  setorder(frozen, PATIENT, Sample)
  setorder(regenerated, PATIENT, Sample)

  if (!identical(regenerated, frozen)) {
    joined <- merge(
      regenerated,
      frozen,
      by = c("Sample", "PATIENT"),
      all = TRUE,
      suffixes = c("_regenerated", "_frozen")
    )
    mismatched <- joined[
      is.na(snv_count_regenerated) |
        is.na(snv_count_frozen) |
        snv_count_regenerated != snv_count_frozen |
        selected_regenerated != selected_frozen
    ]
    preview <- paste(capture.output(print(head(mismatched, 20))), collapse = "\n")
    stop(
      "Regenerated eligibility/selection does not match frozen manifest ", frozen_path,
      ". Refusing to create Sigfit inputs.\n", preview
    )
  }
  invisible(TRUE)
}

option_list <- list(
  make_option("--dataset", type = "character", help = "TCGA or CPTAC"),
  make_option("--cohort", type = "character", help = "Brca, Ccrcc, Hnscc, Lscc, Luad, Ov, or Ucec"),
  make_option(
    "--root",
    type = "character",
    default = "/home/marugoto-1.0.0-regression",
    help = "Repository root [default %default]"
  ),
  make_option("--input-maf", type = "character", default = NA_character_),
  make_option("--frozen-selection", type = "character", default = NA_character_),
  make_option("--output-dir", type = "character", default = NA_character_)
)
opt <- parse_args(OptionParser(option_list = option_list))

if (is.null(opt$dataset) || is.null(opt$cohort)) {
  stop("--dataset and --cohort are required")
}

dataset <- toupper(opt$dataset)
cohort <- opt$cohort
valid_cohorts <- c("Brca", "Ccrcc", "Hnscc", "Lscc", "Luad", "Ov", "Ucec")
tcga_codes <- c(Brca = "BRCA", Ccrcc = "KIRC", Hnscc = "HNSC", Lscc = "LUSC", Luad = "LUAD", Ov = "OV", Ucec = "UCEC")
cptac_stems <- c(Brca = "BRCA", Ccrcc = "CCRCC", Hnscc = "HNSCC", Lscc = "LSCC", Luad = "LUAD", Ov = "OV", Ucec = "UCEC")

if (!dataset %in% c("TCGA", "CPTAC")) {
  stop("--dataset must be TCGA or CPTAC")
}
if (!cohort %in% valid_cohorts) {
  stop("Unsupported cohort: ", cohort)
}

root <- normalizePath(opt$root, mustWork = TRUE)
if (dataset == "TCGA") {
  default_maf <- file.path(
    root,
    "tcga_sigprof_v34_mc3_rerun",
    "input_maf",
    cohort,
    paste0("TCGA-", tcga_codes[[cohort]], "_mc3_public.maf")
  )
  patient_mode <- "tcga"
  ref_genome_pkg <- "BSgenome.Hsapiens.UCSC.hg19"
} else {
  default_maf <- file.path(
    "/home/marugoto-1.0.0-classification",
    paste0(cptac_stems[[cohort]], "_somatic_mutation.maf")
  )
  patient_mode <- "cptac"
  ref_genome_pkg <- "BSgenome.Hsapiens.UCSC.hg38"
}

source_dir <- file.path(root, "poulos2018_cosmic30_20260726", dataset, cohort)
prefix <- paste(dataset, cohort, "poulos2018_cosmic30", sep = "_")
default_frozen <- file.path(source_dir, paste0(prefix, "_selected_samples.csv"))
default_output <- file.path(root, "sigfit_poulos2018_20260726", "inputs", dataset, cohort)

input_maf <- opt$input_maf %||% default_maf
frozen_selection <- opt$frozen_selection %||% default_frozen
output_dir <- opt$output_dir %||% default_output

if (!file.exists(input_maf)) {
  stop("Missing MAF: ", input_maf)
}
if (!file.exists(frozen_selection)) {
  stop("Missing frozen selection manifest: ", frozen_selection)
}

message("[INFO] reading ", input_maf)
mut <- fread(input_maf, sep = "\t", data.table = TRUE)
required_cols <- c(
  "Tumor_Sample_Barcode",
  "Chromosome",
  "Start_Position",
  "Reference_Allele",
  "Tumor_Seq_Allele2",
  "Variant_Type"
)
missing_cols <- setdiff(required_cols, names(mut))
if (length(missing_cols) > 0) {
  stop("Missing MAF columns: ", paste(missing_cols, collapse = ", "))
}

ref_genome <- load_ref_genome(ref_genome_pkg)
max_chr_sizes <- data.table(
  Chromosome = as.character(seqnames(ref_genome)),
  Max_Position = as.numeric(seqlengths(ref_genome))
)

mut <- copy(mut)
setnames(
  mut,
  old = required_cols,
  new = c("Sample", "Chromosome", "Start_Position", "Reference_Allele", "Tumor_Seq_Allele2", "Variant_Type")
)
mut[, Chromosome := normalize_seqnames(Chromosome)]
mut[, Start_Position := as.numeric(Start_Position)]
mut[, Reference_Allele := toupper(as.character(Reference_Allele))]
mut[, Tumor_Seq_Allele2 := toupper(as.character(Tumor_Seq_Allele2))]
mut[, Sample := normalize_sample_ids(Sample, patient_mode)]

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
  stop("No usable SNP rows remain after filtering")
}

# This block intentionally mirrors cptac_sigprof/extract_cptac_sbs.R. Keeping the
# same catalogue builder is what makes the Sigfit/deconstructSigs comparison fair.
sigs_input <- mut.to.sigs.input(
  mut.ref = as.data.frame(mut),
  sample.id = "Sample",
  chr = "Chromosome",
  pos = "Start_Position",
  ref = "Reference_Allele",
  alt = "Tumor_Seq_Allele2",
  bsg = ref_genome
)
sigs_input <- sigs_input[rowSums(sigs_input) > 29L, , drop = FALSE]
if (nrow(sigs_input) == 0) {
  stop("No samples have at least 30 usable SBS96 mutations")
}

eligible <- data.table(
  Sample = rownames(sigs_input),
  PATIENT = normalize_patient_ids(rownames(sigs_input), patient_mode),
  snv_count = as.integer(rowSums(sigs_input))
)
setorder(eligible, PATIENT, Sample)
set.seed(2018)
selected_samples <- eligible[, .SD[sample.int(.N, 1L)], by = PATIENT]$Sample
eligible[, selected := Sample %in% selected_samples]

frozen <- fread(frozen_selection, data.table = TRUE)
assert_identical_manifest(copy(eligible), frozen, frozen_selection)
selected <- eligible[selected == TRUE][order(PATIENT, Sample)]
if (selected[, anyDuplicated(PATIENT)] != 0L) {
  stop("Frozen selection contains more than one sample for a patient")
}

contexts_deconstruct <- colnames(sigs_input)
contexts_sigfit <- context_to_sigfit(contexts_deconstruct)
if (length(contexts_sigfit) != 96L || anyDuplicated(contexts_sigfit)) {
  stop("Expected exactly 96 unique SBS contexts")
}

counts <- as.matrix(sigs_input[selected$Sample, , drop = FALSE])
storage.mode(counts) <- "integer"
colnames(counts) <- contexts_sigfit
if (!identical(as.integer(rowSums(counts)), selected$snv_count)) {
  stop("Selected SBS96 row sums do not match the frozen SNV counts")
}

counts_table <- data.table(
  Sample = selected$Sample,
  PATIENT = selected$PATIENT,
  snv_count = selected$snv_count
)
counts_table <- cbind(counts_table, as.data.table(counts))
mapping <- data.table(
  context_index = seq_along(contexts_deconstruct),
  deconstructSigs_context = contexts_deconstruct,
  sigfit_context = contexts_sigfit
)

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
counts_path <- file.path(output_dir, "sbs96_counts.tsv")
manifest_path <- file.path(output_dir, "selected_samples_rebuilt.csv")
mapping_path <- file.path(output_dir, "context_mapping.tsv")
metadata_path <- file.path(output_dir, "input_metadata.tsv")

fwrite(counts_table, counts_path, sep = "\t")
fwrite(eligible, manifest_path)
fwrite(mapping, mapping_path, sep = "\t")
fwrite(
  data.table(
    dataset = dataset,
    cohort = cohort,
    input_maf = normalizePath(input_maf, mustWork = TRUE),
    frozen_selection = normalizePath(frozen_selection, mustWork = TRUE),
    patient_mode = patient_mode,
    reference_genome_package = ref_genome_pkg,
    minimum_snv = 30L,
    selection_seed = 2018L,
    eligible_sample_n = nrow(eligible),
    selected_sample_n = nrow(selected),
    selected_patient_n = uniqueN(selected$PATIENT),
    selected_snv_sum = sum(selected$snv_count)
  ),
  metadata_path,
  sep = "\t"
)

message("[OK] eligible samples: ", nrow(eligible))
message("[OK] selected samples/patients: ", nrow(selected))
message("[OK] selected SBS96 count sum: ", sum(counts))
message("[OUT] ", counts_path)
