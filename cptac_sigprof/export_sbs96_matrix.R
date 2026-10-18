#!/usr/bin/env Rscript

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

load_ref_genome <- function(pkg_name) {
  suppressPackageStartupMessages(library(pkg_name, character.only = TRUE))
  get(pkg_name, envir = asNamespace(pkg_name))
}

option_list <- list(
  make_option("--input-maf", type = "character", dest = "input_maf", help = "Input mutation table"),
  make_option("--output-matrix", type = "character", dest = "output_matrix", help = "Output SBS96 matrix path"),
  make_option("--sample-col", type = "character", dest = "sample_col", default = "Tumor_Sample_Barcode"),
  make_option("--chr-col", type = "character", dest = "chr_col", default = "Chromosome"),
  make_option("--pos-col", type = "character", dest = "pos_col", default = "Start_Position"),
  make_option("--ref-col", type = "character", dest = "ref_col", default = "Reference_Allele"),
  make_option("--alt-col", type = "character", dest = "alt_col", default = "Tumor_Seq_Allele2"),
  make_option("--variant-type-col", type = "character", dest = "variant_type_col", default = "Variant_Type"),
  make_option("--patient-mode", type = "character", dest = "patient_mode", default = "cptac",
              help = "tcga, cptac, or identity"),
  make_option("--ref-genome-pkg", type = "character", dest = "ref_genome_pkg",
              default = "BSgenome.Hsapiens.UCSC.hg38")
)

opt <- parse_args(OptionParser(option_list = option_list))

required_args <- c("input_maf", "output_matrix")
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

setnames(
  mut,
  old = c(opt$sample_col, opt$chr_col, opt$pos_col, opt$ref_col, opt$alt_col, opt$variant_type_col),
  new = c("Sample", "Chromosome", "Start_Position", "Reference_Allele", "Tumor_Seq_Allele2", "Variant_Type")
)

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

matrix_df <- data.frame(MutationType = colnames(sigs_input), t(sigs_input), check.names = FALSE)
dir.create(dirname(opt$output_matrix), recursive = TRUE, showWarnings = FALSE)
fwrite(matrix_df, opt$output_matrix, sep = "\t")

message("[OK] usable SNP rows: ", nrow(mut))
message("[OK] samples: ", nrow(sigs_input))
message("[OUT] matrix: ", opt$output_matrix)
