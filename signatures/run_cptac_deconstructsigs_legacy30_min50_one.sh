#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <Brca|Ccrcc|Hnscc|Lscc|Luad|Ov|Ucec>" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R_ENV="cptac-sigprof-r"
CONDA_BIN="/opt/anaconda/condabin/conda"
R_SCRIPT="${ROOT}/signatures/extract_cptac_sbs.R"
OUT_ROOT="${ROOT}/cptac_sigprof_legacy30_min50"
COHORT="$1"

declare -A MAFS=(
  [Brca]="/home/marugoto-1.0.0-classification/BRCA_somatic_mutation.maf"
  [Ccrcc]="/home/marugoto-1.0.0-classification/CCRCC_somatic_mutation.maf"
  [Hnscc]="/home/marugoto-1.0.0-classification/HNSCC_somatic_mutation.maf"
  [Lscc]="/home/marugoto-1.0.0-classification/LSCC_somatic_mutation.maf"
  [Luad]="/home/marugoto-1.0.0-classification/LUAD_somatic_mutation.maf"
  [Ov]="/home/marugoto-1.0.0-classification/OV_somatic_mutation.maf"
  [Ucec]="/home/marugoto-1.0.0-classification/UCEC_somatic_mutation.maf"
)

MAF="${MAFS[${COHORT}]:-}"
if [[ -z "${MAF}" ]]; then
  echo "[ERROR] unknown cohort: ${COHORT}" >&2
  exit 1
fi
if [[ ! -f "${MAF}" ]]; then
  echo "[ERROR] MAF not found: ${MAF}" >&2
  exit 1
fi

OUT_DIR="${OUT_ROOT}/by_cancer/${COHORT}"
RAW_XLSX="${OUT_DIR}/CPTAC_${COHORT}_legacy30_min50_raw.xlsx"
RAW_CSV="${OUT_DIR}/CPTAC_${COHORT}_legacy30_min50_raw.csv"
COUNTS_CSV="${OUT_DIR}/CPTAC_${COHORT}_legacy30_min50_sample_snv_count.csv"

if [[ -f "${RAW_XLSX}" && -f "${RAW_CSV}" ]]; then
  echo "[DONE] ${COHORT}: existing outputs"
  exit 0
fi

mkdir -p "${OUT_DIR}"
echo "[RUN] ${COHORT} legacy30_min50"
"${CONDA_BIN}" run --no-capture-output -n "${R_ENV}" Rscript "${R_SCRIPT}" \
  --input-maf "${MAF}" \
  --output-xlsx "${RAW_XLSX}" \
  --output-csv "${RAW_CSV}" \
  --output-counts "${COUNTS_CSV}" \
  --signatures-ref-name legacy30 \
  --tri-counts-method default \
  --associated-signatures "" \
  --min-mutations 50 \
  --signature-cutoff 0.06 \
  --sample-col "Tumor_Sample_Barcode" \
  --chr-col "Chromosome" \
  --pos-col "Start_Position" \
  --ref-col "Reference_Allele" \
  --alt-col "Tumor_Seq_Allele2" \
  --variant-type-col "Variant_Type" \
  --patient-mode cptac \
  --ref-genome-pkg "BSgenome.Hsapiens.UCSC.hg38"

echo "[OK] ${COHORT} legacy30_min50"
