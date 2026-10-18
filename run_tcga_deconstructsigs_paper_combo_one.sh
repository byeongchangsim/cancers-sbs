#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/marugoto-1.0.0-regression"
R_ENV="cptac-sigprof-r"
R_SCRIPT="${ROOT}/cptac_sigprof/extract_cptac_sbs.R"
CONFIG_TSV="${ROOT}/paper_deconstructsigs_configs.tsv"
OUT_ROOT="${ROOT}/tcga_sigprof_paper_sweep"

declare -A CODES=(
  [Brca]="BRCA"
  [Ccrcc]="KIRC"
  [Hnscc]="HNSC"
  [Lscc]="LUSC"
  [Luad]="LUAD"
  [Ov]="OV"
  [Ucec]="UCEC"
)

cohort="${1:-}"
config_id="${2:-}"
if [[ -z "${cohort}" || -z "${config_id}" ]]; then
  echo "usage: $0 <Brca|Ccrcc|Hnscc|Lscc|Luad|Ov|Ucec> <config_id>" >&2
  exit 1
fi

code="${CODES[$cohort]:-}"
if [[ -z "${code}" ]]; then
  echo "[ERROR] Unsupported cohort for paper sweep: ${cohort}" >&2
  exit 1
fi

cfg_line="$(awk -F'\t' -v id="${config_id}" 'NR>1 && $1==id {print $0}' "${CONFIG_TSV}")"
if [[ -z "${cfg_line}" ]]; then
  echo "[ERROR] Unknown config_id: ${config_id}" >&2
  exit 1
fi
IFS=$'\t' read -r _cfg ref_name tri_counts_method associated_mode min_mutations signature_cutoff <<< "${cfg_line}"

associated_signatures=""
if [[ "${associated_mode}" == "cancer_specific" ]]; then
  if [[ "${cohort}" == "Brca" ]]; then
    associated_signatures="SBS1,SBS2,SBS3,SBS5,SBS8,SBS9,SBS13,SBS17a,SBS17b,SBS18,SBS37,SBS40,SBS41"
  elif [[ "${cohort}" == "Ucec" ]]; then
    associated_signatures="SBS1,SBS2,SBS3,SBS5,SBS6,SBS9,SBS10a,SBS10b,SBS13,SBS14,SBS15,SBS18,SBS26,SBS28,SBS40,SBS41"
  fi
fi

maf="${ROOT}/tcga_sigprof_v34_mc3_rerun/input_maf/${cohort}/TCGA-${code}_mc3_public.maf"
out_dir="${OUT_ROOT}/by_config/${config_id}/${cohort}"
raw_xlsx="${out_dir}/TCGA_${code}_${config_id}_raw.xlsx"
raw_csv="${out_dir}/TCGA_${code}_${config_id}_raw.csv"
counts_csv="${out_dir}/TCGA_${code}_${config_id}_sample_snv_count.csv"

mkdir -p "${out_dir}"

if [[ -f "${raw_xlsx}" && -f "${raw_csv}" && -f "${counts_csv}" ]]; then
  echo "[SKIP] ${cohort} ${config_id}: outputs already exist"
  exit 0
fi

if [[ ! -f "${maf}" ]]; then
  echo "[ERROR] Missing MC3 cohort MAF: ${maf}" >&2
  exit 1
fi

echo "[RUN] ${cohort} ${config_id}"
echo "      ref_name=${ref_name} tri_counts_method=${tri_counts_method} associated_mode=${associated_mode} min_mutations=${min_mutations} signature_cutoff=${signature_cutoff}"

conda run --no-capture-output -n "${R_ENV}" Rscript "${R_SCRIPT}" \
  --input-maf "${maf}" \
  --output-xlsx "${raw_xlsx}" \
  --output-csv "${raw_csv}" \
  --output-counts "${counts_csv}" \
  --signatures-ref-name "${ref_name}" \
  --tri-counts-method "${tri_counts_method}" \
  --associated-signatures "${associated_signatures}" \
  --min-mutations "${min_mutations}" \
  --signature-cutoff "${signature_cutoff}" \
  --sample-col "Tumor_Sample_Barcode" \
  --chr-col "Chromosome" \
  --pos-col "Start_Position" \
  --ref-col "Reference_Allele" \
  --alt-col "Tumor_Seq_Allele2" \
  --variant-type-col "Variant_Type" \
  --patient-mode "tcga" \
  --ref-genome-pkg "BSgenome.Hsapiens.UCSC.hg19"

echo "[OK] ${cohort} ${config_id}: worker finished"
