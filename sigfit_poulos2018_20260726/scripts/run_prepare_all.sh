#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/marugoto-1.0.0-regression"
ANALYSIS_ROOT="${ROOT}/sigfit_poulos2018_20260726"
R_ENV="cptac-sigprof-r"
PREPARE="${ANALYSIS_ROOT}/scripts/prepare_sbs96_inputs.R"
AUDIT="${ANALYSIS_ROOT}/scripts/audit_sbs96_inputs.py"

mkdir -p "${ANALYSIS_ROOT}/logs"
for dataset in TCGA CPTAC; do
  for cohort in Brca Ccrcc Hnscc Lscc Luad Ov Ucec; do
    log="${ANALYSIS_ROOT}/logs/prepare_${dataset}_${cohort}.log"
    echo "[RUN] ${dataset} ${cohort}"
    conda run --no-capture-output -n "${R_ENV}" Rscript "${PREPARE}" \
      --dataset "${dataset}" \
      --cohort "${cohort}" \
      >"${log}" 2>&1
  done
done

"/opt/anaconda/envs/cenv/bin/python" "${AUDIT}"

