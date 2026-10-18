#!/usr/bin/env python3
"""Align frozen Sigfit SBS1-SBS30 weights to the locked WSI bag cohort."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/home/marugoto-1.0.0-regression")
ANALYSIS_ROOT = ROOT / "sigfit_poulos2018_20260726"
FIT_ROOT = ANALYSIS_ROOT / "fits"
BAG_ROOT = ROOT / "publication_tumor_only_locked_20260719" / "fixed_bag_cache"
OUTPUT_ROOT = ANALYSIS_ROOT / "model_inputs"
CANCERS = ("Brca", "Ccrcc", "Hnscc", "Lscc", "Luad", "Ov", "Ucec")
DATASETS = ("TCGA", "CPTAC")
SBS_COLUMNS = tuple(f"SBS{index}" for index in range(1, 31))
EXPECTED_COLUMNS = ("PATIENT", *SBS_COLUMNS)
ROW_SUM_TOLERANCE = 1e-8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_weights(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t", dtype={"PATIENT": str})
    if tuple(table.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"{path} must contain only PATIENT followed by SBS1-SBS30; "
            f"found {list(table.columns)}"
        )
    table["PATIENT"] = table["PATIENT"].astype(str)
    if (
        table["PATIENT"].isna().any()
        or table["PATIENT"].eq("").any()
        or table["PATIENT"].duplicated().any()
    ):
        raise ValueError(f"Missing or duplicate patient IDs in {path}")
    for column in SBS_COLUMNS:
        table[column] = pd.to_numeric(table[column], errors="raise")
    values = table.loc[:, SBS_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite Sigfit weights in {path}")
    if (values < 0).any() or (values > 1).any():
        raise ValueError(f"Sigfit weights outside [0, 1] in {path}")
    row_sums = values.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=0.0, atol=ROW_SUM_TOLERANCE):
        bad = np.flatnonzero(np.abs(row_sums - 1.0) > ROW_SUM_TOLERANCE)[:10]
        raise ValueError(f"Sigfit weights do not sum to one in {path}; rows={bad.tolist()}")
    return table


def validate_bag_metadata(path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(path, dtype={"PATIENT": str})
    required = {"PATIENT", "lens", "feature_dim"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Missing bag metadata columns in {path}: {sorted(missing)}")
    metadata = metadata.loc[:, ["PATIENT", "lens", "feature_dim"]].copy()
    if (
        metadata["PATIENT"].isna().any()
        or metadata["PATIENT"].eq("").any()
        or metadata["PATIENT"].duplicated().any()
    ):
        raise ValueError(f"Missing or duplicate bag patients in {path}")
    if not metadata["feature_dim"].eq(2048).all():
        raise ValueError(f"Unexpected feature dimension in {path}")
    if metadata["lens"].lt(1).any() or metadata["lens"].gt(512).any():
        raise ValueError(f"Invalid fixed-bag length in {path}")
    return metadata


def main() -> None:
    summary_rows: list[dict[str, object]] = []
    file_rows: list[dict[str, str]] = []
    for dataset in DATASETS:
        for cancer in CANCERS:
            source = FIT_ROOT / dataset / cancer / "patient_weights.tsv"
            bag_metadata_path = BAG_ROOT / f"{dataset}_{cancer}_bag512.meta.csv"
            if not source.is_file():
                raise FileNotFoundError(source)
            if not bag_metadata_path.is_file():
                raise FileNotFoundError(bag_metadata_path)

            weights = validate_weights(source)
            metadata = validate_bag_metadata(bag_metadata_path)
            aligned = metadata.loc[:, ["PATIENT"]].merge(
                weights,
                on="PATIENT",
                how="inner",
                sort=False,
                validate="one_to_one",
            )
            if aligned.empty:
                raise ValueError(f"No WSI-aligned patients for {dataset} {cancer}")
            if aligned["PATIENT"].duplicated().any():
                raise ValueError(f"Duplicate aligned patients for {dataset} {cancer}")
            aligned_values = aligned.loc[:, SBS_COLUMNS].to_numpy(dtype=float)
            if not np.allclose(
                aligned_values.sum(axis=1), 1.0, rtol=0.0, atol=ROW_SUM_TOLERANCE
            ):
                raise ValueError(f"Aligned weights do not sum to one: {dataset} {cancer}")

            destination = OUTPUT_ROOT / dataset / cancer / "sbs30_targets.tsv"
            destination.parent.mkdir(parents=True, exist_ok=True)
            aligned.to_csv(destination, sep="\t", index=False, lineterminator="\n")
            excel_destination = destination.with_suffix(".xlsx")
            aligned.to_excel(excel_destination, index=False)

            fit_patients = set(weights["PATIENT"])
            bag_patients = set(metadata["PATIENT"])
            summary_rows.append(
                {
                    "dataset": dataset,
                    "cancer": cancer,
                    "sigfit_patient_n": len(weights),
                    "bag_patient_n": len(metadata),
                    "wsi_aligned_n": len(aligned),
                    "sigfit_without_bag_n": len(fit_patients - bag_patients),
                    "bag_without_sigfit_n": len(bag_patients - fit_patients),
                    "weight_row_sum_min": float(aligned_values.sum(axis=1).min()),
                    "weight_row_sum_max": float(aligned_values.sum(axis=1).max()),
                }
            )
            file_rows.append(
                {
                    "dataset": dataset,
                    "cancer": cancer,
                    "sigfit_weights": str(source.relative_to(ROOT)),
                    "sigfit_weights_sha256": sha256(source),
                    "bag_metadata": str(bag_metadata_path.relative_to(ROOT)),
                    "bag_metadata_sha256": sha256(bag_metadata_path),
                    "model_input": str(destination.relative_to(ROOT)),
                    "model_input_sha256": sha256(destination),
                    "model_input_xlsx": str(excel_destination.relative_to(ROOT)),
                    "model_input_xlsx_sha256": sha256(excel_destination),
                }
            )

    summary = pd.DataFrame(summary_rows)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_ROOT / "wsi_alignment_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False, lineterminator="\n")
    manifest = {
        "analysis": "Sigfit Poulos 2018 SBS1-SBS30 WSI target alignment",
        "target_source": "fits/{dataset}/{cohort}/patient_weights.tsv",
        "target_validation": "exactly 30 finite weights in [0,1], row sum one",
        "signature_sparsity_filter": None,
        "signature_nonzero_filter": None,
        "alignment": "one-to-one patient merge against locked fixed-bag metadata",
        "cptac_used_for_tcga_alignment_or_selection": False,
        "files": file_rows,
        "summary_sha256": sha256(summary_path),
    }
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
