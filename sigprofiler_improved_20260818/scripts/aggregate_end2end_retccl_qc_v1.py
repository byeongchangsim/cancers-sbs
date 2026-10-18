#!/opt/anaconda/envs/cenv/bin/python
"""Create patient-level old/new RetCCL summaries under several QC rules."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd


ROOT = Path("/home/marugoto-1.0.0-regression")
RUN = ROOT / "sigprofiler_improved_20260818"
SOURCE = RUN / "inputs/nature_sigprofiler24_v1/matched"
EXTRACTION_INPUT = RUN / "inputs/end2end_retccl_qc_v1"
NEW_FEATURES = RUN / "features_end2end_retccl_qc_v1"
OUTPUT = EXTRACTION_INPUT / "patient_aggregates"
DATASETS = ("TCGA", "CPTAC")
CANCERS = ("BRCA", "HNSC", "KIRC", "LUAD", "LUSC", "OV", "UCEC")
FEATURE_DIMENSION = 2048
FRACTIONS = (0.25, 0.50)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Moments:
    def __init__(self, dimension: int) -> None:
        self.weight = 0.0
        self.total = np.zeros(dimension, dtype=np.float64)
        self.square = np.zeros(dimension, dtype=np.float64)

    def add(self, values: np.ndarray, weights: Optional[np.ndarray] = None) -> None:
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(self.total):
            raise ValueError(f"Invalid aggregate feature shape: {values.shape}")
        if weights is None:
            self.weight += float(len(values))
            self.total += values.sum(axis=0, dtype=np.float64)
            self.square += np.square(values, dtype=np.float64).sum(axis=0)
        else:
            weights = np.asarray(weights, dtype=np.float64).reshape(-1)
            if len(weights) != len(values) or not np.isfinite(weights).all():
                raise ValueError("Invalid aggregate weights")
            self.weight += float(weights.sum())
            self.total += np.einsum("n,nf->f", weights, values, optimize=True)
            self.square += np.einsum(
                "n,nf->f", weights, np.square(values, dtype=np.float64),
                optimize=True,
            )

    def finish(self) -> np.ndarray:
        if self.weight <= 0:
            raise ValueError("Empty patient feature aggregate")
        mean = self.total / self.weight
        variance = np.maximum(self.square / self.weight - np.square(mean), 0.0)
        return np.concatenate([mean, np.sqrt(variance)]).astype(np.float32)


def output_path(dataset: str, cancer: str) -> Path:
    return OUTPUT / f"{dataset}_{cancer}_old_new_qc_mean_std.npz"


def summarize_patient(rows: pd.DataFrame, dataset: str):
    aggregators = {
        "old_all": Moments(FEATURE_DIMENSION),
        "new_all": Moments(FEATURE_DIMENSION),
        "old_top25": Moments(FEATURE_DIMENSION),
        "new_top25": Moments(FEATURE_DIMENSION),
        "old_top50": Moments(FEATURE_DIMENSION),
        "new_top50": Moments(FEATURE_DIMENSION),
        "old_qweighted": Moments(FEATURE_DIMENSION),
        "new_qweighted": Moments(FEATURE_DIMENSION),
    }
    qc_values = []
    tile_count = 0
    row_order_fallbacks = 0
    for row in rows.itertuples(index=False):
        old_path = Path(row.feature_path)
        new_path = NEW_FEATURES / dataset / f"{row.FILENAME}.h5"
        with h5py.File(old_path, "r") as old_handle, h5py.File(new_path, "r") as new_handle:
            old = old_handle["feats"][:].astype(np.float32)
            new = new_handle["feats"][:].astype(np.float32)
            score = new_handle["tile_qc_score"][:].astype(np.float32)
            old_coords = old_handle["coords"][:]
            new_coords = new_handle["coords"][:]
            qc_matrix = np.column_stack([
                new_handle[name][:].astype(np.float32)
                for name in (
                    "tile_qc_occupancy", "tile_qc_tissue_fraction",
                    "tile_qc_edge_density", "tile_qc_focus", "tile_qc_score",
                )
            ])
            if len(score) != len(new):
                raise ValueError(f"New feature/QC row mismatch: {new_path}")
            if not np.array_equal(old_coords, new_coords):
                old_index = {}
                for index, coordinate in enumerate(old_coords):
                    # A small number of historical RetCCL HDF5 files contain
                    # repeated coordinates.  Preserve the first serialized
                    # row deterministically; never duplicate a new QC row.
                    old_index.setdefault(
                        tuple(int(value) for value in coordinate), index
                    )
                new_keep, old_keep = [], []
                for index, coordinate in enumerate(new_coords):
                    match = old_index.get(tuple(int(value) for value in coordinate))
                    if match is not None:
                        new_keep.append(index)
                        old_keep.append(match)
                if not new_keep:
                    old_augmented = old_handle["augmented"][:]
                    if (
                        old.shape == new.shape
                        and len(old_index) == 1
                        and not np.asarray(old_augmented, dtype=bool).any()
                    ):
                        # One historical cache file serialized every coordinate
                        # as (0, 0), but preserved the original row count and
                        # unaugmented row order.  Equal-length row alignment is
                        # the only recoverable, deterministic interpretation.
                        row_order_fallbacks += 1
                    else:
                        raise ValueError(
                            f"No common old/new coordinates: {old_path} / {new_path}"
                        )
                else:
                    new = new[np.asarray(new_keep, dtype=np.int64)]
                    old = old[np.asarray(old_keep, dtype=np.int64)]
                    score = score[np.asarray(new_keep, dtype=np.int64)]
                    qc_matrix = qc_matrix[np.asarray(new_keep, dtype=np.int64)]
            if new.shape[1] != FEATURE_DIMENSION or not np.isfinite(score).all():
                raise ValueError(f"Invalid new feature/QC: {new_path}")

        aggregators["old_all"].add(old)
        aggregators["new_all"].add(new)
        quality_weight = 0.05 + score.astype(np.float64)
        aggregators["old_qweighted"].add(old, quality_weight)
        aggregators["new_qweighted"].add(new, quality_weight)
        for fraction in FRACTIONS:
            count = max(1, int(np.ceil(len(score) * fraction)))
            selected = np.argpartition(score, len(score) - count)[-count:]
            suffix = int(fraction * 100)
            aggregators[f"old_top{suffix}"].add(old[selected])
            aggregators[f"new_top{suffix}"].add(new[selected])
        qc_values.append(qc_matrix)
        tile_count += len(score)

    qc = np.concatenate(qc_values, axis=0)
    qc_summary = np.concatenate([
        qc.mean(axis=0), qc.std(axis=0), qc.min(axis=0), qc.max(axis=0)
    ]).astype(np.float32)
    return (
        {name: value.finish() for name, value in aggregators.items()},
        qc_summary,
        tile_count,
        row_order_fallbacks,
    )


def aggregate(dataset: str, cancer: str) -> None:
    destination = output_path(dataset, cancer)
    metadata_path = destination.with_suffix(".json")
    if destination.exists() and metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if previous.get("status") == "complete" and sha256(destination) == previous.get("archive_sha256"):
            print(f"Already complete: {dataset}/{cancer}")
            return
    if destination.exists() or metadata_path.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    manifest_path = SOURCE / dataset / cancer / "slides.tsv"
    slides = pd.read_csv(manifest_path, sep="\t", dtype={"PATIENT": str})
    patients, tile_counts, qc_rows = [], [], []
    row_order_fallbacks = 0
    variants: dict[str, list[np.ndarray]] = {}
    for patient, rows in slides.groupby("PATIENT", sort=True):
        summaries, qc, tiles, patient_fallbacks = summarize_patient(rows, dataset)
        patients.append(str(patient))
        tile_counts.append(tiles)
        qc_rows.append(qc)
        row_order_fallbacks += patient_fallbacks
        for name, value in summaries.items():
            variants.setdefault(name, []).append(value)
        if len(patients) % 25 == 0:
            print(
                f"{dataset}/{cancer} patients={len(patients)}/{slides.PATIENT.nunique()}",
                flush=True,
            )

    arrays = {
        "patients": np.asarray(patients, dtype=str),
        "tile_counts": np.asarray(tile_counts, dtype=np.int64),
        "qc_summary": np.stack(qc_rows),
        **{name: np.stack(values) for name, values in variants.items()},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + f".partial.{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, destination)
    metadata = {
        "status": "complete",
        "dataset": dataset,
        "cancer": cancer,
        "patients": len(patients),
        "slides": int(len(slides)),
        "tiles": int(sum(tile_counts)),
        "feature_dimension_per_mean_std_block": FEATURE_DIMENSION * 2,
        "variants": sorted(variants),
        "qc_summary_dimension": int(arrays["qc_summary"].shape[1]),
        "row_order_alignment_fallback_slides": int(row_order_fallbacks),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256(manifest_path),
        "extraction_contract_sha256": sha256(EXTRACTION_INPUT / "CONTRACT.json"),
        "archive": str(destination),
        "archive_sha256": sha256(destination),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


def audit() -> None:
    rows, failures = [], []
    for dataset in DATASETS:
        for cancer in CANCERS:
            path = output_path(dataset, cancer)
            metadata = path.with_suffix(".json")
            try:
                value = np.load(path, allow_pickle=False)
                patients = value["patients"].astype(str)
                if len(patients) != len(set(patients)) or len(patients) < 5:
                    raise ValueError("Invalid patient identifiers")
                for name in (
                    "old_all", "new_all", "old_top25", "new_top25",
                    "old_top50", "new_top50", "old_qweighted", "new_qweighted",
                ):
                    if value[name].shape != (len(patients), FEATURE_DIMENSION * 2):
                        raise ValueError(f"Invalid {name} shape: {value[name].shape}")
                    if not np.isfinite(value[name]).all():
                        raise ValueError(f"Non-finite {name}")
                rows.append({
                    "dataset": dataset,
                    "cancer": cancer,
                    "patients": len(patients),
                    "tiles": int(value["tile_counts"].sum()),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                })
                if not metadata.exists():
                    raise FileNotFoundError(metadata)
            except Exception as error:
                failures.append({
                    "dataset": dataset, "cancer": cancer, "error": str(error)
                })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "summary.tsv", sep="\t", index=False)
    result = {
        "status": "complete" if not failures else "incomplete",
        "cohorts_expected": len(DATASETS) * len(CANCERS),
        "cohorts_complete": len(rows),
        "failures": failures,
        "audit_pass": len(rows) == len(DATASETS) * len(CANCERS) and not failures,
    }
    (OUTPUT / "AUDIT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["audit_pass"]:
        raise RuntimeError("Patient aggregate audit failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--cancer", choices=CANCERS)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.audit:
        audit()
    elif args.dataset and args.cancer:
        aggregate(args.dataset, args.cancer)
    else:
        parser.error("Provide --dataset and --cancer, or --audit")


if __name__ == "__main__":
    main()
