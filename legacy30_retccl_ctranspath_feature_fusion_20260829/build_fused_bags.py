#!/opt/anaconda/envs/cenv/bin/python
"""Build coordinate-aligned RetCCL+CTransPath fixed bags."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap


BASE = Path("/home/marugoto-1.0.0-regression")
ROOT = BASE / "legacy30_retccl_ctranspath_feature_fusion_20260829"
SOURCE = BASE / "sigprofiler_improved_20260818/inputs/nature_sigprofiler24_v1/matched"
RETCCL = BASE / "sigprofiler_improved_20260818/features_end2end_retccl_qc_v1"
TCGA_MAIN = Path("/home/end2end-WSI-preprocessing-1.0.0-preprocessing/cache_testa_ctranspath")
TCGA_EXTRA = BASE / "sigprofiler_improved_20260818/features_end2end_ctranspath_completion_v1/TCGA"
CPTAC = Path("/home/end2end-WSI-preprocessing-1.0.0-preprocessing/cache_cptac_ctranspath")
OUT = ROOT / "fixed_bag_cache"
DATASETS = ("TCGA", "CPTAC")
CANCERS = ("BRCA", "HNSC", "KIRC", "LUAD", "LUSC", "OV", "UCEC")
CACHE_CANCER = {
    "BRCA": "Brca", "HNSC": "Hnscc", "KIRC": "Ccrcc", "LUAD": "Luad",
    "LUSC": "Lscc", "OV": "Ov", "UCEC": "Ucec",
}
BAG_SIZE = 1024
RETCCL_DIM = 2048
CTRANS_DIM = 768
FEATURE_DIM = RETCCL_DIM + CTRANS_DIM
BASE_SEED = 20260907


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(dataset: str, cancer: str, patient: str) -> int:
    token = f"{BASE_SEED}|{dataset}|{cancer}|{patient}".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little")


def prefix(mode: str, dataset: str, cancer: str) -> Path:
    return OUT / mode / f"{dataset}_{CACHE_CANCER[cancer]}_bag{BAG_SIZE}"


def ctranspath_path(dataset: str, filename: str) -> Path:
    name = f"{filename}.h5"
    if dataset == "CPTAC":
        return CPTAC / name
    path = TCGA_MAIN / name
    return path if path.exists() else TCGA_EXTRA / name


def keys(coords: np.ndarray, augmented: np.ndarray) -> list[tuple[int, int, bool]]:
    return [
        (int(coord[0]), int(coord[1]), bool(flag))
        for coord, flag in zip(coords, augmented)
    ]


def read_slide(dataset: str, filename: str):
    retccl_path = RETCCL / dataset / f"{filename}.h5"
    ctrans_path = ctranspath_path(dataset, filename)
    with h5py.File(retccl_path, "r") as retccl, h5py.File(ctrans_path, "r") as ctrans:
        ret_coords = retccl["coords"][:]
        ret_aug = retccl["augmented"][:] if "augmented" in retccl else np.zeros(len(ret_coords), bool)
        ret_features = retccl["feats"][:]
        scores = retccl["tile_qc_score"][:].astype(np.float32)
        ctr_coords = ctrans["coords"][:]
        ctr_aug = ctrans["augmented"][:] if "augmented" in ctrans else np.zeros(len(ctr_coords), bool)
        ctr_features = ctrans["feats"][:]
    if ret_features.ndim != 2 or ret_features.shape[1] != RETCCL_DIM:
        raise ValueError(f"Invalid RetCCL shape: {retccl_path}/{ret_features.shape}")
    if ctr_features.ndim != 2 or ctr_features.shape[1] != CTRANS_DIM:
        raise ValueError(f"Invalid CTransPath shape: {ctrans_path}/{ctr_features.shape}")
    if (ret_coords.shape == ctr_coords.shape and np.array_equal(ret_coords, ctr_coords)
            and np.array_equal(ret_aug, ctr_aug)):
        fused = np.concatenate([ret_features, ctr_features], axis=1)
        return fused.astype(np.float16, copy=False), scores, len(ctr_features), len(fused)
    ret_keys = keys(ret_coords, ret_aug)
    if len(set(ret_keys)) != len(ret_keys):
        raise ValueError(f"Duplicate RetCCL coordinate keys: {retccl_path}")
    ret_index = {key: index for index, key in enumerate(ret_keys)}
    ctr_selected, ret_selected = [], []
    for ctr_index, key in enumerate(keys(ctr_coords, ctr_aug)):
        if key in ret_index:
            ctr_selected.append(ctr_index)
            ret_selected.append(ret_index[key])
    if not ctr_selected:
        raise ValueError(f"No coordinate intersection: {retccl_path}/{ctrans_path}")
    ctr_selected = np.asarray(ctr_selected, dtype=np.int64)
    ret_selected = np.asarray(ret_selected, dtype=np.int64)
    fused = np.concatenate(
        [ret_features[ret_selected], ctr_features[ctr_selected]], axis=1,
    )
    return (
        fused.astype(np.float16, copy=False), scores[ret_selected],
        len(ctr_features), len(ctr_selected),
    )


def select_uniform(count: int, rng: np.random.Generator) -> np.ndarray:
    if count <= BAG_SIZE:
        return np.arange(count, dtype=np.int64)
    return np.sort(rng.choice(count, BAG_SIZE, replace=False))


def select_qhybrid(score: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    count = len(score)
    if count <= BAG_SIZE:
        return np.arange(count, dtype=np.int64)
    high_count = BAG_SIZE // 2
    order = np.argsort(-score, kind="stable")
    high = order[:high_count]
    broad = rng.choice(order[high_count:], BAG_SIZE - high_count, replace=False)
    selected = np.concatenate([high, broad])
    rng.shuffle(selected)
    return selected.astype(np.int64, copy=False)


def build(dataset: str, cancer: str) -> None:
    outputs = {mode: prefix(mode, dataset, cancer) for mode in ("uniform", "qhybrid")}
    if all(value.with_suffix(".done").exists() for value in outputs.values()):
        print(f"Already complete: {dataset}/{cancer}")
        return
    for value in outputs.values():
        if any(value.with_suffix(suffix).exists() for suffix in (".bags.npy", ".meta.csv", ".json", ".done")):
            raise FileExistsError(value)
    slides_path = SOURCE / dataset / cancer / "slides.tsv"
    slides = pd.read_csv(slides_path, sep="\t", dtype={"PATIENT": str})
    grouped = list(slides.groupby("PATIENT", sort=True))
    arrays, temporary = {}, {}
    for mode, value in outputs.items():
        value.parent.mkdir(parents=True, exist_ok=True)
        tmp = value.with_name(value.name + f".partial.{os.getpid()}.bags.npy")
        temporary[mode] = tmp
        arrays[mode] = open_memmap(
            tmp, mode="w+", dtype=np.float16,
            shape=(len(grouped), BAG_SIZE, FEATURE_DIM),
        )
    audit_rows = []
    for patient_index, (patient, patient_slides) in enumerate(grouped):
        feature_parts, score_parts = [], []
        source_tiles = aligned_tiles = 0
        for row in patient_slides.itertuples(index=False):
            features, scores, source_n, aligned_n = read_slide(dataset, str(row.FILENAME))
            feature_parts.append(features)
            score_parts.append(scores)
            source_tiles += source_n
            aligned_tiles += aligned_n
        features = np.concatenate(feature_parts)
        scores = np.concatenate(score_parts)
        rng = np.random.default_rng(stable_seed(dataset, cancer, str(patient)))
        selected = {
            "uniform": select_uniform(len(features), rng),
            "qhybrid": select_qhybrid(scores, rng),
        }
        for mode, indices in selected.items():
            arrays[mode][patient_index].fill(0)
            arrays[mode][patient_index, :len(indices)] = features[indices]
        audit_rows.append({
            "PATIENT": str(patient), "slides": len(patient_slides),
            "source_ctrans_tiles": source_tiles, "coordinate_aligned_tiles": aligned_tiles,
            "coordinate_retention": aligned_tiles / source_tiles,
            "available_tiles": len(features), "lens": min(len(features), BAG_SIZE),
            "uniform_mean_qc": float(scores[selected["uniform"]].mean()),
            "qhybrid_mean_qc": float(scores[selected["qhybrid"]].mean()),
        })
        if (patient_index + 1) % 25 == 0:
            print(f"{dataset}/{cancer} {patient_index + 1}/{len(grouped)}", flush=True)
    audit = pd.DataFrame(audit_rows)
    for mode, value in outputs.items():
        arrays[mode].flush()
        del arrays[mode]
        os.replace(temporary[mode], value.with_suffix(".bags.npy"))
        audit[["PATIENT", "lens"]].to_csv(value.with_suffix(".meta.csv"), index=False)
        details = {
            "status": "complete", "dataset": dataset, "cancer": cancer,
            "encoders": ["RetCCL", "CTransPath"], "mode": mode,
            "patients": len(grouped), "slides": len(slides),
            "bag_size": BAG_SIZE, "feature_dimension": FEATURE_DIM,
            "retccl_dimension": RETCCL_DIM, "ctranspath_dimension": CTRANS_DIM,
            "coordinate_alignment": "exact (x,y,augmented) intersection",
            "minimum_patient_coordinate_retention": float(audit.coordinate_retention.min()),
            "median_patient_coordinate_retention": float(audit.coordinate_retention.median()),
            "bags_sha256": sha256(value.with_suffix(".bags.npy")),
            "source_manifest_sha256": sha256(slides_path),
        }
        value.with_suffix(".json").write_text(json.dumps(details, indent=2) + "\n")
        value.with_suffix(".done").write_text("done\n")
    audit.to_csv(OUT / f"{dataset}_{cancer}_alignment_audit.tsv", sep="\t", index=False)
    print(f"Completed {dataset}/{cancer}")


def audit() -> None:
    rows, failures = [], []
    for mode in ("uniform", "qhybrid"):
        for dataset in DATASETS:
            for cancer in CANCERS:
                value = prefix(mode, dataset, cancer)
                try:
                    details = json.loads(value.with_suffix(".json").read_text())
                    bags = np.load(value.with_suffix(".bags.npy"), mmap_mode="r")
                    meta = pd.read_csv(value.with_suffix(".meta.csv"), dtype={"PATIENT": str})
                    if bags.shape != (len(meta), BAG_SIZE, FEATURE_DIM) or bags.dtype != np.float16:
                        raise ValueError(f"Invalid bag: {bags.shape}/{bags.dtype}")
                    if details["bags_sha256"] != sha256(value.with_suffix(".bags.npy")):
                        raise ValueError("Bag hash mismatch")
                    rows.append({
                        "mode": mode, "dataset": dataset, "cancer": cancer,
                        "patients": len(meta),
                        "minimum_coordinate_retention": details["minimum_patient_coordinate_retention"],
                    })
                except Exception as error:
                    failures.append({
                        "mode": mode, "dataset": dataset, "cancer": cancer,
                        "error": str(error),
                    })
    result = {
        "status": "complete" if not failures else "incomplete",
        "caches_expected": 28, "caches_complete": len(rows),
        "failures": failures, "audit_pass": len(rows) == 28 and not failures,
    }
    pd.DataFrame(rows).to_csv(OUT / "summary.tsv", sep="\t", index=False)
    (OUT / "AUDIT.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["audit_pass"]:
        raise RuntimeError("Fused bag audit failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--cancer", choices=CANCERS)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.audit:
        audit()
    elif args.dataset and args.cancer:
        build(args.dataset, args.cancer)
    else:
        raise ValueError("Provide dataset/cancer or --audit")


if __name__ == "__main__":
    main()
