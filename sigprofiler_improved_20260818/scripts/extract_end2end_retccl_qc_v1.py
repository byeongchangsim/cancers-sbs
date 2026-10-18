#!/opt/anaconda/envs/cenv/bin/python
"""Extract row-aligned QC-aware RetCCL features on eight independent GPUs.

The normalized End2End mosaics are immutable inputs.  Every HDF5 is written
to a temporary path, audited, and atomically renamed.  Existing feature files
from earlier analyses are never modified.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import PIL.Image
import torch


ROOT = Path("/home/marugoto-1.0.0-regression")
RUN = ROOT / "sigprofiler_improved_20260818"
END2END = Path("/home/end2end-WSI-preprocessing-1.0.0-preprocessing")
SOURCE = RUN / "inputs/nature_sigprofiler24_v1/matched"
INPUT = RUN / "inputs/end2end_retccl_qc_v1"
OUTPUT = RUN / "features_end2end_retccl_qc_v1"
LOGS = RUN / "logs_end2end_retccl_qc_v1"
CHECKPOINT = END2END / "best_ckpt.pth"
EXPECTED_CHECKPOINT_SHA256 = (
    "931956f31d3f1a3f6047f3172b9e59ee3460d29f7c0c2bb219cbc8e9207795ff"
)
EXTRACTOR = "marugoto-extract-v0.3.0_xiyuewang-retcll-931956f3_qc-v1"
QC_NAMES = (
    "tile_qc_occupancy",
    "tile_qc_tissue_fraction",
    "tile_qc_edge_density",
    "tile_qc_focus",
    "tile_qc_score",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def cache_root(dataset: str) -> Path:
    if dataset == "TCGA":
        return END2END / "cache_testa"
    if dataset == "CPTAC":
        return END2END / "cache_cptac"
    raise ValueError(dataset)


def prepare() -> None:
    for path in (INPUT, OUTPUT, LOGS):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")

    memberships: dict[tuple[str, str], set[str]] = {}
    old_tiles: dict[tuple[str, str], int] = {}
    for dataset in ("TCGA", "CPTAC"):
        for manifest_path in sorted((SOURCE / dataset).glob("*/slides.tsv")):
            cancer = manifest_path.parent.name
            frame = pd.read_csv(manifest_path, sep="\t", dtype=str)
            for row in frame.itertuples(index=False):
                key = (dataset, str(row.FILENAME))
                memberships.setdefault(key, set()).add(cancer)
                old_tiles[key] = int(row.n_tiles)

    rows = []
    for (dataset, filename), cancers in sorted(memberships.items()):
        source_dir = cache_root(dataset) / filename
        source_h5 = cache_root(dataset) / f"{filename}.h5"
        image = source_dir / "norm_slide.jpg"
        info = source_dir / "info.json"
        if not image.exists() or not source_h5.exists() or not info.exists():
            raise FileNotFoundError(
                f"Incomplete immutable cache for {dataset}/{filename}"
            )
        rows.append({
            "dataset": dataset,
            "cancers": ",".join(sorted(cancers)),
            "filename": filename,
            "source_image": str(image),
            "source_h5": str(source_h5),
            "source_info": str(info),
            "old_tile_count": old_tiles[(dataset, filename)],
            "output_h5": str(OUTPUT / dataset / f"{filename}.h5"),
            "source_image_bytes": image.stat().st_size,
        })
    jobs = pd.DataFrame(rows)
    jobs.insert(0, "job_index", np.arange(len(jobs), dtype=int))
    jobs["pilot"] = False

    # Eight representative jobs, one per GPU.  Choose cohort medians rather
    # than tiny slides so the pilot gives a realistic throughput estimate.
    group_rows = []
    for (dataset, cancers), group in jobs.groupby(["dataset", "cancers"]):
        ordered = group.sort_values(["old_tile_count", "filename"])
        group_rows.append(ordered.iloc[len(ordered) // 2])
    representatives = pd.DataFrame(group_rows).sort_values(
        ["dataset", "cancers", "filename"]
    )
    if len(representatives) < 8:
        raise RuntimeError("Fewer than eight cohort representatives")
    pilot_indices = representatives.head(8).job_index.astype(int).tolist()
    jobs.loc[jobs.job_index.isin(pilot_indices), "pilot"] = True

    INPUT.mkdir(parents=True)
    OUTPUT.mkdir(parents=True)
    LOGS.mkdir(parents=True)
    for dataset in ("TCGA", "CPTAC"):
        (OUTPUT / dataset).mkdir()
    jobs.to_csv(INPUT / "jobs.tsv", sep="\t", index=False)
    contract = {
        "status": "prepared",
        "analysis": "End2End cached-Macenko QC-aware RetCCL feature extraction",
        "source_manifests": str(SOURCE),
        "source_features_are_immutable": True,
        "output": str(OUTPUT),
        "jobs": int(len(jobs)),
        "datasets": jobs.groupby("dataset").size().astype(int).to_dict(),
        "cancer_membership": {
            f"{key[0]}:{key[1]}": int(value)
            for key, value in jobs.groupby(["dataset", "cancers"]).size().items()
        },
        "pilot_job_indices": pilot_indices,
        "tile_size": 224,
        "cached_black_threshold": 8,
        "cached_min_occupancy": 0.05,
        "qc_filter_thresholds": {
            "min_tissue_fraction": 0.0,
            "min_edge_density": 0.0,
            "min_focus": 0.0,
        },
        "feature_dimension": 2048,
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "expected_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "selection_contract": (
            "Quality features may be tuned using TCGA cross-validation only; "
            "CPTAC outcomes are not used for model or threshold selection."
        ),
    }
    if contract["checkpoint_sha256"] != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("RetCCL checkpoint hash mismatch")
    write_json_atomic(INPUT / "CONTRACT.json", contract)
    print(json.dumps(contract, indent=2, sort_keys=True))


def load_model(device: torch.device) -> torch.nn.Module:
    sys.path.insert(0, str(END2END))
    from marugoto.marugoto.extract.xiyue_wang.RetCLL import ResNet

    if sha256(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("RetCCL checkpoint hash mismatch")
    model = ResNet.resnet50(
        num_classes=128, mlp=False, two_branch=False, normlinear=True
    )
    state = torch.load(CHECKPOINT, map_location="cpu")
    model.fc = torch.nn.Identity()
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model


def validate_h5(path: Path) -> dict:
    with h5py.File(path, "r") as handle:
        required = {"coords", "feats", "augmented", *QC_NAMES}
        missing = required - set(handle.keys())
        if missing:
            raise ValueError(f"Missing datasets {sorted(missing)} in {path}")
        count, dimension = handle["feats"].shape
        if count < 1 or dimension != 2048:
            raise ValueError(f"Invalid feature matrix {handle['feats'].shape}")
        for name in required - {"feats"}:
            if len(handle[name]) != count:
                raise ValueError(f"Misaligned {name} in {path}")
        score = handle["tile_qc_score"][:]
        if not np.isfinite(score).all() or score.min() < 0 or score.max() > 1:
            raise ValueError(f"Invalid QC scores in {path}")
        if str(handle.attrs.get("extractor", "")) != EXTRACTOR:
            raise ValueError(f"Extractor mismatch in {path}")
        return {
            "tiles": int(count),
            "feature_dimension": int(dimension),
            "qc_score_mean": float(score.mean()),
            "qc_score_min": float(score.min()),
            "qc_score_max": float(score.max()),
            "bytes": int(path.stat().st_size),
        }


def extract_one(
    row: pd.Series,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> dict:
    sys.path.insert(0, str(END2END))
    from wsi_cache_tiles import extract_cached_normalized_tiles, filter_tiles_by_qc

    started = time.time()
    PIL.Image.MAX_IMAGE_PIXELS = None
    with PIL.Image.open(row.source_image) as image:
        image_array = np.asarray(image.convert("RGB"))
    tiles, coords = extract_cached_normalized_tiles(
        image_array,
        tile_size=224,
        black_threshold=8,
        min_occupancy_fraction=0.05,
    )
    del image_array
    tiles, coords, qc = filter_tiles_by_qc(
        tiles,
        coords,
        min_tissue_fraction=0.0,
        min_edge_density=0.0,
        min_focus=0.0,
        black_threshold=8,
    )
    if len(tiles) < 1:
        raise ValueError(f"No occupied tiles for {row.filename}")

    mean = torch.tensor(
        [0.485, 0.456, 0.406], device=device, dtype=torch.float32
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        [0.229, 0.224, 0.225], device=device, dtype=torch.float32
    ).view(1, 3, 1, 1)
    features = []
    with torch.inference_mode():
        for start in range(0, len(tiles), batch_size):
            array = np.ascontiguousarray(tiles[start : start + batch_size])
            batch = torch.from_numpy(array).permute(0, 3, 1, 2)
            batch = batch.to(device=device, dtype=torch.float32, non_blocking=True)
            batch = (batch.div_(255.0) - mean) / std
            features.append(model(batch).to(dtype=torch.float16).cpu().numpy())
    feature_array = np.concatenate(features, axis=0)

    output = Path(row.output_h5)
    temporary = output.with_name(output.name + f".partial.{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    with h5py.File(temporary, "w") as handle:
        handle.create_dataset("coords", data=np.asarray(coords, dtype=np.int32))
        handle.create_dataset("feats", data=feature_array)
        handle.create_dataset(
            "augmented", data=np.zeros(len(feature_array), dtype=bool)
        )
        for name in QC_NAMES:
            handle.create_dataset(name, data=qc[name], compression="gzip")
        handle.attrs["extractor"] = EXTRACTOR
        handle.attrs["tile_qc_version"] = 1
        handle.attrs["source_image"] = str(row.source_image)
        handle.attrs["source_h5"] = str(row.source_h5)
        handle.attrs["retccl_checkpoint_sha256"] = EXPECTED_CHECKPOINT_SHA256
        handle.attrs["cached_black_threshold"] = 8
        handle.attrs["cached_min_occupancy"] = 0.05
    audit = validate_h5(temporary)
    os.replace(temporary, output)
    audit.update({
        "job_index": int(row.job_index),
        "dataset": str(row.dataset),
        "cancers": str(row.cancers),
        "filename": str(row.filename),
        "old_tile_count": int(row.old_tile_count),
        "elapsed_seconds": float(time.time() - started),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    })
    del tiles, feature_array, features, qc
    gc.collect()
    torch.cuda.empty_cache()
    return audit


def worker(
    worker_index: int,
    world_size: int,
    scope: str,
    batch_size: int,
    missing_only: bool = False,
) -> None:
    if not (INPUT / "jobs.tsv").exists():
        raise FileNotFoundError("Run prepare first")
    if not 0 <= worker_index < world_size:
        raise ValueError("worker-index must be in [0, world-size)")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Each worker must see exactly one GPU through CUDA_VISIBLE_DEVICES"
        )
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = load_model(device)
    jobs = pd.read_csv(INPUT / "jobs.tsv", sep="\t")
    if scope == "pilot":
        jobs = jobs.loc[jobs.pilot.astype(str).str.lower().eq("true")].copy()
    if missing_only:
        jobs = jobs.loc[
            ~jobs.output_h5.astype(str).map(lambda value: Path(value).exists())
        ].copy()
    jobs = jobs.sort_values("job_index").reset_index(drop=True)
    jobs = jobs.iloc[worker_index::world_size]

    state_path = LOGS / f"{scope}_worker_{worker_index}.json"
    completed = failed = skipped = 0
    for row in jobs.itertuples(index=False):
        output = Path(row.output_h5)
        state = {
            "status": "running",
            "scope": scope,
            "worker_index": worker_index,
            "world_size": world_size,
            "current_job_index": int(row.job_index),
            "current_filename": str(row.filename),
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json_atomic(state_path, state)
        try:
            if output.exists():
                validate_h5(output)
                skipped += 1
                continue
            audit = extract_one(pd.Series(row._asdict()), model, device, batch_size)
            write_json_atomic(
                LOGS / "completed" / f"{int(row.job_index):05d}.json", audit
            )
            completed += 1
            print(
                f"worker={worker_index} job={row.job_index} "
                f"tiles={audit['tiles']} seconds={audit['elapsed_seconds']:.1f} "
                f"{row.filename}",
                flush=True,
            )
        except Exception as error:
            failed += 1
            failure = {
                **state,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            write_json_atomic(
                LOGS / "failures" / f"{int(row.job_index):05d}.json", failure
            )
            print(
                f"worker={worker_index} job={row.job_index} FAILED: {error}",
                file=sys.stderr,
                flush=True,
            )
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    write_json_atomic(state_path, {
        "status": "complete" if failed == 0 else "complete_with_failures",
        "scope": scope,
        "worker_index": worker_index,
        "world_size": world_size,
        "assigned": int(len(jobs)),
        "completed": completed,
        "failed": failed,
        "skipped": skipped,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    })
    if failed:
        raise RuntimeError(f"Worker {worker_index} had {failed} failures")


def audit(scope: str) -> None:
    jobs = pd.read_csv(INPUT / "jobs.tsv", sep="\t")
    if scope == "pilot":
        jobs = jobs.loc[jobs.pilot.astype(str).str.lower().eq("true")].copy()
    rows, failures = [], []
    for row in jobs.itertuples(index=False):
        path = Path(row.output_h5)
        try:
            value = validate_h5(path)
            rows.append({
                "job_index": int(row.job_index),
                "dataset": row.dataset,
                "cancers": row.cancers,
                "filename": row.filename,
                **value,
            })
        except Exception as error:
            failures.append({
                "job_index": int(row.job_index),
                "filename": row.filename,
                "error": str(error),
            })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame.to_csv(INPUT / f"{scope}_output_audit.tsv", sep="\t", index=False)
    result = {
        "status": "complete" if not failures else "incomplete",
        "scope": scope,
        "expected": int(len(jobs)),
        "complete": int(len(rows)),
        "failures": failures,
        "total_output_bytes": int(frame.bytes.sum()) if not frame.empty else 0,
        "median_tiles": float(frame.tiles.median()) if not frame.empty else None,
        "median_elapsed_seconds": None,
        "audit_pass": len(rows) == len(jobs) and not failures,
    }
    completed = []
    for row in jobs.itertuples(index=False):
        path = LOGS / "completed" / f"{int(row.job_index):05d}.json"
        if path.exists():
            completed.append(json.loads(path.read_text()))
    if completed:
        result["median_elapsed_seconds"] = float(
            np.median([value["elapsed_seconds"] for value in completed])
        )
    write_json_atomic(INPUT / f"{scope.upper()}_AUDIT.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["audit_pass"]:
        raise RuntimeError(f"{scope} extraction audit failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--worker-index", type=int, required=True)
    worker_parser.add_argument("--world-size", type=int, default=8)
    worker_parser.add_argument("--scope", choices=("pilot", "full"), default="full")
    worker_parser.add_argument("--batch-size", type=int, default=256)
    worker_parser.add_argument("--missing-only", action="store_true")
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--scope", choices=("pilot", "full"), required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "worker":
        worker(
            args.worker_index,
            args.world_size,
            args.scope,
            args.batch_size,
            args.missing_only,
        )
    else:
        audit(args.scope)


if __name__ == "__main__":
    main()
