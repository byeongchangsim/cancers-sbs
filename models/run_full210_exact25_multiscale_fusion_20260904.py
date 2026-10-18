#!/opt/anaconda/envs/cenv/bin/python
"""Exact-25-epoch multiscale RetCCL+CTransPath training for all 210 SBS pairs.

TCGA is the only cohort used for fitting, checkpoint selection, architecture
selection, and later stacking.  CPTAC feature bags may be scored by a frozen
checkpoint, but CPTAC SBS outcomes are not read by this program.  The program
adapts the already-audited legacy30 multi-encoder implementation while fixing
the Marugoto training contract at 25 epochs and patience 8.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import threading
from pathlib import Path
from queue import Empty, Queue

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE / "full210_exact25_multiscale_fusion_20260904"
SOURCE_1024 = BASE / "models/run_feature_fusion.py"
SOURCE_2048 = BASE / "models/run_feature_fusion_2048.py"
CACHE_1024 = BASE / "legacy30_retccl_ctranspath_feature_fusion_20260829/fixed_bag_cache"
CACHE_2048 = BASE / "legacy30_retccl_ctranspath_feature_fusion_bag2048_20260829/fixed_bag_cache"
LABEL_ROOT = BASE / "process_endpoint_inputs_legacy30_min50"
PYTHON = "/opt/anaconda/envs/cenv/bin/python"
SCALES = ("1024", "2048")
MAX_EPOCHS = 25
PATIENCE = 8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def implementation(scale: str):
    if scale not in SCALES:
        raise ValueError(scale)
    source = SOURCE_1024 if scale == "1024" else SOURCE_2048
    wrapper = import_file(source, f"exact25_wrapper_{scale}")
    module = wrapper.load_implementation()
    module.ROOT = ROOT / f"scale{scale}"
    module.CACHE = CACHE_1024 if scale == "1024" else CACHE_2048
    module.BAG_SIZE = int(scale)
    module.BATCH_SIZE = 20 if scale == "1024" else 10
    module.MAX_EPOCHS = MAX_EPOCHS
    module.PATIENCE = PATIENCE
    module.WORKER_ARGS = ("--scale", scale)
    module.__file__ = str(Path(__file__).resolve())
    return module


def result_path(module, variant: str, cancer: str, seed_index: int, fold: int) -> Path:
    return module.output_dir(variant, cancer, seed_index, fold) / "RESULT.json"


def register() -> None:
    if ROOT.exists():
        raise FileExistsError(ROOT)
    ROOT.mkdir()
    module = implementation("1024")
    counts = {}
    input_hashes = {
        "driver": sha256(Path(__file__).resolve()),
        "source_1024": sha256(SOURCE_1024),
        "source_2048": sha256(SOURCE_2048),
    }
    for cancer in module.CANCERS:
        cancer_label = module.CANCER_LABEL[cancer]
        tcga_meta = pd.read_csv(
            (CACHE_1024 / "qhybrid" / f"TCGA_{cancer}_bag1024").with_suffix(".meta.csv")
        )
        cptac_meta = pd.read_csv(
            (CACHE_1024 / "qhybrid" / f"CPTAC_{cancer}_bag1024").with_suffix(".meta.csv")
        )
        counts[cancer_label] = {"TCGA": int(len(tcga_meta)), "CPTAC": int(len(cptac_meta))}
        input_hashes[f"TCGA_{cancer_label}_labels"] = sha256(
            LABEL_ROOT / "TCGA" / cancer / f"TCGA_{cancer}_process_endpoints.xlsx"
        )
        # Only feature metadata, never the CPTAC outcome workbook, is registered here.
        input_hashes[f"CPTAC_{cancer_label}_feature_meta"] = sha256(
            (CACHE_1024 / "qhybrid" / f"CPTAC_{cancer}_bag1024").with_suffix(".meta.csv")
        )
    protocol = {
        "status": "locked_before_training",
        "scope": "7 cancers x SBS1-SBS30 (210 pre-specified pairs)",
        "target": "continuous deconstructSigs legacy30_min50 proportional exposures",
        "training_cohort": "TCGA only",
        "external_validation_cohort": "CPTAC only after all TCGA decisions are frozen",
        "cohorts_pooled": False,
        "architecture": "coordinate-aligned RetCCL+CTransPath multi-output target-attention MIL",
        "scales": [1024, 2048],
        "variants": list(module.VARIANTS),
        "outer_cv": "patient-disjoint shuffled 5-fold within TCGA, repeated for 3 fixed seeds",
        "inner_validation": "20% of each TCGA outer-training partition",
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "checkpoint_rule": "maximum inner-validation median Pearson r across eligible SBS1-SBS30",
        "variant_rule": "within each scale/cancer/seed/fold, maximum stored inner-validation checkpoint score",
        "variant_selection_uses_outer_test": False,
        "variant_selection_uses_cptac": False,
        "pair_specific_model_selection": False,
        "same_model_predicts_all_30_signatures_within_context": True,
        "patient_counts": counts,
        "input_sha256": input_hashes,
    }
    (ROOT / "LOCKED_PROTOCOL.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    print(ROOT / "LOCKED_PROTOCOL.json")


def train_fold(scale: str, variant: str, cancer: str, seed_index: int, fold: int, device: str) -> None:
    module = implementation(scale)
    module.train_fold(variant, cancer, seed_index, fold, device)


def launch(gpus: str) -> None:
    if not (ROOT / "LOCKED_PROTOCOL.json").exists():
        raise FileNotFoundError("Run register first")
    gpu_ids = [part.strip() for part in gpus.split(",") if part.strip()]
    module = implementation("1024")
    jobs: Queue[tuple[str, str, str, int, int]] = Queue()
    for scale in SCALES:
        scale_module = implementation(scale)
        for variant in scale_module.VARIANTS:
            for cancer in scale_module.CANCERS:
                for seed_index in range(3):
                    for fold in range(5):
                        if not result_path(scale_module, variant, cancer, seed_index, fold).exists():
                            jobs.put((scale, variant, cancer, seed_index, fold))
    initial_jobs = jobs.qsize()
    failures = []
    lock = threading.Lock()
    log_root = ROOT / "logs"
    log_root.mkdir(exist_ok=True)

    def worker(gpu: str) -> None:
        while True:
            try:
                scale, variant, cancer, seed_index, fold = jobs.get_nowait()
            except Empty:
                return
            log = log_root / f"scale{scale}_{variant}_{cancer}_seed{seed_index}_fold{fold}.log"
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            command = [
                PYTHON, str(Path(__file__).resolve()), "train-fold",
                "--scale", scale, "--variant", variant, "--cancer", cancer,
                "--seed-index", str(seed_index), "--fold", str(fold), "--device", "cuda:0",
            ]
            with log.open("w") as handle:
                completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=env)
            if completed.returncode:
                with lock:
                    failures.append({
                        "scale": scale, "variant": variant, "cancer": cancer,
                        "seed_index": seed_index, "fold": fold, "gpu": gpu,
                        "returncode": completed.returncode, "log": str(log),
                    })
                    (ROOT / "FAILURES_RUNNING.json").write_text(
                        json.dumps(failures, indent=2) + "\n"
                    )
            jobs.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpu_ids]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    (ROOT / "FAILURES.json").write_text(json.dumps(failures, indent=2) + "\n")
    completed_jobs = 0
    for scale in SCALES:
        scale_module = implementation(scale)
        for variant in scale_module.VARIANTS:
            for cancer in scale_module.CANCERS:
                for seed_index in range(3):
                    for fold in range(5):
                        completed_jobs += int(result_path(scale_module, variant, cancer, seed_index, fold).exists())
    audit = {
        "status": "training_complete" if not failures and completed_jobs == 840 else "training_incomplete",
        "jobs_started_this_launch": initial_jobs,
        "expected_total_jobs": 840,
        "completed_total_jobs": completed_jobs,
        "failures": failures,
        "cptac_outcomes_read": False,
    }
    (ROOT / "TRAINING_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    if audit["status"] != "training_complete":
        raise RuntimeError("Training audit failed")


def freeze_selection() -> None:
    protocol = json.loads((ROOT / "LOCKED_PROTOCOL.json").read_text())
    if protocol["max_epochs"] != MAX_EPOCHS or protocol["patience"] != PATIENCE:
        raise RuntimeError("Training protocol mismatch")
    rows = []
    for scale in SCALES:
        module = implementation(scale)
        for cancer in module.CANCERS:
            for seed_index in range(3):
                for fold in range(5):
                    candidates = []
                    for variant in module.VARIANTS:
                        path = result_path(module, variant, cancer, seed_index, fold)
                        result = json.loads(path.read_text())
                        history = pd.read_csv(path.parent / "history.tsv", sep="\t")
                        if int(result["best_epoch"]) > MAX_EPOCHS or len(history) > MAX_EPOCHS:
                            raise RuntimeError(f"Epoch contract failed: {path}")
                        candidates.append((float(result["best_validation_median_r"]), variant, path))
                    score, variant, path = sorted(candidates, key=lambda x: (-x[0], x[1]))[0]
                    checkpoint = path.parent / "best.pt"
                    rows.append({
                        "scale": int(scale), "cancer": module.CANCER_LABEL[cancer],
                        "cancer_dir": cancer, "seed_index": seed_index, "fold": fold,
                        "selected_variant": variant, "inner_validation_median_r": score,
                        "checkpoint_sha256": sha256(checkpoint),
                    })
    selection = pd.DataFrame(rows)
    if len(selection) != 210:
        raise RuntimeError(f"Expected 210 scale-context selections, got {len(selection)}")
    path = ROOT / "FROZEN_INNER_SELECTION.tsv"
    if path.exists():
        raise FileExistsError(path)
    selection.to_csv(path, sep="\t", index=False)
    audit = {
        "status": "selection_frozen_before_CPTAC_outcomes",
        "rows": len(selection),
        "rule": protocol["variant_rule"],
        "outer_test_outcomes_used": False,
        "cptac_outcomes_used": False,
        "selection_sha256": sha256(path),
    }
    (ROOT / "FROZEN_SELECTION_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("register")
    train = sub.add_parser("train-fold")
    train.add_argument("--scale", required=True, choices=SCALES)
    train.add_argument("--variant", required=True)
    train.add_argument("--cancer", required=True)
    train.add_argument("--seed-index", required=True, type=int, choices=(0, 1, 2))
    train.add_argument("--fold", required=True, type=int, choices=range(5))
    train.add_argument("--device", default="cuda:0")
    launch_parser = sub.add_parser("launch")
    launch_parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    sub.add_parser("freeze-selection")
    args = parser.parse_args()
    if args.command == "register":
        register()
    elif args.command == "train-fold":
        train_fold(args.scale, args.variant, args.cancer, args.seed_index, args.fold, args.device)
    elif args.command == "launch":
        launch(args.gpus)
    elif args.command == "freeze-selection":
        freeze_selection()


if __name__ == "__main__":
    main()
