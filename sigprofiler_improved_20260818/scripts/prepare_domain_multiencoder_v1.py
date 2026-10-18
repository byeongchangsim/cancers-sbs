#!/opt/anaconda/envs/cenv/bin/python
"""Prepare and audit run-local CTransPath completion for the 24-task cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import pandas as pd


RUN = Path("/home/marugoto-1.0.0-regression/sigprofiler_improved_20260818")
END2END = Path("/home/end2end-WSI-preprocessing-1.0.0-preprocessing")
INPUT = RUN / "inputs/domain_multiencoder_v1"
OUTPUT = RUN / "features_end2end_ctranspath_completion_v1"
JOBS = RUN / "inputs/end2end_retccl_qc_v1/jobs.tsv"
CHECKPOINT = END2END / "ctranspath.pth"
EXPECTED_SHA256 = "7c998680060c8743551a412583fac689db43cec07053b72dfec6dcd810113539"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def existing_root(dataset: str) -> Path:
    return END2END / (
        "cache_testa_ctranspath" if dataset == "TCGA" else "cache_cptac_ctranspath"
    )


def completion_path(dataset: str, filename: str) -> Path:
    return OUTPUT / dataset / f"{filename}.h5"


def valid(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with h5py.File(path, "r") as handle:
            return (
                "coords" in handle
                and "feats" in handle
                and handle["feats"].ndim == 2
                and handle["feats"].shape[0] == handle["coords"].shape[0]
                and handle["feats"].shape[1] == 768
                and handle["feats"].shape[0] > 0
            )
    except OSError:
        return False


def prepare() -> None:
    if INPUT.exists() or OUTPUT.exists():
        raise FileExistsError("Refusing to overwrite domain_multiencoder_v1 inputs/features")
    if sha256(CHECKPOINT) != EXPECTED_SHA256:
        raise ValueError("CTransPath checkpoint hash mismatch")
    source = pd.read_csv(JOBS, sep="\t", dtype={"filename": str})
    rows, resolved = [], []
    for row in source.itertuples(index=False):
        old = existing_root(row.dataset) / f"{row.filename}.h5"
        new = completion_path(row.dataset, row.filename)
        if valid(old):
            selected, origin = old, "preexisting_audited"
        else:
            selected, origin = new, "run_local_completion"
            rows.append({
                "slide_id": row.filename,
                "dataset": row.dataset,
                # The audited QC-aware RetCCL file supplies the exact tile rows
                # and coordinates to be encoded by CTransPath.
                "old_h5": row.output_h5,
                "norm_slide": row.source_image,
                "out_h5": str(new),
            })
        resolved.append({
            "dataset": row.dataset,
            "cancers": row.cancers,
            "filename": row.filename,
            "ctranspath_h5": str(selected),
            "origin": origin,
            "retccl_qc_h5": row.output_h5,
        })
    INPUT.mkdir(parents=True)
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "TCGA").mkdir()
    (OUTPUT / "CPTAC").mkdir()
    pd.DataFrame(rows).to_csv(INPUT / "ctranspath_completion_jobs.tsv", sep="\t", index=False)
    pd.DataFrame(resolved).to_csv(INPUT / "resolved_features.tsv", sep="\t", index=False)
    contract = {
        "status": "prepared",
        "analysis": "End2End RetCCL plus CTransPath dual-encoder completion",
        "slides": len(resolved),
        "preexisting_ctranspath": sum(x["origin"] == "preexisting_audited" for x in resolved),
        "completion_jobs": len(rows),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "feature_dimension": 768,
        "tile_size": 224,
        "completion_coordinates": "audited row-aligned RetCCL-QC coordinates",
        "cptac_outcomes_opened": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (INPUT / "CONTRACT.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    print(json.dumps(contract, indent=2, sort_keys=True))


def audit() -> None:
    resolved = pd.read_csv(INPUT / "resolved_features.tsv", sep="\t", dtype=str)
    failures, rows = [], []
    for row in resolved.itertuples(index=False):
        path = Path(row.ctranspath_h5)
        try:
            if not valid(path):
                raise ValueError("invalid or missing H5")
            with h5py.File(path, "r") as handle:
                rows.append({
                    "dataset": row.dataset,
                    "cancers": row.cancers,
                    "filename": row.filename,
                    "origin": row.origin,
                    "tiles": int(handle["feats"].shape[0]),
                })
        except Exception as error:  # noqa: BLE001
            failures.append({"path": str(path), "error": str(error)})
    frame = pd.DataFrame(rows)
    result = {
        "status": "complete" if not failures else "incomplete",
        "expected": len(resolved),
        "complete": len(rows),
        "preexisting": int((frame.origin == "preexisting_audited").sum()) if len(frame) else 0,
        "completed_locally": int((frame.origin == "run_local_completion").sum()) if len(frame) else 0,
        "failures": failures,
        "checkpoint_sha256": sha256(CHECKPOINT),
        "audit_pass": len(rows) == len(resolved) and not failures,
    }
    frame.to_csv(INPUT / "feature_audit.tsv", sep="\t", index=False)
    (INPUT / "AUDIT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["audit_pass"]:
        raise RuntimeError("CTransPath completion audit failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "audit"))
    command = parser.parse_args().command
    prepare() if command == "prepare" else audit()


if __name__ == "__main__":
    main()
