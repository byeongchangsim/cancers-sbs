#!/usr/bin/env python
"""Extract CTransPath features using existing tile coordinates and normalized slide JPEGs."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import h5py
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image


Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[1]
TRANSPATH_ROOT = Path(__file__).resolve().parent
CTRANSPATH_CKPT = Path("/home/end2end-WSI-preprocessing-1.0.0-preprocessing/ctranspath.pth")
FEATURE_DIM = 768
PATCH_SIZE = 224
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def build_model(device: torch.device) -> nn.Module:
    sys.path.insert(0, str(TRANSPATH_ROOT))
    from ctran import ConvStem  # noqa: WPS433
    import timm  # noqa: WPS433
    import timm.models.swin_transformer as swin  # noqa: WPS433

    # The released CTransPath checkpoint uses a ConvStem PatchEmbed. The public
    # ctran.py expects a modified timm; monkey-patching gives the same module
    # names while keeping the installed package stable.
    swin.PatchEmbed = ConvStem
    model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False)
    model.head = nn.Identity()
    checkpoint = torch.load(CTRANSPATH_CKPT, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()
    return model


def output_is_valid(path: Path, expected_rows: int) -> bool:
    if not path.exists():
        return False
    try:
        with h5py.File(path, "r") as handle:
            if handle.attrs.get('coordinate_order', '') != 'xy':
                return False
            if "feats" not in handle or "coords" not in handle:
                return False
            return tuple(handle["feats"].shape) == (expected_rows, FEATURE_DIM) and handle["coords"].shape[0] == expected_rows
    except OSError:
        return False


def load_old_metadata(old_h5: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(old_h5, "r") as handle:
        coords = handle["coords"][:].astype(np.int64, copy=False)
        if "augmented" in handle:
            augmented = handle["augmented"][:].astype(bool, copy=False)
        else:
            augmented = np.zeros((coords.shape[0],), dtype=bool)
    return coords, augmented


def crop_batch(image_array: np.ndarray, coords: np.ndarray) -> np.ndarray:
    batch = np.empty((coords.shape[0], PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
    height, width = image_array.shape[:2]
    # End2End stores (x, y), not NumPy's (row, column).
    for idx, (col_x, row_y) in enumerate(coords):
        y = int(row_y)
        x = int(col_x)
        if x < 0 or y < 0 or x + PATCH_SIZE > width or y + PATCH_SIZE > height:
            raise ValueError(f'Tile coordinate outside image: {(x, y)}/{(width, height)}')
        y2 = min(y + PATCH_SIZE, height)
        x2 = min(x + PATCH_SIZE, width)
        crop = image_array[y:y2, x:x2, :]
        if crop.shape == (PATCH_SIZE, PATCH_SIZE, 3):
            batch[idx] = crop
        else:
            padded = np.full((PATCH_SIZE, PATCH_SIZE, 3), 255, dtype=np.uint8)
            if crop.size:
                padded[: crop.shape[0], : crop.shape[1], :] = crop
            batch[idx] = padded
    return batch


def extract_features(
    model: nn.Module,
    device: torch.device,
    image_array: np.ndarray,
    coords: np.ndarray,
    batch_size: int,
    use_amp: bool,
) -> np.ndarray:
    mean = MEAN.to(device)
    std = STD.to(device)
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, coords.shape[0], batch_size):
            coord_batch = coords[start : start + batch_size]
            crop_np = crop_batch(image_array, coord_batch)
            tensor = torch.from_numpy(crop_np).to(device=device, non_blocking=True)
            tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0)
            tensor = tensor.sub_(mean).div_(std)
            if device.type == "cuda" and use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    feats = model(tensor)
            else:
                feats = model(tensor)
            outputs.append(feats.detach().cpu().numpy().astype(np.float16, copy=False))
    return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, FEATURE_DIM), dtype=np.float16)


def write_h5(out_h5: Path, feats: np.ndarray, coords: np.ndarray, augmented: np.ndarray) -> None:
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_h5.with_name(f"{out_h5.name}.tmp.{os.getpid()}")
    with h5py.File(tmp_path, "w") as handle:
        handle.create_dataset("feats", data=feats, dtype="float16")
        handle.create_dataset("coords", data=coords, dtype="int64")
        handle.create_dataset("augmented", data=augmented.astype(bool, copy=False), dtype="bool")
        handle.attrs["encoder"] = "CTransPath"
        handle.attrs["feature_dim"] = FEATURE_DIM
        handle.attrs["patch_size"] = PATCH_SIZE
        handle.attrs["created_by"] = Path(__file__).name
        handle.attrs['coordinate_order'] = 'xy'
    tmp_path.replace(out_h5)


def process_job(job: pd.Series, model: nn.Module, device: torch.device, batch_size: int, use_amp: bool) -> str:
    slide_id = str(job["slide_id"])
    old_h5 = Path(job["old_h5"])
    norm_slide = Path(job["norm_slide"])
    out_h5 = Path(job["out_h5"])
    coords, augmented = load_old_metadata(old_h5)
    if augmented.any():
        raise ValueError('Augmented features require the same transformed RGB view, not recropping the original mosaic')
    if output_is_valid(out_h5, coords.shape[0]):
        return f"skip existing {slide_id} n={coords.shape[0]}"
    if out_h5.exists():
        raise FileExistsError(f'Use a new output path for corrected coordinates: {out_h5}')
    image_bgr = cv2.imread(str(norm_slide), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"cv2.imread failed: {norm_slide}")
    image_array = image_bgr[:, :, ::-1]
    feats = extract_features(model, device, image_array, coords, batch_size=batch_size, use_amp=use_amp)
    write_h5(out_h5, feats, coords, augmented)
    return f"done {slide_id} tiles={coords.shape[0]} feat_dim={feats.shape[1]}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-list", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)
        # In this CUDA/cuDNN build, small histology patch convolutions on H100
        # can stall in cuDNN. Native CUDA kernels are substantially faster here.
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass
    else:
        device = torch.device("cpu")

    jobs = pd.read_csv(args.job_list, sep="\t")
    jobs = jobs.iloc[args.worker_index :: args.num_workers].reset_index(drop=True)
    if args.limit is not None:
        jobs = jobs.iloc[: args.limit].copy()

    log(
        f"worker={args.worker_index}/{args.num_workers} device={device} "
        f"jobs={len(jobs)} batch_size={args.batch_size}"
    )
    model = build_model(device)
    log("model_loaded strict CTransPath")

    ok = 0
    failed = 0
    started = time.time()
    for row_index, job in jobs.iterrows():
        slide_started = time.time()
        try:
            message = process_job(job, model, device, args.batch_size, use_amp=not args.no_amp)
            ok += 1
            log(f"{row_index + 1}/{len(jobs)} {message} elapsed={time.time() - slide_started:.1f}s")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            log(f"ERROR slide={job.get('slide_id')} error={type(exc).__name__}: {exc}")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    log(f"finished ok={ok} failed={failed} elapsed_min={(time.time() - started) / 60:.1f}")


if __name__ == "__main__":
    main()
