from __future__ import annotations

from typing import Optional

import numpy as np


def extract_cached_normalized_tiles(
    image_array: np.ndarray,
    tile_size: int = 224,
    *,
    black_threshold: int = 8,
    min_occupancy_fraction: float = 0.05,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Recover populated tiles and coordinates from a cached JPEG mosaic.

    A strict ``np.any`` test mistakes near-black JPEG compression residue for
    tissue.  Occupancy above a small intensity threshold is therefore used.
    """
    image_array = np.asarray(image_array)
    if image_array.ndim != 3:
        raise ValueError("Expected an H x W x C image array")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if not 0 <= black_threshold <= 255:
        raise ValueError("black_threshold must be between 0 and 255")
    if not 0 <= min_occupancy_fraction <= 1:
        raise ValueError("min_occupancy_fraction must be between 0 and 1")

    height, width, channels = image_array.shape
    tiles: list[np.ndarray] = []
    coords: list[tuple[int, int]] = []
    for y in range(0, height - tile_size + 1, tile_size):
        for x in range(0, width - tile_size + 1, tile_size):
            patch = image_array[y : y + tile_size, x : x + tile_size, :]
            occupied = np.max(patch, axis=-1) > black_threshold
            if float(occupied.mean()) >= min_occupancy_fraction:
                tiles.append(patch)
                coords.append((x, y))

    if not tiles:
        return (
            np.empty((0, tile_size, tile_size, channels), dtype=image_array.dtype),
            coords,
        )
    return np.stack(tiles), coords


def compute_tile_qc(
    tiles: np.ndarray, *, black_threshold: int = 8
) -> dict[str, np.ndarray]:
    """Calculate aligned, bounded QC measurements for RGB tiles."""
    array = np.asarray(tiles)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError("Expected an N x H x W x 3 tile array")
    if array.shape[1] < 2 or array.shape[2] < 2:
        raise ValueError("Tile height and width must both be at least 2 pixels")
    if not 0 <= black_threshold <= 255:
        raise ValueError("black_threshold must be between 0 and 255")
    if len(array) == 0:
        empty = np.empty(0, dtype=np.float32)
        return {
            "tile_qc_occupancy": empty.copy(),
            "tile_qc_tissue_fraction": empty.copy(),
            "tile_qc_edge_density": empty.copy(),
            "tile_qc_focus": empty.copy(),
            "tile_qc_score": empty.copy(),
        }

    rgb = array.astype(np.float32)
    high = rgb.max(axis=-1)
    low = rgb.min(axis=-1)
    brightness = rgb.mean(axis=-1)
    saturation = high - low
    occupancy = (high > black_threshold).mean(axis=(1, 2))
    tissue = (
        (high > black_threshold)
        & (brightness < 242.0)
        & ((saturation > 12.0) | (brightness < 205.0))
    ).mean(axis=(1, 2))

    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    grad_x = np.abs(np.diff(gray, axis=2))
    grad_y = np.abs(np.diff(gray, axis=1))
    edge_density = 0.5 * (
        (grad_x > 12.0).mean(axis=(1, 2))
        + (grad_y > 12.0).mean(axis=(1, 2))
    )
    gradient_energy = 0.5 * (
        np.square(grad_x).mean(axis=(1, 2))
        + np.square(grad_y).mean(axis=(1, 2))
    )
    focus = 1.0 - np.exp(-gradient_energy / 400.0)
    edge_component = np.clip(edge_density / 0.15, 0.0, 1.0)
    score = (
        0.20 * occupancy
        + 0.45 * tissue
        + 0.20 * edge_component
        + 0.15 * focus
    )
    return {
        "tile_qc_occupancy": occupancy.astype(np.float32),
        "tile_qc_tissue_fraction": tissue.astype(np.float32),
        "tile_qc_edge_density": edge_density.astype(np.float32),
        "tile_qc_focus": focus.astype(np.float32),
        "tile_qc_score": score.astype(np.float32),
    }


def filter_tiles_by_qc(
    tiles: np.ndarray,
    coords: list[tuple[int, int]],
    *,
    min_tissue_fraction: float = 0.0,
    min_edge_density: float = 0.0,
    min_focus: float = 0.0,
    max_tiles: Optional[int] = None,
    black_threshold: int = 8,
) -> tuple[np.ndarray, list[tuple[int, int]], dict[str, np.ndarray]]:
    """Filter/rank tiles once and keep image, coordinate, and QC rows aligned."""
    array = np.asarray(tiles)
    if len(array) != len(coords):
        raise ValueError("tile and coordinate counts differ")
    for name, value in {
        "min_tissue_fraction": min_tissue_fraction,
        "min_edge_density": min_edge_density,
        "min_focus": min_focus,
    }.items():
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")
    if max_tiles is not None and max_tiles <= 0:
        raise ValueError("max_tiles must be positive when provided")

    metrics = compute_tile_qc(array, black_threshold=black_threshold)
    keep = (
        (metrics["tile_qc_tissue_fraction"] >= min_tissue_fraction)
        & (metrics["tile_qc_edge_density"] >= min_edge_density)
        & (metrics["tile_qc_focus"] >= min_focus)
    )
    selected = np.flatnonzero(keep)
    if max_tiles is not None and len(selected) > max_tiles:
        ranked = selected[
            np.lexsort((selected, -metrics["tile_qc_score"][selected]))[:max_tiles]
        ]
        selected = np.sort(ranked)

    if len(selected) == len(array) and np.array_equal(
        selected, np.arange(len(array), dtype=selected.dtype)
    ):
        # Avoid a second, potentially multi-gigabyte tile-array copy when QC
        # is being recorded without excluding any of the occupied tiles.
        return array, list(coords), metrics

    filtered_metrics = {name: values[selected] for name, values in metrics.items()}
    filtered_coords = [coords[int(index)] for index in selected]
    return array[selected], filtered_coords, filtered_metrics
