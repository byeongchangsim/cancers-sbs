#!/usr/bin/env python3
"""Rebuild v2.34 figures in the paper_ready_20260817 journal layout.

The SBS panels use the locked fully nested seven-cancer analysis. Clinical
panels retain the v2.33 values, including the mutation-only TCGA polymerase
analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr


BASE = Path("/home/marugoto-1.0.0-regression")
PACKAGE = BASE / "paper_ready_v2_40"
SOURCE = PACKAGE / "source_data"
FIGURE_SOURCE = PACKAGE / "figure_source_data"
PAGE_WIDTH_IN = 170.5 / 25.4

TCGA = "#2F67B1"
CPTAC = "#168C83"
HIGHLIGHT = "#D85C14"
PURPLE = "#8C4B7F"
RED = "#C83E4D"
INK = "#1D2939"
MUTED = "#64748B"
GRID = "#DCE5EE"
PALE = "#F4F7FA"
GREY = "#AEB9C5"
NEGATIVE = "#D9E1E8"
ROW = "#F7F9FB"
GOLD = "#D79A00"

CANCERS = ["BRCA", "HNSC", "KIRC", "LUAD", "LUSC", "OV", "UCEC"]
SIGNATURES = [f"SBS{index}" for index in range(1, 31)]
CANCER_COLORS = ["#E486A8", "#7F63B8", "#3CB878", "#4AA1D5", "#E48A35", "#B458A0", "#EDB62D"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=PACKAGE)
    return parser.parse_args()


def configure_style() -> None:
    font_dir = Path("/usr/share/fonts/truetype/liberation")
    for name in (
        "LiberationSans-Regular.ttf",
        "LiberationSans-Bold.ttf",
        "LiberationSans-Italic.ttf",
        "LiberationSans-BoldItalic.ttf",
    ):
        path = font_dir / name
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
    mpl.rcParams.update(
        {
            "font.family": "Liberation Sans",
            "font.size": 7.5,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.6,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.fontsize": 7.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel(ax: plt.Axes, letter: str, x: float = -0.10, y: float = 1.03) -> None:
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=11.5,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )


def clean(ax: plt.Axes, grid: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)


def add_rows(ax: plt.Axes, count: int) -> None:
    for index in range(count):
        if index % 2 == 0:
            ax.axhspan(index - 0.48, index + 0.48, color=ROW, zorder=0)


def p_label(value: float) -> str:
    if value < 0.001:
        return rf"$P$={value:.1e}"
    return rf"$P$={value:.3f}"


def fisher_ci(r_value: float, n_value: int) -> tuple[float, float]:
    clipped = float(np.clip(r_value, -0.999999999, 0.999999999))
    z_value = np.arctanh(clipped)
    delta = norm.ppf(0.975) / np.sqrt(n_value - 3)
    return float(np.tanh(z_value - delta)), float(np.tanh(z_value + delta))


def save_figure(fig: plt.Figure, output: Path, name: str, supplementary: bool = False) -> None:
    raster_dir = output / ("figures_supp" if supplementary else "figures_main")
    pdf_dir = output / "vector" / ("supp_pdf" if supplementary else "main_pdf")
    svg_dir = output / "vector" / ("supp_svg" if supplementary else "main_svg")
    for directory in (raster_dir, pdf_dir, svg_dir):
        directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_dir / f"{name}.pdf", format="pdf", bbox_inches="tight", pad_inches=0.03, facecolor="white")
    fig.savefig(svg_dir / f"{name}.svg", format="svg", bbox_inches="tight", pad_inches=0.03, facecolor="white")
    fig.savefig(raster_dir / f"{name}.png", format="png", dpi=400, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    fig.savefig(
        raster_dir / f"{name}.tif",
        format="tiff",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.03,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def load_prediction(cancer: str, signature: str, dataset: str) -> pd.DataFrame:
    path = FIGURE_SOURCE / f"{cancer.lower()}_{signature.lower()}_{dataset.lower()}.tsv"
    if not path.is_file():
        path = BASE / "full210_exact25_multiscale_fusion_20260904/nested_regularized_stack/external_results/predictions" / path.name
    frame = pd.read_csv(path, sep="\t")
    return frame.rename(columns={signature: "observed", "prediction": "predicted"})


def scatter_regression(
    ax: plt.Axes,
    frame: pd.DataFrame,
    color: str,
    title: str,
    x_label: str = "Observed proportional exposure",
    y_label: str = "WSI-derived score",
    positive_direction: bool = False,
) -> tuple[float, float]:
    x_values = frame["observed"].to_numpy(float)
    y_values = frame["predicted"].to_numpy(float)
    r_value, p_value = pearsonr(x_values, y_values)
    if positive_direction:
        p_value = p_value / 2 if r_value >= 0 else 1 - p_value / 2
    ax.scatter(
        x_values,
        y_values,
        s=11,
        color=color,
        alpha=0.42,
        edgecolor="white",
        linewidth=0.20,
        rasterized=True,
        zorder=2,
    )
    if np.ptp(x_values) > 0:
        coefficient = np.polyfit(x_values, y_values, 1)
        x_line = np.linspace(float(np.min(x_values)), float(np.max(x_values)), 150)
        ax.plot(x_line, coefficient[0] * x_line + coefficient[1], color=INK, linewidth=1.35, zorder=3)
    ax.set_title(title, loc="left", fontsize=8.1, fontweight="bold", pad=7)
    ax.text(
        0.02,
        0.98,
        f"n={len(frame)}   $r$={r_value:.3f}   {p_label(float(p_value))}",
        transform=ax.transAxes,
        va="top",
        fontsize=6.3,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    clean(ax, "both")
    return float(r_value), float(p_value)


def forest_plot(
    ax: plt.Axes,
    frame: pd.DataFrame,
    dataset: str,
    color: str,
    p_column: bool = False,
) -> None:
    ordered = frame.sort_values(f"pearson_r_{dataset}", ascending=False).reset_index(drop=True)
    values = ordered[f"pearson_r_{dataset}"].to_numpy(float)
    intervals = np.array(
        [fisher_ci(row[f"pearson_r_{dataset}"], int(row[f"n_{dataset}"])) for _, row in ordered.iterrows()]
    )
    y_values = np.arange(len(ordered))
    add_rows(ax, len(ordered))
    ax.errorbar(
        values,
        y_values,
        xerr=np.vstack([values - intervals[:, 0], intervals[:, 1] - values]),
        fmt="none",
        ecolor=INK,
        elinewidth=0.95,
        capsize=2.3,
        zorder=2,
    )
    # Encode cohort consistently. No individual SBS pair receives a unique
    # colour merely because it is discussed in the text.
    ax.scatter(values, y_values, s=36, c=color, edgecolor="white", linewidth=0.55, zorder=3)
    ax.set_yticks(y_values, ordered.pair)
    ax.invert_yaxis()
    ax.axvline(0, color=MUTED, linewidth=0.65, linestyle=(0, (3, 2)))
    lower = min(-0.03, float(intervals[:, 0].min()) - 0.02)
    upper = float(intervals[:, 1].max()) + (0.10 if p_column else 0.035)
    ax.set_xlim(lower, upper)
    ax.set_xlabel(f"{dataset} Pearson $r$ (95% CI)")
    if p_column:
        for row_y, row in enumerate(ordered.itertuples(index=False)):
            field = f"pearson_p_positive_direction_{dataset}"
            ax.text(upper - 0.005, row_y, p_label(float(getattr(row, field))), ha="right", va="center", fontsize=5.1, color=MUTED)
    clean(ax, "x")


def result_matrix(frame: pd.DataFrame, dataset: str, only_pairs: set[str] | None = None) -> np.ndarray:
    matrix = np.full((len(CANCERS), len(SIGNATURES)), np.nan)
    subset = frame[frame.dataset.eq(dataset)]
    if only_pairs is not None:
        subset = subset[subset.pair.isin(only_pairs)]
    for row in subset.itertuples(index=False):
        matrix[CANCERS.index(row.cancer), SIGNATURES.index(row.signature)] = float(row.pearson_r)
    return matrix


def draw_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    norm_value: Normalize,
    cohort: str,
    marks: list[tuple[int, int, str]],
    show_x: bool,
    cmap_name: str = "RdBu_r",
) -> None:
    cmap = mpl.colormaps[cmap_name].copy()
    cmap.set_bad("#E8ECEF")
    ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=cmap, norm=norm_value, interpolation="none")
    ax.set_yticks(range(len(CANCERS)), CANCERS)
    if show_x:
        ax.set_xticks(range(30), SIGNATURES, rotation=55, ha="right", fontsize=5.4)
        ax.set_xlabel("SBS target")
    else:
        ax.set_xticks([])
    ax.set_ylabel("Cancer")
    ax.set_title(cohort, loc="left", fontsize=7.8, fontweight="bold", pad=6)
    for row, column, marker in marks:
        ax.scatter(
            column,
            row,
            s=45 if marker == "o" else 65,
            marker=marker,
            facecolor="none" if marker == "o" else GOLD,
            edgecolor=GOLD if marker == "o" else INK,
            linewidth=1.1 if marker == "o" else 0.5,
            zorder=3,
        )
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)


def signal_bubble_matrix(ax: plt.Axes, paired: pd.DataFrame) -> Normalize:
    """Compact overview using only SBS targets represented among the reported pairs."""
    present_signatures = sorted(
        paired.signature.unique(), key=lambda value: int(str(value).replace("SBS", ""))
    )
    x_lookup = {signature: index for index, signature in enumerate(present_signatures)}
    y_lookup = {cancer: index for index, cancer in enumerate(CANCERS)}
    maximum = max(0.35, float(paired.pearson_r_TCGA.max()))
    norm_value = Normalize(vmin=0, vmax=maximum)

    ax.set_xlim(-0.55, len(present_signatures) - 0.45)
    ax.set_ylim(len(CANCERS) - 0.45, -0.55)
    for index in range(len(CANCERS)):
        if index % 2 == 0:
            ax.axhspan(index - 0.48, index + 0.48, color=ROW, zorder=0)
    ax.set_xticks(range(len(present_signatures)), present_signatures, rotation=45, ha="right")
    ax.set_yticks(range(len(CANCERS)), CANCERS)
    ax.set_xticks(np.arange(-0.5, len(present_signatures), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(CANCERS), 1), minor=True)
    ax.grid(which="minor", color=GRID, linewidth=0.55)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlabel(f"SBS target represented among the {len(paired)} associations")
    ax.set_ylabel("Cancer")
    ax.set_title(f"TCGA effect map for the {len(paired)} cross-cohort associations", loc="left", fontsize=8.4, fontweight="bold", pad=7)

    x_values = np.array([x_lookup[value] for value in paired.signature], dtype=float)
    y_values = np.array([y_lookup[value] for value in paired.cancer], dtype=float)
    effects = paired.pearson_r_TCGA.to_numpy(float)
    sizes = 70 + 250 * np.square(effects / maximum)
    ax.scatter(
        x_values,
        y_values,
        s=sizes,
        c=effects,
        cmap="Blues",
        norm=norm_value,
        edgecolor="white",
        linewidth=0.9,
        zorder=3,
    )
    # An empty OV row is intentionally retained: OV was evaluated as part of
    # the 210-pair landscape but contributed no highlighted pair.
    ax.text(
        len(present_signatures) - 0.60,
        y_lookup["OV"],
        "no highlighted pair",
        ha="right",
        va="center",
        fontsize=5.5,
        color=MUTED,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
    return norm_value


def figure1(summary: pd.DataFrame, paired: pd.DataFrame, output: Path) -> None:
    counts = summary.groupby(["cancer", "dataset"], as_index=False).n.first()
    count_lookup = {(row.cancer, row.dataset): int(row.n) for row in counts.itertuples(index=False)}
    totals = {dataset: sum(count_lookup[(cancer, dataset)] for cancer in CANCERS) for dataset in ("TCGA", "CPTAC")}

    fig = plt.figure(figsize=(PAGE_WIDTH_IN, 7.45))
    grid = fig.add_gridspec(
        4,
        1,
        height_ratios=[0.29, 0.26, 0.16, 0.29],
        left=0.035,
        right=0.98,
        top=0.975,
        bottom=0.035,
        hspace=0.30,
    )

    ax_a = fig.add_subplot(grid[0])
    panel(ax_a, "a", -0.018, 1.03)
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    ax_a.axis("off")
    centers = [0.10, 0.30, 0.50, 0.70, 0.90]
    titles = ["WSI digitisation", "Tissue filtering", "Tessellation", "Feature extraction", "Feature matrix"]
    captions = [
        "diagnostic H&E slides",
        "Canny tissue filtering\nMacenko normalisation",
        "non-overlapping\n224 × 224-px tiles",
        "frozen RetCCL + CTransPath\n2,048 + 768 dimensions",
        "patient bags\n1,024 and 2,048 tiles",
    ]
    for x_value, title, caption in zip(centers, titles, captions):
        ax_a.text(x_value, 0.96, title, ha="center", va="top", fontweight="bold", fontsize=7.2)
        ax_a.text(x_value, 0.05, caption, ha="center", va="bottom", color=MUTED, fontsize=6.2, linespacing=0.95)
    for x_value in (0.08, 0.10, 0.12):
        ax_a.add_patch(Rectangle((x_value - 0.065, 0.37 + (x_value - 0.10) * 0.8), 0.12, 0.25, angle=4, facecolor="#E9EEF3", edgecolor=MUTED, linewidth=0.7))
    ax_a.add_patch(mpl.patches.Ellipse((0.105, 0.50), 0.055, 0.085, facecolor="#C77CA6", edgecolor="#A95C88", linewidth=0.6))
    rng = np.random.default_rng(20260817)
    tissue = np.clip(rng.normal([0.78, 0.53, 0.68], [0.11, 0.13, 0.12], size=(42, 42, 3)), 0, 1)
    tissue[rng.random((42, 42)) < 0.16] = 0.96
    ax_a.imshow(tissue, extent=(0.235, 0.365, 0.29, 0.70), interpolation="bilinear", zorder=1)
    ax_a.add_patch(Rectangle((0.235, 0.29), 0.13, 0.41, fill=False, edgecolor=INK, linewidth=0.7, zorder=2))
    tile = np.clip(rng.normal([0.82, 0.58, 0.72], [0.13, 0.12, 0.11], size=(24, 24, 3)), 0, 1)
    for offset in (0.024, 0.012, 0.0):
        ax_a.imshow(tile, extent=(0.448 + offset, 0.548 + offset, 0.36 + offset, 0.65 + offset), interpolation="bilinear", zorder=3-offset)
        ax_a.add_patch(Rectangle((0.448 + offset, 0.36 + offset), 0.10, 0.29, fill=False, edgecolor=HIGHLIGHT, linewidth=0.55))
    cube_colors = ["#ECD6E0", "#DED0E5", "#9B86CF"]
    for index, (offset, color) in enumerate(zip((-0.055, 0.0, 0.055), cube_colors)):
        ax_a.add_patch(Rectangle((0.695 + offset, 0.34 + 0.03 * index), 0.07, 0.28, facecolor=color, edgecolor=INK, linewidth=0.65))
        ax_a.add_patch(Rectangle((0.765 + offset, 0.36 + 0.03 * index), 0.015, 0.28, facecolor=mpl.colors.to_rgba(color, 0.75), edgecolor=INK, linewidth=0.55))
    for index in range(10):
        x_value = 0.845 + index * 0.012
        height = 0.22 + 0.07 * ((index * 7) % 5) / 4
        ax_a.add_patch(Rectangle((x_value, 0.48 - height / 2), 0.008, height, facecolor="#8BD0CB", edgecolor=CPTAC, linewidth=0.4))
    for left, right in zip(centers[:-1], centers[1:]):
        ax_a.add_patch(FancyArrowPatch((left + 0.075, 0.50), (right - 0.075, 0.50), arrowstyle="-|>", mutation_scale=8, linewidth=0.8, color=MUTED))
    # imshow defaults to an equal data aspect; restore the wide schematic axis.
    ax_a.set_aspect("auto")

    ax_b = fig.add_subplot(grid[1])
    panel(ax_b, "b", -0.018, 1.03)
    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(0, 1)
    ax_b.axis("off")
    ax_b.text(0.02, 0.90, "Patient bag", fontweight="bold", fontsize=7.2)
    bag = rng.uniform(0, 1, size=(7, 6))
    ax_b.imshow(bag, cmap="GnBu", extent=(0.03, 0.095, 0.42, 0.76), interpolation="nearest")
    blocks = [
        (0.15, 0.43, 0.12, 0.34, "Bag\nencoder", "#EEF1F5", MUTED),
        (0.37, 0.62, 0.12, 0.24, "Feature\nembedding", "#EEF1F5", MUTED),
        (0.37, 0.32, 0.12, 0.22, "Attention\nnetwork", "#FFF0E5", HIGHLIGHT),
        (0.68, 0.43, 0.11, 0.34, "Patient\nvector", "#E5F5F3", CPTAC),
        (0.83, 0.43, 0.11, 0.34, "Regression\nhead", "#F2DCE7", PURPLE),
    ]
    for x_value, y_value, width, height, text_value, face, edge in blocks:
        ax_b.add_patch(Rectangle((x_value, y_value), width, height, facecolor=face, edgecolor=edge, linewidth=0.8))
        ax_b.text(x_value + width / 2, y_value + height / 2, text_value, ha="center", va="center", fontweight="bold", fontsize=6.6)
    arrows = [((0.095, 0.59), (0.15, 0.59)), ((0.27, 0.59), (0.37, 0.74)), ((0.27, 0.59), (0.37, 0.43)), ((0.49, 0.74), (0.61, 0.62)), ((0.49, 0.43), (0.61, 0.55)), ((0.63, 0.59), (0.68, 0.59)), ((0.79, 0.59), (0.83, 0.59)), ((0.94, 0.59), (0.985, 0.59))]
    for start, end in arrows:
        ax_b.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8, linewidth=0.8, color=MUTED))
    ax_b.text(0.615, 0.60, "Σ", ha="center", va="center", fontsize=12, fontweight="bold")
    ax_b.text(0.985, 0.59, "ŷ", ha="center", va="center", color=PURPLE, fontsize=10, fontweight="bold")
    ax_b.plot([0.02, 0.98], [0.20, 0.20], color=GRID, linewidth=0.8)
    pipeline = ["Somatic MAF", "deconstructSigs", "Proportional SBS1–SBS30", "Cancer-specific model", "Regression loss"]
    positions = [0.05, 0.25, 0.48, 0.72, 0.92]
    for x_value, text_value in zip(positions, pipeline):
        ax_b.text(x_value, 0.07, text_value, ha="center", va="center", fontweight="bold", fontsize=6.3)
    # Use label-specific clearances: the proportional-SBS and cancer-model
    # labels are wider than the other steps and were previously crossed by
    # the fixed-offset arrows.
    pipeline_arrows = [
        (0.115, 0.170),
        (0.325, 0.382),
        (0.585, 0.635),
        (0.815, 0.850),
    ]
    for left, right in pipeline_arrows:
        ax_b.add_patch(FancyArrowPatch((left, 0.07), (right, 0.07), arrowstyle="-|>", mutation_scale=7, linewidth=0.7, color=MUTED))
    ax_b.text(0.53, 0.88, "Fixed 1:1:1 average: PCA-ridge + 1,024/2,048-tile gated-attention MIL", ha="center", fontsize=6.2, color=MUTED)
    ax_b.set_aspect("auto")

    ax_c = fig.add_subplot(grid[2])
    panel(ax_c, "c", -0.018, 1.03)
    ax_c.set_xlim(0, 1)
    ax_c.set_ylim(0, 1)
    ax_c.axis("off")
    evaluation = [
        ("Cross-validated prediction", "five-fold TCGA × three seeds", TCGA),
        ("Fixed model average", "1:1:1; no fitted weights", HIGHLIGHT),
        ("Model selection", "training data only", GOLD),
        ("External cohort", "frozen predictions in CPTAC", CPTAC),
    ]
    for index, (title, subtitle, color) in enumerate(evaluation):
        x_value = index * 0.25 + 0.015
        ax_c.add_patch(Rectangle((x_value, 0.18), 0.22, 0.60, facecolor="white", edgecolor=GRID, linewidth=0.8))
        ax_c.add_patch(Rectangle((x_value, 0.18), 0.012, 0.60, facecolor=color, edgecolor="none"))
        ax_c.text(x_value + 0.03, 0.59, title, fontweight="bold", fontsize=6.8)
        ax_c.text(x_value + 0.03, 0.34, subtitle, color=MUTED, fontsize=5.9, wrap=True)

    ax_d = fig.add_subplot(grid[3])
    panel(ax_d, "d", -0.018, 1.03)
    ax_d.set_xlim(0, 1)
    ax_d.set_ylim(0, 1)
    ax_d.axis("off")
    for center_x, dataset in ((0.15, "TCGA"), (0.38, "CPTAC")):
        values = np.array([count_lookup[(cancer, dataset)] for cancer in CANCERS], dtype=float)
        # Use most of the panel height so the cohort composition is readable
        # at journal-column scale while remaining clear of the right legend.
        donut = ax_d.inset_axes([center_x - 0.15, 0.04, 0.30, 0.90])
        donut.pie(
            values,
            colors=CANCER_COLORS,
            startangle=90,
            wedgeprops={"width": 0.30, "edgecolor": "white", "linewidth": 0.7},
        )
        donut.text(0, 0.10, dataset, ha="center", va="center", color=TCGA if dataset == "TCGA" else CPTAC, fontweight="bold", fontsize=7.0)
        donut.text(0, -0.20, f"n={totals[dataset]:,}", ha="center", va="center", color=MUTED, fontsize=5.8)
        donut.set_axis_off()
    ax_d.text(0.56, 0.85, "Matched WSI–genomic inventory", fontweight="bold", fontsize=7.5)
    for index, cancer in enumerate(CANCERS):
        column = index % 2
        row = index // 2
        x_value = 0.56 + 0.23 * column
        y_value = 0.68 - 0.18 * row
        ax_d.scatter(x_value, y_value, s=28, color=CANCER_COLORS[index])
        ax_d.text(x_value + 0.025, y_value, cancer, va="center", fontweight="bold", fontsize=6.8)
        ax_d.text(x_value + 0.10, y_value, f"{count_lookup[(cancer, 'TCGA')]} / {count_lookup[(cancer, 'CPTAC')]}", va="center", color=MUTED, fontsize=6.3)
    ax_d.text(0.56, 0.02, f"7 cancers × 30 SBS = 210 pairs · {len(paired)} positive-direction $P$<0.05 in both", color=MUTED, fontsize=6.1)
    save_figure(fig, output, "Figure_1")


def figure2(summary: pd.DataFrame, paired: pd.DataFrame, output: Path) -> None:
    # The complete 210-pair landscape is retained in Supplementary Figure S1.
    # The main figure shows only the locations of the 13 positive-direction
    # signals, avoiding a redundant field of 210 bubbles.
    fig = plt.figure(figsize=(PAGE_WIDTH_IN, 7.65))
    grid = fig.add_gridspec(3, 1, height_ratios=[0.25, 0.30, 0.45], left=0.12, right=0.965, top=0.97, bottom=0.065, hspace=0.42)

    ax_a = fig.add_subplot(grid[0])
    panel(ax_a, "a", -0.08, 1.06)
    color_norm = signal_bubble_matrix(ax_a, paired)
    colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=color_norm, cmap=mpl.colormaps["Blues"]), ax=ax_a, fraction=0.017, pad=0.012)
    colorbar.set_label("TCGA Pearson $r$")

    ax_b = fig.add_subplot(grid[1])
    panel(ax_b, "b", -0.08, 1.04)
    forest_plot(ax_b, paired, "TCGA", TCGA)

    holder = fig.add_subplot(grid[2])
    holder.axis("off")
    sub = grid[2].subgridspec(1, 2, wspace=0.27)
    ax_c = fig.add_subplot(sub[0])
    ax_d = fig.add_subplot(sub[1])
    panel(ax_c, "c", -0.13, 1.04)
    panel(ax_d, "d", -0.13, 1.04)
    scatter_regression(ax_c, load_prediction("BRCA", "SBS3", "TCGA"), TCGA, "BRCA–SBS3 · TCGA", positive_direction=True)
    scatter_regression(ax_d, load_prediction("LUAD", "SBS4", "TCGA"), TCGA, "LUAD–SBS4 · TCGA", positive_direction=True)
    save_figure(fig, output, "Figure_2")


def figure3(summary: pd.DataFrame, paired: pd.DataFrame, output: Path) -> None:
    tcga = summary[summary.dataset.eq("TCGA")][["pair", "cancer", "signature", "pearson_r"]].rename(columns={"pearson_r": "pearson_r_TCGA"})
    cptac = summary[summary.dataset.eq("CPTAC")][["pair", "pearson_r"]].rename(columns={"pearson_r": "pearson_r_CPTAC"})
    wide = tcga.merge(cptac, on="pair", validate="one_to_one")
    estimable = int(wide[["pearson_r_TCGA", "pearson_r_CPTAC"]].notna().all(axis=1).sum())
    paired_set = set(paired.pair)

    fig = plt.figure(figsize=(PAGE_WIDTH_IN, 6.75))
    grid = fig.add_gridspec(2, 2, height_ratios=[0.46, 0.54], width_ratios=[0.39, 0.61], left=0.075, right=0.975, top=0.965, bottom=0.07, hspace=0.34, wspace=0.34)
    ax_a = fig.add_subplot(grid[0, 0])
    panel(ax_a, "a", -0.13, 1.04)
    for row in wide.itertuples(index=False):
        if row.pair in paired_set:
            color, size = CPTAC, 36
        else:
            color, size = GREY, 20
        ax_a.scatter(row.pearson_r_TCGA, row.pearson_r_CPTAC, s=size, color=color, edgecolor="white", linewidth=0.50, alpha=0.95 if row.pair in paired_set else 0.50, zorder=3)
    low = min(float(wide.pearson_r_TCGA.min()), float(wide.pearson_r_CPTAC.min())) - 0.035
    high = max(float(wide.pearson_r_TCGA.max()), float(wide.pearson_r_CPTAC.max())) + 0.035
    ax_a.plot([low, high], [low, high], color=MUTED, linestyle=(0, (3, 3)), linewidth=0.75)
    ax_a.axhline(0, color=GRID, linewidth=0.65)
    ax_a.axvline(0, color=GRID, linewidth=0.65)
    label_positions = {
        "UCEC–SBS6": (0.00, 0.32),
        "BRCA–SBS3": (0.34, 0.27),
        "LUAD–SBS4": (0.27, 0.10),
    }
    for label, text_position in label_positions.items():
        row = wide[wide.pair.eq(label)].iloc[0]
        ax_a.annotate(
            label,
            (row.pearson_r_TCGA, row.pearson_r_CPTAC),
            xytext=text_position,
            textcoords="data",
            fontsize=5.5,
            fontweight="bold",
            arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.45},
        )
    ax_a.set_xlabel("TCGA Pearson $r$")
    ax_a.set_ylabel("CPTAC Pearson $r$")
    ax_a.text(
        0.03,
        0.04,
        f"{estimable}/210 estimable pairs\n{len(paired)} positive-direction signals",
        transform=ax_a.transAxes,
        color=MUTED,
        fontsize=6.2,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
        zorder=5,
    )
    clean(ax_a)

    ax_b = fig.add_subplot(grid[0, 1])
    panel(ax_b, "b", -0.08, 1.04)
    forest_plot(ax_b, paired, "CPTAC", CPTAC)

    holder = fig.add_subplot(grid[1, :])
    holder.axis("off")
    sub = grid[1, :].subgridspec(1, 2, wspace=0.27)
    ax_c = fig.add_subplot(sub[0])
    ax_d = fig.add_subplot(sub[1])
    panel(ax_c, "c", -0.13, 1.04)
    panel(ax_d, "d", -0.13, 1.04)
    scatter_regression(ax_c, load_prediction("BRCA", "SBS3", "CPTAC"), CPTAC, "BRCA–SBS3 · CPTAC", positive_direction=True)
    scatter_regression(ax_d, load_prediction("UCEC", "SBS6", "CPTAC"), CPTAC, "UCEC–SBS6 · CPTAC", positive_direction=True)
    save_figure(fig, output, "Figure_3")


def zscore(values: np.ndarray) -> np.ndarray:
    standard_deviation = float(np.std(values, ddof=0))
    if not np.isfinite(standard_deviation) or standard_deviation == 0:
        raise ValueError("Cannot standardize a constant score")
    return (values - float(np.mean(values))) / standard_deviation


def add_bracket(ax: plt.Axes, x1: float, x2: float, y_value: float, text_value: str) -> None:
    height = 0.10
    ax.plot([x1, x1, x2, x2], [y_value, y_value + height, y_value + height, y_value], color=INK, linewidth=0.75, clip_on=False)
    ax.text((x1 + x2) / 2, y_value + height + 0.03, text_value, ha="center", va="bottom", fontsize=5.9)


def distribution_panel(
    ax: plt.Axes,
    scores: pd.DataFrame,
    specifications: list[tuple[str, str]],
    title: str,
    positive_color: str,
    p_lookup: dict[tuple[str, str], float],
    show_ylabel: bool,
) -> None:
    rng = np.random.default_rng(20260812)
    arrays: list[np.ndarray] = []
    positions: list[float] = []
    colors: list[str] = []
    labels: list[str] = []
    brackets: list[tuple[float, float, float]] = []
    all_values: list[float] = []
    cursor = 0.0
    for cohort, endpoint in specifications:
        frame = scores[scores.cohort.eq(cohort) & scores.endpoint.eq(endpoint)].copy()
        standardized = zscore(frame.score.to_numpy(float))
        pair_positions = (cursor, cursor + 0.72)
        labels_array = frame.label.to_numpy(int)
        for status, position, fill, suffix in ((0, pair_positions[0], NEGATIVE, "−"), (1, pair_positions[1], positive_color, "+")):
            values = standardized[labels_array == status]
            arrays.append(values)
            positions.append(position)
            colors.append(fill)
            all_values.extend(values.tolist())
            labels.append(f"{'TCGA' if cohort.startswith('TCGA') else 'CPTAC'}\n{suffix}")
        brackets.append((pair_positions[0], pair_positions[1], p_lookup[(cohort, endpoint)]))
        cursor += 1.95
    boxes = ax.boxplot(
        arrays,
        positions=positions,
        widths=0.50,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": INK, "linewidth": 0.95},
        whiskerprops={"color": INK, "linewidth": 0.75},
        capprops={"color": INK, "linewidth": 0.75},
        boxprops={"edgecolor": MUTED, "linewidth": 0.75},
    )
    for patch, fill in zip(boxes["boxes"], colors):
        patch.set_facecolor(fill)
        patch.set_alpha(0.86)
    for values, position, fill in zip(arrays, positions, colors):
        point_color = MUTED if fill == NEGATIVE else fill
        ax.scatter(position + rng.normal(0, 0.055, size=len(values)), values, s=5.2, color=point_color, alpha=0.22, edgecolor="none", rasterized=True)
    top = max(all_values) + 0.18
    for index, (x1, x2, p_value) in enumerate(brackets):
        add_bracket(ax, x1, x2, top + index * 0.30, p_label(p_value))
    ax.set_ylim(min(all_values) - 0.22, top + len(brackets) * 0.30 + 0.20)
    ax.set_xticks(positions, labels)
    ax.set_title(title, loc="left", fontsize=8.0, fontweight="bold", pad=7)
    if show_ylabel:
        ax.set_ylabel("Within-cohort standardised WSI score")
    clean(ax, "y")


def clinical_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tcga_metrics = pd.read_csv(SOURCE / "TCGA_nested_clinical_metrics.tsv", sep="\t")
    cptac_metrics = pd.read_csv(SOURCE / "CPTAC_locked_clinical_metrics.tsv", sep="\t")
    # CPTAC contained no qualifying POLD1-positive patient. Its polymerase
    # analysis is retained separately as a POLE-only sensitivity analysis and
    # is not shown as validation of the combined POLE/POLD1 endpoint.
    cptac_metrics = cptac_metrics[cptac_metrics.analysis.ne("POLE/POLD1")].copy()
    metrics = pd.concat([tcga_metrics, cptac_metrics], ignore_index=True)
    tcga = pd.read_csv(SOURCE / "TCGA_nested_clinical_predictions.tsv", sep="\t")
    tcga["cohort"] = np.where(tcga.endpoint.eq("HRD"), "TCGA BRCA", "TCGA UCEC")
    cptac = pd.read_csv(SOURCE / "CPTAC_locked_clinical_predictions.tsv", sep="\t")
    cptac = cptac[cptac.endpoint.ne("POLE/POLD1")].copy()
    cptac["cohort"] = "CPTAC UCEC"
    scores = pd.concat([tcga, cptac], ignore_index=True, sort=False)
    auc = metrics[metrics.metric.eq("ROC AUC")].copy()
    order = [
        ("HRD >=42", "TCGA BRCA"),
        ("MSI/MMR", "TCGA UCEC"),
        ("MSI/MMR", "CPTAC UCEC"),
        ("POLE/POLD1", "TCGA UCEC"),
    ]
    auc["order"] = [order.index((row.analysis, row.cohort)) for row in auc.itertuples(index=False)]
    auc = auc.sort_values("order").reset_index(drop=True)
    return metrics, scores, auc


def figure4(output: Path) -> None:
    metrics, scores, auc = clinical_data()
    fig = plt.figure(figsize=(PAGE_WIDTH_IN, 7.05))
    grid = fig.add_gridspec(2, 1, height_ratios=[0.40, 0.60], left=0.075, right=0.975, top=0.965, bottom=0.07, hspace=0.36)
    top = grid[0].subgridspec(1, 2, width_ratios=[0.38, 0.62], wspace=0.50)

    ax_a = fig.add_subplot(top[0])
    panel(ax_a, "a", -0.13, 1.04)
    hrd = scores[scores.endpoint.eq("HRD") & scores.cohort.eq("TCGA BRCA")].copy()
    correlation = metrics[metrics.analysis.eq("HRD") & metrics.metric.eq("Pearson r")].iloc[0]
    ax_a.scatter(hrd.score, hrd.hrd_score, s=11, color=TCGA, alpha=0.30, edgecolor="white", linewidth=0.15, rasterized=True)
    coefficient = np.polyfit(hrd.score, hrd.hrd_score, 1)
    x_line = np.linspace(float(hrd.score.min()), float(hrd.score.max()), 150)
    ax_a.plot(x_line, coefficient[0] * x_line + coefficient[1], color=INK, linewidth=1.35)
    ax_a.text(
        0.03,
        0.96,
        f"n={int(correlation.n)}   $r$={correlation.estimate:.3f}\n95% CI {correlation.ci_low:.3f}–{correlation.ci_high:.3f}   {p_label(correlation.p_value)}",
        transform=ax_a.transAxes,
        va="top",
        fontsize=6.2,
    )
    ax_a.set_xlabel("Primary WSI-derived SBS3 score")
    ax_a.set_ylabel("Continuous HRD score")
    clean(ax_a, "both")

    ax_b = fig.add_subplot(top[1])
    panel(ax_b, "b", -0.09, 1.04)
    # Keep the external CPTAC MSI/MMR result in panel c and in the manuscript,
    # but omit it from the compact panel-b TCGA AUC summary.
    auc_panel = auc[
        ~(
            auc.analysis.eq("MSI/MMR")
            & auc.cohort.eq("CPTAC UCEC")
        )
    ].reset_index(drop=True)
    labels = [
        "TCGA BRCA\nHRD",
        "TCGA UCEC\nMSI/MMR",
        "TCGA UCEC\n" + r"$\it{POLE}$/$\it{POLD1}$",
    ]
    colors = [TCGA, CPTAC, PURPLE]
    y_values = np.arange(len(auc_panel))
    add_rows(ax_b, len(auc_panel))
    values = auc_panel.estimate.to_numpy(float)
    ax_b.errorbar(values, y_values, xerr=np.vstack([values - auc_panel.ci_low, auc_panel.ci_high - values]), fmt="none", ecolor=INK, elinewidth=0.95, capsize=2.5, zorder=2)
    ax_b.scatter(values, y_values, s=40, c=colors, edgecolor=INK, linewidth=0.40, zorder=3)
    ax_b.set_yticks(y_values, labels)
    ax_b.tick_params(axis="y", labelsize=6.2, pad=3)
    ax_b.invert_yaxis()
    ax_b.set_ylim(len(auc_panel) - 0.55, -0.55)
    ax_b.axvline(0.5, color=MUTED, linestyle=(0, (3, 2)), linewidth=0.75)
    ax_b.set_xlim(0.44, 1.11)
    ax_b.set_xlabel("AUC (95% CI)")
    for row_y, row in enumerate(auc_panel.itertuples(index=False)):
        ax_b.text(1.095, row_y, p_label(float(row.p_value)), ha="right", va="center", fontsize=5.9)
    clean(ax_b, "x")

    holder = fig.add_subplot(grid[1])
    panel(holder, "c", -0.055, 1.04)
    holder.axis("off")
    sub = grid[1].subgridspec(1, 3, width_ratios=[0.27, 0.365, 0.365], wspace=0.28)
    p_lookup = {(str(row.cohort), "HRD" if str(row.analysis).startswith("HRD") else str(row.analysis)): float(row.p_value) for row in auc.itertuples(index=False)}
    distribution_panel(fig.add_subplot(sub[0]), scores, [("TCGA BRCA", "HRD")], "HRD · primary SBS3 score", TCGA, p_lookup, True)
    distribution_panel(fig.add_subplot(sub[1]), scores, [("TCGA UCEC", "MSI/MMR")], "MSI/MMR · primary SBS6 score", CPTAC, p_lookup, False)
    distribution_panel(fig.add_subplot(sub[2]), scores, [("TCGA UCEC", "POLE/POLD1")], r"$\it{POLE}$/$\it{POLD1}$ · primary SBS10 score", PURPLE, p_lookup, False)
    save_figure(fig, output, "Figure_4")


def figure_s1(summary: pd.DataFrame, paired: pd.DataFrame, output: Path) -> None:
    tcga_matrix = result_matrix(summary, "TCGA")
    cptac_matrix = result_matrix(summary, "CPTAC")
    maximum = max(abs(float(np.nanmin(tcga_matrix))), abs(float(np.nanmax(tcga_matrix))), abs(float(np.nanmin(cptac_matrix))), abs(float(np.nanmax(cptac_matrix))))
    norm_value = TwoSlopeNorm(vmin=-maximum, vcenter=0, vmax=maximum)
    tcga_significant = summary[
        summary.dataset.eq("TCGA") & (summary.pearson_r > 0)
        & (summary.pearson_p_positive_direction < 0.05)
    ]
    cptac_significant = summary[
        summary.dataset.eq("CPTAC") & (summary.pearson_r > 0)
        & (summary.pearson_p_positive_direction < 0.05)
    ]
    tcga_marks = [(CANCERS.index(row.cancer), SIGNATURES.index(row.signature), "*") for row in tcga_significant.itertuples(index=False)]
    cptac_marks = [(CANCERS.index(row.cancer), SIGNATURES.index(row.signature), "*") for row in cptac_significant.itertuples(index=False)]
    fig = plt.figure(figsize=(PAGE_WIDTH_IN, 4.15))
    grid = fig.add_gridspec(2, 1, left=0.075, right=0.92, top=0.96, bottom=0.15, hspace=0.36)
    axes = [fig.add_subplot(grid[0]), fig.add_subplot(grid[1])]
    panel(axes[0], "a", -0.075, 1.05)
    panel(axes[1], "b", -0.075, 1.05)
    draw_heatmap(axes[0], tcga_matrix, norm_value, "TCGA", tcga_marks, False)
    draw_heatmap(axes[1], cptac_matrix, norm_value, "CPTAC", cptac_marks, True)
    color_ax = fig.add_axes([0.935, 0.25, 0.014, 0.50])
    colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm_value, cmap=mpl.colormaps["RdBu_r"]), cax=color_ax)
    colorbar.set_label("Pearson $r$")
    axes[1].legend(handles=[Line2D([0], [0], marker="*", markersize=8, markerfacecolor=GOLD, markeredgecolor=INK, markeredgewidth=0.4, linestyle="none", label="Cohort-specific positive-direction $P$<0.05")], loc="lower right", bbox_to_anchor=(1.0, 1.04), frameon=False, fontsize=6.2, borderaxespad=0)
    save_figure(fig, output, "Figure_S1", True)


def figure_s2(summary: pd.DataFrame, paired: pd.DataFrame, output: Path) -> None:
    paired_set = set(paired.pair)
    fig = plt.figure(figsize=(PAGE_WIDTH_IN, 7.10))
    grid = fig.add_gridspec(1, 2, width_ratios=[0.42, 0.58], left=0.11, right=0.95, top=0.96, bottom=0.08, wspace=0.34)
    ax_a = fig.add_subplot(grid[0])
    panel(ax_a, "a", -0.16, 1.03)
    forest_plot(ax_a, paired, "TCGA", TCGA, True)

    right = grid[1].subgridspec(2, 1, hspace=0.26)
    tcga_matrix = result_matrix(summary, "TCGA", paired_set)
    cptac_matrix = result_matrix(summary, "CPTAC", paired_set)
    maximum = max(abs(float(np.nanmin(tcga_matrix))), abs(float(np.nanmax(tcga_matrix))), abs(float(np.nanmin(cptac_matrix))), abs(float(np.nanmax(cptac_matrix))))
    norm_value = TwoSlopeNorm(vmin=-maximum, vcenter=0, vmax=maximum)
    marks = [(CANCERS.index(row.cancer), SIGNATURES.index(row.signature), "o") for row in paired.itertuples(index=False)]
    ax_b = fig.add_subplot(right[0])
    ax_c = fig.add_subplot(right[1])
    panel(ax_b, "b", -0.10, 1.05)
    panel(ax_c, "c", -0.10, 1.05)
    draw_heatmap(ax_b, tcga_matrix, norm_value, "TCGA", marks, False)
    draw_heatmap(ax_c, cptac_matrix, norm_value, "CPTAC", marks, True)
    color_ax = fig.add_axes([0.963, 0.28, 0.012, 0.44])
    colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm_value, cmap=mpl.colormaps["RdBu_r"]), cax=color_ax)
    colorbar.set_label("Pearson $r$")
    save_figure(fig, output, "Figure_S2", True)


def selected_prediction(cancer: str, signature: str, source_name: str) -> pd.DataFrame:
    script_dir = PACKAGE / "reproducibility"
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from select_process_linked_clinical_scores import candidate_predictions

    frame = candidate_predictions(cancer, signature, "TCGA")
    return frame[["observed", source_name]].rename(columns={source_name: "predicted"})


def figure_s3(output: Path) -> None:
    metrics, scores, auc = clinical_data()
    exposure = pd.read_csv(SOURCE / "selected_score_exposure_metrics.tsv", sep="\t")
    selections = pd.read_csv(SOURCE / "TCGA_nested_outer_fold_selection.tsv", sep="\t")
    fig, axes = plt.subplots(2, 3, figsize=(PAGE_WIDTH_IN, 5.25))
    fig.subplots_adjust(left=0.075, right=0.98, top=0.95, bottom=0.10, hspace=0.48, wspace=0.34)

    specifications = [
        ("a", "BRCA", "SBS3", "retatt_score", TCGA),
        ("b", "UCEC", "SBS6", "global_score", CPTAC),
        ("c", "UCEC", "SBS10", "ret4_score", PURPLE),
    ]
    for ax, (letter, cancer, signature, source_name, color) in zip(axes[0], specifications):
        panel(ax, letter, -0.14, 1.05)
        scatter_regression(ax, selected_prediction(cancer, signature, source_name), color, f"{cancer}–{signature} · TCGA")

    ax_d = axes[1, 0]
    panel(ax_d, "d", -0.14, 1.05)
    hrd_auc = auc[auc.analysis.eq("HRD >=42") & auc.cohort.eq("TCGA BRCA")].iloc[0]
    distribution_panel(ax_d, scores, [("TCGA BRCA", "HRD")], "TCGA BRCA HRD-high", TCGA, {("TCGA BRCA", "HRD"): float(hrd_auc.p_value)}, True)

    ax_e = axes[1, 1]
    panel(ax_e, "e", -0.14, 1.05)
    previous = np.array([0.6801260022909505, 0.7093612010597585, 0.6209523016731875])
    order = ["HRD >=42", "MSI/MMR", "POLE/POLD1"]
    nested = np.array([metrics[metrics.analysis.eq(value) & metrics.cohort.str.startswith("TCGA")].iloc[0].estimate for value in order])
    x_values = np.arange(3)
    width = 0.36
    ax_e.bar(x_values - width / 2, previous, width, color=NEGATIVE, label="Previous")
    ax_e.bar(x_values + width / 2, nested, width, color=TCGA, label="Cross-validated")
    ax_e.axhline(0.5, color=MUTED, linewidth=0.75, linestyle=(0, (3, 2)))
    ax_e.set_xticks(x_values, ["HRD", "MSI/MMR", "POLE/POLD1"], rotation=20, ha="right")
    ax_e.set_ylabel("ROC AUC")
    ax_e.set_ylim(0.50, 0.77)
    ax_e.set_title("TCGA AUC comparison", loc="left", fontsize=7.5, fontweight="bold", pad=6)
    ax_e.legend(frameon=False, fontsize=5.9)
    clean(ax_e, "y")

    ax_f = axes[1, 2]
    panel(ax_f, "f", -0.14, 1.05)
    endpoints = ["HRD", "MSI/MMR", "POLE/POLD1"]
    values = [selections[selections.endpoint.eq(endpoint)].legacy_weight.to_numpy(float) for endpoint in endpoints]
    violin = ax_f.violinplot(values, positions=np.arange(3), showextrema=False, widths=0.72)
    for body, color in zip(violin["bodies"], [TCGA, CPTAC, PURPLE]):
        body.set_facecolor(color)
        body.set_edgecolor(MUTED)
        body.set_alpha(0.55)
    for index, value in enumerate(values):
        ax_f.scatter(np.full(len(value), index), value, color=INK, s=18, zorder=3)
    ax_f.set_xticks(np.arange(3), endpoints, rotation=20, ha="right")
    ax_f.set_ylabel("Legacy-score weight")
    ax_f.set_ylim(0, 1)
    ax_f.set_title("Cross-validation weights", loc="left", fontsize=7.5, fontweight="bold", pad=6)
    clean(ax_f, "y")
    save_figure(fig, output, "Figure_S3", True)


def main() -> None:
    args = parse_args()
    configure_style()
    summary = pd.read_csv(SOURCE / "standardized_210_nested_results.tsv", sep="\t")
    paired = pd.read_csv(SOURCE / "positive_directional_in_both.tsv", sep="\t")
    figure1(summary, paired, args.output_root)
    figure2(summary, paired, args.output_root)
    figure3(summary, paired, args.output_root)
    figure4(args.output_root)
    figure_s1(summary, paired, args.output_root)
    figure_s2(summary, paired, args.output_root)
    print(f"Built 4 main and 2 supplementary figures under {args.output_root}")


if __name__ == "__main__":
    main()
