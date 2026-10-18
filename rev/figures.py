#!/usr/bin/env python3
"""Render main Figures 1–3 and provide the shared Figure 4 layout."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr


BASE = Path(__file__).resolve().parents[1]
SOURCE = BASE / "rev"
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
    parser.add_argument("--output-root", type=Path, default=BASE / "results")
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


def save_figure(fig: plt.Figure, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03, facecolor="white")
    fig.savefig(output / f"{name}.tif", format="tiff", dpi=600,
                bbox_inches="tight", pad_inches=0.03, facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def load_prediction(cancer: str, signature: str, dataset: str) -> pd.DataFrame:
    frame = pd.read_csv(SOURCE / "patient_scores.tsv", sep="\t")
    frame = frame[frame.cancer.eq(cancer) & frame.cohort.eq(dataset)]
    return frame[["PATIENT", signature, "prediction_" + signature]].rename(
        columns={signature: "observed", "prediction_" + signature: "predicted"})


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
    ax_b.text(0.02, 0.92, "Patient features", fontweight="bold", fontsize=7.0)
    bag = rng.uniform(0, 1, size=(7, 6))
    ax_b.imshow(bag, cmap="GnBu", extent=(0.025, 0.085, 0.43, 0.72), interpolation="nearest")
    ax_b.text(0.255, 0.92, "Component predictions", ha="center", color=MUTED, fontsize=6.1)
    ax_b.text(0.72, 0.92, "Across-model averaging", ha="center", color=MUTED, fontsize=6.1)

    components = [
        (0.70, "PCA–ridge\nRetCCL patient-mean features", "#EEF1F5", MUTED),
        (0.51, "Gated-attention MIL · 1,024 tiles\nattention-weighted tile pooling", "#FFF0E5", HIGHLIGHT),
        (0.32, "Gated-attention MIL · 2,048 tiles\nattention-weighted tile pooling", "#FFF0E5", HIGHLIGHT),
    ]
    for y_value, label, face, edge in components:
        ax_b.add_patch(Rectangle((0.13, y_value), 0.25, 0.14, facecolor=face, edgecolor=edge, linewidth=0.8))
        ax_b.text(0.255, y_value + 0.07, label, ha="center", va="center", fontweight="bold", fontsize=5.5, linespacing=0.95)
        ax_b.add_patch(FancyArrowPatch((0.085, 0.575), (0.13, y_value + 0.07), arrowstyle="-|>", mutation_scale=7, linewidth=0.7, color=MUTED))
        ax_b.add_patch(Rectangle((0.45, y_value + 0.01), 0.12, 0.12, facecolor="#F4F7FA", edgecolor=MUTED, linewidth=0.7))
        ax_b.text(0.51, y_value + 0.07, "Inner-validation\nstandardisation", ha="center", va="center", fontsize=5.2, linespacing=0.95)
        ax_b.add_patch(FancyArrowPatch((0.38, y_value + 0.07), (0.45, y_value + 0.07), arrowstyle="-|>", mutation_scale=7, linewidth=0.7, color=MUTED))

    ax_b.add_patch(Rectangle((0.64, 0.46), 0.15, 0.24, facecolor="#E5F5F3", edgecolor=CPTAC, linewidth=0.85))
    ax_b.text(0.715, 0.58, "Fixed 1:1:1\narithmetic mean", ha="center", va="center", fontweight="bold", fontsize=6.0)
    for y_value, *_ in components:
        ax_b.add_patch(FancyArrowPatch((0.57, y_value + 0.07), (0.64, 0.58), arrowstyle="-|>", mutation_scale=7, linewidth=0.7, color=MUTED))

    ax_b.add_patch(Rectangle((0.85, 0.46), 0.13, 0.24, facecolor="#F2DCE7", edgecolor=PURPLE, linewidth=0.85))
    ax_b.text(0.915, 0.58, "30 WSI-derived\nSBS scores", ha="center", va="center", fontweight="bold", fontsize=6.0)
    ax_b.add_patch(FancyArrowPatch((0.79, 0.58), (0.85, 0.58), arrowstyle="-|>", mutation_scale=8, linewidth=0.8, color=MUTED))
    ax_b.text(0.55, 0.235, "Attention pooling occurs within each MIL model and the 1:1:1 mean combines model outputs.", ha="center", color=MUTED, fontsize=5.4)

    ax_b.plot([0.02, 0.98], [0.18, 0.18], color=GRID, linewidth=0.8)
    pipeline = ["Somatic MAF", "deconstructSigs", "Proportional SBS1–SBS30", "Cancer-specific model", "Regression loss"]
    positions = [0.05, 0.25, 0.48, 0.72, 0.92]
    for x_value, text_value in zip(positions, pipeline):
        ax_b.text(x_value, 0.065, text_value, ha="center", va="center", fontweight="bold", fontsize=6.3)
    pipeline_arrows = [(0.115, 0.170), (0.325, 0.382), (0.585, 0.635), (0.815, 0.850)]
    for left, right in pipeline_arrows:
        ax_b.add_patch(FancyArrowPatch((left, 0.065), (right, 0.065), arrowstyle="-|>", mutation_scale=7, linewidth=0.7, color=MUTED))
    ax_b.set_aspect("auto")

    ax_c = fig.add_subplot(grid[2])
    panel(ax_c, "c", -0.018, 1.03)
    ax_c.set_xlim(0, 1)
    ax_c.set_ylim(0, 1)
    ax_c.axis("off")
    evaluation = [
        ("Cross-validated prediction", "five-fold TCGA × three seeds", TCGA),
        ("Fixed model average", "1:1:1, no fitted weights", HIGHLIGHT),
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
    # Highlight the 14 associations with nominal positive correlations in both cohorts.
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


def figure4(output: Path, metrics: pd.DataFrame, scores: pd.DataFrame, auc: pd.DataFrame) -> None:
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
    # Figure 4 shows the three primary TCGA clinical associations.
    auc_panel = auc[
        ~(
            auc.analysis.eq("MSI/MMR")
            & auc.cohort.eq("CPTAC UCEC")
        )
    ].reset_index(drop=True)
    labels = [
        "TCGA BRCA\nHRD",
        "TCGA UCEC\nMANTIS MSI",
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
    distribution_panel(fig.add_subplot(sub[1]), scores, [("TCGA UCEC", "MSI/MMR")], "MANTIS MSI · primary SBS6 score", CPTAC, p_lookup, False)
    distribution_panel(fig.add_subplot(sub[2]), scores, [("TCGA UCEC", "POLE/POLD1")], r"$\it{POLE}$/$\it{POLD1}$ · primary SBS10 score", PURPLE, p_lookup, False)
    save_figure(fig, output, "Figure_4")


def main() -> None:
    args = parse_args()
    configure_style()
    summary = pd.read_csv(SOURCE / "cohort_metrics.tsv", sep="\t")
    paired = pd.read_csv(SOURCE / "paired_metrics.tsv", sep="\t")
    paired = paired[paired.nominal_positive_both].copy()
    paired["pair"] = paired.cancer + "–" + paired.signature
    figure1(summary, paired, args.output_root)
    figure2(summary, paired, args.output_root)
    figure3(summary, paired, args.output_root)
    print(f"Figures 1–3 saved to {args.output_root}")


if __name__ == "__main__":
    main()
