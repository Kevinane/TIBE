"""Generate a single-page CNS-style TIBE workflow figure as hybrid vector PDF."""

from __future__ import annotations

import ast
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.path import Path as MplPath
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = "example001"
EXAMPLE = ROOT / "example"
BORDER = EXAMPLE / "border_output" / SAMPLE
FEATURES = EXAMPLE / "features" / SAMPLE
OUTPUT_PDF = ROOT.parent / "output" / "pdf" / "TIBE_example001_CNS_workflow.pdf"
OUTPUT_PREVIEW = EXAMPLE / "workflow_figures" / "TIBE_example001_CNS_workflow_preview.png"

PALETTE = {
    "navy": "#315A7D",
    "teal": "#2A8C82",
    "green": "#5B8F55",
    "gold": "#D79A2B",
    "orange": "#C95C3B",
    "purple": "#6C5B8C",
    "tumor": "#D84A4A",
    "immune": "#3D76B8",
    "stroma": "#42A07B",
    "other": "#9A9A9A",
    "ink": "#263238",
    "muted": "#68737A",
    "line": "#DDE3E6",
    "paper": "#FFFFFF",
}


def load_centroids(path: Path, bounds=None) -> np.ndarray:
    selected: list[np.ndarray] = []
    for chunk in pd.read_csv(path, usecols=["Centroid"], chunksize=100_000):
        points = np.array([ast.literal_eval(value) for value in chunk["Centroid"]], dtype=float)
        points /= 16.0
        if bounds is not None:
            x0, x1, y0, y1 = bounds
            points = points[
                (points[:, 0] >= x0)
                & (points[:, 0] <= x1)
                & (points[:, 1] >= y0)
                & (points[:, 1] <= y1)
            ]
        selected.append(points)
    return np.concatenate(selected) if selected else np.empty((0, 2))


def thin(points: np.ndarray, maximum: int) -> np.ndarray:
    if len(points) <= maximum:
        return points
    return points[:: int(np.ceil(len(points) / maximum))]


def style_panel(ax, letter: str, title: str, subtitle: str, color: str) -> None:
    ax.set_facecolor(PALETTE["paper"])
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(PALETTE["line"])
        spine.set_linewidth(0.8)
    ax.plot([0, 1], [1.015, 1.015], transform=ax.transAxes, color=color, lw=3.2, clip_on=False)
    ax.text(
        0.0,
        1.105,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="white",
        bbox={"boxstyle": "circle,pad=0.33", "facecolor": color, "edgecolor": "none"},
    )
    ax.text(
        0.14,
        1.12,
        title,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        color=PALETTE["ink"],
    )
    ax.text(
        0.14,
        1.065,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=7.5,
        color=PALETTE["muted"],
    )


def add_note(ax, text: str, color: str) -> None:
    patch = FancyBboxPatch(
        (0.04, 0.025),
        0.92,
        0.075,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        transform=ax.transAxes,
        facecolor=color,
        edgecolor="none",
        alpha=0.12,
        zorder=10,
    )
    ax.add_patch(patch)
    ax.text(
        0.5,
        0.062,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.2,
        color=PALETTE["ink"],
        zorder=11,
    )


def main() -> None:
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PREVIEW.parent.mkdir(parents=True, exist_ok=True)

    level4 = Image.open(BORDER / f"{SAMPLE}_level4_remove_background.png").convert("RGB")
    overlay = Image.open(BORDER / f"{SAMPLE}_border_overlay.png").convert("RGB")
    boundary_table = pd.read_csv(BORDER / f"{SAMPLE}_boundary_points.csv")
    boundary = boundary_table[["x", "y"]].to_numpy(dtype=float)
    patches = pd.read_csv(BORDER / f"{SAMPLE}_border_patches.csv")

    median_x = int(patches.center_x.median())
    median_y = int(patches.center_y.median())
    roi_half = 560
    roi = (
        max(0, median_x - roi_half),
        min(level4.width, median_x + roi_half),
        max(0, median_y - roi_half),
        min(level4.height, median_y + roi_half),
    )

    paths = {
        "Tumor": FEATURES / f"{SAMPLE}_Feats_T.csv",
        "Inflammatory": FEATURES / f"{SAMPLE}_Feats_I.csv",
        "Stromal": FEATURES / f"{SAMPLE}_Feats_S.csv",
    }
    local = {name: load_centroids(path, roi) for name, path in paths.items()}
    tumor = load_centroids(paths["Tumor"])
    retained = tumor[MplPath(boundary, closed=True).contains_points(tumor)]

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.titleweight": "bold",
        }
    )
    fig = plt.figure(figsize=(21, 5.2), facecolor="white")
    grid = fig.add_gridspec(1, 6, left=0.018, right=0.988, bottom=0.075, top=0.81, wspace=0.16)
    axes = [fig.add_subplot(grid[0, index]) for index in range(6)]
    accents = [
        PALETTE["navy"],
        PALETTE["teal"],
        PALETTE["green"],
        PALETTE["gold"],
        PALETTE["orange"],
        PALETTE["purple"],
    ]

    # A: input
    ax = axes[0]
    ax.imshow(level4)
    ax.add_patch(
        Rectangle(
            (roi[0], roi[2]),
            roi[1] - roi[0],
            roi[3] - roi[2],
            fill=False,
            edgecolor=PALETTE["navy"],
            linewidth=1.4,
        )
    )
    style_panel(ax, "A", "Input", "H&E whole-slide image", accents[0])
    add_note(ax, "40x WSI · level-4 overview", accents[0])

    # B: cell phenotyping
    ax = axes[1]
    x0, x1, y0, y1 = roi
    ax.imshow(level4.crop((x0, y0, x1, y1)))
    cell_colors = {
        "Tumor": PALETTE["tumor"],
        "Inflammatory": PALETTE["immune"],
        "Stromal": PALETTE["stroma"],
    }
    for name in ("Stromal", "Inflammatory", "Tumor"):
        points = thin(local[name], 9_000)
        ax.scatter(
            points[:, 0] - x0,
            points[:, 1] - y0,
            s=2.1,
            c=cell_colors[name],
            alpha=0.64,
            linewidths=0,
            label=name,
        )
    style_panel(ax, "B", "Cell phenotyping", "HoVer-Net nuclear instances", accents[1])
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=4, color=color, label=label)
        for label, color in cell_colors.items()
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.115),
        fontsize=6.2,
        frameon=True,
        facecolor="white",
        edgecolor=PALETTE["line"],
        borderpad=0.35,
        handletextpad=0.35,
    )
    add_note(ax, "Tumor · inflammatory · stromal", accents[1])

    # C: clustering
    ax = axes[2]
    ax.imshow(level4, alpha=0.22)
    background = thin(tumor, 14_000)
    major = thin(retained, 12_000)
    ax.scatter(background[:, 0], background[:, 1], s=0.35, c="#AEB7BC", alpha=0.32, linewidths=0)
    ax.scatter(major[:, 0], major[:, 1], s=0.65, c=PALETTE["tumor"], alpha=0.68, linewidths=0)
    ax.plot(boundary[:, 0], boundary[:, 1], color=PALETTE["green"], lw=1.0, linestyle=(0, (3, 2)))
    ax.set_xlim(0, level4.width)
    ax.set_ylim(level4.height, 0)
    style_panel(ax, "C", "Tumor clustering", "DBSCAN on tumor centroids", accents[2])
    add_note(ax, "1 retained cluster · >20,000 cells", accents[2])

    # D: interface reconstruction
    ax = axes[3]
    margin = 170
    bx0 = max(0, int(boundary[:, 0].min() - margin))
    bx1 = min(level4.width, int(boundary[:, 0].max() + margin))
    by0 = max(0, int(boundary[:, 1].min() - margin))
    by1 = min(level4.height, int(boundary[:, 1].max() + margin))
    ax.imshow(level4.crop((bx0, by0, bx1, by1)), alpha=0.55)
    shown = thin(retained, 17_000)
    ax.scatter(
        shown[:, 0] - bx0,
        shown[:, 1] - by0,
        s=0.55,
        c=PALETTE["tumor"],
        alpha=0.46,
        linewidths=0,
    )
    ax.plot(
        boundary[:, 0] - bx0,
        boundary[:, 1] - by0,
        color=PALETTE["gold"],
        linewidth=1.8,
    )
    style_panel(ax, "D", "Interface", "Alpha-shape reconstruction", accents[3])
    add_note(ax, "Tumor-non-tumor outer contour", accents[3])

    # E: border-centered sampling
    ax = axes[4]
    focus = patches.iloc[len(patches) // 3]
    radius = 460
    sx0 = max(0, int(focus.center_x - radius))
    sx1 = min(level4.width, int(focus.center_x + radius))
    sy0 = max(0, int(focus.center_y - radius))
    sy1 = min(level4.height, int(focus.center_y + radius))
    ax.imshow(level4.crop((sx0, sy0, sx1, sy1)))
    local_boundary = boundary[
        (boundary[:, 0] >= sx0)
        & (boundary[:, 0] <= sx1)
        & (boundary[:, 1] >= sy0)
        & (boundary[:, 1] <= sy1)
    ]
    ax.plot(
        local_boundary[:, 0] - sx0,
        local_boundary[:, 1] - sy0,
        color=PALETTE["gold"],
        linewidth=1.9,
    )
    local_patches = patches[
        (patches.center_x >= sx0)
        & (patches.center_x <= sx1)
        & (patches.center_y >= sy0)
        & (patches.center_y <= sy1)
    ]
    for row in local_patches.itertuples():
        ax.add_patch(
            Rectangle(
                (row.x_min - sx0, row.y_min - sy0),
                row.x_max - row.x_min,
                row.y_max - row.y_min,
                fill=False,
                edgecolor=PALETTE["teal"],
                linewidth=1.25,
            )
        )
        ax.scatter(
            row.center_x - sx0,
            row.center_y - sy0,
            marker="s",
            s=10,
            c=PALETTE["ink"],
            edgecolors="white",
            linewidths=0.35,
            zorder=5,
        )
    style_panel(ax, "E", "Border sampling", "Distance-constrained centers", accents[4])
    add_note(ax, "250 × 250 px · minimum spacing 250 px", accents[4])

    # F: outputs and QC
    ax = axes[5]
    overlay_crop = overlay.crop((0, 0, min(overlay.width, overlay.height), overlay.height))
    ax.imshow(overlay_crop)
    style_panel(ax, "F", "Outputs and QC", "Coordinates, patches and overlay", accents[5])
    add_note(ax, "24 border patches · visual QC", accents[5])

    # Flow arrows
    for index in range(5):
        left = axes[index].get_position()
        right = axes[index + 1].get_position()
        axes[index].annotate(
            "",
            xy=(right.x0 - 0.004, (right.y0 + right.y1) / 2),
            xytext=(left.x1 + 0.004, (left.y0 + left.y1) / 2),
            xycoords=fig.transFigure,
            textcoords=fig.transFigure,
            arrowprops={
                "arrowstyle": "-|>",
                "lw": 1.1,
                "color": "#8A969C",
                "shrinkA": 0,
                "shrinkB": 0,
                "mutation_scale": 11,
            },
            annotation_clip=False,
        )

    fig.text(
        0.018,
        0.965,
        "TIBE workflow for invasive-border mapping in TNBC",
        fontsize=16,
        fontweight="bold",
        color=PALETTE["ink"],
        ha="left",
        va="top",
    )
    fig.text(
        0.018,
        0.925,
        "Example001 · data-driven visualization of segmentation, spatial aggregation, interface reconstruction and standardized sampling",
        fontsize=8.5,
        color=PALETTE["muted"],
        ha="left",
        va="top",
    )

    fig.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_PREVIEW, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT_PDF.resolve())
    print(OUTPUT_PREVIEW.resolve())


if __name__ == "__main__":
    main()
