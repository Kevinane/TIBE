"""Generate publication-style workflow figures from the completed example run."""

from __future__ import annotations

import ast
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from matplotlib.patches import Rectangle
from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_ROOT = PROJECT_ROOT / "example"
SAMPLE = "example001"
BORDER_ROOT = EXAMPLE_ROOT / "border_output" / SAMPLE
FEATURE_ROOT = EXAMPLE_ROOT / "features" / SAMPLE
OUTPUT_ROOT = EXAMPLE_ROOT / "workflow_figures"

LEVEL4_PATH = BORDER_ROOT / f"{SAMPLE}_level4_remove_background.png"
OVERLAY_PATH = BORDER_ROOT / f"{SAMPLE}_border_overlay.png"
BOUNDARY_PATH = BORDER_ROOT / f"{SAMPLE}_boundary_points.csv"
PATCH_PATH = BORDER_ROOT / f"{SAMPLE}_border_patches.csv"

COLORS = {
    "Tumor": "#e31a1c",
    "Inflammatory": "#1769d2",
    "Stromal": "#2ca02c",
}
STEP_COLORS = ["#28577a", "#238476", "#3c8c38", "#d99100", "#d45122", "#67438c"]


def _save_figure(fig: plt.Figure, filename: str) -> None:
    path = OUTPUT_ROOT / filename
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _load_centroids(path: Path, bounds=None) -> np.ndarray:
    """Load level-0 centroid strings in chunks and return level-4 coordinates."""
    selected = []
    for chunk in pd.read_csv(path, usecols=["Centroid"], chunksize=100_000):
        points = np.array([ast.literal_eval(value) for value in chunk["Centroid"]], dtype=float)
        points /= 16.0
        if bounds is not None:
            x0, x1, y0, y1 = bounds
            keep = (
                (points[:, 0] >= x0)
                & (points[:, 0] <= x1)
                & (points[:, 1] >= y0)
                & (points[:, 1] <= y1)
            )
            points = points[keep]
        selected.append(points)
    return np.concatenate(selected, axis=0) if selected else np.empty((0, 2))


def _display_wsi(ax, image: Image.Image, title: str) -> None:
    ax.imshow(image)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=10)
    ax.axis("off")


def make_step_1(level4: Image.Image, roi) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    _display_wsi(ax, level4, "Step 1 · H&E whole-slide image")
    x0, x1, y0, y1 = roi
    ax.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            edgecolor="#202020",
            linewidth=3,
        )
    )
    ax.text(
        x0,
        y0 - 70,
        "Region used for cell-level visualization",
        fontsize=10,
        color="#202020",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )
    _save_figure(fig, "01_input_wsi.png")


def make_step_2(level4: Image.Image, roi, centroids: dict[str, np.ndarray]) -> None:
    x0, x1, y0, y1 = roi
    crop = level4.crop((x0, y0, x1, y1))
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(crop)
    for label in ("Stromal", "Inflammatory", "Tumor"):
        points = centroids[label]
        ax.scatter(
            points[:, 0] - x0,
            points[:, 1] - y0,
            s=7,
            c=COLORS[label],
            label=label,
            alpha=0.78,
            linewidths=0,
        )
    ax.set_title("Step 2 · Nuclear segmentation and cell classification", fontsize=15, fontweight="bold")
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.92)
    ax.axis("off")
    _save_figure(fig, "02_cell_classification.png")


def make_step_3(
    level4: Image.Image,
    tumor_points: np.ndarray,
    boundary_points: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(level4, alpha=0.20)
    stride = max(1, len(tumor_points) // 50_000)
    shown = tumor_points[::stride]
    ax.scatter(shown[:, 0], shown[:, 1], s=0.8, c="#d8d8d8", alpha=0.55, linewidths=0)
    retained = shown[MplPath(boundary_points, closed=True).contains_points(shown)]
    ax.scatter(retained[:, 0], retained[:, 1], s=1.2, c=COLORS["Tumor"], alpha=0.75, linewidths=0)
    ax.plot(
        boundary_points[:, 0],
        boundary_points[:, 1],
        linestyle="--",
        color="#333333",
        linewidth=1.5,
        label="Retained major aggregate",
    )
    ax.set_xlim(0, level4.width)
    ax.set_ylim(level4.height, 0)
    ax.set_title("Step 3 · DBSCAN tumor-cell clustering", fontsize=15, fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.95)
    ax.axis("off")
    _save_figure(fig, "03_tumor_cell_clustering.png")


def make_step_4(
    level4: Image.Image,
    tumor_points: np.ndarray,
    boundary_points: np.ndarray,
) -> None:
    margin = 250
    x0 = max(0, int(boundary_points[:, 0].min() - margin))
    x1 = min(level4.width, int(boundary_points[:, 0].max() + margin))
    y0 = max(0, int(boundary_points[:, 1].min() - margin))
    y1 = min(level4.height, int(boundary_points[:, 1].max() + margin))
    crop = level4.crop((x0, y0, x1, y1))
    within = tumor_points[
        (tumor_points[:, 0] >= x0)
        & (tumor_points[:, 0] <= x1)
        & (tumor_points[:, 1] >= y0)
        & (tumor_points[:, 1] <= y1)
    ]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(crop, alpha=0.40)
    stride = max(1, len(within) // 35_000)
    ax.scatter(
        within[::stride, 0] - x0,
        within[::stride, 1] - y0,
        s=1.4,
        c=COLORS["Tumor"],
        alpha=0.62,
        linewidths=0,
    )
    ax.plot(
        boundary_points[:, 0] - x0,
        boundary_points[:, 1] - y0,
        color="#e5a000",
        linewidth=3,
        label="Alpha-shape outer contour",
    )
    ax.set_title("Step 4 · Tumor–non-tumor interface reconstruction", fontsize=15, fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.95)
    ax.axis("off")
    _save_figure(fig, "04_interface_reconstruction.png")


def make_step_5(
    level4: Image.Image,
    boundary_points: np.ndarray,
    patches: pd.DataFrame,
) -> None:
    focus = patches.iloc[len(patches) // 3]
    radius = 520
    x0 = max(0, int(focus.center_x - radius))
    x1 = min(level4.width, int(focus.center_x + radius))
    y0 = max(0, int(focus.center_y - radius))
    y1 = min(level4.height, int(focus.center_y + radius))
    crop = level4.crop((x0, y0, x1, y1))
    local_boundary = boundary_points[
        (boundary_points[:, 0] >= x0)
        & (boundary_points[:, 0] <= x1)
        & (boundary_points[:, 1] >= y0)
        & (boundary_points[:, 1] <= y1)
    ]
    local_patches = patches[
        (patches.center_x >= x0)
        & (patches.center_x <= x1)
        & (patches.center_y >= y0)
        & (patches.center_y <= y1)
    ]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(crop)
    ax.plot(
        local_boundary[:, 0] - x0,
        local_boundary[:, 1] - y0,
        color="#e5a000",
        linewidth=3,
    )
    for row in local_patches.itertuples():
        ax.add_patch(
            Rectangle(
                (row.x_min - x0, row.y_min - y0),
                row.x_max - row.x_min,
                row.y_max - row.y_min,
                fill=False,
                edgecolor="#00cddd",
                linewidth=2.5,
            )
        )
        ax.scatter(
            row.center_x - x0,
            row.center_y - y0,
            marker="s",
            s=35,
            c="#111111",
            edgecolors="white",
            linewidths=0.8,
            zorder=5,
        )
    ax.set_title("Step 5 · Border-centered, distance-constrained sampling", fontsize=15, fontweight="bold")
    ax.text(
        0.02,
        0.03,
        "Patch: 250 × 250 px · minimum center spacing: 250 px",
        transform=ax.transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.92},
    )
    ax.axis("off")
    _save_figure(fig, "05_border_centered_sampling.png")


def make_step_6(overlay: Image.Image, patch_paths: list[Path]) -> None:
    fig = plt.figure(figsize=(10, 7))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.65, 1], hspace=0.08, wspace=0.04)
    ax_main = fig.add_subplot(grid[0, :])
    ax_main.imshow(overlay)
    ax_main.axis("off")
    for index, patch_path in enumerate(patch_paths[:3]):
        ax = fig.add_subplot(grid[1, index])
        ax.imshow(Image.open(patch_path).convert("RGB"))
        ax.set_title(patch_path.stem, fontsize=9)
        ax.axis("off")
    fig.suptitle("Step 6 · Outputs and quality control", fontsize=16, fontweight="bold", y=0.98)
    _save_figure(fig, "06_outputs_and_qc.png")


def make_composite(step_paths: list[Path]) -> None:
    titles = [
        ("1", "Input", "H&E whole-slide image"),
        ("2", "Nuclear segmentation", "Tumor · inflammatory · stromal"),
        ("3", "Tumor-cell clustering", "DBSCAN on tumor centroids"),
        ("4", "Interface reconstruction", "Alpha-shape outer contour"),
        ("5", "Border-centered sampling", "250 px patches · 250 px spacing"),
        ("6", "Outputs & QC", "Overlays and extracted patches"),
    ]
    fig, axes = plt.subplots(1, 6, figsize=(24, 5.6))
    for index, (ax, image_path, text) in enumerate(zip(axes, step_paths, titles)):
        number, title, subtitle = text
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        image = image.crop((0, int(height * 0.10), width, height))
        image = ImageOps.fit(
            image,
            (900, 900),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.52),
        )
        ax.imshow(image)
        ax.set_title(title, fontsize=12, fontweight="bold", color=STEP_COLORS[index], pad=27)
        ax.text(
            0.5,
            1.035,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#333333",
        )
        ax.text(
            0.03,
            1.12,
            number,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="white",
            bbox={"boxstyle": "circle,pad=0.35", "facecolor": STEP_COLORS[index], "edgecolor": "none"},
        )
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(STEP_COLORS[index])
            spine.set_linewidth(1.2)
        ax.set_xticks([])
        ax.set_yticks([])
        if index < 5:
            ax.annotate(
                "",
                xy=(1.08, 0.50),
                xytext=(1.01, 0.50),
                xycoords="axes fraction",
                arrowprops={"arrowstyle": "-|>", "color": "#555555", "lw": 1.8},
                annotation_clip=False,
            )
    fig.subplots_adjust(left=0.015, right=0.985, top=0.82, bottom=0.04, wspace=0.12)
    _save_figure(fig, "00_TIBE_six_step_workflow.png")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    level4 = Image.open(LEVEL4_PATH).convert("RGB")
    overlay = Image.open(OVERLAY_PATH).convert("RGB")
    boundary_table = pd.read_csv(BOUNDARY_PATH)
    boundary_points = boundary_table[["x", "y"]].to_numpy()
    patches = pd.read_csv(PATCH_PATH)

    median_x = int(patches.center_x.median())
    median_y = int(patches.center_y.median())
    roi_half_size = 650
    roi = (
        max(0, median_x - roi_half_size),
        min(level4.width, median_x + roi_half_size),
        max(0, median_y - roi_half_size),
        min(level4.height, median_y + roi_half_size),
    )

    feature_files = {
        "Tumor": FEATURE_ROOT / f"{SAMPLE}_Feats_T.csv",
        "Inflammatory": FEATURE_ROOT / f"{SAMPLE}_Feats_I.csv",
        "Stromal": FEATURE_ROOT / f"{SAMPLE}_Feats_S.csv",
    }
    local_centroids = {
        label: _load_centroids(path, roi) for label, path in feature_files.items()
    }
    tumor_points = _load_centroids(feature_files["Tumor"])

    make_step_1(level4, roi)
    make_step_2(level4, roi, local_centroids)
    make_step_3(level4, tumor_points, boundary_points)
    make_step_4(level4, tumor_points, boundary_points)
    make_step_5(level4, boundary_points, patches)
    patch_paths = sorted((BORDER_ROOT / f"{SAMPLE}_patches").glob("Border_*.png"))
    make_step_6(overlay, patch_paths)

    step_paths = [OUTPUT_ROOT / f"0{index}_{name}.png" for index, name in enumerate(
        [
            "input_wsi",
            "cell_classification",
            "tumor_cell_clustering",
            "interface_reconstruction",
            "border_centered_sampling",
            "outputs_and_qc",
        ],
        start=1,
    )]
    make_composite(step_paths)
    print(f"Generated workflow figures in: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
