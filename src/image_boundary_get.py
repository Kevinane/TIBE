#!/usr/bin/env python3
"""
Identify invasive tumor borders and generate interface-centered image patches.

Inputs
------
1. A tumor-cell CSV file containing a ``Centroid`` column in level-0 coordinates.
2. A whole-slide image readable by OpenSlide, such as an NDPI file.

The analysis image is generated directly from the requested WSI pyramid level.
Tumor-cell coordinates are mapped from level 0 to that level using the
downsampling factor reported by OpenSlide.

Outputs
-------
1. The generated analysis-level image.
2. DBSCAN cluster summary.
3. Reconstructed boundary-point coordinates.
4. Border-patch coordinates and image bounds.
5. An overlay showing tumor contours and border patches.
6. Optional cropped patch images.

Patch-level cell counting and ecosystem classification are intentionally excluded.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import openslide
import pandas as pd
from alpha_shapes import Alpha_Shaper, boundary
from sklearn.cluster import DBSCAN


def parse_centroid(value: object) -> tuple[float, float]:
    """Parse a centroid stored as ``[x, y]``, ``(x, y)``, or ``x,y``."""
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) == 2:
        return float(value[0]), float(value[1])

    text = str(value).strip()

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)) and len(parsed) == 2:
            return float(parsed[0]), float(parsed[1])
    except (ValueError, SyntaxError):
        pass

    parts = [part.strip() for part in text.strip("[]()").split(",")]
    if len(parts) != 2:
        raise ValueError(f"Invalid centroid value: {value!r}")

    return float(parts[0]), float(parts[1])


def read_wsi_level(
    wsi_path: Path,
    analysis_level: int,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """
    Read one WSI pyramid level.

    Returns
    -------
    image
        OpenCV-compatible BGR image.
    downsample_factor
        Downsampling factor from level 0 to the selected level.
    level_dimensions
        Width and height of the selected WSI level.
    """
    if not wsi_path.exists():
        raise FileNotFoundError(f"WSI file was not found: {wsi_path}")

    slide = openslide.OpenSlide(str(wsi_path))

    try:
        if analysis_level < 0 or analysis_level >= slide.level_count:
            raise ValueError(
                f"Invalid analysis level {analysis_level}. "
                f"Available levels are 0 to {slide.level_count - 1}."
            )

        level_dimensions = slide.level_dimensions[analysis_level]
        downsample_factor = float(slide.level_downsamples[analysis_level])

        region = slide.read_region(
            location=(0, 0),
            level=analysis_level,
            size=level_dimensions,
        ).convert("RGB")

        image_rgb = np.asarray(region)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    finally:
        slide.close()

    return image_bgr, downsample_factor, level_dimensions


def load_tumor_centroids(
    csv_path: Path,
    centroid_column: str,
    downsample_factor: float,
) -> np.ndarray:
    """Load level-0 tumor-cell centroids and map them to the analysis level."""
    table = pd.read_csv(csv_path)

    if centroid_column not in table.columns:
        raise KeyError(
            f"Column {centroid_column!r} was not found in {csv_path}. "
            f"Available columns: {list(table.columns)}"
        )

    points = np.asarray(
        [parse_centroid(value) for value in table[centroid_column]],
        dtype=np.float64,
    )

    if points.size == 0:
        raise ValueError(f"No centroid coordinates were found in {csv_path}.")

    return np.rint(points / downsample_factor).astype(np.int32)


def cluster_tumor_cells(
    points: np.ndarray,
    eps: float,
    min_samples: int,
    min_cluster_cells: int,
) -> tuple[np.ndarray, pd.DataFrame, list[int]]:
    """Cluster tumor cells and identify large aggregates retained for contouring."""
    labels = DBSCAN(
        eps=eps,
        min_samples=min_samples,
    ).fit_predict(points)

    records: list[dict[str, object]] = []
    retained_labels: list[int] = []

    for label in sorted(int(value) for value in np.unique(labels) if value >= 0):
        tumor_cell_count = int(np.sum(labels == label))
        retained = tumor_cell_count > min_cluster_cells

        records.append(
            {
                "cluster_label": label,
                "tumor_cell_count": tumor_cell_count,
                "retained": retained,
            }
        )

        if retained:
            retained_labels.append(label)

    summary = pd.DataFrame(
        records,
        columns=["cluster_label", "tumor_cell_count", "retained"],
    )

    return labels, summary, retained_labels


def reconstruct_contour(cluster_points: np.ndarray) -> np.ndarray:
    """Reconstruct the outer contour of one tumor-cell aggregate by alpha shape."""
    unique_points = np.unique(cluster_points, axis=0)

    if unique_points.shape[0] < 4:
        raise ValueError(
            "At least four unique points are required for alpha-shape reconstruction."
        )

    shaper = Alpha_Shaper(unique_points, normalize=False)
    _, shape = shaper.optimize()
    boundaries = boundary.get_boundaries(shape)

    if not boundaries:
        raise RuntimeError("Alpha-shape reconstruction returned no exterior boundary.")

    exterior = np.asarray(boundaries[0].exterior, dtype=np.float64)

    if exterior.ndim != 2 or exterior.shape[1] != 2:
        raise RuntimeError(
            "The reconstructed boundary does not contain valid x/y coordinates."
        )

    return np.rint(exterior).astype(np.int32)


def select_patch_centers(
    boundary_points: np.ndarray,
    minimum_spacing: float,
) -> np.ndarray:
    """Greedily retain boundary points separated by at least the specified distance."""
    if boundary_points.size == 0:
        return np.empty((0, 2), dtype=np.int32)

    ordered_points = boundary_points[np.argsort(boundary_points[:, 1])]
    selected: list[np.ndarray] = []

    for point in ordered_points:
        if not selected:
            selected.append(point)
            continue

        previous_points = np.asarray(selected, dtype=np.float64)
        distances = np.linalg.norm(previous_points - point, axis=1)

        if np.all(distances >= minimum_spacing):
            selected.append(point)

    return np.asarray(selected, dtype=np.int32)


def build_patch_table(
    centers: np.ndarray,
    patch_size: int,
    image_width: int,
    image_height: int,
) -> pd.DataFrame:
    """Create patch coordinates and image-boundary validity flags."""
    half_patch = patch_size // 2
    records: list[dict[str, object]] = []

    for index, (center_x, center_y) in enumerate(centers):
        x_min = int(center_x - half_patch)
        y_min = int(center_y - half_patch)
        x_max = int(x_min + patch_size)
        y_max = int(y_min + patch_size)

        fully_inside_image = (
            x_min >= 0
            and y_min >= 0
            and x_max <= image_width
            and y_max <= image_height
        )

        records.append(
            {
                "patch_id": f"Border_{index:04d}",
                "center_x": int(center_x),
                "center_y": int(center_y),
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
                "fully_inside_image": fully_inside_image,
            }
        )

    return pd.DataFrame(records)


def draw_overlay(
    image: np.ndarray,
    contours: Iterable[np.ndarray],
    patch_table: pd.DataFrame,
    line_width: int,
) -> np.ndarray:
    """Draw reconstructed tumor contours and border patches."""
    overlay = image.copy()

    for contour in contours:
        cv2.polylines(
            overlay,
            [contour.reshape(-1, 1, 2)],
            isClosed=True,
            color=(0, 255, 0),
            thickness=line_width,
        )

    for row in patch_table.itertuples(index=False):
        color = (255, 255, 0) if row.fully_inside_image else (0, 165, 255)

        cv2.rectangle(
            overlay,
            (row.x_min, row.y_min),
            (row.x_max, row.y_max),
            color,
            line_width,
        )

    return overlay


def save_patches(
    image: np.ndarray,
    patch_table: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save patches located entirely inside the analysis image."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for row in patch_table.itertuples(index=False):
        if not row.fully_inside_image:
            continue

        patch = image[
            row.y_min : row.y_max,
            row.x_min : row.x_max,
        ]

        cv2.imwrite(
            str(output_dir / f"{row.patch_id}.png"),
            patch,
        )


def identify_border(
    tumor_csv: Path,
    wsi_path: Path,
    output_dir: Path,
    sample_name: str,
    centroid_column: str = "Centroid",
    analysis_level: int = 4,
    dbscan_eps: float = 20.0,
    dbscan_min_samples: int = 10,
    min_cluster_cells: int = 20_000,
    minimum_center_spacing: float = 250.0,
    patch_size: int = 250,
    line_width: int = 4,
    export_patches: bool = False,
    save_analysis_image: bool = True,
) -> None:
    """Run invasive-border identification and border-patch generation."""
    output_dir.mkdir(parents=True, exist_ok=True)

    image, downsample_factor, level_dimensions = read_wsi_level(
        wsi_path=wsi_path,
        analysis_level=analysis_level,
    )

    if save_analysis_image:
        cv2.imwrite(
            str(output_dir / f"{sample_name}_level{analysis_level}.png"),
            image,
        )

    points = load_tumor_centroids(
        tumor_csv,
        centroid_column=centroid_column,
        downsample_factor=downsample_factor,
    )

    labels, cluster_summary, retained_labels = cluster_tumor_cells(
        points,
        eps=dbscan_eps,
        min_samples=dbscan_min_samples,
        min_cluster_cells=min_cluster_cells,
    )

    cluster_summary.to_csv(
        output_dir / f"{sample_name}_dbscan_cluster_summary.csv",
        index=False,
    )

    if not retained_labels:
        raise RuntimeError(
            "No tumor-cell cluster passed the retention criterion "
            f"(tumor-cell count > {min_cluster_cells})."
        )

    contours: list[np.ndarray] = []
    boundary_records: list[dict[str, int]] = []

    for cluster_label in retained_labels:
        cluster_points = points[labels == cluster_label]
        contour = reconstruct_contour(cluster_points)
        contours.append(contour)

        for point_index, (x_coord, y_coord) in enumerate(contour):
            boundary_records.append(
                {
                    "cluster_label": int(cluster_label),
                    "boundary_point_index": point_index,
                    "x": int(x_coord),
                    "y": int(y_coord),
                }
            )

    boundary_table = pd.DataFrame(boundary_records)
    boundary_table.to_csv(
        output_dir / f"{sample_name}_boundary_points.csv",
        index=False,
    )

    all_boundary_points = np.concatenate(contours, axis=0)

    centers = select_patch_centers(
        all_boundary_points,
        minimum_spacing=minimum_center_spacing,
    )

    image_height, image_width = image.shape[:2]

    patch_table = build_patch_table(
        centers,
        patch_size=patch_size,
        image_width=image_width,
        image_height=image_height,
    )

    patch_table.to_csv(
        output_dir / f"{sample_name}_border_patches.csv",
        index=False,
    )

    overlay = draw_overlay(
        image,
        contours=contours,
        patch_table=patch_table,
        line_width=line_width,
    )

    cv2.imwrite(
        str(output_dir / f"{sample_name}_border_overlay.png"),
        overlay,
    )

    if export_patches:
        save_patches(
            image,
            patch_table,
            output_dir / f"{sample_name}_patches",
        )

    parameters = {
        "sample_name": sample_name,
        "tumor_csv": str(tumor_csv),
        "wsi_path": str(wsi_path),
        "centroid_column": centroid_column,
        "analysis_level": analysis_level,
        "analysis_level_width": int(level_dimensions[0]),
        "analysis_level_height": int(level_dimensions[1]),
        "downsample_factor_from_level0": downsample_factor,
        "dbscan_eps_analysis_pixels": dbscan_eps,
        "dbscan_min_samples": dbscan_min_samples,
        "min_cluster_cells_strictly_greater_than": min_cluster_cells,
        "minimum_center_spacing_analysis_pixels": minimum_center_spacing,
        "patch_size_analysis_pixels": patch_size,
        "number_of_retained_clusters": len(retained_labels),
        "number_of_border_patches": int(len(patch_table)),
    }

    with (output_dir / f"{sample_name}_run_parameters.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            parameters,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Completed: {sample_name}")
    print(f"Analysis level: {analysis_level}")
    print(f"Downsampling factor: {downsample_factor}")
    print(f"Retained tumor clusters: {len(retained_labels)}")
    print(f"Generated border patches: {len(patch_table)}")
    print(f"Results: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Read a WSI with OpenSlide, identify tumor borders, "
            "and generate interface-centered patches."
        )
    )

    parser.add_argument(
        "--tumor-csv",
        type=Path,
        required=True,
        help="CSV containing level-0 tumor-cell centroids.",
    )

    parser.add_argument(
        "--wsi-path",
        type=Path,
        required=True,
        help="Whole-slide image readable by OpenSlide, such as an NDPI file.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for output files.",
    )

    parser.add_argument(
        "--sample-name",
        required=True,
        help="Sample identifier used in output filenames.",
    )

    parser.add_argument(
        "--centroid-column",
        default="Centroid",
        help="Name of the centroid column in the tumor-cell CSV.",
    )

    parser.add_argument(
        "--analysis-level",
        type=int,
        default=4,
        help="WSI pyramid level used for border reconstruction and patch generation.",
    )

    parser.add_argument(
        "--dbscan-eps",
        type=float,
        default=20.0,
        help="DBSCAN eps in analysis-level pixels.",
    )

    parser.add_argument(
        "--dbscan-min-samples",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--min-cluster-cells",
        type=int,
        default=20_000,
        help="Retain clusters containing strictly more than this number of tumor cells.",
    )

    parser.add_argument(
        "--minimum-center-spacing",
        type=float,
        default=250.0,
        help="Minimum distance between patch centers in analysis-level pixels.",
    )

    parser.add_argument(
        "--patch-size",
        type=int,
        default=250,
        help="Square patch side length in analysis-level pixels.",
    )

    parser.add_argument(
        "--line-width",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--save-patches",
        action="store_true",
        help="Save individual patches located fully inside the analysis image.",
    )

    parser.add_argument(
        "--do-not-save-analysis-image",
        action="store_true",
        help="Do not save the generated analysis-level image.",
    )

    return parser


def main() -> None:
    """Parse command-line arguments and run the workflow."""
    args = build_parser().parse_args()

    identify_border(
        tumor_csv=args.tumor_csv,
        wsi_path=args.wsi_path,
        output_dir=args.output_dir,
        sample_name=args.sample_name,
        centroid_column=args.centroid_column,
        analysis_level=args.analysis_level,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
        min_cluster_cells=args.min_cluster_cells,
        minimum_center_spacing=args.minimum_center_spacing,
        patch_size=args.patch_size,
        line_width=args.line_width,
        export_patches=args.save_patches,
        save_analysis_image=not args.do_not_save_analysis_image,
    )


if __name__ == "__main__":
    main()
