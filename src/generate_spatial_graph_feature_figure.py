"""Create a CNS-style cell graph and multiscale feature schematic."""

from __future__ import annotations

import ast
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = "example001"
EXAMPLE = ROOT / "example"
FEATURE_ROOT = EXAMPLE / "features" / SAMPLE
BORDER_ROOT = EXAMPLE / "border_output" / SAMPLE
FIGURE_ROOT = ROOT / "docs" / "figures"

PNG_PATH = FIGURE_ROOT / "TIBE_spatial_graph_features.png"
PDF_PATH = FIGURE_ROOT / "TIBE_spatial_graph_features.pdf"
SVG_PATH = FIGURE_ROOT / "TIBE_spatial_graph_features.svg"

COLORS = {
    "teal": "#2A8C82",
    "green": "#5B8F55",
    "tumor": "#D6524A",
    "immune": "#3F78B5",
    "stroma": "#3E9A78",
    "morph": "#4D718C",
    "texture": "#756889",
    "distance": "#C28B32",
    "topology": "#2F887E",
    "ink": "#27343A",
    "muted": "#67757C",
    "light": "#E2E8EA",
    "paper": "#FFFFFF",
}

TYPE_META = {
    1: ("Tumor", COLORS["tumor"]),
    2: ("Inflammatory", COLORS["immune"]),
    3: ("Stromal", COLORS["stroma"]),
}


def parse_centroids(values: pd.Series) -> np.ndarray:
    return np.array([ast.literal_eval(value) for value in values], dtype=float) / 16.0


def load_roi_nodes(bounds) -> pd.DataFrame:
    x0, x1, y0, y1 = bounds
    frames = []
    for suffix in ("T", "I", "S"):
        path = FEATURE_ROOT / f"{SAMPLE}_Feats_{suffix}.csv"
        for chunk in pd.read_csv(path, usecols=["name", "Centroid", "CellType"], chunksize=100_000):
            points = parse_centroids(chunk["Centroid"])
            keep = (
                (points[:, 0] >= x0)
                & (points[:, 0] <= x1)
                & (points[:, 1] >= y0)
                & (points[:, 1] <= y1)
            )
            if keep.any():
                selected = chunk.loc[keep, ["name", "CellType"]].copy()
                selected["x"] = points[keep, 0]
                selected["y"] = points[keep, 1]
                frames.append(selected)
    nodes = pd.concat(frames, ignore_index=True)
    nodes["name"] = nodes["name"].astype(int)
    nodes["CellType"] = nodes["CellType"].astype(int)
    return nodes


def load_roi_edges(node_ids: set[int]) -> pd.DataFrame:
    frames = []
    path = FEATURE_ROOT / f"{SAMPLE}_Edges.csv"
    for chunk in pd.read_csv(path, chunksize=300_000):
        keep = chunk["source"].isin(node_ids) & chunk["target"].isin(node_ids)
        if keep.any():
            frames.append(chunk.loc[keep, ["source", "target", "featype"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["source", "target", "featype"]
    )


def card(ax, letter: str, title: str, subtitle: str, accent: str) -> None:
    ax.set_facecolor("white")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(COLORS["light"])
        spine.set_linewidth(0.9)
    ax.plot([0, 1], [1.01, 1.01], transform=ax.transAxes, color=accent, lw=3, clip_on=False)
    ax.text(
        0.03,
        0.94,
        letter,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        color="white",
        ha="center",
        va="center",
        bbox={"boxstyle": "circle,pad=0.30", "facecolor": accent, "edgecolor": "none"},
    )
    ax.text(
        0.105,
        0.95,
        title,
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["ink"],
        ha="left",
        va="center",
    )
    ax.text(
        0.105,
        0.89,
        subtitle,
        transform=ax.transAxes,
        fontsize=6.8,
        color=COLORS["muted"],
        ha="left",
        va="center",
    )


def draw_feature_list(ax, items, x=0.58, y=0.67, line_height=0.105) -> None:
    for index, item in enumerate(items):
        ax.text(
            x,
            y - index * line_height,
            f"•  {item}",
            transform=ax.transAxes,
            fontsize=7.2,
            color=COLORS["ink"],
            ha="left",
            va="center",
        )


def main() -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    level4 = plt.imread(BORDER_ROOT / f"{SAMPLE}_level4_remove_background.png")
    patches = pd.read_csv(BORDER_ROOT / f"{SAMPLE}_border_patches.csv")
    center_x = float(patches.center_x.median())
    center_y = float(patches.center_y.median())
    half = 330
    bounds = (center_x - half, center_x + half, center_y - half, center_y + half)

    nodes = load_roi_nodes(bounds)
    edges = load_roi_edges(set(nodes["name"]))
    positions = nodes.set_index("name")[["x", "y"]]

    valid = edges["source"].isin(positions.index) & edges["target"].isin(positions.index)
    edges = edges.loc[valid].copy()
    degree = pd.concat([edges["source"], edges["target"]]).value_counts()
    center_node = int(degree.index[0]) if len(degree) else int(nodes.iloc[0]["name"])
    center_position = positions.loc[center_node].to_numpy()

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(14.2, 8.0), facecolor="white")
    outer = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.22, 1, 1],
        height_ratios=[1, 1],
        left=0.035,
        right=0.98,
        bottom=0.065,
        top=0.84,
        wspace=0.12,
        hspace=0.19,
    )
    ax_a = fig.add_subplot(outer[0, 0])
    ax_b = fig.add_subplot(outer[1, 0])
    ax_c = fig.add_subplot(outer[0, 1])
    ax_d = fig.add_subplot(outer[0, 2])
    ax_e = fig.add_subplot(outer[1, 1])
    ax_f = fig.add_subplot(outer[1, 2])

    # A: true cell map
    x0, x1, y0, y1 = bounds
    ax_a.imshow(level4[int(y0):int(y1), int(x0):int(x1)], aspect="auto")
    for cell_type in (3, 2, 1):
        subset = nodes[nodes["CellType"] == cell_type]
        label, color = TYPE_META[cell_type]
        shown = subset.iloc[:: max(1, len(subset) // 4500)]
        ax_a.scatter(
            shown["x"] - x0,
            shown["y"] - y0,
            s=3.2,
            c=color,
            alpha=0.72,
            linewidths=0,
            label=label,
        )
    card(ax_a, "A", "Cell phenotyping", "Centroids inherit broad HoVer-Net classes", COLORS["teal"])
    ax_a.legend(
        loc="lower right",
        fontsize=6.6,
        frameon=True,
        facecolor="white",
        edgecolor=COLORS["light"],
        markerscale=1.7,
    )

    # B: true spatial graph
    ax_b.imshow(level4[int(y0):int(y1), int(x0):int(x1)], alpha=0.20, aspect="auto")
    center_local = center_position - [x0, y0]
    graph_half_width = 82
    line_segments = []
    edge_colors = []
    for row in edges.itertuples():
        source = positions.loc[int(row.source)].to_numpy() - [x0, y0]
        target = positions.loc[int(row.target)].to_numpy() - [x0, y0]
        if not (
            np.all(np.abs(source - center_local) <= graph_half_width)
            and np.all(np.abs(target - center_local) <= graph_half_width)
        ):
            continue
        line_segments.append([source, target])
        edge_colors.append("#7D8D94" if row.featype[0] != row.featype[-1] else "#A7B4B9")
    if line_segments:
        ax_b.add_collection(
            LineCollection(line_segments, colors=edge_colors, linewidths=0.65, alpha=0.58)
        )
    for cell_type in (3, 2, 1):
        subset = nodes[nodes["CellType"] == cell_type]
        ax_b.scatter(
            subset["x"] - x0,
            subset["y"] - y0,
            s=7.0,
            c=TYPE_META[cell_type][1],
            alpha=0.82,
            linewidths=0,
        )
    ax_b.add_patch(
        Circle(
            center_local,
            radius=100 / 16,
            fill=False,
            edgecolor=COLORS["distance"],
            linewidth=1.4,
            linestyle=(0, (3, 2)),
        )
    )
    ax_b.scatter(*center_local, s=33, c=COLORS["ink"], edgecolors="white", linewidths=0.7, zorder=6)
    ax_b.set_xlim(center_local[0] - graph_half_width, center_local[0] + graph_half_width)
    ax_b.set_ylim(center_local[1] + graph_half_width, center_local[1] - graph_half_width)
    card(ax_b, "B", "Cell-level spatial graph", "Cells as nodes · proximity-defined edges", COLORS["green"])
    ax_b.text(
        0.04,
        0.05,
        "6 relation layers: T-T · I-I · S-S · T-I · T-S · I-S",
        transform=ax_b.transAxes,
        fontsize=7,
        color=COLORS["ink"],
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": COLORS["light"], "alpha": 0.94},
    )

    # C: morphology
    card(ax_c, "C", "Nuclear morphology", "Shape descriptors from each segmented contour", COLORS["morph"])
    nucleus = Ellipse((0.30, 0.49), 0.34, 0.48, angle=23, transform=ax_c.transAxes,
                      facecolor="#DCE6EB", edgecolor=COLORS["morph"], linewidth=2)
    ax_c.add_patch(nucleus)
    ax_c.annotate("", xy=(0.42, 0.66), xytext=(0.18, 0.31), xycoords=ax_c.transAxes,
                  arrowprops={"arrowstyle": "|-|", "color": COLORS["morph"], "lw": 1.3})
    ax_c.annotate("", xy=(0.18, 0.57), xytext=(0.41, 0.42), xycoords=ax_c.transAxes,
                  arrowprops={"arrowstyle": "|-|", "color": COLORS["morph"], "lw": 1.3})
    ax_c.text(0.29, 0.72, "major axis", transform=ax_c.transAxes, fontsize=6.6, color=COLORS["muted"])
    ax_c.text(0.19, 0.38, "minor axis", transform=ax_c.transAxes, fontsize=6.6, color=COLORS["muted"])
    draw_feature_list(ax_c, ["Area & perimeter", "Circularity", "Eccentricity", "Solidity & curvature"])

    # D: texture
    card(ax_d, "D", "Nuclear texture", "GLCM and intensity statistics inside the nucleus", COLORS["texture"])
    matrix = np.array(
        [[0.08, 0.13, 0.05, 0.01], [0.12, 0.21, 0.10, 0.03],
         [0.04, 0.11, 0.08, 0.02], [0.01, 0.03, 0.02, 0.01]]
    )
    normalized = matrix / matrix.max()
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            shade = plt.cm.Purples(0.18 + 0.72 * normalized[row, column])
            ax_d.add_patch(
                FancyBboxPatch(
                    (0.08 + column * 0.105, 0.65 - row * 0.105),
                    0.10,
                    0.10,
                    boxstyle="square,pad=0",
                    transform=ax_d.transAxes,
                    facecolor=shade,
                    edgecolor="white",
                    linewidth=0.6,
                )
            )
    ax_d.text(0.13, 0.20, "gray-level co-occurrence", transform=ax_d.transAxes,
              fontsize=6.4, color=COLORS["muted"])
    draw_feature_list(ax_d, ["Contrast", "Correlation", "Homogeneity", "Entropy & intensity"])

    # E: distance
    card(ax_e, "E", "Spatial distance", "Local proximity and neighborhood organization", COLORS["distance"])
    points = np.array([[0.20, 0.42], [0.43, 0.67], [0.48, 0.33], [0.31, 0.22]])
    center = np.array([0.27, 0.52])
    for index, point in enumerate(points):
        ax_e.plot([center[0], point[0]], [center[1], point[1]], transform=ax_e.transAxes,
                  color=COLORS["distance"], lw=1.25)
        mid = (center + point) / 2
        ax_e.text(mid[0], mid[1], f"d{index + 1}", transform=ax_e.transAxes,
                  fontsize=6.5, color=COLORS["distance"])
    ax_e.scatter(points[:, 0], points[:, 1], transform=ax_e.transAxes, s=38,
                 c="#C8D2D6", edgecolors="white", linewidths=0.7)
    ax_e.scatter(*center, transform=ax_e.transAxes, s=62, c=COLORS["distance"],
                 edgecolors="white", linewidths=0.8)
    ax_e.add_patch(Circle(center, 0.22, transform=ax_e.transAxes, fill=False,
                          edgecolor=COLORS["distance"], linestyle=(0, (3, 2)), linewidth=1.0))
    draw_feature_list(ax_e, ["Minimum edge length", "Mean edge length", "k-nearest neighbors", "Distance threshold"], x=0.57)

    # F: graph topology
    card(ax_f, "F", "Graph topology", "Node- and component-level descriptors", COLORS["topology"])
    topology_nodes = {
        0: (0.25, 0.52), 1: (0.12, 0.68), 2: (0.13, 0.37),
        3: (0.38, 0.72), 4: (0.43, 0.48), 5: (0.35, 0.27),
    }
    topology_edges = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 3), (3, 4), (4, 5)]
    for source, target in topology_edges:
        p1, p2 = topology_nodes[source], topology_nodes[target]
        ax_f.plot([p1[0], p2[0]], [p1[1], p2[1]], transform=ax_f.transAxes,
                  color="#A6B5BA", lw=1.15)
    for index, point in topology_nodes.items():
        ax_f.scatter(*point, transform=ax_f.transAxes, s=70 if index == 0 else 38,
                     c=COLORS["topology"] if index == 0 else "#B8D6CF",
                     edgecolors="white", linewidths=0.8, zorder=4)
    ax_f.text(0.20, 0.80, "high-degree node", transform=ax_f.transAxes,
              fontsize=6.5, color=COLORS["topology"])
    draw_feature_list(ax_f, ["Degree & coreness", "Clustering coefficient", "Component size", "Centrality & betweenness"], x=0.57)

    fig.text(
        0.035,
        0.965,
        "Cell-level spatial graph construction and multiscale feature extraction",
        fontsize=16,
        fontweight="bold",
        color=COLORS["ink"],
        ha="left",
        va="top",
    )
    fig.text(
        0.035,
        0.923,
        "TIBE · each segmented nucleus becomes a typed graph node with morphology, texture, distance and topology descriptors",
        fontsize=8.4,
        color=COLORS["muted"],
        ha="left",
        va="top",
    )
    fig.text(
        0.035,
        0.02,
        "Graph rule: maximum interaction distance = 100 level-0 pixels (~25 µm at 0.25 µm/pixel); k ≤ 5 nearest neighbors.",
        fontsize=7.4,
        color=COLORS["muted"],
        ha="left",
    )

    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(PDF_PATH, format="pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(SVG_PATH, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(PNG_PATH.resolve())
    print(PDF_PATH.resolve())
    print(SVG_PATH.resolve())


if __name__ == "__main__":
    main()
