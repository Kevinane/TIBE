"""Extract per-cell spatial-graph features from HoVer-Net output."""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from WSIGraph import constructGraphFromDict


DISTANCE_THRESHOLD = 100
GRAPH_LEVEL = 0
MAX_NEIGHBORS = 5
CELL_TYPE_COLUMNS = ("T", "I", "S")
CELL_TYPE_IDS = {"T": [1], "I": [2], "S": [3]}


def _filter_to_xml_windows(vertex_dataframe, xml_path):
    """Return rows whose centroids fall strictly inside an XML window."""
    try:
        from utils_xml import get_windows
    except ImportError as exc:
        raise ImportError(
            "XML filtering requires a utils_xml.py module providing get_windows()."
        ) from exc

    centroids = np.array(vertex_dataframe["Centroid"].tolist())
    windows = np.array(get_windows(xml_path))
    inside = np.zeros((len(centroids), len(windows)), dtype=np.bool_)
    for index, window in enumerate(windows):
        inside[:, index] = (
            (window[0, 0] < centroids[:, 0])
            & (centroids[:, 0] < window[1, 0])
            & (window[0, 1] < centroids[:, 1])
            & (centroids[:, 1] < window[1, 1])
        )
    row_indices, _ = np.where(inside)
    return vertex_dataframe.iloc[row_indices]


def _columns_by_cell_type(vertex_dataframe):
    columns = defaultdict(list)
    for feature_name in vertex_dataframe.columns.values:
        if "Graph" not in feature_name:
            if feature_name != "Contour":
                for cell_type in CELL_TYPE_COLUMNS:
                    columns[cell_type].append(feature_name)
            continue

        feature_type = feature_name.split("_")[1]
        for cell_type in CELL_TYPE_COLUMNS:
            if cell_type in feature_type:
                columns[cell_type].append(feature_name)
    return columns


def fun3(json_path, wsi_path, output_path, xml_path=None):
    """Construct the graph and write per-cell feature and edge CSV files."""
    sample_name = os.path.basename(wsi_path).split(".")[0]
    with open(json_path, encoding="utf-8") as file:
        print(f"{'Loading json':*^30s}")
        nucleus_info = json.load(file)

    graph_result = constructGraphFromDict(
        wsi_path,
        nucleus_info,
        DISTANCE_THRESHOLD,
        MAX_NEIGHBORS,
        GRAPH_LEVEL,
    )
    global_graph, edge_info = graph_result[:2]
    normal_data = graph_result[2] if len(graph_result) > 2 else None

    vertex_dataframe = global_graph.get_vertex_dataframe()
    if xml_path is not None:
        vertex_dataframe = _filter_to_xml_windows(vertex_dataframe, xml_path)

    output_dir = Path(output_path) / sample_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for cell_type, columns in _columns_by_cell_type(vertex_dataframe).items():
        save_index = vertex_dataframe["CellType"].isin(CELL_TYPE_IDS[cell_type]).values
        csv_path = output_dir / f"{sample_name}_Feats_{cell_type}.csv"
        vertex_dataframe.iloc[save_index].to_csv(csv_path, index=False, columns=columns)

    edge_info.to_csv(output_dir / f"{sample_name}_Edges.csv", index=False)
    if normal_data is not None:
        normal_data.to_csv(output_dir / f"{sample_name}_Feats_N.csv")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-path", required=True, help="HoVer-Net nucleus JSON")
    parser.add_argument("--wsi-path", required=True, help="Source WSI")
    parser.add_argument("--output-dir", required=True, help="Feature output root")
    parser.add_argument("--xml-path", help="Optional annotation XML used to filter cells")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = _parse_args()
    fun3(cli_args.json_path, cli_args.wsi_path, cli_args.output_dir, cli_args.xml_path)
