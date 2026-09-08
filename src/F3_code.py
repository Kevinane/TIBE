"""Compatibility entry point for spatial-graph feature extraction."""

import argparse

from F3_FeatureExtract import fun3


def main(json_path, wsi_path, xml_path, output_path):
    """Retain the historical argument order used by existing callers."""
    fun3(json_path, wsi_path, output_path, xml_path)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-path", required=True, help="HoVer-Net nucleus JSON")
    parser.add_argument("--wsi-path", required=True, help="Source WSI")
    parser.add_argument("--output-dir", required=True, help="Feature output root")
    parser.add_argument("--xml-path", help="Optional annotation XML used to filter cells")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = _parse_args()
    main(cli_args.json_path, cli_args.wsi_path, cli_args.xml_path, cli_args.output_dir)
