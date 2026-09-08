"""Batch wrapper for HoVer-Net WSI segmentation."""

import argparse
from pathlib import Path

from F1_CellSegment import fun1


def main(input_dir, output_dir):
    """Retain the historical public entry point."""
    fun1(input_dir=input_dir, output_dir=output_dir)


def run_batch(input_root, output_dir):
    """Run segmentation once for every immediate subdirectory of *input_root*."""
    input_root = Path(input_root)
    for input_path in sorted(path for path in input_root.iterdir() if path.is_dir()):
        print(input_path.name)
        main(str(input_path), output_dir)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, help="Root containing one directory per case")
    parser.add_argument("--output-dir", required=True, help="Shared HoVer-Net output directory")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = _parse_args()
    run_batch(cli_args.input_root, cli_args.output_dir)
