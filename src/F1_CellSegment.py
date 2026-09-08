"""Run HoVer-Net whole-slide nuclear segmentation."""

import argparse
import os
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_MODEL_PATH = PROJECT_ROOT / "Hover" / "hovernet_fast_pannuke_type_tf2pytorch.tar"
DEFAULT_TYPE_INFO_PATH = PROJECT_ROOT / "Hover" / "type_info.json"


def fun1(input_dir, output_dir):
    """Segment all supported WSIs in *input_dir* and write HoVer-Net outputs."""
    print(input_dir)
    print(output_dir)

    gpu = "0,1"
    model_mode = "fast"
    nr_types = 6
    batch_size = 16
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    method_args = {
        "method": {
            "model_args": {"nr_types": nr_types, "mode": model_mode},
            "model_path": str(DEFAULT_MODEL_PATH),
        },
        "type_info_path": str(DEFAULT_TYPE_INFO_PATH),
    }

    patch_shapes = {"fast": (256, 164), "original": (270, 80)}
    patch_input_shape, patch_output_shape = patch_shapes.get(
        model_mode, patch_shapes["original"]
    )
    run_args = {
        "batch_size": batch_size * torch.cuda.device_count(),
        "nr_inference_workers": 8,
        "nr_post_proc_workers": 16,
        "patch_input_shape": patch_input_shape,
        "patch_output_shape": patch_output_shape,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "presplit_dir": None,
        "input_mask_dir": "",
        "cache_path": "cache",
        "proc_mag": 40,
        "ambiguous_size": 128,
        "chunk_shape": 10000,
        "tile_shape": 2048,
        "save_thumb": True,
        "save_mask": True,
    }

    from Hover.infer.wsi import InferManager

    infer = InferManager(**method_args)
    infer.process_wsi_list(run_args)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Directory containing WSIs")
    parser.add_argument("--output-dir", required=True, help="HoVer-Net output directory")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = _parse_args()
    fun1(cli_args.input_dir, cli_args.output_dir)
