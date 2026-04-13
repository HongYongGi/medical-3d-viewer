#!/usr/bin/env python3
"""Export nnUNet PyTorch model to ONNX format for Rust inference.

Usage:
    python scripts/export_onnx.py \
        --model-dir /path/to/nnUNet_results/DatasetXXX/.../fold_all \
        --output-dir /path/to/output  # optional, defaults to model-dir

This produces:
    - model.onnx: ONNX model with dynamic spatial dims
    - preprocess_config.json: Preprocessing parameters for Rust inference
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


def load_plans_and_dataset(model_dir: Path) -> tuple[dict, dict]:
    """Load plans.json and dataset.json by searching up from model_dir."""
    search = model_dir
    plans, dataset = None, {}
    for _ in range(5):
        if plans is None and (search / "plans.json").exists():
            with open(search / "plans.json") as f:
                plans = json.load(f)
        if not dataset and (search / "dataset.json").exists():
            with open(search / "dataset.json") as f:
                dataset = json.load(f)
        if plans is not None and dataset:
            break
        search = search.parent

    if plans is None:
        raise FileNotFoundError(f"plans.json not found near {model_dir}")
    return plans, dataset


def extract_preprocess_config(plans: dict, dataset: dict, configuration: str = "3d_fullres") -> dict:
    """Extract preprocessing parameters from nnUNet plans."""
    configs = plans.get("configurations", {})
    if configuration not in configs:
        configuration = next(iter(configs), configuration)
    config = configs.get(configuration, {})

    foreground_intensity = plans.get("foreground_intensity_properties_per_channel", {})
    norm_schemes = config.get("normalization_schemes", ["ZScoreNormalization"])

    channel_norms = []
    for ch_idx, scheme in enumerate(norm_schemes):
        ch_key = str(ch_idx)
        ch_info = {
            "scheme": scheme,
            "mean": 0.0,
            "std": 1.0,
            "percentile_00_5": -1000.0,
            "percentile_99_5": 3000.0,
        }
        if ch_key in foreground_intensity:
            props = foreground_intensity[ch_key]
            ch_info["mean"] = float(props.get("mean", 0.0))
            ch_info["std"] = float(props.get("std", 1.0))
            ch_info["percentile_00_5"] = float(props.get("percentile_00_5", -1000.0))
            ch_info["percentile_99_5"] = float(props.get("percentile_99_5", 3000.0))
        channel_norms.append(ch_info)

    mirror_axes = config.get("mirror_axes", None)
    if mirror_axes is None:
        patch = config.get("patch_size", [128, 128, 128])
        mirror_axes = list(range(len(patch)))

    return {
        "configuration": configuration,
        "patch_size": config.get("patch_size", [128, 128, 128]),
        "spacing": config.get("spacing", [1.0, 1.0, 1.0]),
        "num_input_channels": len(norm_schemes),
        "num_classes": plans.get("num_segmentation_heads", 2),
        "channel_normalization": channel_norms,
        "transpose_forward": plans.get("transpose_forward", [0, 1, 2]),
        "transpose_backward": plans.get("transpose_backward", [0, 1, 2]),
        "use_mirroring": False,
        "mirror_axes": mirror_axes,
        "tile_step_size": 0.75,
    }


def load_network_via_predictor(model_dir: Path, configuration: str, plans: dict) -> torch.nn.Module:
    """Load nnUNet network using public API (nnUNetv2 2.x compatible)."""
    try:
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    except ImportError:
        raise ImportError("nnunetv2 is required. pip install nnunetv2")

    # Suppress nnUNet env warnings
    os.environ.setdefault("nnUNet_raw", "/tmp/placeholder")
    os.environ.setdefault("nnUNet_preprocessed", "/tmp/placeholder")

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_mirroring=False,
        perform_everything_on_device=False,
        device=torch.device("cpu"),
        verbose=False,
        verbose_preprocessing=False,
    )

    # initialize_from_trained_model_folder expects the trainer__plans__config folder
    # model_dir can be fold_all or fold_0 etc — go up one level if needed
    trainer_dir = model_dir
    if model_dir.name.startswith("fold_"):
        trainer_dir = model_dir.parent

    predictor.initialize_from_trained_model_folder(
        str(trainer_dir),
        use_folds=("all",),
        checkpoint_name="checkpoint_final.pth",
    )

    # predictor.network holds the loaded nn.Module
    network = predictor.network
    network.eval()
    return network


def export_to_onnx(
    network: torch.nn.Module,
    patch_size: list[int],
    num_channels: int,
    output_path: Path,
) -> None:
    """Export PyTorch model to ONNX with dynamic spatial dimensions."""
    dummy_input = torch.randn(1, num_channels, *patch_size)
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Exporting to: {output_path}")

    dynamic_axes = {
        "input":  {0: "batch", 2: "depth", 3: "height", 4: "width"},
        "output": {0: "batch", 2: "depth", 3: "height", 4: "width"},
    }

    torch.onnx.export(
        network,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )

    import onnx
    model = onnx.load(str(output_path))
    onnx.checker.check_model(model)
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  ✅ ONNX verified: {size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Export nnUNet model to ONNX")
    parser.add_argument("--model-dir", required=True,
                        help="Path to nnUNet fold directory (e.g. .../fold_all)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: same as model-dir)")
    parser.add_argument("--configuration", default="3d_fullres",
                        help="nnUNet configuration name (default: 3d_fullres)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else model_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading plans from {model_dir}")
    plans, dataset = load_plans_and_dataset(model_dir)

    print(f"[2/4] Extracting preprocessing config (config={args.configuration})")
    preprocess_config = extract_preprocess_config(plans, dataset, args.configuration)

    config_path = output_dir / "preprocess_config.json"
    with open(config_path, "w") as f:
        json.dump(preprocess_config, f, indent=2)
    print(f"  Saved: {config_path}")
    print(f"  Patch size:  {preprocess_config['patch_size']}")
    print(f"  Spacing:     {preprocess_config['spacing']}")
    print(f"  Classes:     {preprocess_config['num_classes']}")
    print(f"  Norm scheme: {[n['scheme'] for n in preprocess_config['channel_normalization']]}")

    print(f"[3/4] Loading network weights (CPU)")
    network = load_network_via_predictor(model_dir, args.configuration, plans)

    print(f"[4/4] Exporting to ONNX")
    onnx_path = output_dir / "model.onnx"
    export_to_onnx(
        network,
        preprocess_config["patch_size"],
        preprocess_config["num_input_channels"],
        onnx_path,
    )

    print(f"\n✅ Export complete!")
    print(f"  ONNX model:  {onnx_path}")
    print(f"  Config:      {config_path}")
    print(f"\nUsage with Rust inference:")
    print(f"  nnunet-infer --model {onnx_path} --config {config_path} "
          f"--input case.nii.gz --output seg.nii.gz")


if __name__ == "__main__":
    main()
