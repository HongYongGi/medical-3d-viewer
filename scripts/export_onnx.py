#!/usr/bin/env python3
"""Export nnUNet PyTorch model to ONNX format for Rust inference.

Usage:
    python scripts/export_onnx.py \
        --model-dir /path/to/nnUNet_results/DatasetXXX/.../fold_all \
        --output-dir /path/to/output

This produces:
    - model.onnx: ONNX model with dynamic spatial dims
    - preprocess_config.json: Preprocessing parameters for Rust inference
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def load_plans_and_checkpoint(model_dir: Path) -> tuple[dict, dict, dict]:
    """Load nnUNet plans.json, dataset.json, and checkpoint from model directory."""
    # Navigate to find plans.json (may be in parent dirs)
    search_dir = model_dir
    plans_path = None
    dataset_path = None

    for _ in range(4):  # Search up to 4 levels
        if (search_dir / "plans.json").exists():
            plans_path = search_dir / "plans.json"
        if (search_dir / "dataset.json").exists():
            dataset_path = search_dir / "dataset.json"
        if plans_path and dataset_path:
            break
        search_dir = search_dir.parent

    if plans_path is None:
        raise FileNotFoundError(f"plans.json not found near {model_dir}")

    with open(plans_path) as f:
        plans = json.load(f)
    dataset = {}
    if dataset_path:
        with open(dataset_path) as f:
            dataset = json.load(f)

    # Load checkpoint
    ckpt_path = model_dir / "checkpoint_final.pth"
    if not ckpt_path.exists():
        ckpt_path = model_dir / "checkpoint_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found in {model_dir}")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return plans, dataset, checkpoint


def extract_preprocess_config(plans: dict, dataset: dict, configuration: str = "3d_fullres") -> dict:
    """Extract preprocessing parameters from nnUNet plans."""
    config = plans.get("configurations", {}).get(configuration, {})
    if not config:
        # Try first available configuration
        configs = plans.get("configurations", {})
        if configs:
            configuration = list(configs.keys())[0]
            config = configs[configuration]

    # Get normalization schemes
    foreground_intensity = plans.get("foreground_intensity_properties_per_channel", {})
    norm_schemes = config.get("normalization_schemes", ["ZScoreNormalization"])

    # Build per-channel normalization info
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
            ch_info["mean"] = props.get("mean", 0.0)
            ch_info["std"] = props.get("std", 1.0)
            ch_info["percentile_00_5"] = props.get("percentile_00_5", -1000.0)
            ch_info["percentile_99_5"] = props.get("percentile_99_5", 3000.0)
        channel_norms.append(ch_info)

    return {
        "configuration": configuration,
        "patch_size": config.get("patch_size", [128, 128, 128]),
        "spacing": config.get("spacing", [1.0, 1.0, 1.0]),
        "num_input_channels": len(norm_schemes),
        "num_classes": plans.get("num_segmentation_heads", 2),
        "channel_normalization": channel_norms,
        "transpose_forward": plans.get("transpose_forward", [0, 1, 2]),
        "transpose_backward": plans.get("transpose_backward", [0, 1, 2]),
        "use_mirroring": True,
        "mirror_axes": config.get("mirror_axes", [0, 1, 2]) if "mirror_axes" in config
            else list(range(len(config.get("patch_size", [128, 128, 128])))),
        "tile_step_size": 0.5,
    }


def build_model_from_checkpoint(checkpoint: dict, plans: dict, configuration: str = "3d_fullres"):
    """Reconstruct nnUNet model from checkpoint."""
    try:
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    except ImportError:
        raise ImportError("nnunetv2 is required for ONNX export. pip install nnunetv2")

    # Use nnUNet's own model reconstruction
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_mirroring=False,
        device=torch.device("cpu"),
        verbose=False,
    )

    # Get network architecture from checkpoint
    network = predictor._get_network_from_plans(
        plans,
        configuration,
        checkpoint.get("init_args", {}).get("enable_deep_supervision", False),
    )
    network.load_state_dict(checkpoint["network_weights"])
    network.eval()
    return network


def export_to_onnx(
    network: torch.nn.Module,
    patch_size: list[int],
    num_channels: int,
    output_path: Path,
):
    """Export PyTorch model to ONNX with dynamic spatial dimensions."""
    # Create dummy input: [batch, channels, *spatial]
    dummy_input = torch.randn(1, num_channels, *patch_size)

    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Exporting to: {output_path}")

    dynamic_axes = {
        "input": {0: "batch", 2: "depth", 3: "height", 4: "width"},
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

    # Verify
    import onnx
    model = onnx.load(str(output_path))
    onnx.checker.check_model(model)
    print(f"  ONNX model verified: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Export nnUNet model to ONNX")
    parser.add_argument("--model-dir", required=True, help="Path to nnUNet model fold directory")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: same as model-dir)")
    parser.add_argument("--configuration", default="3d_fullres", help="nnUNet configuration name")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir) if args.output_dir else model_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading plans and checkpoint from {model_dir}")
    plans, dataset, checkpoint = load_plans_and_checkpoint(model_dir)

    print(f"[2/4] Extracting preprocessing config")
    preprocess_config = extract_preprocess_config(plans, dataset, args.configuration)

    config_path = output_dir / "preprocess_config.json"
    with open(config_path, "w") as f:
        json.dump(preprocess_config, f, indent=2)
    print(f"  Saved: {config_path}")
    print(f"  Patch size: {preprocess_config['patch_size']}")
    print(f"  Spacing: {preprocess_config['spacing']}")
    print(f"  Classes: {preprocess_config['num_classes']}")

    print(f"[3/4] Building model from checkpoint")
    network = build_model_from_checkpoint(checkpoint, plans, args.configuration)

    print(f"[4/4] Exporting to ONNX")
    onnx_path = output_dir / "model.onnx"
    export_to_onnx(
        network,
        preprocess_config["patch_size"],
        preprocess_config["num_input_channels"],
        onnx_path,
    )

    print(f"\nExport complete!")
    print(f"  ONNX model: {onnx_path}")
    print(f"  Config: {config_path}")
    print(f"\nUsage with Rust inference:")
    print(f"  nnunet-infer --model {onnx_path} --config {config_path} --input case.nii.gz --output seg.nii.gz")


if __name__ == "__main__":
    main()
