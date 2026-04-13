"""Tests for scripts/export_onnx.py – extract_preprocess_config()."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the scripts directory is importable
_scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from export_onnx import extract_preprocess_config


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------
def _ct_plans() -> dict:
    """CT normalization이 포함된 nnUNet plans 예시."""
    return {
        "configurations": {
            "3d_fullres": {
                "patch_size": [64, 128, 128],
                "spacing": [1.5, 0.75, 0.75],
                "normalization_schemes": ["CTNormalization"],
                "mirror_axes": [0, 1, 2],
            }
        },
        "foreground_intensity_properties_per_channel": {
            "0": {
                "mean": 120.5,
                "std": 300.2,
                "percentile_00_5": -1024.0,
                "percentile_99_5": 2500.0,
            }
        },
        "num_segmentation_heads": 5,
        "transpose_forward": [0, 1, 2],
        "transpose_backward": [0, 1, 2],
    }


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------
class TestExtractPreprocessConfig:
    def test_ct_normalization_parsed(self):
        """CTNormalization 스킴이 올바르게 파싱되어야 합니다."""
        plans = _ct_plans()
        result = extract_preprocess_config(plans, {}, "3d_fullres")

        assert result["num_input_channels"] == 1
        assert result["num_classes"] == 5
        ch = result["channel_normalization"][0]
        assert ch["scheme"] == "CTNormalization"
        assert ch["mean"] == 120.5
        assert ch["std"] == 300.2
        assert ch["percentile_00_5"] == -1024.0
        assert ch["percentile_99_5"] == 2500.0

    def test_patch_size_and_spacing_extracted(self):
        """patch_size와 spacing이 정확하게 추출되어야 합니다."""
        plans = _ct_plans()
        result = extract_preprocess_config(plans, {}, "3d_fullres")

        assert result["patch_size"] == [64, 128, 128]
        assert result["spacing"] == [1.5, 0.75, 0.75]
        assert result["configuration"] == "3d_fullres"

    def test_handles_missing_fields_gracefully(self):
        """필수 필드가 누락되어도 기본값으로 정상 동작해야 합니다."""
        # 최소한의 plans – configurations와 주요 키가 비어있음
        plans: dict = {
            "configurations": {
                "3d_fullres": {
                    "normalization_schemes": ["ZScoreNormalization"],
                }
            },
        }
        result = extract_preprocess_config(plans, {}, "3d_fullres")

        # 기본값이 채워져야 함
        assert result["patch_size"] == [128, 128, 128]
        assert result["spacing"] == [1.0, 1.0, 1.0]
        assert result["num_classes"] == 2  # default num_segmentation_heads
        assert result["num_input_channels"] == 1
        # channel normalization 기본값
        ch = result["channel_normalization"][0]
        assert ch["scheme"] == "ZScoreNormalization"
        assert ch["mean"] == 0.0
        assert ch["std"] == 1.0

    def test_fallback_to_first_configuration(self):
        """요청한 configuration이 없으면 첫 번째 configuration을 사용해야 합니다."""
        plans = {
            "configurations": {
                "2d": {
                    "patch_size": [512, 512],
                    "spacing": [0.5, 0.5],
                    "normalization_schemes": ["ZScoreNormalization"],
                }
            },
            "num_segmentation_heads": 3,
        }
        result = extract_preprocess_config(plans, {}, "3d_fullres")

        # 3d_fullres가 없으므로 "2d"로 fallback
        assert result["configuration"] == "2d"
        assert result["patch_size"] == [512, 512]
        assert result["num_classes"] == 3

    def test_empty_configurations(self):
        """configurations가 완전히 비어있어도 에러 없이 기본값을 반환해야 합니다."""
        plans: dict = {"configurations": {}}
        result = extract_preprocess_config(plans, {}, "3d_fullres")

        assert result["patch_size"] == [128, 128, 128]
        assert result["spacing"] == [1.0, 1.0, 1.0]
