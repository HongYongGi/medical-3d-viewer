"""Tests for NnUNetRunner – Rust/ONNX integration logic."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from medical_viewer.core.config import ModelConfig
from medical_viewer.inference.nnunet_runner import NnUNetRunner


def _make_config(weight_path: str = "") -> ModelConfig:
    return ModelConfig(
        id="test",
        name="Test Model",
        dataset_id=1,
        weight_path=weight_path,
    )


# ------------------------------------------------------------------
# _can_use_rust
# ------------------------------------------------------------------
class TestCanUseRust:
    def test_returns_false_when_no_onnx_model(self):
        """ONNX 모델이 없으면 False를 반환해야 합니다."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # preprocess_config.json만 있고 model.onnx는 없음
            (Path(tmpdir) / "preprocess_config.json").write_text("{}")
            runner = NnUNetRunner(_make_config(weight_path=tmpdir))
            assert runner._can_use_rust() is False

    def test_returns_false_when_binary_not_in_path(self):
        """nnunet-infer 바이너리가 PATH에 없으면 False를 반환해야 합니다."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "model.onnx").write_bytes(b"fake")
            (Path(tmpdir) / "preprocess_config.json").write_text("{}")
            runner = NnUNetRunner(_make_config(weight_path=tmpdir))
            with patch("shutil.which", return_value=None):
                assert runner._can_use_rust() is False

    def test_returns_false_when_weight_path_empty(self):
        """weight_path가 비어있으면 False를 반환해야 합니다."""
        runner = NnUNetRunner(_make_config(weight_path=""))
        assert runner._can_use_rust() is False

    def test_returns_true_when_all_conditions_met(self):
        """ONNX 모델, config, 바이너리 모두 존재하면 True를 반환해야 합니다."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "model.onnx").write_bytes(b"fake")
            (Path(tmpdir) / "preprocess_config.json").write_text("{}")
            runner = NnUNetRunner(_make_config(weight_path=tmpdir))
            with patch("shutil.which", return_value="/usr/local/bin/nnunet-infer"):
                assert runner._can_use_rust() is True


# ------------------------------------------------------------------
# check_model_available
# ------------------------------------------------------------------
class TestCheckModelAvailable:
    def test_detects_onnx_model(self):
        """ONNX 모델이 있으면 True를 반환해야 합니다."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "model.onnx").write_bytes(b"fake")
            runner = NnUNetRunner(_make_config(weight_path=tmpdir))
            assert runner.check_model_available() is True

    def test_detects_pytorch_fold_directories(self):
        """fold_* 디렉토리가 있으면 True를 반환해야 합니다."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "fold_0").mkdir()
            (Path(tmpdir) / "fold_1").mkdir()
            runner = NnUNetRunner(_make_config(weight_path=tmpdir))
            assert runner.check_model_available() is True

    def test_returns_false_for_empty_path(self):
        """weight_path가 빈 문자열이면 False를 반환해야 합니다."""
        runner = NnUNetRunner(_make_config(weight_path=""))
        assert runner.check_model_available() is False

    def test_returns_false_for_empty_directory(self):
        """빈 디렉토리(fold_*도 ONNX도 없음)는 False를 반환해야 합니다."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = NnUNetRunner(_make_config(weight_path=tmpdir))
            assert runner.check_model_available() is False
