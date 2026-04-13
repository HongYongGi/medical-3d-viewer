from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

from ..core.config import ModelConfig


class NnUNetRunner:
    """Wrapper for nnUNet inference.

    Supports two backends:
    - Rust ONNX Runtime (fast, preferred if nnunet-infer binary + .onnx model available)
    - Python nnunetv2 (fallback)
    """

    def __init__(self, model_config: ModelConfig):
        self.config = model_config

    def predict(
        self,
        input_nifti: Path,
        output_dir: Path,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> Path:
        """Run nnUNet inference on a single NIfTI file."""
        output_dir.mkdir(parents=True, exist_ok=True)

        if self._can_use_rust():
            return self._predict_rust(input_nifti, output_dir, progress_callback)
        return self._predict_python(input_nifti, output_dir, progress_callback)

    def _can_use_rust(self) -> bool:
        """Check if Rust ONNX backend is available."""
        if not self.config.weight_path:
            return False
        weight_dir = Path(self.config.weight_path)
        onnx_path = weight_dir / "model.onnx"
        config_path = weight_dir / "preprocess_config.json"
        if not onnx_path.exists() or not config_path.exists():
            return False
        # Check if binary exists
        return shutil.which("nnunet-infer") is not None

    def _predict_rust(
        self,
        input_nifti: Path,
        output_dir: Path,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> Path:
        """Run inference via Rust nnunet-infer binary."""
        weight_dir = Path(self.config.weight_path)
        output_file = output_dir / "segmentation.nii.gz"

        cmd = [
            "nnunet-infer",
            "--model", str(weight_dir / "model.onnx"),
            "--config", str(weight_dir / "preprocess_config.json"),
            "--input", str(input_nifti),
            "--output", str(output_file),
            "--device", "cuda",
            "--mirror", str(True).lower(),
            "--tile-step-size", "0.5",
            "--progress", "true",
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # Read progress from stdout
        for line in iter(process.stdout.readline, ""):
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                if progress_callback:
                    progress_callback(msg.get("progress", 0), msg.get("message", ""))
            except json.JSONDecodeError:
                pass

        process.wait()
        if process.returncode != 0:
            stderr = process.stderr.read()
            raise RuntimeError(f"nnunet-infer failed (exit {process.returncode}): {stderr}")

        if not output_file.exists():
            raise FileNotFoundError(f"Rust inference output not found: {output_file}")

        if progress_callback:
            progress_callback(1.0, "완료!")
        return output_file

    def _predict_python(
        self,
        input_nifti: Path,
        output_dir: Path,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> Path:
        """Run inference via Python nnunetv2 (fallback)."""
        if progress_callback:
            progress_callback(0.0, "nnUNet 초기화 중... (Python 모드)")

        try:
            from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
            import torch
        except ImportError:
            raise ImportError(
                "nnunetv2가 설치되지 않았습니다. "
                "pip install nnunetv2 를 실행하거나, "
                "Rust 추론 엔진을 설정하세요 (scripts/export_onnx.py)"
            )

        if self.config.weight_path:
            os.environ.setdefault(
                "nnUNet_results", str(Path(self.config.weight_path).parent.parent)
            )

        if progress_callback:
            progress_callback(0.1, "모델 로딩 중...")

        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_mirroring=True,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            verbose=False,
            verbose_preprocessing=False,
        )

        folds = ("all",) if self.config.fold == "all" else (int(self.config.fold),)
        predictor.initialize_from_trained_model_folder(
            str(self.config.weight_path),
            use_folds=folds,
            checkpoint_name="checkpoint_final.pth",
        )

        if progress_callback:
            progress_callback(0.3, "전처리 및 추론 중...")

        with tempfile.TemporaryDirectory() as tmp_input:
            tmp_input_path = Path(tmp_input)
            target_name = "case_0000_0000.nii.gz"
            shutil.copy2(Path(input_nifti), tmp_input_path / target_name)

            predictor.predict_from_files(
                list_of_lists_or_source_folder=str(tmp_input_path),
                output_folder_or_list_of_truncated_output_files=str(output_dir),
                save_probabilities=False,
                overwrite=True,
                num_processes_preprocessing=1,
                num_processes_segmentation_export=1,
            )

        if progress_callback:
            progress_callback(0.9, "결과 저장 중...")

        output_file = output_dir / "case_0000.nii.gz"
        if not output_file.exists():
            nii_files = list(output_dir.glob("*.nii.gz"))
            if nii_files:
                output_file = nii_files[0]
            else:
                raise FileNotFoundError(f"추론 결과를 찾을 수 없습니다: {output_dir}")

        if progress_callback:
            progress_callback(1.0, "완료!")
        return output_file

    def check_model_available(self) -> bool:
        """Check if model weights are available (ONNX or PyTorch)."""
        if not self.config.weight_path:
            return False
        weight_path = Path(self.config.weight_path)
        # Check ONNX model
        if (weight_path / "model.onnx").exists():
            return True
        # Check PyTorch checkpoint
        return weight_path.exists() and any(weight_path.glob("fold_*"))
