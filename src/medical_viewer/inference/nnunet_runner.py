from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable

from ..core.config import ModelConfig


class NnUNetRunner:
    """Wrapper for nnUNet v2 inference."""

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

        if progress_callback:
            progress_callback(0.0, "nnUNet 초기화 중...")

        try:
            from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
            import torch
        except ImportError:
            raise ImportError(
                "nnunetv2가 설치되지 않았습니다. "
                "pip install nnunetv2 를 실행해주세요."
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
        if not self.config.weight_path:
            return False
        weight_path = Path(self.config.weight_path)
        return weight_path.exists() and any(weight_path.glob("fold_*"))
