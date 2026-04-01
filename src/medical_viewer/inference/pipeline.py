from __future__ import annotations

from pathlib import Path
from typing import Callable

import nibabel as nib
import numpy as np

from ..core.config import PipelineConfig
from .model_registry import ModelRegistry
from .nnunet_runner import NnUNetRunner


class PipelineRunner:
    """Run multi-model pipelines with result merging."""

    def __init__(self, pipeline_config: PipelineConfig, registry: ModelRegistry):
        self.pipeline = pipeline_config
        self.registry = registry

    def run(
        self,
        input_path: Path,
        output_dir: Path,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        results: dict[str, Path] = {}
        total_steps = len(self.pipeline.steps)
        sorted_steps = sorted(self.pipeline.steps, key=lambda s: s.priority)

        for i, step in enumerate(sorted_steps):
            step_frac_start = i / total_steps
            step_frac_size = 1.0 / total_steps

            model_config = self.registry.get_model(step.model_id)
            runner = NnUNetRunner(model_config)
            step_output = output_dir / step.model_id

            def step_progress(frac: float, msg: str):
                if progress_callback:
                    total_frac = step_frac_start + frac * step_frac_size
                    progress_callback(total_frac, f"[{step.model_id}] {msg}")

            result_path = runner.predict(input_path, step_output, step_progress)
            results[step.model_id] = result_path

        if len(results) > 1 and self.pipeline.merge_strategy == "union":
            merged_path = self._merge_union(results, output_dir)
            results["merged"] = merged_path

        if progress_callback:
            progress_callback(1.0, "파이프라인 완료!")
        return results

    def _merge_union(self, results: dict[str, Path], output_dir: Path) -> Path:
        merged_path = output_dir / "merged_segmentation.nii.gz"
        ref_img = None
        merged_volume = None
        current_label_offset = 0

        for seg_path in results.values():
            img = nib.load(str(seg_path))
            data = img.get_fdata().astype(np.int32)
            if ref_img is None:
                ref_img = img
                merged_volume = np.zeros_like(data)

            unique_labels = np.unique(data)
            unique_labels = unique_labels[unique_labels > 0]
            for label in unique_labels:
                mask = data == label
                new_label = label + current_label_offset
                merged_volume[mask & (merged_volume == 0)] = new_label
            if len(unique_labels) > 0:
                current_label_offset += int(unique_labels.max())

        merged_img = nib.Nifti1Image(merged_volume, ref_img.affine, ref_img.header)
        nib.save(merged_img, str(merged_path))
        return merged_path
