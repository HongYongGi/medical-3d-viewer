"""Automatic nnUNetv2 model weight scanner.

Scans a directory for nnUNetv2 trained model weights and extracts metadata
from dataset.json, plans.json, and directory structure.

Directory structure expected:
    weight_dir/
    ├── Dataset302_Segmentation/
    │   └── nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/
    │       ├── dataset.json
    │       ├── plans.json
    │       └── fold_all/
    │           ├── checkpoint_final.pth
    │           └── checkpoint_best.pth
    └── ...
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

DATASET_DIR_PATTERN = re.compile(r"^Dataset(\d+)_(.+)$")
TRAINER_DIR_PATTERN = re.compile(r"^(.+?)__(.+?)__(.+)$")
FOLD_DIR_PATTERN = re.compile(r"^fold_(\d+|all)$")
SKIP_DIRS = {"predicted_next_stage", "__pycache__", ".git"}


@dataclass
class ScannedModel:
    """A single nnUNetv2 model discovered by scanning."""
    dataset_id: int
    dataset_name: str
    trainer: str
    plans: str
    configuration: str
    folds: list[str]
    weight_path: Path
    labels: dict[int, str] = field(default_factory=dict)
    channel_names: dict[str, str] = field(default_factory=dict)
    num_training: int = 0
    file_ending: str = ".nii.gz"
    has_best_checkpoint: bool = False
    has_final_checkpoint: bool = False
    checkpoint_sizes_mb: dict[str, float] = field(default_factory=dict)

    @property
    def auto_id(self) -> str:
        return f"d{self.dataset_id}_{self.configuration}"

    @property
    def display_name(self) -> str:
        name = self.dataset_name.replace("_", " ")
        return f"{name} ({self.configuration})"

    @property
    def best_fold(self) -> str:
        if "all" in self.folds:
            return "all"
        if self.folds:
            return self.folds[0]
        return "0"

    def to_model_config(self):
        """Convert to ModelConfig for use in the app."""
        from ..core.config import ModelConfig
        return ModelConfig(
            id=self.auto_id,
            name=self.display_name,
            dataset_id=self.dataset_id,
            trainer=self.trainer,
            plans=self.plans,
            configuration=self.configuration,
            fold=self.best_fold,
            weight_path=str(self.weight_path),
            labels=dict(self.labels),
            description=(
                f"자동 탐지 | {self.trainer} | "
                f"Folds: {','.join(self.folds)} | "
                f"학습 데이터: {self.num_training}건"
            ),
            source="auto_scan",
            channel_names=dict(self.channel_names),
            num_training=self.num_training,
        )


class WeightScanner:
    """Scans a directory tree for nnUNetv2 trained model weights."""

    def __init__(self, weight_dir: Path | str):
        self.weight_dir = Path(weight_dir)

    def scan_all(self) -> list[ScannedModel]:
        """Scan the weight directory and return all discovered models."""
        if not self.weight_dir.exists():
            log.warning(f"Weight directory does not exist: {self.weight_dir}")
            return []

        results: list[ScannedModel] = []
        for entry in sorted(self.weight_dir.iterdir()):
            if not entry.is_dir():
                continue
            match = DATASET_DIR_PATTERN.match(entry.name)
            if not match:
                continue
            dataset_id = int(match.group(1))
            dataset_name = match.group(2)
            models = self._parse_dataset_dir(entry, dataset_id, dataset_name)
            results.extend(models)

        log.info(f"Scanned {self.weight_dir}: found {len(results)} models")
        return results

    def _parse_dataset_dir(
        self, dataset_dir: Path, dataset_id: int, dataset_name: str
    ) -> list[ScannedModel]:
        """Parse a single Dataset directory."""
        models = []
        for entry in sorted(dataset_dir.iterdir()):
            if not entry.is_dir() or entry.name in SKIP_DIRS:
                continue
            match = TRAINER_DIR_PATTERN.match(entry.name)
            if not match:
                continue
            trainer = match.group(1)
            plans = match.group(2)
            configuration = match.group(3)

            model = self._parse_trainer_dir(
                entry, dataset_id, dataset_name, trainer, plans, configuration
            )
            if model is not None:
                models.append(model)
        return models

    def _parse_trainer_dir(
        self,
        trainer_dir: Path,
        dataset_id: int,
        dataset_name: str,
        trainer: str,
        plans: str,
        configuration: str,
    ) -> ScannedModel | None:
        """Parse a single trainer/plans/config directory."""
        # Detect folds
        folds = self._detect_folds(trainer_dir)
        if not folds:
            return None

        # Check at least one fold has a checkpoint
        has_best = False
        has_final = False
        ckpt_sizes: dict[str, float] = {}
        for fold_name in folds:
            fold_dir = trainer_dir / f"fold_{fold_name}"
            best = fold_dir / "checkpoint_best.pth"
            final = fold_dir / "checkpoint_final.pth"
            if best.exists():
                has_best = True
                ckpt_sizes[f"fold_{fold_name}/best"] = best.stat().st_size / (1024 * 1024)
            if final.exists():
                has_final = True
                ckpt_sizes[f"fold_{fold_name}/final"] = final.stat().st_size / (1024 * 1024)

        if not has_best and not has_final:
            return None

        # Parse metadata
        labels, channel_names, num_training, file_ending = self._load_dataset_json(
            trainer_dir / "dataset.json"
        )

        return ScannedModel(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            trainer=trainer,
            plans=plans,
            configuration=configuration,
            folds=folds,
            weight_path=trainer_dir,
            labels=labels,
            channel_names=channel_names,
            num_training=num_training,
            file_ending=file_ending,
            has_best_checkpoint=has_best,
            has_final_checkpoint=has_final,
            checkpoint_sizes_mb=ckpt_sizes,
        )

    def _detect_folds(self, trainer_dir: Path) -> list[str]:
        """Detect available folds in a trainer directory."""
        folds = []
        for entry in sorted(trainer_dir.iterdir()):
            if not entry.is_dir():
                continue
            match = FOLD_DIR_PATTERN.match(entry.name)
            if match:
                folds.append(match.group(1))
        return folds

    @staticmethod
    def _load_dataset_json(
        path: Path,
    ) -> tuple[dict[int, str], dict[str, str], int, str]:
        """Parse dataset.json and return (labels, channel_names, num_training, file_ending)."""
        labels: dict[int, str] = {}
        channel_names: dict[str, str] = {}
        num_training = 0
        file_ending = ".nii.gz"

        if not path.exists():
            return labels, channel_names, num_training, file_ending

        try:
            with open(path) as f:
                data = json.load(f)

            # Parse labels: {"background": 0, "aorta": 1, ...}
            raw_labels = data.get("labels", {})
            for name, idx in raw_labels.items():
                if isinstance(idx, int) and idx > 0:  # skip background
                    labels[idx] = name

            # Parse channel names: {"0": "CT"}
            channel_names = data.get("channel_names", {})

            num_training = data.get("numTraining", 0)
            file_ending = data.get("file_ending", ".nii.gz")

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            log.warning(f"Failed to parse {path}: {e}")

        return labels, channel_names, num_training, file_ending
