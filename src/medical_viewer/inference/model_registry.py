from __future__ import annotations
from ..core.config import ModelConfig, PipelineConfig, AppConfig


class ModelRegistry:
    """Registry of available nnUNet models and pipelines."""

    def __init__(self, config: AppConfig):
        self._models: dict[str, ModelConfig] = {m.id: m for m in config.models}
        self._pipelines: dict[str, PipelineConfig] = {p.id: p for p in config.pipelines}

    @property
    def model_ids(self) -> list[str]:
        return list(self._models.keys())

    @property
    def pipeline_ids(self) -> list[str]:
        return list(self._pipelines.keys())

    def get_model(self, model_id: str) -> ModelConfig:
        if model_id not in self._models:
            raise KeyError(f"Model '{model_id}' not found. Available: {self.model_ids}")
        return self._models[model_id]

    def get_pipeline(self, pipeline_id: str) -> PipelineConfig:
        if pipeline_id not in self._pipelines:
            raise KeyError(f"Pipeline '{pipeline_id}' not found. Available: {self.pipeline_ids}")
        return self._pipelines[pipeline_id]

    def get_model_display_options(self) -> list[dict[str, str]]:
        return [
            {"id": m.id, "name": m.name, "description": m.description}
            for m in self._models.values()
        ]

    def get_pipeline_display_options(self) -> list[dict[str, str]]:
        return [
            {"id": p.id, "name": p.name, "description": p.description}
            for p in self._pipelines.values()
        ]

    def get_auto_scanned_models(self) -> list[ModelConfig]:
        return [m for m in self._models.values() if m.source == "auto_scan"]

    def get_manual_models(self) -> list[ModelConfig]:
        return [m for m in self._models.values() if m.source == "manual"]

    def get_models_by_dataset(self, dataset_id: int) -> list[ModelConfig]:
        return [m for m in self._models.values() if m.dataset_id == dataset_id]
