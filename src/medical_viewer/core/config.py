from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class RendererConfig:
    host: str = "localhost"
    port: int = 8080
    url: str = "http://localhost:8080"


@dataclass
class PathsConfig:
    uploads: Path = field(default_factory=lambda: Path("data/uploads"))
    results: Path = field(default_factory=lambda: Path("data/results"))
    models: Path = field(default_factory=lambda: Path("data/models"))

    def __post_init__(self):
        self.uploads = Path(self.uploads)
        self.results = Path(self.results)
        self.models = Path(self.models)
        for p in [self.uploads, self.results, self.models]:
            p.mkdir(parents=True, exist_ok=True)


@dataclass
class NnUNetConfig:
    results_dir: str = ""
    raw_dir: str = ""
    preprocessed_dir: str = ""
    weight_dir: str = ""
    auto_scan: bool = True


@dataclass
class ModelConfig:
    id: str
    name: str
    dataset_id: int
    trainer: str = "nnUNetTrainer"
    plans: str = "nnUNetResEncUNetMPlans"
    configuration: str = "3d_fullres"
    fold: str = "all"
    weight_path: str = ""
    labels: dict[int, str] = field(default_factory=dict)
    description: str = ""
    source: str = "manual"  # "manual" or "auto_scan"
    channel_names: dict[str, str] = field(default_factory=dict)
    num_training: int = 0


@dataclass
class PipelineStep:
    model_id: str
    priority: int = 0


@dataclass
class PipelineConfig:
    id: str
    name: str
    description: str = ""
    steps: list[PipelineStep] = field(default_factory=list)
    merge_strategy: str = "union"


@dataclass
class AppConfig:
    renderer: RendererConfig = field(default_factory=RendererConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    nnunet: NnUNetConfig = field(default_factory=NnUNetConfig)
    models: list[ModelConfig] = field(default_factory=list)
    pipelines: list[PipelineConfig] = field(default_factory=list)


def _expand_env(value: str) -> str:
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


_weight_scan_cache: dict[str, tuple[float, list]] = {}
_WEIGHT_SCAN_TTL = 300  # 5 minutes


def _cached_weight_scan(scanner, weight_dir: str) -> list:
    """Cache weight scan results with TTL to avoid repeated filesystem I/O."""
    import time
    now = time.time()
    if weight_dir in _weight_scan_cache:
        cached_time, cached_result = _weight_scan_cache[weight_dir]
        if now - cached_time < _WEIGHT_SCAN_TTL:
            return cached_result
    result = scanner.scan_all()
    _weight_scan_cache[weight_dir] = (now, result)
    return result


def load_config(config_dir: Path | str = "configs") -> AppConfig:
    config_dir = Path(config_dir)

    with open(config_dir / "app.yaml") as f:
        app_data = yaml.safe_load(f)

    with open(config_dir / "models.yaml") as f:
        models_data = yaml.safe_load(f)

    renderer = RendererConfig(**app_data.get("renderer", {}))
    # Allow environment variable override for Docker/production
    env_renderer_url = os.environ.get("RENDERER_URL")
    if env_renderer_url:
        renderer.url = env_renderer_url
    paths = PathsConfig(**app_data.get("paths", {}))
    nnunet_data = app_data.get("nnunet", {})
    nnunet = NnUNetConfig(
        results_dir=_expand_env(nnunet_data.get("results_dir", "")),
        raw_dir=_expand_env(nnunet_data.get("raw_dir", "")),
        preprocessed_dir=_expand_env(nnunet_data.get("preprocessed_dir", "")),
        weight_dir=_expand_env(nnunet_data.get("weight_dir", "")),
        auto_scan=nnunet_data.get("auto_scan", True),
    )

    models = []
    for m in models_data.get("models", []):
        labels = {int(k): v for k, v in m.get("labels", {}).items()}
        models.append(ModelConfig(
            id=m["id"], name=m["name"], dataset_id=m["dataset_id"],
            trainer=m.get("trainer", "nnUNetTrainer"),
            plans=m.get("plans", "nnUNetResEncUNetMPlans"),
            configuration=m.get("configuration", "3d_fullres"),
            fold=m.get("fold", "all"),
            weight_path=_expand_env(m.get("weight_path", "")),
            labels=labels,
            description=m.get("description", ""),
        ))

    pipelines = []
    for p in models_data.get("pipelines", []):
        steps = [PipelineStep(model_id=s["model_id"], priority=s.get("priority", 0))
                 for s in p.get("steps", [])]
        pipelines.append(PipelineConfig(
            id=p["id"], name=p["name"],
            description=p.get("description", ""),
            steps=steps, merge_strategy=p.get("merge_strategy", "union"),
        ))

    # Auto-scan weight directory and merge with manual models (cached)
    if nnunet.auto_scan and nnunet.weight_dir:
        from ..inference.weight_scanner import WeightScanner
        scanner = WeightScanner(Path(nnunet.weight_dir))
        scanned = _cached_weight_scan(scanner, nnunet.weight_dir)

        # Collect manual model keys for dedup
        manual_keys = {(m.dataset_id, m.configuration) for m in models}

        for s in scanned:
            key = (s.dataset_id, s.configuration)
            if key in manual_keys:
                # Fill in weight_path for manual models that lack it
                for m in models:
                    if m.dataset_id == s.dataset_id and m.configuration == s.configuration:
                        if not m.weight_path:
                            m.weight_path = str(s.weight_path)
                        if not m.labels:
                            m.labels = dict(s.labels)
                continue
            models.append(s.to_model_config())

    return AppConfig(
        renderer=renderer, paths=paths, nnunet=nnunet,
        models=models, pipelines=pipelines,
    )
