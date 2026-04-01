"""nnUNetv2 model management page with auto-scan support."""
from __future__ import annotations

import streamlit as st
from pathlib import Path

from ..core.config import AppConfig
from ..inference.model_registry import ModelRegistry
from ..inference.nnunet_runner import NnUNetRunner


def render_model_manager(config: AppConfig, registry: ModelRegistry):
    """Render model management page."""
    st.header("🧠 모델 관리")

    # Scan status banner
    _render_scan_status(config)

    # Statistics
    all_models = config.models
    auto_models = registry.get_auto_scanned_models()
    manual_models = registry.get_manual_models()
    available = sum(1 for m in all_models if NnUNetRunner(m).check_model_available())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("전체 모델", len(all_models))
    col2.metric("자동 탐지", len(auto_models))
    col3.metric("수동 등록", len(manual_models))
    col4.metric("사용 가능", f"{available}/{len(all_models)}")

    # Rescan button
    if config.nnunet.weight_dir:
        if st.button("🔄 디렉토리 재스캔", use_container_width=False):
            st.cache_data.clear()
            st.rerun()

    st.markdown("---")

    # Auto-scanned models section
    if auto_models:
        st.subheader(f"🔍 자동 탐지된 모델 ({len(auto_models)}개)")
        st.caption(f"경로: `{config.nnunet.weight_dir}`")

        # Group by dataset
        datasets: dict[int, list] = {}
        for m in auto_models:
            datasets.setdefault(m.dataset_id, []).append(m)

        for dataset_id in sorted(datasets.keys()):
            models = datasets[dataset_id]
            _render_dataset_group(dataset_id, models)

    # Manual models section
    if manual_models:
        st.markdown("---")
        st.subheader(f"📝 수동 등록 모델 ({len(manual_models)}개)")
        for model in manual_models:
            _render_model_card(model, tag="수동")

    # No models at all
    if not all_models:
        st.warning("등록된 모델이 없습니다.")
        if not config.nnunet.weight_dir:
            st.info("💡 `configs/app.yaml`에서 `weight_dir`을 설정하면 자동으로 모델을 탐지합니다.")

    # Pipeline section
    st.markdown("---")
    _render_pipelines(config, registry)

    # Help section
    st.markdown("---")
    _render_help(config)


def _render_scan_status(config: AppConfig):
    """Show scan status banner."""
    if config.nnunet.auto_scan and config.nnunet.weight_dir:
        weight_dir = Path(config.nnunet.weight_dir)
        if weight_dir.exists():
            dataset_count = sum(1 for d in weight_dir.iterdir()
                                if d.is_dir() and d.name.startswith("Dataset"))
            st.success(
                f"✅ 자동 스캔 활성화 | "
                f"경로: `{config.nnunet.weight_dir}` | "
                f"Dataset 디렉토리: {dataset_count}개"
            )
        else:
            st.error(f"❌ 가중치 디렉토리를 찾을 수 없습니다: `{config.nnunet.weight_dir}`")
    elif not config.nnunet.weight_dir:
        st.info(
            "💡 자동 스캔이 비활성화되어 있습니다. "
            "`configs/app.yaml`에서 `weight_dir`을 설정하세요."
        )


def _render_dataset_group(dataset_id: int, models: list):
    """Render a group of models from the same dataset."""
    first = models[0]
    group_label = f"**Dataset{dataset_id}** — {first.name.split('(')[0].strip()}"

    with st.expander(f"📦 {group_label} ({len(models)}개 구성)", expanded=True):
        for model in models:
            _render_model_card(model, tag="자동")


def _render_model_card(model, tag: str = ""):
    """Render a single model as a detailed card."""
    runner = NnUNetRunner(model)
    available = runner.check_model_available()
    status_icon = "✅" if available else "❌"
    tag_badge = f" `{tag}`" if tag else ""

    st.markdown(f"##### {status_icon} {model.name}{tag_badge}")

    col_info, col_detail, col_status = st.columns([2, 2, 1])

    with col_info:
        st.markdown(f"**ID:** `{model.id}`")
        st.markdown(f"**Dataset ID:** {model.dataset_id}")
        st.markdown(f"**Configuration:** `{model.configuration}`")
        st.markdown(f"**Trainer:** `{model.trainer}`")
        st.markdown(f"**Plans:** `{model.plans}`")
        st.markdown(f"**Fold:** `{model.fold}`")

    with col_detail:
        # Labels
        if model.labels:
            label_str = ", ".join(f"`{k}: {v}`" for k, v in sorted(model.labels.items()))
            st.markdown(f"**레이블:** {label_str}")
        else:
            st.markdown("**레이블:** (없음)")

        # Channel names
        if model.channel_names:
            ch_str = ", ".join(f"{k}: {v}" for k, v in model.channel_names.items())
            st.markdown(f"**채널:** {ch_str}")

        # Training info
        if model.num_training > 0:
            st.markdown(f"**학습 데이터:** {model.num_training}건")

        st.markdown(f"**설명:** {model.description}")

    with col_status:
        if available:
            st.success("사용 가능")
            weight_path = Path(model.weight_path)
            if weight_path.exists():
                folds = sorted(weight_path.glob("fold_*"))
                for fold_dir in folds:
                    ckpt_best = fold_dir / "checkpoint_best.pth"
                    ckpt_final = fold_dir / "checkpoint_final.pth"
                    ckpt = ckpt_best if ckpt_best.exists() else ckpt_final
                    if ckpt.exists():
                        size_mb = ckpt.stat().st_size / (1024 * 1024)
                        st.caption(f"{fold_dir.name}: {size_mb:.0f}MB")
        else:
            st.error("사용 불가")
            if model.weight_path:
                st.caption(f"경로: `{model.weight_path}`")

    st.markdown("---")


def _render_pipelines(config: AppConfig, registry: ModelRegistry):
    """Render pipeline section."""
    st.subheader("🔗 파이프라인")
    pipelines = config.pipelines
    if not pipelines:
        st.info("등록된 파이프라인이 없습니다.")
        return

    for pipeline in pipelines:
        with st.expander(f"🔗 {pipeline.name} ({pipeline.id})"):
            st.markdown(f"**설명:** {pipeline.description}")
            st.markdown(f"**병합 전략:** {pipeline.merge_strategy}")
            st.markdown("**포함 모델:**")
            for step in pipeline.steps:
                try:
                    model = registry.get_model(step.model_id)
                    runner = NnUNetRunner(model)
                    avail = "✅" if runner.check_model_available() else "❌"
                    st.markdown(f"  {avail} `{step.model_id}` — {model.name} (우선순위: {step.priority})")
                except KeyError:
                    st.markdown(f"  ❓ `{step.model_id}` (미등록)")


def _render_help(config: AppConfig):
    """Render help section."""
    st.subheader("📖 도움말")

    with st.expander("자동 모델 탐지 설정"):
        st.markdown(f"""
**현재 설정:**
- 가중치 경로: `{config.nnunet.weight_dir or '(미설정)'}`
- 자동 스캔: `{'활성화' if config.nnunet.auto_scan else '비활성화'}`

**설정 방법:** `configs/app.yaml` 편집

```yaml
nnunet:
  weight_dir: "/path/to/nnUNet_results"
  auto_scan: true
```

**지원하는 디렉토리 구조:**
```
weight_dir/
├── Dataset302_Segmentation/
│   └── nnUNetTrainer__Plans__3d_fullres/
│       ├── dataset.json    (레이블 자동 추출)
│       ├── plans.json
│       └── fold_all/
│           └── checkpoint_final.pth
└── Dataset404_Heart/
    ├── nnUNetTrainer__Plans__3d_lowres/
    │   └── fold_0~4/
    └── nnUNetTrainer__Plans__3d_fullres/
        └── fold_0~4/
```
        """)

    with st.expander("수동 모델 등록"):
        st.markdown("""
`configs/models.yaml`에 직접 모델을 등록할 수 있습니다.
수동 등록 모델은 자동 탐지보다 **우선**합니다.

```yaml
models:
  - id: "my_model"
    name: "내 모델"
    dataset_id: 100
    weight_path: "/path/to/weights"
    labels:
      1: "장기1"
      2: "장기2"
```
        """)
