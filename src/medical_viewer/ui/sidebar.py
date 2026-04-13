from __future__ import annotations
from collections import defaultdict
import streamlit as st
from ..core.config import ModelConfig
from ..inference.model_registry import ModelRegistry
from ..inference.nnunet_runner import NnUNetRunner


def render_sidebar(registry: ModelRegistry) -> dict:
    """Render sidebar with model/pipeline selection.

    Groups models by dataset, then lets user pick configuration variant.
    """
    st.sidebar.header("⚙️ 모델 설정")

    mode = st.sidebar.radio("실행 모드", ["단일 모델", "파이프라인"], horizontal=True)

    if mode == "단일 모델":
        return _render_model_selector(registry)
    else:
        return _render_pipeline_selector(registry)


def _render_model_selector(registry: ModelRegistry) -> dict:
    """Render model selector grouped by dataset with config variant selection."""
    all_models = [registry.get_model(mid) for mid in registry.model_ids]
    if not all_models:
        st.sidebar.warning("등록된 모델이 없습니다.")
        return {"mode": "model", "selected_id": None}

    # Cache availability check results to avoid repeated filesystem I/O
    availability = {m.id: NnUNetRunner(m).check_model_available() for m in all_models}

    # Group by dataset_id
    dataset_groups: dict[int, list[ModelConfig]] = defaultdict(list)
    for m in all_models:
        dataset_groups[m.dataset_id].append(m)

    # Build dataset display options
    dataset_ids = sorted(dataset_groups.keys())
    dataset_labels = []
    for did in dataset_ids:
        models = dataset_groups[did]
        first = models[0]
        base_name = first.name.split("(")[0].strip()
        variant_count = len(models)
        available = sum(1 for m in models if availability[m.id])
        label = f"Dataset{did} — {base_name}"
        if variant_count > 1:
            label += f" ({variant_count}개 구성)"
        if available > 0:
            label = f"✅ {label}"
        else:
            label = f"❌ {label}"
        dataset_labels.append(label)

    # Step 1: Select dataset
    selected_ds_idx = st.sidebar.selectbox(
        "1. 데이터셋 선택",
        range(len(dataset_ids)),
        format_func=lambda i: dataset_labels[i],
        key="dataset_select",
    )
    selected_dataset_id = dataset_ids[selected_ds_idx]
    variants = dataset_groups[selected_dataset_id]

    # Step 2: Select configuration variant (if multiple)
    if len(variants) == 1:
        selected_model = variants[0]
        st.sidebar.caption(f"구성: `{selected_model.configuration}`")
    else:
        st.sidebar.markdown("**2. 구성(Configuration) 선택**")
        variant_labels = []
        for m in variants:
            avail = "✅" if availability[m.id] else "❌"
            variant_labels.append(f"{avail} {m.configuration} ({m.plans})")

        selected_var_idx = st.sidebar.selectbox(
            "구성 선택",
            range(len(variants)),
            format_func=lambda i: variant_labels[i],
            key="config_select",
            label_visibility="collapsed",
        )
        selected_model = variants[selected_var_idx]

    # Show model details
    with st.sidebar.expander("모델 상세 정보", expanded=False):
        st.markdown(f"**ID:** `{selected_model.id}`")
        st.markdown(f"**Trainer:** `{selected_model.trainer}`")
        st.markdown(f"**Plans:** `{selected_model.plans}`")
        st.markdown(f"**Configuration:** `{selected_model.configuration}`")
        st.markdown(f"**Fold:** `{selected_model.fold}`")
        if selected_model.labels:
            labels_str = ", ".join(f"{k}:{v}" for k, v in sorted(selected_model.labels.items()))
            st.markdown(f"**Labels:** {labels_str}")
        if selected_model.num_training > 0:
            st.markdown(f"**학습 데이터:** {selected_model.num_training}건")
        available = availability[selected_model.id]
        st.markdown(f"**상태:** {'✅ 사용 가능' if available else '❌ 가중치 없음'}")

    st.sidebar.caption(selected_model.description)
    return {"mode": "model", "selected_id": selected_model.id}


def _render_pipeline_selector(registry: ModelRegistry) -> dict:
    """Render pipeline selector."""
    pipelines = registry.get_pipeline_display_options()
    if not pipelines:
        st.sidebar.warning("등록된 파이프라인이 없습니다.")
        return {"mode": "pipeline", "selected_id": None}
    pipeline_names = [p["name"] for p in pipelines]
    selected_idx = st.sidebar.selectbox(
        "파이프라인 선택", range(len(pipelines)),
        format_func=lambda i: pipeline_names[i],
    )
    selected = pipelines[selected_idx]
    st.sidebar.caption(selected["description"])
    return {"mode": "pipeline", "selected_id": selected["id"]}
