from __future__ import annotations
import streamlit as st
from ..inference.model_registry import ModelRegistry


def render_sidebar(registry: ModelRegistry) -> dict:
    """Render sidebar with model/pipeline selection."""
    st.sidebar.header("⚙️ 모델 설정")

    mode = st.sidebar.radio("실행 모드", ["단일 모델", "파이프라인"], horizontal=True)

    if mode == "단일 모델":
        models = registry.get_model_display_options()
        if not models:
            st.sidebar.warning("등록된 모델이 없습니다.")
            return {"mode": "model", "selected_id": None}
        model_names = [m["name"] for m in models]
        selected_idx = st.sidebar.selectbox(
            "모델 선택", range(len(models)), format_func=lambda i: model_names[i],
        )
        selected = models[selected_idx]
        st.sidebar.caption(selected["description"])
        return {"mode": "model", "selected_id": selected["id"]}
    else:
        pipelines = registry.get_pipeline_display_options()
        if not pipelines:
            st.sidebar.warning("등록된 파이프라인이 없습니다.")
            return {"mode": "pipeline", "selected_id": None}
        pipeline_names = [p["name"] for p in pipelines]
        selected_idx = st.sidebar.selectbox(
            "파이프라인 선택", range(len(pipelines)), format_func=lambda i: pipeline_names[i],
        )
        selected = pipelines[selected_idx]
        st.sidebar.caption(selected["description"])
        return {"mode": "pipeline", "selected_id": selected["id"]}
