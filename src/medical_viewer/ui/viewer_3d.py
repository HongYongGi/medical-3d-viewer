from __future__ import annotations
import streamlit as st
import streamlit.components.v1 as components


def render_3d_viewer(session_id: str, renderer_url: str = "http://localhost:8080", height: int = 600):
    """Render 3D viewer by embedding Go renderer's Three.js page."""
    st.subheader("🧊 3D 시각화")
    viewer_url = f"{renderer_url}/viewer/{session_id}"
    try:
        import httpx
        resp = httpx.get(f"{renderer_url}/health", timeout=2.0)
        if resp.status_code == 200:
            components.iframe(viewer_url, height=height, scrolling=False)
        else:
            _render_fallback_3d(session_id)
    except Exception:
        _render_fallback_3d(session_id)


def render_3d_viewer_standalone(volume_data: dict, height: int = 600):
    """Render standalone 3D viewer using plotly."""
    import plotly.graph_objects as go
    import numpy as np
    from skimage import measure

    if "seg_volume" not in volume_data or volume_data["seg_volume"] is None:
        st.info("3D 뷰어: 세그멘테이션 결과가 필요합니다.")
        return

    seg_vol = volume_data["seg_volume"]
    labels = volume_data.get("labels", {})
    spacing = volume_data.get("spacing", (1.0, 1.0, 1.0))

    fig = go.Figure()
    unique_labels = np.unique(seg_vol)
    unique_labels = unique_labels[unique_labels > 0]
    colors = ['#FF4444', '#44FF44', '#4444FF', '#FFFF44', '#FF44FF', '#44FFFF', '#FF8800', '#8800FF']

    for i, label in enumerate(unique_labels):
        mask = (seg_vol == label)
        if mask.sum() < 100:
            continue
        try:
            verts, faces, _, _ = measure.marching_cubes(
                mask.astype(np.float32), level=0.5, spacing=spacing, step_size=2,
            )
            label_name = labels.get(int(label), f"레이블 {int(label)}")
            color = colors[i % len(colors)]
            fig.add_trace(go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color=color, opacity=0.5, name=label_name, showlegend=True,
            ))
        except Exception:
            continue

    fig.update_layout(
        scene=dict(aspectmode='data', xaxis_title='X (mm)', yaxis_title='Y (mm)', zaxis_title='Z (mm)'),
        height=height, margin=dict(l=0, r=0, t=30, b=0), legend=dict(x=0.02, y=0.98),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_fallback_3d(session_id: str):
    st.warning("Go 3D 렌더러에 연결할 수 없습니다. Plotly 기반 3D 뷰어를 사용합니다.")
    st.caption("Go 렌더러 시작: `cd go_renderer && go run cmd/renderer/main.go`")
