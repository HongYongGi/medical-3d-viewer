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
    """Render standalone 3D viewer with CT isosurface + segmentation meshes."""
    import numpy as np

    seg_vol = volume_data.get("seg_volume")
    ct_vol = volume_data.get("ct_volume")
    labels = volume_data.get("labels", {})
    spacing = volume_data.get("spacing", (1.0, 1.0, 1.0))

    has_seg = seg_vol is not None
    has_ct = ct_vol is not None

    if not has_seg and not has_ct:
        st.info("3D 뷰어: CT 또는 세그멘테이션 데이터가 필요합니다.")
        return

    # Controls
    col_ct, col_seg = st.columns(2)
    with col_ct:
        show_ct = st.checkbox("CT 표면 표시", value=has_ct and has_seg, disabled=not has_ct, key="3d_show_ct")
        if show_ct and has_ct:
            ct_threshold = st.slider("CT HU 임계값", -500, 1500, 300, 50, key="3d_ct_thresh",
                                     help="이 값 이상의 HU를 가진 영역을 3D 표면으로 표시합니다 (뼈: 300, 혈관: 150)")
            ct_opacity = st.slider("CT 투명도", 0.05, 0.5, 0.15, 0.05, key="3d_ct_opacity")
    with col_seg:
        show_seg = st.checkbox("세그멘테이션 표시", value=has_seg, disabled=not has_seg, key="3d_show_seg")
        if show_seg and has_seg:
            seg_opacity = st.slider("세그 투명도", 0.1, 1.0, 0.6, 0.1, key="3d_seg_opacity")

    import plotly.graph_objects as go
    fig = go.Figure()
    colors = ['#FF4444', '#44FF44', '#4444FF', '#FFFF44', '#FF44FF', '#44FFFF', '#FF8800', '#8800FF']

    # CT isosurface
    if show_ct and has_ct:
        ct_mesh = _generate_ct_mesh_cached(id(ct_vol), ct_vol.shape, ct_vol, spacing, ct_threshold)
        if ct_mesh is not None:
            verts, faces = ct_mesh
            fig.add_trace(go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color='#E8E8E8', opacity=ct_opacity, name='CT Surface',
                showlegend=True, lighting=dict(ambient=0.5, diffuse=0.8),
            ))

    # Segmentation meshes
    if show_seg and has_seg:
        mesh_data = _generate_meshes_cached(id(seg_vol), seg_vol.shape, seg_vol, spacing)
        for i, (label, verts, faces) in enumerate(mesh_data):
            label_name = labels.get(int(label), f"레이블 {int(label)}")
            color = colors[i % len(colors)]
            fig.add_trace(go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color=color, opacity=seg_opacity if show_seg else 0.5,
                name=label_name, showlegend=True,
            ))

    fig.update_layout(
        scene=dict(aspectmode='data', xaxis_title='X (mm)', yaxis_title='Y (mm)', zaxis_title='Z (mm)',
                   bgcolor='#1a1a2e'),
        height=height, margin=dict(l=0, r=0, t=30, b=0), legend=dict(x=0.02, y=0.98),
        paper_bgcolor='#0E1117',
    )
    st.plotly_chart(fig, use_container_width=True)


@st.cache_data(show_spinner="세그멘테이션 3D 메쉬 생성 중...")
def _generate_meshes_cached(_vol_id: int, _shape: tuple, seg_vol, spacing: tuple) -> list[tuple]:
    import numpy as np
    from skimage import measure
    mesh_data = []
    unique_labels = np.unique(seg_vol)
    unique_labels = unique_labels[unique_labels > 0]
    for label in unique_labels:
        mask = (seg_vol == label)
        if mask.sum() < 100:
            continue
        try:
            verts, faces, _, _ = measure.marching_cubes(
                mask.astype(np.float32), level=0.5, spacing=spacing, step_size=2,
            )
            mesh_data.append((int(label), verts, faces))
        except Exception:
            continue
    return mesh_data


@st.cache_data(show_spinner="CT 3D 표면 생성 중...")
def _generate_ct_mesh_cached(
    _vol_id: int, _shape: tuple, ct_vol, spacing: tuple, threshold: int
) -> tuple | None:
    """Generate CT isosurface mesh at given HU threshold."""
    import numpy as np
    from skimage import measure

    mask = (ct_vol >= threshold)
    if mask.sum() < 500:
        return None
    try:
        # Downsample for performance (step_size=3 for CT)
        verts, faces, _, _ = measure.marching_cubes(
            mask.astype(np.float32), level=0.5, spacing=spacing, step_size=3,
        )
        return (verts, faces)
    except Exception:
        return None


def _render_fallback_3d(session_id: str):
    st.warning("Go 3D 렌더러에 연결할 수 없습니다. Plotly 기반 3D 뷰어를 사용합니다.")
    st.caption("Go 렌더러 시작: `cd go_renderer && go run cmd/renderer/main.go`")
