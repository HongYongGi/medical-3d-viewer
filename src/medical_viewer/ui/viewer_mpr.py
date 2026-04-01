"""MPR Viewer with 3D Slicer-style features.

Features:
- Window Level/Width presets (one-click buttons)
- Colormap selection, Crosshair, HU readout
- Segmentation overlay with per-label toggle + contour/fill modes
- Measurement tools (distance, area)
- Axial/Sagittal/Coronal/Oblique + Linked 3-panel view
"""
from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from ..mpr.slicer import MPRSlicer
from ..mpr.windowing import apply_window, WINDOW_PRESETS, COLORMAPS, auto_window
from ..mpr.measurements import compute_distance, compute_all_label_areas

LABEL_COLORS = {
    1: (255, 50, 50, 180), 2: (50, 255, 50, 180), 3: (50, 50, 255, 180),
    4: (255, 255, 50, 180), 5: (255, 50, 255, 180), 6: (50, 255, 255, 180),
    7: (255, 140, 0, 180), 8: (148, 50, 255, 180), 9: (255, 192, 203, 180),
    10: (139, 69, 19, 180),
}


def _create_mpr_figure(
    ct_slice: np.ndarray,
    seg_slice: np.ndarray | None,
    window_center: float,
    window_width: float,
    show_seg: bool = True,
    seg_opacity: float = 0.4,
    colormap: str = "gray",
    show_crosshair: bool = False,
    crosshair_x: int | None = None,
    crosshair_y: int | None = None,
    show_colorbar: bool = False,
    visible_labels: set | None = None,
    overlay_mode: str = "fill",  # "fill", "contour", "both"
) -> go.Figure:
    windowed = apply_window(ct_slice, window_center, window_width)
    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=windowed, colorscale=colormap, showscale=show_colorbar,
        hovertemplate="x: %{x}<br>y: %{y}<br>HU: %{customdata:.0f}<extra></extra>",
        customdata=ct_slice,
    ))

    if seg_slice is not None and show_seg:
        unique_labels = np.unique(seg_slice)
        unique_labels = unique_labels[unique_labels > 0]
        for label in unique_labels:
            if visible_labels is not None and int(label) not in visible_labels:
                continue
            color = LABEL_COLORS.get(int(label), (200, 200, 200, 128))
            r, g, b, a = color
            mask = (seg_slice == label).astype(np.float32)

            if overlay_mode in ("fill", "both"):
                mask_display = np.where(mask > 0, seg_opacity, np.nan)
                fig.add_trace(go.Heatmap(
                    z=mask_display,
                    colorscale=[[0, f"rgba({r},{g},{b},0)"], [1, f"rgba({r},{g},{b},{a})"]],
                    showscale=False, hoverinfo="skip", zmin=0, zmax=1,
                ))

            if overlay_mode in ("contour", "both"):
                fig.add_trace(go.Contour(
                    z=mask, showscale=False, hoverinfo="skip",
                    contours=dict(start=0.5, end=0.5, size=1, coloring="none"),
                    line=dict(color=f"rgb({r},{g},{b})", width=2),
                ))

    if show_crosshair and crosshair_x is not None and crosshair_y is not None:
        h, w = windowed.shape
        fig.add_shape(type="line", x0=crosshair_x, y0=0, x1=crosshair_x, y1=h-1,
                      line=dict(color="yellow", width=1, dash="dot"))
        fig.add_shape(type="line", x0=0, y0=crosshair_y, x1=w-1, y1=crosshair_y,
                      line=dict(color="yellow", width=1, dash="dot"))

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), height=550,
        yaxis=dict(scaleanchor="x", scaleratio=1, showticklabels=False, showgrid=False),
        xaxis=dict(showticklabels=False, showgrid=False),
        plot_bgcolor='black',
    )
    return fig


def render_mpr_viewer(ct_path: str, seg_path: str | None = None):
    """Render MPR viewer."""
    st.subheader("📐 MPR 다면 재구성")

    ct_slicer = _get_slicer(ct_path)
    seg_slicer = _get_slicer(seg_path) if seg_path else None

    # ---- Window Preset Buttons ----
    st.markdown("##### 🖥️ 윈도우 프리셋")
    preset_cols = st.columns(6)
    preset_names = list(WINDOW_PRESETS.keys())

    if "wl_center" not in st.session_state:
        st.session_state.wl_center = 200.0
        st.session_state.wl_width = 800.0

    for i, name in enumerate(preset_names):
        col = preset_cols[i % 6]
        preset = WINDOW_PRESETS[name]
        short_name = name.replace("CT-", "")
        if col.button(f"{preset['icon']} {short_name}", key=f"preset_{name}", use_container_width=True):
            st.session_state.wl_center = float(preset["center"])
            st.session_state.wl_width = float(preset["width"])
            st.rerun()

    auto_col1, auto_col2 = st.columns([1, 5])
    with auto_col1:
        if st.button("🔄 Auto", key="auto_wl", use_container_width=True):
            mid_slice = ct_slicer.get_axial(ct_slicer.num_axial // 2)
            c, w = auto_window(mid_slice)
            st.session_state.wl_center = c
            st.session_state.wl_width = w
            st.rerun()

    # ---- Manual W/L + Colormap + Options ----
    col_wl, col_ww, col_cmap, col_opts = st.columns([1, 1, 1, 1])
    with col_wl:
        wc = st.number_input("W/L Center", value=st.session_state.wl_center, step=10.0, key="wl_center_input")
        st.session_state.wl_center = wc
    with col_ww:
        ww = st.number_input("W/L Width", value=st.session_state.wl_width, step=10.0, key="wl_width_input")
        st.session_state.wl_width = ww
    with col_cmap:
        cmap_name = st.selectbox("컬러맵", list(COLORMAPS.keys()), index=0, key="colormap_sel")
        colormap = COLORMAPS[cmap_name]
    with col_opts:
        show_crosshair = st.checkbox("십자선", value=False, key="crosshair")
        show_colorbar = st.checkbox("컬러바", value=False, key="colorbar")

    # ---- Segmentation Controls ----
    seg_col1, seg_col2, seg_col3 = st.columns([1, 1, 1])
    with seg_col1:
        show_seg = st.checkbox("세그멘테이션 표시", value=True, disabled=seg_path is None, key="show_seg")
    with seg_col2:
        seg_opacity = st.slider("투명도", 0.0, 1.0, 0.4, 0.05, disabled=seg_path is None, key="seg_opacity")
    with seg_col3:
        overlay_mode = st.selectbox(
            "오버레이 모드",
            ["fill (채우기)", "contour (윤곽선)", "both (채우기+윤곽선)"],
            index=0, key="overlay_mode", disabled=seg_path is None,
        )
        overlay_mode_key = overlay_mode.split(" ")[0]

    # Per-label toggle
    visible_labels = None
    if seg_slicer and show_seg:
        seg_labels = set(np.unique(seg_slicer.volume).astype(int)) - {0}
        if seg_labels:
            st.markdown("**레이블 표시 선택:**")
            label_cols = st.columns(min(len(seg_labels), 8))
            visible_labels = set()
            for i, label in enumerate(sorted(seg_labels)):
                col = label_cols[i % len(label_cols)]
                if col.checkbox(f"L{label}", value=True, key=f"lbl_{label}"):
                    visible_labels.add(label)

    # ---- Slice Views ----
    tab_ax, tab_sag, tab_cor, tab_obl, tab_linked, tab_measure = st.tabs(
        ["Axial", "Sagittal", "Coronal", "Oblique", "📐 Linked View", "📏 측정"]
    )

    render_kwargs = dict(
        window_center=wc, window_width=ww, show_seg=show_seg,
        seg_opacity=seg_opacity, colormap=colormap,
        show_crosshair=show_crosshair, show_colorbar=show_colorbar,
        visible_labels=visible_labels, overlay_mode=overlay_mode_key,
    )

    for tab, view_name, get_slice, get_seg, num_slices, key_prefix in [
        (tab_ax, "Axial", ct_slicer.get_axial,
         (lambda i: seg_slicer.get_axial(i)) if seg_slicer else None,
         ct_slicer.num_axial, "ax"),
        (tab_sag, "Sagittal", ct_slicer.get_sagittal,
         (lambda i: seg_slicer.get_sagittal(i)) if seg_slicer else None,
         ct_slicer.num_sagittal, "sag"),
        (tab_cor, "Coronal", ct_slicer.get_coronal,
         (lambda i: seg_slicer.get_coronal(i)) if seg_slicer else None,
         ct_slicer.num_coronal, "cor"),
    ]:
        with tab:
            idx = st.slider(f"{view_name} Slice", 0, num_slices - 1,
                             num_slices // 2, key=f"{key_prefix}_idx")
            ct_slice = get_slice(idx)
            seg_slice = get_seg(idx) if get_seg else None
            _show_slice_info(ct_slice, idx, view_name, num_slices)
            fig = _create_mpr_figure(ct_slice, seg_slice, **render_kwargs)
            st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_chart")

    with tab_obl:
        st.markdown("**Oblique 단면 설정**")
        col1, col2, col3 = st.columns(3)
        with col1:
            cx = st.number_input("Center X (mm)", value=0.0, step=1.0, key="obl_cx")
            nx = st.number_input("Normal X", value=0.0, step=0.1, key="obl_nx")
        with col2:
            cy = st.number_input("Center Y (mm)", value=0.0, step=1.0, key="obl_cy")
            ny = st.number_input("Normal Y", value=0.0, step=0.1, key="obl_ny")
        with col3:
            cz = st.number_input("Center Z (mm)", value=0.0, step=1.0, key="obl_cz")
            nz = st.number_input("Normal Z", value=1.0, step=0.1, key="obl_nz")
        obl_size = st.slider("Slice Size (px)", 128, 512, 256, 32, key="obl_size")
        center = np.array([cx, cy, cz])
        normal = np.array([nx, ny, nz])
        if np.linalg.norm(normal) > 0:
            ct_slice, _ = ct_slicer.get_oblique(center, normal, size=obl_size)
            seg_slice = None
            if seg_slicer:
                seg_slice = seg_slicer.get_oblique_seg(seg_slicer.volume, center, normal, size=obl_size)
            fig = _create_mpr_figure(ct_slice, seg_slice, **render_kwargs)
            st.plotly_chart(fig, use_container_width=True, key="obl_chart")
        else:
            st.warning("Normal 벡터가 0입니다.")

    with tab_linked:
        _render_linked_view(ct_slicer, seg_slicer, **render_kwargs)

    with tab_measure:
        _render_measurement_tab(ct_slicer, seg_slicer, render_kwargs)


def _show_slice_info(ct_slice: np.ndarray, idx: int, view: str, total: int):
    st.caption(
        f"**{view}** | Slice {idx}/{total-1} | "
        f"HU min: {ct_slice.min():.0f} | max: {ct_slice.max():.0f} | "
        f"mean: {ct_slice.mean():.0f} | std: {ct_slice.std():.0f}"
    )


def _render_linked_view(ct_slicer, seg_slicer, **kwargs):
    """Render 3-panel linked view with crosshair sync."""
    st.markdown("**3D Slicer 스타일 연동 뷰**")

    col_a, col_s, col_c = st.columns(3)

    with col_a:
        ax_idx = st.slider("Axial", 0, ct_slicer.num_axial - 1,
                            ct_slicer.num_axial // 2, key="linked_ax")
    with col_s:
        sag_idx = st.slider("Sagittal", 0, ct_slicer.num_sagittal - 1,
                             ct_slicer.num_sagittal // 2, key="linked_sag")
    with col_c:
        cor_idx = st.slider("Coronal", 0, ct_slicer.num_coronal - 1,
                             ct_slicer.num_coronal // 2, key="linked_cor")

    # Enable crosshair linking
    linked_kwargs = dict(kwargs)
    linked_kwargs["show_crosshair"] = True

    col_a2, col_s2, col_c2 = st.columns(3)

    with col_a2:
        ct_ax = ct_slicer.get_axial(ax_idx)
        seg_ax = seg_slicer.get_axial(ax_idx) if seg_slicer else None
        linked_kwargs["crosshair_x"] = sag_idx
        linked_kwargs["crosshair_y"] = cor_idx
        fig = _create_mpr_figure(ct_ax, seg_ax, **linked_kwargs)
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True, key="linked_ax_chart")

    with col_s2:
        ct_sag = ct_slicer.get_sagittal(sag_idx)
        seg_sag = seg_slicer.get_sagittal(sag_idx) if seg_slicer else None
        linked_kwargs["crosshair_x"] = cor_idx
        linked_kwargs["crosshair_y"] = ax_idx
        fig = _create_mpr_figure(ct_sag, seg_sag, **linked_kwargs)
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True, key="linked_sag_chart")

    with col_c2:
        ct_cor = ct_slicer.get_coronal(cor_idx)
        seg_cor = seg_slicer.get_coronal(cor_idx) if seg_slicer else None
        linked_kwargs["crosshair_x"] = sag_idx
        linked_kwargs["crosshair_y"] = ax_idx
        fig = _create_mpr_figure(ct_cor, seg_cor, **linked_kwargs)
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True, key="linked_cor_chart")


def _render_measurement_tab(ct_slicer, seg_slicer, render_kwargs):
    """Render measurement tools tab."""
    st.markdown("### 📏 측정 도구")

    meas_type = st.radio("측정 유형", ["거리 측정", "면적 측정 (세그멘테이션)"], horizontal=True, key="meas_type")

    if meas_type == "거리 측정":
        st.markdown("**두 점 사이의 거리를 측정합니다.**")
        view = st.selectbox("뷰 선택", ["Axial", "Sagittal", "Coronal"], key="meas_view")
        if view == "Axial":
            idx = st.slider("슬라이스", 0, ct_slicer.num_axial - 1, ct_slicer.num_axial // 2, key="meas_ax_idx")
            ct_slice = ct_slicer.get_axial(idx)
            spacing = (ct_slicer.voxel_spacing[0], ct_slicer.voxel_spacing[1])
        elif view == "Sagittal":
            idx = st.slider("슬라이스", 0, ct_slicer.num_sagittal - 1, ct_slicer.num_sagittal // 2, key="meas_sag_idx")
            ct_slice = ct_slicer.get_sagittal(idx)
            spacing = (ct_slicer.voxel_spacing[1], ct_slicer.voxel_spacing[2])
        else:
            idx = st.slider("슬라이스", 0, ct_slicer.num_coronal - 1, ct_slicer.num_coronal // 2, key="meas_cor_idx")
            ct_slice = ct_slicer.get_coronal(idx)
            spacing = (ct_slicer.voxel_spacing[0], ct_slicer.voxel_spacing[2])

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**점 1**")
            x1 = st.number_input("X1 (px)", value=0, step=1, key="m_x1")
            y1 = st.number_input("Y1 (px)", value=0, step=1, key="m_y1")
        with col2:
            st.markdown("**점 2**")
            x2 = st.number_input("X2 (px)", value=100, step=1, key="m_x2")
            y2 = st.number_input("Y2 (px)", value=100, step=1, key="m_y2")

        m = compute_distance((x1, y1), (x2, y2), spacing)
        st.metric("거리", f"{m.distance_mm:.2f} mm")

        # Show on image
        fig = _create_mpr_figure(ct_slice, None, **render_kwargs)
        fig.add_shape(type="line", x0=x1, y0=y1, x1=x2, y1=y2,
                      line=dict(color="lime", width=2))
        fig.add_annotation(x=(x1+x2)/2, y=(y1+y2)/2,
                           text=f"{m.distance_mm:.1f}mm", showarrow=False,
                           font=dict(color="lime", size=14))
        st.plotly_chart(fig, use_container_width=True, key="meas_dist_chart")

    else:
        if seg_slicer is None:
            st.info("면적 측정에는 세그멘테이션 결과가 필요합니다.")
            return

        idx = st.slider("Axial 슬라이스", 0, ct_slicer.num_axial - 1,
                         ct_slicer.num_axial // 2, key="meas_area_idx")
        ct_slice = ct_slicer.get_axial(idx)
        seg_slice = seg_slicer.get_axial(idx)
        spacing = (ct_slicer.voxel_spacing[0], ct_slicer.voxel_spacing[1])

        labels_dict = {}
        seg_labels_state = st.session_state.get("seg_labels", {})
        if seg_labels_state:
            labels_dict = seg_labels_state

        areas = compute_all_label_areas(seg_slice, spacing, labels_dict)

        if areas:
            st.markdown("**현재 슬라이스 레이블별 면적:**")
            for a in areas:
                col1, col2, col3 = st.columns(3)
                col1.metric(f"L{a.label_id}: {a.label_name}", f"{a.area_mm2:.1f} mm²")
                col2.metric("둘레", f"{a.perimeter_mm:.1f} mm")
                col3.metric("복셀 수", f"{a.area_px}")
        else:
            st.info("이 슬라이스에 세그멘테이션 레이블이 없습니다.")

        fig = _create_mpr_figure(ct_slice, seg_slice, **render_kwargs)
        st.plotly_chart(fig, use_container_width=True, key="meas_area_chart")


@st.cache_resource(show_spinner=False)
def _get_slicer(nifti_path: str) -> MPRSlicer:
    return MPRSlicer(nifti_path)
