"""Segmentation editor tools (3D Slicer-inspired)."""
from __future__ import annotations

import numpy as np
import nibabel as nib
import streamlit as st
from pathlib import Path
from scipy import ndimage
import plotly.graph_objects as go

from ..mpr.windowing import apply_window

from ..core.constants import LABEL_COLORS_RGB as LABEL_COLORS


def render_seg_editor(ct_path: str, seg_path: str):
    """Render segmentation editor UI."""
    st.header("🎨 세그멘테이션 편집")
    st.caption("AI 세그멘테이션 결과를 수동으로 수정합니다.")

    from ..core.volume_cache import get_slicer
    ct_slicer = get_slicer(ct_path)
    seg_slicer = get_slicer(seg_path)
    seg_img = seg_slicer._img
    ct_data = ct_slicer.volume
    seg_data = seg_slicer.volume.astype(np.int32)

    unique_labels = sorted(set(np.unique(seg_data).astype(int)) - {0})
    if not unique_labels:
        st.warning("세그멘테이션에 레이블이 없습니다.")
        return

    if "edited_seg" not in st.session_state or st.session_state.get("_edit_src") != seg_path:
        st.session_state.edited_seg = seg_data.copy()
        st.session_state._edit_src = seg_path

    edited = st.session_state.edited_seg
    edit_labels = sorted(set(np.unique(edited).astype(int)) - {0})

    tool = st.selectbox("편집 도구", [
        "레이블 삭제", "레이블 병합", "Erosion (수축)", "Dilation (팽창)",
        "HU Threshold 정제", "작은 영역 제거", "Smooth (평활화)",
    ], key="seg_tool")

    st.markdown("---")

    if tool == "레이블 삭제":
        label = st.selectbox("삭제할 레이블", edit_labels, key="del_label")
        if st.button("삭제 실행", key="do_delete"):
            edited[edited == label] = 0
            st.session_state.edited_seg = edited
            st.success(f"레이블 {label} 삭제 완료")
            st.rerun()

    elif tool == "레이블 병합":
        c1, c2 = st.columns(2)
        src = c1.selectbox("원본 레이블", edit_labels, key="merge_src")
        dst = c2.selectbox("대상 레이블", edit_labels, key="merge_dst")
        if st.button("병합 실행", key="do_merge") and src != dst:
            edited[edited == src] = dst
            st.session_state.edited_seg = edited
            st.success(f"레이블 {src} → {dst} 병합 완료")
            st.rerun()

    elif tool == "Erosion (수축)":
        label = st.selectbox("대상 레이블", edit_labels, key="erode_label")
        iters = st.slider("반복 횟수", 1, 5, 1, key="erode_iter")
        if st.button("Erosion 실행", key="do_erode"):
            mask = (edited == label)
            eroded = ndimage.binary_erosion(mask, iterations=iters)
            removed = int(mask.sum() - eroded.sum())
            edited[mask & ~eroded] = 0
            st.session_state.edited_seg = edited
            st.success(f"Erosion 완료: {removed:,} 복셀 제거")
            st.rerun()

    elif tool == "Dilation (팽창)":
        label = st.selectbox("대상 레이블", edit_labels, key="dilate_label")
        iters = st.slider("반복 횟수", 1, 5, 1, key="dilate_iter")
        if st.button("Dilation 실행", key="do_dilate"):
            mask = (edited == label)
            dilated = ndimage.binary_dilation(mask, iterations=iters)
            edited[(dilated & ~mask) & (edited == 0)] = label
            added = int(dilated.sum() - mask.sum())
            st.session_state.edited_seg = edited
            st.success(f"Dilation 완료: {added:,} 복셀 추가")
            st.rerun()

    elif tool == "HU Threshold 정제":
        label = st.selectbox("대상 레이블", edit_labels, key="thresh_label")
        c1, c2 = st.columns(2)
        hu_min = c1.number_input("HU 최소값", value=-100.0, step=10.0, key="hu_min")
        hu_max = c2.number_input("HU 최대값", value=500.0, step=10.0, key="hu_max")
        if st.button("Threshold 적용", key="do_thresh"):
            mask = (edited == label)
            outside = (ct_data < hu_min) | (ct_data > hu_max)
            removed = int((mask & outside).sum())
            edited[mask & outside] = 0
            st.session_state.edited_seg = edited
            st.success(f"HU 범위 외 {removed:,} 복셀 제거")
            st.rerun()

    elif tool == "작은 영역 제거":
        label = st.selectbox("대상 레이블", edit_labels, key="island_label")
        min_size = st.number_input("최소 복셀 수", value=100, step=10, key="min_island")
        if st.button("작은 영역 제거", key="do_island"):
            mask = (edited == label)
            labeled_arr, num = ndimage.label(mask)
            removed = 0
            for i in range(1, num + 1):
                comp = (labeled_arr == i)
                if comp.sum() < min_size:
                    edited[comp] = 0
                    removed += int(comp.sum())
            st.session_state.edited_seg = edited
            st.success(f"{removed:,} 복셀 제거 (최소 {min_size} 미만)")
            st.rerun()

    elif tool == "Smooth (평활화)":
        label = st.selectbox("대상 레이블", edit_labels, key="smooth_label")
        sigma = st.slider("Sigma", 0.5, 3.0, 1.0, 0.5, key="smooth_sigma")
        if st.button("Smooth 실행", key="do_smooth"):
            mask = (edited == label).astype(np.float32)
            smoothed = ndimage.gaussian_filter(mask, sigma=sigma) > 0.5
            edited[edited == label] = 0
            edited[smoothed] = label
            st.session_state.edited_seg = edited
            st.success(f"Smooth 완료 (sigma={sigma})")
            st.rerun()

    # Preview
    st.markdown("---")
    st.subheader("편집 결과 미리보기")
    preview_idx = st.slider("Axial 슬라이스", 0, edited.shape[2] - 1,
                            edited.shape[2] // 2, key="preview_idx")

    ct_slice = ct_data[:, :, preview_idx].T
    seg_slice = edited[:, :, preview_idx].T

    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=apply_window(ct_slice, 200, 800), colorscale="gray", showscale=False))
    for lbl in sorted(set(np.unique(seg_slice).astype(int)) - {0}):
        mask = (seg_slice == lbl).astype(np.float32)
        r, g, b = LABEL_COLORS.get(lbl, (200, 200, 200))
        fig.add_trace(go.Heatmap(
            z=np.where(mask > 0, 0.4, np.nan),
            colorscale=[[0, f"rgba({r},{g},{b},0)"], [1, f"rgba({r},{g},{b},180)"]],
            showscale=False, hoverinfo="skip", zmin=0, zmax=1,
        ))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=450,
                      yaxis=dict(scaleanchor="x", scaleratio=1, showticklabels=False),
                      xaxis=dict(showticklabels=False))
    st.plotly_chart(fig, use_container_width=True, key="preview_chart")

    # Stats
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**원본**")
        for l in unique_labels:
            st.text(f"  L{l}: {(seg_data == l).sum():,}")
    with c2:
        st.markdown("**편집 후**")
        for l in sorted(set(np.unique(edited).astype(int)) - {0}):
            st.text(f"  L{l}: {(edited == l).sum():,}")

    # Save / Reset
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("💾 편집 결과 저장", use_container_width=True, key="save_edit"):
            out = Path(seg_path).parent / f"edited_{Path(seg_path).name}"
            nib.save(nib.Nifti1Image(edited.astype(np.int16), seg_img.affine, seg_img.header), str(out))
            st.session_state.seg_path = out
            st.success(f"저장: {out.name}")
    with c2:
        if st.button("🔄 원본 복원", use_container_width=True, key="reset_edit"):
            st.session_state.edited_seg = seg_data.copy()
            st.rerun()
