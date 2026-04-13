"""Medical 3D Viewer - Main Streamlit Application.

병원 관계자를 위한 CT 세그멘테이션 분석 도구.
"""
from __future__ import annotations

import json
import streamlit as st
from pathlib import Path
import nibabel as nib
import numpy as np

from medical_viewer.core.config import load_config
from medical_viewer.core.session import Session
from medical_viewer.core.database import StudyDatabase, StudyRecord
from medical_viewer.core.volume_cache import get_slicer, get_volume
from medical_viewer.inference.model_registry import ModelRegistry
from medical_viewer.inference.nnunet_runner import NnUNetRunner
from medical_viewer.inference.pipeline import PipelineRunner
from medical_viewer.renderer.client import RendererClient
from medical_viewer.ui.seg_editor import render_seg_editor
from medical_viewer.ui.pages import render_navigation
from medical_viewer.ui.sidebar import render_sidebar
from medical_viewer.ui.upload import render_upload
from medical_viewer.ui.viewer_mpr import render_mpr_viewer
from medical_viewer.ui.viewer_3d import render_3d_viewer, render_3d_viewer_standalone
from medical_viewer.ui.progress import ProgressTracker
from medical_viewer.ui.history import render_history_page, render_study_form
from medical_viewer.ui.model_manager import render_model_manager


def init_session_state():
    defaults = {
        "session": Session(),
        "input_path": None,
        "seg_path": None,
        "inference_done": False,
        "seg_labels": {},
        "active_page": "viewer",
        "current_study_id": None,
        "show_patient_form": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main():
    st.set_page_config(
        page_title="Medical 3D Viewer",
        page_icon="\U0001f3e5",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()
    config = _get_config()
    registry = _get_registry(config)
    renderer = RendererClient(config.renderer.url)
    db = StudyDatabase()

    active_page = render_navigation()

    if active_page == "viewer":
        page_viewer(config, registry, renderer, db)
    elif active_page == "editor":
        page_editor()
    elif active_page == "history":
        render_history_page(db)
    elif active_page == "models":
        render_model_manager(config, registry)
    elif active_page == "settings":
        page_settings(config)


def page_viewer(config, registry, renderer, db):
    st.title("\U0001f3e5 CT 세그멘테이션 분석")
    st.caption("CT NIfTI 파일을 업로드하고 AI 세그멘테이션을 실행합니다.")

    selection = render_sidebar(registry)
    session = st.session_state.session
    uploaded_path = render_upload(session.upload_dir(config.paths.uploads))

    if uploaded_path:
        st.session_state.input_path = uploaded_path

    st.sidebar.markdown("---")
    can_run = st.session_state.input_path is not None and selection["selected_id"] is not None

    if st.sidebar.button("\U0001f680 빠른 분석 시작", disabled=not can_run, use_container_width=True,
                         help="환자 정보 없이 바로 분석을 시작합니다."):
        run_inference(config, registry, db, selection, patient_info=None)

    if st.sidebar.button("\U0001f4dd 환자 정보 입력 후 분석", disabled=not can_run, use_container_width=True,
                         help="환자 정보를 입력하여 이력 관리합니다."):
        st.session_state.show_patient_form = True

    st.sidebar.markdown("---")
    st.sidebar.markdown("##### \U0001f4c1 기존 파일 직접 로드")
    seg_file = st.sidebar.file_uploader(
        "세그멘테이션 NIfTI", type=["nii", "nii.gz", "gz"],
        key="seg_upload", label_visibility="collapsed",
    )
    if seg_file is not None:
        seg_dir = session.result_dir(config.paths.results)
        seg_filename = Path(seg_file.name).name
        if not seg_filename or seg_filename.startswith('.'):
            st.sidebar.error("잘못된 파일명입니다.")
            return
        seg_save_path = seg_dir / seg_filename
        if not seg_save_path.resolve().is_relative_to(seg_dir.resolve()):
            st.sidebar.error("잘못된 파일 경로입니다.")
            return
        if not seg_save_path.exists() or seg_save_path.stat().st_size != seg_file.size:
            with open(seg_save_path, "wb") as f:
                f.write(seg_file.getbuffer())
        # Validate shape matches CT
        if st.session_state.input_path:
            from medical_viewer.core.export import validate_segmentation_shape
            valid, msg = validate_segmentation_shape(st.session_state.input_path, seg_save_path)
            if not valid:
                st.sidebar.error(msg)
            else:
                st.session_state.seg_path = seg_save_path
                st.session_state.inference_done = True
        else:
            st.session_state.seg_path = seg_save_path
            st.session_state.inference_done = True

    if st.session_state.show_patient_form and can_run:
        patient_info = render_study_form(db)
        if patient_info is not None:
            st.session_state.show_patient_form = False
            run_inference(config, registry, db, selection, patient_info=patient_info)
            return

    if st.session_state.input_path is None:
        render_welcome(db)
        return

    render_viewers(config, renderer)


def run_inference(config, registry, db, selection, patient_info=None):
    session = st.session_state.session
    input_path = st.session_state.input_path
    output_dir = session.result_dir(config.paths.results)

    study = StudyRecord(
        input_path=str(input_path),
        model_used=selection["selected_id"] if selection["mode"] == "model" else "",
        pipeline_used=selection["selected_id"] if selection["mode"] == "pipeline" else "",
        status="processing",
    )
    if patient_info:
        study.patient_id = patient_info.get("patient_id", "")
        study.patient_name = patient_info.get("patient_name", "")
        study.study_date = patient_info.get("study_date", "")
        study.description = patient_info.get("description", "")
        study.tags = patient_info.get("tags", "")
        study.notes = patient_info.get("notes", "")

    study = db.create_study(study)
    st.session_state.current_study_id = study.id
    tracker = ProgressTracker("추론 실행 중...")

    try:
        if selection["mode"] == "model":
            model_config = registry.get_model(selection["selected_id"])
            runner = NnUNetRunner(model_config)
            if not runner.check_model_available():
                st.error(f"모델 가중치를 찾을 수 없습니다: {model_config.weight_path}")
                st.info("'모델 관리' 페이지에서 모델 상태를 확인하세요.")
                db.update_study(study.id, status="failed")
                return
            seg_path = runner.predict(input_path, output_dir, progress_callback=tracker.update)
            st.session_state.seg_path = seg_path
            st.session_state.seg_labels = model_config.labels
            seg_paths = {selection["selected_id"]: str(seg_path)}
            db.update_study(study.id, status="completed", seg_paths=json.dumps(seg_paths))
        else:
            pipeline_config = registry.get_pipeline(selection["selected_id"])
            runner = PipelineRunner(pipeline_config, registry)
            results = runner.run(input_path, output_dir, progress_callback=tracker.update)
            if "merged" in results:
                st.session_state.seg_path = results["merged"]
            else:
                st.session_state.seg_path = list(results.values())[0]
            seg_paths = {k: str(v) for k, v in results.items()}
            db.update_study(study.id, status="completed", seg_paths=json.dumps(seg_paths))

        tracker.complete("분석 완료!")
        st.session_state.inference_done = True
        st.rerun()

    except ImportError as e:
        st.error(f"nnUNet이 설치되지 않았습니다: {e}")
        st.code("pip install nnunetv2", language="bash")
        db.update_study(study.id, status="failed")
    except Exception as e:
        import logging
        logging.exception("Inference failed")
        st.error(f"분석 중 오류가 발생했습니다. 관리자에게 문의하세요.")
        db.update_study(study.id, status="failed")


def page_editor():
    """Segmentation editor page."""
    input_path = st.session_state.get("input_path")
    seg_path = st.session_state.get("seg_path")
    if input_path and seg_path:
        render_seg_editor(str(input_path), str(seg_path))
    else:
        st.info("세그멘테이션 편집을 위해 먼저 '분석 뷰어'에서 CT와 세그멘테이션을 로드하세요.")


def render_viewers(config, renderer):
    input_path = st.session_state.input_path
    seg_path = st.session_state.seg_path

    view_tab1, view_tab2, view_tab3, view_tab4 = st.tabs(
        ["\U0001f4d0 MPR 다면 재구성", "\U0001f9ca 3D 시각화", "\U0001f4ca 볼륨 정보", "\U0001f4e5 내보내기"]
    )

    with view_tab1:
        render_mpr_viewer(str(input_path), str(seg_path) if seg_path else None)

    with view_tab2:
        session = st.session_state.session
        renderer_available = renderer.health_check()
        if renderer_available and seg_path:
            try:
                renderer.load_volume(session.id, str(input_path), str(seg_path))
                renderer.generate_meshes(session.id)
                render_3d_viewer(session.id, config.renderer.url)
            except Exception as e:
                st.warning(f"Go 렌더러 연결 실패: {e}")
                _render_plotly_3d(seg_path)
        else:
            if not renderer_available:
                st.info(
                    "Go 3D 렌더러가 실행되지 않고 있습니다. Plotly 기반 3D 뷰어를 사용합니다.\n\n"
                    "Go 렌더러 시작: `cd go_renderer && go run cmd/renderer/main.go`"
                )
            _render_plotly_3d(seg_path)

    with view_tab3:
        render_volume_info(input_path, seg_path)

    with view_tab4:
        render_export(seg_path)


def render_export(seg_path):
    """Render export/download tab."""
    st.subheader("📥 결과 내보내기")
    if seg_path is None:
        st.info("세그멘테이션 결과가 필요합니다.")
        return

    from medical_viewer.core.export import export_nifti_bytes, export_stl_bytes

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**NIfTI 세그멘테이션 다운로드**")
        nifti_data = export_nifti_bytes(seg_path)
        st.download_button(
            "📥 NIfTI (.nii.gz) 다운로드",
            data=nifti_data,
            file_name=Path(seg_path).name,
            mime="application/gzip",
            use_container_width=True,
        )

    with col2:
        st.markdown("**STL 메쉬 다운로드 (3D 프린팅)**")
        seg_data = get_volume(seg_path)
        unique_labels = np.unique(seg_data).astype(int)
        unique_labels = unique_labels[unique_labels > 0]
        labels_dict = st.session_state.get("seg_labels", {})

        for label in unique_labels:
            name = labels_dict.get(int(label), f"Label {int(label)}")
            try:
                stl_data = export_stl_bytes(seg_path, int(label))
                st.download_button(
                    f"📥 {name} (L{label}) STL",
                    data=stl_data,
                    file_name=f"label_{label}_{name}.stl",
                    mime="model/stl",
                    key=f"stl_{label}",
                    use_container_width=True,
                )
            except ValueError:
                st.caption(f"L{label}: 복셀 부족으로 메쉬 생성 불가")
            except Exception as e:
                st.caption(f"L{label}: {e}")


def _render_plotly_3d(seg_path):
    """Render 3D with both CT isosurface and segmentation meshes."""
    input_path = st.session_state.get("input_path")
    try:
        volume_data = {
            "labels": st.session_state.get("seg_labels", {}),
        }
        if input_path:
            ct_slicer = get_slicer(str(input_path))
            volume_data["ct_volume"] = ct_slicer.volume
            volume_data["spacing"] = tuple(ct_slicer.voxel_spacing)
            volume_data["affine"] = ct_slicer.affine
            volume_data["ct_path"] = str(input_path)

        if seg_path:
            seg_slicer = get_slicer(str(seg_path))
            volume_data["seg_volume"] = seg_slicer.volume
            volume_data["seg_path"] = str(seg_path)
            if "spacing" not in volume_data:
                volume_data["spacing"] = tuple(seg_slicer.voxel_spacing)

        if volume_data.get("ct_volume") is None and volume_data.get("seg_volume") is None:
            st.info("CT 또는 세그멘테이션 데이터가 필요합니다.")
            return

        render_3d_viewer_standalone(volume_data)
    except Exception as e:
        st.error(f"3D 렌더링 오류: {e}")


def render_volume_info(input_path, seg_path):
    st.subheader("볼륨 정보")
    try:
        ct_slicer = get_slicer(str(input_path))
        img = ct_slicer._img
        header = img.header
        data = ct_slicer.volume
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**CT 볼륨**")
            for k, v in {
                "크기": str(ct_slicer.shape),
                "복셀 간격 (mm)": str(tuple(round(x, 3) for x in ct_slicer.voxel_spacing)),
                "데이터 타입": str(header.get_data_dtype()),
                "방향": str(nib.aff2axcodes(ct_slicer.affine)),
                "HU 범위": f"[{data.min():.0f}, {data.max():.0f}]",
            }.items():
                st.markdown(f"**{k}:** `{v}`")
        with col2:
            if seg_path:
                seg_data = get_volume(seg_path)
                st.markdown("**세그멘테이션**")
                unique_labels = np.unique(seg_data).astype(int)
                unique_labels = unique_labels[unique_labels > 0]
                labels_dict = st.session_state.get("seg_labels", {})
                for label in unique_labels:
                    count = int(np.sum(seg_data == label))
                    name = labels_dict.get(int(label), f"레이블 {label}")
                    vol_ml = count * np.prod(ct_slicer.voxel_spacing) / 1000
                    st.markdown(f"**{label} - {name}:** {count:,} 복셀 ({vol_ml:.1f} mL)")
            else:
                st.info("세그멘테이션 없음")
    except Exception as e:
        st.error(f"볼륨 정보 로드 실패: {e}")


def render_welcome(db):
    col_main, col_recent = st.columns([2, 1])
    with col_main:
        st.markdown("""
### 환영합니다!

CT NIfTI 파일에 대한 **AI 기반 자동 세그멘테이션** + **MPR** + **3D 시각화** 도구입니다.

| 단계 | 설명 |
|------|------|
| **1. 파일 업로드** | 사이드바에서 CT NIfTI 파일 업로드 |
| **2. 모델 선택** | 세그멘테이션 모델 선택 |
| **3. 분석 시작** | '빠른 분석' 또는 '환자 정보 입력 후 분석' |
| **4. 결과 확인** | MPR + 3D 뷰어에서 확인 |

**팁:** nnUNet 없이도 CT만 업로드하면 MPR 뷰어 사용 가능
        """)
    with col_recent:
        st.markdown("#### 최근 분석")
        recent = db.list_studies(limit=5)
        if recent:
            for study in recent:
                icons = {"completed": "✅", "processing": "⏳", "pending": "🔵", "failed": "❌"}
                label = study.patient_name or study.patient_id or f"분석 #{study.id}"
                if st.button(f"{icons.get(study.status, '❓')} {label}",
                             key=f"recent_{study.id}", use_container_width=True):
                    p = Path(study.input_path)
                    if p.exists():
                        st.session_state.input_path = p
                        seg_dict = study.seg_paths_dict()
                        if seg_dict:
                            sp = Path(list(seg_dict.values())[0])
                            if sp.exists():
                                st.session_state.seg_path = sp
                                st.session_state.inference_done = True
                        st.rerun()
        else:
            st.caption("분석 이력이 없습니다.")


def page_settings(config):
    st.header("⚙️ 설정")

    st.subheader("서비스 상태")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Streamlit 앱**")
        st.success("실행 중")
    with col2:
        st.markdown("**Go 3D 렌더러**")
        r = RendererClient(config.renderer.url)
        if r.health_check():
            st.success(f"실행 중 ({config.renderer.url})")
        else:
            st.error("연결 불가")
            st.code("cd go_renderer && go run cmd/renderer/main.go")

    st.subheader("경로 설정")
    st.json({
        "업로드": str(config.paths.uploads), "결과": str(config.paths.results),
        "모델": str(config.paths.models),
        "가중치": config.nnunet.weight_dir or "(미설정)",
    })

    # Data management
    st.subheader("데이터 관리")
    from medical_viewer.core.cleanup import cleanup_old_sessions, get_data_usage

    data_dir = Path("data")
    usage = get_data_usage(data_dir)
    if usage:
        cols = st.columns(len(usage))
        for col, (name, size_mb) in zip(cols, usage.items()):
            col.metric(name, f"{size_mb:.1f} MB")

    st.markdown("---")
    col_clean1, col_clean2 = st.columns(2)
    with col_clean1:
        max_age = st.number_input("정리 기준 (시간)", value=24, min_value=1, step=1, key="cleanup_hours")
    with col_clean2:
        if st.button("🗑️ 오래된 세션 정리", use_container_width=True):
            removed = cleanup_old_sessions(data_dir, max_age_hours=max_age)
            total = sum(removed.values())
            if total > 0:
                st.success(f"{total}개 세션 디렉토리 삭제 완료: {removed}")
            else:
                st.info(f"{max_age}시간 이내의 세션만 존재합니다.")


@st.cache_resource(ttl=300)
def _get_config():
    """Cache config loading (includes weight directory scan). TTL=5min."""
    return load_config()


@st.cache_resource(ttl=300)
def _get_registry(_config):
    """Cache model registry. TTL=5min."""
    return ModelRegistry(_config)


if __name__ == "__main__":
    main()
