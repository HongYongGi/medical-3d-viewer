"""Study history viewer - browse and reload previous analyses."""
from __future__ import annotations

import streamlit as st
from datetime import datetime
from pathlib import Path

from ..core.database import StudyDatabase, StudyRecord


def render_history_page(db: StudyDatabase):
    """Render the study history page."""
    st.header("📋 분석 이력")
    st.caption("이전에 분석한 데이터를 확인하고 다시 열 수 있습니다.")

    col_search, col_status, col_patient = st.columns([2, 1, 1])
    with col_search:
        search = st.text_input("🔍 검색", placeholder="환자명, ID, 설명, 태그로 검색...", key="history_search")
    with col_status:
        status_filter = st.selectbox("상태", ["전체", "완료", "처리중", "대기", "실패"], key="history_status")
        status_map = {"전체": None, "완료": "completed", "처리중": "processing", "대기": "pending", "실패": "failed"}
    with col_patient:
        patients = db.get_unique_patients()
        patient_options = ["전체 환자"] + [f"{p['patient_name']} ({p['patient_id']})" for p in patients]
        patient_sel = st.selectbox("환자", patient_options, key="history_patient")
        patient_id = None
        if patient_sel != "전체 환자":
            idx = patient_options.index(patient_sel) - 1
            patient_id = patients[idx]["patient_id"]

    total = db.count_studies()
    completed = db.count_studies("completed")
    processing = db.count_studies("processing")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("전체 분석", total)
    col2.metric("완료", completed)
    col3.metric("처리중", processing)
    col4.metric("실패", db.count_studies("failed"))

    st.markdown("---")

    studies = db.list_studies(
        status=status_map.get(status_filter),
        patient_id=patient_id,
        search=search if search else None,
        limit=50,
    )

    if not studies:
        st.info("분석 이력이 없습니다. 새로운 CT 파일을 업로드하여 분석을 시작하세요.")
        return

    for study in studies:
        _render_study_card(study, db)


def _render_study_card(study: StudyRecord, db: StudyDatabase):
    status_icons = {"completed": "✅", "processing": "⏳", "pending": "🔵", "failed": "❌"}
    icon = status_icons.get(study.status, "❓")
    try:
        created = datetime.fromisoformat(study.created_at)
        date_str = created.strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        date_str = study.created_at

    title = f"{icon} "
    if study.patient_name:
        title += f"**{study.patient_name}**"
        if study.patient_id:
            title += f" ({study.patient_id})"
    elif study.patient_id:
        title += f"**{study.patient_id}**"
    else:
        title += f"**분석 #{study.id}**"
    title += f" | {date_str}"
    if study.model_used:
        title += f" | 모델: {study.model_used}"
    elif study.pipeline_used:
        title += f" | 파이프라인: {study.pipeline_used}"

    with st.expander(title, expanded=False):
        col_info, col_actions = st.columns([3, 1])
        with col_info:
            if study.description:
                st.markdown(f"**설명:** {study.description}")
            if study.study_date:
                st.markdown(f"**검사일:** {study.study_date}")
            if study.tags:
                tags = study.tags_list()
                st.markdown("**태그:** " + " ".join(f"`{t}`" for t in tags))
            if study.notes:
                st.markdown(f"**메모:** {study.notes}")
            st.markdown(f"**입력 파일:** `{Path(study.input_path).name}`")
            seg_dict = study.seg_paths_dict()
            if seg_dict:
                st.markdown("**세그멘테이션:**")
                for model_id, path in seg_dict.items():
                    st.markdown(f"  - {model_id}: `{Path(path).name}`")
        with col_actions:
            if st.button("📂 열기", key=f"open_{study.id}", use_container_width=True):
                _load_study(study)
            confirm_key = f"confirm_del_{study.id}"
            if st.button("🗑️ 삭제", key=f"del_{study.id}", use_container_width=True):
                st.session_state[confirm_key] = True
            if st.session_state.get(confirm_key, False):
                st.warning("정말 삭제하시겠습니까?")
                c1, c2 = st.columns(2)
                if c1.button("✅ 확인", key=f"yes_{study.id}", use_container_width=True):
                    db.delete_study(study.id)
                    st.session_state.pop(confirm_key, None)
                    st.rerun()
                if c2.button("취소", key=f"no_{study.id}", use_container_width=True):
                    st.session_state.pop(confirm_key, None)
                    st.rerun()


def _load_study(study: StudyRecord):
    input_path = Path(study.input_path)
    if not input_path.exists():
        st.error(f"입력 파일을 찾을 수 없습니다: {input_path}")
        return
    st.session_state.input_path = input_path
    st.session_state.current_study_id = study.id
    seg_dict = study.seg_paths_dict()
    if seg_dict:
        if "merged" in seg_dict:
            seg_path = Path(seg_dict["merged"])
        else:
            seg_path = Path(list(seg_dict.values())[0])
        if seg_path.exists():
            st.session_state.seg_path = seg_path
            st.session_state.inference_done = True
    st.session_state.active_page = "viewer"
    st.rerun()


def render_study_form(db: StudyDatabase) -> dict | None:
    """Render form for entering patient info before analysis."""
    st.subheader("📝 환자 정보 입력")
    st.caption("분석 전 환자 정보를 입력하면 나중에 쉽게 찾을 수 있습니다.")

    with st.form("patient_info_form"):
        col1, col2 = st.columns(2)
        with col1:
            patient_id = st.text_input("환자 ID", placeholder="예: PT-2024-0001", help="병원 내부 환자 식별 번호")
            patient_name = st.text_input("환자명", placeholder="예: 홍길동")
        with col2:
            study_date = st.date_input("검사일")
            description = st.text_input("검사 설명", placeholder="예: TAVR 시술 전 CT")
        tags = st.text_input("태그 (쉼표로 구분)", placeholder="예: TAVR, 대동맥, 긴급", help="나중에 검색할 때 사용할 태그")
        notes = st.text_area("메모", placeholder="추가 메모 사항...", height=80)
        submitted = st.form_submit_button("✅ 저장 후 분석 시작", use_container_width=True)
        if submitted:
            return {
                "patient_id": patient_id, "patient_name": patient_name,
                "study_date": study_date.isoformat() if study_date else "",
                "description": description, "tags": tags, "notes": notes,
            }
    return None
