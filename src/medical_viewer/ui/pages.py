"""Multi-page navigation for the app."""
from __future__ import annotations
import streamlit as st


PAGES = {
    "viewer": {"icon": "🏥", "label": "분석 뷰어", "description": "CT 업로드 및 분석"},
    "history": {"icon": "📋", "label": "분석 이력", "description": "이전 분석 결과 조회"},
    "models": {"icon": "🧠", "label": "모델 관리", "description": "nnUNet 모델 상태 확인"},
    "settings": {"icon": "⚙️", "label": "설정", "description": "앱 설정"},
}


def render_navigation() -> str:
    """Render page navigation in sidebar. Returns selected page key."""
    st.sidebar.markdown("### 📌 메뉴")

    if "active_page" not in st.session_state:
        st.session_state.active_page = "viewer"

    for key, page in PAGES.items():
        label = f"{page['icon']} {page['label']}"
        if st.sidebar.button(
            label,
            key=f"nav_{key}",
            use_container_width=True,
            type="primary" if st.session_state.active_page == key else "secondary",
            help=page["description"],
        ):
            st.session_state.active_page = key
            st.rerun()

    st.sidebar.markdown("---")
    return st.session_state.active_page
