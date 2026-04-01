from __future__ import annotations
from pathlib import Path
import streamlit as st


def render_upload(upload_dir: Path) -> Path | None:
    """Render file upload widget. Returns path to saved file or None."""
    st.sidebar.header("📂 파일 업로드")

    uploaded_file = st.sidebar.file_uploader(
        "CT NIfTI 파일 업로드",
        type=["nii", "nii.gz", "gz"],
        help="NIfTI 형식의 CT 이미지를 업로드하세요 (.nii 또는 .nii.gz)",
    )

    if uploaded_file is not None:
        upload_dir.mkdir(parents=True, exist_ok=True)
        filename = uploaded_file.name
        if not filename.endswith(('.nii', '.nii.gz')):
            filename = filename + '.nii.gz'
        save_path = upload_dir / filename
        if not save_path.exists() or save_path.stat().st_size != uploaded_file.size:
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.sidebar.success(f"업로드 완료: {filename}")
        return save_path
    return None
