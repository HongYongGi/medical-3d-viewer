from __future__ import annotations

import zipfile
import tempfile
from pathlib import Path

import streamlit as st


def render_upload(upload_dir: Path) -> Path | None:
    """Render file upload widget supporting NIfTI and DICOM. Returns path to NIfTI or None."""
    st.sidebar.header("📂 파일 업로드")

    upload_type = st.sidebar.radio(
        "입력 형식", ["NIfTI (.nii/.nii.gz)", "DICOM (ZIP/폴더)"],
        horizontal=True, key="upload_type",
    )

    if upload_type == "NIfTI (.nii/.nii.gz)":
        return _upload_nifti(upload_dir)
    else:
        return _upload_dicom(upload_dir)


def _upload_nifti(upload_dir: Path) -> Path | None:
    """Handle NIfTI file upload."""
    uploaded_file = st.sidebar.file_uploader(
        "CT NIfTI 파일",
        type=["nii", "nii.gz", "gz"],
        help="NIfTI 형식의 CT 이미지 (.nii 또는 .nii.gz), 최대 2GB",
        key="nifti_upload",
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
            size_mb = save_path.stat().st_size / (1024 * 1024)
            st.sidebar.success(f"업로드 완료: {filename} ({size_mb:.0f}MB)")
        return save_path
    return None


def _upload_dicom(upload_dir: Path) -> Path | None:
    """Handle DICOM ZIP upload with automatic NIfTI conversion."""
    uploaded_file = st.sidebar.file_uploader(
        "DICOM ZIP 파일",
        type=["zip"],
        help="DICOM 시리즈가 포함된 ZIP 파일을 업로드하세요. 자동으로 NIfTI로 변환됩니다.",
        key="dicom_upload",
    )

    if uploaded_file is None:
        return None

    upload_dir.mkdir(parents=True, exist_ok=True)

    # Check if already converted
    converted_path = upload_dir / f"{uploaded_file.name}.nii.gz"
    if converted_path.exists():
        st.sidebar.success(f"이전 변환 결과 사용: {converted_path.name}")
        return converted_path

    # Save and extract ZIP
    zip_path = upload_dir / uploaded_file.name
    with open(zip_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.sidebar.info("DICOM → NIfTI 변환 중...")

    try:
        nifti_path = _convert_dicom_to_nifti(zip_path, upload_dir)
        if nifti_path and nifti_path.exists():
            st.sidebar.success(f"변환 완료: {nifti_path.name}")
            return nifti_path
        else:
            st.sidebar.error("DICOM 변환 실패: NIfTI 파일이 생성되지 않았습니다.")
            return None
    except ImportError:
        st.sidebar.error(
            "DICOM 변환에 필요한 패키지가 설치되지 않았습니다.\n\n"
            "`pip install dicom2nifti pydicom`"
        )
        return None
    except Exception as e:
        st.sidebar.error(f"DICOM 변환 오류: {e}")
        return None


def _convert_dicom_to_nifti(zip_path: Path, output_dir: Path) -> Path | None:
    """Convert DICOM ZIP to NIfTI using dicom2nifti."""
    import dicom2nifti

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Extract ZIP
        dicom_dir = tmp_path / "dicom"
        dicom_dir.mkdir()
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(dicom_dir)

        # Find DICOM files (may be in subdirectory)
        dcm_files = list(dicom_dir.rglob("*.dcm")) + list(dicom_dir.rglob("*.DCM"))
        if not dcm_files:
            # Try files without extension (common in DICOM)
            all_files = [f for f in dicom_dir.rglob("*") if f.is_file()]
            if all_files:
                # Use parent directory of first file as DICOM dir
                dicom_dir = all_files[0].parent
            else:
                return None

        # Convert
        nifti_dir = tmp_path / "nifti"
        nifti_dir.mkdir()
        dicom2nifti.convert_directory(str(dicom_dir), str(nifti_dir), compression=True, reorient=True)

        # Find output NIfTI
        nifti_files = list(nifti_dir.glob("*.nii.gz")) + list(nifti_dir.glob("*.nii"))
        if not nifti_files:
            return None

        # Copy to output directory
        src = nifti_files[0]
        dst = output_dir / f"{zip_path.stem}.nii.gz"
        import shutil
        shutil.copy2(src, dst)
        return dst
