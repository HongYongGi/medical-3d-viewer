"""Export segmentation results to various formats."""
from __future__ import annotations

import io
from pathlib import Path

import nibabel as nib
import numpy as np


def export_nifti_bytes(seg_path: str | Path) -> bytes:
    """Read a NIfTI segmentation file and return its bytes for download."""
    with open(seg_path, "rb") as f:
        return f.read()


def export_stl_bytes(
    seg_path: str | Path,
    label: int,
    spacing: tuple[float, float, float] | None = None,
) -> bytes:
    """Generate STL mesh for a single label from segmentation NIfTI."""
    from skimage import measure

    img = nib.load(str(seg_path))
    data = img.get_fdata()
    if spacing is None:
        spacing = tuple(img.header.get_zooms()[:3])

    mask = (data == label).astype(np.float32)
    if mask.sum() < 10:
        raise ValueError(f"Label {label} has too few voxels")

    verts, faces, normals, _ = measure.marching_cubes(
        mask, level=0.5, spacing=spacing, step_size=1,
    )

    return _mesh_to_stl_binary(verts, faces, normals)


def _mesh_to_stl_binary(
    vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray
) -> bytes:
    """Convert mesh to binary STL format."""
    buf = io.BytesIO()

    # Header (80 bytes)
    buf.write(b'\x00' * 80)

    # Number of triangles
    n_tri = len(faces)
    buf.write(np.uint32(n_tri).tobytes())

    # Vectorized: compute all face normals at once
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    edge1 = v1 - v0
    edge2 = v2 - v0
    face_normals = np.cross(edge1, edge2)
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    face_normals /= norms

    # Pack all triangles using structured array
    record = np.zeros(n_tri, dtype=[
        ('normal', '<f4', 3), ('v0', '<f4', 3), ('v1', '<f4', 3),
        ('v2', '<f4', 3), ('attr', '<u2'),
    ])
    record['normal'] = face_normals.astype(np.float32)
    record['v0'] = v0.astype(np.float32)
    record['v1'] = v1.astype(np.float32)
    record['v2'] = v2.astype(np.float32)
    buf.write(record.tobytes())

    return buf.getvalue()


def validate_segmentation_shape(
    ct_path: str | Path, seg_path: str | Path
) -> tuple[bool, str]:
    """Validate that segmentation shape matches CT volume.

    Returns (is_valid, message).
    """
    try:
        ct_img = nib.load(str(ct_path))
        seg_img = nib.load(str(seg_path))
    except Exception as e:
        return False, f"파일 로드 실패: {e}"

    ct_shape = ct_img.shape[:3]
    seg_shape = seg_img.shape[:3]

    if ct_shape != seg_shape:
        return False, (
            f"Shape 불일치: CT {ct_shape} != Seg {seg_shape}. "
            f"CT와 동일한 크기의 세그멘테이션을 사용하세요."
        )

    return True, "검증 통과"
