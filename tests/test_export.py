"""Tests for STL export functionality."""
import sys
sys.path.insert(0, "src")

import struct
import tempfile
import numpy as np
import nibabel as nib


def _make_seg_nifti(shape=(20, 20, 20)):
    """Create a segmentation NIfTI with a cube of label 1."""
    data = np.zeros(shape, dtype=np.int16)
    data[5:15, 5:15, 5:15] = 1  # 10x10x10 cube
    img = nib.Nifti1Image(data, np.eye(4))
    f = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
    nib.save(img, f.name)
    return f.name


def test_mesh_to_stl_binary_format():
    from medical_viewer.core.export import _mesh_to_stl_binary
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    normals = np.array([[0, 0, 1]], dtype=np.float32)

    stl = _mesh_to_stl_binary(vertices, faces, normals)

    # STL binary: 80 header + 4 num_triangles + 50*N triangles
    assert len(stl) == 80 + 4 + 50 * 1
    # Check header is zeros
    assert stl[:80] == b'\x00' * 80
    # Check triangle count
    n_tri = struct.unpack('<I', stl[80:84])[0]
    assert n_tri == 1


def test_export_stl_bytes():
    from medical_viewer.core.export import export_stl_bytes
    path = _make_seg_nifti()
    stl = export_stl_bytes(path, label=1)
    assert len(stl) > 84  # Header + at least 1 triangle
    n_tri = struct.unpack('<I', stl[80:84])[0]
    assert n_tri > 0


def test_export_stl_missing_label():
    from medical_viewer.core.export import export_stl_bytes
    import pytest
    path = _make_seg_nifti()
    with pytest.raises(ValueError):
        export_stl_bytes(path, label=99)


def test_validate_segmentation_shape_match():
    from medical_viewer.core.export import validate_segmentation_shape
    shape = (20, 20, 20)
    ct = np.zeros(shape, dtype=np.float32)
    seg = np.zeros(shape, dtype=np.int16)
    ct_img = nib.Nifti1Image(ct, np.eye(4))
    seg_img = nib.Nifti1Image(seg, np.eye(4))

    ct_f = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
    seg_f = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
    nib.save(ct_img, ct_f.name)
    nib.save(seg_img, seg_f.name)

    valid, msg = validate_segmentation_shape(ct_f.name, seg_f.name)
    assert valid is True


def test_validate_segmentation_shape_mismatch():
    from medical_viewer.core.export import validate_segmentation_shape
    ct = nib.Nifti1Image(np.zeros((20, 20, 20), dtype=np.float32), np.eye(4))
    seg = nib.Nifti1Image(np.zeros((10, 10, 10), dtype=np.int16), np.eye(4))

    ct_f = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
    seg_f = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
    nib.save(ct, ct_f.name)
    nib.save(seg, seg_f.name)

    valid, msg = validate_segmentation_shape(ct_f.name, seg_f.name)
    assert valid is False
    assert "Shape" in msg
