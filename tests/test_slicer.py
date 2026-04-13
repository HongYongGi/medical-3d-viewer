"""Tests for MPR slicer coordinate transforms and slicing."""
import sys
sys.path.insert(0, "src")

import tempfile
import numpy as np
import nibabel as nib
import pytest


def _make_test_nifti(shape=(10, 12, 14), affine=None):
    """Create a temporary NIfTI file with known data."""
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    if affine is None:
        affine = np.diag([1.5, 2.0, 2.5, 1.0])
    img = nib.Nifti1Image(data, affine)
    f = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
    nib.save(img, f.name)
    return f.name, data, affine


def test_slicer_shape():
    from medical_viewer.mpr.slicer import MPRSlicer
    path, data, _ = _make_test_nifti((10, 12, 14))
    slicer = MPRSlicer(path)
    assert slicer.shape == (10, 12, 14)
    assert slicer.num_axial == 14
    assert slicer.num_sagittal == 10
    assert slicer.num_coronal == 12


def test_axial_slice():
    from medical_viewer.mpr.slicer import MPRSlicer
    path, data, _ = _make_test_nifti((10, 12, 14))
    slicer = MPRSlicer(path)
    ax = slicer.get_axial(5)
    expected = data[:, :, 5].T
    np.testing.assert_array_almost_equal(ax, expected)


def test_sagittal_slice():
    from medical_viewer.mpr.slicer import MPRSlicer
    path, data, _ = _make_test_nifti((10, 12, 14))
    slicer = MPRSlicer(path)
    sag = slicer.get_sagittal(3)
    expected = data[3, :, :].T
    np.testing.assert_array_almost_equal(sag, expected)


def test_coronal_slice():
    from medical_viewer.mpr.slicer import MPRSlicer
    path, data, _ = _make_test_nifti((10, 12, 14))
    slicer = MPRSlicer(path)
    cor = slicer.get_coronal(7)
    expected = data[:, 7, :].T
    np.testing.assert_array_almost_equal(cor, expected)


def test_world_voxel_roundtrip():
    from medical_viewer.mpr.slicer import MPRSlicer
    path, _, affine = _make_test_nifti()
    slicer = MPRSlicer(path)
    voxel = np.array([[3.0, 4.0, 5.0]])
    world = slicer.voxel_to_world(voxel)
    back = slicer.world_to_voxel(world)
    np.testing.assert_array_almost_equal(back, voxel, decimal=5)


def test_slice_index_clipping():
    from medical_viewer.mpr.slicer import MPRSlicer
    path, data, _ = _make_test_nifti((10, 12, 14))
    slicer = MPRSlicer(path)
    # Out of bounds should clip, not crash
    ax_neg = slicer.get_axial(-5)
    ax_over = slicer.get_axial(999)
    assert ax_neg.shape == data[:, :, 0].T.shape
    assert ax_over.shape == data[:, :, 13].T.shape


def test_oblique_zero_norm_raises():
    from medical_viewer.mpr.slicer import MPRSlicer
    path, _, _ = _make_test_nifti()
    slicer = MPRSlicer(path)
    with pytest.raises(ValueError, match="Normal"):
        slicer.get_oblique(np.array([0, 0, 0]), np.array([0, 0, 0]))
