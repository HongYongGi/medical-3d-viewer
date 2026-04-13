"""Tests for oblique slice utility functions."""
import sys
sys.path.insert(0, "src")

import numpy as np
import pytest


def test_normal_from_angles_axial():
    from medical_viewer.mpr.oblique import normal_from_angles
    n = normal_from_angles(0, 90)
    np.testing.assert_array_almost_equal(n, [0, 0, 1], decimal=5)


def test_normal_from_angles_sagittal():
    from medical_viewer.mpr.oblique import normal_from_angles
    n = normal_from_angles(0, 0)
    np.testing.assert_array_almost_equal(n, [1, 0, 0], decimal=5)


def test_normal_is_unit_vector():
    from medical_viewer.mpr.oblique import normal_from_angles
    for theta in range(0, 360, 30):
        for phi in range(-90, 91, 30):
            n = normal_from_angles(theta, phi)
            assert abs(np.linalg.norm(n) - 1.0) < 1e-10


def test_compute_plane_from_3_points():
    from medical_viewer.mpr.oblique import compute_plane_from_3_points
    p1 = np.array([0, 0, 0])
    p2 = np.array([1, 0, 0])
    p3 = np.array([0, 1, 0])
    center, normal = compute_plane_from_3_points(p1, p2, p3)
    np.testing.assert_array_almost_equal(center, [1/3, 1/3, 0])
    assert abs(abs(normal[2]) - 1.0) < 1e-10  # Normal should be along Z


def test_compute_plane_colinear_raises():
    from medical_viewer.mpr.oblique import compute_plane_from_3_points
    p1 = np.array([0, 0, 0])
    p2 = np.array([1, 0, 0])
    p3 = np.array([2, 0, 0])
    with pytest.raises(ValueError, match="일직선"):
        compute_plane_from_3_points(p1, p2, p3)


def test_center_from_volume_identity():
    from medical_viewer.mpr.oblique import center_from_volume
    shape = (100, 100, 100)
    affine = np.eye(4)
    center = center_from_volume(shape, affine)
    np.testing.assert_array_almost_equal(center, [50, 50, 50])


def test_center_from_volume_scaled():
    from medical_viewer.mpr.oblique import center_from_volume
    shape = (100, 200, 50)
    affine = np.diag([2.0, 1.0, 3.0, 1.0])
    center = center_from_volume(shape, affine)
    np.testing.assert_array_almost_equal(center, [100, 100, 75])


def test_rotate_normal():
    from medical_viewer.mpr.oblique import rotate_normal
    n = np.array([1.0, 0.0, 0.0])
    rotated = rotate_normal(n, 'z', 90)
    np.testing.assert_array_almost_equal(rotated, [0, 1, 0], decimal=5)
