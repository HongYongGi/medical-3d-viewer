from __future__ import annotations
import numpy as np


def compute_plane_from_3_points(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute plane center and normal from 3 points."""
    center = (p1 + p2 + p3) / 3.0
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)
    return center, normal


def rotate_normal(normal: np.ndarray, axis: str, angle_deg: float) -> np.ndarray:
    """Rotate a normal vector around an axis by given angle in degrees."""
    angle_rad = np.deg2rad(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    if axis == 'x':
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == 'y':
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis == 'z':
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    else:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got '{axis}'")
    rotated = R @ normal
    return rotated / np.linalg.norm(rotated)


def normal_from_angles(theta_deg: float, phi_deg: float) -> np.ndarray:
    """Convert spherical angles to unit normal vector.

    슬라이더 UI에서 입력받은 구면 좌표계 각도를 단위 법선 벡터로 변환합니다.

    Args:
        theta_deg: 방위각 (Z축 기준 XY 평면 회전), 범위 0-360도
        phi_deg: 앙각 (XY 평면으로부터의 기울기), 범위 -90 ~ 90도

    Returns:
        단위 법선 벡터 (shape: (3,))

    Example:
        >>> normal_from_angles(0, 0)
        array([1., 0., 0.])
        >>> normal_from_angles(0, 90)
        array([0., 0., 1.])
    """
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)
    x = np.cos(phi) * np.cos(theta)
    y = np.cos(phi) * np.sin(theta)
    z = np.sin(phi)
    return np.array([x, y, z])


def center_from_volume(volume_shape: tuple, affine: np.ndarray) -> np.ndarray:
    """Get the world-coordinate center of a volume.

    볼륨의 복셀 중심을 월드 좌표계로 변환하여 반환합니다.
    Oblique 슬라이스의 기본 중심점 설정에 사용됩니다.

    Args:
        volume_shape: 볼륨 배열의 shape (최소 3개 원소, (D, H, W) 또는 (D, H, W, C))
        affine: 복셀-월드 좌표 변환 행렬 (4x4)

    Returns:
        월드 좌표계 중심점 (shape: (3,))

    Example:
        >>> shape = (100, 100, 100)
        >>> affine = np.eye(4)
        >>> center_from_volume(shape, affine)
        array([50., 50., 50.])
    """
    center_voxel = np.array(volume_shape[:3]) / 2.0
    center_world = affine[:3, :3] @ center_voxel + affine[:3, 3]
    return center_world
