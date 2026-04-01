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
