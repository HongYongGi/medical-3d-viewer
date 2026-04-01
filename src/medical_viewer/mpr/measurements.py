"""Measurement tools for MPR viewer."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class DistanceMeasurement:
    """Distance measurement between two points."""
    p1: tuple[float, float]  # (x, y) in pixel coords
    p2: tuple[float, float]
    distance_px: float
    distance_mm: float
    label: str = ""


@dataclass
class AreaMeasurement:
    """Area measurement from segmentation label."""
    label_id: int
    label_name: str
    area_px: int
    area_mm2: float
    perimeter_mm: float
    centroid: tuple[float, float]


def compute_distance(p1: tuple, p2: tuple, spacing: tuple[float, float] = (1.0, 1.0)) -> DistanceMeasurement:
    """Compute distance between two points in mm."""
    dx = (p2[0] - p1[0]) * spacing[0]
    dy = (p2[1] - p1[1]) * spacing[1]
    dist_mm = np.sqrt(dx**2 + dy**2)
    dist_px = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    return DistanceMeasurement(p1=p1, p2=p2, distance_px=dist_px, distance_mm=dist_mm)


def compute_slice_area(
    seg_slice: np.ndarray,
    label_id: int,
    spacing: tuple[float, float] = (1.0, 1.0),
    label_name: str = "",
) -> AreaMeasurement | None:
    """Compute area of a segmentation label in a 2D slice."""
    mask = (seg_slice == label_id)
    area_px = int(mask.sum())
    if area_px == 0:
        return None

    area_mm2 = area_px * spacing[0] * spacing[1]

    # Perimeter: count boundary pixels
    from scipy.ndimage import binary_erosion
    eroded = binary_erosion(mask)
    boundary = mask & ~eroded
    perimeter_px = boundary.sum()
    perimeter_mm = perimeter_px * (spacing[0] + spacing[1]) / 2

    # Centroid
    ys, xs = np.where(mask)
    centroid = (float(xs.mean()), float(ys.mean()))

    return AreaMeasurement(
        label_id=label_id,
        label_name=label_name,
        area_px=area_px,
        area_mm2=area_mm2,
        perimeter_mm=perimeter_mm,
        centroid=centroid,
    )


def compute_all_label_areas(
    seg_slice: np.ndarray,
    spacing: tuple[float, float] = (1.0, 1.0),
    label_names: dict[int, str] | None = None,
) -> list[AreaMeasurement]:
    """Compute area for all labels in a segmentation slice."""
    if label_names is None:
        label_names = {}
    results = []
    unique = np.unique(seg_slice)
    for label_id in unique:
        if label_id == 0:
            continue
        name = label_names.get(int(label_id), f"Label {int(label_id)}")
        m = compute_slice_area(seg_slice, int(label_id), spacing, name)
        if m is not None:
            results.append(m)
    return results
