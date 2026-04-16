from __future__ import annotations
import numpy as np

# 3D Slicer compatible window presets
WINDOW_PRESETS = {
    "CT-Abdomen": {"center": 40, "width": 400, "icon": "🫁"},
    "CT-Lung": {"center": -600, "width": 1500, "icon": "🫁"},
    "CT-Bone": {"center": 400, "width": 1800, "icon": "🦴"},
    "CT-Brain": {"center": 40, "width": 80, "icon": "🧠"},
    "CT-Aorta": {"center": 200, "width": 800, "icon": "❤️"},
    "CT-Liver": {"center": 60, "width": 150, "icon": "🫀"},
    "CT-Mediastinum": {"center": 50, "width": 350, "icon": "🫁"},
    "CT-Chest": {"center": -40, "width": 400, "icon": "🫁"},
    "CT-Cardiac": {"center": 120, "width": 240, "icon": "❤️"},
    "CT-Angio": {"center": 300, "width": 600, "icon": "🩸"},
    "CT-Soft-Tissue": {"center": 50, "width": 400, "icon": "🧬"},
    "CT-MIP": {"center": 400, "width": 1500, "icon": "📊"},
}

# Colormap options (3D Slicer compatible)
COLORMAPS = {
    "Gray": "gray",
    "Hot": "hot",
    "Jet": "jet",
    "Bone": "bone",
    "Ocean": "ocean",
    "Viridis": "viridis",
    "Inferno": "inferno",
}


def apply_window(image: np.ndarray, center: float = 40, width: float = 400) -> np.ndarray:
    """Apply CT windowing. Returns uint8 image [0, 255]."""
    lower = center - width / 2
    upper = center + width / 2
    windowed = np.clip(image, lower, upper)
    return ((windowed - lower) / (upper - lower) * 255).astype(np.uint8)


def get_preset(name: str) -> dict:
    return WINDOW_PRESETS.get(name, WINDOW_PRESETS["CT-Abdomen"])


def apply_window_cached(image: np.ndarray, preset_name: str) -> np.ndarray:
    """Apply a named window preset with caching support.

    For use with known presets where (center, width) are fixed.
    The caller can cache results keyed by (slice_index, preset_name).
    """
    preset = WINDOW_PRESETS.get(preset_name)
    if preset is None:
        return apply_window(image, 40, 400)
    return apply_window(image, preset["center"], preset["width"])


def auto_window(image: np.ndarray, percentile_low: float = 1, percentile_high: float = 99) -> tuple[float, float]:
    """Auto-compute window center/width from image percentiles."""
    low = np.percentile(image, percentile_low)
    high = np.percentile(image, percentile_high)
    center = (low + high) / 2
    width = high - low
    return float(center), float(width)
