"""Fast PIL-based slice rendering for MPR viewer.

Replaces Plotly Heatmap with direct PIL Image rendering for ~150-400ms speedup.
Plotly is only used for measurement mode where interactive annotations are needed.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .windowing import apply_window


# Colormap LUTs (pre-computed 256-entry tables)
_COLORMAPS = {
    "gray": None,  # Use grayscale directly
    "hot": np.array([
        [min(255, int(i * 3)), max(0, min(255, int((i - 85) * 3))), max(0, min(255, int((i - 170) * 3)))]
        for i in range(256)
    ], dtype=np.uint8),
    "bone": np.array([
        [min(255, int(i * 0.75 + 64 * (i / 255))), min(255, int(i * 0.75 + 64 * (i / 255))),
         min(255, int(i * 0.75 + 128 * (i / 255)))]
        for i in range(256)
    ], dtype=np.uint8),
}


def render_slice_to_image(
    ct_slice: np.ndarray,
    seg_slice: np.ndarray | None,
    window_center: float,
    window_width: float,
    colormap: str = "gray",
    show_seg: bool = True,
    seg_opacity: float = 0.4,
    visible_labels: set | None = None,
    label_colors: dict | None = None,
) -> Image.Image:
    """Render a CT slice with optional segmentation overlay as a PIL Image.

    Returns an RGB PIL Image ready for st.image().
    """
    # Apply windowing -> uint8 grayscale
    windowed = apply_window(ct_slice, window_center, window_width)

    # Apply colormap
    if colormap == "gray" or colormap not in _COLORMAPS or _COLORMAPS[colormap] is None:
        rgb = np.stack([windowed, windowed, windowed], axis=-1)
    else:
        lut = _COLORMAPS[colormap]
        rgb = lut[windowed]

    # Apply segmentation overlay
    if seg_slice is not None and show_seg and label_colors:
        overlay = rgb.astype(np.float32)
        for label_id, (r, g, b, _a) in label_colors.items():
            if visible_labels is not None and label_id not in visible_labels:
                continue
            mask = seg_slice == label_id
            if not mask.any():
                continue
            overlay[mask] = (
                overlay[mask] * (1 - seg_opacity)
                + np.array([r, g, b], dtype=np.float32) * seg_opacity
            )
        rgb = np.clip(overlay, 0, 255).astype(np.uint8)

    return Image.fromarray(rgb)
