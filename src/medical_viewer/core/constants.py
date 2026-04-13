"""Shared constants for the medical viewer application."""
from __future__ import annotations

# Label colors: RGBA tuples used across MPR viewer, 3D viewer, and seg editor.
# Single source of truth — import from here instead of defining locally.
LABEL_COLORS_RGBA: dict[int, tuple[int, int, int, int]] = {
    1: (255, 50, 50, 180),
    2: (50, 255, 50, 180),
    3: (50, 50, 255, 180),
    4: (255, 255, 50, 180),
    5: (255, 50, 255, 180),
    6: (50, 255, 255, 180),
    7: (255, 140, 0, 180),
    8: (148, 50, 255, 180),
    9: (255, 192, 203, 180),
    10: (139, 69, 19, 180),
}

# RGB-only (for seg_editor and contexts without alpha)
LABEL_COLORS_RGB: dict[int, tuple[int, int, int]] = {
    k: (r, g, b) for k, (r, g, b, _) in LABEL_COLORS_RGBA.items()
}

# Hex colors (for Plotly 3D viewer)
LABEL_COLORS_HEX: list[str] = [
    f"#{r:02X}{g:02X}{b:02X}" for r, g, b, _ in LABEL_COLORS_RGBA.values()
]
