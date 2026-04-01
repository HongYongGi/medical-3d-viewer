"""Tests for MPR windowing functions."""
import numpy as np
import sys
sys.path.insert(0, "src")

from medical_viewer.mpr.windowing import apply_window, auto_window, WINDOW_PRESETS


def test_apply_window_basic():
    img = np.array([[0, 100, 200, 300, 400]], dtype=np.float32)
    result = apply_window(img, center=200, width=400)
    assert result.dtype == np.uint8
    assert result[0, 0] == 0       # 0 is at lower bound
    assert result[0, 2] == 127     # 200 is center -> ~127
    assert result[0, 4] == 255     # 400 is at upper bound


def test_apply_window_clipping():
    img = np.array([[-2000, 3000]], dtype=np.float32)
    result = apply_window(img, center=0, width=100)
    assert result[0, 0] == 0
    assert result[0, 1] == 255


def test_auto_window():
    img = np.random.normal(100, 50, (100, 100)).astype(np.float32)
    center, width = auto_window(img)
    assert 50 < center < 150
    assert width > 0


def test_presets_have_required_keys():
    for name, preset in WINDOW_PRESETS.items():
        assert "center" in preset, f"{name} missing center"
        assert "width" in preset, f"{name} missing width"
        assert "icon" in preset, f"{name} missing icon"
        assert preset["width"] > 0, f"{name} has non-positive width"


if __name__ == "__main__":
    test_apply_window_basic()
    test_apply_window_clipping()
    test_auto_window()
    test_presets_have_required_keys()
    print("All windowing tests passed!")
