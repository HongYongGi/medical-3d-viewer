"""Tests for fast PIL-based slice rendering."""
import sys
sys.path.insert(0, "src")

import numpy as np
from PIL import Image


def test_render_grayscale():
    from medical_viewer.mpr.fast_render import render_slice_to_image
    ct = np.random.randn(64, 64).astype(np.float32) * 500
    img = render_slice_to_image(ct, None, window_center=0, window_width=1000)
    assert isinstance(img, Image.Image)
    assert img.size == (64, 64)
    assert img.mode == "RGB"


def test_render_with_segmentation():
    from medical_viewer.mpr.fast_render import render_slice_to_image
    ct = np.zeros((32, 32), dtype=np.float32)
    seg = np.zeros((32, 32), dtype=np.int32)
    seg[10:20, 10:20] = 1
    seg[5:10, 5:10] = 2

    colors = {
        1: (255, 0, 0, 180),
        2: (0, 255, 0, 180),
    }
    img = render_slice_to_image(
        ct, seg, window_center=0, window_width=100,
        show_seg=True, seg_opacity=0.5, label_colors=colors,
    )
    arr = np.array(img)
    # Label 1 region should have red tint
    assert arr[15, 15, 0] > arr[15, 15, 1]  # R > G in label 1 area
    # Label 2 region should have green tint
    assert arr[7, 7, 1] > arr[7, 7, 0]  # G > R in label 2 area


def test_render_seg_hidden():
    from medical_viewer.mpr.fast_render import render_slice_to_image
    ct = np.ones((16, 16), dtype=np.float32) * 100
    seg = np.ones((16, 16), dtype=np.int32)
    colors = {1: (255, 0, 0, 180)}

    img_with = render_slice_to_image(ct, seg, 100, 200, show_seg=True, label_colors=colors)
    img_without = render_slice_to_image(ct, seg, 100, 200, show_seg=False, label_colors=colors)

    arr_with = np.array(img_with)
    arr_without = np.array(img_without)
    # With seg hidden, should be pure grayscale
    assert np.array_equal(arr_without[:, :, 0], arr_without[:, :, 1])
    # With seg shown, red channel should differ from green
    assert not np.array_equal(arr_with[:, :, 0], arr_with[:, :, 1])


def test_render_visible_labels_filter():
    from medical_viewer.mpr.fast_render import render_slice_to_image
    ct = np.zeros((16, 16), dtype=np.float32)
    seg = np.ones((16, 16), dtype=np.int32)  # all label 1
    colors = {1: (255, 0, 0, 180)}

    # Label 1 visible
    img_visible = render_slice_to_image(
        ct, seg, 0, 100, show_seg=True, visible_labels={1}, label_colors=colors,
    )
    # Label 1 hidden
    img_hidden = render_slice_to_image(
        ct, seg, 0, 100, show_seg=True, visible_labels={2}, label_colors=colors,
    )
    arr_v = np.array(img_visible)
    arr_h = np.array(img_hidden)
    # Hidden should be grayscale (no overlay)
    assert np.array_equal(arr_h[:, :, 0], arr_h[:, :, 1])
    # Visible should have red
    assert arr_v[8, 8, 0] > arr_v[8, 8, 1]


def test_render_hot_colormap():
    from medical_viewer.mpr.fast_render import render_slice_to_image
    ct = np.full((16, 16), 200.0, dtype=np.float32)
    img = render_slice_to_image(ct, None, window_center=200, window_width=400, colormap="hot")
    arr = np.array(img)
    # Hot colormap: bright values should have high red
    assert arr[8, 8, 0] > 100


def test_render_no_seg_no_labels():
    from medical_viewer.mpr.fast_render import render_slice_to_image
    ct = np.random.randn(8, 8).astype(np.float32)
    img = render_slice_to_image(ct, None, 0, 100, label_colors=None)
    assert img.size == (8, 8)


def test_windowing_cached():
    from medical_viewer.mpr.windowing import apply_window_cached
    ct = np.array([[0, 100, 200, 400]], dtype=np.float32)
    result = apply_window_cached(ct, "CT-Abdomen")
    assert result.dtype == np.uint8
    assert result.shape == (1, 4)


def test_windowing_cached_unknown_preset():
    from medical_viewer.mpr.windowing import apply_window_cached
    ct = np.array([[0, 100]], dtype=np.float32)
    result = apply_window_cached(ct, "NonExistentPreset")
    assert result.dtype == np.uint8
