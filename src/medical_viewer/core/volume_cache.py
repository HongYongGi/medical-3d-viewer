"""Shared volume caching utilities.

Provides a single cached entry point for loading NIfTI volumes,
avoiding redundant I/O across app.py, viewer_3d.py, seg_editor.py, etc.
"""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import streamlit as st

from ..mpr.slicer import MPRSlicer


@st.cache_resource(show_spinner=False)
def get_slicer(nifti_path: str) -> MPRSlicer:
    """Return a cached MPRSlicer instance for the given NIfTI path."""
    return MPRSlicer(nifti_path)


def get_volume(nifti_path: str | Path) -> np.ndarray:
    """Return the cached volume data for a NIfTI file via MPRSlicer."""
    return get_slicer(str(nifti_path)).volume


def get_header_info(nifti_path: str | Path) -> dict:
    """Return header metadata without reloading the full volume."""
    slicer = get_slicer(str(nifti_path))
    return {
        "shape": slicer.shape,
        "spacing": tuple(slicer.voxel_spacing),
        "affine": slicer.affine,
    }
