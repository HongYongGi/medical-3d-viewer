"""Tests for pipeline label merging logic."""
import sys
sys.path.insert(0, "src")

import tempfile
import numpy as np
import nibabel as nib


def _make_seg(labels, shape=(10, 10, 10)):
    """Create a temp NIfTI with given label values in distinct regions."""
    data = np.zeros(shape, dtype=np.int32)
    for i, label in enumerate(labels):
        start = i * (shape[0] // len(labels))
        end = (i + 1) * (shape[0] // len(labels))
        data[start:end, :, :] = label
    img = nib.Nifti1Image(data, np.eye(4))
    f = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
    nib.save(img, f.name)
    return f.name


def test_merge_union_two_models():
    """Test that union merge produces continuous labels without gaps."""
    from medical_viewer.inference.pipeline import PipelineRunner
    from pathlib import Path

    # Create non-overlapping segs so all labels survive merge
    shape = (20, 10, 10)
    data1 = np.zeros(shape, dtype=np.int32)
    data1[0:5, :, :] = 1
    data1[5:10, :, :] = 2
    img1 = nib.Nifti1Image(data1, np.eye(4))
    f1 = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
    nib.save(img1, f1.name)

    data2 = np.zeros(shape, dtype=np.int32)
    data2[10:15, :, :] = 1
    data2[15:20, :, :] = 2
    img2 = nib.Nifti1Image(data2, np.eye(4))
    f2 = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
    nib.save(img2, f2.name)

    results = {"model_a": Path(f1.name), "model_b": Path(f2.name)}

    runner = PipelineRunner.__new__(PipelineRunner)
    runner.pipeline = type('P', (), {'merge_strategy': 'union'})()

    with tempfile.TemporaryDirectory() as tmpdir:
        merged_path = runner._merge_union(results, Path(tmpdir))
        merged = nib.load(str(merged_path)).get_fdata().astype(int)
        unique = sorted(set(np.unique(merged)) - {0})

        # Model A: 2 labels -> remapped to 1,2; offset becomes 2
        # Model B: 2 labels -> remapped to 3,4
        assert len(unique) == 4
        assert unique == [1, 2, 3, 4]


def test_merge_union_continuous_offsets():
    """Verify offsets are continuous regardless of original label values."""
    from medical_viewer.inference.pipeline import PipelineRunner
    from pathlib import Path

    # Model with label 10 (high value)
    shape = (10, 10, 10)
    data1 = np.zeros(shape, dtype=np.int32)
    data1[0:5, :, :] = 10
    img1 = nib.Nifti1Image(data1, np.eye(4))
    f1 = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
    nib.save(img1, f1.name)

    data2 = np.zeros(shape, dtype=np.int32)
    data2[5:10, :, :] = 1
    img2 = nib.Nifti1Image(data2, np.eye(4))
    f2 = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
    nib.save(img2, f2.name)

    results = {"a": Path(f1.name), "b": Path(f2.name)}

    runner = PipelineRunner.__new__(PipelineRunner)
    runner.pipeline = type('P', (), {'merge_strategy': 'union'})()

    with tempfile.TemporaryDirectory() as tmpdir:
        merged_path = runner._merge_union(results, Path(tmpdir))
        merged = nib.load(str(merged_path)).get_fdata().astype(int)
        unique = sorted(set(np.unique(merged)) - {0})
        # Model A: 1 label -> remapped to 1, offset becomes 1
        # Model B: 1 label -> remapped to 2
        assert unique == [1, 2]
