"""Tests for WeightScanner auto-detection."""
import json
import tempfile
from pathlib import Path

from medical_viewer.inference.weight_scanner import WeightScanner


def _create_mock_weight_dir(base: Path):
    """Create a mock nnUNet weight directory structure."""
    ds = base / "Dataset100_Test"
    trainer_dir = ds / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    fold_dir = trainer_dir / "fold_all"
    fold_dir.mkdir(parents=True)

    # Create checkpoint files
    (fold_dir / "checkpoint_final.pth").write_bytes(b"mock")
    (fold_dir / "checkpoint_best.pth").write_bytes(b"mock_best")

    # Create dataset.json
    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "organ1": 1, "organ2": 2},
        "numTraining": 50,
        "file_ending": ".nii.gz",
    }
    (trainer_dir / "dataset.json").write_text(json.dumps(dataset_json))

    return base


def test_scan_finds_models():
    with tempfile.TemporaryDirectory() as tmp:
        base = _create_mock_weight_dir(Path(tmp))
        scanner = WeightScanner(base)
        models = scanner.scan_all()
        assert len(models) == 1
        m = models[0]
        assert m.dataset_id == 100
        assert m.dataset_name == "Test"
        assert m.configuration == "3d_fullres"
        assert m.folds == ["all"]
        assert 1 in m.labels and m.labels[1] == "organ1"
        assert m.num_training == 50
        assert m.has_best_checkpoint
        assert m.has_final_checkpoint


def test_scan_empty_dir():
    with tempfile.TemporaryDirectory() as tmp:
        scanner = WeightScanner(Path(tmp))
        assert scanner.scan_all() == []


def test_scan_nonexistent_dir():
    scanner = WeightScanner(Path("/nonexistent/path"))
    assert scanner.scan_all() == []


def test_to_model_config():
    with tempfile.TemporaryDirectory() as tmp:
        base = _create_mock_weight_dir(Path(tmp))
        scanner = WeightScanner(base)
        models = scanner.scan_all()
        config = models[0].to_model_config()
        assert config.id == "d100_3d_fullres"
        assert config.source == "auto_scan"
        assert config.fold == "all"


if __name__ == "__main__":
    test_scan_finds_models()
    test_scan_empty_dir()
    test_scan_nonexistent_dir()
    test_to_model_config()
    print("All weight scanner tests passed!")
