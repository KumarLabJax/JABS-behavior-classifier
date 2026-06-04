"""Tests for multi-class training data export and round-trip retraining."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from jabs.core.constants import FINAL_TRAIN_SEED, MULTICLASS_NONE_BEHAVIOR
from jabs.core.enums import ClassifierMode, ClassifierType
from jabs.project import export_training_data, export_training_data_multiclass
from jabs.project.read_training import load_multiclass_training_data
from jabs.project.track_labels import TrackLabels

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_multiclass_project(tmp_path: Path, behavior_names: list[str]) -> MagicMock:
    """Build a minimal Project-like mock for export tests."""
    n_rows = 6
    per_frame = pd.DataFrame({"feat_a": np.arange(n_rows, dtype=np.float32)})
    window = pd.DataFrame({"feat_b": np.arange(n_rows, dtype=np.float32) * 0.1})
    groups = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)

    # Three frames each: "None" labeled, Walk labeled, Run labeled
    labels_by_behavior: dict[str, np.ndarray] = {
        MULTICLASS_NONE_BEHAVIOR: np.array(
            [
                TrackLabels.Label.BEHAVIOR,
                TrackLabels.Label.NONE,
                TrackLabels.Label.NONE,
                TrackLabels.Label.BEHAVIOR,
                TrackLabels.Label.NONE,
                TrackLabels.Label.NONE,
            ],
            dtype=np.int8,
        ),
    }
    for i, name in enumerate(behavior_names):
        arr = np.full(n_rows, TrackLabels.Label.NONE, dtype=np.int8)
        arr[i + 1] = TrackLabels.Label.BEHAVIOR
        arr[i + 4] = TrackLabels.Label.BEHAVIOR
        labels_by_behavior[name] = arr

    features = {
        "per_frame": per_frame,
        "window": window,
        "labels_by_behavior": labels_by_behavior,
        "groups": groups,
    }
    group_mapping = {
        0: {"video": "vid_a.mp4", "identity": 0},
        1: {"video": "vid_b.mp4", "identity": None},
    }

    project = MagicMock()
    project.dir = tmp_path
    project.get_multiclass_labeled_features.return_value = (features, group_mapping)
    project.get_project_defaults.return_value = {"window_size": 5, "balance_labels": False}
    project.settings_manager = SimpleNamespace(
        classifier_mode=ClassifierMode.MULTICLASS,
        behavior_names=behavior_names,
    )
    project.feature_manager = SimpleNamespace(min_pose_version=6)
    return project


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


def test_multiclass_export_attrs(tmp_path: Path) -> None:
    """Exported file has the expected top-level attributes."""
    import h5py

    behavior_names = ["Walk", "Run"]
    project = _make_multiclass_project(tmp_path, behavior_names)
    out = export_training_data_multiclass(
        project, pose_version=6, classifier_type=ClassifierType.RANDOM_FOREST
    )

    with h5py.File(out, "r") as f:
        assert f.attrs["classifier_mode"] == "multiclass"
        assert f.attrs["classifier_type"] == ClassifierType.RANDOM_FOREST.value
        assert f.attrs["min_pose_version"] == 6
        assert f.attrs["training_seed"] == FINAL_TRAIN_SEED


def test_multiclass_export_class_names(tmp_path: Path) -> None:
    """class_names dataset has background at index 0 followed by behavior names."""
    import h5py

    behavior_names = ["Walk", "Run"]
    project = _make_multiclass_project(tmp_path, behavior_names)
    out = export_training_data_multiclass(
        project, pose_version=6, classifier_type=ClassifierType.RANDOM_FOREST
    )

    with h5py.File(out, "r") as f:
        raw = f["class_names"][:]
        class_names = [n.decode() if isinstance(n, bytes) else str(n) for n in raw]

    assert class_names[0] == MULTICLASS_NONE_BEHAVIOR
    assert class_names[1:] == behavior_names


def test_multiclass_export_label_datasets(tmp_path: Path) -> None:
    """labels/ group contains one dataset per class, indexed by class position."""
    import h5py

    behavior_names = ["Walk", "Run"]
    project = _make_multiclass_project(tmp_path, behavior_names)
    out = export_training_data_multiclass(
        project, pose_version=6, classifier_type=ClassifierType.RANDOM_FOREST
    )

    with h5py.File(out, "r") as f:
        n_classes = len(behavior_names) + 1
        assert set(f["labels"].keys()) == {str(i) for i in range(n_classes)}
        # Dataset shapes must match feature rows
        n_rows = f["features/per_frame/feat_a"].shape[0]
        for i in range(n_classes):
            assert f[f"labels/{i}"].shape == (n_rows,)


def test_multiclass_export_group_mapping(tmp_path: Path) -> None:
    """group_mapping is written correctly; VIDEO-strategy identity stored as -1."""
    import h5py

    behavior_names = ["Walk"]
    project = _make_multiclass_project(tmp_path, behavior_names)
    out = export_training_data_multiclass(
        project, pose_version=6, classifier_type=ClassifierType.RANDOM_FOREST
    )

    with h5py.File(out, "r") as f:
        assert f["group_mapping/0/identity"][0] == 0
        assert f["group_mapping/1/identity"][0] == -1


def test_multiclass_export_no_behavior_names_attr(tmp_path: Path) -> None:
    """Multiclass export does not contain the binary-only 'behavior' attr."""
    import h5py

    project = _make_multiclass_project(tmp_path, ["Walk"])
    out = export_training_data_multiclass(
        project, pose_version=6, classifier_type=ClassifierType.RANDOM_FOREST
    )

    with h5py.File(out, "r") as f:
        assert "behavior" not in f.attrs


# ---------------------------------------------------------------------------
# Reader tests
# ---------------------------------------------------------------------------


def test_load_multiclass_training_data_roundtrip(tmp_path: Path) -> None:
    """load_multiclass_training_data reconstructs class_names and labels_by_behavior."""
    behavior_names = ["Walk", "Run"]
    project = _make_multiclass_project(tmp_path, behavior_names)
    out = export_training_data_multiclass(
        project, pose_version=6, classifier_type=ClassifierType.RANDOM_FOREST
    )

    loaded, group_mapping = load_multiclass_training_data(out)

    assert loaded["class_names"] == [MULTICLASS_NONE_BEHAVIOR, "Walk", "Run"]
    assert loaded["behavior_names"] == ["Walk", "Run"]
    assert set(loaded["labels_by_behavior"].keys()) == {MULTICLASS_NONE_BEHAVIOR, "Walk", "Run"}
    assert loaded["classifier_type"] == ClassifierType.RANDOM_FOREST
    assert loaded["training_seed"] == FINAL_TRAIN_SEED
    assert isinstance(loaded["per_frame"], pd.DataFrame)
    assert isinstance(loaded["window"], pd.DataFrame)
    assert loaded["per_frame"].shape[0] == loaded["window"].shape[0]
    # Identity None sentinel decoded correctly
    assert group_mapping[1]["identity"] is None
    assert group_mapping[0]["identity"] == 0


def test_load_multiclass_rejects_binary_file(tmp_path: Path) -> None:
    """load_multiclass_training_data raises ValueError on a binary training file."""
    binary_project = MagicMock()
    binary_project.dir = tmp_path
    binary_project.get_labeled_features.return_value = (
        {
            "per_frame": pd.DataFrame({"feat_a": [1.0, 2.0]}),
            "window": pd.DataFrame({"feat_b": [0.1, 0.2]}),
            "labels": np.array([1, 0], dtype=np.int8),
            "groups": np.array([0, 1], dtype=np.int32),
        },
        {0: {"video": "v.mp4", "identity": 0}},
    )
    binary_project.settings_manager = SimpleNamespace(
        get_behavior=lambda _: {"window_size": 5, "balance_labels": False},
    )
    binary_project.feature_manager = SimpleNamespace(min_pose_version=6)

    binary_out = export_training_data(binary_project, "Walk", 6, ClassifierType.RANDOM_FOREST)

    with pytest.raises(ValueError, match="not a multi-class training file"):
        load_multiclass_training_data(binary_out)


# ---------------------------------------------------------------------------
# Round-trip retrain test
# ---------------------------------------------------------------------------


def test_multiclass_from_training_file_roundtrip(tmp_path: Path) -> None:
    """MultiClassClassifier.from_training_file trains successfully from an export."""
    from jabs.classifier.multi_class_classifier import MultiClassClassifier

    behavior_names = ["Walk", "Run"]
    project = _make_multiclass_project(tmp_path, behavior_names)
    out = export_training_data_multiclass(
        project, pose_version=6, classifier_type=ClassifierType.RANDOM_FOREST
    )

    clf = MultiClassClassifier.from_training_file(out)

    assert clf.behavior_names == behavior_names
    assert clf.get_class_names() == [MULTICLASS_NONE_BEHAVIOR, "Walk", "Run"]
    assert clf.classifier_type == ClassifierType.RANDOM_FOREST
    assert clf._classifier_file == out.name
    assert clf._classifier_source == "training_file"


# ---------------------------------------------------------------------------
# CLI branching tests
# ---------------------------------------------------------------------------


def test_cli_binary_requires_behavior(tmp_path: Path) -> None:
    """export-training without --behavior on a binary project raises an error."""
    from click.testing import CliRunner

    from jabs.scripts.cli.cli import cli

    runner = CliRunner()

    # Patch Project so no real directory is needed
    import jabs.scripts.cli.cli as cli_module

    fake_project = MagicMock()
    fake_project.settings_manager = SimpleNamespace(
        classifier_mode=ClassifierMode.BINARY,
    )

    original_project = cli_module.Project
    cli_module.Project = MagicMock(return_value=fake_project)
    try:
        result = runner.invoke(cli, ["export-training", str(tmp_path)])
    finally:
        cli_module.Project = original_project

    assert result.exit_code != 0
    assert "--behavior is required" in result.output


def test_cli_multiclass_does_not_require_behavior(tmp_path: Path) -> None:
    """export-training without --behavior on a multiclass project succeeds."""
    from click.testing import CliRunner

    import jabs.scripts.cli.cli as cli_module
    from jabs.scripts.cli.cli import cli

    behavior_names = ["Walk", "Run"]
    project = _make_multiclass_project(tmp_path, behavior_names)
    project.settings_manager.classifier_mode = ClassifierMode.MULTICLASS
    project.feature_manager.min_pose_version = 6

    original_project = cli_module.Project
    original_export = cli_module.export_training_data_multiclass
    cli_module.Project = MagicMock(return_value=project)
    out_path = tmp_path / "mc.h5"
    cli_module.export_training_data_multiclass = MagicMock(return_value=out_path)
    try:
        runner = CliRunner()
        result = runner.invoke(cli, ["export-training", str(tmp_path)])
    finally:
        cli_module.Project = original_project
        cli_module.export_training_data_multiclass = original_export

    assert result.exit_code == 0, result.output
    assert "Exported training data" in result.output
