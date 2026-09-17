"""Tests for reading binary training data exported by ``export_training_data``."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from jabs.core.enums import ClassifierType
from jabs.project import export_training_data
from jabs.project.read_training import load_training_data


def _make_binary_project(tmp_path: Path, group_mapping: dict[int, dict]) -> MagicMock:
    """Build a minimal Project-like mock for binary export tests."""
    n_rows = 2 * len(group_mapping)
    features = {
        "per_frame": pd.DataFrame({"feat_a": np.arange(n_rows, dtype=np.float32)}),
        "window": pd.DataFrame({"feat_b": np.arange(n_rows, dtype=np.float32) * 0.1}),
        "labels": np.tile(np.array([1, 0], dtype=np.int8), len(group_mapping)),
        "groups": np.repeat(np.array(sorted(group_mapping), dtype=np.int32), 2),
    }

    project = MagicMock()
    project.dir = tmp_path
    project.get_labeled_features.return_value = (features, group_mapping)
    project.settings_manager = SimpleNamespace(
        get_behavior=lambda _: {"window_size": 5, "balance_labels": False},
    )
    return project


def test_load_training_data_decodes_missing_identity(tmp_path: Path) -> None:
    """A group with no identity round-trips back to None, not the -1 sentinel."""
    project = _make_binary_project(
        tmp_path,
        {
            0: {"video": "vid_a.mp4", "identity": 0},
            1: {"video": "vid_b.mp4", "identity": None},
        },
    )
    out = export_training_data(project, "Walk", 6, ClassifierType.RANDOM_FOREST)

    _, group_mapping = load_training_data(out)

    assert group_mapping[0]["identity"] == 0
    assert group_mapping[1]["identity"] is None


def test_load_training_data_identity_is_builtin_int(tmp_path: Path) -> None:
    """An identity-specific group decodes to a plain int, as documented."""
    project = _make_binary_project(tmp_path, {0: {"video": "vid_a.mp4", "identity": 2}})
    out = export_training_data(project, "Walk", 6, ClassifierType.RANDOM_FOREST)

    _, group_mapping = load_training_data(out)

    assert isinstance(group_mapping[0]["identity"], int)
    assert group_mapping[0]["identity"] == 2


def test_load_training_data_decodes_video_name(tmp_path: Path) -> None:
    """Video names decode to str rather than the bytes h5py hands back."""
    project = _make_binary_project(tmp_path, {0: {"video": "vid_a.mp4", "identity": 0}})
    out = export_training_data(project, "Walk", 6, ClassifierType.RANDOM_FOREST)

    _, group_mapping = load_training_data(out)

    assert group_mapping[0]["video"] == "vid_a.mp4"


def test_load_training_data_uses_filename_pattern_label(tmp_path: Path) -> None:
    """A FILENAME_PATTERN group reports its regex-extracted label as the video."""
    project = _make_binary_project(
        tmp_path,
        {
            0: {
                "video": None,
                "identity": None,
                "label": "cage_1234",
                "videos": ["cage_1234_a.mp4", "cage_1234_b.mp4"],
            },
        },
    )
    out = export_training_data(project, "Walk", 6, ClassifierType.RANDOM_FOREST)

    _, group_mapping = load_training_data(out)

    assert group_mapping[0]["identity"] is None
    assert group_mapping[0]["video"] == "cage_1234"
