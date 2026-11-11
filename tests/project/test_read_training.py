"""Tests for reading binary training data exported by ``export_training_data``."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import h5py
import numpy as np
import pandas as pd
import pytest

from jabs.core.enums import ClassifierType, ProjectDistanceUnit
from jabs.project import export_training_data
from jabs.project.read_training import load_training_data


def _make_binary_project(
    tmp_path: Path,
    group_mapping: dict[int, dict],
    behavior_settings: dict | None = None,
) -> MagicMock:
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
    settings = (
        {"window_size": 5, "balance_labels": False}
        if behavior_settings is None
        else behavior_settings
    )
    project.settings_manager = SimpleNamespace(get_behavior=lambda _: settings)
    return project


def test_load_training_data_decodes_group_mapping(tmp_path: Path) -> None:
    """Group identities and video names decode to plain Python values.

    A group with no identity round-trips back to None, not the -1 sentinel; an
    identity-specific group decodes to a builtin int, as documented; and video
    names decode to str rather than the bytes h5py hands back.
    """
    project = _make_binary_project(
        tmp_path,
        {
            0: {"video": "vid_a.mp4", "identity": 0},
            1: {"video": "vid_b.mp4", "identity": None},
            2: {"video": "vid_c.mp4", "identity": 2},
        },
    )
    out = export_training_data(project, "Walk", 6, ClassifierType.RANDOM_FOREST)

    _, group_mapping = load_training_data(out)

    assert group_mapping[0]["identity"] == 0
    assert group_mapping[1]["identity"] is None
    assert group_mapping[2]["identity"] == 2
    assert isinstance(group_mapping[2]["identity"], int)
    assert group_mapping[0]["video"] == "vid_a.mp4"
    assert isinstance(group_mapping[0]["video"], str)


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


@pytest.mark.parametrize(
    ("cm_units", "expected"),
    [
        (True, ProjectDistanceUnit.CM),
        (False, ProjectDistanceUnit.PIXEL),
        (ProjectDistanceUnit.CM, ProjectDistanceUnit.CM),
        (ProjectDistanceUnit.PIXEL, ProjectDistanceUnit.PIXEL),
    ],
    ids=["bool-cm", "bool-pixel", "enum-cm", "enum-pixel"],
)
def test_load_training_data_reads_distance_unit_from_settings(
    tmp_path: Path, cm_units: bool | ProjectDistanceUnit, expected: ProjectDistanceUnit
) -> None:
    """The reported unit follows the exported ``cm_units`` setting.

    ``export_training_data`` writes no top-level ``distance_unit`` attribute, so a
    reader that consults only the attribute reports PIXEL for every file it
    writes - including exports of a cm-unit project. ``cm_units`` is a bool, but
    files exported by older releases may hold a ``ProjectDistanceUnit`` from the
    project defaults, so both spellings are checked.
    """
    project = _make_binary_project(
        tmp_path,
        {0: {"video": "vid_a.mp4", "identity": 0}},
        behavior_settings={"window_size": 5, "cm_units": cm_units},
    )
    out = export_training_data(project, "Walk", 6, ClassifierType.RANDOM_FOREST)

    features, _ = load_training_data(out)

    assert features["distance_unit"] == expected


def test_load_training_data_prefers_legacy_distance_unit_attr(tmp_path: Path) -> None:
    """A file that carries the old attribute still reads back from it.

    Older JABS releases stored the unit as a top-level ``distance_unit``
    attribute. Such a file keeps reading back as it used to, even when the
    settings it also carries disagree.
    """
    project = _make_binary_project(
        tmp_path,
        {0: {"video": "vid_a.mp4", "identity": 0}},
        behavior_settings={"window_size": 5, "cm_units": False},
    )
    out = export_training_data(project, "Walk", 6, ClassifierType.RANDOM_FOREST)
    with h5py.File(out, "r+") as h5:
        h5.attrs["distance_unit"] = ProjectDistanceUnit.CM.name

    features, _ = load_training_data(out)

    assert features["distance_unit"] == ProjectDistanceUnit.CM
