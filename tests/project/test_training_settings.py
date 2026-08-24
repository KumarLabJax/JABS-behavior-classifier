"""Tests for project settings serialization in exported training files."""

import json
from pathlib import Path

import h5py
import pytest

from jabs.project.export_training import JSON_ENCODED_SETTING_ATTR, write_project_settings
from jabs.project.read_training import read_project_settings


def _round_trip(tmp_path: Path, settings: dict) -> dict:
    """Write settings to a training file and read them back."""
    path = tmp_path / "training.h5"
    with h5py.File(path, "w") as out_h5:
        write_project_settings(out_h5, settings, "settings")
    with h5py.File(path, "r") as in_h5:
        return read_project_settings(in_h5["settings"])


def test_scalar_settings_round_trip(tmp_path: Path) -> None:
    """Scalar settings survive the write/read cycle unchanged."""
    settings = {"window_size": 5, "social": True, "cm_units": False}
    assert _round_trip(tmp_path, settings) == settings


def test_nested_dict_settings_round_trip(tmp_path: Path) -> None:
    """A one-level-deep settings sub-group survives the write/read cycle."""
    settings = {"window_size": 5, "static_objects": {"lixit": True, "food_hopper": False}}
    assert _round_trip(tmp_path, settings) == settings


def test_list_setting_round_trips_as_list(tmp_path: Path) -> None:
    """A list setting (e.g. postprocessing stages) reads back as a list, not a JSON string."""
    stages = [
        {"name": "duration", "enabled": True, "params": {"min_frames": 3}},
        {"name": "stitching", "enabled": False, "params": {"max_gap": 5}},
    ]
    result = _round_trip(tmp_path, {"window_size": 5, "postprocessing": stages})

    assert result["postprocessing"] == stages
    assert result["window_size"] == 5


def test_empty_list_setting_round_trips(tmp_path: Path) -> None:
    """An empty list reads back as an empty list."""
    assert _round_trip(tmp_path, {"postprocessing": []})["postprocessing"] == []


def test_nested_list_setting_round_trips(tmp_path: Path) -> None:
    """A list nested inside a settings sub-group is decoded too."""
    stages = [{"name": "duration", "params": {"min_frames": 3}}]
    result = _round_trip(tmp_path, {"behavior": {"postprocessing": stages}})

    assert result["behavior"]["postprocessing"] == stages


def test_list_setting_dataset_is_tagged(tmp_path: Path) -> None:
    """The JSON-encoded dataset carries the attribute the reader looks for."""
    stages = [{"name": "duration"}]
    path = tmp_path / "training.h5"
    with h5py.File(path, "w") as out_h5:
        write_project_settings(out_h5, {"postprocessing": stages, "social": True}, "settings")

    with h5py.File(path, "r") as in_h5:
        postprocessing = in_h5["settings/postprocessing"]
        assert postprocessing.attrs[JSON_ENCODED_SETTING_ATTR]
        assert json.loads(postprocessing[...].item()) == stages
        assert JSON_ENCODED_SETTING_ATTR not in in_h5["settings/social"].attrs


def test_untagged_string_setting_is_not_decoded(tmp_path: Path) -> None:
    """Settings written without the tag are returned as stored (older training files)."""
    path = tmp_path / "training.h5"
    with h5py.File(path, "w") as out_h5:
        group = out_h5.require_group("settings")
        group.create_dataset("postprocessing", data=json.dumps([{"name": "duration"}]))

    with h5py.File(path, "r") as in_h5:
        settings = read_project_settings(in_h5["settings"])

    assert settings["postprocessing"] == b'[{"name": "duration"}]'


@pytest.mark.parametrize(
    "value",
    [["a", "b"], [1, 2, 3], [[1, 2], [3, 4]], [{"a": 1}]],
    ids=["strings", "ints", "nested_lists", "dicts"],
)
def test_list_element_types_round_trip(tmp_path: Path, value: list) -> None:
    """Lists of assorted JSON-compatible element types round-trip."""
    assert _round_trip(tmp_path, {"setting": value})["setting"] == value
