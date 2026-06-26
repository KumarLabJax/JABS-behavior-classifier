"""Tests for convert_to_nwb helper functions."""

import datetime
import json
from unittest import mock

import h5py
import numpy as np
import pytest
from click.testing import CliRunner

from jabs.scripts.cli.convert_to_nwb import (
    _collect_hdf5_attributes,
    _h5_attr_to_jsonable,
    _parse_session_start_time,
    run_conversion,
)


def test_parse_utc_offset():
    """Test parsing a UTC offset datetime string."""
    dt = _parse_session_start_time("2024-03-15T10:30:00+00:00")
    assert dt == datetime.datetime(2024, 3, 15, 10, 30, 0, tzinfo=datetime.timezone.utc)


def test_parse_negative_offset():
    """Test parsing a negative offset datetime string."""
    dt = _parse_session_start_time("2024-03-15T10:30:00-05:00")
    expected_tz = datetime.timezone(datetime.timedelta(hours=-5))
    assert dt == datetime.datetime(2024, 3, 15, 10, 30, 0, tzinfo=expected_tz)


def test_parse_z_suffix():
    """Test parsing a datetime string with 'Z' suffix (UTC)."""
    dt = _parse_session_start_time("2024-03-15T10:30:00Z")
    assert dt.tzinfo == datetime.timezone.utc
    assert dt.year == 2024 and dt.month == 3 and dt.day == 15


def test_parse_naive_assumes_utc(caplog):
    """Test that naive datetime strings are assumed to be UTC and log a warning."""
    import logging

    with caplog.at_level(logging.WARNING):
        dt = _parse_session_start_time("2024-03-15T10:30:00")

    assert dt.tzinfo == datetime.timezone.utc
    assert "no timezone" in caplog.text.lower() or "utc" in caplog.text.lower()


def test_parse_invalid_raises():
    """Test that invalid datetime strings raise ValueError."""
    with pytest.raises(ValueError, match="ISO 8601"):
        _parse_session_start_time("not-a-date")


@pytest.mark.parametrize("value", [42, None, 3.14, True], ids=["int", "null", "float", "bool"])
def test_parse_non_string_raises(value):
    """Test that ValueError is raised if value is not a string."""
    with pytest.raises(ValueError, match="must be a string"):
        _parse_session_start_time(value)


# ---------------------------------------------------------------------------
# run_conversion write-mode wiring
# ---------------------------------------------------------------------------


def _patch_conversion_internals(monkeypatch):
    """Patch the pose-loading and save boundaries of run_conversion; return the save mock."""
    pose = mock.Mock(num_identities=2, num_frames=10, fps=30)
    monkeypatch.setattr("jabs.scripts.cli.convert_to_nwb.open_pose_file", lambda *a, **k: pose)
    monkeypatch.setattr(
        "jabs.scripts.cli.convert_to_nwb.pose_to_pose_data",
        lambda *a, **k: mock.sentinel.pose_data,
    )
    save_mock = mock.Mock()
    monkeypatch.setattr("jabs.scripts.cli.convert_to_nwb.save", save_mock)
    return save_mock


def test_run_conversion_multisubject_forwarded(monkeypatch, tmp_path):
    """run_conversion(multisubject=True) forwards multisubject=True to save()."""
    save_mock = _patch_conversion_internals(monkeypatch)

    run_conversion(tmp_path / "in_pose_est_v6.h5", tmp_path / "out.nwb", multisubject=True)

    save_mock.assert_called_once()
    assert save_mock.call_args.kwargs["multisubject"] is True


def test_run_conversion_defaults_to_per_identity(monkeypatch, tmp_path):
    """run_conversion defaults to multisubject=False (per-identity output)."""
    save_mock = _patch_conversion_internals(monkeypatch)

    run_conversion(tmp_path / "in_pose_est_v6.h5", tmp_path / "out.nwb")

    assert save_mock.call_args.kwargs["multisubject"] is False


# ---------------------------------------------------------------------------
# convert-to-nwb CLI wiring
# ---------------------------------------------------------------------------


def test_cli_multisubject_flag_forwarded(monkeypatch, tmp_path):
    """The --multisubject flag is forwarded to run_conversion."""
    from jabs.scripts.cli.cli import cli

    run_mock = mock.Mock()
    monkeypatch.setattr("jabs.scripts.cli.cli.run_conversion", run_mock)
    input_path = tmp_path / "session_pose_est_v6.h5"
    input_path.write_bytes(b"")  # must exist for click.Path(exists=True)
    output = tmp_path / "session.nwb"

    result = CliRunner().invoke(
        cli, ["convert-to-nwb", str(input_path), str(output), "--multisubject"]
    )

    assert result.exit_code == 0, result.output
    assert run_mock.call_args.kwargs["multisubject"] is True


def test_cli_defaults_to_per_identity(monkeypatch, tmp_path):
    """Without --multisubject the CLI requests per-identity output (multisubject=False)."""
    from jabs.scripts.cli.cli import cli

    run_mock = mock.Mock()
    monkeypatch.setattr("jabs.scripts.cli.cli.run_conversion", run_mock)
    input_path = tmp_path / "session_pose_est_v6.h5"
    input_path.write_bytes(b"")
    output = tmp_path / "session.nwb"

    result = CliRunner().invoke(cli, ["convert-to-nwb", str(input_path), str(output)])

    assert result.exit_code == 0, result.output
    assert run_mock.call_args.kwargs["multisubject"] is False


def test_cli_per_identity_flag_removed(tmp_path):
    """The old --per-identity flag no longer exists."""
    from jabs.scripts.cli.cli import cli

    input_path = tmp_path / "session_pose_est_v6.h5"
    input_path.write_bytes(b"")
    output = tmp_path / "session.nwb"

    result = CliRunner().invoke(
        cli, ["convert-to-nwb", str(input_path), str(output), "--per-identity"]
    )

    assert result.exit_code != 0
    assert "no such option" in result.output.lower()


# ---------------------------------------------------------------------------
# HDF5 attribute normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (b"hello", "hello"),
        (np.bytes_(b"world"), "world"),
        ("plain", "plain"),
        (np.int64(7), 7),
        (np.float64(1.5), 1.5),
        (np.bool_(True), True),
        (np.array(7), 7),
        (np.array(1.5), 1.5),
        (np.array(b"scalar"), "scalar"),
        (42, 42),
        (3.14, 3.14),
        (None, None),
    ],
    ids=[
        "bytes",
        "np_bytes",
        "str",
        "np_int",
        "np_float",
        "np_bool",
        "zerod_int_array",
        "zerod_float_array",
        "zerod_bytes_array",
        "py_int",
        "py_float",
        "none",
    ],
)
def test_h5_attr_to_jsonable_scalars(value, expected):
    """Scalar HDF5 attribute values normalize to plain JSON-friendly types."""
    assert _h5_attr_to_jsonable(value) == expected


def test_h5_attr_to_jsonable_numeric_array():
    """A numeric numpy array becomes a plain list of Python numbers."""
    result = _h5_attr_to_jsonable(np.array([6, 0, 0], dtype=np.uint16))
    assert result == [6, 0, 0]
    assert all(isinstance(x, int) for x in result)


def test_h5_attr_to_jsonable_byte_string_array():
    """An array of fixed-length byte strings is decoded to a list of str."""
    result = _h5_attr_to_jsonable(np.array([b"a", b"b"], dtype="S1"))
    assert result == ["a", "b"]


def test_h5_attr_to_jsonable_unsupported_falls_back_to_str(caplog):
    """An unrecognized type is preserved as its string representation with a warning."""
    import logging

    value = complex(1, 2)
    with caplog.at_level(logging.WARNING):
        result = _h5_attr_to_jsonable(value)

    assert result == str(value)
    assert "unsupported type" in caplog.text.lower()


def test_collect_hdf5_attributes(tmp_path):
    """All attributes across the file are collected, keyed by object path."""
    h5_path = tmp_path / "sample_pose_est_v6.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.attrs["experimenter"] = "Jane Doe"
        h5.attrs["custom_flag"] = np.int64(1)
        poseest = h5.create_group("poseest")
        poseest.attrs["version"] = np.array([6, 0], dtype=np.uint16)
        poseest.attrs["cm_per_pixel"] = np.float64(0.07)
        points = poseest.create_dataset("points", data=np.zeros((2, 12, 2)))
        points.attrs["note"] = b"raw bytes note"
        # group with no attributes should be omitted
        h5.create_group("static_objects")

    collected = _collect_hdf5_attributes(h5_path)

    assert collected["/"] == {"experimenter": "Jane Doe", "custom_flag": 1}
    assert collected["poseest"] == {"version": [6, 0], "cm_per_pixel": pytest.approx(0.07)}
    assert collected["poseest/points"] == {"note": "raw bytes note"}
    assert "static_objects" not in collected


def test_collect_hdf5_attributes_is_json_serializable(tmp_path):
    """The collected attributes survive json.dumps without a custom encoder."""
    h5_path = tmp_path / "sample_pose_est_v6.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.attrs["str_attr"] = "value"
        h5.attrs["int_array"] = np.array([1, 2, 3], dtype=np.int32)
        h5.create_group("poseest").attrs["bytes_attr"] = b"bytes"

    collected = _collect_hdf5_attributes(h5_path)

    # Should not raise; round-trips back to the same structure.
    assert json.loads(json.dumps(collected)) == collected
