"""Tests for convert_to_nwb helper functions."""

import datetime
import json
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import pytest
from click.testing import CliRunner

from jabs.core.abstract.pose_est import PoseEstimation
from jabs.core.types.pose import PoseData
from jabs.pose_estimation import open_pose_file
from jabs.scripts.cli.convert_to_nwb import (
    _collect_hdf5_attributes,
    _h5_attr_to_jsonable,
    _parse_session_start_time,
    pose_to_pose_data,
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


def _valid_pose_data(num_identities=2, num_frames=10):
    """Build a PoseData whose subject metadata satisfies the DANDI pre-flight check.

    run_conversion validates subject metadata before writing, so the write-mode
    wiring tests need real data rather than a sentinel.
    """
    body_parts = [kpt.name for kpt in PoseEstimation.KeypointIndex]
    num_keypoints = len(body_parts)
    subjects = {
        f"subject_{i + 1}": {
            "subject_id": f"M{i + 1}",
            "species": "Mus musculus",
            "sex": "M",
            "age": "P70D",
        }
        for i in range(num_identities)
    }
    return PoseData(
        points=np.zeros((num_identities, num_frames, num_keypoints, 2)),
        point_mask=np.ones((num_identities, num_frames, num_keypoints), dtype=bool),
        identity_mask=np.ones((num_identities, num_frames), dtype=bool),
        body_parts=body_parts,
        edges=[],
        fps=30,
        subjects=subjects,
    )


def _patch_conversion_internals(monkeypatch, pose_data=None):
    """Patch the pose-loading and save boundaries of run_conversion; return the save mock."""
    pose = mock.Mock(num_identities=2, num_frames=10, fps=30)
    monkeypatch.setattr("jabs.scripts.cli.convert_to_nwb.open_pose_file", lambda *a, **k: pose)
    monkeypatch.setattr(
        "jabs.scripts.cli.convert_to_nwb.pose_to_pose_data",
        lambda *a, **k: pose_data if pose_data is not None else _valid_pose_data(),
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


def test_run_conversion_rejects_invalid_subjects_before_saving(monkeypatch, tmp_path):
    """Invalid subject metadata must abort before save(), leaving nothing on disk.

    Per-identity output writes one file per identity in a loop, so validating after
    the first write would leave a partial, unpublishable set behind.
    """
    invalid = _valid_pose_data()
    object.__setattr__(invalid, "subjects", None)
    save_mock = _patch_conversion_internals(monkeypatch, pose_data=invalid)

    with pytest.raises(ValueError, match="missing or malformed"):
        run_conversion(tmp_path / "in_pose_est_v6.h5", tmp_path / "out.nwb")

    save_mock.assert_not_called()
    assert list(tmp_path.glob("*.nwb")) == []


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


# ---------------------------------------------------------------------------
# segmentation contours
# ---------------------------------------------------------------------------

SAMPLE_POSE_V6 = Path(__file__).parent.parent / "data" / "sample_pose_est_v6.h5"


def test_pose_to_pose_data_builds_segmentation():
    """A v6 pose file's contours reach PoseData, identity-ordered and unflipped.

    seg_data is stored (x, y) in the pose file, unlike poseest/points which is (y, x),
    so the contours must come through with no axis flip.
    """
    pose = open_pose_file(SAMPLE_POSE_V6)
    data = pose_to_pose_data(pose)

    seg = data.segmentation_data
    assert seg is not None
    num_identities = len(list(pose.identities))
    assert seg.contours.shape[0] == num_identities
    assert seg.contours.shape[1] == pose.num_frames

    for i in pose.identities:
        np.testing.assert_array_equal(seg.contours[i], pose.get_segmentation_data(i))
        np.testing.assert_array_equal(seg.is_external[i], pose.get_segmentation_flags(i) > 0)


def test_pose_to_pose_data_keeps_pose_file_contour_width():
    """Contours keep seg_data's own integer width instead of being widened.

    The contour array is the largest thing JABS holds for a video - int16 over a
    full-length recording is already gigabytes - so widening it would double both the
    peak during the stack and what the writer holds for the length of the export.
    """
    pose = open_pose_file(SAMPLE_POSE_V6)
    seg = pose_to_pose_data(pose).segmentation_data

    source_dtype = pose.get_segmentation_data(0).dtype
    assert source_dtype == np.int16, "fixture must be narrower than int32 to test anything"
    assert seg.contours.dtype == source_dtype


def test_pose_to_pose_data_vertex_counts_match_padding():
    """vertex_counts counts exactly the vertices that are not the -1 padding sentinel."""
    pose = open_pose_file(SAMPLE_POSE_V6)
    seg = pose_to_pose_data(pose).segmentation_data

    expected = np.all(seg.contours != -1, axis=-1).sum(axis=-1)
    np.testing.assert_array_equal(seg.vertex_counts, expected)
    # the fixture must actually exercise padding, or this asserts nothing
    assert seg.vertex_counts.min() == 0
    assert 0 < seg.vertex_counts.max() <= seg.contours.shape[3]


def test_pose_to_pose_data_segmentation_disabled():
    """segmentation=False drops the contours even when the pose file has them."""
    pose = open_pose_file(SAMPLE_POSE_V6)
    assert pose.has_segmentation
    assert pose_to_pose_data(pose, segmentation=False).segmentation_data is None


def test_pose_to_pose_data_no_segmentation_in_older_pose():
    """A v5 pose file has no contours, so segmentation_data stays None."""
    pose = open_pose_file(SAMPLE_POSE_V6.with_name("sample_pose_est_v5.h5"))
    assert pose_to_pose_data(pose).segmentation_data is None


def test_run_conversion_forwards_segmentation(monkeypatch, tmp_path):
    """run_conversion passes its segmentation flag down to pose_to_pose_data."""
    pose = mock.Mock(num_identities=2, num_frames=10, fps=30)
    monkeypatch.setattr("jabs.scripts.cli.convert_to_nwb.open_pose_file", lambda *a, **k: pose)
    monkeypatch.setattr("jabs.scripts.cli.convert_to_nwb.save", mock.Mock())
    # run_conversion validates subject metadata before writing, so this needs real
    # PoseData rather than a sentinel.
    to_pose_data = mock.Mock(return_value=_valid_pose_data())
    monkeypatch.setattr("jabs.scripts.cli.convert_to_nwb.pose_to_pose_data", to_pose_data)

    run_conversion(tmp_path / "in_pose_est_v6.h5", tmp_path / "out.nwb", segmentation=False)

    assert to_pose_data.call_args.kwargs["segmentation"] is False


@pytest.mark.parametrize(
    ("flag", "expected"),
    [([], True), (["--segmentation"], True), (["--no-segmentation"], False)],
    ids=["default", "explicit_on", "off"],
)
def test_cli_segmentation_flag_forwarded(monkeypatch, tmp_path, flag, expected):
    """--segmentation/--no-segmentation reaches run_conversion, defaulting to on."""
    from jabs.scripts.cli.cli import cli

    run_mock = mock.Mock()
    monkeypatch.setattr("jabs.scripts.cli.cli.run_conversion", run_mock)
    input_path = tmp_path / "session_pose_est_v6.h5"
    input_path.write_bytes(b"")
    output = tmp_path / "session.nwb"

    result = CliRunner().invoke(cli, ["convert-to-nwb", str(input_path), str(output), *flag])

    assert result.exit_code == 0, result.output
    assert run_mock.call_args.kwargs["segmentation"] is expected


def test_unused_contour_slots_are_not_marked_external():
    """The -1 that identity-sorting leaves in unused flag slots must not read as True.

    PoseEstimationV6._segmentation_sort fills its output with -1 before scattering the
    per-identity flags into it, so an identity's unused contour slots come back as -1
    rather than False. A plain bool cast would turn those into True and claim an unused
    slot is an external boundary.
    """
    pose = open_pose_file(SAMPLE_POSE_V6)
    seg = pose_to_pose_data(pose).segmentation_data

    raw_flags = np.stack([pose.get_segmentation_flags(i) for i in pose.identities], axis=0)
    assert (raw_flags == -1).any(), "fixture no longer exercises the -1 sentinel"
    assert not seg.is_external[raw_flags == -1].any()
