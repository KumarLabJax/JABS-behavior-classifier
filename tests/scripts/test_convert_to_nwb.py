"""Tests for convert_to_nwb helper functions."""

import datetime
import json
import logging
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import pytest
from click.testing import CliRunner

from jabs.core.abstract.pose_est import PoseEstimation
from jabs.core.types.pose import PoseData
from jabs.io.internal.pose import resolve_identity_subjects
from jabs.pose_estimation import open_pose_file
from jabs.scripts.cli.convert_to_nwb import (
    _collect_hdf5_attributes,
    _h5_attr_to_jsonable,
    _parse_session_start_time,
    pose_to_pose_data,
    run_conversion,
)
from jabs.scripts.cli.dandi_subject_metadata import validate_subjects


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
    with caplog.at_level(logging.WARNING):
        dt = _parse_session_start_time("2024-03-15T10:30:00")

    assert dt.tzinfo == datetime.timezone.utc
    # the warning has to say both what is wrong and what was assumed
    assert "no timezone" in caplog.text.lower()
    assert "utc" in caplog.text.lower()


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


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [({}, False), ({"multisubject": True}, True)],
    ids=["default_per_identity", "multisubject"],
)
def test_run_conversion_forwards_multisubject(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kwargs: dict[str, bool], expected: bool
) -> None:
    """run_conversion forwards multisubject to save(), defaulting to per-identity (False).

    Args:
        monkeypatch: Pytest fixture used to patch the pose-loading and save boundaries.
        tmp_path: Pytest temporary directory.
        kwargs: Extra keyword arguments passed to run_conversion.
        expected: The multisubject value save() is expected to receive.
    """
    save_mock = _patch_conversion_internals(monkeypatch)

    run_conversion(tmp_path / "in_pose_est_v6.h5", tmp_path / "out.nwb", **kwargs)

    save_mock.assert_called_once()
    assert save_mock.call_args.kwargs["multisubject"] is expected


def test_run_conversion_rejects_invalid_subjects_before_saving(monkeypatch, tmp_path):
    """Invalid subject metadata must abort before save() is called.

    Per-identity output writes one file per identity in a loop, so validating after
    the first write would leave a partial, unpublishable set behind.
    """
    invalid = _valid_pose_data()
    object.__setattr__(invalid, "subjects", None)
    save_mock = _patch_conversion_internals(monkeypatch, pose_data=invalid)

    with pytest.raises(ValueError, match="missing or malformed"):
        run_conversion(tmp_path / "in_pose_est_v6.h5", tmp_path / "out.nwb")

    save_mock.assert_not_called()


# ---------------------------------------------------------------------------
# convert-to-nwb CLI wiring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("flag", "expected"),
    [([], False), (["--multisubject"], True)],
    ids=["default_per_identity", "multisubject"],
)
def test_cli_multisubject_flag_forwarded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, flag: list[str], expected: bool
) -> None:
    """--multisubject reaches run_conversion; without it the CLI requests per-identity output.

    Args:
        monkeypatch: Pytest fixture used to patch run_conversion.
        tmp_path: Pytest temporary directory.
        flag: Extra command-line arguments passed to convert-to-nwb.
        expected: The multisubject value run_conversion is expected to receive.
    """
    from jabs.scripts.cli.cli import cli

    run_mock = mock.Mock()
    monkeypatch.setattr("jabs.scripts.cli.cli.run_conversion", run_mock)
    input_path = tmp_path / "session_pose_est_v6.h5"
    input_path.write_bytes(b"")  # must exist for click.Path(exists=True)
    output = tmp_path / "session.nwb"

    result = CliRunner().invoke(cli, ["convert-to-nwb", str(input_path), str(output), *flag])

    assert result.exit_code == 0, result.output
    assert run_mock.call_args.kwargs["multisubject"] is expected


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


def test_contours_are_a_view_of_the_pose_file_array():
    """The exported contours share memory with the pose file's array instead of copying it.

    seg_data can be gigabytes for a long video and the pose object already holds it, so
    stacking per-identity slices would double the peak memory of every export.
    """
    pose = open_pose_file(SAMPLE_POSE_V6)
    seg = pose_to_pose_data(pose).segmentation_data

    for i in pose.identities:
        assert np.shares_memory(seg.contours[i], pose.get_segmentation_data(i))


def test_contours_fall_back_to_stacking_without_the_bulk_accessor():
    """A pose object lacking get_segmentation_data_by_identity still exports, by stacking."""
    pose = open_pose_file(SAMPLE_POSE_V6)
    expected = pose_to_pose_data(pose).segmentation_data

    class _WithoutBulkAccessor:
        def __getattr__(self, name):
            if name == "get_segmentation_data_by_identity":
                raise AttributeError(name)
            return getattr(pose, name)

    seg = pose_to_pose_data(_WithoutBulkAccessor()).segmentation_data

    np.testing.assert_array_equal(seg.contours, expected.contours)
    np.testing.assert_array_equal(seg.vertex_counts, expected.vertex_counts)
    assert not np.shares_memory(seg.contours[0], pose.get_segmentation_data(0))


def test_segmentation_is_skipped_without_external_flags(monkeypatch, caplog):
    """Contours without seg_external_flag are not exported, rather than guessing hole vs outer.

    ContourSeries requires is_external and cannot say "unknown", so filling it in would
    write a boundary type the pose file never asserted.
    """
    pose = open_pose_file(SAMPLE_POSE_V6)
    monkeypatch.setattr(pose, "get_segmentation_flags", lambda identity: None)

    with caplog.at_level(logging.WARNING):
        data = pose_to_pose_data(pose)

    assert data.segmentation_data is None
    assert "no seg_external_flag" in caplog.text
    # the rest of the pose still converts
    assert data.points.shape[0] == len(list(pose.identities))


# --- identity renaming via the subjects file ------------------------------------------


def _renaming_pose(num_identities=1, external_ids=None):
    """A minimal fake pose whose identities are unnamed unless external_ids is given."""
    pose = mock.Mock(
        num_frames=4,
        fps=30,
        identities=list(range(num_identities)),
        cm_per_pixel=None,
        has_segmentation=False,
        static_objects={},
        external_identities=external_ids,
        pose_file="/tmp/sample_pose_est_v6.h5",
        hash=None,
    )
    pose.get_identity_poses.return_value = (
        np.zeros((4, 12, 2)),
        np.ones((4, 12), dtype=bool),
    )
    pose.identity_mask.return_value = np.ones(4, dtype=bool)
    pose.get_connected_segments.return_value = []
    pose.get_bounding_boxes.return_value = None
    return pose


def test_name_renames_the_identity(monkeypatch):
    """A 'name' in the subject entry becomes the identity name, not just the subject_id.

    The identity name is what names the NWB container and the per-identity output file,
    which subject_id cannot reach.
    """
    monkeypatch.setattr(
        "jabs.scripts.cli.convert_to_nwb._collect_hdf5_attributes", lambda path: {}
    )
    subjects = {"subject_1": {"name": "NV1-B2A", "species": "Mus musculus", "sex": "M"}}

    data = pose_to_pose_data(_renaming_pose(), subjects=subjects)

    assert data.external_ids == ["NV1-B2A"]
    # re-keyed, or the writer would look up "NV1-B2A" and find nothing
    assert set(data.subjects) == {"NV1-B2A"}
    # 'name' is not a Subject field and must not reach the output metadata
    assert "name" not in data.subjects["NV1-B2A"]
    assert data.subjects["NV1-B2A"]["species"] == "Mus musculus"


def test_without_name_nothing_changes(monkeypatch):
    """Subjects with no 'name' leave external_ids alone, so identities stay subject_N."""
    monkeypatch.setattr(
        "jabs.scripts.cli.convert_to_nwb._collect_hdf5_attributes", lambda path: {}
    )
    subjects = {"subject_1": {"species": "Mus musculus", "sex": "M"}}

    data = pose_to_pose_data(_renaming_pose(), subjects=subjects)

    assert data.external_ids is None
    assert set(data.subjects) == {"subject_1"}


def test_partial_rename_keeps_the_other_identities(monkeypatch):
    """Renaming one identity leaves the rest with the names they already had."""
    monkeypatch.setattr(
        "jabs.scripts.cli.convert_to_nwb._collect_hdf5_attributes", lambda path: {}
    )
    subjects = {"subject_2": {"name": "mouse_b", "species": "Mus musculus", "sex": "F"}}

    data = pose_to_pose_data(_renaming_pose(num_identities=3), subjects=subjects)

    assert data.external_ids == ["subject_1", "mouse_b", "subject_3"]


def test_name_is_sanitized_with_a_warning(monkeypatch, caplog):
    """A name that is not a legal container name is sanitized, and the user is told."""
    monkeypatch.setattr(
        "jabs.scripts.cli.convert_to_nwb._collect_hdf5_attributes", lambda path: {}
    )
    subjects = {"subject_1": {"name": "NV1/B2A", "species": "Mus musculus", "sex": "M"}}

    with caplog.at_level(logging.WARNING):
        data = pose_to_pose_data(_renaming_pose(), subjects=subjects)

    assert data.external_ids == ["NV1_B2A"]
    assert "NV1_B2A" in caplog.text


def test_duplicate_names_raise(monkeypatch):
    """Two identities cannot share a name: it would collide in the container and filename."""
    monkeypatch.setattr(
        "jabs.scripts.cli.convert_to_nwb._collect_hdf5_attributes", lambda path: {}
    )
    subjects = {
        "subject_1": {"name": "same", "species": "Mus musculus", "sex": "M"},
        "subject_2": {"name": "same", "species": "Mus musculus", "sex": "F"},
    }

    with pytest.raises(ValueError, match="Identity names must be unique"):
        pose_to_pose_data(_renaming_pose(num_identities=2), subjects=subjects)


def test_rename_preserves_raw_external_ids_of_other_identities(monkeypatch):
    """Renaming one identity must not re-key the others to their sanitized names.

    The writer looks subjects up by the raw external ID first, so writing the sanitized
    form back to external_ids would orphan metadata keyed by an ID that needed
    sanitizing - and only when some *other* identity happens to be renamed.
    """
    monkeypatch.setattr(
        "jabs.scripts.cli.convert_to_nwb._collect_hdf5_attributes", lambda path: {}
    )
    subjects = {
        "mouse/a": {"species": "Mus musculus", "sex": "M"},
        "mouse/b": {"name": "renamed_b", "species": "Mus musculus", "sex": "F"},
    }

    data = pose_to_pose_data(
        _renaming_pose(num_identities=2, external_ids=["mouse/a", "mouse/b"]),
        subjects=subjects,
    )

    # the untouched identity keeps its raw ID, the renamed one takes the new name
    assert data.external_ids == ["mouse/a", "renamed_b"]
    resolved = resolve_identity_subjects(data)
    assert [e.matched_key for e in resolved] == ["mouse/a", "renamed_b"]
    assert all(e.metadata.get("species") == "Mus musculus" for e in resolved)


def test_rename_colliding_after_sanitization_raises(monkeypatch):
    """Two ids that differ only in sanitized-away characters collide as container names."""
    monkeypatch.setattr(
        "jabs.scripts.cli.convert_to_nwb._collect_hdf5_attributes", lambda path: {}
    )
    subjects = {
        "mouse/a": {"species": "Mus musculus", "sex": "M"},
        "z": {"name": "mouse_a", "species": "Mus musculus", "sex": "F"},
    }

    with pytest.raises(ValueError, match="Identity names must be unique"):
        pose_to_pose_data(
            _renaming_pose(num_identities=2, external_ids=["mouse/a", "z"]), subjects=subjects
        )


def test_non_dict_subject_entry_is_left_for_validation(monkeypatch):
    """A non-dict entry must reach subject_metadata_problems, not crash before it.

    The CLI only checks that the top level of the subjects JSON is an object, so the
    per-identity check lives in validation - which never runs if the rename raises an
    AttributeError on the way past.
    """
    monkeypatch.setattr(
        "jabs.scripts.cli.convert_to_nwb._collect_hdf5_attributes", lambda path: {}
    )
    subjects = {
        "subject_1": "M123",  # a string where an object belongs
        "subject_2": {"name": "mouse_b", "species": "Mus musculus", "sex": "F"},
    }

    data = pose_to_pose_data(_renaming_pose(num_identities=2), subjects=subjects)

    with pytest.raises(ValueError, match="metadata must be a JSON object, got str"):
        validate_subjects(data)


def test_two_identities_can_swap_names(monkeypatch):
    """Swapping two names is legitimate; the collision check must not reject it.

    Popping as we insert would test the second name against a key that is itself about
    to move, making the outcome depend on the order the entries happen to be processed.
    """
    monkeypatch.setattr(
        "jabs.scripts.cli.convert_to_nwb._collect_hdf5_attributes", lambda path: {}
    )
    subjects = {
        "a": {"name": "b", "species": "Mus musculus", "sex": "M"},
        "b": {"name": "a", "species": "Mus musculus", "sex": "F"},
    }

    data = pose_to_pose_data(
        _renaming_pose(num_identities=2, external_ids=["a", "b"]), subjects=subjects
    )

    assert data.external_ids == ["b", "a"]
    assert data.subjects["b"]["sex"] == "M"
    assert data.subjects["a"]["sex"] == "F"


@pytest.mark.parametrize("value", [["NV1-B2A"], 0, False, 3.5], ids=str)
def test_non_string_name_is_rejected(monkeypatch, value):
    """str() would coerce these into plausible container names and write real files."""
    monkeypatch.setattr(
        "jabs.scripts.cli.convert_to_nwb._collect_hdf5_attributes", lambda path: {}
    )
    subjects = {"subject_1": {"name": value, "species": "Mus musculus", "sex": "M"}}

    with pytest.raises(ValueError, match="must be a string"):
        pose_to_pose_data(_renaming_pose(), subjects=subjects)


def test_blank_name_is_stripped_alongside_a_real_rename(monkeypatch):
    """A blank name renames nothing, but must not survive into the output metadata."""
    monkeypatch.setattr(
        "jabs.scripts.cli.convert_to_nwb._collect_hdf5_attributes", lambda path: {}
    )
    subjects = {
        "subject_1": {"name": "", "species": "Mus musculus", "sex": "M"},
        "subject_2": {"name": "mouse_b", "species": "Mus musculus", "sex": "F"},
    }

    data = pose_to_pose_data(_renaming_pose(num_identities=2), subjects=subjects)

    assert "name" not in data.subjects["subject_1"]
    assert data.external_ids == ["subject_1", "mouse_b"]


def test_blank_name_alone_is_stripped_and_renames_nothing(monkeypatch):
    """The only 'name' being blank must still strip it, without inventing external_ids."""
    monkeypatch.setattr(
        "jabs.scripts.cli.convert_to_nwb._collect_hdf5_attributes", lambda path: {}
    )
    subjects = {"subject_1": {"name": "", "species": "Mus musculus", "sex": "M"}}

    data = pose_to_pose_data(_renaming_pose(), subjects=subjects)

    assert "name" not in data.subjects["subject_1"]
    assert data.external_ids is None


def test_rename_keeps_keys_that_match_no_identity(monkeypatch):
    """A typo'd key must survive the rename so validate_subjects can still report it.

    Rebuilding subjects from the resolved metadata would drop it, and the user would see
    only an unexplained "species is missing" for the identity it was meant for.
    """
    monkeypatch.setattr(
        "jabs.scripts.cli.convert_to_nwb._collect_hdf5_attributes", lambda path: {}
    )
    subjects = {
        "subject_1": {"name": "mouse_a", "species": "Mus musculus", "sex": "M"},
        "subejct_2": {"species": "Mus musculus", "sex": "F"},  # typo, matches nothing
    }

    data = pose_to_pose_data(_renaming_pose(num_identities=2), subjects=subjects)

    assert set(data.subjects) == {"mouse_a", "subejct_2"}


def test_name_equal_to_the_existing_name_is_stripped(monkeypatch):
    """A no-op rename still has to have 'name' removed; it is not a Subject field."""
    monkeypatch.setattr(
        "jabs.scripts.cli.convert_to_nwb._collect_hdf5_attributes", lambda path: {}
    )
    subjects = {"subject_1": {"name": "subject_1", "species": "Mus musculus", "sex": "M"}}

    data = pose_to_pose_data(_renaming_pose(), subjects=subjects)

    assert data.external_ids == ["subject_1"]
    assert "name" not in data.subjects["subject_1"]


def test_rename_onto_an_unrelated_key_raises(monkeypatch):
    """Renaming onto a key that belongs to something else would discard one of them."""
    monkeypatch.setattr(
        "jabs.scripts.cli.convert_to_nwb._collect_hdf5_attributes", lambda path: {}
    )
    subjects = {
        "subject_1": {"name": "mouse_b", "species": "Mus musculus", "sex": "M"},
        "mouse_b": {"species": "Mus musculus", "sex": "F"},
    }

    with pytest.raises(ValueError, match="collides with the --subjects key"):
        pose_to_pose_data(_renaming_pose(num_identities=2), subjects=subjects)


def test_name_overrides_a_pose_file_external_id(monkeypatch):
    """A pose file's own external ID can be overridden too, keyed by that ID."""
    monkeypatch.setattr(
        "jabs.scripts.cli.convert_to_nwb._collect_hdf5_attributes", lambda path: {}
    )
    subjects = {"orig_a": {"name": "renamed_a", "species": "Mus musculus", "sex": "M"}}

    data = pose_to_pose_data(_renaming_pose(external_ids=["orig_a"]), subjects=subjects)

    assert data.external_ids == ["renamed_a"]
    assert set(data.subjects) == {"renamed_a"}
