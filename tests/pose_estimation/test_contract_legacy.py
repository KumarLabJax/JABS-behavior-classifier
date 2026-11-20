"""PoseLike behavior of legacy pose files, which ``open_pose_file`` converts.

The recorded contract goldens are checked on ``Pose`` in
``test_contract_unified.py``.
"""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from jabs.core.enums import PoseCapability
from jabs.core.pose_protocol import source_format_from_path
from jabs.pose_estimation import open_pose_file

from .contract import (
    sample_path,
)


def test_v3_identity_to_track_from_cache_is_int32(tmp_path: Path) -> None:
    """A v3 identity_to_track read back from a cached conversion is int32 too."""
    cache = tmp_path / "cache"
    cache.mkdir()
    first = open_pose_file(sample_path(3), cache)
    second = open_pose_file(sample_path(3), cache)
    assert second.identity_to_track.dtype == np.int32
    np.testing.assert_array_equal(first.identity_to_track, second.identity_to_track)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("video_pose_est_v2.h5", "pose_est_v2"),
        ("a_b_pose_est_v8.h5", "pose_est_v8"),
        ("video_pose.h5", "jabs.pose-file"),
        ("video.h5", None),
    ],
)
def test_source_format_from_path(name: str, expected: str | None) -> None:
    """The filename rule shared by every PoseLike implementation."""
    assert source_format_from_path(Path(name)) == expected


def test_close_and_context_manager() -> None:
    """The context manager returns self, and the pose stays readable after close()."""
    pose = open_pose_file(sample_path(2))
    with pose as entered:
        assert entered is pose
    pose.close()
    assert pose.num_frames > 0


def test_segmentation_requires_capability() -> None:
    """A pose without segmentation raises NotImplementedError."""
    pose = open_pose_file(sample_path(4))
    with pytest.raises(NotImplementedError):
        pose.segmentation_window(0, 0, 10)
    with pytest.raises(NotImplementedError):
        pose.iter_segmentation(0)


def test_segmentation_window_bounds() -> None:
    """Windows must lie within the file and blocks must be positive."""
    pose = open_pose_file(sample_path(6))
    with pytest.raises(ValueError):
        pose.segmentation_window(0, 10, 5)
    with pytest.raises(ValueError):
        pose.segmentation_window(0, 0, pose.num_frames + 1)
    with pytest.raises(ValueError):
        pose.segmentation_window(pose.num_identities, 0, 1)
    with pytest.raises(ValueError):
        pose.iter_segmentation(0, block=0)
    empty = pose.segmentation_window(0, 5, 5)
    assert empty.contour_count.shape == (0,)


# --- synthetic v6 file with a hole at contour index 0 -------------------------

_FRAMES = 4
_POINTS = 6


def _write_hole_v6(path: Path, model_metadata: object | None) -> None:
    """Write a two-identity v6 file whose identity 0 has an empty contour slot 0.

    Identity 0, every frame: slot 0 is padding, slot 1 a 3-point contour flagged
    internal (0), slot 2 padding, slot 3 a full 6-point contour flagged external (1).
    Frame 1 has no contours at all for identity 0.
    """
    instances, contours = 2, 4
    points = np.zeros((_FRAMES, instances, 12, 2), dtype=np.uint16)
    points[...] = np.arange(12, dtype=np.uint16)[:, np.newaxis] + 10
    confidence = np.ones((_FRAMES, instances, 12), dtype=np.float32)
    seg = np.full((_FRAMES, instances, contours, _POINTS, 2), -1, dtype=np.int16)
    flags = np.zeros((_FRAMES, instances, contours), dtype=bool)
    seg[:, 0, 1, :3] = [[1, 2], [3, 4], [5, 6]]
    seg[:, 0, 3, :] = np.arange(_POINTS * 2).reshape(_POINTS, 2) + 100
    flags[:, 0, 3] = True
    seg[1, 0] = -1
    seg[:, 1, 0, :4] = [[7, 8], [9, 10], [11, 12], [13, 14]]
    flags[:, 1, 0] = True
    with h5py.File(path, "w") as h5:
        grp = h5.create_group("poseest")
        grp.attrs["version"] = np.array([6, 0], dtype=np.uint16)
        if model_metadata is not None:
            grp.attrs["model_metadata_json"] = model_metadata
        grp["points"] = points
        grp["confidence"] = confidence
        grp["id_mask"] = np.zeros((_FRAMES, instances), dtype=bool)
        grp["instance_embed_id"] = np.tile(np.array([1, 2], dtype=np.uint32), (_FRAMES, 1))
        grp["instance_id_center"] = np.zeros((instances, 16), dtype=np.float64)
        grp["instance_count"] = np.full(_FRAMES, instances, dtype=np.uint8)
        grp["longterm_seg_id"] = np.tile(np.array([1, 2], dtype=np.uint32), (_FRAMES, 1))
        grp["instance_seg_id"] = np.tile(np.array([1, 2], dtype=np.uint32), (_FRAMES, 1))
        grp["seg_data"] = seg
        grp["seg_external_flag"] = flags


@pytest.fixture
def hole_v6(tmp_path: Path) -> Path:
    """A synthetic v6 file with a hole contour at index 0."""
    path = tmp_path / "hole_pose_est_v6.h5"
    _write_hole_v6(path, json.dumps({"model": "test", "version": 2}))
    return path


def test_hole_at_contour_index_zero_packs_valid_contours_first(hole_v6: Path) -> None:
    """Valid contours move to the front, keeping their order, lengths and flags."""
    pose = open_pose_file(hole_v6)
    window = pose.segmentation_window(0, 0, _FRAMES)
    assert window.start == 0
    np.testing.assert_array_equal(window.contour_count, [2, 0, 2, 2])
    for i in (0, 2, 3):
        np.testing.assert_array_equal(window.contour_length[i, :2], [3, _POINTS])
        np.testing.assert_array_equal(window.contours[i, 0, :3], [[1, 2], [3, 4], [5, 6]])
        np.testing.assert_array_equal(
            window.contours[i, 1], np.arange(_POINTS * 2).reshape(_POINTS, 2) + 100
        )
        np.testing.assert_array_equal(window.external_flag[i, :2], [False, True])
    other = pose.segmentation_window(1, 1, 3)
    assert other.start == 1
    np.testing.assert_array_equal(other.contour_count, [1, 1])
    np.testing.assert_array_equal(other.contour_length[:, 0], [4, 4])
    np.testing.assert_array_equal(other.external_flag[:, 0], [True, True])


def test_hole_v6_iter_segmentation_blocks(hole_v6: Path) -> None:
    """iter_segmentation yields consecutive windows that tile the file."""
    pose = open_pose_file(hole_v6)
    windows = list(pose.iter_segmentation(0, block=3))
    assert [w.start for w in windows] == [0, 3]
    np.testing.assert_array_equal(np.concatenate([w.contour_count for w in windows]), [2, 0, 2, 2])


def test_model_metadata_parsed(hole_v6: Path) -> None:
    """model_metadata is the parsed model_metadata_json attribute."""
    pose = open_pose_file(hole_v6)
    assert pose.model_metadata == {"model": "test", "version": 2}
    assert pose.source_format == "pose_est_v6"
    assert pose.capabilities == frozenset({PoseCapability.SOCIAL, PoseCapability.SEGMENTATION})


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        ("", None),
        ("   ", None),
        (np.bytes_(b'{"a": 1}'), {"a": 1}),
        ('{"a": [1, 2]}', {"a": [1, 2]}),
        ("not json", None),
    ],
    ids=["absent", "empty", "blank", "bytes", "str", "invalid"],
)
def test_model_metadata_edge_cases(tmp_path: Path, raw: object, expected: object) -> None:
    """Absent, empty or invalid metadata is None; bytes and str are parsed."""
    path = tmp_path / "meta_pose_est_v6.h5"
    _write_hole_v6(path, raw)
    assert open_pose_file(path).model_metadata == expected
