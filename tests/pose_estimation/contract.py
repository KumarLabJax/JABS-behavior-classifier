"""Reusable behavioral contract checks for :class:`jabs.core.pose_protocol.PoseLike`.

:func:`check_pose_contract` compares a pose object against expectations recorded
in a golden ``.npz`` file. It touches only ``PoseLike`` members, so the same
goldens apply to any implementation: the legacy readers that recorded them and
the unified ``Pose`` reading a converted copy of the same sample file.

The goldens are kept small and portable:

* Per-frame data (points, point masks, segmentation) is recorded for a subset of
  frames only (:func:`golden_frames`, :func:`segmentation_windows`).
* Values are compared, never dtypes: legacy readers return float64 points and
  integer masks, ``Pose`` returns float32 points and bool masks.
* Points are compared only where the point mask is true, segmentation only on
  valid contours (the first ``contour_count`` contours of a frame and their
  first ``contour_length`` points), since padding is unspecified.
* Scalar and string expectations live in one JSON document stored as a 0-d
  string array, so loading never needs ``allow_pickle``.

Recording goldens:
    Goldens are written from the legacy readers and committed. They are never
    recorded by a test run; a test fails when its golden is missing. To
    (re)record them after an intended behavior change, run from the repo root::

        uv run python tests/pose_estimation/contract.py --record

    which calls :func:`record_sample_goldens`. Review the diff of the
    regenerated files before committing them.
"""

import argparse
import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from jabs.core.enums import PoseCapability
from jabs.core.pose_protocol import PoseLike
from jabs.core.pose_skeleton import KeypointIndex
from jabs.core.types import SegmentationWindow

logger = logging.getLogger(__name__)

GOLDEN_DIR = Path(__file__).parent / "goldens"
SAMPLE_DIR = Path(__file__).parent.parent / "data"
SAMPLE_VERSIONS = (2, 3, 4, 6)

# frames between sampled frames for per-frame expectations
_FRAME_STRIDE = 50
# frames between frames whose segmentation is sampled on its own; sparser than
# _FRAME_STRIDE because contours are by far the largest expectations
_SEGMENTATION_STRIDE = 300
# length of each fixed window of consecutive frames
_WINDOW_LENGTH = 10
# a fixed window crosses this frame so iter_segmentation's block boundary
# (default block of 300 frames) is covered
_BLOCK_BOUNDARY = 300

_META_KEY = "meta_json"
# iter_segmentation's default block
_DEFAULT_BLOCK = 300


def sample_path(version: int) -> Path:
    """Return the path of the sample legacy pose file for a format version.

    Args:
        version: Legacy pose format major version.

    Returns:
        Path of ``tests/data/sample_pose_est_v<version>.h5``.
    """
    return SAMPLE_DIR / f"sample_pose_est_v{version}.h5"


def golden_path(version: int) -> Path:
    """Return the path of the golden expectations for a sample file.

    Args:
        version: Legacy pose format major version of the sample file.

    Returns:
        Path of ``goldens/sample_pose_est_v<version>.npz``.
    """
    return GOLDEN_DIR / f"sample_pose_est_v{version}.npz"


def segmentation_windows(num_frames: int) -> list[tuple[int, int]]:
    """Return the fixed ``[start, stop)`` windows segmentation is recorded over.

    One window at the start, one crossing the 300-frame block boundary, one in
    the middle and one at the end.

    Args:
        num_frames: Number of frames in the pose file.

    Returns:
        Sorted, de-duplicated ``(start, stop)`` pairs within ``[0, num_frames)``.
    """
    half = _WINDOW_LENGTH // 2
    starts = [
        0,
        _BLOCK_BOUNDARY - half,
        num_frames // 2,
        num_frames - _WINDOW_LENGTH,
    ]
    windows = {
        (max(0, s), min(num_frames, max(0, s) + _WINDOW_LENGTH)) for s in starts if s < num_frames
    }
    return sorted(w for w in windows if w[0] < w[1])


def golden_frames(num_frames: int) -> npt.NDArray[np.int64]:
    """Return the frame indexes per-frame expectations are recorded at.

    Every 50th frame plus every frame of :func:`segmentation_windows`.

    Args:
        num_frames: Number of frames in the pose file.

    Returns:
        Sorted unique frame indexes.
    """
    frames = set(range(0, num_frames, _FRAME_STRIDE))
    for start, stop in segmentation_windows(num_frames):
        frames.update(range(start, stop))
    return np.array(sorted(frames), dtype=np.int64)


def _valid_segmentation(
    window: SegmentationWindow,
) -> tuple[
    npt.NDArray[np.uint32],
    npt.NDArray[np.uint32],
    npt.NDArray[np.int32],
    npt.NDArray[np.bool_] | None,
]:
    """Flatten the valid contours of a window, dropping all padding.

    Returns:
        ``(contour_count, lengths, points, flags)``: the per-frame contour
        counts; the length of every valid contour in (frame, contour) order;
        the valid points of those contours concatenated, shape ``(sum, 2)``;
        and the external flag of every valid contour, or None without flags.
    """
    count = np.asarray(window.contour_count, dtype=np.uint32)
    lengths: list[int] = []
    points: list[npt.NDArray[np.int32]] = []
    flags: list[bool] = []
    for i, n in enumerate(count):
        for c in range(int(n)):
            length = int(window.contour_length[i, c])
            lengths.append(length)
            points.append(np.asarray(window.contours[i, c, :length], dtype=np.int32))
            if window.external_flag is not None:
                flags.append(bool(window.external_flag[i, c]))
    flat_points = np.concatenate(points) if points else np.zeros((0, 2), dtype=np.int32)
    flat_flags = None if window.external_flag is None else np.array(flags, dtype=bool)
    return count, np.array(lengths, dtype=np.uint32), flat_points, flat_flags


def _segmentation_samples(num_frames: int) -> list[tuple[int, int]]:
    """Windows checked for segmentation: the fixed windows plus every 300th frame alone."""
    single = [(f, f + 1) for f in range(0, num_frames, _SEGMENTATION_STRIDE)]
    return segmentation_windows(num_frames) + single


def _segments_to_json(segments: Iterable[Iterable[KeypointIndex]]) -> list[list[int]]:
    """Convert connected segments to nested lists of keypoint indexes."""
    return [[int(k.value) for k in segment] for segment in segments]


def record_expectations(pose: PoseLike) -> dict[str, npt.NDArray[Any]]:
    """Record the contract expectations of a pose object.

    Args:
        pose: Pose object to record. Only ``PoseLike`` members are read.

    Returns:
        Mapping of ``.npz`` key to array, suitable for :func:`save_golden`.
    """
    frames = golden_frames(pose.num_frames)
    has_track = pose.identity_to_track is not None
    meta = {
        "num_frames": pose.num_frames,
        "num_identities": pose.num_identities,
        "identities": [int(i) for i in pose.identities],
        "cm_per_pixel": None if pose.cm_per_pixel is None else float(pose.cm_per_pixel),
        "capabilities": sorted(c.value for c in pose.capabilities),
        "source_format": pose.source_format,
        "model_metadata": pose.model_metadata,
        "has_identity_to_track": has_track,
        "static_objects": sorted(pose.static_objects),
        "external_identities": pose.external_identities,
        "num_lixit_keypoints": pose.num_lixit_keypoints,
        "hash": pose.hash,
        "fps": pose.fps,
        "dynamic_objects": sorted(pose.dynamic_objects),
        "connected_segments": _segments_to_json(pose.get_connected_segments()),
        "display_names": [pose.identity_index_to_display(i) for i in pose.identities],
    }
    expected: dict[str, npt.NDArray[Any]] = {
        _META_KEY: np.array(json.dumps(meta)),
        "frames": frames,
        "reduced_point_mask": np.asarray(pose.get_reduced_point_mask(), dtype=bool),
    }
    if has_track:
        expected["identity_to_track"] = np.asarray(pose.identity_to_track, dtype=np.int32)
    for name, value in pose.static_objects.items():
        expected[f"static/{name}"] = np.asarray(value, dtype=np.float64)

    segmentation = PoseCapability.SEGMENTATION in pose.capabilities
    for identity in pose.identities:
        points, point_mask = pose.get_identity_poses(identity)
        mask = np.asarray(point_mask[frames], dtype=bool)
        # padding where the mask is false is not compared; zero it so it compresses
        expected[f"id{identity}/points"] = np.where(
            mask[..., np.newaxis], np.asarray(points[frames], dtype=np.float64), 0.0
        )
        expected[f"id{identity}/point_mask"] = mask
        expected[f"id{identity}/identity_mask"] = np.asarray(
            pose.identity_mask(identity), dtype=bool
        )
        expected[f"id{identity}/bearings"] = np.asarray(
            pose.compute_all_bearings(identity)[frames], dtype=np.float32
        )
        if not segmentation:
            continue
        for start, stop in _segmentation_samples(pose.num_frames):
            count, lengths, seg_points, flags = _valid_segmentation(
                pose.segmentation_window(identity, start, stop)
            )
            key = f"id{identity}/seg/{start}-{stop}"
            expected[f"{key}/count"] = count
            expected[f"{key}/lengths"] = lengths
            # legacy contours are int16; storing them that way halves the golden
            expected[f"{key}/points"] = seg_points.astype(np.int16)
            if flags is not None:
                expected[f"{key}/flags"] = flags
    return expected


def save_golden(path: Path, expected: dict[str, npt.NDArray[Any]]) -> None:
    """Write expectations to a compressed ``.npz`` file.

    Args:
        path: Destination ``.npz`` path; its directory is created if needed.
        expected: Expectations from :func:`record_expectations`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **expected)


def load_golden(path: Path) -> dict[str, npt.NDArray[Any]]:
    """Load expectations written by :func:`save_golden`.

    Args:
        path: The golden ``.npz`` file.

    Returns:
        Mapping of ``.npz`` key to array.

    Raises:
        FileNotFoundError: If the golden has not been recorded.
    """
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _check_segmentation(
    pose: PoseLike, identity: int, expected: dict[str, npt.NDArray[Any]]
) -> None:
    """Compare the valid contours of every recorded window for one identity."""
    for start, stop in _segmentation_samples(pose.num_frames):
        key = f"id{identity}/seg/{start}-{stop}"
        window = pose.segmentation_window(identity, start, stop)
        n = stop - start
        assert window.start == start, key
        assert window.contours.dtype == np.int32, key
        assert window.contour_count.dtype == np.uint32, key
        assert window.contour_length.dtype == np.uint32, key
        assert window.contours.ndim == 4 and window.contours.shape[0] == n, key
        assert window.contours.shape[-1] == 2, key
        assert window.contour_count.shape == (n,), key
        assert window.contour_length.shape == window.contours.shape[:2], key
        count, lengths, points, flags = _valid_segmentation(window)
        np.testing.assert_array_equal(count, expected[f"{key}/count"], err_msg=key)
        np.testing.assert_array_equal(lengths, expected[f"{key}/lengths"], err_msg=key)
        np.testing.assert_array_equal(points, expected[f"{key}/points"], err_msg=key)
        if f"{key}/flags" in expected:
            assert window.external_flag is not None, key
            assert window.external_flag.dtype == np.bool_, key
            np.testing.assert_array_equal(flags, expected[f"{key}/flags"], err_msg=key)
        else:
            assert window.external_flag is None, key


def _check_get_points(
    pose: PoseLike,
    identity: int,
    frames: npt.NDArray[np.int64],
    points: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_],
    identity_mask: npt.NDArray[np.bool_],
) -> None:
    """get_points is (None, None) where absent, else agrees with get_identity_poses."""
    for row in range(0, len(frames), _WINDOW_LENGTH):
        frame = int(frames[row])
        frame_points, frame_mask = pose.get_points(frame, identity)
        if not identity_mask[frame]:
            assert frame_points is None and frame_mask is None, (identity, frame)
            continue
        assert frame_points is not None and frame_mask is not None, (identity, frame)
        np.testing.assert_array_equal(np.asarray(frame_mask, dtype=bool), mask[row])
        np.testing.assert_allclose(
            np.asarray(frame_points, dtype=np.float64)[mask[row]],
            np.asarray(points[frame], dtype=np.float64)[mask[row]],
        )


def _check_iter_segmentation(
    pose: PoseLike, identity: int, expected: dict[str, npt.NDArray[Any]]
) -> None:
    """Default-block iter_segmentation tiles the file and holds the recorded contours."""
    windows = list(pose.iter_segmentation(identity))
    assert [w.start for w in windows] == list(range(0, pose.num_frames, _DEFAULT_BLOCK))
    assert sum(len(w.contour_count) for w in windows) == pose.num_frames
    for start, stop in _segmentation_samples(pose.num_frames):
        key = f"id{identity}/seg/{start}-{stop}"
        # rebuild [start, stop) from the blocks it falls in
        parts = []
        for w in windows:
            lo, hi = max(start, w.start), min(stop, w.start + len(w.contour_count))
            if lo < hi:
                parts.append(_slice_window(w, lo - w.start, hi - w.start))
        counts = np.concatenate([p[0] for p in parts])
        np.testing.assert_array_equal(counts, expected[f"{key}/count"], err_msg=key)
        np.testing.assert_array_equal(
            np.concatenate([p[1] for p in parts]), expected[f"{key}/lengths"], err_msg=key
        )
        np.testing.assert_array_equal(
            np.concatenate([p[2] for p in parts]), expected[f"{key}/points"], err_msg=key
        )
        if f"{key}/flags" in expected:
            assert all(p[3] is not None for p in parts), key
            np.testing.assert_array_equal(
                np.concatenate([p[3] for p in parts]), expected[f"{key}/flags"], err_msg=key
            )
        else:
            assert all(p[3] is None for p in parts), key


def _slice_window(
    window: SegmentationWindow, lo: int, hi: int
) -> tuple[
    npt.NDArray[np.uint32],
    npt.NDArray[np.uint32],
    npt.NDArray[np.int32],
    npt.NDArray[np.bool_] | None,
]:
    """Valid count, lengths, points and flags of rows ``[lo, hi)`` of a window."""
    sub = SegmentationWindow(
        contours=window.contours[lo:hi],
        contour_count=window.contour_count[lo:hi],
        contour_length=window.contour_length[lo:hi],
        external_flag=None if window.external_flag is None else window.external_flag[lo:hi],
        start=window.start + lo,
    )
    return _valid_segmentation(sub)


def check_pose_contract(pose: PoseLike, expected: dict[str, npt.NDArray[Any]]) -> None:
    """Assert that a pose object matches recorded contract expectations.

    Args:
        pose: Pose object under test. Only ``PoseLike`` members are read.
        expected: Expectations from :func:`load_golden`.

    Raises:
        AssertionError: On the first member that differs from the expectation.
    """
    meta = json.loads(str(expected[_META_KEY]))

    assert pose.num_frames == meta["num_frames"]
    assert pose.num_identities == meta["num_identities"]
    assert [int(i) for i in pose.identities] == meta["identities"]
    if meta["cm_per_pixel"] is None:
        assert pose.cm_per_pixel is None
    else:
        assert pose.cm_per_pixel is not None
        assert np.isclose(float(pose.cm_per_pixel), meta["cm_per_pixel"], rtol=1e-6)
    assert pose.source_format == meta["source_format"]
    assert pose.model_metadata == meta["model_metadata"]
    assert pose.external_identities == meta["external_identities"]
    assert pose.num_lixit_keypoints == meta["num_lixit_keypoints"]
    assert pose.hash == meta["hash"]
    assert pose.fps == meta["fps"]
    assert sorted(pose.dynamic_objects) == meta["dynamic_objects"]
    for name in meta["dynamic_objects"]:
        assert pose.get_dynamic_object(name) is not None
    assert pose.get_dynamic_object("not_a_dynamic_object") is None
    assert _segments_to_json(pose.get_connected_segments()) == meta["connected_segments"]
    assert [pose.identity_index_to_display(i) for i in pose.identities] == meta["display_names"]
    np.testing.assert_array_equal(
        np.asarray(pose.get_reduced_point_mask(), dtype=bool), expected["reduced_point_mask"]
    )

    # capabilities and the boolean accessors that mirror them
    capabilities = pose.capabilities
    assert isinstance(capabilities, frozenset)
    assert sorted(c.value for c in capabilities) == meta["capabilities"]
    assert pose.has_social == (PoseCapability.SOCIAL in capabilities)
    assert pose.has_social == (pose.num_identities > 1)
    assert pose.has_segmentation == (PoseCapability.SEGMENTATION in capabilities)
    assert pose.has_bounding_boxes == (PoseCapability.BOUNDING_BOXES in capabilities)

    # static objects
    assert sorted(pose.static_objects) == meta["static_objects"]
    for name in meta["static_objects"]:
        assert pose.has_static_object(name)
        np.testing.assert_allclose(
            np.asarray(pose.static_objects[name], dtype=np.float64),
            expected[f"static/{name}"],
            err_msg=f"static object {name}",
        )
    assert not pose.has_static_object("not_a_static_object")

    # track mapping
    identity_to_track = pose.identity_to_track
    if meta["has_identity_to_track"]:
        assert identity_to_track is not None
        assert identity_to_track.dtype == np.int32
        assert identity_to_track.shape == (pose.num_identities, pose.num_frames)
        np.testing.assert_array_equal(identity_to_track, expected["identity_to_track"])
    else:
        assert identity_to_track is None

    # per identity: masks exact, points where the mask is true
    frames = expected["frames"]
    for identity in pose.identities:
        prefix = f"id{identity}"
        points, point_mask = pose.get_identity_poses(identity)
        assert points.shape[:2] == (pose.num_frames, point_mask.shape[1])
        mask = np.asarray(point_mask[frames], dtype=bool)
        np.testing.assert_array_equal(
            mask, expected[f"{prefix}/point_mask"], err_msg=f"{prefix} point mask"
        )
        np.testing.assert_allclose(
            np.asarray(points[frames], dtype=np.float64)[mask],
            expected[f"{prefix}/points"][mask],
            err_msg=f"{prefix} points",
        )
        identity_mask = np.asarray(pose.identity_mask(identity), dtype=bool)
        np.testing.assert_array_equal(
            identity_mask, expected[f"{prefix}/identity_mask"], err_msg=f"{prefix} identity mask"
        )
        np.testing.assert_array_equal(
            np.asarray(pose.get_identity_point_mask(identity), dtype=bool)[frames],
            mask,
            err_msg=f"{prefix} get_identity_point_mask",
        )
        np.testing.assert_allclose(
            np.asarray(pose.compute_all_bearings(identity)[frames], dtype=np.float32),
            expected[f"{prefix}/bearings"],
            rtol=1e-5,
            equal_nan=True,
            err_msg=f"{prefix} bearings",
        )
        _check_get_points(pose, identity, frames, points, mask, identity_mask)
        boxes = pose.get_bounding_boxes(identity)
        if PoseCapability.BOUNDING_BOXES in capabilities:
            assert boxes is not None and boxes.shape == (pose.num_frames, 2, 2)
        else:
            assert boxes is None
        if PoseCapability.SEGMENTATION in capabilities:
            _check_segmentation(pose, identity, expected)
            _check_iter_segmentation(pose, identity, expected)


def record_sample_goldens(versions: tuple[int, ...] = SAMPLE_VERSIONS) -> list[Path]:
    """Record the goldens of the sample legacy pose files with the legacy readers.

    Args:
        versions: Sample file versions to record.

    Returns:
        Paths of the written golden files.
    """
    # imported here: recording is a legacy-reader job, checking is not
    from jabs.pose_estimation import open_pose_file

    written = []
    for version in versions:
        path = golden_path(version)
        with open_pose_file(sample_path(version)) as pose:
            save_golden(path, record_expectations(pose))
        logger.info("Recorded %s (%d bytes)", path, path.stat().st_size)
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> None:
    """Command line entry point; ``--record`` rewrites the sample goldens.

    Args:
        argv: Command line arguments, defaults to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--record", action="store_true", help="record goldens from the legacy readers"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not args.record:
        parser.error("nothing to do; pass --record to rewrite the goldens")
    record_sample_goldens()


if __name__ == "__main__":
    main()
