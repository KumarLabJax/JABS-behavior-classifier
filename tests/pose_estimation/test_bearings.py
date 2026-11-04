"""Tests for PoseEstimation.compute_all_bearings against the original per-frame loop."""

import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

import jabs.pose_estimation
from jabs.pose_estimation import PoseEstimation

_DATA_DIR = Path(__file__).parent.parent / "data"
_SAMPLE_FILES = [f"sample_pose_est_v{v}.h5" for v in (2, 3, 4, 5, 6)]


@pytest.fixture(scope="module", params=_SAMPLE_FILES)
def sample_pose(request: pytest.FixtureRequest) -> Iterator[PoseEstimation]:
    """Each sample pose file, opened from a temporary copy."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / request.param
        shutil.copy(_DATA_DIR / request.param, path)
        yield jabs.pose_estimation.open_pose_file(path)


def _reference_bearings(poses: PoseEstimation, identity: int) -> npt.NDArray[np.float32]:
    """Original per-frame loop from compute_all_bearings, kept as a reference."""
    use_nose = not poses.get_reduced_point_mask()[PoseEstimation.KeypointIndex.BASE_NECK.value]
    bearings = np.full(poses.num_frames, np.nan, dtype=np.float32)
    for i in range(poses.num_frames):
        points, _ = poses.get_points(i, identity)
        if points is not None:
            bearings[i] = poses.compute_bearing(points, use_nose)
    return bearings


def test_bearings_match_reference(sample_pose: PoseEstimation) -> None:
    """Bearings equal the original per-frame computation for every identity."""
    for identity in sample_pose.identities:
        np.testing.assert_array_equal(
            sample_pose.compute_all_bearings(identity),
            _reference_bearings(sample_pose, identity),
            err_msg=f"identity {identity}",
        )


def test_bearings_fall_back_to_nose_without_base_neck(
    sample_pose: PoseEstimation, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pose without a base neck keypoint uses the nose, as the original loop did."""
    reduced = np.array(sample_pose.get_reduced_point_mask(), dtype=bool)
    reduced[PoseEstimation.KeypointIndex.BASE_NECK.value] = False
    monkeypatch.setattr(sample_pose, "get_reduced_point_mask", lambda: reduced)

    identity = sample_pose.identities[0]
    np.testing.assert_array_equal(
        sample_pose.compute_all_bearings(identity), _reference_bearings(sample_pose, identity)
    )
