"""Tests for Pose.compute_all_bearings against the original per-frame loop.

Each sample legacy file is converted and opened as ``Pose``.
"""

import shutil
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from jabs.core.pose_skeleton import KeypointIndex
from jabs.pose_estimation.conversion_cache import open_unified
from jabs.pose_estimation.pose import Pose

_DATA_DIR = Path(__file__).parent.parent / "data"
_SAMPLE_FILES = [f"sample_pose_est_v{v}.h5" for v in (2, 3, 4, 5, 6)]


@pytest.fixture(scope="module", params=_SAMPLE_FILES)
def sample_pose(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[Pose]:
    """Each sample pose file, converted from a temporary copy and opened as Pose."""
    root = tmp_path_factory.mktemp("bearings")
    path = root / request.param
    shutil.copy(_DATA_DIR / request.param, path)
    with open_unified(path, root / "cache") as pose:
        yield pose


def _reference_bearings(poses: Pose, identity: int) -> npt.NDArray[np.float32]:
    """Original per-frame loop from compute_all_bearings, kept as a reference."""
    use_nose = not poses.get_reduced_point_mask()[KeypointIndex.BASE_NECK.value]
    bearings = np.full(poses.num_frames, np.nan, dtype=np.float32)
    for i in range(poses.num_frames):
        points, _ = poses.get_points(i, identity)
        if points is not None:
            bearings[i] = poses.compute_bearing(points, use_nose)
    return bearings


def test_bearings_match_reference(sample_pose: Pose) -> None:
    """Bearings equal the original per-frame computation for every identity."""
    for identity in sample_pose.identities:
        np.testing.assert_array_equal(
            sample_pose.compute_all_bearings(identity),
            _reference_bearings(sample_pose, identity),
            err_msg=f"identity {identity}",
        )


def test_bearings_fall_back_to_nose_without_base_neck(
    sample_pose: Pose, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pose without a base neck keypoint uses the nose, as the original loop did."""
    reduced = np.array(sample_pose.get_reduced_point_mask(), dtype=bool)
    reduced[KeypointIndex.BASE_NECK.value] = False
    monkeypatch.setattr(sample_pose, "get_reduced_point_mask", lambda: reduced)

    identity = sample_pose.identities[0]
    np.testing.assert_array_equal(
        sample_pose.compute_all_bearings(identity), _reference_bearings(sample_pose, identity)
    )
