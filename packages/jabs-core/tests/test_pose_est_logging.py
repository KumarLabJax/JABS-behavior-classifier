"""Tests that PoseEstimation logs through its own module logger, not the root logger."""

import logging
from pathlib import Path

import numpy as np
import pytest

from jabs.core.abstract.pose_est import PoseEstimation

POSE_EST_LOGGER = "jabs.core.abstract.pose_est"


class _StubPose(PoseEstimation):
    """Minimal concrete PoseEstimation used to exercise the base class directly."""

    def get_points(self, frame_index, identity, scale=None):
        """Not exercised by these tests."""
        raise NotImplementedError

    def get_identity_poses(self, identity, scale=None):
        """Not exercised by these tests."""
        raise NotImplementedError

    def get_identity_point_mask(self, identity):
        """Not exercised by these tests."""
        raise NotImplementedError

    def get_reduced_point_mask(self):
        """Report every keypoint except BASE_NECK as valid, forcing the nose fallback."""
        mask = np.ones(len(PoseEstimation.KeypointIndex), dtype=bool)
        mask[PoseEstimation.KeypointIndex.BASE_NECK.value] = False
        return mask

    def identity_mask(self, identity):
        """Not exercised by these tests."""
        raise NotImplementedError

    @property
    def identity_to_track(self):
        """Not exercised by these tests."""
        raise NotImplementedError

    @property
    def format_major_version(self):
        """Stub version, only needs to satisfy the abstract interface."""
        return 2


@pytest.fixture
def stub_pose(tmp_path: Path) -> _StubPose:
    """A stub pose object backed by a real (empty) file so hashing succeeds."""
    pose_file = tmp_path / "video_pose_est_v2.h5"
    pose_file.write_bytes(b"")
    return _StubPose(pose_file)


def test_bearing_fallback_warning_uses_module_logger(stub_pose, caplog) -> None:
    """The nose-fallback warning is attributed to the module logger, not the root logger."""
    with caplog.at_level(logging.WARNING, logger=POSE_EST_LOGGER):
        stub_pose.compute_all_bearings(identity=0)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert warnings[0].name == POSE_EST_LOGGER


def test_cache_delete_failure_warning_uses_module_logger(tmp_path, caplog, monkeypatch) -> None:
    """A cache file that cannot be deleted warns through the module logger."""
    pose_file = tmp_path / "video_pose_est_v2.h5"
    pose_file.write_bytes(b"")

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    # _cache_file_path() derives this name from the pose file name
    (cache_dir / "video_pose_est_v2_cache.h5").write_bytes(b"")

    def _fail_unlink(self, *args, **kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "unlink", _fail_unlink)

    with caplog.at_level(logging.WARNING, logger=POSE_EST_LOGGER):
        _StubPose(pose_file, cache_dir=cache_dir)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert warnings[0].name == POSE_EST_LOGGER
    assert "Unable to delete old cache file" in warnings[0].getMessage()
