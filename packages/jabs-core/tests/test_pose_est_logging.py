"""Tests that PoseEstimation logs through its own module logger, not the root logger."""

import logging
from pathlib import Path
from typing import NoReturn

import numpy as np
import numpy.typing as npt
import pytest

from jabs.core.abstract.pose_est import PoseEstimation

POSE_EST_LOGGER = "jabs.core.abstract.pose_est"


class _StubPose(PoseEstimation):
    """Minimal concrete PoseEstimation used to exercise the base class directly."""

    def get_points(self, frame_index: int, identity: int, scale: float | None = None) -> NoReturn:
        """Not exercised by these tests."""
        raise NotImplementedError

    def get_identity_poses(self, identity: int, scale: float | None = None) -> NoReturn:
        """Not exercised by these tests."""
        raise NotImplementedError

    def get_identity_point_mask(self, identity: int) -> NoReturn:
        """Not exercised by these tests."""
        raise NotImplementedError

    def get_reduced_point_mask(self) -> npt.NDArray[np.bool_]:
        """Report every keypoint except BASE_NECK as valid, forcing the nose fallback."""
        mask = np.ones(len(PoseEstimation.KeypointIndex), dtype=bool)
        mask[PoseEstimation.KeypointIndex.BASE_NECK.value] = False
        return mask

    def identity_mask(self, identity: int) -> NoReturn:
        """Not exercised by these tests."""
        raise NotImplementedError

    @property
    def identity_to_track(self) -> NoReturn:
        """Not exercised by these tests."""
        raise NotImplementedError

    @property
    def format_major_version(self) -> int:
        """Stub version, only needs to satisfy the abstract interface."""
        return 2


def _pose_est_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """Captured WARNING records emitted by the pose_est module logger."""
    return [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING and record.name == POSE_EST_LOGGER
    ]


@pytest.fixture
def stub_pose(tmp_path: Path) -> _StubPose:
    """A stub pose object backed by a real (empty) file so hashing succeeds."""
    pose_file = tmp_path / "video_pose_est_v2.h5"
    pose_file.write_bytes(b"")
    return _StubPose(pose_file)


def test_bearing_fallback_warning_uses_module_logger(
    stub_pose: _StubPose, caplog: pytest.LogCaptureFixture
) -> None:
    """The nose-fallback warning is attributed to the module logger, not the root logger."""
    with caplog.at_level(logging.WARNING, logger=POSE_EST_LOGGER):
        stub_pose.compute_all_bearings(identity=0)

    assert len(_pose_est_warnings(caplog)) == 1


def test_cache_delete_failure_warning_uses_module_logger(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cache file that cannot be deleted warns through the module logger."""
    pose_file = tmp_path / "video_pose_est_v2.h5"
    pose_file.write_bytes(b"")

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    # _cache_file_path() derives this name from the pose file name
    (cache_dir / "video_pose_est_v2_cache.h5").write_bytes(b"")

    def _fail_unlink(self: Path, *args: object, **kwargs: object) -> NoReturn:
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "unlink", _fail_unlink)

    with caplog.at_level(logging.WARNING, logger=POSE_EST_LOGGER):
        _StubPose(pose_file, cache_dir=cache_dir)

    warnings = _pose_est_warnings(caplog)
    assert len(warnings) == 1
    assert "Unable to delete old cache file" in warnings[0].getMessage()
