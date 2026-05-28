"""Unit tests for jabs.scripts.classify helpers."""

from pathlib import Path

import pytest

from jabs.scripts.classify import _require_pose_file_name


@pytest.mark.parametrize(
    "name",
    [
        "sample_pose_est_v2.h5",
        "sample_pose_est_v6.h5",
        "sample_pose_est_v12.h5",
        "nested_name_pose_est_v8.h5",
    ],
)
def test_require_pose_file_name_accepts_canonical(name: str) -> None:
    """Canonical ``*_pose_est_vN.h5`` filenames pass validation."""
    _require_pose_file_name(Path("/some/dir") / name)


@pytest.mark.parametrize(
    "name",
    [
        "sample.mp4",
        "sample.h5",
        "sample_v6.h5",
        "sample_pose_est.h5",
        "sample_pose_est_v6.hdf5",
        "predictions_pose_est_v6.h5.bak",
    ],
)
def test_require_pose_file_name_rejects_invalid(name: str) -> None:
    """Names that do not match ``*_pose_est_vN.h5`` are rejected."""
    with pytest.raises(ValueError, match="not a valid pose file path"):
        _require_pose_file_name(Path(name))
