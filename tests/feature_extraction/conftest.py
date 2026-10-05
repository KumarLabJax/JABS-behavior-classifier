"""Shared fixtures for feature extraction tests."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import h5py
import pytest

import jabs.pose_estimation as pose_est
from jabs.pose_estimation import PoseEstimation

_SAMPLE_POSE_V5 = Path(__file__).parent.parent / "data" / "sample_pose_est_v5.h5"

# 100 frames from the middle of the 1800-frame sample. The slice keeps all three
# identities and the gaps in identity 0 (frames 668-680), so NaN handling is still exercised.
_SHORT_FRAMES = slice(620, 720)


def _write_short_pose(source: Path, dest: Path, frames: slice) -> None:
    """Copy an HDF5 pose file, keeping only ``frames`` of every per-frame dataset.

    Datasets whose first axis is the frame axis are sliced. Everything else (static
    objects, per-identity arrays, group and dataset attributes) is copied unchanged.

    Args:
        source: Pose file to copy.
        dest: Path of the shortened copy to write.
        frames: Frame range to keep.
    """
    with h5py.File(source, "r") as src, h5py.File(dest, "w") as dst:
        num_frames = src["poseest/points"].shape[0]
        for key, value in src.attrs.items():
            dst.attrs[key] = value

        def _copy(name: str, obj: h5py.Group | h5py.Dataset) -> None:
            if isinstance(obj, h5py.Group):
                target = dst.require_group(name)
            else:
                data = obj[()]
                if obj.ndim > 0 and obj.shape[0] == num_frames:
                    data = data[frames]
                target = dst.create_dataset(name, data=data)
            for key, value in obj.attrs.items():
                target.attrs[key] = value

        src.visititems(_copy)


@pytest.fixture(scope="module")
def pose_est_v5_short(tmp_path_factory: pytest.TempPathFactory) -> Iterator[PoseEstimation]:
    """Open a 100-frame slice of the sample v5 pose file.

    Window features cost time per frame, so tests that only need real feature names,
    NaN gaps and cache round trips use this instead of the full 1800-frame sample.

    Yields:
        PoseEstimationV5 backed by a temporary shortened copy of the sample file.
    """
    pose_path = tmp_path_factory.mktemp("short_pose") / _SAMPLE_POSE_V5.name
    _write_short_pose(_SAMPLE_POSE_V5, pose_path, _SHORT_FRAMES)
    yield pose_est.open_pose_file(pose_path)
    shutil.rmtree(pose_path.parent, ignore_errors=True)
