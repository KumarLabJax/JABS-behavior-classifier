"""Pose file fixtures for the CLI tests: one legacy and one unified file of the same pose."""

import shutil
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest

from jabs.core.utils import hash_file
from jabs.pose_estimation import conversion_cache

LEGACY_CORPUS = (
    Path(__file__).parents[2] / "packages" / "jabs-io" / "tests" / "data" / "legacy-pose"
)

# a 120-frame v8 file: three identities, bounding boxes, model metadata
LEGACY_SOURCE = LEGACY_CORPUS / "v8-with-centers.h5"
FRAMES = 120

# a legacy file the converter refuses (an unknown poseest dataset)
REFUSED_SOURCE = LEGACY_CORPUS / "v8-unknown-dataset.h5"

POSE_KINDS = ("legacy", "unified")

PoseFileFactory = Callable[[Path, str, str], Path]


@pytest.fixture(autouse=True)
def _user_cache_in_tmp(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep conversions made without a project cache out of the real per-user cache."""
    user_cache = tmp_path_factory.mktemp("user-cache")
    monkeypatch.setattr(conversion_cache, "user_cache_dir", lambda _app: str(user_cache))


@pytest.fixture(scope="session")
def unified_source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The unified conversion of ``LEGACY_SOURCE``, converted once per session."""
    work = tmp_path_factory.mktemp("unified-source")
    legacy = work / "source_pose_est_v8.h5"
    shutil.copyfile(LEGACY_SOURCE, legacy)
    return conversion_cache.ensure_converted(legacy, hash_file(legacy), work)


@pytest.fixture
def make_pose_file(unified_source: Path) -> PoseFileFactory:
    """Return a factory writing ``<stem>_pose_est_v8.h5`` (legacy) or ``<stem>_pose.h5`` (unified).

    Both kinds hold the same pose.
    """

    def make(directory: Path, stem: str, kind: str) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        if kind == "legacy":
            path = directory / f"{stem}_pose_est_v8.h5"
            shutil.copyfile(LEGACY_SOURCE, path)
        elif kind == "unified":
            path = directory / f"{stem}_pose.h5"
            shutil.copyfile(unified_source, path)
        else:
            raise ValueError(f"unknown pose kind {kind!r}")
        return path

    return make


def write_video(path: Path, frames: int = FRAMES) -> Path:
    """Write a tiny MJPG video with ``frames`` frames and return its path."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (32, 32))
    try:
        for _ in range(frames):
            writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    finally:
        writer.release()
    return path
