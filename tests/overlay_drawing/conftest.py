"""Pose file fixtures for the overlay drawing tests: legacy and unified copies of one pose."""

import shutil
from pathlib import Path

import pytest

from jabs.core.utils import hash_file
from jabs.pose_estimation import conversion_cache

# a 3600-frame v6 file with segmentation for every identity
SEGMENTATION_SOURCE = Path(__file__).parents[1] / "data" / "sample_pose_est_v6.h5"

# a v8 file with bounding boxes, and one generated without them
LEGACY_CORPUS = (
    Path(__file__).parents[2] / "packages" / "jabs-io" / "tests" / "data" / "legacy-pose"
)
BBOX_SOURCE = LEGACY_CORPUS / "v8-with-centers.h5"
NO_BBOX_SOURCE = LEGACY_CORPUS / "v8-without-centers.h5"

POSE_KINDS = ("legacy", "unified")


@pytest.fixture(autouse=True)
def _user_cache_in_tmp(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep conversions made without a project cache out of the real per-user cache."""
    user_cache = tmp_path_factory.mktemp("user-cache")
    monkeypatch.setattr(conversion_cache, "user_cache_dir", lambda _app: str(user_cache))


def copy_pose(source: Path, directory: Path, stem: str, kind: str, version: int) -> Path:
    """Copy ``source`` into ``directory`` as a legacy or a unified pose file.

    Args:
        source: A legacy pose file.
        directory: Where to write the copy.
        stem: Video stem for the copy's name.
        kind: ``"legacy"`` for ``<stem>_pose_est_v<version>.h5``, ``"unified"`` for
            its conversion, named ``<stem>_pose.h5``.
        version: Legacy major version of ``source``.

    Returns:
        The copy.
    """
    directory.mkdir(parents=True, exist_ok=True)
    legacy = directory / f"{stem}_pose_est_v{version}.h5"
    shutil.copyfile(source, legacy)
    if kind == "legacy":
        return legacy
    converted = conversion_cache.ensure_converted(legacy, hash_file(legacy), directory / "cache")
    unified = directory / f"{stem}_pose.h5"
    shutil.copyfile(converted, unified)
    legacy.unlink()
    return unified


@pytest.fixture(scope="session")
def segmentation_poses(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Legacy and unified copies of ``SEGMENTATION_SOURCE``, converted once per session."""
    work = tmp_path_factory.mktemp("segmentation-poses")
    return {
        kind: copy_pose(SEGMENTATION_SOURCE, work / kind, "sample", kind, 6) for kind in POSE_KINDS
    }
