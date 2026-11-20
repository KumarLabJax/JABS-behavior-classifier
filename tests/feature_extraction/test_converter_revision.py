"""Feature and hull caches of a converted pose file follow the converter revision.

A converter change regenerates conversions; features and convex hulls computed
from the old conversion must not be served for the new one. A unified pose
file was never converted, so its caches do not depend on the revision.
"""

import shutil
from pathlib import Path
from unittest import mock

import pytest

from jabs.core.utils import hash_file
from jabs.feature_extraction import IdentityFeatures
from jabs.pose_estimation import conversion_cache
from jabs.pose_estimation.conversion_cache import ensure_converted, open_unified

SAMPLE_V4 = Path(__file__).parents[1] / "data" / "sample_pose_est_v4.h5"


@pytest.fixture
def pose_files(tmp_path: Path) -> dict[str, Path]:
    """A legacy v4 file and a unified file converted from it."""
    legacy = tmp_path / "videos" / "legacy_pose_est_v4.h5"
    legacy.parent.mkdir()
    shutil.copyfile(SAMPLE_V4, legacy)
    unified = tmp_path / "videos" / "unified_pose.h5"
    shutil.copyfile(ensure_converted(legacy, hash_file(legacy), tmp_path / "scratch"), unified)
    return {"legacy": legacy, "unified": unified}


@pytest.fixture
def compute_spy(monkeypatch: pytest.MonkeyPatch) -> mock.Mock:
    """Count per-frame feature computations (cache misses)."""
    name = "_IdentityFeatures__initialize_from_pose_estimation"
    real = getattr(IdentityFeatures, name)
    spy = mock.Mock()

    def compute(self: IdentityFeatures, pose: object) -> None:
        spy(pose)
        real(self, pose)

    monkeypatch.setattr(IdentityFeatures, name, compute)
    return spy


def _features(path: Path, cache: Path, feature_dir: Path) -> None:
    """Load identity 0's per-frame features through the cache."""
    with open_unified(path, cache) as pose:
        IdentityFeatures(path, 0, feature_dir, pose)


@pytest.mark.parametrize(("kind", "recomputed"), [("legacy", True), ("unified", False)])
def test_revision_bump_invalidates_features_of_converted_files_only(
    pose_files: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compute_spy: mock.Mock,
    kind: str,
    recomputed: bool,
) -> None:
    """After a bump a converted file's features are recomputed; a unified file's are reused."""
    cache, feature_dir = tmp_path / "cache", tmp_path / "features"
    _features(pose_files[kind], cache, feature_dir)
    _features(pose_files[kind], cache, feature_dir)
    assert compute_spy.call_count == 1

    bumped = conversion_cache._converter_revision() + 1
    monkeypatch.setattr(conversion_cache, "_converter_revision", lambda: bumped)
    _features(pose_files[kind], cache, feature_dir)
    assert compute_spy.call_count == (2 if recomputed else 1)


def test_hull_cache_of_a_converted_file_is_keyed_by_revision(
    pose_files: dict[str, Path], tmp_path: Path
) -> None:
    """A conversion's hulls live under ``<hash>-r<revision>``; a unified file's under ``<hash>``."""
    cache = tmp_path / "cache"
    for kind in ("legacy", "unified"):
        with open_unified(pose_files[kind], cache) as pose:
            pose.get_identity_convex_hulls(0)
    revision = conversion_cache._converter_revision()
    legacy_hash, unified_hash = (hash_file(pose_files[k]) for k in ("legacy", "unified"))
    assert (cache / "convex_hulls" / "legacy" / f"{legacy_hash}-r{revision}").is_dir()
    assert (cache / "convex_hulls" / "unified" / unified_hash).is_dir()
