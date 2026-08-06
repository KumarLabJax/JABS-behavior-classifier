"""Tests for per-video feature cache status scanning and aggregation."""

from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from jabs.core.enums import CacheFormat
from jabs.core.types import FeatureCacheMetadata, PerFrameCacheData
from jabs.io.feature_cache import IdentityCacheInfo
from jabs.io.feature_cache.hdf5 import HDF5FeatureCacheWriter
from jabs.project.feature_cache_status import (
    VideoFeatureCacheStatus,
    scan_project_feature_cache,
    scan_video_feature_cache,
)

_N_FRAMES = 10
_CURRENT_VERSION = 17


def _identity_cache(
    identity: int = 0,
    window_sizes: frozenset[int] = frozenset(),
    feature_version: int = _CURRENT_VERSION,
    cache_format: CacheFormat = CacheFormat.PARQUET,
    per_frame_present: bool = True,
    distance_scale_factor: float | None = None,
    size_bytes: int = 100,
    directory: Path | None = None,
) -> IdentityCacheInfo:
    """Build an IdentityCacheInfo without touching the filesystem."""
    return IdentityCacheInfo(
        directory=directory or Path(f"/features/video/{identity}"),
        identity=identity,
        cache_format=cache_format,
        feature_version=feature_version,
        pose_hash="hash",
        num_frames=_N_FRAMES,
        distance_scale_factor=distance_scale_factor,
        window_sizes=window_sizes,
        per_frame_present=per_frame_present,
        size_bytes=size_bytes,
    )


def _status(*caches: IdentityCacheInfo, expected_identity_count: int | None = None):
    """Build a VideoFeatureCacheStatus around the given identity caches."""
    return VideoFeatureCacheStatus(
        video="video.mp4",
        cache_dir=Path("/features/video"),
        identity_caches=caches,
        current_feature_version=_CURRENT_VERSION,
        expected_identity_count=expected_identity_count,
    )


def _write_hdf5_cache(
    identity_dir: Path,
    identity: int,
    window_sizes: tuple[int, ...] = (),
    feature_version: int = _CURRENT_VERSION,
) -> None:
    """Write a small HDF5 feature cache for one identity."""
    writer = HDF5FeatureCacheWriter()
    metadata = FeatureCacheMetadata(
        feature_version=feature_version,
        identity=identity,
        num_frames=_N_FRAMES,
        pose_hash="posehash",
    )
    data = PerFrameCacheData(
        frame_valid=np.ones(_N_FRAMES, dtype=np.uint8),
        features={"mod feat": np.zeros(_N_FRAMES)},
    )
    writer.write_per_frame(identity_dir, metadata, data)
    for size in window_sizes:
        writer.write_window(identity_dir, metadata, size, {"mod mean feat": np.zeros(_N_FRAMES)})


# ---------------------------------------------------------------------------
# VideoFeatureCacheStatus aggregation
# ---------------------------------------------------------------------------


def test_empty_status_reports_nothing_cached():
    """A status with no identity caches reports no cached features."""
    status = _status(expected_identity_count=3)

    assert status.has_cached_features is False
    assert status.cached_identity_count == 0
    assert status.window_sizes == ()
    assert status.partial_window_sizes == ()
    assert status.is_complete is False
    assert status.is_stale is False
    assert status.cm_units is None
    assert status.size_bytes == 0


def test_window_sizes_are_intersection_across_identities():
    """Only window sizes cached for every identity count as available."""
    status = _status(
        _identity_cache(identity=0, window_sizes=frozenset({5, 10, 30})),
        _identity_cache(identity=1, window_sizes=frozenset({5, 10})),
        _identity_cache(identity=2, window_sizes=frozenset({10})),
    )

    assert status.window_sizes == (10,)
    assert status.partial_window_sizes == (5, 30)


def test_window_sizes_merge_across_pose_hash_directories():
    """Caches sharing an identity are merged rather than counted separately."""
    status = _status(
        _identity_cache(identity=0, window_sizes=frozenset({5}), directory=Path("/f/v/hash_a/0")),
        _identity_cache(identity=0, window_sizes=frozenset({10}), directory=Path("/f/v/hash_b/0")),
        expected_identity_count=1,
    )

    assert status.cached_identity_count == 1
    assert status.window_sizes == (5, 10)
    assert status.partial_window_sizes == ()
    assert status.is_complete is True


def test_has_window_features_requires_every_identity():
    """With no identities named, every identity in the video must be covered."""
    status = _status(
        _identity_cache(identity=0, window_sizes=frozenset({5, 10})),
        _identity_cache(identity=1, window_sizes=frozenset({5})),
        expected_identity_count=2,
    )

    assert status.has_window_features(5) is True
    assert status.has_window_features(10) is False


def test_has_window_features_unknown_identity_count():
    """Without a known identity count, full coverage cannot be confirmed."""
    status = _status(_identity_cache(identity=0, window_sizes=frozenset({5})))

    assert status.has_window_features(5) is False


def test_has_window_features_for_named_identities():
    """Naming the identities that matter ignores the uncached ones."""
    status = _status(
        _identity_cache(identity=0, window_sizes=frozenset({5})),
        expected_identity_count=3,
    )

    assert status.has_window_features(5, [0]) is True
    assert status.has_window_features(5, [0, 1]) is False
    # nothing needed, so nothing is missing
    assert status.has_window_features(5, []) is True


def test_has_window_features_without_any_cache():
    """A video with no cache never covers a window size."""
    assert _status(expected_identity_count=2).has_window_features(5) is False
    assert _status().has_window_features(5, [0]) is False


@pytest.mark.parametrize(
    ("cached_identities", "expected_count", "complete"),
    [(2, 2, True), (1, 2, False), (2, None, False)],
    ids=["all-cached", "partially-cached", "unknown-identity-count"],
)
def test_is_complete_requires_every_identity(cached_identities, expected_count, complete):
    """is_complete is only True when every known identity has a cache."""
    caches = tuple(_identity_cache(identity=i) for i in range(cached_identities))
    status = _status(*caches, expected_identity_count=expected_count)

    assert status.is_complete is complete


def test_is_complete_false_when_per_frame_features_missing():
    """An identity whose per-frame features are absent makes the cache incomplete."""
    status = _status(
        _identity_cache(identity=0),
        _identity_cache(identity=1, per_frame_present=False),
        expected_identity_count=2,
    )

    assert status.is_complete is False


def test_stale_cache_detected_from_feature_version():
    """A cache written by another feature version is reported as stale."""
    status = _status(_identity_cache(feature_version=_CURRENT_VERSION - 1))

    assert status.is_stale is True
    assert status.feature_versions == (_CURRENT_VERSION - 1,)


def test_current_cache_not_stale():
    """A cache written by the current feature version is not stale."""
    assert _status(_identity_cache()).is_stale is False


def test_mixed_cache_formats_reported():
    """Both formats are reported when the cache directory is mixed."""
    status = _status(
        _identity_cache(identity=0, cache_format=CacheFormat.HDF5),
        _identity_cache(identity=1, cache_format=CacheFormat.PARQUET),
    )

    assert set(status.cache_formats) == {CacheFormat.HDF5, CacheFormat.PARQUET}


@pytest.mark.parametrize(
    ("scales", "expected"),
    [((0.1, 0.1), True), ((None, None), False), ((0.1, None), None)],
    ids=["cm", "pixels", "mixed"],
)
def test_cm_units_reflects_distance_scale_factors(scales, expected):
    """cm_units is True/False only when every cache agrees."""
    caches = tuple(
        _identity_cache(identity=i, distance_scale_factor=scale) for i, scale in enumerate(scales)
    )

    assert _status(*caches).cm_units is expected


def test_size_bytes_sums_identity_caches():
    """The reported size is the total across every identity cache."""
    status = _status(
        _identity_cache(identity=0, size_bytes=100),
        _identity_cache(identity=1, size_bytes=250),
    )

    assert status.size_bytes == 350


# ---------------------------------------------------------------------------
# scan_video_feature_cache
# ---------------------------------------------------------------------------


def test_scan_video_finds_flat_layout(tmp_path):
    """Identity directories directly under the video directory are found."""
    feature_dir = tmp_path / "features"
    _write_hdf5_cache(feature_dir / "video1" / "0", identity=0, window_sizes=(5, 10))
    _write_hdf5_cache(feature_dir / "video1" / "1", identity=1, window_sizes=(5,))

    status = scan_video_feature_cache(
        feature_dir,
        "video1.mp4",
        current_feature_version=_CURRENT_VERSION,
        expected_identity_count=2,
    )

    assert status.cache_dir == feature_dir / "video1"
    assert status.cached_identity_count == 2
    assert status.window_sizes == (5,)
    assert status.partial_window_sizes == (10,)
    assert status.cache_formats == (CacheFormat.HDF5,)
    assert status.is_complete is True


def test_scan_video_finds_pose_hash_layout(tmp_path):
    """Identity directories nested under a pose-hash directory are found."""
    feature_dir = tmp_path / "features"
    _write_hdf5_cache(feature_dir / "video1" / "deadbeef" / "0", identity=0, window_sizes=(5,))

    status = scan_video_feature_cache(
        feature_dir, "video1.mp4", current_feature_version=_CURRENT_VERSION
    )

    assert status.cached_identity_count == 1
    assert status.window_sizes == (5,)
    assert status.identity_caches[0].directory == feature_dir / "video1" / "deadbeef" / "0"


def test_scan_video_accepts_pose_filename(tmp_path):
    """A pose filename resolves to the same cache directory as the video."""
    feature_dir = tmp_path / "features"
    _write_hdf5_cache(feature_dir / "video1" / "0", identity=0)

    status = scan_video_feature_cache(
        feature_dir, "video1_pose_est_v6.h5", current_feature_version=_CURRENT_VERSION
    )

    assert status.cache_dir == feature_dir / "video1"
    assert status.has_cached_features is True


def test_scan_video_with_no_cache_directory(tmp_path):
    """A video with no cache directory reports nothing cached, without error."""
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()

    status = scan_video_feature_cache(
        feature_dir, "video1.mp4", current_feature_version=_CURRENT_VERSION
    )

    assert status.cache_dir == feature_dir / "video1"
    assert status.has_cached_features is False


def test_scan_video_ignores_stray_files_and_empty_directories(tmp_path):
    """Files and cache-free directories under the video directory are skipped."""
    feature_dir = tmp_path / "features"
    video_dir = feature_dir / "video1"
    _write_hdf5_cache(video_dir / "0", identity=0)
    (video_dir / "1").mkdir()  # identity dir with no cache files
    (video_dir / "notes.txt").write_text("stray file")

    status = scan_video_feature_cache(
        feature_dir, "video1.mp4", current_feature_version=_CURRENT_VERSION
    )

    assert status.cached_identity_count == 1


# ---------------------------------------------------------------------------
# scan_project_feature_cache
# ---------------------------------------------------------------------------


def test_scan_project_covers_every_video(tmp_path):
    """Every project video appears in the result, cached or not."""
    feature_dir = tmp_path / "features"
    _write_hdf5_cache(feature_dir / "video1" / "0", identity=0, window_sizes=(5,))

    project = mock.MagicMock()
    project.feature_dir = feature_dir
    project.video_manager.videos = ["video1.mp4", "video2.mp4"]
    project.video_manager.get_video_identity_count.return_value = 1

    statuses = scan_project_feature_cache(project)

    assert set(statuses) == {"video1.mp4", "video2.mp4"}
    assert statuses["video1.mp4"].window_sizes == (5,)
    assert statuses["video2.mp4"].has_cached_features is False


def test_scan_project_stops_early_when_asked(tmp_path):
    """A should_continue predicate returning False ends the scan with partial results."""
    feature_dir = tmp_path / "features"
    for video in ("video1", "video2", "video3"):
        _write_hdf5_cache(feature_dir / video / "0", identity=0)

    project = mock.MagicMock()
    project.feature_dir = feature_dir
    project.video_manager.videos = ["video1.mp4", "video2.mp4", "video3.mp4"]
    project.video_manager.get_video_identity_count.return_value = 1

    # allow the first two videos, then stop
    calls = iter([True, True, False])
    statuses = scan_project_feature_cache(project, should_continue=lambda: next(calls))

    assert set(statuses) == {"video1.mp4", "video2.mp4"}


def test_scan_project_tolerates_unknown_identity_count(tmp_path):
    """A video whose identity count cannot be read is still scanned."""
    feature_dir = tmp_path / "features"
    _write_hdf5_cache(feature_dir / "video1" / "0", identity=0)

    project = mock.MagicMock()
    project.feature_dir = feature_dir
    project.video_manager.videos = ["video1.mp4"]
    project.video_manager.get_video_identity_count.side_effect = ValueError("no pose file")

    status = scan_project_feature_cache(project)["video1.mp4"]

    assert status.expected_identity_count is None
    assert status.has_cached_features is True


def test_per_frame_features_merged_across_pose_hash_directories():
    """An identity cached under several pose hashes only needs per-frame features once.

    Matches how window sizes merge: a partially written cache under one pose hash
    must not make the identity (or the video) look incomplete.
    """
    status = _status(
        _identity_cache(identity=0, per_frame_present=False, directory=Path("/f/v/hash_a/0")),
        _identity_cache(identity=0, per_frame_present=True, directory=Path("/f/v/hash_b/0")),
        expected_identity_count=1,
    )

    assert status.identities_missing_per_frame == ()
    assert status.is_complete is True


def test_identity_with_no_per_frame_features_anywhere_is_reported():
    """An identity whose only cache lacks per-frame features is named."""
    status = _status(
        _identity_cache(identity=0),
        _identity_cache(identity=1, per_frame_present=False),
        expected_identity_count=2,
    )

    assert status.identities_missing_per_frame == (1,)
    assert status.is_complete is False


def test_cache_formats_sorted_by_enum_value():
    """Formats are ordered by their on-disk value, not by the enum's repr."""
    status = _status(
        _identity_cache(identity=0, cache_format=CacheFormat.PARQUET),
        _identity_cache(identity=1, cache_format=CacheFormat.HDF5),
    )

    assert status.cache_formats == (CacheFormat.HDF5, CacheFormat.PARQUET)
    assert [fmt.value for fmt in status.cache_formats] == ["hdf5", "parquet"]


def test_stale_cache_is_not_coverage():
    """A cache from another feature version will be recomputed, so it is not coverage.

    This is the routine case after a JABS upgrade that bumps FEATURE_VERSION: the
    window features are on disk, but the run rebuilds them.
    """
    status = _status(
        _identity_cache(
            identity=0, window_sizes=frozenset({5}), feature_version=_CURRENT_VERSION - 1
        ),
        expected_identity_count=1,
    )

    # still reported as present on disk, and flagged stale, for display purposes
    assert status.window_sizes == (5,)
    assert status.is_stale is True
    # but not counted as cached for the train/classify warning
    assert status.has_window_features(5) is False
    assert status.has_window_features(5, [0]) is False


def test_cache_without_per_frame_features_is_not_coverage():
    """Window features are recomputed along with the per-frame features they need."""
    status = _status(
        _identity_cache(identity=0, window_sizes=frozenset({5}), per_frame_present=False),
        expected_identity_count=1,
    )

    assert status.has_window_features(5) is False


def test_coverage_is_not_stitched_together_from_different_pose_hashes():
    """A window size only counts when one cache directory is loadable and has it.

    Here the size exists only in a stale directory and the current directory lacks
    it, so neither could serve the run even though merging the fields separately
    would suggest otherwise.
    """
    status = _status(
        _identity_cache(
            identity=0,
            window_sizes=frozenset({5}),
            feature_version=_CURRENT_VERSION - 1,
            directory=Path("/f/v/stale_hash/0"),
        ),
        _identity_cache(
            identity=0,
            window_sizes=frozenset({10}),
            directory=Path("/f/v/current_hash/0"),
        ),
        expected_identity_count=1,
    )

    assert status.has_window_features(5) is False
    # the current directory's own window size is still coverage
    assert status.has_window_features(10) is True


def test_current_cache_is_coverage():
    """A current cache with per-frame features covers its window sizes."""
    status = _status(
        _identity_cache(identity=0, window_sizes=frozenset({5})),
        expected_identity_count=1,
    )

    assert status.has_window_features(5) is True
