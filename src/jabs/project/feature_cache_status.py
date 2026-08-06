"""Per-video status of a JABS project's on-disk feature cache.

Aggregates the per-identity cache summaries produced by
:func:`jabs.io.feature_cache.inspect_identity_cache` into one status object per
video, answering the questions a user asks about a project's cache: are
features cached for this video, for which window sizes, in which format, and is
what is on disk still current?

Scanning only reads small metadata files (HDF5 attributes or ``metadata.json``),
never feature data, so a whole project can be scanned quickly. Nothing here
validates a cache against its pose file: doing so would require hashing every
pose file, which is far too expensive for a status scan. A cache whose pose file
has changed is therefore still reported as present; it will be detected and
recomputed when the features are actually loaded.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from jabs.core.enums import CacheFormat
from jabs.core.utils import pose_file_stem
from jabs.io.feature_cache import IdentityCacheInfo, inspect_identity_cache

if TYPE_CHECKING:
    from .project import Project

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoFeatureCacheStatus:
    """Status of the cached features for a single video.

    Attributes:
        video: Video filename this status describes.
        cache_dir: Directory holding this video's cached features
            (``<project>/jabs/features/<video stem>``). May not exist.
        identity_caches: Per-identity cache summaries found under ``cache_dir``,
            ordered by identity. Identities with no cache are absent. More than
            one entry can share an identity when caches exist under multiple
            pose-hash subdirectories (written by the CLI tools' ``--use-pose-hash``
            option).
        current_feature_version: The running application's ``FEATURE_VERSION``,
            used to decide whether the cached features are stale.
        expected_identity_count: Number of identities in the video, or ``None``
            when unknown. Used to report whether every identity is cached.
    """

    video: str
    cache_dir: Path
    identity_caches: tuple[IdentityCacheInfo, ...]
    current_feature_version: int
    expected_identity_count: int | None = None

    def _window_sizes_by_identity(self) -> dict[int, frozenset[int]]:
        """Group the cached window sizes by identity.

        Reports what is on disk, whether or not it could still be loaded; see
        :meth:`_loadable_window_sizes_by_identity` for the stricter view used to
        answer coverage questions.

        Caches for the same identity under different pose-hash subdirectories
        are merged, so an identity counts as having a window size if any of its
        caches provides it.
        """
        grouped: dict[int, set[int]] = {}
        for info in self.identity_caches:
            grouped.setdefault(info.identity, set()).update(info.window_sizes)
        return {identity: frozenset(sizes) for identity, sizes in grouped.items()}

    def _loadable_window_sizes_by_identity(self) -> dict[int, frozenset[int]]:
        """Group by identity the window sizes that would actually load.

        A cache only counts when it was written by the current feature version and
        holds per-frame features: ``IdentityFeatures`` discards a cache whose
        version differs and recomputes it, and window features are recomputed
        along with the per-frame features they accompany.

        Sizes are only merged across an identity's pose-hash subdirectories from
        caches that pass those checks, so a window size present only in a stale
        directory does not count because a sibling directory happens to be current.
        """
        grouped: dict[int, set[int]] = {}
        for info in self.identity_caches:
            loadable = (
                info.per_frame_present and info.feature_version == self.current_feature_version
            )
            grouped.setdefault(info.identity, set()).update(
                info.window_sizes if loadable else frozenset()
            )
        return {identity: frozenset(sizes) for identity, sizes in grouped.items()}

    @property
    def has_cached_features(self) -> bool:
        """Whether any identity has cached features."""
        return bool(self.identity_caches)

    @property
    def cached_identity_count(self) -> int:
        """Number of distinct identities with a cache present."""
        return len({info.identity for info in self.identity_caches})

    @property
    def identities_missing_per_frame(self) -> tuple[int, ...]:
        """Sorted identities whose caches hold no per-frame features.

        Evaluated per identity rather than per cache directory: an identity with
        caches under several pose-hash subdirectories only needs per-frame
        features in one of them, matching how :attr:`window_sizes` merges.
        """
        present: dict[int, bool] = {}
        for info in self.identity_caches:
            present[info.identity] = present.get(info.identity, False) or info.per_frame_present
        return tuple(sorted(identity for identity, found in present.items() if not found))

    @property
    def is_complete(self) -> bool:
        """Whether every identity in the video has per-frame features cached.

        ``False`` when the identity count is unknown, so callers can treat this
        as "known to be complete" rather than "not known to be incomplete".
        """
        if self.expected_identity_count is None or not self.has_cached_features:
            return False
        return (
            self.cached_identity_count >= self.expected_identity_count
            and not self.identities_missing_per_frame
        )

    @property
    def window_sizes(self) -> tuple[int, ...]:
        """Sorted window sizes cached for every identity that has a cache.

        These are the window sizes that can be loaded from the cache for the
        whole video. Note the qualifier: when only some identities are cached
        (see :attr:`is_complete`), this reflects only those identities.
        """
        sets = list(self._window_sizes_by_identity().values())
        if not sets:
            return ()
        return tuple(sorted(frozenset.intersection(*sets)))

    def has_window_features(
        self, window_size: int, identities: Collection[int] | None = None
    ) -> bool:
        """Whether cached features cover a window size for the given identities.

        Used to decide whether features would have to be computed on demand, so
        only caches that would actually load count: a stale one, or one missing its
        per-frame features, is recomputed on use and is therefore not coverage.

        Two causes of recomputation cannot be detected from a status scan, so this
        can still answer ``True`` for a cache that will be rebuilt: a pose file that
        changed since the cache was written (detecting it would mean hashing every
        pose file), and a distance-unit setting that no longer matches the cache
        (compare :attr:`cm_units` against the run's settings to catch that).

        Args:
            window_size: Window size the features are needed for.
            identities: Identities that need features. When ``None``, every
                identity in the video is required, which means ``False`` is
                returned if the identity count is unknown. An empty collection
                means nothing is needed, so the answer is ``True``.

        Returns:
            True when every required identity has this window size cached in a
            form that can be loaded.
        """
        cached = self._loadable_window_sizes_by_identity()
        if identities is None:
            if self.expected_identity_count is None:
                return False
            identities = range(self.expected_identity_count)
        return all(window_size in cached.get(identity, frozenset()) for identity in identities)

    @property
    def partial_window_sizes(self) -> tuple[int, ...]:
        """Sorted window sizes cached for some, but not all, cached identities."""
        sets = list(self._window_sizes_by_identity().values())
        if not sets:
            return ()
        return tuple(sorted(frozenset.union(*sets) - frozenset.intersection(*sets)))

    @property
    def cache_formats(self) -> tuple[CacheFormat, ...]:
        """Distinct storage formats found, sorted by value.

        More than one format means the cache is mixed, which can happen when a
        project's cache format setting changed without clearing the cache.
        """
        return tuple(
            sorted(
                {info.cache_format for info in self.identity_caches},
                key=lambda cache_format: cache_format.value,
            )
        )

    @property
    def feature_versions(self) -> tuple[int, ...]:
        """Sorted distinct feature versions the caches were written with."""
        return tuple(sorted({info.feature_version for info in self.identity_caches}))

    @property
    def is_stale(self) -> bool:
        """Whether any cache was written by a different feature version.

        A stale cache is discarded and recomputed the next time features are
        needed.
        """
        return any(version != self.current_feature_version for version in self.feature_versions)

    @property
    def cm_units(self) -> bool | None:
        """Whether the caches were computed in cm units.

        Returns:
            ``True`` when every cache carries a distance scale factor, ``False``
            when none do, and ``None`` when the caches disagree or there are
            none. A mismatch with the project's current unit setting causes the
            cache to be recomputed.
        """
        if not self.identity_caches:
            return None
        scaled = {info.distance_scale_factor is not None for info in self.identity_caches}
        if len(scaled) != 1:
            return None
        return scaled.pop()

    @property
    def size_bytes(self) -> int:
        """Total size on disk of every cache file found for this video."""
        return sum(info.size_bytes for info in self.identity_caches)


def _iter_identity_dirs(video_cache_dir: Path) -> Iterator[Path]:
    """Yield the per-identity cache directories for one video.

    Handles both on-disk layouts: the flat ``features/<video>/<identity>`` used
    by the GUI, and ``features/<video>/<pose hash>/<identity>`` written when the
    CLI tools are run with ``--use-pose-hash``.

    Args:
        video_cache_dir: The ``features/<video stem>`` directory to walk.

    Yields:
        Directories whose name is an identity index, in name order.
    """
    try:
        children = sorted(video_cache_dir.iterdir())
    except OSError:
        logger.debug("Could not list feature cache directory %s", video_cache_dir, exc_info=True)
        return

    for child in children:
        if not child.is_dir():
            continue
        if child.name.isdigit():
            yield child
            continue
        # a pose-hash subdirectory: its own children are the identity directories
        try:
            grandchildren = sorted(child.iterdir())
        except OSError:
            logger.debug("Could not list feature cache directory %s", child, exc_info=True)
            continue
        for grandchild in grandchildren:
            if grandchild.is_dir() and grandchild.name.isdigit():
                yield grandchild


def scan_video_feature_cache(
    feature_dir: Path,
    video: str,
    current_feature_version: int,
    expected_identity_count: int | None = None,
) -> VideoFeatureCacheStatus:
    """Inspect the cached features for one video.

    Args:
        feature_dir: The project's features directory.
        video: Video filename (or pose filename) to scan the cache for.
        current_feature_version: The running application's ``FEATURE_VERSION``.
        expected_identity_count: Number of identities in the video, when known.

    Returns:
        The video's cache status. A video with no cache yields a status whose
        ``has_cached_features`` is ``False``; missing directories are not an
        error.
    """
    cache_dir = feature_dir / pose_file_stem(video)
    caches: list[IdentityCacheInfo] = []
    if cache_dir.is_dir():
        for identity_dir in _iter_identity_dirs(cache_dir):
            info = inspect_identity_cache(identity_dir)
            if info is not None:
                caches.append(info)

    return VideoFeatureCacheStatus(
        video=video,
        cache_dir=cache_dir,
        identity_caches=tuple(sorted(caches, key=lambda info: (info.identity, info.directory))),
        current_feature_version=current_feature_version,
        expected_identity_count=expected_identity_count,
    )


def scan_project_video_feature_cache(project: Project, video: str) -> VideoFeatureCacheStatus:
    """Inspect the cached features for one video of a project.

    Convenience wrapper that supplies the project's features directory, the
    running ``FEATURE_VERSION``, and the video's identity count.

    Args:
        project: Project the video belongs to.
        video: Video filename.

    Returns:
        The video's cache status.
    """
    # imported here rather than at module scope: jabs.feature_extraction imports
    # jabs.project, so a module-level import would be circular.
    from jabs.feature_extraction import FEATURE_VERSION

    try:
        identity_count = project.video_manager.get_video_identity_count(video)
    except (KeyError, ValueError):
        logger.debug("Identity count unavailable for %s", video, exc_info=True)
        identity_count = None

    return scan_video_feature_cache(
        project.feature_dir,
        video,
        current_feature_version=FEATURE_VERSION,
        expected_identity_count=identity_count,
    )


def scan_project_feature_cache(
    project: Project, should_continue: Callable[[], bool] | None = None
) -> dict[str, VideoFeatureCacheStatus]:
    """Inspect the cached features for every video in a project.

    This only reads cache metadata, but it touches every per-identity cache
    directory in the project, so callers in the GUI should run it off the main
    thread.

    Args:
        project: Project to scan.
        should_continue: Optional predicate checked before each video; the scan
            stops early and returns partial results when it returns ``False``.
            Lets a caller running the scan in a thread abandon it, for example at
            application shutdown.

    Returns:
        Mapping of video filename to its cache status. Every video in the project
        is present, including those with no cached features, unless the scan
        stopped early.
    """
    statuses: dict[str, VideoFeatureCacheStatus] = {}
    videos = list(project.video_manager.videos)
    for video in videos:
        if should_continue is not None and not should_continue():
            logger.debug(
                "Feature cache scan stopped early after %d of %d videos",
                len(statuses),
                len(videos),
            )
            return statuses
        statuses[video] = scan_project_video_feature_cache(project, video)

    cached = sum(1 for status in statuses.values() if status.has_cached_features)
    logger.info(
        "Feature cache scan complete: %d of %d videos have cached features",
        cached,
        len(statuses),
    )
    return statuses
