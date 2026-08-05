"""Read-only inspection of an on-disk feature cache.

Describes what a per-identity cache directory contains (storage format, the
metadata it was written with, and which window sizes are available) without
loading any feature data.

Inspection deliberately performs none of the validation the readers in this
package do: it reports what is on disk and leaves interpretation (is this
feature version current? does this pose hash still match the pose file?) to the
caller. This makes it safe to use for cheap, bulk status reporting such as
scanning an entire project.

Note:
    Like :func:`~jabs.io.feature_cache.detect_cache_format`, the on-disk
    filenames are written as literals here rather than imported from the
    backend modules. They are stable on-disk format contracts; changing one
    constitutes a format-breaking change requiring a versioned migration, so
    drift is not a practical risk.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import h5py

from jabs.core.enums import CacheFormat

logger = logging.getLogger(__name__)

_HDF5_FILENAME = "features.h5"
_HDF5_PER_FRAME_GROUP = "features/per_frame"
_HDF5_FEATURES_GROUP = "features"
_HDF5_WINDOW_GROUP_PREFIX = "window_features_"

_PARQUET_METADATA_FILENAME = "metadata.json"
_PARQUET_PER_FRAME_FILENAME = "per_frame.parquet"
_PARQUET_WINDOW_FILENAME_TEMPLATE = "window_{size}.parquet"


@dataclass(frozen=True)
class IdentityCacheInfo:
    """Summary of the feature cache stored in one per-identity directory.

    Attributes:
        directory: The per-identity cache directory that was inspected.
        identity: Identity index recorded in the cache metadata.
        cache_format: Storage format the cache was written in.
        feature_version: Value of ``FEATURE_VERSION`` at the time the cache was
            written. Compare against the current value to detect a stale cache.
        pose_hash: Hash of the pose file the features were computed from.
        num_frames: Frame count recorded in the cache.
        distance_scale_factor: Pixels-to-cm scale factor the cache was computed
            with, or ``None`` when the cache does not use cm units.
        window_sizes: Window sizes whose features are present on disk and
            loadable.
        per_frame_present: Whether per-frame features are present. A cache
            missing them is incomplete and will be recomputed on next use.
        size_bytes: Total size on disk of all files in the cache directory.
    """

    directory: Path
    identity: int
    cache_format: CacheFormat
    feature_version: int
    pose_hash: str
    num_frames: int
    distance_scale_factor: float | None
    window_sizes: frozenset[int]
    per_frame_present: bool
    size_bytes: int


def inspect_identity_cache(identity_dir: Path) -> IdentityCacheInfo | None:
    """Inspect one per-identity cache directory.

    Checks for the Parquet sentinel (``metadata.json``) first, then the HDF5
    cache file, matching the precedence used by
    :func:`~jabs.io.feature_cache.detect_cache_format`.

    Args:
        identity_dir: Per-identity cache directory to inspect.

    Returns:
        An :class:`IdentityCacheInfo` describing the cache, or ``None`` when no
        cache is present or its metadata cannot be read (for example a
        truncated file, or a cache being written concurrently). An unreadable
        cache is reported the same way as an absent one because both mean the
        features would have to be recomputed.
    """
    if (identity_dir / _PARQUET_METADATA_FILENAME).exists():
        return _inspect_parquet_cache(identity_dir)
    if (identity_dir / _HDF5_FILENAME).exists():
        return _inspect_hdf5_cache(identity_dir)
    return None


def _directory_size(directory: Path) -> int:
    """Return the total size in bytes of the files directly inside a directory."""
    total = 0
    for path in directory.iterdir():
        if path.is_file():
            try:
                total += path.stat().st_size
            except OSError:
                logger.debug("Could not stat cache file %s", path, exc_info=True)
    return total


def _inspect_hdf5_cache(identity_dir: Path) -> IdentityCacheInfo | None:
    """Inspect an HDF5 (``features.h5``) cache directory."""
    path = identity_dir / _HDF5_FILENAME
    try:
        with h5py.File(path, "r") as f:
            window_sizes: set[int] = set()
            if _HDF5_FEATURES_GROUP in f:
                for key in f[_HDF5_FEATURES_GROUP]:
                    if key.startswith(_HDF5_WINDOW_GROUP_PREFIX):
                        suffix = key[len(_HDF5_WINDOW_GROUP_PREFIX) :]
                        if suffix.isdigit():
                            window_sizes.add(int(suffix))
            scale = f.attrs.get("distance_scale_factor", None)
            return IdentityCacheInfo(
                directory=identity_dir,
                identity=int(f.attrs["identity"]),
                cache_format=CacheFormat.HDF5,
                feature_version=int(f.attrs["version"]),
                pose_hash=str(f.attrs["pose_hash"]),
                num_frames=int(f.attrs["num_frames"]),
                distance_scale_factor=float(scale) if scale is not None else None,
                window_sizes=frozenset(window_sizes),
                per_frame_present=_HDF5_PER_FRAME_GROUP in f,
                size_bytes=_directory_size(identity_dir),
            )
    except (OSError, KeyError, TypeError, ValueError):
        logger.debug("Could not inspect HDF5 feature cache at %s", path, exc_info=True)
        return None


def _inspect_parquet_cache(identity_dir: Path) -> IdentityCacheInfo | None:
    """Inspect a Parquet cache directory.

    Only window sizes that are both registered in ``metadata.json`` and backed
    by an existing ``window_{size}.parquet`` file are reported, so the result
    reflects what could actually be loaded. A registered size whose file is
    missing (for example an interrupted write) is omitted.
    """
    path = identity_dir / _PARQUET_METADATA_FILENAME
    try:
        with path.open(encoding="utf-8") as f:
            raw: dict = json.load(f)

        window_sizes = {
            size
            for size in (int(s) for s in raw["cached_window_sizes"])
            if (identity_dir / _PARQUET_WINDOW_FILENAME_TEMPLATE.format(size=size)).exists()
        }
        scale = raw["distance_scale_factor"]
        return IdentityCacheInfo(
            directory=identity_dir,
            identity=int(raw["identity"]),
            cache_format=CacheFormat.PARQUET,
            feature_version=int(raw["feature_version"]),
            pose_hash=str(raw["pose_hash"]),
            num_frames=int(raw["num_frames"]),
            distance_scale_factor=float(scale) if scale is not None else None,
            window_sizes=frozenset(window_sizes),
            per_frame_present=(identity_dir / _PARQUET_PER_FRAME_FILENAME).exists(),
            size_bytes=_directory_size(identity_dir),
        )
    except (OSError, KeyError, TypeError, ValueError):
        logger.debug("Could not inspect Parquet feature cache at %s", path, exc_info=True)
        return None
