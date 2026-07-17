"""Persistent cache of per-video pose attributes to skip the load-time pose scan.

The project-load scan opens every pose HDF5 file to read a handful of small,
intrinsic attributes (frame count, identity count, static objects, lixit
keypoint count, cm-per-pixel flag). Those values change only when the pose file
itself changes, so they are cached in ``jabs/cache/pose_attribute_cache.json``
keyed by video filename and gated by a cheap ``stat`` token of the pose file. A
subsequent load then rescans only the videos whose pose file is new or changed.

The cache is an optimization only: a missing, unreadable, or schema-mismatched
cache simply triggers a full rescan, and a failed write is logged and ignored.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Bump when the cached entry schema changes so stale caches are ignored wholesale.
SCHEMA_VERSION = 1


def pose_token(pose_path: Path) -> str:
    """Return a cheap change-detection token for a pose file.

    Uses a single ``stat`` (size and modification time) rather than opening or
    hashing the file, so computing the token stays far cheaper than the scan it
    guards.

    Args:
        pose_path: Path to the pose HDF5 file.

    Returns:
        A ``"<size>:<mtime_ns>"`` token string.
    """
    st = pose_path.stat()
    return f"{st.st_size}:{st.st_mtime_ns}"


def load(cache_path: Path | None) -> dict[str, dict]:
    """Load the per-video attribute map from the cache file.

    Args:
        cache_path: Path to the cache JSON file, or ``None`` when caching is
            disabled (``use_cache=False``).

    Returns:
        Mapping of video filename to its cached entry. Returns an empty mapping
        when caching is disabled, or the file is missing, unreadable, or written
        by a different schema version.
    """
    if cache_path is None or not cache_path.exists():
        return {}
    try:
        with cache_path.open("r") as f:
            data = json.load(f)
    except (OSError, ValueError):
        logger.warning("Could not read pose attribute cache %s; rescanning", cache_path)
        return {}
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        return {}
    videos = data.get("videos")
    return videos if isinstance(videos, dict) else {}


def save(cache_path: Path | None, videos: dict[str, dict]) -> None:
    """Atomically write the per-video attribute map to the cache file.

    A write failure is logged and swallowed: the cache is an optimization, so
    failing to persist it must never break project loading.

    Args:
        cache_path: Path to the cache JSON file, or ``None`` when caching is
            disabled (in which case this is a no-op).
        videos: Mapping of video filename to its cached entry.
    """
    if cache_path is None:
        return
    payload = {"schema_version": SCHEMA_VERSION, "videos": videos}
    tmp = cache_path.with_suffix(".json.tmp")
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        tmp.replace(cache_path)
    except OSError:
        logger.warning("Could not write pose attribute cache %s", cache_path, exc_info=True)
