"""Feature cache I/O backends for jabs-io."""

from __future__ import annotations

import logging
from pathlib import Path

from jabs.core.enums import CacheFormat

logger = logging.getLogger(__name__)


def detect_cache_format(identity_dir: Path) -> CacheFormat | None:
    """Return the storage format of an existing cache, or ``None`` if absent.

    Checks for ``metadata.json`` first (the Parquet sentinel file), then falls
    back to ``features.h5``. Returns ``None`` when neither file is present so
    the caller knows to compute features from scratch.

    Args:
        identity_dir: Per-identity cache directory to inspect.

    Returns:
        ``CacheFormat.PARQUET`` if ``metadata.json`` exists,
        ``CacheFormat.HDF5`` if ``features.h5`` exists,
        ``None`` if no cache is present.

    Note:
        The sentinel filenames are written as literals here rather
        than imported from the backend modules. They are stable on-disk format
        contracts; changing either name constitutes a format-breaking change that
        requires a versioned migration, so drift is not a practical risk.
    """
    if (identity_dir / "metadata.json").exists():
        return CacheFormat.PARQUET
    if (identity_dir / "features.h5").exists():
        return CacheFormat.HDF5
    return None


def clear_cache(identity_dir: Path) -> None:
    """Delete all feature cache files from ``identity_dir``, regardless of format.

    Removes ``metadata.json``, ``per_frame.parquet``, ``window_*.parquet``, and
    ``features.h5`` if present. The directory itself is not removed.

    This is the correct way to discard a cache before writing in a different
    format, and is also used by the "Clear Feature Cache" GUI action.

    Args:
        identity_dir: Per-identity cache directory to clear.
    """
    for path in identity_dir.glob("*.parquet"):
        path.unlink(missing_ok=True)
    (identity_dir / "metadata.json").unlink(missing_ok=True)
    (identity_dir / "features.h5").unlink(missing_ok=True)
    logger.debug("Cleared feature cache in %s", identity_dir)
