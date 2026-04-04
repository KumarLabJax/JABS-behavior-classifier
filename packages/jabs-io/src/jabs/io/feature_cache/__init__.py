"""Feature cache I/O backends for jabs-io."""

from __future__ import annotations

from pathlib import Path

from jabs.core.enums import CacheFormat


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
    """
    if (identity_dir / "metadata.json").exists():
        return CacheFormat.PARQUET
    if (identity_dir / "features.h5").exists():
        return CacheFormat.HDF5
    return None
