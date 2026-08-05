"""Human-readable text for feature cache status.

Used by the video info dialog's feature cache section. Deliberately free of Qt
imports: these are pure string helpers.
"""

from __future__ import annotations

from collections.abc import Sequence

from jabs.core.enums import CacheFormat

_SIZE_UNITS = ("bytes", "KiB", "MiB", "GiB", "TiB")

_CACHE_FORMAT_NAMES = {CacheFormat.HDF5: "HDF5", CacheFormat.PARQUET: "Parquet"}


def format_byte_size(num_bytes: int) -> str:
    """Format a byte count for display.

    Args:
        num_bytes: Size in bytes.

    Returns:
        The size scaled to the largest unit that leaves a value of at least one,
        for example ``"842 bytes"``, ``"12.4 MiB"``.
    """
    size = float(num_bytes)
    for unit in _SIZE_UNITS:
        if size < 1024 or unit == _SIZE_UNITS[-1]:
            if unit == "bytes":
                return f"{int(size)} bytes"
            return f"{size:.1f} {unit}"
        size /= 1024
    # unreachable: the loop always returns on the last unit
    raise AssertionError  # pragma: no cover


def format_window_sizes(sizes: Sequence[int]) -> str:
    """Format a collection of window sizes as a comma-separated list.

    Args:
        sizes: Window sizes, in the order they should be displayed.

    Returns:
        The sizes joined by commas, or ``"none"`` when empty.
    """
    return ", ".join(str(size) for size in sizes) if sizes else "none"


def format_cache_formats(formats: Sequence[CacheFormat]) -> str:
    """Format the storage formats found in a cache for display.

    Args:
        formats: Distinct cache formats found.

    Returns:
        The format's display name, or ``"Mixed (HDF5, Parquet)"`` style text when
        more than one format is present. Empty input yields ``"unknown"``.
    """
    names = [_CACHE_FORMAT_NAMES.get(fmt, str(fmt)) for fmt in formats]
    if not names:
        return "unknown"
    if len(names) == 1:
        return names[0]
    return f"Mixed ({', '.join(names)})"
