"""Tests for feature cache status text formatting."""

import pytest

from jabs.core.enums import CacheFormat
from jabs.ui.feature_cache_text import (
    format_byte_size,
    format_cache_formats,
    format_window_sizes,
)


@pytest.mark.parametrize(
    ("num_bytes", "expected"),
    [
        (0, "0 bytes"),
        (512, "512 bytes"),
        (2048, "2.0 KiB"),
        (1024 * 1024 + 512 * 1024, "1.5 MiB"),
        (3 * 1024**3, "3.0 GiB"),
        (2 * 1024**4, "2.0 TiB"),
        (1024**5, "1024.0 TiB"),
    ],
    ids=["zero", "bytes", "kib", "mib", "gib", "tib", "beyond-tib"],
)
def test_format_byte_size(num_bytes, expected):
    """Byte counts are scaled to the largest unit that leaves a value >= 1."""
    assert format_byte_size(num_bytes) == expected


def test_format_window_sizes_lists_sizes():
    """Window sizes are joined with commas in the order given."""
    assert format_window_sizes((5, 10, 30)) == "5, 10, 30"


def test_format_window_sizes_empty():
    """An empty window size collection reads as 'none'."""
    assert format_window_sizes(()) == "none"


def test_format_cache_formats_single():
    """A single format is named directly."""
    assert format_cache_formats((CacheFormat.PARQUET,)) == "Parquet"


def test_format_cache_formats_mixed():
    """Multiple formats are reported as mixed."""
    assert format_cache_formats((CacheFormat.HDF5, CacheFormat.PARQUET)) == "Mixed (HDF5, Parquet)"


def test_format_cache_formats_empty():
    """No formats reads as unknown."""
    assert format_cache_formats(()) == "unknown"
