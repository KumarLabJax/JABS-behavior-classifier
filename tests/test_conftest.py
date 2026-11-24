"""The session-wide fixtures in ``tests/conftest.py``."""

from pathlib import Path

from platformdirs import user_cache_dir

from jabs.pose_estimation import conversion_cache


def test_per_user_conversion_cache_is_redirected() -> None:
    """Conversions without a cache_dir stay out of the real per-user cache."""
    assert conversion_cache.cache_root(None) != Path(user_cache_dir("jabs"))
