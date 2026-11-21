"""Shared test configuration for the application test suite."""

import os
from collections.abc import Iterator

import pytest

from jabs.pose_estimation import conversion_cache

# Render Qt widgets offscreen so UI tests don't open windows and steal focus. Set before
# any test module creates a QApplication; override with e.g. QT_QPA_PLATFORM=cocoa to
# watch the widgets.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session", autouse=True)
def _per_user_conversion_cache(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Keep conversions opened without a cache_dir out of the real per-user cache.

    Tests that patch ``conversion_cache.user_cache_dir`` themselves still win.
    A ``JABS_CACHE_DIR`` set in the developer's environment would override the
    redirect, so it is cleared.
    """
    user_cache = tmp_path_factory.mktemp("user_cache")
    with pytest.MonkeyPatch.context() as patch:
        patch.delenv(conversion_cache.CACHE_DIR_ENV, raising=False)
        patch.setattr(conversion_cache, "user_cache_dir", lambda _app: str(user_cache))
        yield
