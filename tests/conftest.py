"""Shared test configuration for the application test suite."""

import os
from collections.abc import Generator, Iterator

import pytest

from jabs.pose_estimation import conversion_cache

# Render Qt widgets offscreen so UI tests don't open windows and steal focus. Set before
# any test module creates a QApplication; override with e.g. QT_QPA_PLATFORM=cocoa to
# watch the widgets.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Legacy *_pose_est_v<N>.h5 files open through the conversion cache as Pose for the
# whole session. Tests marked legacy_reader run with it unset (see below).
_FORCE_UNIFIED_ENV = "JABS_FORCE_UNIFIED"
os.environ[_FORCE_UNIFIED_ENV] = "1"

_LEGACY_READER_MARK = "legacy_reader"


@pytest.fixture(scope="session", autouse=True)
def _per_user_conversion_cache(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Keep conversions opened without a cache_dir out of the real per-user cache.

    Tests that patch ``conversion_cache.user_cache_dir`` themselves still win.
    """
    user_cache = tmp_path_factory.mktemp("user_cache")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(conversion_cache, "user_cache_dir", lambda _app: str(user_cache))
        yield


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item: pytest.Item) -> Generator[None, None, None]:
    """Unset the switch for a legacy_reader test before any of its fixtures run.

    A hook rather than a fixture, so module- and session-scoped fixtures the test
    sets up open legacy files with the legacy reader too.
    """
    if item.get_closest_marker(_LEGACY_READER_MARK) is not None:
        os.environ.pop(_FORCE_UNIFIED_ENV, None)
    yield


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item: pytest.Item) -> Generator[None, None, None]:
    """Turn the switch back on once a legacy_reader test is torn down."""
    yield
    os.environ[_FORCE_UNIFIED_ENV] = "1"
