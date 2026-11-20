"""The test session opens legacy pose files as ``Pose`` unless a test is ``legacy_reader``.

``tests/conftest.py`` sets ``JABS_FORCE_UNIFIED=1`` for the session and unsets it
for the setup, call and teardown of tests marked ``legacy_reader``, before any of
their fixtures (module-scoped ones included) are set up.
"""

import os
import shutil
from pathlib import Path

import pytest

from jabs.pose_estimation import Pose, PoseEstimation, conversion_cache, open_pose_file

_SAMPLE_V4 = Path(__file__).parent / "data" / "sample_pose_est_v4.h5"


@pytest.fixture(scope="module")
def module_env() -> str | None:
    """The switch as a module-scoped fixture sees it when it is set up."""
    return os.environ.get("JABS_FORCE_UNIFIED")


def test_switch_is_on_by_default() -> None:
    """An unmarked test runs with legacy files converted."""
    assert os.environ.get("JABS_FORCE_UNIFIED") == "1"


def test_unmarked_test_opens_legacy_file_as_pose(tmp_path: Path) -> None:
    """open_pose_file converts a legacy file in an unmarked test."""
    source = tmp_path / _SAMPLE_V4.name
    shutil.copyfile(_SAMPLE_V4, source)
    assert isinstance(open_pose_file(source, tmp_path / "cache"), Pose)


@pytest.mark.legacy_reader
def test_switch_is_off_for_legacy_reader_tests(module_env: str | None) -> None:
    """A legacy_reader test, and the module fixtures it sets up, run with the switch off."""
    assert os.environ.get("JABS_FORCE_UNIFIED") is None
    assert module_env is None


@pytest.mark.legacy_reader
def test_legacy_reader_test_opens_legacy_reader(tmp_path: Path) -> None:
    """open_pose_file uses the legacy reader in a legacy_reader test."""
    assert isinstance(open_pose_file(_SAMPLE_V4, tmp_path), PoseEstimation)


def test_switch_is_restored_after_a_legacy_reader_test() -> None:
    """The switch is back on for the next unmarked test."""
    assert os.environ.get("JABS_FORCE_UNIFIED") == "1"


def test_per_user_conversion_cache_is_redirected() -> None:
    """Conversions without a cache_dir stay out of the real per-user cache."""
    from platformdirs import user_cache_dir

    assert conversion_cache.cache_root(None) != Path(user_cache_dir("jabs"))
