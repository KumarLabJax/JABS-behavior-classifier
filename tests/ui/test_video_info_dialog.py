"""Tests for the feature cache section of the video info dialog."""

from pathlib import Path

import pytest

from jabs.core.enums import CacheFormat
from jabs.io.feature_cache import IdentityCacheInfo
from jabs.project import VideoFeatureCacheStatus

try:
    from PySide6.QtWidgets import QApplication, QLabel

    from jabs.ui.dialogs.video_info_dialog import VideoInfoDialog

    SKIP_UI_TESTS = False
    SKIP_REASON = None
except ImportError as e:
    SKIP_UI_TESTS = True
    SKIP_REASON = f"Qt/UI dependencies not available: {e}"

pytestmark = pytest.mark.skipif(
    SKIP_UI_TESTS,
    reason=SKIP_REASON if SKIP_UI_TESTS else "",
)

_CURRENT_VERSION = 17

# The dialog reads the video and pose files directly; pointing it at paths that
# do not exist exercises its existing error handling and leaves only the feature
# cache section (built from the status object) under test.
_VIDEO_PATH = Path("/does/not/exist/video1.mp4")
_POSE_PATH = Path("/does/not/exist/video1_pose_est_v6.h5")


@pytest.fixture(scope="module", autouse=True)
def qapp():
    """Ensure a QApplication exists for widget tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def _status(
    *,
    identities: int = 2,
    window_sizes: tuple[frozenset[int], ...] | None = None,
    feature_version: int = _CURRENT_VERSION,
    cache_format: CacheFormat = CacheFormat.PARQUET,
    per_frame_present: bool = True,
    distance_scale_factor: float | None = None,
    expected_identity_count: int | None = 2,
) -> VideoFeatureCacheStatus:
    """Build a status with one cache per identity.

    Args:
        identities: Number of cached identities.
        window_sizes: Per-identity window size sets; defaults to ``{5, 10}`` for
            every identity.
        feature_version: Feature version recorded in each cache.
        cache_format: Storage format recorded in each cache.
        per_frame_present: Whether per-frame features are present.
        distance_scale_factor: Distance scale recorded in each cache.
        expected_identity_count: Identity count of the video, or ``None``.
    """
    sizes = window_sizes or tuple(frozenset({5, 10}) for _ in range(identities))
    caches = tuple(
        IdentityCacheInfo(
            directory=Path("/features/video1") / str(i),
            identity=i,
            cache_format=cache_format,
            feature_version=feature_version,
            pose_hash="hash",
            num_frames=100,
            distance_scale_factor=distance_scale_factor,
            window_sizes=sizes[i],
            per_frame_present=per_frame_present,
            size_bytes=2048,
        )
        for i in range(identities)
    )
    return VideoFeatureCacheStatus(
        video="video1.mp4",
        cache_dir=Path("/features/video1"),
        identity_caches=caches,
        current_feature_version=_CURRENT_VERSION,
        expected_identity_count=expected_identity_count,
    )


def _label_texts(dialog) -> list[str]:
    """Return the text of every label in the dialog."""
    return [label.text() for label in dialog.findChildren(QLabel)]


def _value_for(dialog, label_text: str) -> str:
    """Return the text of the field value following a form label.

    Args:
        dialog: Dialog to search.
        label_text: The form row label, including its trailing colon.

    Returns:
        The value label's text.
    """
    texts = _label_texts(dialog)
    assert label_text in texts, f"{label_text!r} not in {texts!r}"
    return texts[texts.index(label_text) + 1]


def _dialog(status):
    """Build a VideoInfoDialog for the given feature cache status."""
    return VideoInfoDialog(_VIDEO_PATH, _POSE_PATH, identity_count=2, feature_cache_status=status)


def test_dialog_shows_feature_cache_section():
    """The dialog reports the cache directory, coverage, window sizes and size."""
    dialog = _dialog(_status())

    assert "<b>Feature Cache</b>" in _label_texts(dialog)
    assert _value_for(dialog, "Directory:") == "/features/video1"
    assert _value_for(dialog, "Identities cached:") == "2 of 2"
    assert _value_for(dialog, "Window sizes:") == "5, 10"
    assert _value_for(dialog, "Format:") == "Parquet"
    assert _value_for(dialog, "Feature version:") == str(_CURRENT_VERSION)
    assert _value_for(dialog, "Size on disk:") == "4.0 KiB"


def test_dialog_reports_no_cached_features():
    """A video with no cached features shows the directory and 'None'."""
    dialog = _dialog(_status(identities=0))

    assert _value_for(dialog, "Cached features:") == "None"
    assert "Window sizes:" not in _label_texts(dialog)


def test_dialog_reports_unknown_status():
    """Without a status the section says the cache could not be determined."""
    dialog = _dialog(None)

    assert _value_for(dialog, "Status:") == "Unable to determine"


def test_dialog_flags_stale_feature_version():
    """A cache from an older feature version is flagged with the current one."""
    dialog = _dialog(_status(feature_version=_CURRENT_VERSION - 1))

    version_text = _value_for(dialog, "Feature version:")
    assert version_text.startswith(str(_CURRENT_VERSION - 1))
    assert f"current is {_CURRENT_VERSION}" in version_text


def test_dialog_reports_partial_window_sizes():
    """Window sizes cached for only some identities are listed separately."""
    dialog = _dialog(_status(window_sizes=(frozenset({5, 10}), frozenset({5}))))

    assert _value_for(dialog, "Window sizes:") == "5"
    assert _value_for(dialog, "Partial window sizes:").startswith("10")


def test_dialog_reports_incomplete_identity_coverage():
    """A cache missing an identity is marked incomplete."""
    dialog = _dialog(_status(identities=1, expected_identity_count=3))

    assert _value_for(dialog, "Identities cached:") == "1 of 3 (incomplete)"


def test_dialog_warns_about_missing_per_frame_features():
    """A cache without per-frame features shows a warning row."""
    dialog = _dialog(_status(per_frame_present=False))

    assert "Per-frame features are missing" in _value_for(dialog, "Warning:")


@pytest.mark.parametrize(
    ("scale", "expected"), [(0.05, "cm"), (None, "pixels")], ids=["cm", "pixels"]
)
def test_dialog_reports_distance_units(scale, expected):
    """The units the cache was computed with are reported."""
    dialog = _dialog(_status(distance_scale_factor=scale))

    assert _value_for(dialog, "Distance units:") == expected
