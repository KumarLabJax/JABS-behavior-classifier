"""Tests for the overlays that replaced the decoder's cv2 annotations."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

try:
    from PySide6 import QtCore
    from PySide6.QtWidgets import QApplication

    from jabs.ui.player_widget.frame_with_overlays import FrameWithOverlaysWidget
    from jabs.ui.player_widget.overlays import closest_identity_overlay as closest_module
    from jabs.ui.player_widget.overlays import landmark_overlay as landmark_module
    from jabs.ui.player_widget.overlays import track_overlay as track_module
    from jabs.ui.player_widget.overlays.closest_identity_overlay import ClosestIdentityOverlay
    from jabs.ui.player_widget.overlays.landmark_overlay import LandmarkOverlay
    from jabs.ui.player_widget.overlays.track_overlay import TrackOverlay

    _RECT = QtCore.QRect(0, 0, 800, 800)

    SKIP_UI_TESTS = False
    SKIP_REASON = ""
except ImportError as e:
    SKIP_UI_TESTS = True
    SKIP_REASON = f"Qt/UI dependencies not available: {e}"

pytestmark = pytest.mark.skipif(SKIP_UI_TESTS, reason=SKIP_REASON)


@pytest.fixture(scope="module", autouse=True)
def qapp():
    """Ensure a QApplication exists for widget tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def _overlay(cls, enabled: bool, active_identity: int | None = 1, frame: int = 7):
    """Build one overlay on a stand-in frame widget, bypassing Qt parenting."""
    parent = SimpleNamespace(
        pixmap=lambda: SimpleNamespace(isNull=lambda: False),
        pose=SimpleNamespace(identities=[0, 1, 2]),
        current_frame=frame,
        active_identity=active_identity,
        scaled_pix_width=800,
        image_to_widget_coords_cropped=lambda x, y, rect: (int(x), int(y)),
    )
    overlay = cls.__new__(cls)
    overlay._parent = parent
    overlay._priority = 0
    overlay._enabled = enabled
    if cls is ClosestIdentityOverlay:
        overlay._cache_key = None
        overlay._cached = (None, None)
    return overlay


@pytest.mark.parametrize(
    "overlay_name",
    ["track", "landmarks"],
)
def test_disabled_overlays_draw_nothing(monkeypatch, overlay_name: str):
    """Each of these is opt-in and must not touch the painter until asked for.

    The overlay classes are resolved here rather than in the parametrize list, which
    is evaluated at import time and so cannot name anything that needs Qt.
    """
    cases = {
        "track": (TrackOverlay, track_module, "draw_identity_track"),
        "landmarks": (LandmarkOverlay, landmark_module, "draw_landmarks"),
    }
    cls, module, function = cases[overlay_name]
    draw = MagicMock()
    monkeypatch.setattr(module, function, draw)

    _overlay(cls, enabled=False).paint(MagicMock(), _RECT)

    draw.assert_not_called()


def test_the_track_draws_only_the_active_identity(monkeypatch):
    """The track follows the animal being labeled; all of them would be unreadable."""
    draw = MagicMock()
    monkeypatch.setattr(track_module, "draw_identity_track", draw)

    _overlay(TrackOverlay, enabled=True, active_identity=2).paint(MagicMock(), _RECT)

    draw.assert_called_once()
    assert draw.call_args.args[3] == 2


def test_the_track_is_skipped_without_an_active_identity(monkeypatch):
    """Nothing is selected, so there is no track to follow."""
    draw = MagicMock()
    monkeypatch.setattr(track_module, "draw_identity_track", draw)

    _overlay(TrackOverlay, enabled=True, active_identity=None).paint(MagicMock(), _RECT)

    draw.assert_not_called()


def test_markers_are_measured_once_per_frame(monkeypatch):
    """A repaint that changes neither the frame nor the selection must not re-measure.

    The measurement walks every identity's hull and view angle, and repaints happen
    for reasons that have nothing to do with it - a resize, a crop, another overlay
    being toggled.
    """
    calls: list[tuple] = []

    def fake_closest(pose, subject, frame, half_fov=None):
        calls.append((subject, frame, half_fov))
        return 2 if half_fov is None else 1

    monkeypatch.setattr(closest_module, "closest_identity", fake_closest)
    monkeypatch.setattr(closest_module, "draw_identity_marker", MagicMock())
    overlay = _overlay(ClosestIdentityOverlay, enabled=True)
    overlay.get_centroid = lambda identity: SimpleNamespace(x=10.0, y=20.0)

    overlay.paint(MagicMock(), _RECT)
    overlay.paint(MagicMock(), _RECT)

    assert len(calls) == 2, "one call per marker, then served from the cache"

    # A new frame has to be measured again.
    overlay.parent.current_frame = 8
    overlay.paint(MagicMock(), _RECT)

    assert len(calls) == 4


def test_both_markers_are_drawn_when_they_differ(monkeypatch):
    """One marker for the nearest animal in view, one for the nearest overall."""
    monkeypatch.setattr(
        closest_module,
        "closest_identity",
        lambda pose, subject, frame, half_fov=None: 1 if half_fov is None else 2,
    )
    draw = MagicMock()
    monkeypatch.setattr(closest_module, "draw_identity_marker", draw)
    overlay = _overlay(ClosestIdentityOverlay, enabled=True)
    overlay.get_centroid = lambda identity: SimpleNamespace(x=10.0, y=20.0)

    overlay.paint(MagicMock(), _RECT)

    assert draw.call_count == 2
    colors = [call.args[4] for call in draw.call_args_list]
    assert colors[0] != colors[1], "the two markers must be distinguishable"
    # In-view marker first, then nearest overall: the order the cv2 drawing used.
    assert colors == [closest_module.CLOSEST_FOV_MARKER_COLOR, closest_module.CLOSEST_MARKER_COLOR]


def test_one_marker_when_the_nearest_animal_is_also_the_nearest_in_view(monkeypatch):
    """The same animal is marked once, as the cv2 drawing did."""
    monkeypatch.setattr(
        closest_module, "closest_identity", lambda pose, subject, frame, half_fov=None: 2
    )
    draw = MagicMock()
    monkeypatch.setattr(closest_module, "draw_identity_marker", draw)
    overlay = _overlay(ClosestIdentityOverlay, enabled=True)
    overlay.get_centroid = lambda identity: SimpleNamespace(x=10.0, y=20.0)

    overlay.paint(MagicMock(), _RECT)

    assert draw.call_count == 1


@pytest.mark.parametrize(
    "flag",
    ["track_overlay_enabled", "closest_identity_overlay_enabled", "landmark_overlay_enabled"],
)
def test_each_overlay_starts_off_and_repaints_when_toggled(monkeypatch, flag: str):
    """These are all opt-in, and turning one on repaints rather than re-decoding."""
    widget = FrameWithOverlaysWidget()
    update = MagicMock()
    monkeypatch.setattr(widget, "update", update)

    assert getattr(widget, flag) is False

    setattr(widget, flag, True)

    assert getattr(widget, flag) is True
    update.assert_called_once_with()

    setattr(widget, flag, True)
    update.assert_called_once_with()


def test_selecting_an_identity_repaints_the_overlays(monkeypatch):
    """Several overlays draw the active identity differently, so they must repaint.

    That repaint used to come for free, because changing identity re-decoded the frame
    to redraw the track and the closest-animal markers into it.
    """
    widget = FrameWithOverlaysWidget()
    update = MagicMock()
    monkeypatch.setattr(widget, "update", update)

    widget.set_active_identity(3)

    assert widget.active_identity == 3
    update.assert_called_once_with()

    widget.set_active_identity(3)
    update.assert_called_once_with()
