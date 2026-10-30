"""Tests for the shared brightness/contrast controls of the player's ControlOverlay."""

from __future__ import annotations

import pytest

try:
    from PySide6 import QtCore, QtGui, QtWidgets
    from PySide6.QtWidgets import QApplication

    from jabs.ui.player_widget.overlays.control_overlay import ControlOverlay, _ImageAdjustment

    _BRIGHTNESS_BADGE = QtCore.QRect(0, 0, 10, 10)
    _CONTRAST_BADGE = QtCore.QRect(20, 0, 10, 10)

    class _StubFrameWidget(QtWidgets.QWidget):
        """Minimal stand-in for FrameWithOverlaysWidget.

        ControlOverlay only needs a parent widget that reports a non-null pixmap
        and the geometry of the scaled frame inside it.
        """

        def __init__(self) -> None:
            super().__init__()
            self._pixmap = QtGui.QPixmap(100, 100)
            self._pixmap.fill(QtGui.QColor("black"))
            self.scaled_pix_x = 0
            self.scaled_pix_y = 0
            self.scaled_pix_width = 100
            self.scaled_pix_height = 100
            self.is_cropped = False

        def pixmap(self) -> QtGui.QPixmap:
            """Return the stub frame pixmap."""
            return self._pixmap

    SKIP_UI_TESTS = False
    SKIP_REASON = ""
except ImportError as e:
    SKIP_UI_TESTS = True
    SKIP_REASON = f"Qt/UI dependencies not available: {e}"
    _BRIGHTNESS_BADGE = _CONTRAST_BADGE = None

pytestmark = pytest.mark.skipif(SKIP_UI_TESTS, reason=SKIP_REASON)


@pytest.fixture(scope="module", autouse=True)
def qapp():
    """Ensure a QApplication exists for widget tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def overlay() -> ControlOverlay:
    """A ControlOverlay whose adjustment badges have known positions.

    The badge rectangles are normally set while painting; the tests set them
    directly so a click can be aimed at one without rendering a frame.
    """
    control_overlay = ControlOverlay(_StubFrameWidget())
    control_overlay._brightness.badge = QtCore.QRect(_BRIGHTNESS_BADGE)
    control_overlay._contrast.badge = QtCore.QRect(_CONTRAST_BADGE)
    return control_overlay


def _click(overlay: ControlOverlay, badge: QtCore.QRect) -> bool:
    """Send a left-button press at the center of ``badge``."""
    center = badge.center()
    event = QtGui.QMouseEvent(
        QtCore.QEvent.Type.MouseButtonPress,
        QtCore.QPointF(center),
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    return overlay.handle_mouse_press(event)


def _reset_button(adjustment: _ImageAdjustment) -> QtWidgets.QPushButton:
    """Return the reset button of an open adjustment slider."""
    return adjustment.container.findChildren(QtWidgets.QPushButton)[0]


@pytest.mark.parametrize(
    ("attribute", "badge", "name", "expected_range"),
    [
        ("_brightness", _BRIGHTNESS_BADGE, "brightness", (50, 200)),
        ("_contrast", _CONTRAST_BADGE, "contrast", (100, 200)),
    ],
    ids=["brightness", "contrast"],
)
def test_badge_click_toggles_its_own_slider(
    overlay: ControlOverlay,
    attribute: str,
    badge: QtCore.QRect,
    name: str,
    expected_range: tuple[int, int],
) -> None:
    """Clicking a badge opens that adjustment's slider, and clicking again closes it."""
    adjustment = getattr(overlay, attribute)

    assert _click(overlay, badge) is True
    assert adjustment.slider_open is True
    assert (adjustment.slider.minimum(), adjustment.slider.maximum()) == expected_range
    assert adjustment.slider.value() == _ImageAdjustment.NEUTRAL_VALUE
    # the stylesheet rule selects on the container's object name, so they must agree
    assert adjustment.container.objectName() == f"{name}SliderContainer"
    assert f"#{name}SliderContainer" in adjustment.container.styleSheet()

    assert _click(overlay, badge) is True
    assert adjustment.slider_open is False
    assert adjustment.slider is None
    assert adjustment.container is None


@pytest.mark.parametrize(
    ("attribute", "badge", "signal_name"),
    [
        ("_brightness", _BRIGHTNESS_BADGE, "brightness_changed"),
        ("_contrast", _CONTRAST_BADGE, "contrast_changed"),
    ],
    ids=["brightness", "contrast"],
)
def test_slider_emits_multiplier_and_remembers_value(
    overlay: ControlOverlay, attribute: str, badge: QtCore.QRect, signal_name: str
) -> None:
    """Moving a slider emits its own signal as a multiplier and the value is restored."""
    adjustment = getattr(overlay, attribute)
    emitted: list[float] = []
    getattr(overlay, signal_name).connect(emitted.append)

    _click(overlay, badge)
    adjustment.slider.setValue(150)

    assert emitted == [1.5]
    assert adjustment.value == 150

    # closing and reopening restores the last value rather than the neutral one
    _click(overlay, badge)
    _click(overlay, badge)
    assert adjustment.slider.value() == 150


@pytest.mark.parametrize(
    ("attribute", "badge"),
    [("_brightness", _BRIGHTNESS_BADGE), ("_contrast", _CONTRAST_BADGE)],
    ids=["brightness", "contrast"],
)
def test_reset_button_restores_neutral_value(
    overlay: ControlOverlay, attribute: str, badge: QtCore.QRect
) -> None:
    """The reset button above a slider returns it to the neutral value."""
    adjustment = getattr(overlay, attribute)

    _click(overlay, badge)
    adjustment.slider.setValue(175)
    _reset_button(adjustment).click()

    assert adjustment.slider.value() == _ImageAdjustment.NEUTRAL_VALUE
    assert adjustment.value == _ImageAdjustment.NEUTRAL_VALUE


def test_opening_one_slider_closes_the_other(overlay: ControlOverlay) -> None:
    """Only one adjustment slider is open at a time."""
    _click(overlay, _BRIGHTNESS_BADGE)
    assert overlay._brightness.slider_open is True

    _click(overlay, _CONTRAST_BADGE)

    assert overlay._brightness.slider_open is False
    assert overlay._brightness.slider is None
    assert overlay._contrast.slider_open is True


def test_leaving_the_widget_closes_every_slider(overlay: ControlOverlay) -> None:
    """Leaving the frame hides all adjustment sliders."""
    _click(overlay, _BRIGHTNESS_BADGE)

    overlay.handle_leave(QtCore.QEvent(QtCore.QEvent.Type.Leave))

    for adjustment in overlay._adjustments:
        assert adjustment.slider_open is False
        assert adjustment.slider is None
        assert adjustment.container is None
