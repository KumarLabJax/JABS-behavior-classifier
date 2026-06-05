import pytest

try:
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import QApplication

    from jabs.ui.colors import BEHAVIOR_COLOR
    from jabs.ui.main_control_widget.main_control_widget import MainControlWidget

    SKIP_UI_TESTS = False
    SKIP_REASON = None
except ImportError as e:
    SKIP_UI_TESTS = True
    SKIP_REASON = f"Qt/UI dependencies not available: {e}"

pytestmark = pytest.mark.skipif(
    SKIP_UI_TESTS,
    reason=SKIP_REASON if SKIP_UI_TESTS else "",
)


@pytest.fixture(scope="module", autouse=True)
def qapp():
    """Ensure a QApplication exists for widget tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_default_label_button_color_is_orange() -> None:
    """A freshly built control widget uses the default orange behavior tint."""
    widget = MainControlWidget()

    style = widget._label_behavior_button.styleSheet()
    assert f"rgba{BEHAVIOR_COLOR.getRgb()}" in style
    assert "color: white" in style


def test_set_behavior_button_color_none_restores_default() -> None:
    """Passing None restores the default orange (binary-mode) tint."""
    widget = MainControlWidget()

    widget.set_behavior_button_color(QColor(10, 20, 30))
    widget.set_behavior_button_color(None)

    style = widget._label_behavior_button.styleSheet()
    assert f"rgba{BEHAVIOR_COLOR.getRgb()}" in style


def test_set_behavior_button_color_applies_behavior_color() -> None:
    """A behavior color tints the button gradient with that color."""
    widget = MainControlWidget()

    widget.set_behavior_button_color(QColor(10, 20, 30))

    style = widget._label_behavior_button.styleSheet()
    assert "rgba(10, 20, 30, 255)" in style


def test_set_behavior_button_color_picks_readable_text() -> None:
    """Text color adapts to the base color's luminance for readability."""
    widget = MainControlWidget()

    widget.set_behavior_button_color(QColor(20, 20, 20))  # dark -> white text
    assert "color: white" in widget._label_behavior_button.styleSheet()

    widget.set_behavior_button_color(QColor(240, 240, 240))  # light -> black text
    assert "color: black" in widget._label_behavior_button.styleSheet()


def test_set_behavior_button_color_disabled_text_contrasts() -> None:
    """Disabled text color contrasts with the derived disabled background."""
    widget = MainControlWidget()

    # dark behavior color -> dark disabled background -> light disabled text
    widget.set_behavior_button_color(QColor(20, 20, 20))
    assert "color: #cccccc" in widget._label_behavior_button.styleSheet()

    # light behavior color -> light disabled background -> dark disabled text
    widget.set_behavior_button_color(QColor(240, 240, 240))
    assert "color: #555555" in widget._label_behavior_button.styleSheet()


def test_default_disabled_text_is_grey() -> None:
    """Binary-mode default keeps grey disabled text (unchanged)."""
    widget = MainControlWidget()

    widget.set_behavior_button_color(None)
    assert "color: grey" in widget._label_behavior_button.styleSheet()
