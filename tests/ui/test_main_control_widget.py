from types import SimpleNamespace

import pytest

try:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import QApplication

    import jabs.ui.main_control_widget.main_control_widget as main_control_widget_module
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


class _RejectedDialog:
    """Stands in for the QInputDialog the user cancels with "Quit JABS"."""

    def __getattr__(self, _name):
        return lambda *args, **kwargs: None

    def windowFlags(self):
        return Qt.WindowType.Dialog

    def exec(self):
        return 0  # rejected


def _quit_prompt_stub(monkeypatch, close_accepted: bool) -> SimpleNamespace:
    """Patch the dialog and return a stub self whose window closes as requested."""
    monkeypatch.setattr(
        main_control_widget_module,
        "QtWidgets",
        SimpleNamespace(QInputDialog=_RejectedDialog),
    )
    closed = []

    def close() -> bool:
        closed.append(True)
        return close_accepted

    window = SimpleNamespace(close=close)
    return SimpleNamespace(window=lambda: window, closed=closed)


def test_first_label_quit_closes_the_main_window_before_exiting(monkeypatch) -> None:
    """Choosing "Quit JABS" at the first-behavior prompt runs window cleanup first.

    Closing the main window delivers MainWindow.closeEvent (stopping background
    threads and shutting down the process pool) before the interpreter exits.
    """
    stub = _quit_prompt_stub(monkeypatch, close_accepted=True)

    with pytest.raises(SystemExit) as exit_info:
        MainControlWidget._get_first_label(stub)

    assert exit_info.value.code == 0
    assert stub.closed == [True]


def test_first_label_quit_does_not_exit_when_the_close_is_declined(monkeypatch) -> None:
    """A declined close means the application is not quitting, so it must not exit.

    Exiting anyway would skip the cleanup that closing the window performs.
    """
    stub = _quit_prompt_stub(monkeypatch, close_accepted=False)

    # returns normally instead of raising SystemExit
    MainControlWidget._get_first_label(stub)

    assert stub.closed == [True]
