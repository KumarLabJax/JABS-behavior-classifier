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


def _assert_default_button_tint(widget: "MainControlWidget") -> None:
    """Assert the Label Behavior button has the binary-mode default tint.

    Args:
        widget: Control widget whose Label Behavior button style to check.
    """
    style = widget._label_behavior_button.styleSheet()
    assert f"rgba{BEHAVIOR_COLOR.getRgb()}" in style
    assert "color: white" in style
    # the binary-mode default keeps grey disabled text
    assert "color: grey" in style


def test_label_button_has_the_default_orange_tint_and_none_restores_it() -> None:
    """A new widget starts with the default orange tint, and passing None restores it.

    The constructor applies the default through ``set_behavior_button_color(None)``,
    so both the freshly built widget and the restored one are checked.
    """
    widget = MainControlWidget()

    _assert_default_button_tint(widget)

    widget.set_behavior_button_color(QColor(10, 20, 30))
    widget.set_behavior_button_color(None)

    _assert_default_button_tint(widget)


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


def test_all_kfold_checkbox_reports_the_cross_validation_change() -> None:
    """Toggling "All k-fold" emits kfold_changed so the train button is re-evaluated.

    The checkbox overrides the k slider, so the number of cross-validation groups
    the labels must support changes even though the slider value does not.
    """
    widget = MainControlWidget()
    emitted = []
    widget.kfold_changed.connect(lambda: emitted.append(widget.all_kfold))

    widget._all_kfold_checkbox.setChecked(True)
    assert not widget._kslider.isEnabled()

    widget._all_kfold_checkbox.setChecked(False)
    assert widget._kslider.isEnabled()

    assert emitted == [True, False]
