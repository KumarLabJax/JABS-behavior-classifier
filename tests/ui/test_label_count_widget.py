import pytest

try:
    from PySide6.QtWidgets import QApplication

    from jabs.ui.main_control_widget.label_count_widget import FrameLabelCountWidget

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


def test_default_class_labels() -> None:
    """The summary defaults to the binary-mode row headers."""
    widget = FrameLabelCountWidget()

    assert [lbl.text() for lbl in widget._positive_row_labels] == ["Behavior", "Behavior"]
    assert [lbl.text() for lbl in widget._negative_row_labels] == [
        "Not Behavior",
        "Not Behavior",
    ]


def test_set_class_labels_retitles_both_rows() -> None:
    """set_class_labels updates the frame and bout row headers for both classes."""
    widget = FrameLabelCountWidget()

    widget.set_class_labels("Walk", "None")

    assert [lbl.text() for lbl in widget._positive_row_labels] == ["Walk", "Walk"]
    assert [lbl.text() for lbl in widget._negative_row_labels] == ["None", "None"]


def test_set_class_labels_can_restore_defaults() -> None:
    """Switching back to binary wording restores the standard headers."""
    widget = FrameLabelCountWidget()

    widget.set_class_labels("Walk", "None")
    widget.set_class_labels("Behavior", "Not Behavior")

    assert [lbl.text() for lbl in widget._positive_row_labels] == ["Behavior", "Behavior"]
    assert [lbl.text() for lbl in widget._negative_row_labels] == [
        "Not Behavior",
        "Not Behavior",
    ]
