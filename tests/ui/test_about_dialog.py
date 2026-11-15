"""Tests for the About dialog's optional extras section."""

import pytest

from jabs.core.utils import ExtraStatus

try:
    from PySide6.QtWidgets import QApplication, QLabel

    from jabs.ui.dialogs.about_dialog import (
        EXTRAS_USER_GUIDE_PAGE,
        AboutDialog,
        extra_status_text,
    )

    SKIP_UI_TESTS = False
    SKIP_REASON = ""
except ImportError as e:
    SKIP_UI_TESTS = True
    SKIP_REASON = f"Qt/UI dependencies not available: {e}"

pytestmark = pytest.mark.skipif(SKIP_UI_TESTS, reason=SKIP_REASON)


@pytest.fixture(scope="module", autouse=True)
def qapp():
    """Ensure a QApplication exists for widget tests."""
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (ExtraStatus("nwb", ("pynwb",), ()), "installed"),
        (ExtraStatus("nwb", ("pynwb",), ("pynwb",)), "not installed"),
        (
            ExtraStatus("extra-classifiers", ("catboost", "lightgbm"), ("catboost",)),
            "partially installed (missing: catboost)",
        ),
    ],
    ids=["installed", "not-installed", "partial"],
)
def test_extra_status_text(status: ExtraStatus, expected: str) -> None:
    """Each extra's status reads as installed, not installed, or names what is missing."""
    assert extra_status_text(status) == expected


def _label_text(dialog: AboutDialog) -> str:
    """All label text in the dialog, joined, for checking what it shows."""
    return "\n".join(label.text() for label in dialog.findChildren(QLabel))


def test_dialog_lists_each_extra_with_its_status() -> None:
    """The extras passed in are listed by name with their status."""
    dialog = AboutDialog(
        "JABS",
        extras=[
            ExtraStatus("mlflow", ("mlflow",), ()),
            ExtraStatus("yaml", ("pyyaml",), ("pyyaml",)),
        ],
    )

    text = _label_text(dialog)

    # the status is its own table cell, so match it exactly
    assert "mlflow</code></td><td style='padding-left: 12px'>installed</td>" in text
    assert "yaml</code></td><td style='padding-left: 12px'>not installed</td>" in text


def test_dialog_explains_when_extras_cannot_be_determined() -> None:
    """With no extras to report, the dialog says so instead of showing an empty table."""
    dialog = AboutDialog("JABS", extras=[])

    assert "Unable to determine installed extras" in _label_text(dialog)


def test_user_guide_link_closes_dialog_and_requests_the_extras_page() -> None:
    """Following the link closes the dialog, so the guide is not blocked behind it."""
    dialog = AboutDialog("JABS", extras=[])
    dialog.show()
    requested: list[str] = []
    dialog.user_guide_page_requested.connect(requested.append)

    link = next(label for label in dialog.findChildren(QLabel) if "How to install" in label.text())
    link.linkActivated.emit(EXTRAS_USER_GUIDE_PAGE)

    assert requested == [EXTRAS_USER_GUIDE_PAGE]
    assert not dialog.isVisible()
