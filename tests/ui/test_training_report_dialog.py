"""Tests for TrainingReportDialog UI component."""

from textwrap import dedent

import pytest

# Try to import Qt/UI modules; if Qt/EGL is not available (e.g., headless CI),
# mark all tests in this module to be skipped
try:
    from PySide6.QtWidgets import QApplication

    from jabs.ui.dialogs import TrainingReportDialog

    SKIP_UI_TESTS = False
    SKIP_REASON = None
except ImportError as e:
    # Qt/PySide6 not available (likely headless environment)
    SKIP_UI_TESTS = True
    SKIP_REASON = f"Qt/UI dependencies not available: {e}"

pytestmark = pytest.mark.skipif(
    SKIP_UI_TESTS,
    reason=SKIP_REASON if SKIP_UI_TESTS else "",
)


@pytest.fixture
def sample_markdown():
    """Create sample markdown content for testing.

    Returns:
        String containing sample markdown report.
    """
    return dedent("""
        # Training Report: Grooming

        **Date:** January 03, 2026 at 02:30:45 PM

        ## Training Summary

        - **Behavior:** Grooming
        - **Classifier:** Random Forest
        - **Distance Unit:** cm
        - **Training Time:** 12.35 seconds

        ### Label Counts

        - **Behavior frames:** 1,250
        - **Not-behavior frames:** 3,840

        ## Feature Importance

        | Rank | Feature Name | Importance |
        |------|-------------|------------|
        | 1    | nose_speed  | 0.16       |
        | 2    | ear_angle   | 0.14       |
    """).strip()


@pytest.fixture(scope="module", autouse=True)
def qapp():
    """Create QApplication instance for Qt widget tests.

    This fixture is automatically used for all tests in this module to ensure
    a QApplication exists before any Qt widgets are created.

    Returns:
        QApplication instance.
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app
    # QApplication cleanup happens automatically


class TestTrainingReportDialog:
    """Tests for TrainingReportDialog class."""

    def test_dialog_creation(self, sample_markdown):
        """Test that dialog can be created."""
        dialog = TrainingReportDialog(sample_markdown)

        assert dialog is not None
        assert dialog.windowTitle() == "Training Report"

    def test_custom_title(self, sample_markdown):
        """Test that custom title is set."""
        dialog = TrainingReportDialog(sample_markdown, title="Custom Title")

        assert dialog.windowTitle() == "Custom Title"

    def test_markdown_content_stored(self, sample_markdown):
        """Test that markdown content is stored for clipboard."""
        dialog = TrainingReportDialog(sample_markdown)

        assert dialog._markdown_content == sample_markdown

    def test_web_view_created(self, sample_markdown):
        """Test that web view is created and added to dialog."""
        dialog = TrainingReportDialog(sample_markdown)

        assert dialog.web_view is not None

    def test_copy_button_exists(self, sample_markdown):
        """Test that copy button is created."""
        dialog = TrainingReportDialog(sample_markdown)

        # Find the copy button by looking for widgets with tooltip
        widgets = dialog.findChildren(object)
        copy_buttons = [
            w
            for w in widgets
            if hasattr(w, "toolTip") and w.toolTip() == "Copy Markdown to Clipboard"
        ]

        assert len(copy_buttons) > 0

    def test_copy_to_clipboard(self, sample_markdown):
        """Test copying markdown to clipboard."""
        dialog = TrainingReportDialog(sample_markdown)

        # Call the copy method directly
        dialog._copy_markdown_to_clipboard()

        # Check clipboard content
        clipboard = QApplication.clipboard()
        assert clipboard.text() == sample_markdown

    def test_markdown_to_html_conversion(self):
        """Test markdown to HTML conversion."""
        simple_markdown = "# Header\n\nSome **bold** text."
        dialog = TrainingReportDialog(simple_markdown)

        html = dialog._markdown_to_html(simple_markdown)

        assert "<!DOCTYPE html>" in html
        assert "<html>" in html
        assert "<h1>" in html
        assert "<strong>bold</strong>" in html

    def test_html_includes_styling(self, sample_markdown):
        """Test that generated HTML includes CSS styling."""
        dialog = TrainingReportDialog(sample_markdown)

        html = dialog._markdown_to_html(sample_markdown)

        assert "<style>" in html
        assert "font-family:" in html
        assert "table {" in html

    def test_markdown_tables_converted(self):
        """Test that markdown tables are converted to HTML tables."""
        markdown_with_table = dedent("""
            | Column 1 | Column 2 |
            |----------|----------|
            | Value 1  | Value 2  |
        """).strip()
        dialog = TrainingReportDialog(markdown_with_table)

        html = dialog._markdown_to_html(markdown_with_table)

        assert "<table>" in html
        assert "<th>" in html
        assert "<td>" in html

    def test_markdown_code_blocks_supported(self):
        """Test that code blocks are supported in markdown."""
        markdown_with_code = dedent("""
            ```python
            def hello():
                print("world")
            ```
        """).strip()
        dialog = TrainingReportDialog(markdown_with_code)

        html = dialog._markdown_to_html(markdown_with_code)

        assert "<code>" in html or "<pre>" in html


class TestTrainingReportDialogIntegration:
    """Integration tests for TrainingReportDialog."""

    def test_parent_widget_set(self, sample_markdown):
        """Test that parent widget can be set to None."""
        dialog = TrainingReportDialog(sample_markdown, parent=None)

        # Dialog should be created successfully with None parent
        assert dialog is not None
        assert dialog.parent() is None

    def test_dialog_can_be_shown(self, sample_markdown):
        """Test that dialog can be shown (non-modal)."""
        dialog = TrainingReportDialog(sample_markdown)

        # Should not raise an error
        dialog.show()
        dialog.close()

    def test_multiple_dialogs(self, sample_markdown):
        """Test that multiple dialogs can be created."""
        dialog1 = TrainingReportDialog(sample_markdown, title="Report 1")
        dialog2 = TrainingReportDialog(sample_markdown, title="Report 2")

        assert dialog1.windowTitle() == "Report 1"
        assert dialog2.windowTitle() == "Report 2"
        assert dialog1 is not dialog2
