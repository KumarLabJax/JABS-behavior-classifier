"""Cross-validation settings group for configuring model training and validation."""

import html
import re

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QComboBox, QLabel, QLineEdit, QSizePolicy

from jabs.core.constants import CV_GROUPING_KEY, CV_GROUPING_REGEX_KEY
from jabs.core.enums import (
    DEFAULT_CV_GROUPING_STRATEGY,
    CrossValidationGroupingStrategy,
    filename_group_key,
)

from .collapsible_section import CollapsibleSection
from .settings_group import SettingsGroup

# Debounce delay (ms) before recomputing the filename-pattern preview while typing.
_PREVIEW_DEBOUNCE_MS = 200


def _count_phrase(count: int, noun: str) -> str:
    """Return ``"<count> <noun>"`` with a naive plural ``s`` suffix."""
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


class CrossValidationSettingsGroup(SettingsGroup):
    """
    Settings group for cross-validation configuration.

    This group controls how data is split during model training and validation.
    """

    def __init__(
        self,
        videos: list[tuple[str, bool]] | None = None,
        parent=None,
    ) -> None:
        """Initialize the cross-validation settings group.

        Args:
            videos: ``(video_filename, is_excluded)`` pairs for every video in the
                project, used to render the filename-pattern grouping preview. When
                omitted, the preview is empty (the rest of the group still works).
            parent: Parent widget.
        """
        # Stored before super().__init__ so it is available to _create_controls().
        self._video_entries: list[tuple[str, bool]] = list(videos) if videos else []
        super().__init__("Cross-Validation", parent)

    def _create_controls(self) -> None:
        """Create the cross-validation settings controls."""
        # Cross-validation grouping combo box
        self._cv_grouping_combo = QComboBox()
        # Add enum values as items, storing the enum as userData
        for enum_val in CrossValidationGroupingStrategy:
            self._cv_grouping_combo.addItem(enum_val.value, enum_val)
        self._cv_grouping_combo.setCurrentIndex(
            self._cv_grouping_combo.findData(DEFAULT_CV_GROUPING_STRATEGY)
        )
        self._cv_grouping_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.add_control_row("CV Grouping:", self._cv_grouping_combo)

        # Regex input, only relevant (and visible) for the "Filename Pattern" strategy.
        self._cv_grouping_regex_input = QLineEdit()
        self._cv_grouping_regex_input.setPlaceholderText(r"e.g. cage_(\d+)")
        self._cv_grouping_regex_input.setMinimumWidth(220)
        regex_row = self.add_control_row("Filename Pattern:", self._cv_grouping_regex_input)
        regex_label_item = self._grid_layout.itemAtPosition(regex_row, 0)
        self._regex_label = regex_label_item.widget() if regex_label_item is not None else None

        # Inline validation message shown below the input when the regex is invalid.
        self._regex_error_label = QLabel()
        self._regex_error_label.setWordWrap(True)
        self._regex_error_label.setStyleSheet("color: #c0392b;")
        self.add_widget_row(self._regex_error_label)

        # Live preview of how project videos partition into groups under the pattern.
        self._preview_summary_label = QLabel()
        self._preview_summary_label.setWordWrap(True)
        self._preview_summary_label.setStyleSheet("color: #555;")
        self.add_widget_row(self._preview_summary_label)

        self._preview_detail = QLabel()
        self._preview_detail.setTextFormat(Qt.TextFormat.RichText)
        self._preview_detail.setWordWrap(True)
        self._preview_detail.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )
        self._preview_section = CollapsibleSection("Preview groups", self._preview_detail, self)
        self._preview_section.sizeChanged.connect(self._relayout_preview)
        self.add_widget_row(self._preview_section)

        # Debounce preview recomputation while the user is typing a pattern.
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(_PREVIEW_DEBOUNCE_MS)
        self._preview_timer.timeout.connect(self._refresh_preview)

        self._cv_grouping_combo.currentIndexChanged.connect(self._update_regex_row_visibility)
        self._cv_grouping_regex_input.textChanged.connect(self._update_regex_validation)
        self._cv_grouping_regex_input.textChanged.connect(self._preview_timer.start)

        self._update_regex_row_visibility()

    def _is_filename_pattern_selected(self) -> bool:
        """Return True if the "Filename Pattern" grouping strategy is selected."""
        return (
            self._cv_grouping_combo.currentData()
            == CrossValidationGroupingStrategy.FILENAME_PATTERN
        )

    def _update_regex_row_visibility(self) -> None:
        """Show the regex input only when the filename-pattern strategy is selected."""
        visible = self._is_filename_pattern_selected()
        if self._regex_label is not None:
            self._regex_label.setVisible(visible)
        self._cv_grouping_regex_input.setVisible(visible)
        self._update_regex_validation()
        self._refresh_preview()
        # Notify the layout so the group grows/shrinks as the rows appear/hide.
        self.updateGeometry()

    def _update_regex_validation(self) -> None:
        """Refresh the inline regex validation message."""
        message = ""
        if self._is_filename_pattern_selected():
            regex = self._cv_grouping_regex_input.text().strip()
            if not regex:
                message = "Enter a regular expression to group videos by filename."
            else:
                try:
                    re.compile(regex)
                except re.error as e:
                    message = f"Invalid regular expression: {e}"
        self._regex_error_label.setText(message)
        self._regex_error_label.setVisible(bool(message))

    def _compiled_preview_pattern(self) -> re.Pattern[str] | None:
        """Return the compiled pattern to preview, or None when not applicable.

        Returns None when the filename-pattern strategy is not selected or the
        current regex is empty or invalid (the inline error already covers those).
        """
        if not self._is_filename_pattern_selected():
            return None
        regex = self._cv_grouping_regex_input.text().strip()
        if not regex:
            return None
        try:
            return re.compile(regex)
        except re.error:
            return None

    def _refresh_preview(self) -> None:
        """Recompute and render the filename-pattern grouping preview."""
        self._preview_timer.stop()
        pattern = self._compiled_preview_pattern()
        if pattern is None:
            self._preview_summary_label.setVisible(False)
            self._preview_section.setVisible(False)
            return

        matched: dict[str, list[tuple[str, bool]]] = {}
        unmatched: list[tuple[str, bool]] = []
        for name, excluded in self._video_entries:
            if pattern.search(name) is None:
                unmatched.append((name, excluded))
            else:
                key = filename_group_key(name, pattern)
                matched.setdefault(key, []).append((name, excluded))

        n_videos = len(self._video_entries)
        # Each unmatched video forms its own group, so it counts toward the total.
        n_groups = len(matched) + len(unmatched)

        if n_videos == 0:
            self._preview_summary_label.setText("No videos in the project to preview.")
            self._preview_summary_label.setVisible(True)
            self._preview_section.setVisible(False)
            return

        summary = f"{_count_phrase(n_videos, 'video')} → {_count_phrase(n_groups, 'group')}"
        if unmatched:
            summary += f" ({_count_phrase(len(unmatched), 'unmatched video')})"
        self._preview_summary_label.setText(summary)
        self._preview_summary_label.setVisible(True)

        self._preview_detail.setText(self._render_preview_detail(matched, unmatched))
        self._preview_section.setVisible(True)

    @staticmethod
    def _render_video(name: str, excluded: bool) -> str:
        """Render one video filename for the preview, marking excluded videos."""
        safe = html.escape(name)
        if excluded:
            return f'{safe} <span style="color:#888;">(excluded)</span>'
        return safe

    def _render_preview_detail(
        self,
        matched: dict[str, list[tuple[str, bool]]],
        unmatched: list[tuple[str, bool]],
    ) -> str:
        """Build the rich-text breakdown of groups and their member videos."""
        lines: list[str] = []
        for key in sorted(matched):
            members = ", ".join(self._render_video(name, excl) for name, excl in matched[key])
            lines.append(f"<b>{html.escape(key)}</b> &rarr; {members}")
        if unmatched:
            members = ", ".join(self._render_video(name, excl) for name, excl in unmatched)
            lines.append(f"<i>unmatched (each its own group)</i>: {members}")
        return "<br>".join(lines)

    def _relayout_preview(self) -> None:
        """Grow/shrink the group (and the dialog page) when the preview is toggled."""
        self._preview_detail.adjustSize()
        self._preview_section.adjustSize()
        self.adjustSize()
        parent = self.parentWidget()
        if parent is not None:
            parent.adjustSize()
            parent_layout = parent.layout()
            if parent_layout is not None:
                parent_layout.activate()
            parent.adjustSize()
            dialog = self._find_parent_dialog()
            if dialog is not None and hasattr(dialog, "_sync_page_width"):
                # Defer so the scrollbar settles before the width is synced.
                QTimer.singleShot(0, dialog._sync_page_width)

    def _create_documentation(self) -> QLabel:
        """Create help documentation for cross-validation settings."""
        help_label = QLabel(self)
        help_label.setTextFormat(Qt.TextFormat.RichText)
        help_label.setWordWrap(True)
        help_label.setText(
            """
            <h3>What is Cross-Validation Grouping?</h3>
            <p>Cross-validation grouping determines how training data is split when
            evaluating model performance using leave-one-group-out cross-validation.</p>

            <ul>
              <li><b>Individual Animal:</b> Each group represents a single animal identity
              within a single video. During cross-validation, all labeled data for one
              animal from one video is held out for validation while the remaining animals'
              data is used for training.</li>

              <li><b>Video:</b> Each group represents a single video recording. During
              cross-validation, all labeled data from one video is held out for validation
              while data from other videos is used for training.</li>

              <li><b>Filename Pattern:</b> Each group is defined by a regular expression
              applied to the video filename. All videos whose filenames produce the same
              key are placed in the same group, letting you group videos by an identifier
              embedded in their names (for example, a cage ID). If the pattern contains a
              capture group, the captured text is used as the key; otherwise the entire
              match is used. Videos that do not match the pattern are each placed in their
              own group.<br>
              <b>Example - group by cage ID:</b> if your videos are named like
              <tt>cage_0042_2026-06-16.mp4</tt>, the pattern <tt>cage_(\\d+)</tt> extracts
              the cage number (<tt>0042</tt>), so every video recorded from the same cage
              forms a single cross-validation group.</li>
            </ul>

            <p><b>Note:</b> For cross-validation to work properly, you need labeled data
            from multiple groups (multiple animals, videos, or filename-pattern groups,
            depending on the grouping method selected). For rare behaviors, it may be
            easier to meet the minimum label requirements per group at the video or
            filename-pattern level rather than at the individual animal level within a
            single video.</p>
            """
        )
        help_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        return help_label

    def get_values(self) -> dict:
        """
        Get current cross-validation settings values.

        Returns:
            Dictionary with setting names and their current values.
        """
        # Return the enum, not just the string. The regex is always returned (even
        # for non-pattern strategies) so a previously entered pattern is preserved.
        return {
            CV_GROUPING_KEY: self._cv_grouping_combo.currentData(),
            CV_GROUPING_REGEX_KEY: self._cv_grouping_regex_input.text().strip(),
        }

    def set_values(self, values: dict) -> None:
        """
        Set cross-validation settings values from a dictionary.

        Args:
            values: Dictionary with setting names and values to apply.
        """
        cv_grouping = values.get(CV_GROUPING_KEY, CrossValidationGroupingStrategy.INDIVIDUAL)

        # The grouping setting is saved as the string value, try to convert string from settings dict back to enum
        try:
            enum_val = CrossValidationGroupingStrategy(cv_grouping)
            index = self._cv_grouping_combo.findData(enum_val)
        except ValueError:
            # Invalid setting value, we'll treat it as not found
            index = -1

        if index >= 0:
            self._cv_grouping_combo.setCurrentIndex(index)
        else:
            # Fall back to default if invalid value or not found
            self._cv_grouping_combo.setCurrentIndex(
                self._cv_grouping_combo.findData(CrossValidationGroupingStrategy.INDIVIDUAL)
            )

        regex = values.get(CV_GROUPING_REGEX_KEY, "")
        self._cv_grouping_regex_input.setText(regex if isinstance(regex, str) else "")
        self._update_regex_row_visibility()

    def validate(self) -> str | None:
        """Validate the filename-pattern regex when that strategy is selected.

        Returns:
            An error message if the "Filename Pattern" strategy is selected with an
            empty or invalid regular expression; otherwise None.
        """
        if not self._is_filename_pattern_selected():
            return None
        regex = self._cv_grouping_regex_input.text().strip()
        if not regex:
            return "Filename Pattern cross-validation grouping requires a regular expression."
        try:
            re.compile(regex)
        except re.error as e:
            return f"The filename grouping pattern is not a valid regular expression: {e}"
        return None
