from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from jabs.core.constants import CLASSIFIER_MODE_KEY, CV_GROUPING_KEY, CV_GROUPING_REGEX_KEY
from jabs.core.enums import ClassifierMode, CrossValidationGroupingStrategy

try:
    from PySide6.QtWidgets import QApplication

    from jabs.ui.settings_dialog.classifier_mode_settings_group import (
        ClassifierModeSettingsGroup,
    )
    from jabs.ui.settings_dialog.cross_validation_settings_group import (
        CrossValidationSettingsGroup,
    )
    from jabs.ui.settings_dialog.settings_dialog import _OverlapCheckThread

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
    """Ensure a QApplication exists for UI-related tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_overlap_check_thread_emits_complete() -> None:
    """Successful overlap checks emit the conflicting video list."""
    project = SimpleNamespace(
        get_overlapping_behavior_label_videos=MagicMock(return_value=["video1.avi"])
    )
    emitted_results = []
    emitted_errors = []

    thread = _OverlapCheckThread(project)
    thread.check_complete.connect(emitted_results.append)
    thread.check_failed.connect(emitted_errors.append)

    thread.run()

    assert emitted_results == [["video1.avi"]]
    assert emitted_errors == []


def test_overlap_check_thread_emits_error_on_exception() -> None:
    """Failed overlap checks emit an error instead of silently ending the thread."""
    expected_error = RuntimeError("pose load failed")
    project = SimpleNamespace(
        get_overlapping_behavior_label_videos=MagicMock(side_effect=expected_error)
    )
    emitted_results = []
    emitted_errors = []

    thread = _OverlapCheckThread(project)
    thread.check_complete.connect(emitted_results.append)
    thread.check_failed.connect(emitted_errors.append)

    thread.run()

    assert emitted_results == []
    assert emitted_errors == [expected_error]


def test_classifier_mode_group_roundtrips_enum_not_label() -> None:
    """Set/get bind to the ClassifierMode value, independent of the display label."""
    group = ClassifierModeSettingsGroup()

    group.set_values({CLASSIFIER_MODE_KEY: ClassifierMode.MULTICLASS.value})
    assert group.get_values() == {CLASSIFIER_MODE_KEY: ClassifierMode.MULTICLASS}

    group.set_values({CLASSIFIER_MODE_KEY: ClassifierMode.BINARY.value})
    assert group.get_values() == {CLASSIFIER_MODE_KEY: ClassifierMode.BINARY}


def test_cv_grouping_group_roundtrips_strategy_and_regex() -> None:
    """Set/get round-trips the grouping strategy enum and the filename regex."""
    group = CrossValidationSettingsGroup()

    group.set_values(
        {
            CV_GROUPING_KEY: CrossValidationGroupingStrategy.FILENAME_PATTERN.value,
            CV_GROUPING_REGEX_KEY: r"cage_(\d+)",
        }
    )
    values = group.get_values()
    assert values[CV_GROUPING_KEY] == CrossValidationGroupingStrategy.FILENAME_PATTERN
    assert values[CV_GROUPING_REGEX_KEY] == r"cage_(\d+)"


def test_cv_grouping_group_validate_blocks_empty_pattern() -> None:
    """The Filename Pattern strategy with no regex fails validation."""
    group = CrossValidationSettingsGroup()
    group.set_values(
        {
            CV_GROUPING_KEY: CrossValidationGroupingStrategy.FILENAME_PATTERN.value,
            CV_GROUPING_REGEX_KEY: "",
        }
    )
    assert group.validate() is not None


def test_cv_grouping_group_validate_blocks_invalid_pattern() -> None:
    """The Filename Pattern strategy with an invalid regex fails validation."""
    group = CrossValidationSettingsGroup()
    group.set_values(
        {
            CV_GROUPING_KEY: CrossValidationGroupingStrategy.FILENAME_PATTERN.value,
            CV_GROUPING_REGEX_KEY: "cage_(",
        }
    )
    assert group.validate() is not None


def test_cv_grouping_group_validate_passes_for_valid_pattern() -> None:
    """A valid Filename Pattern regex passes validation."""
    group = CrossValidationSettingsGroup()
    group.set_values(
        {
            CV_GROUPING_KEY: CrossValidationGroupingStrategy.FILENAME_PATTERN.value,
            CV_GROUPING_REGEX_KEY: r"cage_(\d+)",
        }
    )
    assert group.validate() is None


def test_cv_grouping_group_validate_ignores_regex_for_other_strategies() -> None:
    """Non-pattern strategies do not validate the regex field, even if it is invalid."""
    group = CrossValidationSettingsGroup()
    group.set_values(
        {
            CV_GROUPING_KEY: CrossValidationGroupingStrategy.VIDEO.value,
            CV_GROUPING_REGEX_KEY: "cage_(",  # invalid, but ignored for VIDEO grouping
        }
    )
    assert group.validate() is None


_PREVIEW_VIDEOS = [
    ("cage_0042_day1.mp4", False),
    ("cage_0042_day2.mp4", True),  # excluded from training
    ("cage_0043_day1.mp4", False),
    ("calibration.mp4", False),  # does not match cage_(\d+)
]


def _select_filename_pattern(group, regex: str) -> None:
    group.set_values(
        {
            CV_GROUPING_KEY: CrossValidationGroupingStrategy.FILENAME_PATTERN.value,
            CV_GROUPING_REGEX_KEY: regex,
        }
    )


def test_cv_grouping_preview_summarizes_groups_and_unmatched() -> None:
    """The preview summary counts videos, groups, and unmatched files."""
    group = CrossValidationSettingsGroup(videos=_PREVIEW_VIDEOS)
    _select_filename_pattern(group, r"cage_(\d+)")

    summary = group._preview_summary_label.text()
    # 4 videos -> cage 0042, cage 0043, plus the unmatched calibration video = 3 groups.
    assert "4 videos" in summary
    assert "3 groups" in summary
    assert "1 unmatched video" in summary
    assert not group._preview_summary_label.isHidden()
    assert not group._preview_section.isHidden()


def test_cv_grouping_preview_lists_groups_and_marks_excluded() -> None:
    """The breakdown lists group keys, members, and marks excluded videos."""
    group = CrossValidationSettingsGroup(videos=_PREVIEW_VIDEOS)
    _select_filename_pattern(group, r"cage_(\d+)")

    detail = group._preview_detail.text()
    assert "0042" in detail
    assert "0043" in detail
    assert "calibration.mp4" in detail
    assert "unmatched" in detail
    # The excluded video is annotated; non-excluded videos are not.
    assert "(excluded)" in detail
    assert detail.count("(excluded)") == 1


@pytest.mark.parametrize("regex", ["", "cage_("], ids=["empty", "invalid"])
def test_cv_grouping_preview_hidden_for_empty_or_invalid_regex(regex) -> None:
    """No preview is shown when the regex is empty or does not compile."""
    group = CrossValidationSettingsGroup(videos=_PREVIEW_VIDEOS)
    _select_filename_pattern(group, regex)

    assert group._preview_summary_label.isHidden()
    assert group._preview_section.isHidden()


def test_cv_grouping_preview_hidden_for_non_pattern_strategy() -> None:
    """No preview is shown for the Video or Individual Animal strategies."""
    group = CrossValidationSettingsGroup(videos=_PREVIEW_VIDEOS)
    group.set_values(
        {
            CV_GROUPING_KEY: CrossValidationGroupingStrategy.VIDEO.value,
            CV_GROUPING_REGEX_KEY: r"cage_(\d+)",
        }
    )

    assert group._preview_summary_label.isHidden()
    assert group._preview_section.isHidden()


def test_cv_grouping_preview_handles_no_videos() -> None:
    """With no project videos, the preview reports that there is nothing to show."""
    group = CrossValidationSettingsGroup(videos=[])
    _select_filename_pattern(group, r"cage_(\d+)")

    assert "No videos" in group._preview_summary_label.text()
    assert not group._preview_summary_label.isHidden()
    assert group._preview_section.isHidden()
