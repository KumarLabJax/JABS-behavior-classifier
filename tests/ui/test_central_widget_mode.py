"""Tests for the central_widget_mode per-mode dispatch helpers."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pytest

from jabs.core.constants import MULTICLASS_NONE_BEHAVIOR
from jabs.core.enums import ClassifierMode
from jabs.project import TrackLabels, VideoLabels

try:
    from jabs.ui.main_window import central_widget_mode

    SKIP_UI_TESTS = False
    SKIP_REASON = None
except ImportError as e:
    SKIP_UI_TESTS = True
    SKIP_REASON = f"Qt/UI dependencies not available: {e}"

pytestmark = pytest.mark.skipif(
    SKIP_UI_TESTS,
    reason=SKIP_REASON if SKIP_UI_TESTS else "",
)

_NUM_FRAMES = 30
_BEHAVIOR = TrackLabels.Label.BEHAVIOR
_NOT_BEHAVIOR = TrackLabels.Label.NOT_BEHAVIOR


def _track(labels: VideoLabels, behavior: str, identity: str = "0") -> npt.NDArray[np.int8]:
    """Return the label values of one identity's track for a behavior."""
    return labels.get_track_labels(identity, behavior).get_labels()


def _runs(*runs: tuple[int, int, int]) -> npt.NDArray[np.int8]:
    """Build the expected label values: unlabeled except for ``(start, end, value)`` runs.

    Both ends of a run are inclusive, as in ``TrackLabels``.
    """
    expected = np.full(_NUM_FRAMES, TrackLabels.Label.NONE, dtype=np.int8)
    for start, end, value in runs:
        expected[start : end + 1] = value
    return expected


# ---------------------------------------------------------------------------
# load_video_predictions
# ---------------------------------------------------------------------------


def test_load_video_predictions_binary_returns_none_class_names() -> None:
    """Binary mode delegates to load_predictions and returns None for class_names."""
    prediction_manager = SimpleNamespace(
        load_predictions=MagicMock(
            return_value=({0: np.zeros(3)}, {0: np.zeros(3)}, {0: np.zeros(3)})
        ),
        load_multiclass_predictions=MagicMock(),
    )

    preds, probs, postprocessed, class_names = central_widget_mode.load_video_predictions(
        prediction_manager,
        ClassifierMode.BINARY,
        video_name="video.avi",
        behavior="Walk",
    )

    prediction_manager.load_predictions.assert_called_once_with("video.avi", "Walk")
    prediction_manager.load_multiclass_predictions.assert_not_called()
    assert class_names is None
    assert 0 in preds and 0 in probs and 0 in postprocessed


def test_load_video_predictions_multiclass_returns_class_names() -> None:
    """Multi-class mode delegates to load_multiclass_predictions and forwards class_names."""
    prediction_manager = SimpleNamespace(
        load_predictions=MagicMock(),
        load_multiclass_predictions=MagicMock(
            return_value=({0: np.zeros(3)}, {0: np.zeros((3, 3))}, {}, ["None", "Walk", "Run"])
        ),
    )

    preds, probs, postprocessed, class_names = central_widget_mode.load_video_predictions(
        prediction_manager,
        ClassifierMode.MULTICLASS,
        video_name="video.avi",
        behavior="Walk",
    )

    prediction_manager.load_multiclass_predictions.assert_called_once_with("video.avi")
    prediction_manager.load_predictions.assert_not_called()
    assert class_names == ["None", "Walk", "Run"]
    assert probs[0].ndim == 2


# ---------------------------------------------------------------------------
# apply_behavior_label
# ---------------------------------------------------------------------------


def test_apply_behavior_label_binary_does_not_clear_competing() -> None:
    """Binary mode labels the current track and leaves other behaviors' labels alone."""
    labels = VideoLabels("t.avi", _NUM_FRAMES)
    labels.get_track_labels("0", "Run").label_behavior(5, 25)

    central_widget_mode.apply_behavior_label(
        labels,
        ClassifierMode.BINARY,
        identity_str="0",
        current_behavior="Walk",
        start=10,
        end=20,
    )

    np.testing.assert_array_equal(_track(labels, "Walk"), _runs((10, 20, _BEHAVIOR)))
    np.testing.assert_array_equal(_track(labels, "Run"), _runs((5, 25, _BEHAVIOR)))


def test_apply_behavior_label_multiclass_clears_competing_then_labels() -> None:
    """Multi-class mode clears non-current behavior tracks on the range, then labels current."""
    labels = VideoLabels("t.avi", _NUM_FRAMES)
    labels.get_track_labels("0", "Run").label_behavior(5, 25)
    labels.get_track_labels("0", "Walk").label_behavior(0, 2)

    central_widget_mode.apply_behavior_label(
        labels,
        ClassifierMode.MULTICLASS,
        identity_str="0",
        current_behavior="Walk",
        start=10,
        end=20,
    )

    # the competing behavior loses only the labeled range
    np.testing.assert_array_equal(
        _track(labels, "Run"), _runs((5, 9, _BEHAVIOR), (21, 25, _BEHAVIOR))
    )
    np.testing.assert_array_equal(
        _track(labels, "Walk"), _runs((0, 2, _BEHAVIOR), (10, 20, _BEHAVIOR))
    )


# ---------------------------------------------------------------------------
# apply_not_behavior_label
# ---------------------------------------------------------------------------


def test_apply_not_behavior_label_binary_returns_current_behavior_false() -> None:
    """Binary mode labels the current track as not-behavior and reports (behavior, False)."""
    labels = VideoLabels("t.avi", _NUM_FRAMES)
    labels.get_track_labels("0", "Run").label_behavior(0, 29)

    behavior_key, is_positive = central_widget_mode.apply_not_behavior_label(
        labels,
        ClassifierMode.BINARY,
        identity_str="0",
        current_behavior="Walk",
        start=5,
        end=15,
    )

    assert behavior_key == "Walk"
    assert is_positive is False
    np.testing.assert_array_equal(_track(labels, "Walk"), _runs((5, 15, _NOT_BEHAVIOR)))
    np.testing.assert_array_equal(_track(labels, "Run"), _runs((0, 29, _BEHAVIOR)))


def test_apply_not_behavior_label_multiclass_returns_none_key_true() -> None:
    """Multi-class mode clears competing tracks and labels the NONE track as positive."""
    labels = VideoLabels("t.avi", _NUM_FRAMES)
    labels.get_track_labels("0", "Walk").label_behavior(0, 10)
    labels.get_track_labels("0", "Run").label_behavior(12, 29)

    behavior_key, is_positive = central_widget_mode.apply_not_behavior_label(
        labels,
        ClassifierMode.MULTICLASS,
        identity_str="0",
        current_behavior="Walk",
        start=5,
        end=15,
    )

    assert behavior_key == MULTICLASS_NONE_BEHAVIOR
    assert is_positive is True
    np.testing.assert_array_equal(
        _track(labels, MULTICLASS_NONE_BEHAVIOR), _runs((5, 15, _BEHAVIOR))
    )
    np.testing.assert_array_equal(_track(labels, "Walk"), _runs((0, 4, _BEHAVIOR)))
    np.testing.assert_array_equal(_track(labels, "Run"), _runs((16, 29, _BEHAVIOR)))


# ---------------------------------------------------------------------------
# build_timeline_label_arrays
# ---------------------------------------------------------------------------


def test_build_timeline_label_arrays_multiclass_uses_merged_arrays() -> None:
    """Multi-class returns one merged label array per identity from VideoLabels."""
    labels = VideoLabels("t.avi", _NUM_FRAMES)
    labels.get_track_labels("0", "Walk").label_behavior(2, 3)
    labels.get_track_labels("1", "Run").label_behavior(0, 1)

    result = central_widget_mode.build_timeline_label_arrays(
        labels,
        ClassifierMode.MULTICLASS,
        num_identities=2,
        current_behavior="Walk",
        behaviors=["Walk", "Run"],
    )

    # class index 0 is unlabeled and 1 is the None class, so behaviors start at 2
    expected_first = np.zeros(_NUM_FRAMES, dtype=np.int16)
    expected_first[2:4] = 2
    expected_second = np.zeros(_NUM_FRAMES, dtype=np.int16)
    expected_second[0:2] = 3
    assert len(result) == 2
    np.testing.assert_array_equal(result[0], expected_first)
    np.testing.assert_array_equal(result[1], expected_second)


def test_build_timeline_label_arrays_binary_uses_lut_indices() -> None:
    """Binary returns one LUT-index array per identity for current_behavior only."""
    labels = VideoLabels("t.avi", _NUM_FRAMES)
    walk = labels.get_track_labels("0", "Walk")
    walk.label_behavior(2, 3)
    walk.label_not_behavior(4, 5)
    labels.get_track_labels("0", "Run").label_behavior(6, 7)  # another behavior is ignored
    labels.get_track_labels("1", "Run").label_behavior(0, 1)

    result = central_widget_mode.build_timeline_label_arrays(
        labels,
        ClassifierMode.BINARY,
        num_identities=2,
        current_behavior="Walk",
        behaviors=["Walk", "Run"],
    )

    # label values shift up by one: unlabeled -> 0, not behavior -> 1, behavior -> 2
    expected_first = np.zeros(_NUM_FRAMES, dtype=np.int16)
    expected_first[2:4] = 2
    expected_first[4:6] = 1
    assert len(result) == 2
    np.testing.assert_array_equal(result[0], expected_first)
    np.testing.assert_array_equal(result[1], np.zeros(_NUM_FRAMES, dtype=np.int16))
