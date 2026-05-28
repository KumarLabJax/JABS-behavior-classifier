"""Tests for the central_widget_mode per-mode dispatch helpers."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from jabs.core.constants import MULTICLASS_NONE_BEHAVIOR
from jabs.core.enums import ClassifierMode

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


def _make_track_mock() -> MagicMock:
    track = MagicMock()
    track.label_behavior = MagicMock()
    track.label_not_behavior = MagicMock()
    track.clear_labels = MagicMock()
    return track


def test_apply_behavior_label_binary_does_not_clear_competing() -> None:
    """Binary mode labels the current track and never iterates competing behaviors."""
    current_track = _make_track_mock()
    labels = SimpleNamespace(
        get_track_labels=MagicMock(return_value=current_track),
        iter_behavior_labels=MagicMock(),
    )

    central_widget_mode.apply_behavior_label(
        labels,
        ClassifierMode.BINARY,
        identity_str="0",
        current_behavior="Walk",
        start=10,
        end=20,
    )

    labels.iter_behavior_labels.assert_not_called()
    labels.get_track_labels.assert_called_once_with("0", "Walk")
    current_track.label_behavior.assert_called_once_with(10, 20)


def test_apply_behavior_label_multiclass_clears_competing_then_labels() -> None:
    """Multi-class mode clears non-current behavior tracks on the range, then labels current."""
    competing_track = _make_track_mock()
    current_track = _make_track_mock()
    labels = SimpleNamespace(
        get_track_labels=MagicMock(return_value=current_track),
        iter_behavior_labels=MagicMock(
            return_value=iter([("Walk", current_track), ("Run", competing_track)])
        ),
    )

    central_widget_mode.apply_behavior_label(
        labels,
        ClassifierMode.MULTICLASS,
        identity_str="0",
        current_behavior="Walk",
        start=10,
        end=20,
    )

    competing_track.clear_labels.assert_called_once_with(10, 20)
    current_track.clear_labels.assert_not_called()
    current_track.label_behavior.assert_called_once_with(10, 20)


# ---------------------------------------------------------------------------
# apply_not_behavior_label
# ---------------------------------------------------------------------------


def test_apply_not_behavior_label_binary_returns_current_behavior_false() -> None:
    """Binary mode labels the current track as not-behavior and reports (behavior, False)."""
    current_track = _make_track_mock()
    labels = SimpleNamespace(
        get_track_labels=MagicMock(return_value=current_track),
        iter_behavior_labels=MagicMock(),
    )

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
    labels.iter_behavior_labels.assert_not_called()
    current_track.label_not_behavior.assert_called_once_with(5, 15)


def test_apply_not_behavior_label_multiclass_returns_none_key_true() -> None:
    """Multi-class mode clears competing tracks and labels the NONE track as positive."""
    none_track = _make_track_mock()
    competing_track = _make_track_mock()
    labels = SimpleNamespace(
        get_track_labels=MagicMock(return_value=none_track),
        iter_behavior_labels=MagicMock(
            return_value=iter([(MULTICLASS_NONE_BEHAVIOR, none_track), ("Walk", competing_track)])
        ),
    )

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
    competing_track.clear_labels.assert_called_once_with(5, 15)
    none_track.clear_labels.assert_not_called()
    labels.get_track_labels.assert_called_once_with("0", MULTICLASS_NONE_BEHAVIOR)
    none_track.label_behavior.assert_called_once_with(5, 15)


# ---------------------------------------------------------------------------
# build_timeline_label_arrays
# ---------------------------------------------------------------------------


def test_build_timeline_label_arrays_multiclass_uses_merged_arrays() -> None:
    """Multi-class returns one merged label array per identity from VideoLabels."""
    expected = [
        np.array([0, 1, 2, 0], dtype=np.int16),
        np.array([1, 0, 0, 2], dtype=np.int16),
    ]
    labels = SimpleNamespace(
        build_multiclass_label_array=MagicMock(side_effect=expected),
        get_track_labels=MagicMock(),
    )

    result = central_widget_mode.build_timeline_label_arrays(
        labels,
        ClassifierMode.MULTICLASS,
        num_identities=2,
        current_behavior="Walk",
        behaviors=["Walk", "Run"],
    )

    labels.get_track_labels.assert_not_called()
    assert labels.build_multiclass_label_array.call_count == 2
    labels.build_multiclass_label_array.assert_any_call("0", ["Walk", "Run"])
    labels.build_multiclass_label_array.assert_any_call("1", ["Walk", "Run"])
    np.testing.assert_array_equal(result[0], expected[0])
    np.testing.assert_array_equal(result[1], expected[1])


def test_build_timeline_label_arrays_binary_uses_lut_indices(monkeypatch) -> None:
    """Binary returns one LUT-index array per identity for current_behavior only."""
    lut_call_args: list = []

    def fake_lut(track):
        lut_call_args.append(track)
        return np.array([1, 2, 3], dtype=np.int16)

    monkeypatch.setattr(
        "jabs.ui.main_window.central_widget_mode.track_labels_to_lut_indices",
        fake_lut,
    )

    track_a = MagicMock(name="track_a")
    track_b = MagicMock(name="track_b")
    labels = SimpleNamespace(
        get_track_labels=MagicMock(side_effect=[track_a, track_b]),
        build_multiclass_label_array=MagicMock(),
    )

    result = central_widget_mode.build_timeline_label_arrays(
        labels,
        ClassifierMode.BINARY,
        num_identities=2,
        current_behavior="Walk",
        behaviors=["Walk", "Run"],
    )

    labels.build_multiclass_label_array.assert_not_called()
    labels.get_track_labels.assert_any_call("0", "Walk")
    labels.get_track_labels.assert_any_call("1", "Walk")
    assert lut_call_args == [track_a, track_b]
    assert len(result) == 2
