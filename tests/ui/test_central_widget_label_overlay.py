"""Tests for the label overlay values CentralWidget pushes to the player widget."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from jabs.core.enums import ClassifierMode

try:
    from jabs.ui.main_window.central_widget import CentralWidget
    from jabs.ui.player_widget import PlayerWidget

    SKIP_UI_TESTS = False
    SKIP_REASON = ""
except ImportError as e:
    SKIP_UI_TESTS = True
    SKIP_REASON = f"Qt/UI dependencies not available: {e}"

pytestmark = pytest.mark.skipif(SKIP_UI_TESTS, reason=SKIP_REASON)

_MANUAL = [np.array([1, 1, 0, 0], dtype=np.int8)]
_PREDICTED = [np.array([0, 1, 1, 0], dtype=np.int8)]
_LUT = np.array([[0, 0, 0, 0], [255, 0, 0, 255]], dtype=np.uint8)

# Sentinel for "present", so a test can pass None for a video's labels.
_PRESENT = object()


def _widget(
    mode,
    *,
    classifier_mode: ClassifierMode = ClassifierMode.BINARY,
    color_lut: np.ndarray | None = None,
    manual_labels: list[np.ndarray] | None = _MANUAL,
    predicted_labels: list[np.ndarray] | None = _PREDICTED,
    populated: bool = False,
) -> SimpleNamespace:
    """Stand-in exposing what the label overlay refresh reads from self."""
    widget = SimpleNamespace(
        _label_overlay_mode=mode,
        _label_overlay_populated=populated,
        _project=SimpleNamespace(
            settings_manager=SimpleNamespace(classifier_mode=classifier_mode)
        ),
        _jabs_timeline=SimpleNamespace(multiclass_color_lut=color_lut),
        _player_widget=SimpleNamespace(set_labels=MagicMock(), set_label_color_lut=MagicMock()),
        _manual_overlay_labels=MagicMock(return_value=manual_labels),
        _prediction_overlay_labels=MagicMock(return_value=predicted_labels),
    )
    widget._clear_label_overlay = lambda: CentralWidget._clear_label_overlay(widget)
    return widget


def test_the_overlay_shows_labels_alone() -> None:
    """View > Label Overlay > Labels sends the manual labels and no predictions."""
    widget = _widget(PlayerWidget.LabelOverlayMode.LABEL)

    CentralWidget._refresh_label_overlay(widget)

    widget._player_widget.set_labels.assert_called_once_with(_MANUAL, None)
    widget._prediction_overlay_labels.assert_not_called()


def test_the_overlay_shows_predictions_alone() -> None:
    """View > Label Overlay > Predictions sends the predictions and no manual labels."""
    widget = _widget(PlayerWidget.LabelOverlayMode.PREDICTION)

    CentralWidget._refresh_label_overlay(widget)

    widget._player_widget.set_labels.assert_called_once_with(None, _PREDICTED)
    widget._manual_overlay_labels.assert_not_called()


def test_the_overlay_shows_both_at_once() -> None:
    """View > Label Overlay > Labels and Predictions sends both, for markers side by side."""
    widget = _widget(PlayerWidget.LabelOverlayMode.BOTH)

    CentralWidget._refresh_label_overlay(widget)

    widget._player_widget.set_labels.assert_called_once_with(_MANUAL, _PREDICTED)


def test_both_sources_share_the_multiclass_color_table() -> None:
    """Labels and predictions are LUT indices into the same table in multi-class mode."""
    widget = _widget(
        PlayerWidget.LabelOverlayMode.BOTH,
        classifier_mode=ClassifierMode.MULTICLASS,
        color_lut=_LUT,
    )

    CentralWidget._refresh_label_overlay(widget)

    widget._player_widget.set_label_color_lut.assert_called_once_with(_LUT)
    widget._manual_overlay_labels.assert_called_once_with(True)
    widget._prediction_overlay_labels.assert_called_once_with(True)


def test_multiclass_without_a_color_table_shows_nothing() -> None:
    """A multi-class label value cannot be colored without the timeline's table."""
    widget = _widget(
        PlayerWidget.LabelOverlayMode.BOTH,
        classifier_mode=ClassifierMode.MULTICLASS,
        color_lut=None,
        populated=True,
    )

    CentralWidget._refresh_label_overlay(widget)

    widget._player_widget.set_labels.assert_called_once_with(None, None)
    widget._player_widget.set_label_color_lut.assert_called_once_with(None)


def test_a_mode_with_nothing_to_show_clears_the_overlay() -> None:
    """Asking for both in an unclassified, unlabeled video leaves the overlay empty."""
    widget = _widget(
        PlayerWidget.LabelOverlayMode.BOTH,
        manual_labels=None,
        predicted_labels=None,
        populated=True,
    )

    CentralWidget._refresh_label_overlay(widget)

    widget._player_widget.set_labels.assert_called_once_with(None, None)
    assert widget._label_overlay_populated is False


def test_switching_the_overlay_off_clears_it() -> None:
    """The markers come off the frame when the overlay is switched off."""
    widget = _widget(PlayerWidget.LabelOverlayMode.NONE, populated=True)

    CentralWidget._refresh_label_overlay(widget)

    widget._player_widget.set_labels.assert_called_once_with(None, None)
    widget._player_widget.set_label_color_lut.assert_called_once_with(None)
    assert widget._label_overlay_populated is False


def test_an_already_empty_overlay_is_left_alone() -> None:
    """Labeling with the overlay off must not reload the displayed frame on every edit."""
    widget = _widget(PlayerWidget.LabelOverlayMode.NONE, populated=False)

    CentralWidget._refresh_label_overlay(widget)

    widget._player_widget.set_labels.assert_not_called()
    widget._player_widget.set_label_color_lut.assert_not_called()


def _label_source_widget(
    *,
    classifier_mode: ClassifierMode = ClassifierMode.BINARY,
    labels=_PRESENT,
    label_list: list | None = None,
    prediction_list: list | None = None,
) -> SimpleNamespace:
    """Stand-in exposing what the per-source array builders read from self."""
    video_labels = (
        SimpleNamespace(
            build_multiclass_label_array=lambda identity, behaviors: np.array(
                [int(identity)], dtype=np.int8
            )
        )
        if labels is _PRESENT
        else labels
    )
    return SimpleNamespace(
        _project=SimpleNamespace(
            settings_manager=SimpleNamespace(classifier_mode=classifier_mode)
        ),
        _labels=video_labels,
        _pose_est=SimpleNamespace(num_identities=2),
        _controls=SimpleNamespace(behaviors=["Grooming", "Rearing"]),
        _prediction_list=prediction_list,
        _get_label_list=lambda: label_list or [],
        _build_multiclass_overlay_labels=lambda: _PREDICTED,
    )


def test_binary_manual_labels_come_from_the_selected_behavior() -> None:
    """The manual marker follows the behavior the label track is showing."""
    track = SimpleNamespace(get_labels=lambda: _MANUAL[0])

    labels = CentralWidget._manual_overlay_labels(
        _label_source_widget(label_list=[track]), multiclass=False
    )

    assert len(labels) == 1
    assert labels[0] is _MANUAL[0]


def test_no_selected_behavior_means_no_manual_labels() -> None:
    """Before a behavior and identity are picked there is nothing to draw."""
    assert CentralWidget._manual_overlay_labels(_label_source_widget(), multiclass=False) is None


def test_multiclass_manual_labels_cover_every_identity() -> None:
    """Multi-class labels are merged across behaviors, one array per identity."""
    labels = CentralWidget._manual_overlay_labels(
        _label_source_widget(classifier_mode=ClassifierMode.MULTICLASS), multiclass=True
    )

    assert [int(arr[0]) for arr in labels] == [0, 1]


def test_an_unlabeled_video_has_no_manual_labels() -> None:
    """A video with no annotations loaded has nothing to draw."""
    assert (
        CentralWidget._manual_overlay_labels(_label_source_widget(labels=None), multiclass=False)
        is None
    )


def test_binary_predictions_come_from_the_displayed_prediction_list() -> None:
    """The prediction marker shows what the timeline is showing."""
    widget = _label_source_widget(prediction_list=_PREDICTED)

    assert CentralWidget._prediction_overlay_labels(widget, multiclass=False) is _PREDICTED


def test_an_unclassified_video_has_no_predictions() -> None:
    """Nothing is drawn for a video that has not been classified."""
    widget = _label_source_widget(prediction_list=None)

    assert CentralWidget._prediction_overlay_labels(widget, multiclass=False) is None
