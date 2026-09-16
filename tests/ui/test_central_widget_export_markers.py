"""Tests for the label markers CentralWidget hands to the video export."""

from types import SimpleNamespace

import numpy as np
import pytest

from jabs.core.enums import ClassifierMode, PredictionType
from jabs.project import TrackLabels

try:
    from jabs.ui.main_window.central_widget import (
        _NO_LABELS_REASON,
        _NO_PREDICTIONS_REASON,
        _STALE_PREDICTIONS_REASON,
        CentralWidget,
    )

    SKIP_UI_TESTS = False
    SKIP_REASON = None
except ImportError as e:
    SKIP_UI_TESTS = True
    SKIP_REASON = f"Qt/UI dependencies not available: {e}"

pytestmark = pytest.mark.skipif(
    SKIP_UI_TESTS,
    reason=SKIP_REASON if SKIP_UI_TESTS else "",
)

_LABELS = [np.array([1, 1, 0, -1], dtype=np.int8)]
_UNLABELED = [np.full(4, TrackLabels.Label.NONE.value, dtype=np.int8)]
_RAW = [np.array([1, 1, 0, 0], dtype=np.int8)]
_POSTPROCESSED = [np.array([1, 1, 1, 0], dtype=np.int8)]
_MULTICLASS_LABELS = [np.array([0, 2, 3, 1], dtype=np.int16)]
_MULTICLASS_PREDICTIONS = [np.array([1, 2, 3, 0], dtype=np.int16)]
_LUT = np.array([[0, 0, 0, 255], [1, 1, 1, 255], [2, 2, 2, 255], [3, 3, 3, 255]], dtype=np.uint8)

# Sentinel for "present"; the export builders only check these for None.
_PRESENT = object()


def _widget(
    *,
    classifier_mode=None,
    manual_labels=_PRESENT,
    predictions=None,
    postprocessed: bool = False,
    color_lut=None,
    project=_PRESENT,
    loaded_video=_PRESENT,
    pose_est=_PRESENT,
    multiclass_class_names=_PRESENT,
) -> SimpleNamespace:
    """Stand-in exposing what the export's label marker builders read from self."""
    mode = ClassifierMode.BINARY if classifier_mode is None else classifier_mode
    multiclass = mode == ClassifierMode.MULTICLASS
    pose = SimpleNamespace(num_identities=1) if pose_est is _PRESENT else pose_est
    # By default the saved record's class list matches the project's behaviors.
    class_names = (
        ["None", "Grooming", "Rearing"]
        if multiclass_class_names is _PRESENT
        else multiclass_class_names
    )
    labels = (
        (_MULTICLASS_LABELS if multiclass else _LABELS)
        if (manual_labels is _PRESENT)
        else manual_labels
    )
    widget = SimpleNamespace(
        _project=(
            None
            if project is None
            else SimpleNamespace(settings_manager=SimpleNamespace(classifier_mode=mode))
        ),
        _loaded_video=loaded_video,
        _pose_est=pose,
        _predictions={0: _RAW[0]} if predictions is None else predictions,
        _multiclass_export_lut=color_lut,
        _controls=SimpleNamespace(behaviors=["Grooming", "Rearing"]),
        _multiclass_class_names=class_names,
        _showing_postprocessed_predictions=postprocessed,
        behavior="Grooming",
        _manual_overlay_labels=lambda multiclass: labels,
        _get_prediction_list=lambda: (_POSTPROCESSED if postprocessed else _RAW, []),
        _build_multiclass_overlay_labels=lambda: _MULTICLASS_PREDICTIONS,
    )
    # The public export methods gather from these two, which are what the states
    # below are really exercising.
    widget._export_manual_labels = lambda: CentralWidget._export_manual_labels(widget)
    widget._export_predicted_labels = lambda: CentralWidget._export_predicted_labels(widget)
    return widget


def _overlay(widget: SimpleNamespace, *, labels: bool = False, predictions: bool = False):
    """Build the export overlay the way the options dialog's choices would."""
    return CentralWidget.label_marker_overlay(
        widget, draw_labels=labels, draw_predictions=predictions
    )


def test_binary_overlay_carries_the_displayed_predictions() -> None:
    """The export draws exactly what the player's label overlay would."""
    overlay = _overlay(_widget(), predictions=True)

    assert overlay is not None
    assert overlay.predicted_labels is _RAW
    assert overlay.manual_labels is None
    assert overlay.color_lut is None
    assert overlay.caption == "Grooming predictions (raw)"


def test_binary_overlay_follows_the_postprocessed_choice() -> None:
    """Switching the timeline to post-processed predictions exports those instead."""
    overlay = _overlay(_widget(postprocessed=True), predictions=True)

    assert overlay.predicted_labels is _POSTPROCESSED
    assert overlay.caption == "Grooming predictions (post-processed)"


def test_binary_overlay_can_carry_the_manual_labels_alone() -> None:
    """An unclassified video can still export what has been labeled by hand."""
    overlay = _overlay(_widget(predictions={}), labels=True)

    assert overlay is not None
    assert overlay.manual_labels is _LABELS
    assert overlay.predicted_labels is None
    assert overlay.caption == "Grooming labels"


def test_binary_overlay_can_carry_both_sources() -> None:
    """Ticking both boxes exports a marker for each, the way the player draws them."""
    overlay = _overlay(_widget(), labels=True, predictions=True)

    assert overlay.manual_labels is _LABELS
    assert overlay.predicted_labels is _RAW
    assert overlay.marker_count == 2
    assert overlay.caption == "Grooming labels (left) and predictions (raw, right)"


def test_nothing_selected_builds_no_overlay() -> None:
    """With both boxes unticked there is nothing for the export to draw."""
    assert _overlay(_widget()) is None


def test_an_unticked_source_is_left_out_even_when_available() -> None:
    """The choice decides what is drawn, not what happens to be available."""
    overlay = _overlay(_widget(), labels=True)

    assert overlay.predicted_labels is None


def test_multiclass_overlay_uses_the_project_color_table() -> None:
    """Multi-class exports carry the same colors and class names as the timeline."""
    overlay = _overlay(
        _widget(classifier_mode=ClassifierMode.MULTICLASS, color_lut=_LUT), predictions=True
    )

    assert overlay is not None
    assert overlay.color_lut is _LUT
    assert [name for name, _ in overlay.legend] == [
        "None",
        "Grooming",
        "Rearing",
        "no prediction",
    ]


def test_multiclass_overlay_can_carry_both_sources() -> None:
    """Multi-class labels and predictions both index the project's color table."""
    overlay = _overlay(
        _widget(classifier_mode=ClassifierMode.MULTICLASS, color_lut=_LUT),
        labels=True,
        predictions=True,
    )

    assert overlay.manual_labels is _MULTICLASS_LABELS
    assert overlay.predicted_labels is _MULTICLASS_PREDICTIONS
    assert overlay.color_lut is _LUT
    assert overlay.caption == "Multi-class labels (left) and predictions (raw, right)"


@pytest.mark.parametrize(
    "widget_kwargs",
    [
        {"predictions": {}},
        {"project": None},
        {"loaded_video": None},
        {"pose_est": None},
        {"classifier_mode": ClassifierMode.MULTICLASS, "color_lut": None},
    ],
    ids=[
        "nothing-classified",
        "no-project",
        "no-video",
        "no-pose",
        "multiclass-without-a-color-table",
    ],
)
def test_predictions_are_unavailable_when_there_is_nothing_to_draw(widget_kwargs: dict) -> None:
    """Every state with no drawable predictions is reported, not exported blank."""
    widget = _widget(**widget_kwargs)

    _labels_reason, predictions_reason = CentralWidget.label_markers_unavailable(widget)

    assert predictions_reason == _NO_PREDICTIONS_REASON
    assert _overlay(widget, predictions=True) is None


@pytest.mark.parametrize(
    "widget_kwargs",
    [
        {"manual_labels": None},
        {"project": None},
        {"loaded_video": None},
        {"pose_est": None},
        {"classifier_mode": ClassifierMode.MULTICLASS, "color_lut": None},
    ],
    ids=[
        "no-labels-loaded",
        "no-project",
        "no-video",
        "no-pose",
        "multiclass-without-a-color-table",
    ],
)
def test_labels_are_unavailable_when_there_is_nothing_to_draw(widget_kwargs: dict) -> None:
    """Without label arrays to hand over there is nothing for the export to draw."""
    widget = _widget(**widget_kwargs)

    labels_reason, _predictions_reason = CentralWidget.label_markers_unavailable(widget)

    assert labels_reason == _NO_LABELS_REASON
    assert _overlay(widget, labels=True) is None


@pytest.mark.parametrize(
    ("classifier_mode", "color_lut", "manual_labels"),
    [
        (ClassifierMode.BINARY, None, _UNLABELED),
        (ClassifierMode.MULTICLASS, _LUT, [np.zeros(4, dtype=np.int16)]),
    ],
    ids=["binary", "multiclass"],
)
def test_the_labels_are_offered_even_when_nothing_is_labeled_yet(
    classifier_mode, color_lut, manual_labels
) -> None:
    """Labels are always there to be drawn, so the export always offers them."""
    widget = _widget(
        classifier_mode=classifier_mode, color_lut=color_lut, manual_labels=manual_labels
    )

    labels_reason, _predictions_reason = CentralWidget.label_markers_unavailable(widget)

    assert labels_reason is None
    assert _overlay(widget, labels=True) is not None


def test_both_sources_are_available_in_the_ordinary_case() -> None:
    """A labeled and classified video offers a choice between the two and both."""
    assert CentralWidget.label_markers_unavailable(_widget()) == (None, None)


@pytest.mark.parametrize(
    ("prediction_type", "keys_match", "expected"),
    [
        (PredictionType.POSTPROCESSED, True, True),
        (PredictionType.POSTPROCESSED, False, False),
        (PredictionType.RAW, True, False),
    ],
    ids=["requested-and-available", "requested-but-incomplete", "not-requested"],
)
def test_postprocessed_flag_requires_the_data_to_exist(
    prediction_type, keys_match: bool, expected: bool
) -> None:
    """Asking for post-processed predictions is not enough; they have to be there.

    The display silently falls back to raw predictions when they are not, and the
    caption has to fall back with it rather than mislabeling the export.
    """
    widget = SimpleNamespace(
        prediction_type=prediction_type,
        _predictions={0: _RAW[0]},
        _predictions_postprocessed={0: _POSTPROCESSED[0]} if keys_match else {},
    )

    assert CentralWidget._showing_postprocessed_predictions.fget(widget) is expected


@pytest.mark.parametrize(
    ("project", "classifier_mode", "timeline_lut", "expected"),
    [
        (_PRESENT, ClassifierMode.MULTICLASS, _LUT, _LUT),
        (_PRESENT, ClassifierMode.BINARY, _LUT, None),
        (None, ClassifierMode.MULTICLASS, _LUT, None),
        (_PRESENT, ClassifierMode.MULTICLASS, None, None),
    ],
    ids=["multiclass", "binary", "no-project", "table-not-built-yet"],
)
def test_the_export_color_table_is_the_timelines_in_multiclass_mode_only(
    project, classifier_mode, timeline_lut, expected
) -> None:
    """The table doubles as the multi-class discriminator, so binary mode has none."""
    widget = SimpleNamespace(
        _project=(
            None
            if project is None
            else SimpleNamespace(settings_manager=SimpleNamespace(classifier_mode=classifier_mode))
        ),
        _jabs_timeline=SimpleNamespace(multiclass_color_lut=timeline_lut),
    )

    assert CentralWidget._multiclass_export_lut.fget(widget) is expected


@pytest.mark.parametrize(
    "stored_names",
    [
        ["None", "Rearing", "Grooming"],
        ["None", "Grooming"],
        ["None", "Grooming", "Rearing", "Locomotion"],
        None,
    ],
    ids=["reordered", "behavior-removed", "behavior-added", "names-unknown"],
)
def test_multiclass_predictions_from_a_record_that_does_not_match_are_refused(
    stored_names: list[str] | None,
) -> None:
    """Stale class indices would be burned in under another behavior's name and color.

    The prediction values are class indices from the saved record, while the color
    table and legend come from the project's current behavior list. If the two
    disagree, there is no safe way to label the markers, so nothing is exported.
    """
    widget = _widget(
        classifier_mode=ClassifierMode.MULTICLASS,
        color_lut=_LUT,
        multiclass_class_names=stored_names,
    )

    labels_reason, predictions_reason = CentralWidget.label_markers_unavailable(widget)

    # A stale record is not a missing one: telling the user to classify a video they
    # already classified would send them looking for a problem that is not there.
    assert predictions_reason == _STALE_PREDICTIONS_REASON
    assert _overlay(widget, predictions=True) is None
    # The labels are built from the project's current behavior list, so they cannot go
    # stale with the record and are still exportable on their own.
    assert labels_reason is None
    assert _overlay(widget, labels=True) is not None


def test_multiclass_predictions_from_a_matching_record_are_accepted() -> None:
    """The ordinary case, where nothing has changed since the video was classified."""
    overlay = _overlay(
        _widget(
            classifier_mode=ClassifierMode.MULTICLASS,
            color_lut=_LUT,
            multiclass_class_names=["None", "Grooming", "Rearing"],
        ),
        predictions=True,
    )

    assert overlay is not None
    assert [name for name, _ in overlay.legend] == [
        "None",
        "Grooming",
        "Rearing",
        "no prediction",
    ]
