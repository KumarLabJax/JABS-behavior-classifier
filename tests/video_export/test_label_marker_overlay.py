"""Tests for the label marker overlay spec handed to the exporter."""

import numpy as np
import pytest

try:
    from PySide6.QtGui import QColor  # noqa: F401

    from jabs.overlay_drawing import BACKGROUND_COLOR, BEHAVIOR_COLOR, NOT_BEHAVIOR_COLOR
    from jabs.video_export import LabelMarkerOverlay

    SKIP_UI_TESTS = False
    SKIP_REASON = ""
except ImportError as e:
    SKIP_UI_TESTS = True
    SKIP_REASON = f"Qt/UI dependencies not available: {e}"

pytestmark = pytest.mark.skipif(SKIP_UI_TESTS, reason=SKIP_REASON)

_LABELS = [np.array([1, 0, -1])]
_PREDICTIONS = [np.array([0, 1, 1])]
_MULTICLASS_LUT = np.array(
    [[10, 10, 10, 255], [20, 20, 20, 255], [30, 30, 30, 255], [40, 40, 40, 255]],
    dtype=np.uint8,
)
_CLASS_NAMES = ["None", "Grooming", "Rearing"]


def test_binary_caption_names_the_behavior_and_prediction_type() -> None:
    """An exported video has no menus, so the caption says what it is showing."""
    overlay = LabelMarkerOverlay.for_binary(
        behavior="Grooming", predicted_labels=_PREDICTIONS, postprocessed=True
    )

    assert overlay.caption == "Grooming predictions (post-processed)"


def test_binary_caption_says_raw_when_not_postprocessed() -> None:
    """Raw and post-processed predictions can differ substantially; say which is drawn."""
    overlay = LabelMarkerOverlay.for_binary(
        behavior="Grooming", predicted_labels=_PREDICTIONS, postprocessed=False
    )

    assert overlay.caption == "Grooming predictions (raw)"


def test_binary_caption_for_labels_alone_says_nothing_about_predictions() -> None:
    """Exporting labels only must not claim a raw or post-processed prediction type."""
    overlay = LabelMarkerOverlay.for_binary(
        behavior="Grooming", manual_labels=_LABELS, postprocessed=True
    )

    assert overlay.caption == "Grooming labels"


def test_binary_caption_for_both_says_which_marker_is_which() -> None:
    """Side by side, only the order tells the two markers apart, so the caption says it."""
    overlay = LabelMarkerOverlay.for_binary(
        behavior="Grooming",
        manual_labels=_LABELS,
        predicted_labels=_PREDICTIONS,
        postprocessed=False,
    )

    assert overlay.caption == "Grooming labels (left) and predictions (raw, right)"


def test_binary_legend_explains_every_marker_color() -> None:
    """The three binary marker colors mean nothing on their own, so all are named."""
    overlay = LabelMarkerOverlay.for_binary(behavior="Grooming", predicted_labels=_PREDICTIONS)

    assert overlay.legend == (
        ("behavior", BEHAVIOR_COLOR),
        ("not behavior", NOT_BEHAVIOR_COLOR),
        ("no prediction", BACKGROUND_COLOR),
    )
    assert overlay.color_lut is None


@pytest.mark.parametrize(
    ("sources", "expected"),
    [
        ({"manual_labels": _LABELS}, "no label"),
        ({"predicted_labels": _PREDICTIONS}, "no prediction"),
        (
            {"manual_labels": _LABELS, "predicted_labels": _PREDICTIONS},
            "no label or prediction",
        ),
    ],
    ids=["labels", "predictions", "both"],
)
def test_the_gray_swatch_is_named_for_the_markers_being_drawn(
    sources: dict, expected: str
) -> None:
    """Gray means unlabeled on one marker and unpredicted on the other."""
    overlay = LabelMarkerOverlay.for_binary(behavior="Grooming", **sources)

    assert overlay.legend[-1] == (expected, BACKGROUND_COLOR)


def test_multiclass_legend_pairs_names_with_their_table_colors() -> None:
    """Legend colors come from the same table the markers are drawn from."""
    overlay = LabelMarkerOverlay.for_multiclass(
        color_lut=_MULTICLASS_LUT,
        class_names=_CLASS_NAMES,
        predicted_labels=[np.array([0, 1, 2, 3])],
    )

    assert overlay.caption == "Multi-class predictions (raw)"
    assert [name for name, _ in overlay.legend] == [
        "None",
        "Grooming",
        "Rearing",
        "no prediction",
    ]
    # Table index 0 is the no-prediction color and comes last in the legend, so each
    # class has to be paired with the color one index further along.
    assert [color.getRgb() for _, color in overlay.legend] == [
        (20, 20, 20, 255),
        (30, 30, 30, 255),
        (40, 40, 40, 255),
        (10, 10, 10, 255),
    ]


def test_multiclass_labels_and_predictions_share_one_legend() -> None:
    """Both multi-class sources index the same table, so one legend covers the pair."""
    overlay = LabelMarkerOverlay.for_multiclass(
        color_lut=_MULTICLASS_LUT,
        class_names=_CLASS_NAMES,
        manual_labels=[np.array([0, 2, 3, 1])],
        predicted_labels=[np.array([0, 1, 2, 3])],
    )

    assert overlay.caption == "Multi-class labels (left) and predictions (raw, right)"
    assert [name for name, _ in overlay.legend] == [
        "None",
        "Grooming",
        "Rearing",
        "no label or prediction",
    ]
    assert overlay.color_lut is _MULTICLASS_LUT


def test_multiclass_rejects_a_name_list_that_does_not_cover_the_table() -> None:
    """Too few names would silently label a class with another class's color."""
    lut = np.array([[0, 0, 0, 255], [1, 1, 1, 255], [2, 2, 2, 255]], dtype=np.uint8)

    with pytest.raises(ValueError, match="class_names must name every color"):
        LabelMarkerOverlay.for_multiclass(
            color_lut=lut, class_names=["None"], predicted_labels=_PREDICTIONS
        )


@pytest.mark.parametrize(
    "sources",
    [{}, {"manual_labels": []}, {"manual_labels": None, "predicted_labels": []}],
    ids=["neither", "empty-labels", "empty-both"],
)
def test_an_overlay_with_no_markers_is_rejected(sources: dict) -> None:
    """A caption and legend on every frame for markers that never appear is worse than none."""
    with pytest.raises(ValueError, match="needs manual labels, predictions, or both"):
        LabelMarkerOverlay(**sources)


def test_an_empty_source_is_not_named_in_the_caption_or_legend() -> None:
    """A source with no identities draws nothing, so the caption must not promise it."""
    overlay = LabelMarkerOverlay.for_binary(
        behavior="Grooming", manual_labels=[], predicted_labels=_PREDICTIONS
    )

    assert overlay.marker_count == 1
    assert overlay.caption == "Grooming predictions (raw)"
    assert overlay.legend[-1][0] == "no prediction"


def test_one_source_draws_one_marker() -> None:
    """Predictions alone put a single marker beside each animal."""
    overlay = LabelMarkerOverlay(predicted_labels=_PREDICTIONS)

    assert overlay.marker_count == 1
    assert overlay.marker_values(0, 1) == (1,)


def test_both_sources_draw_the_label_first_then_the_prediction() -> None:
    """The marker order is the only thing that says which source a marker came from."""
    overlay = LabelMarkerOverlay(manual_labels=_LABELS, predicted_labels=_PREDICTIONS)

    assert overlay.marker_count == 2
    assert overlay.marker_values(0, 0) == (1, 0)
    assert overlay.marker_values(0, 1) == (0, 1)


@pytest.mark.parametrize(
    ("identity", "frame", "expected"),
    [
        (0, 0, (1, 4)),
        (0, 2, (-1, 6)),
        (1, 1, (5, None)),
        (2, 0, (None, None)),
        (0, 3, (None, None)),
        (-1, 0, (None, None)),
        (0, -1, (None, None)),
    ],
    ids=[
        "first-identity",
        "later-frame",
        "one-source-short-an-identity",
        "identity-past-the-end",
        "frame-past-the-end",
        "negative-identity",
        "negative-frame",
    ],
)
def test_marker_values_are_none_outside_the_arrays(
    identity: int, frame: int, expected: tuple
) -> None:
    """Values can be shorter than the video; a missing one draws no marker."""
    overlay = LabelMarkerOverlay(
        manual_labels=[np.array([1, 0, -1]), np.array([4, 5, 6])],
        predicted_labels=[np.array([4, 5, 6])],
    )

    assert overlay.marker_values(identity, frame) == expected
