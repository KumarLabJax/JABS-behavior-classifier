"""Tests for the prediction overlay spec handed to the exporter."""

import numpy as np
import pytest

try:
    from PySide6.QtGui import QColor  # noqa: F401

    from jabs.overlay_drawing import BACKGROUND_COLOR, BEHAVIOR_COLOR, NOT_BEHAVIOR_COLOR
    from jabs.video_export import PredictionOverlay

    SKIP_UI_TESTS = False
    SKIP_REASON = ""
except ImportError as e:
    SKIP_UI_TESTS = True
    SKIP_REASON = f"Qt/UI dependencies not available: {e}"

pytestmark = pytest.mark.skipif(SKIP_UI_TESTS, reason=SKIP_REASON)


def test_binary_caption_names_the_behavior_and_prediction_type() -> None:
    """An exported video has no menus, so the caption says what it is showing."""
    overlay = PredictionOverlay.for_binary(
        [np.array([1, 0, -1])], behavior="Grooming", postprocessed=True
    )

    assert overlay.caption == "Grooming predictions (post-processed)"


def test_binary_caption_says_raw_when_not_postprocessed() -> None:
    """Raw and post-processed predictions can differ substantially; say which is drawn."""
    overlay = PredictionOverlay.for_binary(
        [np.array([1, 0, -1])], behavior="Grooming", postprocessed=False
    )

    assert overlay.caption == "Grooming predictions (raw)"


def test_binary_legend_explains_every_marker_color() -> None:
    """The three binary marker colors mean nothing on their own, so all are named."""
    overlay = PredictionOverlay.for_binary(
        [np.array([1])], behavior="Grooming", postprocessed=False
    )

    assert overlay.legend == (
        ("behavior", BEHAVIOR_COLOR),
        ("not behavior", NOT_BEHAVIOR_COLOR),
        ("no prediction", BACKGROUND_COLOR),
    )
    assert overlay.color_lut is None


def test_multiclass_legend_pairs_names_with_their_table_colors() -> None:
    """Legend colors come from the same table the markers are drawn from."""
    lut = np.array(
        [[10, 10, 10, 255], [20, 20, 20, 255], [30, 30, 30, 255], [40, 40, 40, 255]],
        dtype=np.uint8,
    )

    overlay = PredictionOverlay.for_multiclass(
        [np.array([0, 1, 2, 3])],
        color_lut=lut,
        class_names=["None", "Grooming", "Rearing"],
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


def test_multiclass_rejects_a_name_list_that_does_not_cover_the_table() -> None:
    """Too few names would silently label a class with another class's color."""
    lut = np.array([[0, 0, 0, 255], [1, 1, 1, 255], [2, 2, 2, 255]], dtype=np.uint8)

    with pytest.raises(ValueError, match="class_names must name every color"):
        PredictionOverlay.for_multiclass([np.array([0])], color_lut=lut, class_names=["None"])


@pytest.mark.parametrize(
    ("identity", "frame", "expected"),
    [(0, 0, 1), (0, 2, -1), (1, 1, 5), (2, 0, None), (0, 3, None), (-1, 0, None), (0, -1, None)],
    ids=[
        "first-identity",
        "later-frame",
        "second-identity",
        "identity-past-the-end",
        "frame-past-the-end",
        "negative-identity",
        "negative-frame",
    ],
)
def test_label_value_returns_none_outside_the_arrays(
    identity: int, frame: int, expected: int | None
) -> None:
    """Predictions can be shorter than the video; a missing value draws no marker."""
    overlay = PredictionOverlay(labels=[np.array([1, 0, -1]), np.array([4, 5, 6])])

    assert overlay.label_value(identity, frame) == expected
