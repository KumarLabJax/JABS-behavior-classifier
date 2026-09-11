"""Tests for compositing the JABS overlays onto a single frame."""

import numpy as np
import pytest

try:
    from PySide6.QtWidgets import QApplication  # noqa: F401

    from jabs.overlay_drawing import BEHAVIOR_COLOR, NOT_BEHAVIOR_COLOR
    from jabs.video_export import PredictionOverlay, render_overlay_frame

    SKIP_UI_TESTS = False
    SKIP_REASON = ""
except ImportError as e:
    SKIP_UI_TESTS = True
    SKIP_REASON = f"Qt/UI dependencies not available: {e}"

pytestmark = pytest.mark.skipif(SKIP_UI_TESTS, reason=SKIP_REASON)

from ._fakes import BACKGROUND, HEIGHT, WIDTH, StubPose


def test_render_overlay_frame_draws_onto_a_copy(blank_frame: np.ndarray) -> None:
    """The overlay is drawn on a copy; the caller's frame buffer is left alone."""
    result = render_overlay_frame(blank_frame, StubPose(), 0, draw_segmentation=False)

    assert result.shape == blank_frame.shape
    assert result.dtype == np.uint8
    assert (blank_frame == BACKGROUND).all(), "input frame was mutated"
    assert (result != blank_frame).any(), "no overlay was drawn"


def test_render_overlay_frame_varies_with_frame_index(blank_frame: np.ndarray) -> None:
    """Different frames render differently, so the pose actually tracks the frame."""
    first = render_overlay_frame(blank_frame, StubPose(), 0, draw_segmentation=False)
    later = render_overlay_frame(blank_frame, StubPose(), 5, draw_segmentation=False)

    assert (first != later).any()


def test_render_overlay_frame_draws_every_identity(blank_frame: np.ndarray) -> None:
    """Two identities mark more pixels than one; an export has no active identity."""
    one = render_overlay_frame(blank_frame, StubPose(identities=[0]), 0, draw_segmentation=False)
    two = render_overlay_frame(
        blank_frame, StubPose(identities=[0, 1]), 0, draw_segmentation=False
    )

    assert int((two != blank_frame).any(axis=2).sum()) > int(
        (one != blank_frame).any(axis=2).sum()
    )


def test_render_overlay_frame_accepts_non_uint8_input() -> None:
    """A non-uint8 source frame is coerced rather than rejected."""
    frame = np.full((HEIGHT, WIDTH, 3), float(BACKGROUND), dtype=np.float64)

    result = render_overlay_frame(frame, StubPose(), 0, draw_segmentation=False)

    assert result.dtype == np.uint8


def test_segmentation_skipped_when_the_pose_file_has_none(blank_frame: np.ndarray) -> None:
    """A v6+ pose file without segmentation data renders pose only, without error.

    Segmentation became optional in v6, so a pose version check alone is not enough:
    asking for contours a file does not have must be a no-op, not a failure.
    """
    pose = StubPose(has_segmentation=False)

    result = render_overlay_frame(blank_frame, pose, 0, draw_segmentation=True)

    assert (result != blank_frame).any(), "pose should still be drawn"
    assert pose.segmentation_calls == [], "should not query segmentation it does not have"


def test_segmentation_queried_when_the_pose_file_has_it(blank_frame: np.ndarray) -> None:
    """When the file does carry segmentation, it is requested for every identity."""
    pose = StubPose(identities=[0, 1], has_segmentation=True)

    render_overlay_frame(blank_frame, pose, 3, draw_segmentation=True)

    assert pose.segmentation_calls == [(3, 0), (3, 1)]


def test_segmentation_not_queried_when_switched_off(blank_frame: np.ndarray) -> None:
    """--no-segmentation / unchecked box skips it even when the data exists."""
    pose = StubPose(has_segmentation=True)

    render_overlay_frame(blank_frame, pose, 0, draw_segmentation=False)

    assert pose.segmentation_calls == []


def test_no_overlays_returns_an_unmodified_copy(blank_frame: np.ndarray) -> None:
    """With every overlay switched off the frame is copied through untouched."""
    result = render_overlay_frame(
        blank_frame, StubPose(), 0, draw_pose=False, draw_segmentation=False
    )

    assert (result == blank_frame).all()
    assert result is not blank_frame


def test_pose_can_be_switched_off_while_predictions_are_drawn(blank_frame: np.ndarray) -> None:
    """Prediction markers do not depend on the skeleton being drawn."""
    overlay = PredictionOverlay(labels=[np.ones(10, dtype=np.int8)])

    with_pose = render_overlay_frame(
        blank_frame, StubPose(), 0, draw_segmentation=False, prediction_overlay=overlay
    )
    without_pose = render_overlay_frame(
        blank_frame,
        StubPose(),
        0,
        draw_pose=False,
        draw_segmentation=False,
        prediction_overlay=overlay,
    )

    assert (without_pose != blank_frame).any(), "marker should still be drawn"
    changed_with_pose = int((with_pose != blank_frame).any(axis=2).sum())
    changed_without_pose = int((without_pose != blank_frame).any(axis=2).sum())
    assert changed_without_pose < changed_with_pose


def test_prediction_marker_uses_the_label_color(blank_frame: np.ndarray) -> None:
    """The marker beside an identity is colored by that identity's prediction."""
    pose = StubPose()
    behavior = render_overlay_frame(
        blank_frame,
        pose,
        0,
        draw_pose=False,
        draw_segmentation=False,
        prediction_overlay=PredictionOverlay(labels=[np.ones(10, dtype=np.int8)]),
    )
    not_behavior = render_overlay_frame(
        blank_frame,
        pose,
        0,
        draw_pose=False,
        draw_segmentation=False,
        prediction_overlay=PredictionOverlay(labels=[np.zeros(10, dtype=np.int8)]),
    )

    behavior_pixels = _marker_colors(behavior, blank_frame)
    not_behavior_pixels = _marker_colors(not_behavior, blank_frame)
    assert _bgr(BEHAVIOR_COLOR) in behavior_pixels
    assert _bgr(NOT_BEHAVIOR_COLOR) in not_behavior_pixels


def test_prediction_marker_follows_the_frame(blank_frame: np.ndarray) -> None:
    """The marker tracks the identity's centroid rather than sitting still."""
    overlay = PredictionOverlay(labels=[np.ones(10, dtype=np.int8)])

    first = render_overlay_frame(
        blank_frame,
        StubPose(),
        0,
        draw_pose=False,
        draw_segmentation=False,
        prediction_overlay=overlay,
    )
    later = render_overlay_frame(
        blank_frame,
        StubPose(),
        5,
        draw_pose=False,
        draw_segmentation=False,
        prediction_overlay=overlay,
    )

    assert (first != later).any()


def test_identity_without_a_convex_hull_gets_no_marker(blank_frame: np.ndarray) -> None:
    """An identity with no pose on this frame has nowhere for a marker to sit."""
    pose = StubPose(hulls_present=False)

    result = render_overlay_frame(
        blank_frame,
        pose,
        0,
        draw_pose=False,
        draw_segmentation=False,
        prediction_overlay=PredictionOverlay(labels=[np.ones(10, dtype=np.int8)]),
    )

    assert (result == blank_frame).all()


def test_frame_beyond_the_predictions_gets_no_marker(blank_frame: np.ndarray) -> None:
    """Predictions shorter than the video leave the remaining frames unmarked."""
    overlay = PredictionOverlay(labels=[np.ones(3, dtype=np.int8)])

    within = render_overlay_frame(
        blank_frame,
        StubPose(),
        2,
        draw_pose=False,
        draw_segmentation=False,
        prediction_overlay=overlay,
    )
    beyond = render_overlay_frame(
        blank_frame,
        StubPose(),
        5,
        draw_pose=False,
        draw_segmentation=False,
        prediction_overlay=overlay,
    )

    assert (within != blank_frame).any()
    assert (beyond == blank_frame).all()


def test_multiclass_marker_uses_the_color_table(blank_frame: np.ndarray) -> None:
    """Multi-class labels index the project's color table."""
    lut = np.array([[0, 0, 0, 255], [17, 85, 153, 255]], dtype=np.uint8)
    overlay = PredictionOverlay(labels=[np.ones(10, dtype=np.int8)], color_lut=lut)

    result = render_overlay_frame(
        blank_frame,
        StubPose(),
        0,
        draw_pose=False,
        draw_segmentation=False,
        prediction_overlay=overlay,
    )

    assert (153, 85, 17) in _marker_colors(result, blank_frame), "BGR of the table color"


def _bgr(color) -> tuple[int, int, int]:
    """Convert a QColor to the BGR tuple the rendered frame holds."""
    return (color.blue(), color.green(), color.red())


def _marker_colors(rendered: np.ndarray, source: np.ndarray) -> set[tuple[int, int, int]]:
    """Return the distinct colors the overlay painted onto the frame."""
    changed = (rendered != source).any(axis=2)
    return {tuple(int(c) for c in pixel) for pixel in rendered[changed]}


def test_segmentation_contours_are_drawn(blank_frame: np.ndarray) -> None:
    """Contours reach the rendered frame, in the active color an export draws them in."""
    pose = StubPose(has_segmentation=True, contours=True)

    result = render_overlay_frame(blank_frame, pose, 0, draw_pose=False)

    changed = _marker_colors(result, blank_frame)
    assert changed, "no contour was drawn"
    # Frames are BGR, so the red the exporter draws contours in is the last channel.
    assert any(red > green and red > blue for blue, green, red in changed)


def test_segmentation_contours_can_be_switched_off(blank_frame: np.ndarray) -> None:
    """An unticked segmentation box leaves the frame alone, data or no data."""
    pose = StubPose(has_segmentation=True, contours=True)

    result = render_overlay_frame(blank_frame, pose, 0, draw_pose=False, draw_segmentation=False)

    assert (result == blank_frame).all()
    assert pose.segmentation_calls == []
