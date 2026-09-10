"""Tests for compositing the pose overlay onto a single frame."""

import numpy as np
import pytest

try:
    from PySide6.QtWidgets import QApplication  # noqa: F401

    from jabs.video_export import render_overlay_frame

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
