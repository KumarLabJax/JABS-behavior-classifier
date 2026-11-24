"""Tests for the shared bounding box drawing used by the exports."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jabs.core.enums import PoseCapability
from jabs.pose_estimation import open_pose_file

from .conftest import BBOX_SOURCE, NO_BBOX_SOURCE, POSE_KINDS, copy_pose

try:
    from PySide6 import QtGui
    from PySide6.QtWidgets import QApplication

    from jabs.overlay_drawing import (
        BOUNDING_BOX_COLOR,
        draw_identity_bounding_box,
        native_bounding_box_sizes,
        pose_has_bounding_boxes,
    )

    SKIP_UI_TESTS = False
    SKIP_REASON = ""
except ImportError as e:
    SKIP_UI_TESTS = True
    SKIP_REASON = f"Qt/UI dependencies not available: {e}"

pytestmark = pytest.mark.skipif(SKIP_UI_TESTS, reason=SKIP_REASON)

WIDTH, HEIGHT = 200, 160


@pytest.fixture(scope="module", autouse=True)
def qapp():
    """Ensure a QApplication exists, so the identity tab text can be drawn."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class StubPose:
    """Minimal pose stand-in with one bounding box per frame for identity 0."""

    def __init__(
        self,
        bbox: list[list[float]] | None = None,
        has_bounding_boxes: bool = True,
    ) -> None:
        self.has_bounding_boxes = has_bounding_boxes
        self._bboxes = None if bbox is None else np.array([bbox], dtype=np.float32)

    def get_bounding_boxes(self, identity: int) -> np.ndarray | None:
        """Return the canned boxes, shape (num_frames, 2, 2)."""
        return self._bboxes

    @staticmethod
    def identity_index_to_display(identity: int) -> str:
        """Return the identity index as text."""
        return str(identity)


def _draw(pose: StubPose) -> QtGui.QImage:
    """Draw identity 0's box on frame 0 of a black image and return the image."""
    image = QtGui.QImage(WIDTH, HEIGHT, QtGui.QImage.Format.Format_RGB888)
    image.fill(QtGui.QColor(0, 0, 0))
    painter = QtGui.QPainter(image)
    try:
        draw_identity_bounding_box(painter, pose, 0, 0, width=WIDTH, height=HEIGHT)
    finally:
        painter.end()
    return image


def _box_colored(image: QtGui.QImage, x: int, y: int) -> bool:
    return image.pixelColor(x, y) == BOUNDING_BOX_COLOR


def test_draws_box_outline_and_leaves_inside_untouched() -> None:
    """The box is an outline: its edges are drawn and its interior is not filled."""
    image = _draw(StubPose([[60, 60], [140, 140]]))

    assert _box_colored(image, 60, 100)  # left edge
    assert _box_colored(image, 140, 100)  # right edge
    assert _box_colored(image, 100, 140)  # bottom edge
    assert image.pixelColor(100, 100) == QtGui.QColor(0, 0, 0)


def test_draws_identity_tab_above_the_box() -> None:
    """The identity tab sits above the box's top-left corner."""
    image = _draw(StubPose([[60, 60], [140, 140]]))

    above = [image.pixelColor(x, 50) for x in range(55, 80)]
    assert any(color != QtGui.QColor(0, 0, 0) for color in above)


@pytest.mark.parametrize(
    "pose",
    [
        StubPose([[-1, -1], [-1, -1]]),
        StubPose([[80, 80], [40, 40]]),
        StubPose(None),
        StubPose([[60, 60], [140, 140]], has_bounding_boxes=False),
    ],
    ids=["missing", "degenerate", "no-data", "no-bounding-boxes"],
)
def test_draws_nothing_without_a_usable_box(pose: StubPose) -> None:
    """Missing, degenerate or unavailable boxes leave the frame untouched."""
    image = _draw(pose)
    array = np.frombuffer(image.constBits(), dtype=np.uint8)

    assert not array.any()


def test_pose_has_bounding_boxes_follows_the_capability() -> None:
    """Only a pose with bounding boxes has any to draw."""
    assert pose_has_bounding_boxes(StubPose([[0, 0], [1, 1]]))
    assert not pose_has_bounding_boxes(StubPose([[0, 0], [1, 1]], has_bounding_boxes=False))


@pytest.mark.parametrize("kind", POSE_KINDS)
@pytest.mark.parametrize(
    ("source", "expected"), [(BBOX_SOURCE, True), (NO_BBOX_SOURCE, False)], ids=["boxes", "none"]
)
def test_real_pose_bounding_boxes_follow_capabilities(
    tmp_path: Path, kind: str, source: Path, expected: bool
) -> None:
    """A real legacy or unified file is drawable exactly when it has BOUNDING_BOXES."""
    with open_pose_file(copy_pose(source, tmp_path, "clip", kind, 8)) as pose:
        assert (PoseCapability.BOUNDING_BOXES in pose.capabilities) is expected
        assert pose_has_bounding_boxes(pose) is expected
        image = QtGui.QImage(800, 800, QtGui.QImage.Format.Format_RGB888)
        image.fill(QtGui.QColor(0, 0, 0))
        painter = QtGui.QPainter(image)
        try:
            draw_identity_bounding_box(painter, pose, 0, 0, width=800, height=800)
        finally:
            painter.end()

    assert bool(np.frombuffer(image.constBits(), dtype=np.uint8).any()) is expected


def test_sizes_grow_with_the_frame() -> None:
    """Larger frames get thicker lines and larger text, never smaller than the base."""
    small = native_bounding_box_sizes(400, 400)
    large = native_bounding_box_sizes(1920, 1080)

    assert all(s >= 1 for s in small)
    assert all(lg >= sm for lg, sm in zip(large, small, strict=True))
    assert large[0] > small[0]
