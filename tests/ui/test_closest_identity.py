"""Tests for working out which animal is nearest the subject."""

import numpy as np
import pytest
from shapely.geometry import Polygon

try:
    from jabs.pose_estimation import PoseEstimation
    from jabs.ui.player_widget.closest_identity import HALF_FOV_DEGREES, closest_identity

    SKIP_UI_TESTS = False
    SKIP_REASON = ""
except ImportError as e:
    SKIP_UI_TESTS = True
    SKIP_REASON = f"Qt/UI dependencies not available: {e}"

pytestmark = pytest.mark.skipif(SKIP_UI_TESTS, reason=SKIP_REASON)


def _square(cx: float, cy: float, size: float = 10) -> Polygon:
    half = size / 2
    return Polygon(
        [
            (cx - half, cy - half),
            (cx + half, cy - half),
            (cx + half, cy + half),
            (cx - half, cy + half),
        ]
    )


class StubPose:
    """Pose stand-in with animals at fixed positions and a subject facing +x."""

    def __init__(self, centers: dict[int, tuple[float, float] | None], facing: bool = True):
        self._centers = centers
        self._facing = facing
        self.identities = list(centers)

    def get_identity_convex_hulls(self, identity: int):
        """One hull per frame; a None center means no pose on that frame."""
        center = self._centers[identity]
        return [None if center is None else _square(*center)]

    def get_points(self, frame_index: int, identity: int):
        """Nose ahead of the neck on the x axis, so the subject faces +x."""
        n_kp = len(PoseEstimation.KeypointIndex)
        points = np.zeros((n_kp, 2), dtype=np.float32)
        mask = np.ones(n_kp, dtype=np.uint8)
        cx, cy = self._centers[identity]
        points[PoseEstimation.KeypointIndex.NOSE] = (cx + 5, cy)
        points[PoseEstimation.KeypointIndex.BASE_NECK] = (cx, cy)
        if not self._facing:
            mask[PoseEstimation.KeypointIndex.NOSE] = 0
        return points, mask


def test_nearest_animal_is_reported() -> None:
    """Without a field of view, distance alone decides."""
    pose = StubPose({0: (0, 0), 1: (100, 0), 2: (30, 0)})

    assert closest_identity(pose, 0, 0) == 2


def test_the_subject_is_never_its_own_nearest_animal() -> None:
    """A lone animal has nobody near it."""
    assert closest_identity(StubPose({0: (0, 0)}), 0, 0) is None


def test_an_animal_with_no_pose_on_the_frame_is_skipped() -> None:
    """A missing hull means the animal cannot be measured against."""
    pose = StubPose({0: (0, 0), 1: (30, 0), 2: None})

    assert closest_identity(pose, 0, 0) == 1


def test_no_answer_when_the_subject_has_no_pose() -> None:
    """Nothing to measure from."""
    pose = StubPose({0: None, 1: (30, 0)})

    assert closest_identity(pose, 0, 0) is None


def test_the_field_of_view_excludes_animals_behind_the_subject() -> None:
    """The subject faces +x, so the nearer animal behind it does not qualify.

    This is the difference between the two markers the overlay draws: the nearest
    animal overall, and the nearest one the subject can actually see.
    """
    pose = StubPose({0: (0, 0), 1: (-20, 0), 2: (60, 0)})

    assert closest_identity(pose, 0, 0) == 1
    assert closest_identity(pose, 0, 0, HALF_FOV_DEGREES) == 2


def test_a_subject_without_nose_and_neck_has_no_field_of_view() -> None:
    """Facing direction needs both keypoints; without them nobody qualifies."""
    pose = StubPose({0: (0, 0), 1: (60, 0)}, facing=False)

    assert closest_identity(pose, 0, 0, HALF_FOV_DEGREES) is None
    assert closest_identity(pose, 0, 0) == 1


def test_a_half_angle_of_180_or_more_ignores_facing() -> None:
    """At 180 degrees the field of view covers everything, so distance decides."""
    pose = StubPose({0: (0, 0), 1: (-20, 0), 2: (60, 0)})

    assert closest_identity(pose, 0, 0, 180) == 1
