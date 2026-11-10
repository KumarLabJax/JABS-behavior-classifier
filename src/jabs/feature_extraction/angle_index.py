"""Named joint angles and the keypoint triples they are measured from."""

import enum
from typing import Final

from jabs.pose_estimation import PoseEstimation

_KP = PoseEstimation.KeypointIndex


class AngleIndex(enum.IntEnum):
    """enum defining the indexes of the angle features"""

    NOSE_BASE_NECK_RIGHT_FRONT_PAW = 0
    NOSE_BASE_NECK_LEFT_FRONT_PAW = 1
    RIGHT_FRONT_PAW_BASE_NECK_CENTER_SPINE = 2
    LEFT_FRONT_PAW_BASE_NECK_CENTER_SPINE = 3
    BASE_NECK_CENTER_SPINE_BASE_TAIL = 4
    RIGHT_REAR_PAW_BASE_TAIL_CENTER_SPINE = 5
    LEFT_REAR_PAW_BASE_TAIL_CENTER_SPINE = 6
    RIGHT_REAR_PAW_BASE_TAIL_MID_TAIL = 7
    LEFT_REAR_PAW_BASE_TAIL_MID_TAIL = 8
    CENTER_SPINE_BASE_TAIL_MID_TAIL = 9
    BASE_TAIL_MID_TAIL_TIP_TAIL = 10

    @staticmethod
    def get_angle_name(i: "AngleIndex") -> str:
        """map angle index to a string name

        Args:
            i: angle to look up.

        Returns:
            The angle's name, e.g. ``"NOSE-BASE_NECK-RIGHT_FRONT_PAW"``. These
            strings become part of the feature column names stored in the
            feature cache, so changing one invalidates cached features.

        Raises:
            KeyError: If the angle has no entry in the keypoint table.
        """
        return _ANGLE_NAMES[i]

    @staticmethod
    def get_angle_indices(i: "AngleIndex") -> list[PoseEstimation.KeypointIndex]:
        """get the keypoint indices for a given angle index

        Args:
            i: angle to look up.

        Returns:
            The three keypoints the angle is measured from, ordered
            ``[point, vertex, point]``. A fresh list is returned on each call,
            so callers cannot mutate the shared table through it.

        Raises:
            KeyError: If the angle has no entry in the keypoint table.
        """
        return list(_ANGLE_KEYPOINTS[i])


# The three keypoints each named angle is measured from, ordered
# (point, vertex, point): the angle is the one at the middle keypoint, between
# the rays running from it to the other two.
_ANGLE_KEYPOINTS: Final[
    dict[
        AngleIndex,
        tuple[
            PoseEstimation.KeypointIndex,
            PoseEstimation.KeypointIndex,
            PoseEstimation.KeypointIndex,
        ],
    ]
] = {
    AngleIndex.NOSE_BASE_NECK_RIGHT_FRONT_PAW: (_KP.NOSE, _KP.BASE_NECK, _KP.RIGHT_FRONT_PAW),
    AngleIndex.NOSE_BASE_NECK_LEFT_FRONT_PAW: (_KP.NOSE, _KP.BASE_NECK, _KP.LEFT_FRONT_PAW),
    AngleIndex.RIGHT_FRONT_PAW_BASE_NECK_CENTER_SPINE: (
        _KP.RIGHT_FRONT_PAW,
        _KP.BASE_NECK,
        _KP.CENTER_SPINE,
    ),
    AngleIndex.LEFT_FRONT_PAW_BASE_NECK_CENTER_SPINE: (
        _KP.LEFT_FRONT_PAW,
        _KP.BASE_NECK,
        _KP.CENTER_SPINE,
    ),
    AngleIndex.BASE_NECK_CENTER_SPINE_BASE_TAIL: (
        _KP.BASE_NECK,
        _KP.CENTER_SPINE,
        _KP.BASE_TAIL,
    ),
    AngleIndex.RIGHT_REAR_PAW_BASE_TAIL_CENTER_SPINE: (
        _KP.RIGHT_REAR_PAW,
        _KP.BASE_TAIL,
        _KP.CENTER_SPINE,
    ),
    AngleIndex.LEFT_REAR_PAW_BASE_TAIL_CENTER_SPINE: (
        _KP.LEFT_REAR_PAW,
        _KP.BASE_TAIL,
        _KP.CENTER_SPINE,
    ),
    AngleIndex.RIGHT_REAR_PAW_BASE_TAIL_MID_TAIL: (
        _KP.RIGHT_REAR_PAW,
        _KP.BASE_TAIL,
        _KP.MID_TAIL,
    ),
    AngleIndex.LEFT_REAR_PAW_BASE_TAIL_MID_TAIL: (
        _KP.LEFT_REAR_PAW,
        _KP.BASE_TAIL,
        _KP.MID_TAIL,
    ),
    AngleIndex.CENTER_SPINE_BASE_TAIL_MID_TAIL: (
        _KP.CENTER_SPINE,
        _KP.BASE_TAIL,
        _KP.MID_TAIL,
    ),
    AngleIndex.BASE_TAIL_MID_TAIL_TIP_TAIL: (_KP.BASE_TAIL, _KP.MID_TAIL, _KP.TIP_TAIL),
}

# An angle's name is its three keypoint names joined by "-", so it is derived
# from the table above rather than written out a second time: a hand-maintained
# name table can disagree with the keypoints it claims to describe, and these
# names end up in the feature cache where a disagreement would be mislabeled
# data rather than an error. tests/feature_extraction/test_angle_index.py pins
# the exact strings so the derivation cannot silently rename a cached feature.
_ANGLE_NAMES: Final[dict[AngleIndex, str]] = {
    angle: "-".join(keypoint.name for keypoint in keypoints)
    for angle, keypoints in _ANGLE_KEYPOINTS.items()
}
