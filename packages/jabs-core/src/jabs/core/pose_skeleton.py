"""Skeleton constants for the 12-keypoint JABS mouse pose."""

import enum
import logging

logger = logging.getLogger(__name__)

MINIMUM_CONFIDENCE = 0.3
"""Keypoints with a confidence at or below this value are treated as invalid."""


class KeypointIndex(enum.IntEnum):
    """enum defining the 12 keypoint indexes"""

    NOSE = 0
    LEFT_EAR = 1
    RIGHT_EAR = 2
    BASE_NECK = 3
    LEFT_FRONT_PAW = 4
    RIGHT_FRONT_PAW = 5
    CENTER_SPINE = 6
    LEFT_REAR_PAW = 7
    RIGHT_REAR_PAW = 8
    BASE_TAIL = 9
    MID_TAIL = 10
    TIP_TAIL = 11


# Connected segments to use when full 12 keypoints are available.
FULL_CONNECTED_SEGMENTS: tuple[tuple[KeypointIndex, ...], ...] = (
    (
        KeypointIndex.LEFT_FRONT_PAW,
        KeypointIndex.CENTER_SPINE,
        KeypointIndex.RIGHT_FRONT_PAW,
    ),
    (
        KeypointIndex.LEFT_REAR_PAW,
        KeypointIndex.BASE_TAIL,
        KeypointIndex.RIGHT_REAR_PAW,
    ),
    (
        KeypointIndex.NOSE,
        KeypointIndex.BASE_NECK,
        KeypointIndex.CENTER_SPINE,
        KeypointIndex.BASE_TAIL,
        KeypointIndex.MID_TAIL,
        KeypointIndex.TIP_TAIL,
    ),
    (
        KeypointIndex.LEFT_EAR,
        KeypointIndex.NOSE,
        KeypointIndex.RIGHT_EAR,
    ),
)

# Pose based on the Envision Hydra model will have fewer keypoints,
# so we adjust the connected segments accordingly.
NVSN_CONNECTED_SEGMENTS: tuple[tuple[KeypointIndex, ...], ...] = (
    (
        KeypointIndex.LEFT_EAR,
        KeypointIndex.NOSE,
        KeypointIndex.RIGHT_EAR,
    ),
    (
        KeypointIndex.NOSE,
        KeypointIndex.BASE_TAIL,
        KeypointIndex.TIP_TAIL,
    ),
)

# Body parts, by name, that do not count toward identity presence.
TAIL_KEYPOINTS: tuple[str, ...] = ("MID_TAIL", "TIP_TAIL")
