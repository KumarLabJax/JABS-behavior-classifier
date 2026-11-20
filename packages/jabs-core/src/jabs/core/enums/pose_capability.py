"""Enum for optional capabilities of a pose source."""

from enum import Enum


class PoseCapability(str, Enum):
    """Optional data a pose source can provide, used to gate features.

    Static objects are not capabilities; they are checked by name.

    Attributes:
        SOCIAL: The pose has more than one identity.
        SEGMENTATION: The pose carries instance segmentation contours.
        BOUNDING_BOXES: The pose carries per-identity bounding boxes.
    """

    SOCIAL = "social"
    SEGMENTATION = "segmentation"
    BOUNDING_BOXES = "bounding_boxes"
