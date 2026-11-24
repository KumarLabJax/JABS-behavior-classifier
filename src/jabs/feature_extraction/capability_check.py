"""Refuse feature settings a pose file cannot provide, before classifying it."""

from collections.abc import Mapping
from typing import Any

from jabs.core.enums import PoseCapability

__all__ = ["PoseCapabilityError", "require_pose_capabilities"]

# feature setting -> the pose capability its features need
_SETTING_CAPABILITIES: tuple[tuple[str, PoseCapability], ...] = (
    ("social", PoseCapability.SOCIAL),
    ("segmentation", PoseCapability.SEGMENTATION),
)

_REASONS = {
    "social": "its format holds one animal (a pose_est_v2 file, or a unified file with one slot)",
    "segmentation": "it has no segmentation data",
}


class PoseCapabilityError(ValueError):
    """Feature settings turn on features the pose file cannot provide.

    Attributes:
        pose_name: The pose file's name, for the message.
        missing: The settings (``"social"``, ``"segmentation"``) the pose lacks.
    """

    def __init__(self, pose_name: str, missing: tuple[str, ...]) -> None:
        self.pose_name = pose_name
        self.missing = missing
        reasons = "; ".join(f"{name}: {_REASONS[name]}" for name in missing)
        super().__init__(
            f"{pose_name}: the classifier was trained with {' and '.join(missing)} features, "
            f"which this pose file cannot provide ({reasons}). Classify it with a classifier "
            f"trained without them."
        )


def require_pose_capabilities(
    settings: Mapping[str, Any],
    capabilities: frozenset[PoseCapability],
    pose_name: str,
) -> None:
    """Raise when ``settings`` need features ``capabilities`` cannot provide.

    A classifier trained with social (or segmentation) features needs their
    columns; without this check, a pose lacking them fails deep in pandas with
    a ``KeyError`` naming the missing columns.

    Args:
        settings: A classifier's or behavior's feature settings.
        capabilities: The pose file's capabilities.
        pose_name: The pose file's name, for the message.

    Raises:
        PoseCapabilityError: If a feature the settings turn on needs a
            capability the pose lacks.
    """
    missing = tuple(
        name
        for name, capability in _SETTING_CAPABILITIES
        if settings.get(name) and capability not in capabilities
    )
    if missing:
        raise PoseCapabilityError(pose_name, missing)
