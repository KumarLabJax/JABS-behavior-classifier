"""Deprecated home of the legacy pose reader base class.

The per-version legacy pose readers and their abstract base class were removed;
JABS opens every pose file as ``jabs.pose_estimation.Pose``, typed against
:class:`jabs.core.pose_protocol.PoseLike`. ``PoseEstimation`` stays for one
release as a name that only carries the skeleton constants, now in
:mod:`jabs.core.pose_skeleton`, and warns on every access.
"""

import warnings

from jabs.core.pose_skeleton import (
    FULL_CONNECTED_SEGMENTS,
    MINIMUM_CONFIDENCE,
    NVSN_CONNECTED_SEGMENTS,
    KeypointIndex,
)

__all__ = ["MINIMUM_CONFIDENCE"]

_DEPRECATED_NAME = "PoseEstimation"
_DEPRECATION_MESSAGE = (
    "jabs.core.abstract.PoseEstimation is deprecated and will be removed in a future "
    "release; the legacy pose readers are gone. Use jabs.core.pose_skeleton for "
    "KeypointIndex, FULL_CONNECTED_SEGMENTS and NVSN_CONNECTED_SEGMENTS, and "
    "jabs.core.pose_protocol.PoseLike to type a pose."
)


class _PoseEstimation:
    """Carrier of the skeleton constants under the deprecated ``PoseEstimation`` name.

    The attributes are the same objects as :mod:`jabs.core.pose_skeleton`'s.
    """

    KeypointIndex = KeypointIndex
    FULL_CONNECTED_SEGMENTS = FULL_CONNECTED_SEGMENTS
    NVSN_CONNECTED_SEGMENTS = NVSN_CONNECTED_SEGMENTS


def _deprecated_pose_estimation(stacklevel: int) -> type[_PoseEstimation]:
    """Warn that ``PoseEstimation`` is deprecated and return its constant carrier.

    Args:
        stacklevel: ``warnings.warn`` stack level that points at the caller's
            access.

    Returns:
        The class carrying the skeleton constants.
    """
    warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=stacklevel + 1)
    return _PoseEstimation


def __getattr__(name: str) -> type[_PoseEstimation]:
    """Return the deprecated ``PoseEstimation`` name with a ``DeprecationWarning``.

    Args:
        name: Attribute looked up on this module.

    Returns:
        The class carrying the skeleton constants, for ``PoseEstimation``.

    Raises:
        AttributeError: For any other name.
    """
    if name == _DEPRECATED_NAME:
        return _deprecated_pose_estimation(stacklevel=2)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
