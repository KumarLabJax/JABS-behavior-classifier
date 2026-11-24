"""JABS abstract base classes.

``PoseEstimation`` is deprecated (see :mod:`jabs.core.abstract.pose_est`) and
is provided lazily, so that importing this package does not warn.
"""

from typing import Any

__all__: list[str] = []


def __getattr__(name: str) -> Any:
    """Return the deprecated ``PoseEstimation`` name with a ``DeprecationWarning``.

    Args:
        name: Attribute looked up on this package.

    Returns:
        The class carrying the skeleton constants, for ``PoseEstimation``.

    Raises:
        AttributeError: For any other name, so submodule imports still work.
    """
    if name == "PoseEstimation":
        from .pose_est import _deprecated_pose_estimation

        return _deprecated_pose_estimation(stacklevel=2)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
