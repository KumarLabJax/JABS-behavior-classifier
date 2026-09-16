"""Validation of the ordered behavior-name lists used by multi-class projects.

A multi-class project describes its classes as an ordered list of behavior names
plus the reserved :data:`~jabs.core.constants.MULTICLASS_NONE_BEHAVIOR` class,
which always occupies the first class index. Colors, label arrays, and classifier
class indices are all derived from that list by position, so every consumer needs
the same two guarantees: the reserved name is not in the list, and no name appears
twice.
"""

from collections.abc import Sequence

from jabs.core.constants import MULTICLASS_NONE_BEHAVIOR


def validate_behavior_names(behavior_names: Sequence[str]) -> None:
    """Check that a multi-class behavior-name list can be used as a class ordering.

    Args:
        behavior_names: Ordered list of project behavior names, excluding the
            reserved ``MULTICLASS_NONE_BEHAVIOR`` class. An empty list is
            accepted; callers that require at least one behavior check for that
            themselves.

    Raises:
        ValueError: If ``behavior_names`` contains the reserved
            ``MULTICLASS_NONE_BEHAVIOR`` name or any duplicate entries.
    """
    if MULTICLASS_NONE_BEHAVIOR in behavior_names:
        raise ValueError(
            f"behavior_names must not include the reserved name {MULTICLASS_NONE_BEHAVIOR!r}"
        )
    if len(behavior_names) != len(set(behavior_names)):
        raise ValueError("behavior_names must not contain duplicates")
