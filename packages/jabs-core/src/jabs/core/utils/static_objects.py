"""Rules for interpreting the static objects stored in a pose file.

A pose file's ``static_objects`` group is read in two places that must agree:
the v5+ pose readers, which expose the objects a feature can actually use, and
the project scan, which reports what the project as a whole supports. Keeping
the rules here is what stops the two from drifting apart and advertising a
landmark feature that the pose reader will never enable.
"""

from __future__ import annotations

from collections.abc import Sequence


def lixit_keypoint_count(shape: Sequence[int]) -> int:
    """Number of keypoints per lixit held by a ``static_objects/lixit`` dataset.

    The dataset is ``#lixit x 3 x 2`` when each lixit is described by three
    keypoints (tip, left side, right side) and ``#lixit x 2`` when only the tip
    is stored. A dataset holding no lixit at all carries no keypoints, and such
    a ``lixit`` entry is dropped rather than treated as a usable static object.

    Args:
        shape: Shape of the ``lixit`` dataset, from either an ``h5py`` dataset
            (no data load needed) or the array read from one.

    Returns:
        ``3`` or ``1`` for a populated dataset, and ``0`` when the dataset holds
        no lixit, which callers take as "this pose file has no lixit".
    """
    if len(shape) == 0 or shape[0] == 0:
        return 0
    return 3 if len(shape) == 3 else 1
