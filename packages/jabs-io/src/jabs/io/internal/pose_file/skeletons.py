"""The skeletons this specification defines by name.

Derived from ``jabs.core`` rather than transcribed. The keypoint names and
connections already exist as ``PoseEstimation.KeypointIndex`` and
``FULL_CONNECTED_SEGMENTS``, and a second hand-written copy would drift — the
first copy had already picked up a case mismatch against the enum.

The connected segments are polylines; a skeleton stores edges. Drawing an edge
only when both endpoints are valid produces the same picture as splitting a
polyline at missing keypoints, which is what JABS's ``gen_line_fragments``
exists to do, so the conversion loses nothing.
"""

import itertools

from jabs.core.abstract.pose_est import PoseEstimation
from jabs.io.internal.pose_file.types import Skeleton

JABS_MOUSE12 = "jabs.mouse12"


def _edges_from(segments: tuple) -> tuple[tuple[int, int], ...]:
    """Expand polyline segments into deduplicated edges.

    Args:
        segments: Iterables of keypoint indexes, each a polyline.

    Returns:
        Edge pairs, in first-seen order.
    """
    edges: list[tuple[int, int]] = []
    for segment in segments:
        for start, end in itertools.pairwise([int(point) for point in segment]):
            if (start, end) not in edges and (end, start) not in edges:
                edges.append((start, end))
    return tuple(edges)


def jabs_mouse12() -> Skeleton:
    """The JABS 12-keypoint mouse skeleton.

    Returns:
        The skeleton, with body-part names and edges taken from ``jabs.core``.
    """
    return Skeleton(
        body_parts=tuple(keypoint.name for keypoint in PoseEstimation.KeypointIndex),
        edges=_edges_from(PoseEstimation.FULL_CONNECTED_SEGMENTS),
        description="JABS 12-keypoint mouse skeleton",
    )
