"""Helpers deriving simple geometry from a pose estimation object."""

from collections.abc import Generator, Iterable

import numpy as np
import numpy.typing as npt

from jabs.core.abstract import PoseEstimation


def gen_line_fragments(
    connected_segments: Iterable[Iterable[PoseEstimation.KeypointIndex]],
    exclude_points: np.ndarray,
) -> Generator[list[int], None, None]:
    """generate line fragments from the connected segments.

    This will break up segments if a point within the segment is excluded,
    or will remove the segment completely if it does not have at least two points

    Args:
        connected_segments: Iterable of Iterables of KeypointIndex, where each inner
            Iterable represents a segment of connected keypoints
        exclude_points: numpy array of points to exclude when generating segments

    Yields:
        yields lists of Keypoint indexes that make up the segments to draw
    """
    curr_fragment = []
    for curr_pt_indexes in connected_segments:
        for curr_pt_index in curr_pt_indexes:
            if curr_pt_index.value in exclude_points:
                if len(curr_fragment) >= 2:
                    yield curr_fragment
                curr_fragment = []
            else:
                curr_fragment.append(curr_pt_index.value)
        if len(curr_fragment) >= 2:
            yield curr_fragment
        curr_fragment = []


def identity_centroids(poses: PoseEstimation, identity: int) -> npt.NDArray[np.float32]:
    """Compute the per-frame centroid of an identity's convex hull, in pixel units.

    The convex hull omits the tail keypoints, so its center is robust to dropout of any
    single keypoint. That is why features prefer it to a single keypoint as a position
    for the animal.

    A frame is NaN both when the identity is absent and when it is present but has no
    convex hull. Those are distinct cases: ``get_identity_convex_hulls()`` needs three
    valid body keypoints to build a hull, while ``identity_mask()`` requires only one for
    some pose versions (v3), so a frame can be masked present and still have no hull.

    Args:
        poses: Pose estimation object to read the identity mask and convex hulls from.
        identity: Identity to compute centroids for.

    Returns:
        Array of shape ``(#frames, 2)`` of ``(x, y)`` centroid coordinates in pixel
        units. Rows are NaN for frames with no centroid.
    """
    centroids = np.full((poses.num_frames, 2), np.nan, dtype=np.float32)
    frame_valid = poses.identity_mask(identity)
    convex_hulls = poses.get_identity_convex_hulls(identity)
    for i in np.arange(poses.num_frames)[frame_valid == 1]:
        hull = convex_hulls[i]
        if hull is not None:
            centroids[i, :] = np.asarray(hull.centroid.xy).squeeze()
    return centroids
