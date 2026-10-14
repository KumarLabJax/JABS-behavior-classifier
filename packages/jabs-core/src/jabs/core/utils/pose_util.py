"""Helpers deriving identity presence and simple geometry from pose estimation data."""

from collections.abc import Generator, Iterable

import numpy as np
import numpy.typing as npt

from jabs.core.abstract import PoseEstimation


def build_identity_mask(
    point_mask: npt.NDArray[np.integer] | npt.NDArray[np.bool_],
    min_valid_keypoints: int,
) -> npt.NDArray[np.uint8]:
    """Build a per-frame identity presence mask from a keypoint validity mask.

    An identity counts as present in a frame when at least ``min_valid_keypoints`` of
    its body keypoints are valid. The last two keypoints (``MID_TAIL`` and
    ``TIP_TAIL``) are dropped before counting, matching the body-only keypoint set
    :meth:`PoseEstimation.get_identity_convex_hulls` builds its hulls from.

    The threshold is the caller's because it differs by pose version: v2 and v4
    require the three keypoints a convex hull needs, while v3 requires only one, so
    a v3 frame can be masked present and still have no hull.

    Args:
        point_mask: Per-keypoint validity mask with the keypoints on the last axis,
            shape ``(..., #keypoints)``. Single-identity readers pass
            ``(#frames, #keypoints)``; identity-indexed readers pass
            ``(#identities, #frames, #keypoints)``. Any dtype that sums as 0/1 per
            keypoint is accepted.
        min_valid_keypoints: Number of valid body keypoints a frame needs for the
            identity to count as present.

    Returns:
        ``uint8`` array of 0/1 values with shape ``point_mask.shape[:-1]``.
    """
    valid_body_keypoints = np.sum(point_mask[..., :-2], axis=-1)
    return (valid_body_keypoints >= min_valid_keypoints).astype(np.uint8)


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
