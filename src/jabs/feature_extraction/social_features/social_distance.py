"""Helper computations for social distance, field-of-view, and huddle topology features."""

from __future__ import annotations

import math
from collections import deque

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

from jabs.core.utils import signed_angle_degrees
from jabs.pose_estimation import PoseEstimation

# Maximum number of contour vertices evaluated per identity per frame when
# computing segmentation boundary distances; longer contours are uniformly
# strided to bound per-frame KD-tree construction cost.
_MAX_SEG_CONTOUR_VERTICES = 64

# Fraction of an animal's characteristic body convex-hull size (sqrt(area))
# representing the soft flank/fur margin outside the internal skeletal keypoints.
_CONTACT_AREA_SCALE = 0.35
_CONTACT_PERIMETER_SCALE = 0.08
_SEG_CONTACT_TOLERANCE_PX = 2.0


def _hull_contact_margin(hull: object) -> float:
    """Compute the pixel-space body contact margin for a skeletal convex hull.

    Because the skeletal convex hull spans internal keypoints rather than the
    outer fur coat, two huddling mice whose coats are in physical contact have a
    nonzero hull-to-hull gap proportional to their body width. Scaling this margin
    by the square root of hull area (and perimeter fallback for collinear poses)
    makes the contact criterion resolution-invariant.

    Args:
        hull: Shapely geometry representing an identity's body convex hull.

    Returns:
        Pixel-space contact margin for this hull.
    """
    area = float(getattr(hull, "area", 0.0))
    length = float(getattr(hull, "length", 0.0))
    return max(_CONTACT_AREA_SCALE * math.sqrt(max(area, 0.0)), _CONTACT_PERIMETER_SCALE * length)


def _extract_valid_contour_points(
    frame_seg: npt.NDArray[np.generic],
) -> npt.NDArray[np.float32] | None:
    """Extract valid ``(x, y)`` contour vertices from a single frame's segmentation array.

    Args:
        frame_seg: Array of shape ``(num_contours, num_vertices, 2)`` padded with ``-1``.

    Returns:
        2-D ``float32`` array of valid vertices (strided to at most
        ``_MAX_SEG_CONTOUR_VERTICES`` rows), or ``None`` if no valid vertices exist.
    """
    valid_mask = frame_seg[..., 0] != -1
    if not np.any(valid_mask):
        return None
    pts = frame_seg[valid_mask].astype(np.float32, copy=False)
    if len(pts) > _MAX_SEG_CONTOUR_VERTICES:
        step = max(1, len(pts) // _MAX_SEG_CONTOUR_VERTICES)
        pts = pts[::step]
    return pts


def _compute_identity_centroid_speeds(
    hulls: list[object],
    num_frames: int,
    fps: float,
    pixel_scale: float,
) -> npt.NDArray[np.float32]:
    """Compute per-frame convex-hull centroid speed for one identity.

    Matches the formula used by ``CentroidVelocityMag`` in ``base_features``,
    operating directly on the pre-fetched convex hulls so it requires no extra
    pose file reads.

    Args:
        hulls: List of per-frame Shapely convex hulls (or ``None``) for the identity.
        num_frames: Total number of frames in the video.
        fps: Video frame rate in frames per second.
        pixel_scale: Scale factor converting pixels to target distance units.

    Returns:
        1-D ``float32`` array of centroid speeds of length ``num_frames``.
    """
    if num_frames < 2:
        return np.full(num_frames, np.nan, dtype=np.float32)

    centroids = np.full((num_frames, 2), np.nan, dtype=np.float32)
    for frame_idx, hull in enumerate(hulls):
        if hull is not None:
            centroids[frame_idx, :] = np.asarray(hull.centroid.xy).squeeze()

    velocity = np.gradient(centroids, axis=0)
    speed = np.linalg.norm(velocity, axis=-1) * fps * pixel_scale
    return speed.astype(np.float32, copy=False)


class ClosestIdentityInfo:
    """Pre-computed social proximity, field-of-view, and group topology data for one identity.

    This helper computes nearest-neighbor identities, hull and segmentation distances,
    field-of-view angles, multi-animal contact counts, transitive connected group sizes,
    and nearest-neighbor speeds once for a given subject identity, then shares the cached
    arrays across all social feature classes in ``SocialFeatureGroup``.

    Args:
        poses: Pose estimation data for the video.
        identity: Subject identity index.
        pixel_scale: Scale factor to convert pixel distances to real-world units (cm).
    """

    # Taken from the ICY paper where the full field of view is 240 degrees.
    HALF_FOV_DEGREE = 120

    def __init__(self, poses: PoseEstimation, identity: int, pixel_scale: float) -> None:
        idx = PoseEstimation.KeypointIndex

        self._poses = poses
        self._identity = identity
        self._pixel_scale = pixel_scale

        num_frames = poses.num_frames
        identities = list(poses.identities)
        other_identities = [curr_id for curr_id in identities if curr_id != identity]

        self._closest_identities = np.full(num_frames, -1, dtype=np.int16)
        self._closest_fov_identities = np.full(num_frames, -1, dtype=np.int16)
        self._fov_angles = np.full(num_frames, np.nan, dtype=np.float32)

        self._closest_distances = np.full(num_frames, np.nan, dtype=np.float32)
        self._closest_fov_distances = np.full(num_frames, np.nan, dtype=np.float32)
        self._second_closest_distances = np.full(num_frames, np.nan, dtype=np.float32)
        self._closest_segmentation_distances = np.full(num_frames, np.nan, dtype=np.float32)
        self._social_contact_counts = np.full(num_frames, np.nan, dtype=np.float32)
        self._social_connected_group_sizes = np.full(num_frames, np.nan, dtype=np.float32)
        self._closest_neighbor_speeds = np.full(num_frames, np.nan, dtype=np.float32)

        hulls_by_id: dict[int, list[object]] = {
            curr_id: poses.get_identity_convex_hulls(curr_id) for curr_id in identities
        }
        self_hulls = hulls_by_id[identity]

        fps = float(getattr(poses, "fps", 30))
        speeds_by_id: dict[int, npt.NDArray[np.float32]] = {
            curr_id: _compute_identity_centroid_speeds(
                hulls_by_id[curr_id], num_frames, fps, pixel_scale
            )
            for curr_id in other_identities
        }

        has_seg = bool(
            getattr(poses, "format_major_version", 0) >= 6
            and getattr(poses, "has_segmentation", False)
        )
        seg_by_id: dict[int, npt.NDArray[np.generic] | None] = {}
        if has_seg:
            for curr_id in identities:
                get_seg = getattr(poses, "get_segmentation_data", None)
                seg_by_id[curr_id] = get_seg(curr_id) if get_seg is not None else None

        for frame in range(num_frames):
            points, mask = poses.get_points(frame, identity, pixel_scale)

            # Skip this frame if the identity is not present.
            if points is None:
                continue

            self_shape = self_hulls[frame]
            if self_shape is None:
                continue

            # Nose and base-neck keypoints define the subject's facing direction for
            # field-of-view (FoV) features, whereas body-hull proximity depends only
            # on valid convex hulls so nose occlusion during huddling does not drop
            # closest-neighbor identity or distance measurements.
            has_fov_keypoints = mask[idx.NOSE] == 1 and mask[idx.BASE_NECK] == 1
            if has_fov_keypoints:
                self_base_neck_point = points[idx.BASE_NECK, :]
                self_nose_point = points[idx.NOSE, :]

            closest_dist: float | None = None
            second_closest_dist: float | None = None
            closest_fov_dist: float | None = None

            self_margin = _hull_contact_margin(self_shape)
            valid_other_ids: list[int] = []
            direct_contacts: list[int] = []

            for curr_id in other_identities:
                other_shape = hulls_by_id[curr_id][frame]
                if other_shape is None:
                    continue

                valid_other_ids.append(curr_id)
                curr_dist = float(self_shape.distance(other_shape))

                if curr_dist <= self_margin + _hull_contact_margin(other_shape):
                    direct_contacts.append(curr_id)

                if closest_dist is None or curr_dist < closest_dist:
                    second_closest_dist = closest_dist
                    closest_dist = curr_dist
                    self._closest_identities[frame] = curr_id
                elif second_closest_dist is None or curr_dist < second_closest_dist:
                    second_closest_dist = curr_dist

                if has_fov_keypoints:
                    other_centroid = (
                        np.array(other_shape.centroid.xy).squeeze() * self._pixel_scale
                    )
                    view_angle = signed_angle_degrees(
                        self_nose_point, self_base_neck_point, other_centroid
                    )
                    if abs(view_angle) <= self.HALF_FOV_DEGREE and (
                        closest_fov_dist is None or curr_dist < closest_fov_dist
                    ):
                        self._closest_fov_identities[frame] = curr_id
                        self._fov_angles[frame] = view_angle
                        closest_fov_dist = curr_dist

            if closest_dist is not None:
                scaled_closest = closest_dist * self._pixel_scale
                self._closest_distances[frame] = scaled_closest
                closest_id = int(self._closest_identities[frame])
                self._closest_neighbor_speeds[frame] = speeds_by_id[closest_id][frame]

                seg_dist_px = closest_dist
                if has_seg and closest_dist > 0.0:
                    self_seg_arr = seg_by_id.get(identity)
                    other_seg_arr = seg_by_id.get(closest_id)
                    if self_seg_arr is not None and other_seg_arr is not None:
                        pts_self = _extract_valid_contour_points(self_seg_arr[frame])
                        pts_other = _extract_valid_contour_points(other_seg_arr[frame])
                        if pts_self is not None and pts_other is not None:
                            d_vals, _ = cKDTree(pts_self).query(pts_other, k=1)
                            seg_dist_px = min(float(np.min(d_vals)), closest_dist)
                            if (
                                seg_dist_px <= _SEG_CONTACT_TOLERANCE_PX
                                and closest_id not in direct_contacts
                            ):
                                direct_contacts.append(closest_id)

                self._closest_segmentation_distances[frame] = seg_dist_px * self._pixel_scale

            if second_closest_dist is not None:
                self._second_closest_distances[frame] = second_closest_dist * self._pixel_scale

            if closest_fov_dist is not None:
                self._closest_fov_distances[frame] = closest_fov_dist * self._pixel_scale

            contact_count = len(direct_contacts)
            self._social_contact_counts[frame] = float(contact_count)

            # Compute connected component size via BFS when partial contacts exist
            # and unvisited animals remain that could form a transitive chain (A-B-C).
            if contact_count == 0:
                self._social_connected_group_sizes[frame] = 1.0
            elif contact_count == len(valid_other_ids):
                self._social_connected_group_sizes[frame] = float(1 + contact_count)
            else:
                visited: set[int] = {identity, *direct_contacts}
                unvisited: set[int] = set(valid_other_ids) - visited
                queue: deque[int] = deque(direct_contacts)
                while queue and unvisited:
                    curr_node = queue.popleft()
                    curr_hull = hulls_by_id[curr_node][frame]
                    curr_margin = _hull_contact_margin(curr_hull)
                    newly_reached: list[int] = []
                    for cand_id in unvisited:
                        cand_hull = hulls_by_id[cand_id][frame]
                        if float(curr_hull.distance(cand_hull)) <= (
                            curr_margin + _hull_contact_margin(cand_hull)
                        ):
                            newly_reached.append(cand_id)
                    for cand_id in newly_reached:
                        unvisited.remove(cand_id)
                        visited.add(cand_id)
                        queue.append(cand_id)
                self._social_connected_group_sizes[frame] = float(len(visited))

    @property
    def closest_identities(self) -> npt.NDArray[np.int16]:
        """Return the closest identities for each frame."""
        return self._closest_identities

    @property
    def closest_fov_identities(self) -> npt.NDArray[np.int16]:
        """Return the closest identities in the field of view for each frame."""
        return self._closest_fov_identities

    @property
    def closest_fov_angles(self) -> npt.NDArray[np.float32]:
        """Return the angles of the closest animals in the field of view."""
        return self._fov_angles

    @property
    def closest_distances(self) -> npt.NDArray[np.float32]:
        """Return the pre-computed hull distances to the closest identity per frame."""
        return self._closest_distances

    @property
    def closest_fov_distances(self) -> npt.NDArray[np.float32]:
        """Return the pre-computed hull distances to the closest FoV identity per frame."""
        return self._closest_fov_distances

    @property
    def second_closest_distances(self) -> npt.NDArray[np.float32]:
        """Return the hull distances to the second-closest identity per frame."""
        return self._second_closest_distances

    @property
    def closest_segmentation_distances(self) -> npt.NDArray[np.float32]:
        """Return the segmentation contour distances to the closest identity per frame."""
        return self._closest_segmentation_distances

    @property
    def social_contact_counts(self) -> npt.NDArray[np.float32]:
        """Return the number of other identities directly in body contact per frame."""
        return self._social_contact_counts

    @property
    def social_connected_group_sizes(self) -> npt.NDArray[np.float32]:
        """Return the size of the subject's connected social contact cluster per frame."""
        return self._social_connected_group_sizes

    @property
    def closest_neighbor_speeds(self) -> npt.NDArray[np.float32]:
        """Return the centroid speed of the closest identity per frame."""
        return self._closest_neighbor_speeds

    def compute_distances(
        self, closest_identities: npt.NDArray[np.integer]
    ) -> npt.NDArray[np.float32]:
        """Compute frame-wise distances between the subject's hull and the specified identity's hull.

        When ``closest_identities`` is the cached ``self.closest_identities`` or
        ``self.closest_fov_identities`` array, returns a copy of the distances already
        computed during construction.

        Args:
            closest_identities: Array of closest identity indices for each frame.

        Returns:
            Array of distances for each frame, with NaN where distance cannot be computed.
        """
        if closest_identities is self._closest_identities:
            return self._closest_distances.copy()
        if closest_identities is self._closest_fov_identities:
            return self._closest_fov_distances.copy()

        values = np.full(self._poses.num_frames, np.nan, dtype=np.float32)
        self_convex_hulls = self._poses.get_identity_convex_hulls(self._identity)

        for frame in range(self._poses.num_frames):
            points, _ = self._poses.get_points(frame, self._identity, self._pixel_scale)
            if points is None:
                continue

            closest_id = int(closest_identities[frame])
            if closest_id != -1:
                self_shape = self_convex_hulls[frame]
                other_shape = self._poses.get_identity_convex_hulls(closest_id)[frame]
                if self_shape is not None and other_shape is not None:
                    values[frame] = float(self_shape.distance(other_shape)) * self._pixel_scale

        return values

    def compute_pairwise_social_distances(
        self,
        social_points: list[PoseEstimation.KeypointIndex],
        closest_identities: npt.NDArray[np.integer],
    ) -> dict[str, npt.NDArray[np.float32]]:
        """Compute pairwise distances between specified keypoints of the subject and closest identity.

        Args:
            social_points: List of keypoint indices to use for pairwise distance calculation.
            closest_identities: Array of closest identity indices for each frame.

        Returns:
            Dictionary mapping keypoint pair names to arrays of distances for each frame.
        """
        return_dict: dict[str, npt.NDArray[np.float32]] = {}
        for p1 in social_points:
            for p2 in social_points:
                return_dict[f"social dist. {p1.name}-{p2.name}"] = np.full(
                    self._poses.num_frames, np.nan, dtype=np.float32
                )

        for frame in range(self._poses.num_frames):
            closest_id = int(closest_identities[frame])
            points, _ = self._poses.get_points(frame, self._identity, self._pixel_scale)

            if points is None or closest_id == -1:
                continue

            closest_points, _ = self._poses.get_points(frame, closest_id, self._pixel_scale)
            if closest_points is None:
                continue

            for p1 in social_points:
                for p2 in social_points:
                    key = f"social dist. {p1.name}-{p2.name}"
                    return_dict[key][frame] = math.dist(points[p1.value], closest_points[p2.value])

        return return_dict
