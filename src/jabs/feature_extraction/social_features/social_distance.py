"""Helper computations for social distance, field-of-view, and huddle topology features."""

from __future__ import annotations

import logging
import math
import weakref
from collections import deque
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint
from shapely.geometry.base import BaseGeometry

from jabs.core.utils import signed_angle_degrees
from jabs.core.utils.pose_util import identity_centroids
from jabs.pose_estimation import PoseEstimation

logger = logging.getLogger(__name__)

# Maximum number of contour vertices evaluated per identity per frame when
# computing segmentation boundary distances; longer contours are uniformly
# strided to bound per-frame KD-tree construction cost.
_MAX_SEG_CONTOUR_VERTICES = 64

# Fraction of an animal's characteristic body convex-hull size (sqrt(area))
# representing the soft flank/fur margin outside the internal skeletal keypoints.
_CONTACT_AREA_SCALE = 0.35
_CONTACT_PERIMETER_SCALE = 0.08

# When only 3 nearly-collinear spine keypoints are visible, the triangle's area
# approaches 0 even though its perimeter (~2 * spine_length) remains normal.
# Taking 0.25 * perimeter (~0.5 * spine_length) prevents the characteristic body
# width estimate from collapsing on collinear 3-keypoint frames, while 1.0 px
# guards against division by zero on degenerate geometries.
_COLLINEAR_PERIMETER_SCALE = 0.25
_MIN_BODY_CHAR_LEN_PX = 1.0

# Resolution-invariant segmentation contour contact tolerance: scales with the
# mean characteristic body size of the two animals (~4% of body width) while
# enforcing a 2.0 px floor because non-overlapping integer-pixel instance masks
# of two touching animals are separated by a 1.0 to sqrt(2) (~1.41 px) raster
# grid gap even when their coats are pressed directly together.
_SEG_CONTACT_BODY_SCALE = 0.04
_SEG_CONTACT_MIN_RASTER_PX = 2.0


def _characteristic_body_length(hull: BaseGeometry) -> float:
    """Compute the characteristic pixel-space body width of a convex hull.

    Uses ``sqrt(area)`` as the primary body-size scale, with a ``0.25 * perimeter``
    fallback when only 3 nearly-collinear keypoints are visible (where polygon area
    collapses toward zero while perimeter remains ``~2 * spine_length``) and a
    ``1.0 px`` floor to prevent division by zero.
    """
    area = max(float(hull.area), 0.0)
    length = max(float(hull.length), 0.0)
    return max(
        math.sqrt(area),
        _COLLINEAR_PERIMETER_SCALE * length,
        _MIN_BODY_CHAR_LEN_PX,
    )


def _hull_contact_margin(hull: BaseGeometry, *, is_segmentation_fallback: bool = False) -> float:
    """Compute the pixel-space body contact margin for a convex hull.

    Because a skeletal convex hull spans internal keypoints rather than the outer
    fur coat, two huddling mice whose coats are in physical contact have a nonzero
    hull-to-hull gap proportional to their body width. Scaling this margin by the
    square root of hull area (and perimeter fallback for collinear poses) makes the
    contact criterion resolution-invariant.

    When ``is_segmentation_fallback`` is True, the hull was built from the outer
    segmentation contour and already wraps the coat, so the margin is ``0.0`` to
    avoid double-counting the fur allowance.

    Args:
        hull: Shapely geometry representing an identity's body convex hull.
        is_segmentation_fallback: Whether ``hull`` was constructed from segmentation
            contour vertices rather than skeletal keypoints.

    Returns:
        Pixel-space contact margin for this hull.
    """
    if is_segmentation_fallback:
        return 0.0
    area = max(float(hull.area), 0.0)
    length = max(float(hull.length), 0.0)
    return max(_CONTACT_AREA_SCALE * math.sqrt(area), _CONTACT_PERIMETER_SCALE * length)


def _seg_contact_tolerance(hull_a: BaseGeometry, hull_b: BaseGeometry) -> float:
    """Compute the resolution-invariant fur contour contact tolerance in pixels.

    Scales with the mean characteristic body size of the two animals so the fur
    contact criterion generalizes across camera resolutions and zoom levels, while
    enforcing a ``2.0 px`` floor for integer-pixel raster mask quantization.
    """
    mean_body_len = 0.5 * (
        _characteristic_body_length(hull_a) + _characteristic_body_length(hull_b)
    )
    return max(_SEG_CONTACT_MIN_RASTER_PX, _SEG_CONTACT_BODY_SCALE * mean_body_len)


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


def _contour_min_distance(
    pts_a: npt.NDArray[np.float32] | None,
    pts_b: npt.NDArray[np.float32] | None,
    hull_dist_fallback: float,
) -> float:
    """Compute minimum Euclidean distance between two contour point sets, bounded by hull distance."""
    if hull_dist_fallback <= 0.0 or pts_a is None or pts_b is None:
        return hull_dist_fallback
    d_vals, _ = cKDTree(pts_a).query(pts_b, k=1)
    return min(float(np.min(d_vals)), hull_dist_fallback)


def _populate_segmentation_fallback_hulls(
    hulls: list[BaseGeometry | None],
    seg_arr: npt.NDArray[np.generic] | None,
) -> tuple[list[BaseGeometry | None], npt.NDArray[np.bool_]]:
    """Fill missing skeletal convex hulls from segmentation contours when ``< 3`` keypoints are valid.

    During deep multi-animal huddling, a buried animal may have fewer than 3 skeletal
    keypoints above the confidence threshold even though its segmentation mask remains
    visible. Constructing a fallback convex hull from the contour vertices preserves
    social distance and group topology features on those frames.

    Returns:
        Tuple of ``(updated_hulls, is_fallback_mask)`` where ``is_fallback_mask[frame]``
        is True on frames populated from ``seg_arr`` so hull-margin contact tests do not
        double-count the fur coat.
    """
    num_frames = len(hulls)
    is_fallback = np.zeros(num_frames, dtype=bool)
    if seg_arr is None:
        return list(hulls), is_fallback

    updated = list(hulls)
    for frame_idx, hull in enumerate(updated):
        if hull is None and frame_idx < len(seg_arr):
            pts = _extract_valid_contour_points(seg_arr[frame_idx])
            if pts is not None and len(pts) >= 3:
                cand_hull = MultiPoint(pts).convex_hull
                if cand_hull.area > 0.0:
                    updated[frame_idx] = cand_hull
                    is_fallback[frame_idx] = True
    return updated, is_fallback


def _compute_identity_centroid_speeds(
    poses: PoseEstimation,
    identity: int,
) -> npt.NDArray[np.float32]:
    """Compute per-frame convex-hull centroid speed for one identity, in pixels per second.

    Matches ``CentroidVelocityMag`` in ``base_features`` by delegating centroid
    computation to ``identity_centroids(poses, identity)``, ensuring identical
    ``identity_mask`` handling and avoiding centroid jumps on frames that use a
    segmentation fallback hull. Speeds are left unscaled so the cache holding them
    does not depend on ``pixel_scale``.

    Args:
        poses: Pose estimation object for the video.
        identity: Identity index to compute speeds for.

    Returns:
        1-D ``float32`` array of centroid speeds of length ``poses.num_frames``.
    """
    num_frames = poses.num_frames
    if num_frames < 2:
        return np.full(num_frames, np.nan, dtype=np.float32)

    centroid_centers = identity_centroids(poses, identity)
    velocity = np.gradient(centroid_centers, axis=0)
    speed = np.linalg.norm(velocity, axis=-1) * float(poses.fps)
    return speed.astype(np.float32, copy=False)


def _frame_contact_graph(
    frame: int,
    hulls_by_id: dict[int, list[BaseGeometry | None]],
    is_fallback_by_id: dict[int, npt.NDArray[np.bool_]],
    seg_points_by_id: dict[int, npt.NDArray[np.float32] | None],
) -> dict[int, set[int]]:
    """Return a symmetric adjacency map of identities in physical body contact at one frame.

    Evaluates every unordered pair of present identities once using a single symmetric
    criterion (skeletal hull margin contact or outer segmentation contour contact),
    guaranteeing that all animals in a huddle cluster agree on ``social contact count``
    and ``social connected group size``.
    """
    present = [i for i, hulls in hulls_by_id.items() if hulls[frame] is not None]
    adjacency: dict[int, set[int]] = {i: set() for i in present}
    for a_idx, id_a in enumerate(present):
        hull_a = hulls_by_id[id_a][frame]
        assert hull_a is not None
        margin_a = _hull_contact_margin(
            hull_a, is_segmentation_fallback=bool(is_fallback_by_id[id_a][frame])
        )
        pts_a = seg_points_by_id.get(id_a)
        tree_a: cKDTree | None = None

        for id_b in present[a_idx + 1 :]:
            hull_b = hulls_by_id[id_b][frame]
            assert hull_b is not None
            margin_b = _hull_contact_margin(
                hull_b, is_segmentation_fallback=bool(is_fallback_by_id[id_b][frame])
            )
            hull_gap = float(hull_a.distance(hull_b))
            touching = not (
                is_fallback_by_id[id_a][frame] and is_fallback_by_id[id_b][frame]
            ) and hull_gap <= (margin_a + margin_b)
            if not touching:
                pts_b = seg_points_by_id.get(id_b)
                if pts_a is not None and pts_b is not None:
                    if tree_a is None:
                        tree_a = cKDTree(pts_a)
                    seg_gap = min(float(tree_a.query(pts_b, k=1)[0].min()), hull_gap)
                    touching = seg_gap <= _seg_contact_tolerance(hull_a, hull_b)
                elif hull_gap <= _seg_contact_tolerance(hull_a, hull_b):
                    touching = True
            if touching:
                adjacency[id_a].add(id_b)
                adjacency[id_b].add(id_a)
    return adjacency


def _frame_contour_points(
    seg_arr: npt.NDArray[np.generic] | None, frame: int
) -> npt.NDArray[np.float32] | None:
    """Return one frame's valid contour vertices, or None if there are none."""
    if seg_arr is None or frame >= len(seg_arr):
        return None
    return _extract_valid_contour_points(seg_arr[frame])


def _connected_component_sizes(adjacency: dict[int, set[int]]) -> dict[int, int]:
    """Return the size of the connected component containing each identity."""
    sizes: dict[int, int] = {}
    for start in adjacency:
        if start in sizes:
            continue
        visited = {start}
        queue: deque[int] = deque([start])
        while queue:
            for neighbor in adjacency[queue.popleft()] - visited:
                visited.add(neighbor)
                queue.append(neighbor)
        for member in visited:
            sizes[member] = len(visited)
    return sizes


@dataclass
class _VideoSocialCache:
    """Video-level pre-computed hulls, speeds, and contact topology for all identities.

    ``IdentityFeatures`` builds a new ``SocialFeatureGroup`` (and ``ClosestIdentityInfo``)
    for each identity, but callers reuse the same ``PoseEstimation`` object across
    identities. ``for_pose()`` keys the cache on that object so the per-video work is
    done once rather than once per subject animal. Only compact per-identity arrays are
    kept; contour vertices are re-extracted on demand.
    """

    has_seg: bool
    hulls_by_id: dict[int, list[BaseGeometry | None]]
    is_fallback_by_id: dict[int, npt.NDArray[np.bool_]]
    seg_by_id: dict[int, npt.NDArray[np.generic] | None]
    speeds_px_by_id: dict[int, npt.NDArray[np.float32]]
    contact_counts_by_id: dict[int, npt.NDArray[np.float32]]
    group_sizes_by_id: dict[int, npt.NDArray[np.float32]]

    @classmethod
    def for_pose(cls, poses: PoseEstimation) -> _VideoSocialCache:
        """Return the shared cache for ``poses``, building it on first use."""
        cache = _VIDEO_SOCIAL_CACHES.get(poses)
        if cache is None:
            cache = cls.build(poses)
            _VIDEO_SOCIAL_CACHES[poses] = cache
        return cache

    @classmethod
    def build(cls, poses: PoseEstimation) -> _VideoSocialCache:
        """Build shared social geometry and contact caches for all identities in ``poses``."""
        num_frames = poses.num_frames
        identities = list(poses.identities)
        logger.debug(
            "Building social feature cache for %d identities, %d frames",
            len(identities),
            num_frames,
        )
        has_seg = bool(poses.format_major_version >= 6 and poses.has_segmentation)

        seg_by_id: dict[int, npt.NDArray[np.generic] | None] = {}
        if has_seg:
            for curr_id in identities:
                seg_by_id[curr_id] = poses.get_segmentation_data(curr_id)

        hulls_by_id: dict[int, list[BaseGeometry | None]] = {}
        is_fallback_by_id: dict[int, npt.NDArray[np.bool_]] = {}
        for curr_id in identities:
            raw_hulls: list[BaseGeometry | None] = list(poses.get_identity_convex_hulls(curr_id))
            hulls_by_id[curr_id], is_fallback_by_id[curr_id] = (
                _populate_segmentation_fallback_hulls(raw_hulls, seg_by_id.get(curr_id))
            )

        speeds_px_by_id: dict[int, npt.NDArray[np.float32]] = {
            curr_id: _compute_identity_centroid_speeds(poses, curr_id) for curr_id in identities
        }

        contact_counts_by_id = {
            curr_id: np.full(num_frames, np.nan, dtype=np.float32) for curr_id in identities
        }
        group_sizes_by_id = {
            curr_id: np.full(num_frames, np.nan, dtype=np.float32) for curr_id in identities
        }
        for frame in range(num_frames):
            frame_seg_pts = {
                curr_id: _frame_contour_points(seg_by_id.get(curr_id), frame)
                for curr_id in identities
            }
            adjacency = _frame_contact_graph(frame, hulls_by_id, is_fallback_by_id, frame_seg_pts)
            for curr_id, neighbors in adjacency.items():
                contact_counts_by_id[curr_id][frame] = len(neighbors)
            for curr_id, size in _connected_component_sizes(adjacency).items():
                group_sizes_by_id[curr_id][frame] = size

        return cls(
            has_seg=has_seg,
            hulls_by_id=hulls_by_id,
            is_fallback_by_id=is_fallback_by_id,
            seg_by_id=seg_by_id,
            speeds_px_by_id=speeds_px_by_id,
            contact_counts_by_id=contact_counts_by_id,
            group_sizes_by_id=group_sizes_by_id,
        )


# Shared caches keyed weakly on the PoseEstimation object, so a cache is released
# together with the pose data it was built from.
_VIDEO_SOCIAL_CACHES: weakref.WeakKeyDictionary[PoseEstimation, _VideoSocialCache] = (
    weakref.WeakKeyDictionary()
)


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
        self._second_closest_identities = np.full(num_frames, -1, dtype=np.int16)
        self._closest_fov_identities = np.full(num_frames, -1, dtype=np.int16)
        self._fov_angles = np.full(num_frames, np.nan, dtype=np.float32)

        self._closest_distances = np.full(num_frames, np.nan, dtype=np.float32)
        self._closest_fov_distances = np.full(num_frames, np.nan, dtype=np.float32)
        self._second_closest_distances = np.full(num_frames, np.nan, dtype=np.float32)
        self._closest_segmentation_distances = np.full(num_frames, np.nan, dtype=np.float32)
        self._second_closest_segmentation_distances = np.full(num_frames, np.nan, dtype=np.float32)
        self._closest_segmentation_distances_body_units = np.full(
            num_frames, np.nan, dtype=np.float32
        )
        self._social_chain_bottleneck_distances = np.full(num_frames, np.nan, dtype=np.float32)
        self._closest_neighbor_speeds = np.full(num_frames, np.nan, dtype=np.float32)
        self._second_closest_neighbor_speeds = np.full(num_frames, np.nan, dtype=np.float32)

        cache = _VideoSocialCache.for_pose(poses)
        has_seg = cache.has_seg
        hulls_by_id = cache.hulls_by_id
        seg_by_id = cache.seg_by_id
        speeds_by_id = {
            curr_id: speeds * pixel_scale for curr_id, speeds in cache.speeds_px_by_id.items()
        }
        self_hulls = hulls_by_id[identity]

        # Contact count and connected group size come from the video's symmetric
        # per-frame contact graph, so all animals in a huddle cluster agree.
        self._social_contact_counts = cache.contact_counts_by_id[identity].copy()
        self._social_connected_group_sizes = cache.group_sizes_by_id[identity].copy()

        for frame in range(num_frames):
            points, mask = poses.get_points(frame, identity, pixel_scale)
            self_shape = self_hulls[frame]

            # Skip this frame if neither valid keypoints nor a fallback segmentation hull exist.
            if self_shape is None:
                continue

            # Nose and base-neck keypoints define the subject's facing direction for
            # field-of-view (FoV) features, whereas body-hull proximity depends only
            # on valid convex hulls so nose occlusion during huddling does not drop
            # closest-neighbor identity or distance measurements.
            has_fov_keypoints = (
                points is not None
                and mask is not None
                and mask[idx.NOSE] == 1
                and mask[idx.BASE_NECK] == 1
            )
            if has_fov_keypoints:
                self_base_neck_point = points[idx.BASE_NECK, :]
                self_nose_point = points[idx.NOSE, :]

            closest_dist: float | None = None
            second_closest_dist: float | None = None
            closest_fov_dist: float | None = None

            for curr_id in other_identities:
                other_shape = hulls_by_id[curr_id][frame]
                if other_shape is None:
                    continue

                curr_dist = float(self_shape.distance(other_shape))

                if closest_dist is None or curr_dist < closest_dist:
                    second_closest_dist = closest_dist
                    self._second_closest_identities[frame] = self._closest_identities[frame]
                    closest_dist = curr_dist
                    self._closest_identities[frame] = curr_id
                elif second_closest_dist is None or curr_dist < second_closest_dist:
                    second_closest_dist = curr_dist
                    self._second_closest_identities[frame] = curr_id

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

            pts_self = _frame_contour_points(seg_by_id.get(identity), frame)
            pts_closest: npt.NDArray[np.float32] | None = None
            seg_dist_px: float | None = None

            if closest_dist is not None:
                self._closest_distances[frame] = closest_dist * self._pixel_scale
                closest_id = int(self._closest_identities[frame])
                self._closest_neighbor_speeds[frame] = speeds_by_id[closest_id][frame]

                seg_dist_px = closest_dist
                if has_seg:
                    pts_closest = _frame_contour_points(seg_by_id.get(closest_id), frame)
                    seg_dist_px = _contour_min_distance(pts_self, pts_closest, closest_dist)

                self._closest_segmentation_distances[frame] = seg_dist_px * self._pixel_scale
                body_char_len = _characteristic_body_length(self_shape)
                self._closest_segmentation_distances_body_units[frame] = (
                    seg_dist_px / body_char_len
                )

            if second_closest_dist is not None and seg_dist_px is not None:
                self._second_closest_distances[frame] = second_closest_dist * self._pixel_scale
                second_id = int(self._second_closest_identities[frame])
                self._second_closest_neighbor_speeds[frame] = speeds_by_id[second_id][frame]

                seg_dist_2_px = second_closest_dist
                pts_second: npt.NDArray[np.float32] | None = None
                if has_seg:
                    pts_second = _frame_contour_points(seg_by_id.get(second_id), frame)
                    seg_dist_2_px = _contour_min_distance(
                        pts_self, pts_second, second_closest_dist
                    )

                self._second_closest_segmentation_distances[frame] = (
                    seg_dist_2_px * self._pixel_scale
                )

                # Continuous single-linkage bottleneck distance connecting the triad
                # (identity, closest_id, second_id): the second-smallest edge of the
                # 3-animal triangle is the maximum edge of its minimum spanning tree,
                # independent of whether hull ranking and contour ranking agree.
                closest_id = int(self._closest_identities[frame])
                closest_hull = hulls_by_id[closest_id][frame]
                second_hull = hulls_by_id[second_id][frame]
                assert closest_hull is not None and second_hull is not None
                hull_12 = float(closest_hull.distance(second_hull))
                dist_12_px = (
                    _contour_min_distance(pts_closest, pts_second, hull_12) if has_seg else hull_12
                )
                bottleneck_px = sorted((seg_dist_px, seg_dist_2_px, dist_12_px))[1]
                self._social_chain_bottleneck_distances[frame] = bottleneck_px * self._pixel_scale

            if closest_fov_dist is not None:
                self._closest_fov_distances[frame] = closest_fov_dist * self._pixel_scale

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
    def second_closest_segmentation_distances(self) -> npt.NDArray[np.float32]:
        """Return the segmentation contour distances to the second-closest identity per frame."""
        return self._second_closest_segmentation_distances

    @property
    def closest_segmentation_distances_body_units(self) -> npt.NDArray[np.float32]:
        """Return the closest segmentation distance normalized by characteristic body size."""
        return self._closest_segmentation_distances_body_units

    @property
    def social_contact_counts(self) -> npt.NDArray[np.float32]:
        """Return the number of other identities directly in body contact per frame."""
        return self._social_contact_counts

    @property
    def social_connected_group_sizes(self) -> npt.NDArray[np.float32]:
        """Return the size of the subject's connected social contact cluster per frame."""
        return self._social_connected_group_sizes

    @property
    def social_chain_bottleneck_distances(self) -> npt.NDArray[np.float32]:
        """Return the single-linkage minimax distance connecting the 3 nearest animals per frame."""
        return self._social_chain_bottleneck_distances

    @property
    def closest_neighbor_speeds(self) -> npt.NDArray[np.float32]:
        """Return the centroid speed of the closest identity per frame."""
        return self._closest_neighbor_speeds

    @property
    def second_closest_neighbor_speeds(self) -> npt.NDArray[np.float32]:
        """Return the centroid speed of the second-closest identity per frame."""
        return self._second_closest_neighbor_speeds

    def compute_distances(
        self, closest_identities: npt.NDArray[np.integer]
    ) -> npt.NDArray[np.float32]:
        """Compute frame-wise distances between the subject's hull and the specified identity's hull.

        Args:
            closest_identities: Array of closest identity indices for each frame.

        Returns:
            Array of distances for each frame, with NaN where distance cannot be computed.
        """
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
