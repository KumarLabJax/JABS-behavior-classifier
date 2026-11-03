from __future__ import annotations

import typing
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from jabs.feature_extraction.feature_base_class import Feature
from jabs.pose_estimation import PoseEstimation

if typing.TYPE_CHECKING:
    from .moment_cache import MomentInfo


@dataclass
class _ContourStats:
    """Per-frame OpenCV contour measurements used by ShapeDescriptors, in pixel units."""

    rect_width: npt.NDArray[np.float64]
    rect_height: npt.NDArray[np.float64]
    perimeter: npt.NDArray[np.float64]
    hull_perimeter: npt.NDArray[np.float64]
    hull_area: npt.NDArray[np.float64]
    hole_area: npt.NDArray[np.float64]
    num_external: npt.NDArray[np.int64]
    num_internal: npt.NDArray[np.int64]


class ShapeDescriptors(Feature):
    """Feature related to shape descriptors of the segmentation data.

    Ellipse-fit was adopted from https://doi.org/10.1038/nmeth.2281
    Additional shape features are taken from definitions in http://dx.doi.org/10.5772/6237
    """

    _name = "shape_descriptor"
    # TODO: we're discarding centroid angle and ellipse-fit angle (theta)
    # These need to be handled similar to other angle terms (circular statistics)

    def __init__(self, poses: PoseEstimation, pixel_scale: float, moment_cache: MomentInfo):
        super().__init__(poses, pixel_scale)
        self._moment_cache = moment_cache
        self._pixel_scale = pixel_scale

    def per_frame(self, identity: int) -> dict[str, np.ndarray]:
        """Computes per-frame shape descriptor features for a specific identity.

        For each frame, calculates various shape features based on segmentation moments and contours,
        including ellipse dimensions, perimeter, elongation, rectangularity, convexity, solidity,
        Euler number, hole area ratio, and centroid speed.

        Args:
            identity (int): The identity index for which to compute shape descriptors.

        Returns:
            dict[str, np.ndarray]: Dictionary mapping feature names to per-frame arrays of values.
        """
        num_frames = self._poses.num_frames
        x = np.full(num_frames, np.nan, dtype=np.float32)
        y = np.full(num_frames, np.nan, dtype=np.float32)
        ellipse_w = np.full(num_frames, np.nan, dtype=np.float32)
        ellipse_l = np.full(num_frames, np.nan, dtype=np.float32)
        perimeter_sum = np.full(num_frames, np.nan, dtype=np.float32)
        elongation = np.full(num_frames, np.nan, dtype=np.float32)
        rectangularity = np.full(num_frames, np.nan, dtype=np.float32)
        convexity = np.full(num_frames, np.nan, dtype=np.float32)
        solidity = np.full(num_frames, np.nan, dtype=np.float32)
        euler_number = np.full(num_frames, np.nan, dtype=np.float32)
        hole_area_ratio = np.full(num_frames, np.nan, dtype=np.float32)

        # Only frames with segmentation data have values (avoids division by 0).
        # Since these features are based on cached moments, they are already adjusted
        # for pixel scaling.
        m00_all = self._moment_cache.get_moment_array("m00")
        frames = np.flatnonzero(~np.isnan(m00_all))

        def moment(key: str) -> npt.NDArray[np.float32]:
            return self._moment_cache.get_moment_array(key)[frames]

        m00 = m00_all[frames]

        # Ellipse features
        x[frames] = moment("m10") / m00
        y[frames] = moment("m01") / m00
        frame_x = x[frames]
        frame_y = y[frames]
        a = moment("m20") / m00 - np.square(frame_x)
        b = 2 * (moment("m11") / m00 - frame_x * frame_y)
        c = moment("m02") / m00 - np.square(frame_y)
        ellipse_w[frames] = 0.5 * np.sqrt(8 * (a + c - np.sqrt(np.square(b) + np.square(a - c))))
        ellipse_l[frames] = 0.5 * np.sqrt(8 * (a + c + np.sqrt(np.square(b) + np.square(a - c))))
        # Theta needs the be handled uniquely because it's only 0-pi and needs circular statistics
        # theta = 0.5 * np.arctan(2 * b / (a - c))

        # Shape features. The contour measurements have not yet been adjusted for pixel
        # scaling.
        contour_stats = self._contour_stats(frames)
        rect_w, rect_h = contour_stats.rect_width, contour_stats.rect_height
        perimeter_sum[frames] = contour_stats.perimeter * self._pixel_scale
        elongation[frames] = 1 - (np.minimum(rect_w, rect_h) / np.maximum(rect_w, rect_h))
        # the moments are float32, and these denominators were previously Python floats,
        # which NumPy converts to float32 before dividing; keep that to preserve values
        rectangularity[frames] = m00 / (rect_w * rect_h * self._pixel_scale**2).astype(np.float32)
        convexity[frames] = contour_stats.hull_perimeter / contour_stats.perimeter
        solidity[frames] = m00 / (contour_stats.hull_area * self._pixel_scale**2).astype(
            np.float32
        )
        euler_number[frames] = contour_stats.num_external - contour_stats.num_internal
        hole_area_ratio[frames] = (contour_stats.hole_area * self._pixel_scale**2) / m00

        # Calculate the centroid speeds
        centroid_speeds = np.hypot(np.gradient(x), np.gradient(y)) * self._poses.fps

        values = {
            "centroid_speed": centroid_speeds,
            "ellipse_w": ellipse_w,
            "ellipse_l": ellipse_l,
            "perimeter": perimeter_sum,
            "elongation": elongation,
            "rectangularity": rectangularity,
            "convexity": convexity,
            "solidity": solidity,
            "euler_number": euler_number,
            "hole_area_ratio": hole_area_ratio,
        }

        return values

    def _contour_stats(self, frames: npt.NDArray[np.intp]) -> _ContourStats:
        """Measure the segmentation contours of each frame with OpenCV.

        OpenCV's contour functions take one contour at a time, so this is the only part
        of the feature computed frame by frame.

        Args:
            frames: Frames to measure; each must have segmentation contours.

        Returns:
            Per-frame contour measurements, in pixel units, aligned with ``frames``.
        """
        n = len(frames)
        stats = _ContourStats(
            rect_width=np.empty(n, dtype=np.float64),
            rect_height=np.empty(n, dtype=np.float64),
            perimeter=np.empty(n, dtype=np.float64),
            hull_perimeter=np.empty(n, dtype=np.float64),
            hull_area=np.empty(n, dtype=np.float64),
            hole_area=np.empty(n, dtype=np.float64),
            num_external=np.empty(n, dtype=np.int64),
            num_internal=np.empty(n, dtype=np.int64),
        )
        for i, frame in enumerate(frames):
            contour_list = self._moment_cache.get_trimmed_contours(frame)
            contour_flags = self._moment_cache.get_flags(frame)[: len(contour_list)]
            all_points = np.concatenate(contour_list, axis=0)
            stats.rect_width[i], stats.rect_height[i] = cv2.minAreaRect(all_points)[1]
            stats.perimeter[i] = np.sum([cv2.arcLength(x, True) for x in contour_list])
            # The convex hull of the object (which may have multiple pieces) requires that
            # all lines between points are contained. Therefore, we can just concatenate
            # all points together when generating the hull
            hull = cv2.convexHull(all_points)
            stats.hull_perimeter[i] = cv2.arcLength(hull, True)
            #### WARNING
            # These contourArea uses a different calculation (Green formula) compared to the raster used in the cached moments.
            # This will cause slight differences in the actual value (eg solidity may be >1 despite the definition being convex=1)
            # TODO: To fix this, the hull and holes should be rendered on a raster and area calculated
            stats.hull_area[i] = cv2.contourArea(hull)
            # "Holes" are only internal contours
            stats.hole_area[i] = np.sum(
                [cv2.contourArea(contour_list[int(x)]) for x in np.where(contour_flags == 0)[0]]
            )
            stats.num_external[i] = np.sum(contour_flags == 1)
            stats.num_internal[i] = np.sum(contour_flags == 0)
        return stats
