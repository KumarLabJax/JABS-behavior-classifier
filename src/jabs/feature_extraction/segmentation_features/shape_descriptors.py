import typing

import cv2
import numpy as np

from jabs.feature_extraction.feature_base_class import Feature
from jabs.pose_estimation import PoseEstimation

if typing.TYPE_CHECKING:
    from .moment_cache import MomentInfo


class ShapeDescriptors(Feature):
    """Feature related to shape descriptors of the segmentation data.

    Ellipse-fit was adopted from https://doi.org/10.1038/nmeth.2281
    Additional shape features are taken from definitions in http://dx.doi.org/10.5772/6237
    """

    _name = "shape_descriptor"
    # TODO: we're discarding centroid angle and ellipse-fit angle (theta)
    # These need to be handled similar to other angle terms (circular statistics)

    def __init__(self, poses: PoseEstimation, pixel_scale: float, moment_cache: "MomentInfo"):
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
        x = np.full(self._poses.num_frames, np.nan, dtype=np.float32)
        y = np.full(self._poses.num_frames, np.nan, dtype=np.float32)
        ellipse_w = np.full(self._poses.num_frames, np.nan, dtype=np.float32)
        ellipse_l = np.full(self._poses.num_frames, np.nan, dtype=np.float32)
        perimeter_sum = np.full(self._poses.num_frames, np.nan, dtype=np.float32)
        elongation = np.full(self._poses.num_frames, np.nan, dtype=np.float32)
        rectangularity = np.full(self._poses.num_frames, np.nan, dtype=np.float32)
        convexity = np.full(self._poses.num_frames, np.nan, dtype=np.float32)
        solidity = np.full(self._poses.num_frames, np.nan, dtype=np.float32)
        euler_number = np.full(self._poses.num_frames, np.nan, dtype=np.float32)
        hole_area_ratio = np.full(self._poses.num_frames, np.nan, dtype=np.float32)

        # We don't use vectorized ops so that division by 0 safeties can be checked before calculation
        for frame in range(self._poses.num_frames):
            # Safety for division by 0 (no segmentation to calculate on)
            if np.isnan(self._moment_cache.get_moment(frame, "m00")):
                continue

            # Ellipse features
            # Since these features are based on cached moments, they are already adjusted for pixel scaling
            x[frame] = self._moment_cache.get_moment(frame, "m10") / self._moment_cache.get_moment(
                frame, "m00"
            )
            y[frame] = self._moment_cache.get_moment(frame, "m01") / self._moment_cache.get_moment(
                frame, "m00"
            )
            a = self._moment_cache.get_moment(frame, "m20") / self._moment_cache.get_moment(
                frame, "m00"
            ) - np.square(x[frame])
            b = 2 * (
                self._moment_cache.get_moment(frame, "m11")
                / self._moment_cache.get_moment(frame, "m00")
                - x[frame] * y[frame]
            )
            c = self._moment_cache.get_moment(frame, "m02") / self._moment_cache.get_moment(
                frame, "m00"
            ) - np.square(y[frame])
            ellipse_w[frame] = 0.5 * np.sqrt(
                8 * (a + c - np.sqrt(np.square(b) + np.square(a - c)))
            )
            ellipse_l[frame] = 0.5 * np.sqrt(
                8 * (a + c + np.sqrt(np.square(b) + np.square(a - c)))
            )
            # Theta needs the be handled uniquely because it's only 0-pi and needs circular statistics
            # theta = 0.5 * np.arctan(2 * b / (a - c))

            # Shape features
            # Pre-calculate some notable features
            # Note that these have not yet been adjusted for pixel scaling
            contour_list = self._moment_cache.get_trimmed_contours(frame)
            contour_flags = self._moment_cache.get_flags(frame)
            min_bound_rect = cv2.minAreaRect(np.concatenate(contour_list, axis=0))
            perimeters = [cv2.arcLength(x, True) for x in contour_list]
            # The convex hull of the object (which may have multiple pieces) requires that all lines between points are contained.
            # Therefore, we can just concatenate all points together when generating the hull
            hull = cv2.convexHull(np.concatenate(contour_list, axis=0))
            hull_perimeter = cv2.arcLength(hull, True)
            #### WARNING
            # These contourArea uses a different calculation (Green formula) compared to the raster used in the cached moments.
            # This will cause slight differences in the actual value (eg solidity may be >1 despite the definition being convex=1)
            # TODO: To fix this, the hull and holes should be rendered on a raster and area calculated
            hull_area = cv2.contourArea(hull)
            # "Holes" are only internal contours
            hole_areas = np.sum(
                [
                    cv2.contourArea(contour_list[int(x)])
                    for x in np.where(contour_flags[: len(contour_list)] == 0)[0]
                ]
            )

            # Place the shape features into the return value
            perimeter_sum[frame] = np.sum(perimeters) * self._pixel_scale
            elongation[frame] = 1 - (np.min(min_bound_rect[1]) / np.max(min_bound_rect[1]))
            rectangularity[frame] = self._moment_cache.get_moment(frame, "m00") / (
                min_bound_rect[1][0] * min_bound_rect[1][1] * self._pixel_scale**2
            )
            convexity[frame] = hull_perimeter / np.sum(perimeters)
            solidity[frame] = self._moment_cache.get_moment(frame, "m00") / (
                hull_area * self._pixel_scale**2
            )
            euler_number[frame] = np.sum(contour_flags[: len(contour_list)] == 1) - np.sum(
                contour_flags[: len(contour_list)] == 0
            )
            hole_area_ratio[frame] = (
                hole_areas * self._pixel_scale**2
            ) / self._moment_cache.get_moment(frame, "m00")

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
