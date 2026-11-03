from math import comb

import cv2
import numpy as np
import numpy.typing as npt

from jabs.pose_estimation import PoseEstimationV6

# raw (spatial) moment keys returned by cv2.moments; these depend on the image origin,
# unlike the central (mu) and normalized (nu) moments
_RAW_MOMENT_KEYS = ("m00", "m10", "m01", "m20", "m11", "m02", "m30", "m21", "m12", "m03")


def _shift_raw_moments(moments: dict[str, float], x0: float, y0: float) -> dict[str, float]:
    """Translate raw image moments computed with the origin at (x0, y0) back to (0, 0).

    For pixel coordinates x = x' + x0 and y = y' + y0, the binomial expansion gives
    m_pq = sum_i sum_j C(p, i) C(q, j) x0^(p-i) y0^(q-j) m'_ij. Central and normalized
    moments are translation invariant and are returned unchanged.

    Args:
        moments: Moments from cv2.moments for an image whose (0, 0) pixel is at (x0, y0).
        x0: X offset of the image origin.
        y0: Y offset of the image origin.

    Returns:
        Copy of ``moments`` with the raw moments relative to (0, 0).
    """
    shifted = dict(moments)
    for key in _RAW_MOMENT_KEYS:
        p, q = int(key[1]), int(key[2])
        shifted[key] = sum(
            comb(p, i) * comb(q, j) * x0 ** (p - i) * y0 ** (q - j) * moments[f"m{i}{j}"]
            for i in range(p + 1)
            for j in range(q + 1)
        )
    return shifted


class MomentInfo:
    """this info is needed to compute a number of different image moment features.

    It can be done once for a given identity, and then an instance of this object can be passed into all the
    features that need it. Image moments provided here are adjusted for pixel scaling.

    get_moment(frame, key) retrieves the calculated image moment

    Args:
        poses (PoseEstimationV6): V6+ Pose estimation data for one video.
        identity (int): Identity to compute moments for.
        pixel_scale (float): Scale factor to convert pixel distances to cm.
    """

    def __init__(self, poses: PoseEstimationV6, identity: int, pixel_scale: float):
        # These keys are necessary because opencv returns a dict which may not always be sorted
        self._moment_keys = list(cv2.moments(np.empty(0)))
        self._moment_conversion_powers = [
            self.get_pixel_power(feature_name) for feature_name in self._moment_keys
        ]
        # per-moment factor converting from pixel to cm units
        moment_scale: npt.NDArray[np.float64] = np.power(
            float(pixel_scale), np.array(self._moment_conversion_powers, dtype=np.float64)
        )
        self._poses = poses
        self._pixel_scale = pixel_scale
        self._moments = np.zeros(
            (self._poses.num_frames, len(self._moment_keys)), dtype=np.float32
        )
        self._seg_data = self._poses.get_segmentation_data(identity)
        self._seg_flags = self._poses.get_segmentation_flags(identity)

        # Parse out the contour matrix into a list of contour lists
        tmp_contour_data = []
        for frame in range(self._moments.shape[0]):
            tmp_contour_data.append(self.trim_contour_list(self._seg_data[frame, ...]))

        self._seg_data = tmp_contour_data

        for frame, contours in enumerate(self._seg_data):
            # No segmentation data was present, skip calculating moments
            if len(contours) < 1:
                self._moments[frame, :] = np.nan
                continue
            moments = self.calculate_moments(contours)
            self._moments[frame, :] = [moments[key] for key in self._moment_keys] * moment_scale

    def get_pixel_power(self, key):
        """get the degree that pixels influence this image moment

        Args:
            key: key of the image moment

        Returns:
            power that should be used for converting from pixels to cm
            space
        """
        # Only works for image moments 0-9 on either dimension
        # opencv only does the first 3 moments (0-2)
        return int(key[-1]) + int(key[-2]) + 2

    def get_moment(self, frame, key):
        """retrieve a single moment value

        Args:
            frame: frame to retrieve moment data
            key: key of moment data to retrieve

        Returns:
            moment value
        """
        key_idx = self._moment_keys.index(key)
        return self._moments[frame, key_idx]

    def get_all_moments(self, frame):
        """retrieve moments for a frame

        Args:
            frame: frame to retrieve moment data

        Returns:
            dict of moment data
        """
        return dict(zip(self._moment_keys, self._moments[frame], strict=False))

    def get_trimmed_contours(self, frame):
        """retrieves a contour for a specific frame

        Args:
            frame: frame to retrieve contour data

        Returns:
            an opencv-complaint list of contours
        """
        return self._seg_data[frame]

    def get_flags(self, frame):
        """retrieves the internal/external flags for a specific frame

        Args:
            frame: frame to retrieve flags

        Returns:
            a binary vector of whether the segmentation contours are
            external (1) or internal (0)
        """
        return self._seg_flags[frame]

    def trim_contour(self, arr):
        """removes -1s from contour data

        Args:
            arr: contour, padded with -1s

        Returns:
            opencv-complaint contour
        """
        assert arr.ndim == 2
        return_arr = arr[np.all(arr != -1, axis=1), :]
        if len(return_arr) > 0:
            return return_arr.astype(np.int32)

    def trim_contour_list(self, arr):
        """trims a fully padded 3D matrix into opencv-compliant contour list

        Args:
            arr: a full matrix of contours, padded with -1s

        Returns:
            opencv-complaint contour list
        """
        assert arr.ndim == 3
        return [self.trim_contour(x) for x in arr if np.any(x != -1)]

    @staticmethod
    def calculate_moments(contour_list: list[npt.NDArray[np.int32]]) -> dict[str, float]:
        """Renders the contour data onto an image to calculate the moments

        The contours are filled into an image covering only their bounding box, rather
        than the whole frame, and the raw moments are then translated back to frame
        coordinates. This gives the same pixels as rendering onto a full frame while
        scanning far fewer of them.

        Args:
            contour_list: list of polygons, the format opencv returns
                from cv2.findContours

        Returns:
            dict of cv2.moments image moments, in frame pixel coordinates
        """
        points = np.concatenate(contour_list).reshape(-1, 2)
        x0, y0 = points.min(axis=0)
        width, height = points.max(axis=0) - (x0, y0) + 1
        # Render the contours, translated so the bounding box starts at (0, 0)
        render = np.zeros((int(height), int(width), 1), dtype=np.uint8)
        _ = cv2.drawContours(render, contour_list, -1, [1], -1, offset=(-int(x0), -int(y0)))
        return _shift_raw_moments(cv2.moments(render), float(x0), float(y0))
