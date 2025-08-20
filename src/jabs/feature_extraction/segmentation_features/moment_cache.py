import cv2
import numpy as np

from jabs.pose_estimation import PoseEstimationV6


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
                moments = dict.fromkeys(self._moment_keys, np.nan)
            else:
                moments = self.calculate_moments(contours)
            # Update the output array with the desired moments for each frame.
            for j in range(len(self._moment_keys)):
                self._moments[frame, j] = moments[self._moment_keys[j]] * np.power(
                    self._pixel_scale, self._moment_conversion_powers[j]
                )

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
    def calculate_moments(contour_list):
        """Renders the contour data onto a frame to calculate the moments

        Args:
            contour_list: list of polygons, the format opencv returns
                from cv2.findContours

        Returns:
            dict of cv2.moments image moments
        """
        frame_size = np.max(np.concatenate(contour_list)) + 1
        # Render the contours on a frame
        render = np.zeros([frame_size, frame_size, 1], dtype=np.uint8)
        _ = cv2.drawContours(render, contour_list, -1, [1], -1)
        return cv2.moments(render)
