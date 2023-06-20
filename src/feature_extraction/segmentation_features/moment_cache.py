import numpy as np
import cv2

from src.pose_estimation import PoseEstimation

class MomentInfo:
    """
    this info is needed to compute a number of different image moment features.
    It can be done once for a given identity, and then an instance of this
    object can be passed into all the features that need it

    get_moment(frame, key) retrieves the calculated image moment
    """

    def __init__(self, poses: PoseEstimation, identity: int,
                 pixel_scale: float):
        # These keys are necessary because opencv returns a dict which may not always be sorted
        self._moment_keys = [feature_name for feature_name in cv2.moments(np.empty(0))]
        self._moment_conversion_powers = [self.get_pixel_power(feature_name) for feature_name in self._moment_keys]
        self._poses = poses
        self._pixel_scale = pixel_scale
        self._moments = np.zeros((self._poses.num_frames, len(self._moment_keys)), dtype=np.float32)
        self._seg_data = self._poses.get_segmentation_data(identity)

        for frame in range(self._moments.shape[0]):
            contours = self.trim_contour_list(self._seg_data[frame, ...])
            # No segmentation data was present, skip calculating moments
            if len(contours)<1:
                continue
            moments = self.calculate_moments(contours)
            # Update the output array with the desired moments for each frame.
            for j in range(len(self._moment_keys)):
                # self._moments[frame, j] = moments[self._moment_keys[j]]*np.power(self._pixel_scale, self._moment_conversion_powers[j])
                self._moments[frame, j] = moments[self._moment_keys[j]]
    
    def get_pixel_power(self, key):
        """
        get the degree that pixels influence this image moment
        :param key: key of the image moment
        :return: power that should be used for converting from pixels to cm space
        """
        # Only works for image moments 0-9 on either dimension
        # opencv only does the first 3 moments (0-2)
        return int(key[-1]) + int(key[-2]) + 2

    def get_moment(self, frame, key):
        """
        retrieve a single moment value
        :param frame: frame to retrieve moment data
        :param key: key of moment data to retrieve
        :return: moment value
        """
        key_idx = self._moment_keys.index(key)
        return self._moments[frame, key_idx]

    def get_all_moments(self, frame):
        """
        retrieve moments for a frame
        :param frame: frame to retrieve moment data
        :return: dict of moment data
        """
        return {key:value for key,value in zip(self._moment_keys,self._moments[frame])}

    def trim_contour(self, arr):
        """
        removes -1s from contour data
        :param arr: contour, padded with -1s
        :return: opencv-complaint contour
        """
        assert arr.ndim == 2
        return_arr = arr[np.all(arr!=-1, axis=1),:]
        if len(return_arr)>0:
            return return_arr.astype(np.int32)

    def trim_contour_list(self, arr):
        """
        trims a fully padded 3D matrix into opencv-compliant contour list
        :param arr: a full matrix of contours, padded with -1s
        :return: opencv-complaint contour list
        """
        assert arr.ndim == 3
        return [self.trim_contour(x) for x in arr if np.any(x!=-1)]

    @staticmethod
    def calculate_moments(contour_list):
        """
        Renders the contour data onto a frame to calculate the moments
        :param contour_list: list of polygons, the format opencv returns from cv2.findContours
        :return: dict of cv2.moments image moments
        """
        frame_size = np.max(np.concatenate(contour_list)) + 1
        # Render the contours on a frame
        render = np.zeros([frame_size, frame_size, 1], dtype=np.uint8)
        _ = cv2.drawContours(render, contour_list, -1, [1], -1)
        return cv2.moments(render)
