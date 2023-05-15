import cv2
import numpy as np

from src.pose_estimation import PoseEstimation
from src.feature_extraction.feature_base_class import Feature


class Moments(Feature):
    """feature for the image moments of the contours.
    """

    _name = 'moments'
    # Might be better to hard code the moment names
    _feature_names = \
        [feature_name for feature_name in cv2.moments(np.empty(0))]

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__(poses, pixel_scale)

    def per_frame(self, identity: int) -> np.ndarray:

        values = np.zeros((self._poses.num_frames, len(self._feature_names)), dtype=np.float32)

        # invoke v6's new get_segmentation_data method.
        seg_data = self._poses.get_segmentation_data(identity)

        for frame in range(values.shape[0]):
            contours = seg_data[frame, ...]
            contours = contours.reshape(
                contours.shape[0] * contours.shape[1], contours.shape[-1]
                ).astype(np.int32)

            Moments = cv2.moments(
                ((contours[(contours[..., 0] > -1) & (contours[..., 1] > -1)]) * self._pixel_scale).astype(np.float32) 
                )

            # Update the output array with the desired moments for each frame.
            for j in range(len(self._feature_names)):
                values[frame, j] = Moments[self._feature_names[j]]

        return values

    def per_frame_pixel(self, identity: int) -> np.ndarray:
        """This method does not convert the units from cm to px.  The purpose of this is so that the moments can
        be used to find the centroid of a mouse in frame.
        """
        values = np.zeros((self._poses.num_frames, len(self._feature_names)), dtype=np.float32)

        # invoke v6's new get_segmentation_data method.
        seg_data = self._poses.get_segmentation_data(identity)

        for frame in range(values.shape[0]):
            contours = seg_data[frame, ...]
            contours = contours.reshape(
                contours.shape[0] * contours.shape[1], contours.shape[-1]
                ).astype(np.int32)

            Moments = cv2.moments(
                (contours[(contours[..., 0] > -1) & (contours[..., 1] > -1)])
                )

            # Update the output array with the desired moments for each frame.
            for j in range(len(self._feature_names)):
                values[frame, j] = Moments[self._feature_names[j]]

        return values