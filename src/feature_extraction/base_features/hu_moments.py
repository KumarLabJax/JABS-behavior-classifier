import cv2
import numpy as np

from src.pose_estimation import PoseEstimation
from src.feature_extraction.feature_base_class import Feature


class HuMoments(Feature):
    """Feature for the hu image moments of the segmentation contours.
    """

    _name = 'hu moments'
    _feature_names = [f"hu{i}" for i in range(1, 8)]

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__(poses, pixel_scale)

    def per_frame(self, identity: int) -> np.ndarray:
        values = np.zeros((self._poses.num_frames, len(self._feature_names)))
        
        # Computing moments... again, it would obviously be more efficient to
        # only make this calculation once, will return to this design decision
        # later when we see how features are being used.
        seg_data = self._poses.get_segmentation_data(identity)

        for frame in range(values.shape[0]):
            contours = seg_data[frame, ...]
            contours = contours.reshape(
                contours.shape[0] * contours.shape[1], contours.shape[-1]
                ).astype(np.int32)

            # Compute the moments from contours
            moments = cv2.moments(
                contours[(contours[..., 0] > -1) & (contours[..., 1] > -1)]
                )

            values[frame, :] = cv2.HuMoments(moments).T

        return values
