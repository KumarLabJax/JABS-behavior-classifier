import cv2
import numpy as np

from src.pose_estimation import PoseEstimation
from src.feature_extraction.feature_base_class import Feature


class EllipseFit(Feature):
    """Feature for the best fit ellipse from the segmentation contours.
    """

    _name = 'ellipse fit'
    _feature_names = ['x', 'y', 'a', 'b', 'c', 'w', 'l', 'theta']

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__(poses, pixel_scale)

    def per_frame(self, identity: int) -> np.ndarray:
        values = np.zeros((self._poses.num_frames, len(self._feature_names)), dtype=np.float32)

        seg_data = self._poses.get_segmentation_data(identity)

        x = self.feature_names.index('x')
        y = self.feature_names.index('y')
        a = self.feature_names.index('a')
        b = self.feature_names.index('b')
        c = self.feature_names.index('c')
        w = self.feature_names.index('w')
        ln = self.feature_names.index('l')
        t = self.feature_names.index('theta')

        for frame in range(values.shape[0]):
            contours = seg_data[frame, ...]
            contours = contours.reshape(
                contours.shape[0] * contours.shape[1], contours.shape[-1]
                ).astype(np.int32)

            moments = cv2.moments(
                ((contours[(contours[..., 0] > -1) & (contours[..., 1] > -1)]) * self._pixel_scale).astype(np.float32)
                )
            values[frame, x] = moments['m10'] / moments['m00']
            values[frame, y] = moments['m01'] / moments['m00']
            values[frame, a] = moments['m20'] / moments['m00'] \
                - values[frame, x]**2
            values[frame, b] = 2*(
                moments['m11'] / moments['m00'] -
                values[frame, x] * values[frame, y]
                )
            values[frame, c] = moments['m02'] / moments['m00'] \
                - values[frame, y]**2
            values[frame, w] = 0.5 * np.sqrt(
                8*(values[frame, a] + values[frame, c] -
                    np.sqrt(values[frame, b]**2 +
                            (values[frame, a] - values[frame, c])**2))
                )
            values[frame, ln] = 0.5 * np.sqrt(
                8*(values[frame, a] + values[frame, c] +
                    np.sqrt(values[frame, b]**2 +
                            (values[frame, a] - values[frame, c])**2))
                )
            values[frame, t] = 0.5 * np.arctan(
                2 * values[frame, b] / (values[frame, a] - values[frame, c])
                )

        return values
