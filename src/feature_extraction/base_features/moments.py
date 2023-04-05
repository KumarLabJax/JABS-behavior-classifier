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
        # This runs about 8 times faster than the V1 method below.

        values = np.zeros((self._poses.num_frames, len(self._feature_names)))

        # Access segmentation data from underlying PoseEstimation object and
        # invoke v6's new get_segmentation_data method.
        seg_data = self._poses.get_segmentation_data(identity)

        for frame in range(values.shape[0]):
            contours = seg_data[frame, ...]
            contours = contours.reshape(
                contours.shape[0] * contours.shape[1], contours.shape[-1]
                ).astype(np.int32)

            # Compute the moments from
            Moments = cv2.moments(
                contours[(contours[..., 0] > -1) & (contours[..., 1] > -1)]
                )

            # Update the output array with the desired moments for each frame.
            for j in range(len(self._feature_names)):
                values[frame, j] = Moments[self._feature_names[j]]

        return values

    def per_frameV1(self, identity: int) -> np.ndarray:
        ''' [deprecated] I think this method can be improved.  Specifically, I
        don't think the canvas C is needed.  This method is slow.
        '''

        values = np.zeros((self._poses.num_frames, len(self._feature_names)))

        # Access segmentation data from underlying PoseEstimation object and
        # invoke v6's new get_segmentation_data method. 
        seg_data = self._poses.get_segmentation_data(identity)

        # Performance needs to be tested.  This proccess appears quite slow.
        # Not sure how it compares to the performance of computing other base
        # features.  I think you can compute moments directly from contours
        # this should speed up the function.
        # https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/
        for frame in range(values.shape[0]):
            # Create a canvas (2D array) to draw the contours onto.
            C = np.zeros(
                shape=(
                    np.max((seg_data[..., 0])),
                    np.max((seg_data[..., 1]))
                    )
                )

            # Separate all contours for a given identity & frame into a list,
            # similar to the return of findContours.
            contours = [seg_data[frame, cnt_num, :, :]
                        for cnt_num in range(seg_data.shape[1])]

            # Discard values < 0.
            contours = [contour[contour >= 0] for contour in contours]

            # Filter missing contours and coerce type to satisfy drawContours
            # method.
            contours = [contour.reshape((len(contour)//2, 2)).astype(int)
                        for contour in contours if len(contour) > 0]

            # Draw the contours onto the temporary canvas.
            cv2.drawContours(C, contours, -1, 1, 1)

            # Compute the moments from the canvas
            Moments = cv2.moments(C)

            # Update the output array with the desired moments for each frame.
            for j in range(len(self._feature_names)):
                values[frame, j] = Moments[self._feature_names[j]]

        return values
