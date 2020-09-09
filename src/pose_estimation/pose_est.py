import enum
from abc import ABC, abstractmethod


class PoseEstimation(ABC):
    """
    abstract base class for PoseEstimation objects. Used as the base class for
    PoseEstimationV2 and PoseEstimationV3
    """
    class KeypointIndex(enum.IntEnum):
        """ enum defining the 12 keypoint indexes """
        NOSE = 0
        LEFT_EAR = 1
        RIGHT_EAR = 2
        BASE_NECK = 3
        LEFT_FRONT_PAW = 4
        RIGHT_FRONT_PAW = 5
        CENTER_SPINE = 6
        LEFT_REAR_PAW = 7
        RIGHT_REAR_PAW = 8
        BASE_TAIL = 9
        MID_TAIL = 10
        TIP_TAIL = 11

    def __init__(self):
        self._num_frames = 0
        self._identities = []
        super().__init__()

    @property
    def num_frames(self):
        """ return the number of frames in the pose_est file """
        return self._num_frames

    @property
    def identities(self):
        """ return list of integer identities generated from file """
        return self._identities

    @abstractmethod
    def get_points(self, frame_index, identity):
        """
        return points and point masks for an individual frame
        :param frame_index: frame index of points and masks to be returned
        :param identity: identity to return points for
        :return: numpy array of points (12,2), numpy array of point masks (12,)
        """
        pass

    @abstractmethod
    def get_identity_poses(self, identity):
        """
        return all points and point masks
        :param identity: identity to return points for
        :return: numpy array of points (#frames, 12, 2), numpy array of point
        masks (#frames, 12)
        """
        pass

    @abstractmethod
    def identity_mask(self, identity):
        """
        get the identity mask (indicates if specified identity is present in
        each frame)
        :param identity: identity to get masks for
        :return: numpy array of size (#frames,)
        """
        pass
