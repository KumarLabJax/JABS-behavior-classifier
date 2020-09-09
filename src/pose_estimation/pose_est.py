import enum
from abc import ABC, abstractmethod


class PoseEstimation(ABC):
    """

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
        return self._num_frames

    @property
    def identities(self):
        """ return list of integer identities generated from file """
        return self._identities

    @abstractmethod
    def get_points(self, frame_index, identity):
        pass

    @abstractmethod
    def get_identity_poses(self, identity):
        pass

    @abstractmethod
    def identity_mask(self, identity):
        pass
