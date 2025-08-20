import enum

from jabs.pose_estimation import PoseEstimation


class AngleIndex(enum.IntEnum):
    """enum defining the indexes of the angle features"""

    NOSE_BASE_NECK_RIGHT_FRONT_PAW = 0
    NOSE_BASE_NECK_LEFT_FRONT_PAW = 1
    RIGHT_FRONT_PAW_BASE_NECK_CENTER_SPINE = 2
    LEFT_FRONT_PAW_BASE_NECK_CENTER_SPINE = 3
    BASE_NECK_CENTER_SPINE_BASE_TAIL = 4
    RIGHT_REAR_PAW_BASE_TAIL_CENTER_SPINE = 5
    LEFT_REAR_PAW_BASE_TAIL_CENTER_SPINE = 6
    RIGHT_REAR_PAW_BASE_TAIL_MID_TAIL = 7
    LEFT_REAR_PAW_BASE_TAIL_MID_TAIL = 8
    CENTER_SPINE_BASE_TAIL_MID_TAIL = 9
    BASE_TAIL_MID_TAIL_TIP_TAIL = 10

    @staticmethod
    def get_angle_name(i: "AngleIndex"):
        """map angle index to a string name"""
        strings = {
            AngleIndex.NOSE_BASE_NECK_RIGHT_FRONT_PAW: "NOSE-BASE_NECK-RIGHT_FRONT_PAW",
            AngleIndex.NOSE_BASE_NECK_LEFT_FRONT_PAW: "NOSE-BASE_NECK-LEFT_FRONT_PAW",
            AngleIndex.RIGHT_FRONT_PAW_BASE_NECK_CENTER_SPINE: "RIGHT_FRONT_PAW-BASE_NECK-CENTER_SPINE",
            AngleIndex.LEFT_FRONT_PAW_BASE_NECK_CENTER_SPINE: "LEFT_FRONT_PAW-BASE_NECK-CENTER_SPINE",
            AngleIndex.BASE_NECK_CENTER_SPINE_BASE_TAIL: "BASE_NECK-CENTER_SPINE-BASE_TAIL",
            AngleIndex.RIGHT_REAR_PAW_BASE_TAIL_CENTER_SPINE: "RIGHT_REAR_PAW-BASE_TAIL-CENTER_SPINE",
            AngleIndex.LEFT_REAR_PAW_BASE_TAIL_CENTER_SPINE: "LEFT_REAR_PAW-BASE_TAIL-CENTER_SPINE",
            AngleIndex.RIGHT_REAR_PAW_BASE_TAIL_MID_TAIL: "RIGHT_REAR_PAW-BASE_TAIL-MID_TAIL",
            AngleIndex.LEFT_REAR_PAW_BASE_TAIL_MID_TAIL: "LEFT_REAR_PAW-BASE_TAIL-MID_TAIL",
            AngleIndex.CENTER_SPINE_BASE_TAIL_MID_TAIL: "CENTER_SPINE-BASE_TAIL-MID_TAIL",
            AngleIndex.BASE_TAIL_MID_TAIL_TIP_TAIL: "BASE_TAIL-MID_TAIL-TIP_TAIL",
        }
        return strings[i]

    @staticmethod
    def get_angle_indices(i: "AngleIndex"):
        """get the keypoint indices for a given angle index"""
        angles = {
            AngleIndex.NOSE_BASE_NECK_RIGHT_FRONT_PAW: [
                PoseEstimation.KeypointIndex.NOSE,
                PoseEstimation.KeypointIndex.BASE_NECK,
                PoseEstimation.KeypointIndex.RIGHT_FRONT_PAW,
            ],
            AngleIndex.NOSE_BASE_NECK_LEFT_FRONT_PAW: [
                PoseEstimation.KeypointIndex.NOSE,
                PoseEstimation.KeypointIndex.BASE_NECK,
                PoseEstimation.KeypointIndex.LEFT_FRONT_PAW,
            ],
            AngleIndex.RIGHT_FRONT_PAW_BASE_NECK_CENTER_SPINE: [
                PoseEstimation.KeypointIndex.RIGHT_FRONT_PAW,
                PoseEstimation.KeypointIndex.BASE_NECK,
                PoseEstimation.KeypointIndex.CENTER_SPINE,
            ],
            AngleIndex.LEFT_FRONT_PAW_BASE_NECK_CENTER_SPINE: [
                PoseEstimation.KeypointIndex.LEFT_FRONT_PAW,
                PoseEstimation.KeypointIndex.BASE_NECK,
                PoseEstimation.KeypointIndex.CENTER_SPINE,
            ],
            AngleIndex.BASE_NECK_CENTER_SPINE_BASE_TAIL: [
                PoseEstimation.KeypointIndex.BASE_NECK,
                PoseEstimation.KeypointIndex.CENTER_SPINE,
                PoseEstimation.KeypointIndex.BASE_TAIL,
            ],
            AngleIndex.RIGHT_REAR_PAW_BASE_TAIL_CENTER_SPINE: [
                PoseEstimation.KeypointIndex.RIGHT_REAR_PAW,
                PoseEstimation.KeypointIndex.BASE_TAIL,
                PoseEstimation.KeypointIndex.CENTER_SPINE,
            ],
            AngleIndex.LEFT_REAR_PAW_BASE_TAIL_CENTER_SPINE: [
                PoseEstimation.KeypointIndex.LEFT_REAR_PAW,
                PoseEstimation.KeypointIndex.BASE_TAIL,
                PoseEstimation.KeypointIndex.CENTER_SPINE,
            ],
            AngleIndex.RIGHT_REAR_PAW_BASE_TAIL_MID_TAIL: [
                PoseEstimation.KeypointIndex.RIGHT_REAR_PAW,
                PoseEstimation.KeypointIndex.BASE_TAIL,
                PoseEstimation.KeypointIndex.MID_TAIL,
            ],
            AngleIndex.LEFT_REAR_PAW_BASE_TAIL_MID_TAIL: [
                PoseEstimation.KeypointIndex.LEFT_REAR_PAW,
                PoseEstimation.KeypointIndex.BASE_TAIL,
                PoseEstimation.KeypointIndex.MID_TAIL,
            ],
            AngleIndex.CENTER_SPINE_BASE_TAIL_MID_TAIL: [
                PoseEstimation.KeypointIndex.CENTER_SPINE,
                PoseEstimation.KeypointIndex.BASE_TAIL,
                PoseEstimation.KeypointIndex.MID_TAIL,
            ],
            AngleIndex.BASE_TAIL_MID_TAIL_TIP_TAIL: [
                PoseEstimation.KeypointIndex.BASE_TAIL,
                PoseEstimation.KeypointIndex.MID_TAIL,
                PoseEstimation.KeypointIndex.TIP_TAIL,
            ],
        }
        return angles[i]
