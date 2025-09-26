from .pose_est_v6 import PoseEstimationV6


class PoseEstimationV7(PoseEstimationV6):
    """Pose estimation version 7

    Currently handled the same as v6 because we're not using the v7 dynamic_objects dataset yet.
    """

    @property
    def format_major_version(self) -> int:
        """Returns the major version of the pose file format."""
        return 7
