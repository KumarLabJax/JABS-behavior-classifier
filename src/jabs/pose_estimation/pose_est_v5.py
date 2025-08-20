from pathlib import Path

import h5py
import numpy as np

from .pose_est import PoseEstimation
from .pose_est_v4 import PoseEstimationV4

OBJECTS_STORED_YX = [
    "lixit",
    "food_hopper",
]


class PoseEstimationV5(PoseEstimationV4):
    """Pose estimation handler for version 5 pose files with static object support.

    Extends PoseEstimationV4 to add reading and management of static object data
    (such as lixit and food hopper positions) from pose v5 HDF5 files. Handles
    additional datasets introduced in v5, including logic for different lixit
    keypoint configurations.

    Args:
        file_path (Path): Path to the pose HDF5 file.
        cache_dir (Path | None): Optional cache directory for intermediate data.
        fps (int): Frames per second for the video.

    """

    def __init__(self, file_path: Path, cache_dir: Path | None = None, fps: int = 30):
        super().__init__(file_path, cache_dir, fps)

        # with V5 we try to infer if we use the full connected segments
        # or the Envision connected segments
        self._connected_segments = None

        # V5 files are the same as V4, except they have some additional datasets
        # in addition to the posest data. The pose data is all loaded from
        # calling super().__init__(), so now we just need to load the additional
        # data

        self._static_objects = {}
        self._lixit_keypoints = 0

        # open the hdf5 pose file
        with h5py.File(self._path, "r") as pose_h5:
            # extract data from the HDF5 file
            for g in pose_h5:
                # skip over the poseest group, since that's already been
                # processed
                if g == "poseest":
                    continue

                # v5 adds a 'static_objects' dataset, but is otherwise the same as v4
                if g == "static_objects":
                    for d in pose_h5["static_objects"]:
                        static_object_data = pose_h5["static_objects"][d][:]
                        if d in OBJECTS_STORED_YX:
                            static_object_data = np.flip(static_object_data, axis=-1)
                        self._static_objects[d] = static_object_data

        if "lixit" in self._static_objects:
            # drop "lixit" from the static objects if it is an empty array
            if self._static_objects["lixit"].shape[0] == 0:
                del self._static_objects["lixit"]
            else:
                # if the lixit data is not empty, we need to get the number of
                # keypoints in the lixit data
                if self._static_objects["lixit"].ndim == 3:
                    # if the lixit data is 3D, it means we have 3 points per
                    # lixit (tip, left side, right side -- in that order) and the shape is #lixit x 3 x 2
                    self._lixit_keypoints = 3
                else:
                    # if the lixit data is 2D, it means we have 1 point per
                    # lixit (tip) and the shape is #lixit x 2
                    self._lixit_keypoints = 1

    def get_connected_segments(self):
        """Get the segments to use for rendering connections between the keypoints

        Returns:
            list of tuples, where each tuple contains the indexes of the keypoints
            that form a connected segment
        """
        # if we have already cached the connected segments, we're done
        if self._connected_segments is not None:
            return self._connected_segments

        # if we don't see any of the points that are missing from Envision Hydranet
        # pose we use NVSN_CONNECTED_SEGMENTS, otherwise we use FULL_CONNECTED_SEGMENTS
        points_missing_from_nvsn = [
            PoseEstimation.KeypointIndex.BASE_NECK.value,
            PoseEstimation.KeypointIndex.LEFT_FRONT_PAW.value,
            PoseEstimation.KeypointIndex.RIGHT_FRONT_PAW.value,
            PoseEstimation.KeypointIndex.CENTER_SPINE.value,
            PoseEstimation.KeypointIndex.LEFT_REAR_PAW.value,
            PoseEstimation.KeypointIndex.RIGHT_REAR_PAW.value,
            PoseEstimation.KeypointIndex.MID_TAIL.value,
        ]
        reduced_mask = self.get_reduced_point_mask()
        self._connected_segments = (
            PoseEstimation.FULL_CONNECTED_SEGMENTS
            if np.any(reduced_mask[points_missing_from_nvsn])
            else PoseEstimation.NVSN_CONNECTED_SEGMENTS
        )

        return self._connected_segments

    @property
    def format_major_version(self) -> int:
        """get the major version of the pose file format"""
        return 5

    @property
    def num_lixit_keypoints(self) -> int:
        """get the number of lixit keypoints"""
        return self._lixit_keypoints
