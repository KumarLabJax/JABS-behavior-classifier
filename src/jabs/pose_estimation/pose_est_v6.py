from typing import Any

import numpy as np
import typing
from pathlib import Path
import h5py

from .pose_est_v5 import PoseEstimationV5


class PoseEstimationV6(PoseEstimationV5):
    """Version 6 of the Pose Estimation class."""

    def __init__(
        self, file_path: Path, cache_dir: typing.Optional[Path] = None, fps: int = 30
    ):
        """
        Args:
            file_path: Path object representing the location of the pose file
            cache_dir: optional cache directory, used to cache convex
                hulls and transformed pose files for faster loading
            fps: frames per second, used for scaling time series
                features from "per frame" to "per second"
        """
        super().__init__(file_path, cache_dir, fps)

        # v6 properties
        # Image segmentation data read from pose v6 files.
        self._segmentation_dict = {
            "instance_seg_id": None,
            "longterm_seg_id": None,
            "seg_external_flag": None,
            "seg_data": None,
        }

        # open the hdf5 pose file and extract segmentation data.
        with h5py.File(self._path, "r") as pose_h5:
            for seg_key in set(pose_h5["poseest"].keys()) & set(
                self._segmentation_dict.keys()
            ):
                self._segmentation_dict[seg_key] = pose_h5[f"poseest/{seg_key}"][:]
            # transpose seg_data similar to the way the points are transposed.

        # sort the segmentation data
        self._segmentation_dict["seg_data"] = self._segmentation_sort(
            self._segmentation_dict["seg_data"],
            self._segmentation_dict["longterm_seg_id"],
        )
        self._segmentation_dict["seg_external_flag"] = self._segmentation_sort(
            self._segmentation_dict["seg_external_flag"],
            self._segmentation_dict["longterm_seg_id"],
        )

    def get_seg_id(
        self, frame_index: int, identity: int
    ) -> np.ndarray[Any, Any] | None:
        if self._segmentation_dict["longterm_seg_id"] is None:
            return None
        else:
            return self._segmentation_dict["longterm_seg_id"][frame_index, identity]

    @classmethod
    def _segmentation_sort(
        cls, seg_data: np.ndarray, longterm_seg_id: np.ndarray
    ) -> np.ndarray:
        """This method attempts to sort the segmentation data according to the longterm segmentation id.

        Args:
            seg_data: segmentation data with the first 2 dimensions
                being [frame,animal]
            longterm_seg_id: identities to sort by, beginning with 1 (0
                reserved for invalid data)

        Returns:
            sorted segmentation data
        """
        # Copy the data into the new array, sorted
        # Note that the -1 is the default for missing data
        sorted_seg_data = np.zeros_like(seg_data) - 1
        # Need to do a loop here because numpy doesn't allow sorting 2D indices for differently shaped arrays
        for animal_idx in np.arange(seg_data.shape[1]):
            # Detect which frames have valid data
            detected_idxs = longterm_seg_id == animal_idx + 1
            animal_preset_frames = np.any(detected_idxs, axis=1)
            # Sort the data
            sorted_seg_data[animal_preset_frames, animal_idx, ...] = seg_data[
                np.where(detected_idxs)
            ]
        return sorted_seg_data

    def get_segmentation_data(self, identity: int) -> np.ndarray | None:
        """Given a particular identity, return the appropriate segmentation data.

        Args:
            identity: identity to return segmentation data for.

        Returns:
            the ndarray of segmentation data (if it exists) otherwise
            the function returns None.
        """

        if self._segmentation_dict["seg_data"] is None:
            return None
        else:
            return self._segmentation_dict["seg_data"][:, identity, ...]

    def get_segmentation_flags(self, identity: int) -> np.ndarray | None:
        """Given a particular identity, return the appropriate segmentation
        internal/external flags.

        Args:
            identity: identity to return segmentation flags for.

        Returns:
            the ndarray of segmentation flags (if it exists) otherwise
            the function returns None.
        """

        if self._segmentation_dict["seg_external_flag"] is None:
            return None
        else:
            return self._segmentation_dict["seg_external_flag"][:, identity, ...]

    def get_segmentation_data_per_frame(
        self, frame_index, identity: int
    ) -> np.ndarray | None:
        """Given a particular identity, return the appropriate segmentation data.

        Args:
            identity: identity to return segmentation data for.
            frame_index: index of the frame to return segmentation data
                for.

        Returns:
            the ndarray of segmentation data (if it exists) otherwise
            the function returns None.
        """

        if self._segmentation_dict["seg_data"] is None:
            return None
        else:
            return self._segmentation_dict["seg_data"][frame_index, identity, ...]

    @property
    def format_major_version(self) -> int:
        return 6
