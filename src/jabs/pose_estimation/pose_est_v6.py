from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .pose_est_v5 import PoseEstimationV5


class PoseEstimationV6(PoseEstimationV5):
    """Pose estimation handler for version 6 pose files with segmentation support.

    Extends PoseEstimationV5 to add reading and management of image segmentation data
    (instance and long-term segmentation IDs, segmentation flags, and segmentation masks)
    from pose v6 HDF5 files. Provides methods to access segmentation data per identity
    and per frame, as well as utilities for sorting and retrieving segmentation arrays.

    Args:
        file_path (Path): Path to the pose HDF5 file.
        cache_dir (Path | None): Optional cache directory for intermediate data.
        fps (int): Frames per second for the video.

    Methods:
        get_seg_id(frame_index, identity): Get long-term segmentation ID for a frame and identity.
        get_segmentation_data(identity): Get segmentation mask array for an identity.
        get_segmentation_flags(identity): Get segmentation internal/external flags for an identity.
        get_segmentation_data_per_frame(frame_index, identity): Get segmentation mask for a specific frame and identity.
        format_major_version: Returns the major version of the pose file format (6).
    """

    def __init__(self, file_path: Path, cache_dir: Path | None = None, fps: int = 30):
        super().__init__(file_path, cache_dir, fps)

        # v6 properties
        # Image segmentation data read from pose v6 files.
        self._segmentation_dict: dict[str, np.ndarray | None] = {
            "instance_seg_id": None,
            "longterm_seg_id": None,
            "seg_external_flag": None,
            "seg_data": None,
        }

        # open the hdf5 pose file and extract segmentation data, this is not cached
        with h5py.File(self._path, "r") as pose_h5:
            for seg_key in set(pose_h5["poseest"].keys()) & set(self._segmentation_dict.keys()):
                self._segmentation_dict[seg_key] = pose_h5[f"poseest/{seg_key}"][:]
            # transpose seg_data similar to the way the points are transposed.

        # sort the segmentation data
        if self._segmentation_dict["seg_data"] is not None:
            self._segmentation_dict["seg_data"] = self._segmentation_sort(
                self._segmentation_dict["seg_data"],
                self._segmentation_dict["longterm_seg_id"],
            )
        if self._segmentation_dict["seg_external_flag"] is not None:
            self._segmentation_dict["seg_external_flag"] = self._segmentation_sort(
                self._segmentation_dict["seg_external_flag"],
                self._segmentation_dict["longterm_seg_id"],
            )

    def get_seg_id(self, frame_index: int, identity: int) -> np.ndarray[Any, Any] | None:
        """get segmentation for a given frame and identity."""
        if self._segmentation_dict["longterm_seg_id"] is None:
            return None
        else:
            return self._segmentation_dict["longterm_seg_id"][frame_index, identity]

    @classmethod
    def _segmentation_sort(cls, seg_data: np.ndarray, longterm_seg_id: np.ndarray) -> np.ndarray:
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
        """Given a particular identity, return the appropriate segmentation internal/external flags.

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

    def get_segmentation_data_per_frame(self, frame_index, identity: int) -> np.ndarray | None:
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
        """Returns the major version of the pose file format."""
        return 6

    @property
    def has_segmentation(self) -> bool:
        """Returns True if segmentation data is available."""
        return self._segmentation_dict["seg_data"] is not None
