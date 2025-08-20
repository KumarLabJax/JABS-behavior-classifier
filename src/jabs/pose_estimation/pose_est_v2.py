from pathlib import Path

import h5py
import numpy as np

from .pose_est import MINIMUM_CONFIDENCE, PoseEstimation


class PoseEstimationV2(PoseEstimation):
    """read in pose_est_v2.h5 file"""

    def __init__(self, file_path: Path, cache_dir: Path | None = None, fps: int = 30):
        """initialize new object from h5 file

        Args:
            file_path: path to pose_est_v2.h5 file
            cache_dir: optional cache directory, used to cache convex
                hulls for faster loading
            fps: frames per second, used for scaling time series
                featuresfrom "per frame" to "per second"
        """
        super().__init__(file_path, cache_dir, fps)

        # we will make this look like the PoseEstimationV3 but with a single
        # identity so the main program won't care which type it is
        self._identities = [0]
        self._max_instances = 1

        self._path = file_path

        # open the hdf5 pose file
        with h5py.File(self._path, "r") as pose_h5:
            # extract data from the HDF5 file
            pose_grp = pose_h5["poseest"]

            # load contents
            # keypoints are stored as (y,x)
            self._points = np.flip(pose_grp["points"][:].astype(np.float64), axis=-1)
            self._point_mask = np.zeros(self._points.shape[:-1], dtype=np.uint16)
            self._point_mask[:] = pose_grp["confidence"][:] > MINIMUM_CONFIDENCE

            # get pixel size
            self._cm_per_pixel = pose_grp.attrs.get("cm_per_pixel", None)

        self._num_frames = self._points.shape[0]

        # build an array that indicates if the identity exists for a each frame
        # require at least 3 body points, not just tail
        init_func = np.vectorize(
            lambda x: 0 if np.sum(self._point_mask[x][:-2]) < 3 else 1,
            otypes=[np.uint8],
        )
        self._identity_mask = np.fromfunction(init_func, (self._num_frames,), dtype=np.int_)

    @property
    def identity_to_track(self):
        """get the identity to track mapping

        for pose_est_v2, this is always None because jabs doesn't do any track to identity mapping for the single
        mouse pose files
        """
        return None

    @property
    def format_major_version(self):
        """get the major version of the pose file format"""
        return 2

    def get_points(self, frame_index: int, identity: int, scale: float | None = None):
        """return points and point masks for an individual frame

        Args:
            frame_index: frame index of points and masks to be returned
            identity: included for compatibility with pose_est_v3.
                Should always be zero.
            scale: optional scale factor, set to cm_per_pixel to convert
                poses from pixel coordinates to cm coordinates

        Returns:
            numpy array of points (12,2), numpy array of point masks (12,)
        """
        if identity not in self.identities:
            raise ValueError("Invalid identity")

        if not self._identity_mask[frame_index]:
            return None, None

        if scale is not None:
            return self._points[frame_index] * scale, self._point_mask[frame_index]
        else:
            return self._points[frame_index], self._point_mask[frame_index]

    def get_identity_poses(self, identity: int, scale: float | None = None):
        """return all points and point masks

        Args:
            identity: included for compatibility with pose_est_v3.
                Should always be zero.
            scale: optional scale factor, set to cm_per_pixel to convert
                poses from pixel coordinates to cm coordinates

        Returns:
            numpy array of points (#frames, 12, 2), numpy array of point masks (#frames, 12)
        """
        if identity not in self.identities:
            raise ValueError("Invalid identity")

        if scale is not None:
            return self._points * scale, self._point_mask
        else:
            return self._points, self._point_mask

    def identity_mask(self, identity):
        """get the identity mask (indicates if specified identity is present in each frame)

        Args:
            identity: included for compatibility with pose_est_v3 interface. Should always be zero.

        Returns:
            numpy array of size (#frames,)
        """
        if identity not in self.identities:
            raise ValueError("Invalid identity")
        return self._identity_mask

    def get_identity_point_mask(self, identity):
        """get the point mask array for a given identity

        Args:
            identity: identity to return point mask for

        Returns:
            array of point masks (#frames, 12)
        """
        return self._point_mask

    def get_reduced_point_mask(self):
        """Returns a boolean array of length 12 indicating which keypoints are valid.

        Determines which keypoints are valid for any identity across all frames.

        Returns:
            numpy array of shape (12,) with boolean values indicating validity
            of each keypoint.
        """
        return np.any(self._point_mask, axis=0)
