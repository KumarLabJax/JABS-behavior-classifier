from pathlib import Path

import h5py
import numpy as np

from jabs.constants import COMPRESSION, COMPRESSION_OPTS_DEFAULT

from .pose_est import MINIMUM_CONFIDENCE, PoseEstimation, PoseHashException


class _CacheFileVersion(Exception):
    pass


class PoseIdEmbeddingException(Exception):
    """Exception raised for invalid instance_embed_id values in pose file."""

    pass


class PoseEstimationV4(PoseEstimation):
    """
    Handler for version 4 pose estimation HDF5 files.

    This class parses and manages pose data from v4 pose files, including reordering
    keypoints by identity and handling identity masks. It provides access to pose
    points, confidence masks, and identity presence per frame. All pose estimation
    versions >= v4 inherit from this class, as it introduces long term identity.

    Args:
        file_path (Path): Path to the pose HDF5 file.
        cache_dir (Path | None): Optional cache directory for intermediate data.
        fps (int): Frames per second for the video.

    Properties:
        identity_to_track: Always returns None for v4+ files.
        format_major_version: Returns the major version of the pose file format (4).

    Methods:
        get_points(frame_index, identity, scale): Get points and mask for an identity in a frame.
        get_identity_poses(identity, scale): Get all points and masks for an identity.
        identity_mask(identity): Get the identity mask for a given identity.
        get_identity_point_mask(identity): Get the point mask array for a given identity.
    """

    # bump to force regeneration of pose cache files for v4 or any subclass
    _CACHE_FILE_VERSION = 4

    def __init__(self, file_path: Path, cache_dir: Path | None = None, fps: int = 30):
        super().__init__(file_path, cache_dir, fps)

        # these are not relevant for v4 pose files, but are included
        self._identity_to_track = None
        self._identity_map = None

        use_cache = False
        if cache_dir is not None:
            try:
                self._load_from_cache()
                use_cache = True
            except (OSError, KeyError, PoseHashException):
                # if load_from_cache() raises an exception, we'll read from
                # the source pose file below because use_cache will still be
                # set to false, just ignore the exceptions here
                pass

        if not use_cache:
            # open the hdf5 pose file
            with h5py.File(self._path, "r") as pose_h5:
                # extract data from the HDF5 file
                pose_grp = pose_h5["poseest"]

                # get pixel size
                self._cm_per_pixel = pose_grp.attrs.get("cm_per_pixel", None)

                # load contents
                # keypoints are stored as (y,x)
                all_points = np.flip(pose_grp["points"][:], axis=-1)
                all_confidence = pose_grp["confidence"][:]
                id_mask = pose_grp["id_mask"][:]
                instance_embed_id = pose_grp["instance_embed_id"][:]
                if "external_identity_mapping" in pose_grp:
                    # If the external identity mapping is stored as integers, convert to strings.
                    raw_ids = pose_grp["external_identity_mapping"][:]
                    if np.issubdtype(raw_ids.dtype, np.integer):
                        self._external_identities = [str(x) for x in raw_ids.tolist()]
                    else:
                        # If stored as strings (possibly bytes), decode to Python str if necessary.
                        self._external_identities = [
                            x.decode("utf-8") if isinstance(x, bytes) else str(x)
                            for x in raw_ids.tolist()
                        ]

                self._num_frames = len(all_points)

                max_instance_id = (
                    np.max(np.ma.array(instance_embed_id[...], mask=id_mask[...]))
                    if instance_embed_id.shape[1] > 0
                    else 0
                )

                if "instance_id_center" in pose_grp:
                    self._num_identities = pose_grp["instance_id_center"].shape[0]
                elif max_instance_id > 0:
                    self._num_identities = max_instance_id
                else:
                    print(f"Warning: No identities found in pose file: {file_path}")
                    self._num_identities = 0

            # Validate instance_embed_id range: must be in [0, self._num_identities]
            if max_instance_id > self._num_identities:
                raise PoseIdEmbeddingException(
                    f"Invalid instance_embed_id, values out of range: {file_path.name}"
                )

            # generate list of identities based on the max number of instances
            # in the pose file
            if self._num_identities > 0:
                self._identities = [*range(self._num_identities)]

                # tmp array used to reorder points
                # sometimes not all identities are used so need to shrink the array
                tmp_shape = np.array(np.shape(all_points))
                tmp_shape[1] = self._num_identities
                points_tmp = np.full(tmp_shape, np.nan, dtype=np.float64)

                # first use instance_embed_id to group points by identity
                points_tmp[
                    np.where(id_mask == 0)[0], instance_embed_id[id_mask == 0] - 1, :, :
                ] = all_points[id_mask == 0, :, :]

                # transpose to make the first index the "identity" rather than frame
                # indexes before transpose: [frame][ident][point idx][pt axis]
                # indexes after transpose: [ident][frame][point idx][pt axis]
                points_tmp = np.transpose(points_tmp, [1, 0, 2, 3])

                # transform confidence values for mask as well
                confidence_by_id_tmp = np.zeros(tmp_shape[:3], dtype=all_confidence.dtype)
                confidence_by_id_tmp[
                    np.where(id_mask == 0)[0], instance_embed_id[id_mask == 0] - 1, :
                ] = all_confidence[id_mask == 0, :]
                confidence_by_id = np.transpose(confidence_by_id_tmp, [1, 0, 2])

                # enforce partial poses get nan values
                points_tmp[confidence_by_id <= MINIMUM_CONFIDENCE] = np.nan

                # copy data into object
                self._points = points_tmp

                self._point_mask = confidence_by_id > MINIMUM_CONFIDENCE

                # build a mask for each identity that indicates if it exists or not
                # in the frame
                # require a minimum number of points to be > 3
                # this is because the convex hull requires 3 points
                init_func = np.vectorize(
                    lambda x, y: np.sum(self._point_mask[x][y][:-2]) >= 3,
                    otypes=[np.uint8],
                )

                self._identity_mask = np.fromfunction(
                    init_func, (self._num_identities, self._num_frames), dtype=np.int_
                )
            else:
                self._identities = []
                self._point_mask = None
                self._points = None
                self._identity_mask = None

            # cache pose data
            if cache_dir:
                self._cache_poses()

    @property
    def identity_to_track(self):
        """return identity_to_track mapping

        Note: returns None for >=v4 pose files because JABS doesn't do track to identity mapping, the pose file
        includes long term identity information
        """
        return None

    @property
    def format_major_version(self):
        """return the major version of the pose file format"""
        return 4

    def get_points(self, frame_index: int, identity: int, scale: float | None = None):
        """get points and mask for an identity for a given frame

        Args:
            frame_index: index of frame
            identity: identity that we want the points for
            scale: optional scale factor, set to cm_per_pixel to convert
            fps: video frames per second
        poses from pixel coordinates to cm coordinates

        Returns:
            points, mask if identity has data for this frame
        """
        if not self._identity_mask[identity, frame_index]:
            return None, None

        if scale is not None:
            return (
                self._points[identity, frame_index, ...] * scale,
                self._point_mask[identity, frame_index, :],
            )
        else:
            return (
                self._points[identity, frame_index, ...],
                self._point_mask[identity, frame_index, :],
            )

    def get_identity_poses(self, identity: int, scale: float | None = None):
        """return all points and point masks

        Args:
            identity: included for compatibility with pose_est_v3.
                Should
            scale: optional scale factor, set to cm_per_pixel to convert
        always be zero.
        poses from pixel coordinates to cm coordinates

        Returns:
            numpy array of points (#frames, 12, 2), numpy array of point
        masks (#frames, 12)
        """
        if scale is not None:
            return (
                self._points[identity, ...] * scale,
                self._point_mask[identity, ...],
            )
        else:
            return self._points[identity, ...], self._point_mask[identity, ...]

    def identity_mask(self, identity):
        """get the identity mask for a given identity"""
        return self._identity_mask[identity, :]

    def get_identity_point_mask(self, identity):
        """get the point mask array for a given identity

        Args:
            identity: identity to return point mask for

        Returns:
            array of point masks (#frames, 12)
        """
        return self._point_mask[identity, :]

    def get_reduced_point_mask(self):
        """Returns a boolean array of length 12 indicating which keypoints are valid.

        Determines which keypoints are valid for any identity across all frames.

        Returns:
            numpy array of shape (12,) with boolean values indicating validity
            of each keypoint.
        """
        return np.any(self._point_mask, axis=(0, 1))

    def _load_from_cache(self):
        """Load data from a cached pose file.

        We do some transformation of the pose files so that, for example, we can index them by identity. The
        cache file allows us to avoid doing this every time the pose file is loaded.

        Returns:
            None
        """
        cache_file_path = self._cache_file_path()

        with h5py.File(cache_file_path, "r") as cache_h5:
            if cache_h5.attrs["source_pose_hash"] != self._hash:
                raise PoseHashException

            pose_grp = cache_h5["poseest"]
            self._num_identities = int(cache_h5.attrs["num_identities"])
            self._num_frames = int(cache_h5.attrs["num_frames"])
            self._identities = [*range(self._num_identities)]
            if "external_identity_mapping" in pose_grp:
                # If the external identity mapping is stored as integers, convert to strings
                raw_ids = pose_grp["external_identity_mapping"][:]
                if np.issubdtype(raw_ids.dtype, np.integer):
                    self._external_identities = [str(x) for x in raw_ids.tolist()]
                else:
                    # If stored as strings (possibly bytes), decode to Python str if necessary.
                    self._external_identities = [
                        x.decode("utf-8") if isinstance(x, bytes) else str(x)
                        for x in raw_ids.tolist()
                    ]

            # get pixel size
            self._cm_per_pixel = pose_grp.attrs.get("cm_per_pixel", None)

            if self._num_identities > 0:
                self._points = pose_grp["points"][:]
                self._point_mask = pose_grp["point_mask"][:]
                self._identity_mask = pose_grp["identity_mask"][:]

    def _cache_poses(self):
        """cache the pose data in an h5 file in the project cache directory

        Returns:
            None
        """
        filename = self._path.name.replace(".h5", "_cache.h5")
        cache_file_path = self._cache_dir / filename

        with h5py.File(cache_file_path, "w") as cache_h5:
            cache_h5.attrs["cache_file_version"] = self._CACHE_FILE_VERSION
            cache_h5.attrs["source_pose_hash"] = self.hash
            cache_h5.attrs["num_identities"] = self._num_identities
            cache_h5.attrs["num_frames"] = self._num_frames
            group = cache_h5.create_group("poseest")
            if self._cm_per_pixel is not None:
                group.attrs["cm_per_pixel"] = self._cm_per_pixel

            if self._num_identities > 0:
                if self._external_identities:
                    # Always store external identities as strings in the cache
                    string_dt = h5py.string_dtype(encoding="utf-8")
                    group.create_dataset(
                        "external_identity_mapping",
                        data=np.array(self._external_identities, dtype=object),
                        dtype=string_dt,
                    )
                group.create_dataset(
                    "points",
                    data=self._points,
                    compression=COMPRESSION,
                    compression_opts=COMPRESSION_OPTS_DEFAULT,
                )
                group.create_dataset(
                    "point_mask",
                    data=self._point_mask,
                    compression=COMPRESSION,
                    compression_opts=COMPRESSION_OPTS_DEFAULT,
                )
                group.create_dataset(
                    "identity_mask",
                    data=self._identity_mask,
                    compression=COMPRESSION,
                    compression_opts=COMPRESSION_OPTS_DEFAULT,
                )
