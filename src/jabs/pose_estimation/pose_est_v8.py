from pathlib import Path

import h5py
import numpy as np

from jabs.constants import COMPRESSION, COMPRESSION_OPTS_DEFAULT

from .pose_est_v7 import PoseEstimationV7


class PoseEstimationV8(PoseEstimationV7):
    """Pose estimation version 8

    Adds bounding box support.
    """

    # force a bump in cache file version if either parent class or this class changes
    _CACHE_FILE_VERSION = PoseEstimationV7._CACHE_FILE_VERSION + 1

    def __init__(self, file_path: Path, cache_dir: Path | None = None, fps: int = 30) -> None:
        super().__init__(file_path, cache_dir, fps)
        self._has_bounding_boxes = False
        self._bboxes: np.ndarray | None = None

        # Try to load reorganized bboxes (identity-first) from cache
        # if not able to use cached bboxes, load from source
        if not self._load_bboxes_from_cache(cache_dir):
            self._load_from_h5(cache_dir)

    @property
    def format_major_version(self) -> int:
        """Returns the major version of the pose file format."""
        return 8

    def _load_from_h5(self, cache_dir: Path | None) -> None:
        """Load bounding boxes from source HDF5 file, reorganizing by identity.

        Args:
            cache_dir: directory to use for caching reorganized pose files, or None to disable caching.
        """
        with h5py.File(self._path, "r") as pose_h5:
            ds = pose_h5.get("poseest/bbox")
            if ds is None or not ds.attrs.get("bboxes_generated", False):
                # No bounding box data
                # Update cache to reflect absence of bboxes
                if cache_dir is not None:
                    try:
                        filename = self._path.name.replace(".h5", "_cache.h5")
                        cache_file_path = self._cache_dir / filename
                        with h5py.File(cache_file_path, "a") as cache_h5:
                            grp = cache_h5.require_group("poseest")
                            if "bboxes" in grp:
                                del grp["bboxes"]
                            empty_ds = grp.create_dataset("bboxes", shape=(0,), dtype=np.float32)
                            empty_ds.attrs["bboxes_generated"] = False
                    except OSError:
                        pass
                return

            bboxes = ds[:]

            # Load identity mapping arrays, needed to reorganize bboxes by identity
            instance_embed_id = pose_h5["poseest/instance_embed_id"][:]
            id_mask = pose_h5["poseest/id_mask"][:]

            # Determine number of identities the same way v4 does: mask out invalids and take max
            if instance_embed_id.shape[1] > 0:
                valid = id_mask == 0
                if valid.any():
                    # instance_embed_id is 1-based; take max over valid entries
                    self._num_identities = int(instance_embed_id[valid].max())
                else:
                    print(f"Warning: All identities masked in pose file: {self._path}")
                    self._num_identities = 0
            else:
                print(f"Warning: No identities found in pose file: {self._path}")
                self._num_identities = 0

            # Prepare an array grouped by identity, matching the v4 keypoint transform logic.
            # Shapes:
            #   bboxes: [frame][ident_instance][2][2]
            #   id_mask: [frame][ident_instance] with 0 where valid, 1 where padded/missing
            num_frames = bboxes.shape[0]
            bboxes_tmp = np.full(
                (num_frames, self._num_identities, 2, 2), np.nan, dtype=bboxes.dtype
            )

            # First use instance_embed_id to group bboxes by identity
            # IMPORTANT: valid entries are where id_mask == 0 (not == 1).
            valid = id_mask == 0
            if valid.any() and self._num_identities > 0:
                ids_flat = instance_embed_id[valid]
                pos = ids_flat > 0
                # Align rows and source slices with the filtered positives
                rows = np.where(valid)[0][pos]
                ids0 = ids_flat[pos] - 1
                # Guard against any out-of-range IDs
                in_range = ids0 < self._num_identities
                if in_range.any():
                    rows = rows[in_range]
                    ids0 = ids0[in_range]
                    src = bboxes[valid, :, :][pos, :, :][in_range, :, :]
                    bboxes_tmp[rows, ids0, :, :] = src

            # Transpose so that identity becomes the first index
            # Before: [frame][ident][2][2]
            # After:  [ident][frame][2][2]
            bboxes_by_ident = np.transpose(bboxes_tmp, (1, 0, 2, 3))
            self._bboxes = bboxes_by_ident.astype(np.float32)

            self._has_bounding_boxes = True

            # Write reorganized bboxes to cache for faster future loads
            if cache_dir is not None:
                filename = self._path.name.replace(".h5", "_cache.h5")
                cache_file_path = self._cache_dir / filename
                with h5py.File(cache_file_path, "a") as cache_h5:
                    cache_h5.attrs["cache_file_version"] = self._CACHE_FILE_VERSION
                    grp = cache_h5.require_group("poseest")
                    if "bboxes" in grp:
                        del grp["bboxes"]
                    ds_out = grp.create_dataset(
                        "bboxes",
                        data=self._bboxes,
                        compression=COMPRESSION,
                        compression_opts=COMPRESSION_OPTS_DEFAULT,
                    )
                    ds_out.attrs["bboxes_generated"] = True

    def _load_bboxes_from_cache(self, cache_dir: Path | None) -> bool:
        """Attempt to load bounding boxes from cache.

        Args:
            cache_dir: directory to use for caching reorganized pose files, or None to disable caching.

        Returns:
            True if bounding boxes were successfully loaded from cache, False otherwise.
        """
        use_cache = False
        if cache_dir is not None:
            try:
                filename = self._path.name.replace(".h5", "_cache.h5")
                cache_file_path = self._cache_dir / filename
                with h5py.File(cache_file_path, "r") as cache_h5:
                    if "poseest/bboxes" in cache_h5:
                        ds_cache = cache_h5["poseest/bboxes"]
                        bgen_cache = ds_cache.attrs.get("bboxes_generated", False)
                        if bgen_cache and ds_cache.size > 0:
                            self._bboxes = ds_cache[:]
                            self._has_bounding_boxes = True
                            use_cache = True
                        else:
                            # Cached dataset exists but marked as not generated; treat as absent
                            # set use_cache to True to skip source loading
                            use_cache = True
            except (OSError, KeyError):
                # Cache missing or unreadable; fall back to source
                pass
        return use_cache

    def get_bounding_boxes(self, identity: int) -> np.ndarray | None:
        """Get bounding box array for an identity index.

        Args:
            identity: identity index (0 to num_identities-1)

        Returns:
            bounding box array of shape [num_frames, 2, 2] or None if no bounding box data
            is available. bounding box format is [upper_left_x, upper_left_y], [lower_right_x, lower_right_y].
        """
        if self._bboxes is None:
            return None
        if identity < 0 or identity >= self._num_identities:
            raise ValueError(f"Identity {identity} out of range (0 to {self._num_identities - 1})")
        return self._bboxes[identity, :, :, :]

    @property
    def has_bounding_boxes(self) -> bool:
        """Returns True if bounding box data is available."""
        return self._has_bounding_boxes
