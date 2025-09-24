from pathlib import Path

import h5py
import numpy as np

from .pose_est_v7 import PoseEstimationV7


class PoseEstimationV8(PoseEstimationV7):
    """Pose estimation version 8

    Adds bounding box support.
    """

    def __init__(self, file_path: Path, cache_dir: Path | None = None, fps: int = 30) -> None:
        super().__init__(file_path, cache_dir, fps)
        self._has_bounding_boxes = False

        # v8 properties
        with h5py.File(self._path, "r") as pose_h5:
            ds = pose_h5.get("poseest/bbox")
            if ds is None:
                # No bounding box data
                self._bboxes = None
                return
            bgen = ds.attrs.get("bboxes_generated", False)
            if not bool(bgen):
                # Bboxes not marked as generated
                self._bboxes = None
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
                    print(f"Warning: All identities masked in pose file: {file_path}")
                    self._num_identities = 0
            else:
                print(f"Warning: No identities found in pose file: {file_path}")
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

    @property
    def format_major_version(self) -> int:
        """Returns the major version of the pose file format."""
        return 8

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
