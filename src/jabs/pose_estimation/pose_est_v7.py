"""Pose estimation handler for version 7 pose files with dynamic object support."""

import logging
from pathlib import Path

import h5py
import numpy as np

from jabs.core.types import DynamicObjectData

from .pose_est_v6 import PoseEstimationV6

logger = logging.getLogger(__name__)

# Attribute name on dynamic_objects/[name]/points datasets that specifies
# the axis ordering of the stored coordinates.  Valid values: "xy" or "yx".
# When this attribute is absent the reader defaults to "yx" (matching the
# fecal_boli network which was trained with HRNet and stores points as y, x).
_AXIS_ORDER_ATTR = "axis_order"
_DEFAULT_AXIS_ORDER = "yx"


class PoseEstimationV7(PoseEstimationV6):
    """Pose estimation handler for version 7 pose files with dynamic object support.

    Extends PoseEstimationV6 to add reading and management of dynamic object data
    (e.g. fecal boli positions) from pose v7 HDF5 files.  Dynamic objects differ
    from static objects in that their position or count may change over time, and
    predictions are made only at a subset of frames recorded in sample_indices.

    The dynamic_objects/[name]/points dataset may carry an axis_order
    attribute specifying whether coordinates are stored as "xy" or "yx".
    If the attribute is absent the reader defaults to "yx" and flips the last
    axis so that all data is returned in (x, y) order.

    Args:
        file_path: Path to the pose HDF5 file.
        cache_dir: Optional cache directory for intermediate data.
        fps: Frames per second for the video.
    """

    def __init__(self, file_path: Path, cache_dir: Path | None = None, fps: int = 30) -> None:
        """Initialize PoseEstimationV7 from an HDF5 pose file.

        Args:
            file_path: Path to the pose v7 HDF5 file.
            cache_dir: Optional cache directory for intermediate data.
            fps: Frames per second for the video.
        """
        super().__init__(file_path, cache_dir, fps)

        with h5py.File(self._path, "r") as pose_h5:
            if "dynamic_objects" not in pose_h5:
                return

            for obj_name in pose_h5["dynamic_objects"]:
                obj_group = pose_h5[f"dynamic_objects/{obj_name}"]

                required = {"points", "counts", "sample_indices"}
                if not required.issubset(obj_group.keys()):
                    logger.warning(
                        "Dynamic object %r is missing required datasets %r; skipping.",
                        obj_name,
                        required - set(obj_group.keys()),
                    )
                    continue

                points_ds = obj_group["points"]
                axis_order: str = points_ds.attrs.get(_AXIS_ORDER_ATTR, _DEFAULT_AXIS_ORDER)

                points = points_ds[:].astype(np.float64)
                counts = obj_group["counts"][:].astype(np.int64)
                sample_indices = obj_group["sample_indices"][:].astype(np.int64)

                if points.ndim not in (3, 4):
                    logger.warning(
                        "Dynamic object %r has unexpected points shape %s; skipping.",
                        obj_name,
                        points.shape,
                    )
                    continue

                if axis_order == "yx":
                    points = np.flip(points, axis=-1).copy()
                elif axis_order == "xy":
                    pass
                else:
                    logger.warning(
                        "Dynamic object %r has unknown axis_order=%r; defaulting to 'yx'.",
                        obj_name,
                        axis_order,
                    )
                    points = np.flip(points, axis=-1).copy()

                # Normalize to 4-D (n_predictions, max_count, n_keypoints, 2).
                # Some older single-keypoint objects (e.g. fecal_boli) are stored as 3-D
                # in the HDF5 file; expand to (n_predictions, max_count, 1, 2).
                if points.ndim == 3:
                    points = points[:, :, np.newaxis, :]

                self._dynamic_objects[obj_name] = DynamicObjectData(
                    points=points,
                    counts=counts,
                    sample_indices=sample_indices,
                )
                logger.debug(
                    "Loaded dynamic object %r: %d predictions, max_count=%d, n_keypoints=%d, axis_order=%r",
                    obj_name,
                    len(sample_indices),
                    points.shape[1],
                    points.shape[2],
                    axis_order,
                )

    @property
    def format_major_version(self) -> int:
        """Returns the major version of the pose file format."""
        return 7
