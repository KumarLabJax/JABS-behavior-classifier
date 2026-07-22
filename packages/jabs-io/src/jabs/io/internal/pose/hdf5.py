"""HDF5 adapter for PoseData using the legacy ``pose_est_vN`` layout.

Legacy v2 layout::

    / (root)
      poseest/ (group)
        attrs: version = uint16[2] = [2, 0]
        points (dataset, uint16, shape n_frames x 12 x 2, order (y, x))
          attrs: config (str), model (str)
        confidence (dataset, float32, shape n_frames x 12)

Only single-identity v2 is supported today; the ``legacy=`` selector reserves
space for further legacy versions and the future backwards-compatible default.
"""

import logging
from pathlib import Path

import h5py
import numpy as np

from jabs.core.enums import JabsPoseVersion, StorageFormat
from jabs.core.types.pose import PoseData
from jabs.io.base import HDF5Adapter
from jabs.io.registry import register_adapter

logger = logging.getLogger(__name__)


@register_adapter(StorageFormat.HDF5, PoseData, priority=10)
class PoseHDF5Adapter(HDF5Adapter):
    """Write PoseData to a legacy ``pose_est_vN`` HDF5 file.

    Overrides ``write`` directly (like PredictionHDF5Adapter) so it can branch on
    the ``legacy`` version selector. ``read`` is not implemented in this increment;
    the legacy readers live in the ``jabs.pose_estimation`` monolith.
    """

    @classmethod
    def can_handle(cls, data_type: type) -> bool:  # noqa: D102
        return data_type is PoseData

    def _write_one(self, data, group) -> None:
        raise NotImplementedError("Use write() directly for PoseHDF5Adapter")

    def _read_one(self, group, data_type=None):
        raise NotImplementedError("Use read() directly for PoseHDF5Adapter")

    def write(  # noqa: D102
        self,
        data: PoseData,
        path: str | Path,
        *,
        legacy: JabsPoseVersion = JabsPoseVersion.V2,
        **kwargs,
    ) -> None:
        if legacy is not JabsPoseVersion.V2:
            raise ValueError(f"Unsupported legacy pose version: {legacy!r} (only V2 is supported)")
        # Validate and prepare BEFORE opening the file: h5py.File(path, "w") truncates
        # any existing file at open time, so a validation failure here must not touch disk.
        points_yx, confidence = self._prepare_v2(data)
        with h5py.File(path, "w") as h5:
            pose_grp = h5.require_group("poseest")
            points_ds = pose_grp.create_dataset("points", data=points_yx)
            points_ds.attrs["config"] = str(data.metadata.get("config", ""))
            points_ds.attrs["model"] = str(data.metadata.get("model", ""))
            pose_grp.create_dataset("confidence", data=confidence)
            pose_grp.attrs["version"] = np.asarray([2, 0], dtype=np.uint16)
            if data.cm_per_pixel is not None:
                pose_grp.attrs["cm_per_pixel"] = float(data.cm_per_pixel)

    def read(self, path: str | Path, data_type: type | None = None, **kwargs):  # noqa: D102
        raise NotImplementedError(
            "Reading pose HDF5 into PoseData is not implemented; use "
            "jabs.pose_estimation.PoseEstimationV2 for legacy v2 reads."
        )

    @staticmethod
    def _prepare_v2(data: PoseData) -> tuple[np.ndarray, np.ndarray]:
        """Validate a single-identity PoseData and build the on-disk v2 arrays.

        Pure (no file I/O) so all validation runs before the output file is opened.

        Args:
            data: The pose data to write.

        Returns:
            Tuple of (points as uint16 in (y, x) order, confidence as float32).

        Raises:
            ValueError: If the data is multi-identity, missing confidence, or has
                non-finite or out-of-uint16-range coordinates.
        """
        if data.points.shape[0] != 1:
            raise ValueError(
                f"Legacy v2 is single-identity; got {data.points.shape[0]} identities"
            )
        if data.confidence is None:
            raise ValueError("Legacy v2 requires confidence; PoseData.confidence is None")

        # canonical (x, y) -> on-disk (y, x). v2 stores pixel coordinates as uint16, so
        # guard against silent wrap/truncation before casting (NaN -> 0, negatives wrap).
        points_xy_to_yx = np.flip(data.points[0], axis=-1)
        if not np.all(np.isfinite(points_xy_to_yx)):
            raise ValueError("points contain non-finite values; cannot write legacy v2")
        uint16_max = np.iinfo(np.uint16).max
        if np.any(points_xy_to_yx < 0) or np.any(points_xy_to_yx > uint16_max):
            raise ValueError(
                f"points out of uint16 range [0, {uint16_max}] for legacy v2 pixel coordinates"
            )
        points_yx = np.rint(points_xy_to_yx).astype(np.uint16)
        confidence = data.confidence[0].astype(np.float32)
        return points_yx, confidence
