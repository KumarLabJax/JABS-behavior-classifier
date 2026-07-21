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
        with h5py.File(path, "w") as h5:
            self._write_v2(data, h5)

    def read(self, path: str | Path, data_type: type | None = None, **kwargs):  # noqa: D102
        raise NotImplementedError(
            "Reading pose HDF5 into PoseData is not implemented; use "
            "jabs.pose_estimation.PoseEstimationV2 for legacy v2 reads."
        )

    @staticmethod
    def _write_v2(data: PoseData, h5: h5py.File) -> None:
        """Write a single-identity PoseData into an open HDF5 file as v2."""
        if data.points.shape[0] != 1:
            raise ValueError(
                f"Legacy v2 is single-identity; got {data.points.shape[0]} identities"
            )
        if data.confidence is None:
            raise ValueError("Legacy v2 requires confidence; PoseData.confidence is None")

        # canonical (x, y) -> on-disk (y, x); float -> uint16
        points_yx = np.flip(data.points[0], axis=-1).astype(np.uint16)
        confidence = data.confidence[0].astype(np.float32)

        pose_grp = h5.require_group("poseest")
        points_ds = pose_grp.create_dataset("points", data=points_yx)
        points_ds.attrs["config"] = str(data.metadata.get("config", ""))
        points_ds.attrs["model"] = str(data.metadata.get("model", ""))
        pose_grp.create_dataset("confidence", data=confidence)
        pose_grp.attrs["version"] = np.asarray([2, 0], dtype=np.uint16)
