from __future__ import annotations

from mmpose.datasets.transforms.formatting import PackPoseInputs
from mmpose.registry import TRANSFORMS


@TRANSFORMS.register_module()
class PackPoseInputsWithAE(PackPoseInputs):
    """PackPoseInputs variant that keeps AE-specific labels."""

    label_mapping_table = dict(PackPoseInputs.label_mapping_table)
    label_mapping_table.update(keypoint_indices="keypoint_indices")
