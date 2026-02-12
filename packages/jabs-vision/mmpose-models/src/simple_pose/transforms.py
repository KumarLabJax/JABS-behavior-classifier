from __future__ import annotations

from mmpose.datasets.transforms.formatting import PackPoseInputs
from mmpose.registry import TRANSFORMS


@TRANSFORMS.register_module()
class PackPoseInputsWithAE(PackPoseInputs):
    """Pack pose inputs and preserve AE supervision labels.

    This class behaves the same as :class:`PackPoseInputs`, but extends
    ``label_mapping_table`` so ``results['keypoint_indices']`` is packed into
    ``data_sample.gt_instance_labels.keypoint_indices``.

    The extra field is required by Associative Embedding (AE) style losses to
    match keypoints to person instances during training.
    """

    label_mapping_table = dict(PackPoseInputs.label_mapping_table)  # noqa: RUF012
    label_mapping_table.update(keypoint_indices="keypoint_indices")
