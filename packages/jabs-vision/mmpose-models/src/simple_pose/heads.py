from __future__ import annotations

import torch
from torch import Tensor

from mmpose.models.heads.heatmap_heads.ae_head import AssociativeEmbeddingHead
from mmpose.utils.typing import ConfigType, OptSampleList
from mmpose.registry import MODELS


@MODELS.register_module()
class AssociativeEmbeddingHeadNoKptWeight(AssociativeEmbeddingHead):
    """AE head that ignores keypoint weights in heatmap loss.

    The AE tag loss still uses per-instance keypoint indices/weights, but
    heatmap supervision is unweighted to avoid shape mismatches.
    """

    def loss(self,
             feats: tuple[Tensor, ...],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        pred_heatmaps, pred_tags = self.forward(feats)

        if not self.tag_per_keypoint:
            pred_tags = pred_tags.repeat((1, self.num_keypoints, 1, 1))

        gt_heatmaps = torch.stack(
            [d.gt_fields.heatmaps for d in batch_data_samples])
        gt_masks = torch.stack(
            [d.gt_fields.heatmap_mask for d in batch_data_samples])
        keypoint_indices = [
            d.gt_instance_labels.keypoint_indices for d in batch_data_samples
        ]

        loss_kpt = self.loss_module.keypoint_loss(
            pred_heatmaps, gt_heatmaps, None, gt_masks)
        loss_pull, loss_push = self.loss_module.tag_loss(
            pred_tags, keypoint_indices)

        return {
            "loss_kpt": loss_kpt,
            "loss_pull": loss_pull,
            "loss_push": loss_push,
        }
