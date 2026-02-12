from __future__ import annotations

import torch
from mmpose.models.heads.heatmap_heads.ae_head import AssociativeEmbeddingHead
from mmpose.registry import MODELS
from mmpose.utils.typing import ConfigType, OptSampleList
from torch import Tensor


@MODELS.register_module()
class AssociativeEmbeddingHeadNoKptWeight(AssociativeEmbeddingHead):
    """AE head that ignores keypoint weights in heatmap loss.

    The AE tag loss still uses per-instance keypoint indices/weights, but
    heatmap supervision is unweighted to avoid shape mismatches.
    """

    def loss(self,
             feats: tuple[Tensor, ...],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:  # noqa: B006
        """Compute the loss for the AE head without keypoint weights.

        Args:
            feats: Feature maps from the backbone.
            batch_data_samples: Batch of data samples containing ground truth.
            train_cfg: Training configuration. Defaults to {}.

        Returns:
            Dictionary containing loss_kpt, loss_pull, and loss_push.
        """
        if batch_data_samples is None:
            raise ValueError(
                "batch_data_samples must be provided when computing training "
                "losses."
            )
        if len(batch_data_samples) == 0:
            raise ValueError(
                "batch_data_samples must be a non-empty list of data samples."
            )

        pred_heatmaps, pred_tags = self.forward(feats)  # type: ignore[arg-type]

        if not self.tag_per_keypoint:
            pred_tags = pred_tags.repeat((1, self.num_keypoints, 1, 1))

        gt_heatmaps_list: list[Tensor] = []
        gt_masks_list: list[Tensor] = []
        for data_sample in batch_data_samples:
            heatmaps = data_sample.gt_fields.heatmaps # type: ignore
            heatmap_mask = data_sample.gt_fields.heatmap_mask # type: ignore
            if not isinstance(heatmaps, Tensor):
                raise TypeError("Expected gt_fields.heatmaps to be a Tensor.")
            if not isinstance(heatmap_mask, Tensor):
                raise TypeError("Expected gt_fields.heatmap_mask to be a Tensor.")
            gt_heatmaps_list.append(heatmaps)
            gt_masks_list.append(heatmap_mask)

        gt_heatmaps = torch.stack(gt_heatmaps_list)
        gt_masks = torch.stack(gt_masks_list)
        keypoint_indices = [
            d.gt_instance_labels.keypoint_indices for d in batch_data_samples # type: ignore
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
