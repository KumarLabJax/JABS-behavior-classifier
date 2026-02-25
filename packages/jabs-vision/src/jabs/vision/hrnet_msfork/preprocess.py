"""Preprocessing utilities for HRNet inference."""

import numpy as np
import torch
from torch import Tensor

_HRNET_MEAN = torch.tensor([0.45, 0.45, 0.45], dtype=torch.float32).view(1, 3, 1, 1)
_HRNET_STD = torch.tensor([0.225, 0.225, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


def preprocess_hrnet(frame: np.ndarray) -> Tensor:
    """Convert uint8 HWC frame into normalized NCHW tensor for HRNet.

    Args:
        frame: Input frame in HWC format.

    Returns:
        Tensor of shape (1, 3, H, W), normalized for HRNet.
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected frame shape (H, W, 3), received {frame.shape}")

    image = torch.as_tensor(frame, dtype=torch.float32) / 255.0
    image = image.unsqueeze(0).permute(0, 3, 1, 2)
    return (image - _HRNET_MEAN) / _HRNET_STD
