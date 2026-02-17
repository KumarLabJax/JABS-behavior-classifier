"""Tests for HRNet preprocessing helpers."""

import numpy as np
import torch

from jabs.vision.hrnet_leoxiaobin import preprocess_hrnet


def test_preprocess_hrnet_shape_dtype_and_normalization() -> None:
    """Frames are converted to normalized NCHW float tensors."""
    frame = np.array(
        [
            [[0, 255, 128], [255, 0, 128]],
            [[64, 64, 64], [128, 128, 128]],
        ],
        dtype=np.uint8,
    )

    tensor = preprocess_hrnet(frame)

    assert tensor.shape == (1, 3, 2, 2)
    assert tensor.dtype == torch.float32

    expected_ch0 = (0.0 - 0.45) / 0.225
    expected_ch1 = (1.0 - 0.45) / 0.225
    assert torch.isclose(tensor[0, 0, 0, 0], torch.tensor(expected_ch0))
    assert torch.isclose(tensor[0, 1, 0, 0], torch.tensor(expected_ch1))
