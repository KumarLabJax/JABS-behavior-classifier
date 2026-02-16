"""Core interfaces and protocols for jabs-vision models."""

from abc import ABC, abstractmethod

import torch
from torch import Tensor

# =============================================================================
# Abstract Base Classes (for implementation guidance)
# =============================================================================


class BaseVisionModel(ABC, torch.nn.Module):
    """Abstract base class for vision models.

    Provides common functionality and enforces the interface contract.
    """

    @abstractmethod
    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Forward pass returning named outputs."""
        pass


# =============================================================================
# Output Key Conventions
# =============================================================================


class OutputKeys:
    """Standard keys for model output dictionaries.

    Using these constants prevents typos and documents the contract.
    """

    # Keypoint outputs
    HEATMAPS = "heatmaps"
    COORDS = "coords"
    KEYPOINT_CONFIDENCE = "keypoint_confidence"
    CONFIDENCE_MAPS = "confidence_maps"  # Learned confidence (not derived from heatmaps)
