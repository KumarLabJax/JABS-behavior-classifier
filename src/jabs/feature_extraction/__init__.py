"""The `feature_extraction` package provides modules and utilities for extracting behavioral and pose-based features from pose estimation data.

It includes:
    - Core feature extraction classes for computing per-frame and windowed features.
    - Base features such as joint angles, angular velocities, centroid velocities, pairwise distances, and keypoint speeds.
    - Feature grouping and management utilities.
    - Versioning and configuration for feature extraction workflows.

This package serves as the foundation for higher-level behavioral analysis and downstream processing of pose data.
"""

from .features import FEATURE_VERSION, IdentityFeatures

DEFAULT_WINDOW_SIZE = 5

__all__ = [
    "DEFAULT_WINDOW_SIZE",
    "FEATURE_VERSION",
    "IdentityFeatures",
]
