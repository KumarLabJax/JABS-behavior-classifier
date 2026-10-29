"""Social feature extraction package.

This package provides classes and utilities to compute social interaction features from pose estimation data, including
pairwise distances, closest distances, multi-animal huddle topology, partner speed, and field-of-view (FoV) based
metrics. These features are useful for analyzing proximity, huddling, and orientation-based social behaviors in
multi-subject tracking scenarios.

Modules:
    - closest_distances: Closest, second-closest, and segmentation distances to neighboring animals.
    - closest_fov_angles: Angle to the closest animal within subject's FoV.
    - closest_fov_distances: Closest distance to animal within FoV.
    - huddle_features: Multi-animal contact count, connected group size, and closest neighbor speed.
    - pairwise_social_distances: Pairwise distances for keypoint subsets.
"""

from .closest_distances import ClosestDistances
from .closest_fov_angles import ClosestFovAngles
from .closest_fov_distances import ClosestFovDistances
from .huddle_features import HuddleFeatures
from .pairwise_social_distances import (
    PairwiseSocialDistances,
    PairwiseSocialFovDistances,
)
from .social_group import SocialFeatureGroup

__all__ = [
    "ClosestDistances",
    "ClosestFovAngles",
    "ClosestFovDistances",
    "HuddleFeatures",
    "PairwiseSocialDistances",
    "PairwiseSocialFovDistances",
    "SocialFeatureGroup",
]
