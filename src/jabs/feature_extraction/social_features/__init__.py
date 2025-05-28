"""Social feature extraction package.

This package provides classes and utilities to compute social interaction features from pose estimation data, including
pairwise distances, closest distances, and field-of-view (FoV) based metrics. These features are useful for analyzing
proximity and orientation-based social behaviors in multi-subject tracking scenarios.

Modules:
    - closest_distances: Closest distance between subject and nearest other animal.
    - closest_fov_angles: Angle to the closest animal within subject's FoV.
    - closest_fov_distances: Closest distance to animal within FoV.
    - pairwise_social_distances: Pairwise distances for keypoint subsets.
"""

from .closest_distances import ClosestDistances
from .closest_fov_angles import ClosestFovAngles
from .closest_fov_distances import ClosestFovDistances
from .pairwise_social_distances import (
    PairwiseSocialDistances,
    PairwiseSocialFovDistances,
)
from .social_group import SocialFeatureGroup
