"""The `base_features` package provides core feature extraction modules.

It includes classes for computing joint angles, angular velocities, centroid velocities, pairwise distances,
keypoint speeds, and velocity directions. These features serve as fundamental building blocks for higher-level
behavioral analysis.

Modules:
    - angles: Computes joint angles.
    - angular_velocity: Calculates angular velocities of joints.
    - base_group: Manages base feature modules.
    - centroid_velocity: Computes centroid velocity magnitude and direction.
    - pairwise_distances: Calculates pairwise distances between keypoints.
    - point_speeds: Computes per-frame speeds of keypoints.
    - point_velocities: Computes velocity directions for keypoints.
"""

from .angles import Angles
from .angular_velocity import AngularVelocity
from .base_group import BaseFeatureGroup
from .centroid_velocity import CentroidVelocityDir, CentroidVelocityMag
from .pairwise_distances import PairwisePointDistances
from .point_speeds import PointSpeeds
from .point_velocities import PointVelocityDirs
