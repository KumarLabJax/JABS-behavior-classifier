"""Segmentation feature extraction package for pose estimation analysis.

This package provides classes and utilities to compute segmentation-based features from pose estimation data,
including image moments, Hu moments, shape descriptors, and grouped segmentation features. These features
facilitate quantitative analysis of object shapes, contours, and their temporal dynamics in multi-subject
tracking scenarios.

Modules:
    - hu_moments: Extraction of Hu invariant moments from segmentation contours.
    - moment_cache: Efficient caching and retrieval of image moments for each frame and identity.
    - moments: Calculation of central and normalized image moments.
    - segment_group: Grouping and management of segmentation-based features.
    - shape_descriptors: Computation of geometric shape descriptors from segmentation data.
"""

from .hu_moments import HuMoments
from .moment_cache import MomentInfo
from .moments import Moments
from .segment_group import SegmentationFeatureGroup
from .shape_descriptors import ShapeDescriptors
