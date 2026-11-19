"""Conversion of legacy ``pose_est_v4``..``v8`` HDF5 files to ``jabs.pose-file``."""

import logging

from jabs.io.internal.pose_file.convert.convert import (
    ConversionReport,
    VideoOverrides,
    convert_legacy,
)
from jabs.io.internal.pose_file.convert.errors import LegacyConversionError
from jabs.io.internal.pose_file.convert.inspect import LegacyLayout, inspect_legacy
from jabs.io.internal.pose_file.convert.objects import object_components
from jabs.io.internal.pose_file.convert.plan import PlacementPlan, plan_placement
from jabs.io.internal.pose_file.convert.segmentation import segmentation_components
from jabs.io.internal.pose_file.convert.sources import pose_components

logger = logging.getLogger(__name__)

__all__ = [
    "ConversionReport",
    "LegacyConversionError",
    "LegacyLayout",
    "PlacementPlan",
    "VideoOverrides",
    "convert_legacy",
    "inspect_legacy",
    "object_components",
    "plan_placement",
    "pose_components",
    "segmentation_components",
]
