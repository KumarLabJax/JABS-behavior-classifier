"""Conversion of legacy ``pose_est_v4``..``v8`` HDF5 files to ``jabs.pose-file``."""

from jabs.io.internal.pose_file.convert.errors import LegacyConversionError
from jabs.io.internal.pose_file.convert.inspect import LegacyLayout, inspect_legacy
from jabs.io.internal.pose_file.convert.objects import object_components
from jabs.io.internal.pose_file.convert.plan import PlacementPlan, plan_placement
from jabs.io.internal.pose_file.convert.sources import pose_components

__all__ = [
    "LegacyConversionError",
    "LegacyLayout",
    "PlacementPlan",
    "inspect_legacy",
    "object_components",
    "plan_placement",
    "pose_components",
]
