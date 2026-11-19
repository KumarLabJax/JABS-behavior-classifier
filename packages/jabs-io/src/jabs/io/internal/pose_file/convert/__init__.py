"""Conversion of legacy ``pose_est_v4``..``v8`` HDF5 files to ``jabs.pose-file``."""

from jabs.io.internal.pose_file.convert.errors import LegacyConversionError
from jabs.io.internal.pose_file.convert.inspect import LegacyLayout, inspect_legacy

__all__ = ["LegacyConversionError", "LegacyLayout", "inspect_legacy"]
