"""Pose estimation adapters (NWB requires the [nwb] extra)."""

from jabs.io.internal.pose.hdf5 import PoseHDF5Adapter
from jabs.io.internal.pose.nwb import (
    IdentitySubject,
    PoseNWBAdapter,
    resolve_identity_subjects,
    sanitize_identity_name,
    subject_value_is_absent,
)

__all__ = [
    "IdentitySubject",
    "PoseHDF5Adapter",
    "PoseNWBAdapter",
    "resolve_identity_subjects",
    "sanitize_identity_name",
    "subject_value_is_absent",
]
