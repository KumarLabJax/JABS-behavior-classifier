"""Pose estimation adapters (NWB requires the [nwb] extra)."""

from jabs.io.internal.pose.hdf5 import PoseHDF5Adapter
from jabs.io.internal.pose.nwb import (
    IdentitySubject,
    PoseNWBAdapter,
    resolve_identity_subjects,
    sanitize_identity_name,
    subject_value_is_absent,
)
from jabs.io.internal.pose.nwb_video import (
    NWBVideoLinkInfo,
    link_external_video,
    read_video_link_info,
    relative_video_path,
    require_video_link_support,
)

__all__ = [
    "IdentitySubject",
    "NWBVideoLinkInfo",
    "PoseHDF5Adapter",
    "PoseNWBAdapter",
    "link_external_video",
    "read_video_link_info",
    "relative_video_path",
    "require_video_link_support",
    "resolve_identity_subjects",
    "sanitize_identity_name",
    "subject_value_is_absent",
]
