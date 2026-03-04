"""Convert a JABS pose estimation file (any version) to NWB format."""

import logging
from pathlib import Path

import numpy as np

from jabs.core.abstract.pose_est import PoseEstimation
from jabs.core.types.pose import PoseData
from jabs.io import save
from jabs.pose_estimation import open_pose_file

logger = logging.getLogger(__name__)


def _segments_to_edges(segments) -> list[tuple[int, int]]:
    """Convert connected segment paths to a list of (src, dst) edge pairs.

    A segment like (0, 3, 6, 9) produces edges (0,3), (3,6), (6,9).

    Args:
        segments: Iterable of sequences of keypoint indices.

    Returns:
        List of (src, dst) index tuples.
    """
    edges = []
    for segment in segments:
        for i in range(len(segment) - 1):
            edges.append((int(segment[i]), int(segment[i + 1])))
    return edges


def pose_to_pose_data(pose: PoseEstimation) -> PoseData:
    """Convert any PoseEstimation object to a PoseData dataclass.

    Handles all supported JABS pose versions (v2-v8).  Optional attributes
    introduced in later versions (``static_objects``, ``external_identities``,
    ``cm_per_pixel``) are read with safe fallbacks for older formats.

    Args:
        pose: A loaded PoseEstimation object (any version).

    Returns:
        A PoseData instance ready for NWB export.
    """
    all_points = []
    all_point_masks = []
    for identity in pose.identities:
        points, mask = pose.get_identity_poses(identity)
        all_points.append(points)
        all_point_masks.append(mask)

    points_array = np.stack(all_points, axis=0)
    point_mask_array = np.stack(all_point_masks, axis=0)

    identity_mask_array = np.stack(
        [pose.identity_mask(identity) for identity in pose.identities],
        axis=0,
    )

    body_parts = [kpt.name for kpt in PoseEstimation.KeypointIndex]
    edges = _segments_to_edges(pose.get_connected_segments())

    cm_per_pixel = getattr(pose, "cm_per_pixel", None)
    static_objects = getattr(pose, "static_objects", {})
    external_ids = getattr(pose, "external_identities", None)

    file_hash = getattr(pose, "hash", None)
    metadata: dict = {
        "source_file": str(pose.pose_file),
        "pose_format_version": pose.format_major_version,
    }
    if file_hash is not None:
        metadata["source_file_hash"] = file_hash

    return PoseData(
        points=points_array,
        point_mask=point_mask_array,
        identity_mask=identity_mask_array,
        body_parts=body_parts,
        edges=edges,
        fps=pose.fps,
        cm_per_pixel=cm_per_pixel,
        static_objects=static_objects,
        external_ids=external_ids,
        metadata=metadata,
    )


def run_conversion(
    input_path: Path,
    output_path: Path,
    per_identity: bool = False,
    session_description: str | None = None,
) -> None:
    """Convert a JABS pose HDF5 file to NWB and write to disk.

    The pose format version is inferred from the filename (e.g.
    ``_pose_est_v6.h5`` → v6).  Supported versions: v2-v8.

    Args:
        input_path: Path to the input JABS pose HDF5 file.
        output_path: Destination path for the NWB file.  In per-identity mode
            this is used as a naming template; actual files are written
            alongside it as ``{stem}_{identity_name}.nwb``.
        per_identity: If True, write one NWB file per identity.
        session_description: Optional NWB session description string.

    Raises:
        ValueError: If the input file is not a recognised JABS pose file.
        FileNotFoundError: If the input file does not exist.
    """
    logger.info("Loading %s", input_path)
    pose = open_pose_file(input_path)
    identity_word = "identity" if pose.num_identities == 1 else "identities"
    logger.info(
        "%d %s, %d frames, %d fps", pose.num_identities, identity_word, pose.num_frames, pose.fps
    )

    pose_data = pose_to_pose_data(pose)

    write_kwargs: dict = {"per_identity_files": per_identity}
    if session_description is not None:
        write_kwargs["session_description"] = session_description

    logger.info("Writing NWB to %s", output_path)
    save(pose_data, output_path, **write_kwargs)
