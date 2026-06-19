"""Convert a JABS pose estimation file (any version) to NWB format."""

import datetime
import logging
from pathlib import Path

import h5py
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


def _h5_attr_to_jsonable(value: object) -> object:
    """Convert an HDF5 attribute value to a JSON-serializable Python object.

    ``h5py`` returns attribute values as numpy scalars or arrays, ``bytes``
    (for fixed-length string attributes), or native Python objects.  This
    normalizes them to plain JSON-friendly types (``str``, ``int``, ``float``,
    ``bool``, ``list``, ``None``) so they can be embedded losslessly in the NWB
    metadata JSON.  A value of an unrecognized type is preserved as its string
    representation rather than dropped.

    Args:
        value: A raw attribute value as returned by ``h5py``.

    Returns:
        A JSON-serializable representation of ``value``.
    """
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return [_h5_attr_to_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _h5_attr_to_jsonable(value.item())
    if isinstance(value, list | tuple):
        return [_h5_attr_to_jsonable(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    logger.warning(
        "Preserving HDF5 attribute of unsupported type %s as a string", type(value).__name__
    )
    return str(value)


def _collect_hdf5_attributes(path: Path) -> dict[str, dict[str, object]]:
    """Collect every attribute from every object in an HDF5 file.

    Walks the whole file - the root group, all sub-groups, and all datasets -
    recording each object's attributes keyed by its HDF5 path (``"/"`` for the
    root).  Objects that carry no attributes are omitted.  Attribute values are
    normalized to JSON-serializable types via :func:`_h5_attr_to_jsonable` so
    they survive the NWB metadata round-trip.

    Args:
        path: Path to the HDF5 file to read.

    Returns:
        Mapping of HDF5 object path to a dict of that object's attributes.
    """
    collected: dict[str, dict[str, object]] = {}

    def _record(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        if len(obj.attrs) == 0:
            return
        collected[name] = {key: _h5_attr_to_jsonable(val) for key, val in obj.attrs.items()}

    with h5py.File(path, "r") as h5:
        _record("/", h5)  # visititems does not visit the root group itself
        h5.visititems(_record)
    return collected


def pose_to_pose_data(
    pose: PoseEstimation,
    subjects: dict[str, dict] | None = None,
) -> PoseData:
    """Convert any PoseEstimation object to a PoseData dataclass.

    Handles all supported JABS pose versions (v2-v8).

    Every attribute stored anywhere in the source pose HDF5 file is captured
    into ``PoseData.metadata["hdf5_attributes"]`` (keyed by HDF5 object path)
    so arbitrary provenance attributes are not lost in the NWB conversion.

    Args:
        pose: A loaded PoseEstimation object (any version).
        subjects: Optional per-animal biological metadata, keyed by identity
            name (matching external_identities values).  Passed through
            directly to PoseData.subjects.

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

    per_identity_boxes = [pose.get_bounding_boxes(i) for i in pose.identities]
    bounding_boxes: np.ndarray | None = None
    if all(b is not None for b in per_identity_boxes):
        bounding_boxes = np.stack(per_identity_boxes, axis=0)  # (num_identities, num_frames, 2, 2)

    file_hash = getattr(pose, "hash", None)
    metadata: dict = {
        "source_file": str(pose.pose_file),
        "pose_format_version": pose.format_major_version,
    }
    if file_hash is not None:
        metadata["source_file_hash"] = file_hash

    hdf5_attributes = _collect_hdf5_attributes(Path(pose.pose_file))
    if hdf5_attributes:
        metadata["hdf5_attributes"] = hdf5_attributes

    return PoseData(
        points=points_array,
        point_mask=point_mask_array,
        identity_mask=identity_mask_array,
        body_parts=body_parts,
        edges=edges,
        fps=pose.fps,
        cm_per_pixel=cm_per_pixel,
        bounding_boxes=bounding_boxes,
        static_objects=static_objects,
        external_ids=external_ids,
        subjects=subjects,
        metadata=metadata,
    )


_SESSION_METADATA_FIELDS = frozenset(
    {
        "session_start_time",
        "experimenter",
        "lab",
        "institution",
        "experiment_description",
        "session_id",
    }
)


def _parse_session_start_time(value: str) -> datetime.datetime:
    """Parse an ISO 8601 datetime string into a timezone-aware datetime.

    Accepts any offset-aware ISO 8601 string.  The trailing ``Z`` shorthand
    for UTC is normalized to ``+00:00`` for Python 3.10 compatibility.  If the
    string carries no timezone offset, UTC is assumed and a warning is logged.

    Args:
        value: ISO 8601 datetime string, e.g. ``"2024-03-15T10:30:00-05:00"``
            or ``"2024-03-15T10:30:00Z"``.

    Returns:
        A timezone-aware :class:`datetime.datetime` object.

    Raises:
        ValueError: If ``value`` is not a string.
        ValueError: If the string cannot be parsed as an ISO 8601 datetime.
    """
    if not isinstance(value, str):
        raise ValueError(
            f"session_start_time must be a string, got {type(value).__name__!r}: {value!r}"
        )
    normalized = value
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        dt = datetime.datetime.fromisoformat(normalized)
    except ValueError as e:
        raise ValueError(
            f"session_start_time {value!r} is not a valid ISO 8601 datetime: {e}"
        ) from e
    if dt.tzinfo is None:
        logger.warning("session_start_time %r has no timezone; assuming UTC", value)
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt


def run_conversion(
    input_path: Path,
    output_path: Path,
    per_identity: bool = False,
    session_description: str | None = None,
    subjects: dict[str, dict] | None = None,
    session_metadata: dict | None = None,
) -> None:
    """Convert a JABS pose HDF5 file to NWB and write to disk.

    The pose format version is inferred from the filename (e.g.
    "_pose_est_v6.h5" → v6).  Supported versions: v2-v8.

    Args:
        input_path: Path to the input JABS pose HDF5 file.
        output_path: Destination path for the NWB file.  In per-identity mode
            this is used as a naming template; actual files are written
            alongside it as "{stem}_{identity_name}.nwb".
        per_identity: If True, write one NWB file per identity.
        session_description: Optional NWB session description string.
        subjects: Optional per-animal biological metadata dict, keyed by
            identity name.  See PoseData.subjects for the expected
            structure.
        session_metadata: Optional dict of NWB session-level metadata.
            Supported keys: ``session_start_time`` (ISO 8601 string),
            ``experimenter`` (str or list[str]), ``lab``, ``institution``,
            ``experiment_description``, ``session_id``.  Unknown keys are
            ignored with a warning.

    Raises:
        ValueError: If the input file is not a recognized JABS pose file, or
            if ``session_start_time`` cannot be parsed.
        FileNotFoundError: If the input file does not exist.
    """
    logger.info("Loading %s", input_path)
    pose = open_pose_file(input_path)
    identity_word = "identity" if pose.num_identities == 1 else "identities"
    logger.info(
        "%d %s, %d frames, %d fps", pose.num_identities, identity_word, pose.num_frames, pose.fps
    )

    pose_data = pose_to_pose_data(pose, subjects=subjects)

    write_kwargs: dict = {"per_identity_files": per_identity}
    if session_description is not None:
        write_kwargs["session_description"] = session_description

    if session_metadata is not None:
        unknown = set(session_metadata) - _SESSION_METADATA_FIELDS
        if unknown:
            logger.warning("Ignoring unrecognized session_metadata keys: %s", sorted(unknown))

        if "session_start_time" in session_metadata:
            write_kwargs["session_start_time"] = _parse_session_start_time(
                session_metadata["session_start_time"]
            )
        for key in _SESSION_METADATA_FIELDS - {"session_start_time"}:
            if key in session_metadata:
                write_kwargs[key] = session_metadata[key]

    logger.info("Writing NWB to %s", output_path)
    save(pose_data, output_path, **write_kwargs)
