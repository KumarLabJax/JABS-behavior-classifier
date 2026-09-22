"""Convert a JABS pose estimation file (any version) to NWB format."""

import collections
import dataclasses
import datetime
import logging
from pathlib import Path

import h5py
import numpy as np

from jabs.core.abstract.pose_est import PoseEstimation
from jabs.core.types.pose import PoseData
from jabs.io import save
from jabs.io.internal.pose import (
    resolve_identity_subjects,
    sanitize_identity_name,
    subject_value_is_absent,
)
from jabs.pose_estimation import open_pose_file
from jabs.scripts.cli.dandi_subject_metadata import validate_subjects

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
        # h5py returns scalar attributes as 0-d arrays; .tolist() yields a bare
        # scalar (not a list), so unwrap to the scalar before recursing.
        if value.shape == ():
            return _h5_attr_to_jsonable(value.item())
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


# Key in a --subjects entry that renames the identity it applies to. Stripped before the
# metadata reaches PoseData.subjects: it is not a pynwb Subject field, and leaving it in
# would also write it through to jabs_metadata.
_IDENTITY_NAME_KEY = "name"


def _carries_name(entry) -> bool:
    """Whether this identity's subject entry has a ``name`` key at all, blank or not.

    Metadata that is not a dict answers False: the CLI only checks that the top level of
    the subjects JSON is an object, so a non-dict entry reaches here, and reporting it is
    ``subject_metadata_problems``'s job. Touching it here would raise an AttributeError
    in place of the message written for it.
    """
    return isinstance(entry.metadata, dict) and _IDENTITY_NAME_KEY in entry.metadata


def _name_override(entry) -> object:
    """The ``name`` this identity asks for, or None when its metadata is unusable."""
    return entry.metadata.get(_IDENTITY_NAME_KEY) if isinstance(entry.metadata, dict) else None


def _apply_identity_names(data: PoseData) -> PoseData:
    """Name identities from the ``name`` field of their subject metadata.

    A pose file without external identities leaves its animals called ``subject_1``,
    ``subject_2``, ... , which names the NWB container, the per-identity output file and
    the bounding box series. ``subject_id`` cannot change any of those - it only labels
    the Subject - so a ``name`` in the subject entry sets the identity name instead. This
    is chiefly how a pose file that has no external identities gets them; a file that
    already carries them normally keeps what it has.

    The name is applied by filling ``external_ids``, which is both what the writer reads
    the identity name from and where the name is recorded for a reader to restore, so
    nothing downstream needs to know this happened. An identity left unnamed keeps the id
    it already had, which for such a file is its ``subject_N`` placeholder. ``subjects``
    is re-keyed to match: the writer looks metadata up by identity name, so leaving the
    old key in place would orphan the metadata the name was attached to.

    Args:
        data: Pose data whose ``subjects`` may carry ``name`` overrides.

    Returns:
        ``data`` unchanged when no entry carries a name, otherwise a copy with
        ``subjects`` stripped of the key and, if anything was actually renamed,
        ``external_ids`` filled in.

    Raises:
        ValueError: If a ``name`` is not a string, if the resulting identity names are
            not unique, or if a new name collides with a ``subjects`` key belonging to
            something else.
    """
    resolved = resolve_identity_subjects(data)
    # Keyed on the key being present, not on it naming anything: a blank name renames
    # nothing but still has to be stripped before it reaches the output metadata.
    if not any(_carries_name(entry) for entry in resolved):
        return data

    # Resolve every name before touching subjects, so two identities renamed to the same
    # thing are reported as the duplicate they are rather than as a key collision.
    names: list[str] = []
    renames: list[tuple] = []
    for entry in resolved:
        override = _name_override(entry)
        if subject_value_is_absent(override):
            # Not every identity has to be renamed; the rest keep the id they had. That
            # is lookup_keys[0], the *raw* external ID, not the sanitized container name:
            # the writer looks subjects up by the raw ID first, so writing the sanitized
            # form back would orphan metadata keyed by an ID that needed sanitizing.
            names.append(entry.lookup_keys[0])
            continue
        if not isinstance(override, str):
            # str() would turn a list or a number into a plausible-looking container
            # name and write a real file under it, with only the sanitization warning
            # to hint that anything was wrong.
            raise ValueError(
                f"The 'name' for --subjects key {entry.matched_key!r} must be a string, "
                f"got {type(override).__name__}: {override!r}."
            )
        name = sanitize_identity_name(override)
        if name != override.strip():
            logger.warning(
                "Identity name %r is not usable as an NWB container name; using %r instead",
                override,
                name,
            )
        logger.info("Renaming identity %s to %s", entry.identity_name, name)
        names.append(name)
        renames.append((entry, name))

    # Compare the container names the writer will derive, not the ids themselves: two
    # ids that differ only in characters sanitization strips would collide there.
    container_names = [sanitize_identity_name(n) for n in names]
    duplicates = sorted(
        n for n, count in collections.Counter(container_names).items() if count > 1
    )
    if duplicates:
        raise ValueError(
            f"Identity names must be unique, but {', '.join(repr(d) for d in duplicates)} "
            f"is used more than once: {container_names}. "
            "Check the 'name' fields in --subjects."
        )

    # Re-key in place rather than rebuilding from the resolved metadata: a key that
    # matches no identity has to survive, or validate_subjects can no longer report it
    # and a typo'd key becomes an unexplained "species is missing".
    subjects: dict[str, dict] = dict(data.subjects or {})
    for entry in resolved:
        if entry.matched_key is not None and _carries_name(entry):
            subjects[entry.matched_key] = {
                k: v for k, v in subjects[entry.matched_key].items() if k != _IDENTITY_NAME_KEY
            }

    # Pop every renamed entry before inserting any of them, so a collision is reported
    # only against a key that survives the pops. Swapping two identities' names is a
    # legitimate edit, and checking as we go would reject it on the first of the pair.
    moved = [
        (entry, name, subjects.pop(entry.matched_key) if entry.matched_key is not None else {})
        for entry, name in renames
    ]
    for entry, name, _ in moved:
        if name in subjects:
            raise ValueError(
                f"Renaming identity {entry.identity_name!r} to {name!r} collides with the "
                f"--subjects key {name!r}, which would discard one of them. Rename the "
                "identity to something else, or drop the conflicting key."
            )
    for _, name, metadata in moved:
        subjects[name] = metadata

    # Only a real rename changes the identity names; a file whose only 'name' is blank
    # keeps whatever external_ids it already had.
    external_ids = names if renames else data.external_ids
    return dataclasses.replace(data, external_ids=external_ids, subjects=subjects or None)


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
            name (matching external_identities values, or "subject_1",
            "subject_2", ... when the pose file has none).  Passed through to
            PoseData.subjects, except for a ``name`` field, which renames the
            identity it belongs to - see :func:`_apply_identity_names`.

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

    return _apply_identity_names(
        PoseData(
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
    )


_SESSION_METADATA_FIELDS = frozenset(
    {
        "session_start_time",
        "experimenter",
        "lab",
        "institution",
        "experiment_description",
        "session_id",
        "keywords",
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
    multisubject: bool = False,
    session_description: str | None = None,
    subjects: dict[str, dict] | None = None,
    session_metadata: dict | None = None,
) -> None:
    """Convert a JABS pose HDF5 file to NWB and write to disk.

    The pose format version is inferred from the filename (e.g.
    "_pose_est_v6.h5" → v6).  Supported versions: v2-v8.

    By default one NWB file is written per identity.  In that mode
    ``output_path`` is a naming template; actual files are written alongside it
    as "{stem}_{identity_name}.nwb".  Pass ``multisubject=True`` to instead write
    a single combined file at ``output_path`` using the ndx-multisubjects
    extension.

    Args:
        input_path: Path to the input JABS pose HDF5 file.
        output_path: Destination path for the NWB file.  In the default
            per-identity mode this is used as a naming template; actual files are
            written alongside it as "{stem}_{identity_name}.nwb".  In multisubject
            mode the single combined file is written at this path.
        multisubject: If True, write a single multi-subject NWB file using the
            ndx-multisubjects extension instead of one file per identity.
        session_description: Optional NWB session description string.
        subjects: Optional per-animal biological metadata dict, keyed by
            identity name.  See PoseData.subjects for the expected
            structure.
        session_metadata: Optional dict of NWB session-level metadata.
            Supported keys: ``session_start_time`` (ISO 8601 string),
            ``experimenter`` (str or list[str]), ``lab``, ``institution``,
            ``experiment_description``, ``session_id``, ``keywords``
            (list[str]).  Unknown keys are ignored with a warning.

    Raises:
        ValueError: If the input file is not a recognized JABS pose file, if
            ``session_start_time`` cannot be parsed, or if any identity's subject
            metadata does not meet the DANDI archive's requirements (see
            :mod:`jabs.scripts.cli.dandi_subject_metadata`).  Subject metadata is
            validated before any file is written.
        FileNotFoundError: If the input file does not exist.
    """
    logger.info("Loading %s", input_path)
    pose = open_pose_file(input_path)
    identity_word = "identity" if pose.num_identities == 1 else "identities"
    logger.info(
        "%d %s, %d frames, %d fps", pose.num_identities, identity_word, pose.num_frames, pose.fps
    )

    pose_data = pose_to_pose_data(pose, subjects=subjects)

    # Validate before writing: per-identity output writes one file per identity in a
    # loop, so failing partway would leave an incomplete set on disk, and the whole
    # set would be unpublishable anyway.
    validate_subjects(pose_data)

    write_kwargs: dict = {"multisubject": multisubject}
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

    if multisubject:
        logger.info("Writing multisubject NWB to %s", output_path)
    else:
        logger.info(
            "Writing per-identity NWB files in %s (using %s as a naming template)",
            output_path.parent,
            output_path.name,
        )
    save(pose_data, output_path, **write_kwargs)
