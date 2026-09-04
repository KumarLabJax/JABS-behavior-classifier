"""Reading revision-1 pose files (ADR 0002).

A reader asks what a file *contains*, never how old it is: identification is
the ``jabs_format`` root attribute, and nothing here inspects
``schema_revision``.

Three entry points, in increasing cost. :func:`read_manifest` learns the
contents without touching a payload. :func:`read_component` reads one
component, optionally only a window of its frame axis. :func:`read_pose_file`
loads everything.

**The reader refuses rather than guesses.** The manifest is a second source of
truth about shapes, dtypes and skeletons, so it can disagree with the arrays
beside it. Where a disagreement would make the returned data wrong — a shape
that is not the dataset's, a skeleton whose body parts do not span the keypoint
axis, an encoding this build cannot decode — the read fails with
:class:`PoseFileError` instead of handing back something plausible.
"""

import json
from pathlib import Path

import h5py
import numpy as np

from jabs.io.internal.pose_file.manifest import parse_manifest, parse_provenance
from jabs.io.internal.pose_file.schema import FORMAT_ID, validate_manifest
from jabs.io.internal.pose_file.types import Attachment, Component, PoseFile

# Only the baseline encoding can be decoded. Handing back RLE run values or a
# ragged point buffer as though it were a dense array is worse than failing.
_SUPPORTED_ENCODINGS = frozenset({"dense"})


class PoseFileError(Exception):
    """Raised when a pose file cannot be read as it declares itself."""


class NotAPoseFileError(PoseFileError):
    """Raised when a file is not a ``jabs.pose-file`` at all."""


def _attr_text(value: object) -> str | None:
    """Decode an HDF5 string attribute to ``str``.

    h5py hands back ``bytes``/``np.bytes_`` for fixed-length ASCII attributes,
    which is what the C, Fortran, MATLAB and Julia HDF5 APIs write by default.
    Comparing those against a ``str`` with ``!=`` rejects genuine files, which
    would defeat the cross-implementation interop the specification exists for.

    Args:
        value: The raw attribute value.

    Returns:
        The decoded text, or None when the value is absent or not string-like.
    """
    if value is None:
        return None
    if isinstance(value, bytes | np.bytes_):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    flat = np.atleast_1d(np.asarray(value)).ravel()
    if flat.size == 0:
        return None
    return _attr_text(flat[0].item() if hasattr(flat[0], "item") else flat[0])


def _describe_other_format(h5: h5py.File) -> str:
    """Name what a non-pose-file actually is, as far as can be told.

    Args:
        h5: An open HDF5 file lacking a recognised ``jabs_format``.

    Returns:
        A phrase naming the format, for use in an error message.
    """
    if "poseest" not in h5:
        return "not a JABS pose file"
    try:
        version = h5["poseest"].attrs.get("version")
        major = int(np.atleast_1d(np.asarray(version)).ravel()[0])
    except (AttributeError, IndexError, TypeError, ValueError, KeyError):
        return "not a JABS pose file"
    return f"a legacy pose_est_v{major} file"


def read_manifest(path: str | Path) -> dict:
    """Read and validate a pose file's contents declaration.

    Args:
        path: The pose file.

    Returns:
        The parsed manifest document.

    Raises:
        NotAPoseFileError: If the file is not a ``jabs.pose-file``.
        PoseFileError: If the manifest is unreadable or fails its schema.
    """
    with h5py.File(path, "r") as h5:
        if _attr_text(h5.attrs.get("jabs_format")) != FORMAT_ID:
            raise NotAPoseFileError(f"{path} is {_describe_other_format(h5)}, not a {FORMAT_ID}")
        try:
            raw = h5["manifest"][()]
        except (KeyError, TypeError) as error:
            raise PoseFileError(f"{path}: /manifest is missing or unreadable: {error}") from error
    try:
        manifest = json.loads(raw)
    except ValueError as error:
        raise PoseFileError(f"{path}: /manifest is not valid JSON: {error}") from error

    errors = validate_manifest(manifest)
    if errors:
        raise PoseFileError(f"{path}: manifest does not satisfy the schema: " + "; ".join(errors))
    return manifest


def _spec_for(manifest: dict, component_id: str) -> dict:
    """Find one component's manifest entry.

    Args:
        manifest: A validated manifest.
        component_id: The component's id.

    Returns:
        The manifest entry.

    Raises:
        KeyError: If the manifest declares no such component.
    """
    for spec in manifest["components"]:
        if spec["id"] == component_id:
            return spec
    raise KeyError(f"no component {component_id!r} declared in this pose file")


def _dataset_for(h5: h5py.File, spec: dict) -> h5py.Dataset:
    """Resolve a component's payload, checking it is really a dataset.

    Args:
        h5: The open file.
        spec: The component's manifest entry.

    Returns:
        The payload dataset.

    Raises:
        PoseFileError: If the path is absent, is a group, or disagrees with the
            declared dtype or shape.
    """
    component_id = spec["id"]
    node = h5.get(spec["path"])
    if node is None:
        raise PoseFileError(f"{component_id}: no dataset at declared path {spec['path']}")
    if not isinstance(node, h5py.Dataset):
        raise PoseFileError(
            f"{component_id}: declared path {spec['path']} is a "
            f"{type(node).__name__}, not a dataset"
        )
    if list(node.shape) != list(spec["shape"]):
        raise PoseFileError(
            f"{component_id}: manifest declares shape {tuple(spec['shape'])} but the dataset "
            f"is {node.shape}"
        )
    if node.dtype.name != spec["dtype"] and not (
        spec["dtype"] == "string" and node.dtype.kind in "OSU"
    ):
        raise PoseFileError(
            f"{component_id}: manifest declares dtype {spec['dtype']!r} but the dataset is "
            f"{node.dtype.name!r}"
        )
    return node


def _check_decodable(spec: dict) -> None:
    """Refuse a component whose encoding this build cannot decode.

    Args:
        spec: The component's manifest entry.

    Raises:
        NotImplementedError: If the encoding is not the baseline.
    """
    kind = spec.get("encoding", {}).get("kind")
    if kind not in _SUPPORTED_ENCODINGS:
        raise NotImplementedError(
            f"{spec['id']}: cannot decode encoding {kind!r}; only "
            f"{sorted(_SUPPORTED_ENCODINGS)} is implemented, and returning the payload as "
            "dense would hand back offsets or run values as data"
        )


def _check_skeleton(manifest: dict, spec: dict) -> None:
    """Refuse a keypoint component whose skeleton cannot label its axis.

    Args:
        manifest: The validated manifest.
        spec: The component's manifest entry.

    Raises:
        PoseFileError: If the skeleton is unknown, or its body parts do not
            span the keypoint axis.
    """
    skeleton_id = spec.get("skeleton")
    if skeleton_id is None or "keypoint" not in spec["axes"]:
        return
    skeleton = manifest.get("skeletons", {}).get(skeleton_id)
    if skeleton is None:
        raise PoseFileError(f"{spec['id']}: references unknown skeleton {skeleton_id!r}")
    declared = spec["shape"][spec["axes"].index("keypoint")]
    if declared != len(skeleton["body_parts"]):
        raise PoseFileError(
            f"{spec['id']}: keypoint axis is {declared} wide but skeleton {skeleton_id!r} "
            f"names {len(skeleton['body_parts'])} body parts, so every keypoint would be "
            "mislabeled"
        )


def _component_from(spec: dict, data: np.ndarray) -> Component:
    """Build a Component from its manifest entry and payload.

    Args:
        spec: The component's manifest entry.
        data: The payload.

    Returns:
        The component, preserving every declaration the manifest carried.
    """
    return Component(
        id=spec["id"],
        axes=tuple(spec["axes"]),
        data=data,
        missing=spec["missing"],
        units=spec.get("units"),
        coord_order=spec.get("coord_order"),
        skeleton=spec.get("skeleton"),
        provenance=spec.get("provenance"),
        sparse_index=(spec.get("sparse") or {}).get("index"),
        description=spec.get("description"),
        layout=spec.get("layout"),
        encoding=spec.get("encoding", {"kind": "dense"}),
        extra=spec.get("extra"),
        # Preserved so a read-write cycle does not relocate a payload that a
        # third-party producer deliberately stored elsewhere.
        stored_path=spec["path"],
    )


def read_component(path: str | Path, component_id: str, frames: slice | None = None) -> np.ndarray:
    """Read one component's payload.

    Args:
        path: The pose file.
        component_id: The component to read.
        frames: An optional window over the component's ``frame`` axis. Only
            that window is read from disk, and it is applied to whichever axis
            the manifest names ``frame`` — not necessarily axis 0.

    Returns:
        The payload, or the requested window of it.

    Raises:
        KeyError: If the manifest declares no such component.
        ValueError: If ``frames`` is given for a component with no frame axis.
        NotImplementedError: If the encoding cannot be decoded.
        PoseFileError: If the file disagrees with its own manifest.
        NotAPoseFileError: If the file is not a ``jabs.pose-file``.
    """
    manifest = read_manifest(path)
    spec = _spec_for(manifest, component_id)
    _check_decodable(spec)
    _check_skeleton(manifest, spec)

    if frames is not None and "frame" not in spec["axes"]:
        raise ValueError(
            f"{component_id} has axes {spec['axes']} and no frame axis, so a frame "
            "window is meaningless; a sparse component is windowed through its index"
        )
    with h5py.File(path, "r") as h5:
        dataset = _dataset_for(h5, spec)
        if frames is None:
            return dataset[()]
        axis = spec["axes"].index("frame")
        selector = (slice(None),) * axis + (frames,)
        return dataset[selector]


def read_pose_file(path: str | Path) -> PoseFile:
    """Read a whole pose file.

    Args:
        path: The pose file.

    Returns:
        The file's contents, including attachments and namespaced extras, so
        that writing the result back loses nothing.

    Raises:
        NotImplementedError: If a component's encoding cannot be decoded.
        PoseFileError: If the file disagrees with its own manifest.
        NotAPoseFileError: If the file is not a ``jabs.pose-file``.
    """
    manifest = read_manifest(path)
    parsed = parse_manifest(manifest)
    with h5py.File(path, "r") as h5:
        try:
            provenance_raw = json.loads(h5["provenance"][()])
        except (KeyError, TypeError, ValueError) as error:
            raise PoseFileError(
                f"{path}: /provenance is missing or unreadable: {error}"
            ) from error
        provenance = parse_provenance(provenance_raw)

        components = []
        for spec in parsed.component_specs:
            _check_decodable(spec)
            _check_skeleton(manifest, spec)
            components.append(_component_from(spec, _dataset_for(h5, spec)[()]))

        attachments = []
        for spec in parsed.attachment_specs:
            node = h5.get(spec["path"])
            if not isinstance(node, h5py.Dataset):
                raise PoseFileError(
                    f"declared attachment {spec['path']} is missing or is not a dataset"
                )
            attachments.append(
                Attachment(
                    path=spec["path"],
                    data=node[()],
                    description=spec.get("description"),
                    content_type=spec.get("content_type"),
                )
            )

    return PoseFile(
        dimensions=parsed.dimensions,
        video=parsed.video,
        skeletons=parsed.skeletons,
        components=tuple(components),
        provenance=provenance,
        attachments=tuple(attachments),
        extra=parsed.extra,
    )
