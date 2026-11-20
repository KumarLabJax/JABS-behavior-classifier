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
import logging
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt

from jabs.io.internal.pose_file.alternates import Alternate, RaggedIndex, alternate_key
from jabs.io.internal.pose_file.embeddings import EmbeddingDescriptor
from jabs.io.internal.pose_file.manifest import parse_manifest, parse_provenance
from jabs.io.internal.pose_file.schema import (
    FORMAT_ID,
    validate_manifest,
    validate_provenance,
)
from jabs.io.internal.pose_file.types import (
    Attachment,
    Component,
    PoseFile,
    accepts_requested_layout,
    requested_layout_problems,
)

logger = logging.getLogger(__name__)

# Only the baseline encoding can be decoded. Handing back RLE run values or a
# ragged point buffer as though it were a dense array is worse than failing.
_SUPPORTED_ENCODINGS = frozenset({"dense"})

# The only alternate encoding this build can model. An alternate it cannot
# model is refused rather than skipped: dropping it would let a read-then-write
# lose it without a word.
_SUPPORTED_ALTERNATE_ENCODINGS = frozenset({"ragged"})


class PoseFileError(Exception):
    """Raised when a pose file cannot be read as it declares itself."""


class NotAPoseFileError(PoseFileError):
    """Raised when a file is not a ``jabs.pose-file`` at all."""


def attr_text(value: object) -> str | None:
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
    scalar = flat[0].item() if hasattr(flat[0], "item") else flat[0]
    # Only one unwrapping step: recursing on a non-string scalar such as
    # jabs_format = 1 would hand the same object back forever.
    if isinstance(scalar, bytes | np.bytes_):
        return scalar.decode("utf-8", errors="replace")
    if isinstance(scalar, str):
        return scalar
    return None


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
        if attr_text(h5.attrs.get("jabs_format")) != FORMAT_ID:
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


def _payload(node: h5py.Dataset, spec: dict) -> np.ndarray:
    """Read a component's payload, decoding text as text.

    Args:
        node: The payload dataset.
        spec: The component's manifest entry.

    Returns:
        The payload. A ``string`` component comes back as unicode rather than
        the bytes h5py hands back by default.
    """
    if spec["dtype"] == "string":
        return np.asarray(node.asstr()[()])
    return node[()]


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


def _check_alternate_decodable(label: str, alternate_spec: dict) -> None:
    """Refuse an alternate whose encoding this build cannot model.

    Args:
        label: The alternate's ``<component id>:alternate:<name>`` key.
        alternate_spec: The alternate's manifest entry.

    Raises:
        NotImplementedError: If the encoding is not ``ragged``.
    """
    kind = alternate_spec["encoding"].get("kind")
    if kind not in _SUPPORTED_ALTERNATE_ENCODINGS:
        raise NotImplementedError(
            f"{label}: cannot read alternate encoding {kind!r}; only "
            f"{sorted(_SUPPORTED_ALTERNATE_ENCODINGS)} is implemented, and skipping it would "
            "drop it from any file written back"
        )


def _offsets_for(h5: h5py.File, path: str, label: str) -> npt.NDArray[np.generic]:
    """Read one ragged offsets dataset, checking it is really a dataset.

    Args:
        h5: The open file.
        path: The offsets' declared path.
        label: The alternate that declared it, for the error message.

    Returns:
        The offsets, in their stored dtype.

    Raises:
        PoseFileError: If the path is absent, is not a dataset, or is not
            one-dimensional.
    """
    node = h5.get(path)
    if not isinstance(node, h5py.Dataset):
        raise PoseFileError(f"{label}: declared offsets {path} are missing or are not a dataset")
    # Checked before reading: a scalar dataset reads back as a bare value
    # (bytes, for a string) that RaggedIndex cannot even ask the shape of.
    if node.ndim != 1:
        raise PoseFileError(
            f"{label}: declared offsets {path} must be a one-dimensional dataset, got shape "
            f"{node.shape} ({node.dtype})"
        )
    return node[()]


def _alternates_for(
    h5: h5py.File, spec: dict, indexes: dict[tuple[str, str], RaggedIndex]
) -> tuple[Alternate, ...]:
    """Read a component's alternates.

    Args:
        h5: The open file.
        spec: The component's manifest entry.
        indexes: The offsets already read, keyed by ``(group_offsets,
            instance_offsets)`` path pair. Updated in place, so alternates
            naming the same two paths share one :class:`RaggedIndex`: the
            writer writes a shared index once, and two equal copies would be
            two writes to one path.

    Returns:
        The alternates, in manifest order.

    Raises:
        NotImplementedError: If an alternate's encoding cannot be modeled.
        PoseFileError: If an alternate disagrees with the file or with itself.
    """
    alternates = []
    for alternate_spec in spec.get("alternates", []):
        label = alternate_key(spec["id"], alternate_spec["name"])
        _check_alternate_decodable(label, alternate_spec)
        encoding = alternate_spec["encoding"]
        pair = (encoding["group_offsets"], encoding["instance_offsets"])
        # _dataset_for and _payload name what they read by "id"; the alternate
        # shares its component's id, so its own key keeps the messages apart.
        labeled = {**alternate_spec, "id": label}
        payload = _payload(_dataset_for(h5, labeled), labeled)
        try:
            index = indexes.get(pair)
            if index is None:
                index = RaggedIndex(
                    group_offsets_path=pair[0],
                    instance_offsets_path=pair[1],
                    group_offsets=_offsets_for(h5, pair[0], label),
                    instance_offsets=_offsets_for(h5, pair[1], label),
                )
                indexes[pair] = index
            alternates.append(
                Alternate(
                    name=alternate_spec["name"],
                    path=alternate_spec["path"],
                    axes=tuple(alternate_spec["axes"]),
                    data=payload,
                    missing=alternate_spec["missing"],
                    index=index,
                    # Absent means leaf, as the format defines it.
                    level=encoding.get("level", "leaf"),
                    units=alternate_spec.get("units"),
                    coord_order=alternate_spec.get("coord_order"),
                    description=alternate_spec.get("description"),
                    layout=alternate_spec.get("layout"),
                    extra=alternate_spec.get("extra"),
                )
            )
        except ValueError as error:
            raise PoseFileError(f"{label}: {error}") from error
    return tuple(alternates)


def _component_from(
    spec: dict, data: np.ndarray, *, alternates: tuple[Alternate, ...] = ()
) -> Component:
    """Build a Component from its manifest entry and payload.

    Args:
        spec: The component's manifest entry.
        data: The payload.
        alternates: The component's alternates, already read.

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
        alternates=alternates,
        # The manifest was schema-validated by read_manifest, so the
        # descriptor carries every required key.
        embedding=EmbeddingDescriptor.from_json(spec["embedding"])
        if "embedding" in spec
        else None,
        requested_layout=_requested_layout(spec, data.ndim),
    )


def _requested_layout(spec: dict, ndim: int) -> dict | None:
    """The manifest layout to carry forward as a request, when it can be one.

    A layout recorded for an embedding or a foreign component is kept so that
    a read-write cycle preserves it. One the writer could not honor -- szip,
    say, which a third party may legitimately have used -- is dropped rather
    than failing the read, and the writer's policy applies instead.

    Args:
        spec: The component's manifest entry.
        ndim: The payload's rank.

    Returns:
        The layout, or None when there is none to request.
    """
    layout = spec.get("layout")
    if layout is None or not accepts_requested_layout(spec["id"]):
        return None
    problems = requested_layout_problems(layout, ndim)
    if problems:
        logger.debug(
            "%s: dropping recorded layout the writer cannot honor (%s)",
            spec["id"],
            "; ".join(problems),
        )
        return None
    return layout


def read_component(
    path: str | Path,
    component_id: str,
    frames: slice | None = None,
    *,
    manifest: dict | None = None,
) -> np.ndarray:
    """Read one component's payload.

    Reading the manifest means opening the file and running a full JSON Schema
    validation. A caller reading many windows of one file -- a viewer
    following a playhead, especially over a network -- should call
    ``read_manifest`` once and pass the result as ``manifest``: a pose file is
    read-only once written, so the manifest cannot change between calls.

    Args:
        path: The pose file.
        component_id: The component to read.
        frames: An optional window over the component's ``frame`` axis. Only
            that window is read from disk, and it is applied to whichever axis
            the manifest names ``frame`` — not necessarily axis 0.
        manifest: The file's manifest, as returned by ``read_manifest``. When
            given, the file's own manifest is neither re-read nor
            re-validated. Passing any other document is the caller's error.

    Returns:
        The payload, or the requested window of it.

    Raises:
        KeyError: If the manifest declares no such component.
        ValueError: If ``frames`` is given for a component with no frame axis.
        NotImplementedError: If the encoding cannot be decoded.
        PoseFileError: If the file disagrees with its own manifest.
        NotAPoseFileError: If the file is not a ``jabs.pose-file``.
    """
    if manifest is None:
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
            return _payload(dataset, spec)
        axis = spec["axes"].index("frame")
        selector = (slice(None),) * axis + (frames,)
        # asstr() on the window too, so a string component does not hand back
        # bytes for a window and str for a whole read.
        if spec["dtype"] == "string":
            return np.asarray(dataset.asstr()[selector])
        return dataset[selector]


def read_pose_file(path: str | Path) -> PoseFile:
    """Read a whole pose file.

    Alternates come back on their components. They are derived from the
    baseline data, so per ADR 0002 a caller that changes a component's data
    before writing it back must drop or regenerate its alternates, e.g.
    ``dataclasses.replace(c, data=..., alternates=())``; the default
    ``deep=False`` write does not catch a stale one.

    Args:
        path: The pose file.

    Returns:
        The file's contents, including alternates, attachments and namespaced
        extras, so that writing the result back loses nothing.

    Raises:
        NotImplementedError: If a component's encoding cannot be decoded, or
            an alternate's cannot be modeled.
        PoseFileError: If the file disagrees with its own manifest, or its
            declarations disagree with one another.
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
        if not isinstance(provenance_raw, dict):
            raise PoseFileError(f"{path}: /provenance is not a JSON object")
        # Validated before parsing: parse_provenance indexes required fields,
        # so a schema-invalid record would leak a KeyError past the documented
        # PoseFileError contract.
        provenance_errors = validate_provenance(provenance_raw)
        if provenance_errors:
            raise PoseFileError(
                f"{path}: provenance does not satisfy the schema: " + "; ".join(provenance_errors)
            )
        provenance = parse_provenance(provenance_raw)

        components = []
        indexes: dict[tuple[str, str], RaggedIndex] = {}
        for spec in parsed.component_specs:
            _check_decodable(spec)
            _check_skeleton(manifest, spec)
            payload = _payload(_dataset_for(h5, spec), spec)
            alternates = _alternates_for(h5, spec, indexes)
            try:
                components.append(_component_from(spec, payload, alternates=alternates))
            except ValueError as error:
                raise PoseFileError(f"{path}: {spec['id']}: {error}") from error

        attachments = []
        for spec in parsed.attachment_specs:
            node = h5.get(spec["path"])
            if not isinstance(node, h5py.Dataset):
                raise PoseFileError(
                    f"declared attachment {spec['path']} is missing or is not a dataset"
                )
            try:
                attachments.append(
                    Attachment(
                        path=spec["path"],
                        data=node[()],
                        description=spec.get("description"),
                        content_type=spec.get("content_type"),
                    )
                )
            except ValueError as error:
                raise PoseFileError(f"{path}: attachment {spec['path']}: {error}") from error

    # The typed objects enforce cross-references the schema cannot (a mask
    # naming an undeclared component, for one) by raising ValueError. On the
    # read path that is a file disagreeing with itself, which callers catch as
    # PoseFileError, so it must not escape as a bare ValueError.
    try:
        return PoseFile(
            dimensions=parsed.dimensions,
            video=parsed.video,
            skeletons=parsed.skeletons,
            components=tuple(components),
            provenance=provenance,
            attachments=tuple(attachments),
            extra=parsed.extra,
        )
    except ValueError as error:
        raise PoseFileError(f"{path}: the file's contents are inconsistent: {error}") from error
