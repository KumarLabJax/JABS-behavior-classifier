"""Reading revision-1 pose files (ADR 0002).

A reader asks what a file *contains*, never how old it is: identification is
the ``jabs_format`` root attribute, and nothing here inspects
``schema_revision``.

Three entry points, in increasing cost. :func:`read_manifest` learns the
contents without touching a payload. :func:`read_component` reads one
component, optionally only a window of its frame axis. :func:`read_pose_file`
loads everything.
"""

import json
from pathlib import Path

import h5py
import numpy as np

from jabs.io.internal.pose_file.manifest import parse_manifest, parse_provenance
from jabs.io.internal.pose_file.schema import FORMAT_ID
from jabs.io.internal.pose_file.types import Component, PoseFile


class NotAPoseFileError(Exception):
    """Raised when a file is not a ``jabs.pose-file``."""


def _describe_other_format(h5: h5py.File) -> str:
    """Name what a non-pose-file actually is, as far as can be told.

    Args:
        h5: An open HDF5 file lacking the ``jabs_format`` attribute.

    Returns:
        A phrase naming the format, for use in an error message.
    """
    version = h5.get("poseest", {}).attrs.get("version") if "poseest" in h5 else None
    if version is not None:
        major = int(np.asarray(version).ravel()[0])
        return f"a legacy pose_est_v{major} file"
    return "not a JABS pose file"


def read_manifest(path: str | Path) -> dict:
    """Read a pose file's contents declaration.

    Args:
        path: The pose file.

    Returns:
        The parsed manifest document.

    Raises:
        NotAPoseFileError: If the file is not a ``jabs.pose-file``.
    """
    with h5py.File(path, "r") as h5:
        if h5.attrs.get("jabs_format") != FORMAT_ID:
            raise NotAPoseFileError(f"{path} is {_describe_other_format(h5)}, not a {FORMAT_ID}")
        return json.loads(h5["manifest"][()])


def _spec_for(manifest: dict, component_id: str) -> dict:
    """Find one component's manifest entry.

    Args:
        manifest: A parsed manifest.
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


def read_component(path: str | Path, component_id: str, frames: slice | None = None) -> np.ndarray:
    """Read one component's payload.

    Args:
        path: The pose file.
        component_id: The component to read.
        frames: An optional window over the component's ``frame`` axis. Only
            that window is read from disk.

    Returns:
        The payload, or the requested window of it.

    Raises:
        KeyError: If the manifest declares no such component.
        ValueError: If ``frames`` is given for a component with no frame axis.
        NotAPoseFileError: If the file is not a ``jabs.pose-file``.
    """
    manifest = read_manifest(path)
    spec = _spec_for(manifest, component_id)
    if frames is not None and "frame" not in spec["axes"]:
        raise ValueError(
            f"{component_id} has axes {spec['axes']} and no frame axis, so a frame "
            "window is meaningless; a sparse component is windowed through its index"
        )
    with h5py.File(path, "r") as h5:
        dataset = h5[spec["path"]]
        return dataset[frames] if frames is not None else dataset[()]


def read_pose_file(path: str | Path) -> PoseFile:
    """Read a whole pose file.

    Args:
        path: The pose file.

    Returns:
        The file's contents.

    Raises:
        NotAPoseFileError: If the file is not a ``jabs.pose-file``.
    """
    manifest = read_manifest(path)
    parsed = parse_manifest(manifest)
    with h5py.File(path, "r") as h5:
        provenance = parse_provenance(json.loads(h5["provenance"][()]))
        components = tuple(
            Component(
                id=spec["id"],
                axes=tuple(spec["axes"]),
                data=h5[spec["path"]][()],
                missing=spec["missing"],
                units=spec.get("units"),
                coord_order=spec.get("coord_order"),
                skeleton=spec.get("skeleton"),
                provenance=spec.get("provenance"),
                sparse_index=(spec.get("sparse") or {}).get("index"),
                description=spec.get("description"),
                layout=spec.get("layout"),
            )
            for spec in parsed.component_specs
        )
    return PoseFile(
        dimensions=parsed.dimensions,
        video=parsed.video,
        skeletons=parsed.skeletons,
        components=components,
        provenance=provenance,
    )
