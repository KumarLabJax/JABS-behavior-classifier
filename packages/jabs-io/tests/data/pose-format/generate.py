"""Generate the pose file conformance fixtures.

The ``.h5`` files beside this script are committed. Regenerate them by running
it from the repository root::

    uv run python packages/jabs-io/tests/data/pose-format/generate.py

The corpus is deliberately small and hand-checkable. Its purpose is to give any
independent implementation of ADR 0002 something to fail against — three
repositories have already reimplemented the identity scatter, and fixtures are
the cheapest defense against a fourth.

Invalid fixtures are produced by writing a valid file and then corrupting it,
because that is the only way to build a file the writer would refuse to
produce.
"""

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import TypeAlias

import h5py
import numpy as np
import numpy.typing as npt

from jabs.io.internal.pose_file.skeletons import jabs_mouse12
from jabs.io.internal.pose_file.types import (
    Component,
    HistoryEntry,
    PoseFile,
    Provenance,
    ProvenanceRecord,
    VideoInfo,
)
from jabs.io.internal.pose_file.writer import write_pose_file

HERE = Path(__file__).parent

# Pinned so the corpus is byte-reproducible: a wall-clock stamp gave the three
# valid fixtures three different timestamps from a single run, which makes any
# regeneration look like a change.
CREATED = "2026-09-04T00:00:00Z"

FRAMES = 8
SLOTS = 3
IDENTITIES = 2

MOUSE12 = jabs_mouse12()


INFER = ProvenanceRecord(
    producer="jabs-io conformance fixtures",
    version="1",
    created="2026-09-03T00:00:00Z",
    parameters={"confidence_threshold": 0.3},
)
HISTORY = (
    HistoryEntry(
        operation="infer",
        tool="jabs-io conformance fixtures",
        version="1",
        time="2026-09-03T00:00:00Z",
    ),
)


def _video() -> VideoInfo:
    return VideoInfo(
        frame_count=FRAMES,
        width=800,
        height=800,
        fps=30.0,
        cm_per_pixel=0.13082914,
        cm_per_pixel_source="default_alignment",
    )


def _pose_arrays():
    rng = np.random.default_rng(11)
    points = rng.uniform(0, 800, (FRAMES, SLOTS, 12, 2)).astype(np.float32)
    confidence = rng.uniform(0, 1, (FRAMES, SLOTS, 12)).astype(np.float32)
    # ADR 0002: NaN means no coordinate, never low confidence. Points below 0.1
    # get confidence 0 and NaN; points in (0.1, 0.3] keep their coordinates and
    # are only marked invalid, so the fixture exercises both sides of the rule.
    confidence[confidence < 0.1] = 0
    point_valid = confidence > 0.3
    slot_occupied = point_valid.any(axis=2)
    points[confidence == 0] = np.nan
    return points, confidence, point_valid, slot_occupied


def _core_components():
    points, confidence, point_valid, slot_occupied = _pose_arrays()
    return (
        Component(
            id="jabs.pose.points",
            axes=("frame", "slot", "keypoint", "coord"),
            data=points,
            missing={"policy": "nan"},
            units="pixel",
            coord_order="xy",
            skeleton="jabs.mouse12",
            provenance="jabs.pose",
        ),
        Component(
            id="jabs.pose.confidence",
            axes=("frame", "slot", "keypoint"),
            data=confidence,
            missing={"policy": "none"},
            units="unitless",
            skeleton="jabs.mouse12",
            provenance="jabs.pose",
        ),
        Component(
            id="jabs.pose.point_valid",
            axes=("frame", "slot", "keypoint"),
            data=point_valid,
            missing={"policy": "none"},
            skeleton="jabs.mouse12",
            provenance="jabs.pose",
        ),
        Component(
            id="jabs.pose.slot_occupied",
            axes=("frame", "slot"),
            data=slot_occupied,
            missing={"policy": "none"},
            provenance="jabs.pose",
        ),
    )


def valid_minimal() -> PoseFile:
    """One component and nothing else: the smallest conforming file."""
    points, _, _, _ = _pose_arrays()
    return PoseFile(
        dimensions={"frame": FRAMES, "slot": SLOTS, "identity": IDENTITIES},
        video=_video(),
        skeletons={"jabs.mouse12": MOUSE12},
        components=(
            Component(
                id="jabs.pose.points",
                axes=("frame", "slot", "keypoint", "coord"),
                data=points,
                missing={"policy": "nan"},
                units="pixel",
                coord_order="xy",
                skeleton="jabs.mouse12",
                provenance="jabs.pose",
            ),
        ),
        provenance=Provenance(records={"jabs.pose": INFER}, history=HISTORY),
    )


def valid_full() -> PoseFile:
    """The core pose components, identity centers, and a foreign component.

    The foreign component is the extensibility claim made concrete: a reader
    that has never heard of ``org.example.lab.whisker_angle`` can still subset
    it correctly, because the manifest says its axis 0 is ``frame``.
    """
    return PoseFile(
        dimensions={"frame": FRAMES, "slot": SLOTS, "identity": IDENTITIES},
        video=_video(),
        skeletons={"jabs.mouse12": MOUSE12},
        components=(
            *_core_components(),
            Component(
                id="jabs.identity.centers",
                axes=("identity", "embedding"),
                data=np.arange(IDENTITIES * 4, dtype=np.float32).reshape(IDENTITIES, 4),
                missing={"policy": "none"},
                provenance="jabs.identity",
            ),
            Component(
                id="org.example.lab.whisker_angle",
                axes=("frame", "slot"),
                data=np.linspace(0, 1, FRAMES * SLOTS, dtype=np.float32).reshape(FRAMES, SLOTS),
                missing={"policy": "nan"},
                units="radian",
                description="A component JABS knows nothing about.",
            ),
        ),
        provenance=Provenance(
            records={
                "jabs.pose": INFER,
                "jabs.identity": ProvenanceRecord(
                    producer="jabs-io conformance fixtures",
                    version="1",
                    created="2026-09-03T00:00:00Z",
                    algorithm={"name": "fixture", "tracklet_stitch": "none"},
                ),
            },
            history=HISTORY,
        ),
    )


def valid_sparse() -> PoseFile:
    """A dynamic-object component predicted on a subset of frames.

    Note the self-reference on the index component. The schema requires a
    ``sample`` axis and a ``sparse`` declaration to accompany each other in
    both directions, so the index -- which itself has a ``sample`` axis --
    names itself as its own index. The ADR's dynamic-objects table does not
    show this; see the note in the pull request.
    """
    samples = np.array([0, 3, 6], dtype=np.uint32)
    counts = np.array([1, 2, 1], dtype=np.uint32)
    positions = np.zeros((len(samples), 2, 1, 2), dtype=np.float32)
    positions[:, :, :, 0] = 100.0
    positions[:, :, :, 1] = 200.0
    index_id = "jabs.dynamic_objects.fecal_boli.frame_index"
    return PoseFile(
        dimensions={"frame": FRAMES, "slot": SLOTS, "identity": IDENTITIES},
        video=_video(),
        skeletons={"jabs.mouse12": MOUSE12},
        components=(
            *_core_components(),
            Component(
                id=index_id,
                axes=("sample",),
                data=samples,
                missing={"policy": "none"},
                units="frame",
                sparse_index=index_id,
                description="Frames on which fecal boli were counted.",
            ),
            Component(
                id="jabs.dynamic_objects.fecal_boli.counts",
                axes=("sample",),
                data=counts,
                missing={"policy": "none"},
                units="unitless",
                sparse_index=index_id,
            ),
            Component(
                id="jabs.dynamic_objects.fecal_boli.points",
                axes=("sample", "object", "point", "coord"),
                data=positions,
                missing={"policy": "nan"},
                units="pixel",
                coord_order="xy",
                sparse_index=index_id,
            ),
        ),
        provenance=Provenance(records={"jabs.pose": INFER}, history=HISTORY),
    )


# --- segmentation: one set of contours, written in each encoding ------------
#
# Per (frame, slot), a list of contours; each contour a list of (x, y) points.
# Instance (1, 0) has only one-point contours: there the leaf and group counts
# of an instance are equal, which is the case where a reader that guesses the
# ragged level instead of reading it goes wrong.


def _contours() -> dict[tuple[int, int], list[list[tuple[int, int]]]]:
    shapes = {
        (0, 0): [[(10, 10), (20, 10), (20, 20)], [(12, 12), (14, 12), (14, 14), (12, 14)]],
        (0, 1): [[(100, 100), (140, 100), (140, 140), (120, 160), (100, 140)]],
        (1, 0): [[(30, 30)], [(40, 40)]],
        (1, 1): [[(200, 200), (210, 200), (205, 210)]],
        (3, 2): [[(300, 300), (330, 300), (330, 330), (300, 330)], [(310, 310), (315, 315)]],
        (7, 0): [[(700, 700), (710, 700), (705, 710)]],
    }
    return {(f, s): shapes.get((f, s), []) for f in range(FRAMES) for s in range(SLOTS)}


def _external(contour_index: int) -> bool:
    """The first contour of an instance is its outer boundary; the rest are holes."""
    return contour_index == 0


def _ragged_arrays():
    """Leaf payload, per-contour flags, and the two shared CSR offsets."""
    points, flags, contour_offsets, instance_offsets = [], [], [0], [0]
    for (_, _), contours in sorted(_contours().items()):
        for index, contour in enumerate(contours):
            points.extend(contour)
            flags.append(_external(index))
            contour_offsets.append(len(points))
        instance_offsets.append(len(contour_offsets) - 1)
    return (
        np.array(points, dtype=np.int32).reshape(-1, 2),
        np.array(flags, dtype=bool),
        np.array(contour_offsets, dtype=np.uint64),
        np.array(instance_offsets, dtype=np.uint64),
    )


def _dense_segmentation_components():
    contours = _contours()
    max_contours = max(len(c) for c in contours.values())
    max_points = max(len(p) for c in contours.values() for p in c)
    data = np.full((FRAMES, SLOTS, max_contours, max_points, 2), -1, dtype=np.int32)
    count = np.zeros((FRAMES, SLOTS), dtype=np.uint32)
    length = np.zeros((FRAMES, SLOTS, max_contours), dtype=np.uint32)
    flag = np.zeros((FRAMES, SLOTS, max_contours), dtype=bool)
    for (f, s), instance in contours.items():
        count[f, s] = len(instance)
        for c, contour in enumerate(instance):
            data[f, s, c, : len(contour)] = contour
            length[f, s, c] = len(contour)
            flag[f, s, c] = _external(c)
    return (
        Component(
            id="jabs.segmentation.contours",
            axes=("frame", "slot", "contour", "point", "coord"),
            data=data,
            missing={"policy": "length", "length": "jabs.segmentation.contour_length"},
            units="pixel",
            coord_order="xy",
            provenance="jabs.pose",
        ),
        Component(
            id="jabs.segmentation.contour_count",
            axes=("frame", "slot"),
            data=count,
            missing={"policy": "none"},
            provenance="jabs.pose",
        ),
        Component(
            id="jabs.segmentation.contour_length",
            axes=("frame", "slot", "contour"),
            data=length,
            missing={"policy": "none"},
            provenance="jabs.pose",
        ),
        Component(
            id="jabs.segmentation.external_flag",
            axes=("frame", "slot", "contour"),
            data=flag,
            missing={"policy": "none"},
            provenance="jabs.pose",
        ),
    )


def valid_dense_segmentation() -> PoseFile:
    """Pose plus dense segmentation: the base the ragged alternates are added to."""
    return PoseFile(
        dimensions={"frame": FRAMES, "slot": SLOTS, "identity": IDENTITIES},
        video=_video(),
        skeletons={"jabs.mouse12": MOUSE12},
        components=_core_components() + _dense_segmentation_components(),
        provenance=Provenance(records={"jabs.pose": INFER}, history=HISTORY),
    )


def _ragged_encoding(prefix: str, level: str) -> dict:
    return {
        "kind": "ragged",
        "level": level,
        "group_offsets": f"{prefix}/contour_offsets",
        "instance_offsets": f"{prefix}/instance_offsets",
    }


def _write_ragged_datasets(h5: h5py.File, prefix: str) -> tuple[int, int]:
    """Write the ragged payloads and offsets under a group; return (points, contours)."""
    points, flags, contour_offsets, instance_offsets = _ragged_arrays()
    h5.create_dataset(f"{prefix}/contours", data=points)
    h5.create_dataset(f"{prefix}/external_flag", data=flags)
    h5.create_dataset(f"{prefix}/contour_offsets", data=contour_offsets)
    h5.create_dataset(f"{prefix}/instance_offsets", data=instance_offsets)
    return points.shape[0], flags.shape[0]


def _ragged_entries(prefix: str, n_points: int, n_contours: int) -> tuple[dict, dict]:
    """Manifest fields for the leaf-level contours and the group-level flags."""
    contours = {
        "path": f"{prefix}/contours",
        "axes": ["point", "coord"],
        "dtype": "int32",
        "shape": [n_points, 2],
        "units": "pixel",
        "coord_order": "xy",
        "encoding": _ragged_encoding(prefix, "leaf"),
        "missing": {"policy": "none"},
    }
    flags = {
        "path": f"{prefix}/external_flag",
        "axes": ["contour"],
        "dtype": "bool",
        "shape": [n_contours],
        "encoding": _ragged_encoding(prefix, "group"),
        "missing": {"policy": "none"},
    }
    return contours, flags


def _write_ragged_baseline(path: Path) -> None:
    """Segmentation stored ragged as the component itself, not as an alternate.

    The writer implements only the dense encoding, so the file is written with
    pose alone and the ragged datasets and their manifest entries are added.
    """
    write_pose_file(valid_minimal(), path, created=CREATED)
    prefix = "/jabs/segmentation"
    with h5py.File(path, "r+") as h5:
        n_points, n_contours = _write_ragged_datasets(h5, prefix)
    contours, flags = _ragged_entries(prefix, n_points, n_contours)

    def add(manifest: dict) -> None:
        manifest["components"] += [
            {"id": "jabs.segmentation.contours", **contours, "provenance": "jabs.pose"},
            {"id": "jabs.segmentation.external_flag", **flags, "provenance": "jabs.pose"},
        ]

    _rewrite_manifest(path, add)


def _write_ragged_alternate(path: Path) -> None:
    """Dense segmentation with the ragged alternate beside it, as ADR 0002 defines it.

    Adding an alternate is an ``encode`` operation, so the history records it.
    """
    write_pose_file(valid_dense_segmentation(), path, created=CREATED)
    prefix = "/jabs/segmentation/ragged"
    with h5py.File(path, "r+") as h5:
        n_points, n_contours = _write_ragged_datasets(h5, prefix)
    contours, flags = _ragged_entries(prefix, n_points, n_contours)

    def add(manifest: dict) -> None:
        _entry(manifest, "jabs.segmentation.contours")["alternates"] = [
            {"name": "ragged", **contours}
        ]
        _entry(manifest, "jabs.segmentation.external_flag")["alternates"] = [
            {"name": "ragged", **flags}
        ]

    _rewrite_manifest(path, add)
    _rewrite_document(
        path,
        "provenance",
        lambda p: p["history"].append(
            {
                "operation": "encode",
                "tool": "jabs-io conformance fixtures",
                "version": "1",
                "time": "2026-09-03T00:00:00Z",
                "synthesized": [
                    "jabs.segmentation.contours:alternate:ragged",
                    "jabs.segmentation.external_flag:alternate:ragged",
                ],
            }
        ),
    )


def _rle_runs() -> tuple[np.ndarray, np.ndarray]:
    """Column-major runs per (frame, slot), each starting with background.

    One instance is foreground from its first pixel, so its runs start with a
    zero-length background run, as the specification requires.
    """
    pixels = 800 * 800
    runs, offsets = [], [0]
    for f in range(FRAMES):
        for s in range(SLOTS):
            if (f, s) == (0, 0):
                instance = [0, 50, pixels - 50]
            elif (f + s) % 3 == 0:
                instance = [1000 * (s + 1), 200, pixels - 1000 * (s + 1) - 200]
            else:
                instance = [pixels]
            runs.extend(instance)
            offsets.append(len(runs))
    return np.array(runs, dtype=np.uint32), np.array(offsets, dtype=np.uint64)


def _write_rle(path: Path) -> None:
    """A foreign per-instance mask in the RLE encoding."""
    write_pose_file(valid_minimal(), path, created=CREATED)
    runs, offsets = _rle_runs()
    with h5py.File(path, "r+") as h5:
        h5.create_dataset("/org.example.lab/mask", data=runs)
        h5.create_dataset("/org.example.lab/mask_offsets", data=offsets)

    def add(manifest: dict) -> None:
        manifest["components"].append(
            {
                "id": "org.example.lab.mask",
                "path": "/org.example.lab/mask",
                "axes": ["run"],
                "dtype": "uint32",
                "shape": [int(runs.shape[0])],
                "encoding": {
                    "kind": "rle",
                    "instance_offsets": "/org.example.lab/mask_offsets",
                    "order": "column-major",
                },
                "missing": {"policy": "none"},
                "provenance": "jabs.pose",
            }
        )

    _rewrite_manifest(path, add)


def _rewrite_document(path: Path, name: str, mutate) -> None:
    """Replace a written file's /manifest or /provenance document."""
    with h5py.File(path, "r+") as h5:
        document = json.loads(h5[name][()])
        mutate(document)
        del h5[name]
        h5.create_dataset(
            name,
            data=json.dumps(document),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )


def _rewrite_manifest(path: Path, mutate) -> None:
    """Replace a written file's manifest, leaving its arrays untouched."""
    with h5py.File(path, "r+") as h5:
        manifest = json.loads(h5["manifest"][()])
        mutate(manifest)
        del h5["manifest"]
        h5.create_dataset(
            "manifest",
            data=json.dumps(manifest),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )


def _entry(manifest: dict, component_id: str) -> dict:
    return next(c for c in manifest["components"] if c["id"] == component_id)


# --- One invalid fixture per error row -------------------------------------
#
# ADR 0002 promises "a set of deliberately invalid files, one per error check".
# Each entry below breaks exactly one rule of its source file, which is a valid
# fixture: a file that broke two would tell a porter less about which check
# they got wrong. The conformance suite asserts that each reports its check and
# no other error.


# A change applied to a copied fixture, and an edit applied to one array.
Mutation: TypeAlias = Callable[[Path], None]
ArrayEdit: TypeAlias = Callable[[npt.NDArray[np.generic]], npt.NDArray[np.generic]]


def _recreate(h5: h5py.File, path: str, data: npt.NDArray[np.generic]) -> None:
    """Replace a dataset's contents, shape and dtype at the same path."""
    del h5[path]
    h5.create_dataset(path, data=data)


def _declare(
    manifest: dict, component_id: str, data: npt.NDArray[np.generic], **fields: object
) -> None:
    """Make a component's declared shape and dtype match a recreated dataset."""
    _entry(manifest, component_id).update(
        shape=list(data.shape), dtype=np.dtype(data.dtype).name, **fields
    )


def _whisker(data: npt.NDArray[np.generic]) -> Mutation:
    """Replace the foreign whisker_angle payload, declaration included."""

    def mutate(path: Path) -> None:
        with h5py.File(path, "r+") as h5:
            _recreate(h5, "/org.example.lab/whisker_angle", data)
        _rewrite_manifest(path, lambda m: _declare(m, "org.example.lab.whisker_angle", data))

    return mutate


def _arrays(**datasets: ArrayEdit) -> Mutation:
    """Mutate datasets in place: each value is ``fn(array) -> array``."""

    def mutate(path: Path) -> None:
        with h5py.File(path, "r+") as h5:
            for dataset, fn in datasets.items():
                h5[dataset][...] = fn(h5[dataset][()])

    return mutate


def _manifest(fn: Callable[[dict], object]) -> Mutation:
    """Rewrite the manifest with ``fn``, leaving the arrays untouched."""
    return lambda path: _rewrite_manifest(path, fn)


def _set_root(name: str, value: object | None) -> Mutation:
    """Set a root attribute, or delete it when ``value`` is None."""

    def mutate(path: Path) -> None:
        with h5py.File(path, "r+") as h5:
            if value is None:
                del h5.attrs[name]
            else:
                h5.attrs[name] = value

    return mutate


def _in_order(*steps: Mutation) -> Mutation:
    """Apply several mutations in turn."""

    def mutate(path: Path) -> None:
        for step in steps:
            step(path)

    return mutate


def _add_dataset(path: str, data: npt.NDArray[np.generic]) -> Mutation:
    """Create a dataset the manifest may or may not declare."""

    def mutate(file: Path) -> None:
        with h5py.File(file, "r+") as h5:
            h5.create_dataset(path, data=data)

    return mutate


def _replace_dataset(path: str, data: npt.NDArray[np.generic]) -> Mutation:
    """Replace a dataset's contents, shape and dtype, leaving its declaration alone."""

    def mutate(file: Path) -> None:
        with h5py.File(file, "r+") as h5:
            _recreate(h5, path, data)

    return mutate


def _first_alternate(manifest: dict, component_id: str) -> dict:
    """A component's first declared alternate."""
    return _entry(manifest, component_id)["alternates"][0]


SEG = "jabs.segmentation"
RAGGED = "/jabs/segmentation"
RAGGED_ALT = "/jabs/segmentation/ragged"
NOTES = np.frombuffer(b"an opaque payload", dtype=np.uint8)


def _invalid_corpus(full: Path) -> list[tuple[str, Path | None, Mutation]]:
    """``(fixture, source, mutate)`` for every error row the original six did not cover."""
    minimal = HERE / "valid-minimal.h5"
    sparse = HERE / "valid-sparse.h5"
    ragged = HERE / "valid-ragged.h5"
    alternate = HERE / "valid-ragged-alternate.h5"
    rle = HERE / "valid-rle.h5"
    index = "jabs.dynamic_objects.fecal_boli.frame_index"
    counts = "jabs.dynamic_objects.fecal_boli.counts"
    return [
        # Root attributes and the two documents.
        ("invalid-not-hdf5.h5", None, lambda path: path.write_bytes(b"not an HDF5 file\n")),
        ("invalid-schema-revision-missing.h5", minimal, _set_root("schema_revision", None)),
        (
            "invalid-manifest-root-mismatch.h5",
            minimal,
            _set_root("schema_revision", np.int32(0)),
        ),
        ("invalid-manifest-schema.h5", minimal, _manifest(lambda m: m.pop("video"))),
        (
            "invalid-timestamp.h5",
            minimal,
            _manifest(lambda m: m.update(created="the fourth of September")),
        ),
        (
            "invalid-provenance-schema.h5",
            minimal,
            lambda path: _rewrite_document(path, "provenance", lambda p: p.update(history={})),
        ),
        # Component structure.
        (
            "invalid-axes-arity.h5",
            full,
            _manifest(lambda m: _entry(m, "org.example.lab.whisker_angle").update(axes=["frame"])),
        ),
        (
            "invalid-frame-axis-length.h5",
            full,
            _whisker(np.zeros((FRAMES + 1, SLOTS), dtype=np.float32)),
        ),
        (
            "invalid-axis-length.h5",
            full,
            _whisker(np.zeros((FRAMES, SLOTS + 1), dtype=np.float32)),
        ),
        (
            "invalid-provenance-reference.h5",
            minimal,
            _manifest(lambda m: _entry(m, "jabs.pose.points").update(provenance="jabs.nowhere")),
        ),
        (
            "invalid-coord-declarations.h5",
            minimal,
            _manifest(lambda m: _entry(m, "jabs.pose.points").pop("units")),
        ),
        (
            "invalid-coord-axis-length.h5",
            minimal,
            _in_order(
                _replace_dataset(
                    "/jabs/pose/points", np.zeros((FRAMES, SLOTS, 12, 3), np.float32)
                ),
                _manifest(
                    lambda m: _entry(m, "jabs.pose.points").update(shape=[FRAMES, SLOTS, 12, 3])
                ),
            ),
        ),
        (
            "invalid-nan-policy.h5",
            full,
            _manifest(
                lambda m: _entry(m, "jabs.pose.point_valid").update(missing={"policy": "nan"})
            ),
        ),
        (
            "invalid-mask-axes.h5",
            full,
            _in_order(
                _add_dataset("/org.example.lab/side", np.ones((FRAMES, SLOTS, 12), dtype=bool)),
                _manifest(
                    lambda m: (
                        m["components"].append(
                            {
                                "id": "org.example.lab.side",
                                "path": "/org.example.lab/side",
                                "axes": ["frame", "slot", "whisker"],
                                "dtype": "bool",
                                "shape": [FRAMES, SLOTS, 12],
                                "encoding": {"kind": "dense"},
                                "missing": {"policy": "none"},
                                "provenance": "jabs.pose",
                            }
                        ),
                        _entry(m, "jabs.pose.confidence").update(
                            missing={"policy": "mask", "mask": "org.example.lab.side"}
                        ),
                    )
                ),
            ),
        ),
        (
            "invalid-mask-dtype.h5",
            full,
            _in_order(
                _replace_dataset(
                    "/jabs/pose/point_valid", np.ones((FRAMES, SLOTS, 12), dtype=np.uint8)
                ),
                _manifest(
                    lambda m: (
                        _entry(m, "jabs.pose.point_valid").update(dtype="uint8"),
                        _entry(m, "jabs.pose.confidence").update(
                            missing={"policy": "mask", "mask": "jabs.pose.point_valid"}
                        ),
                    )
                ),
            ),
        ),
        (
            "invalid-length-exceeds-axis.h5",
            alternate,
            _arrays(
                **{
                    "/jabs/segmentation/contour_length": lambda a: np.where(
                        np.arange(a.size).reshape(a.shape) == 0, 99, a
                    ).astype(a.dtype)
                }
            ),
        ),
        (
            "invalid-contour-count-exceeds-axis.h5",
            alternate,
            _arrays(
                **{
                    "/jabs/segmentation/contour_count": lambda a: np.where(
                        np.arange(a.size).reshape(a.shape) == 0, 99, a
                    ).astype(a.dtype)
                }
            ),
        ),
        (
            "invalid-duplicate-component-id.h5",
            full,
            _manifest(
                lambda m: m["components"].append(dict(_entry(m, "org.example.lab.whisker_angle")))
            ),
        ),
        (
            "invalid-namespace.h5",
            full,
            _manifest(
                lambda m: _entry(m, "org.example.lab.whisker_angle").update(id="lab.whisker_angle")
            ),
        ),
        # Dimensions and video.
        (
            "invalid-identity-exceeds-slot.h5",
            minimal,
            _manifest(lambda m: m["dimensions"].update(identity=SLOTS + 1)),
        ),
        (
            "invalid-frame-count.h5",
            minimal,
            _manifest(lambda m: m["video"].update(frame_count=FRAMES + 1)),
        ),
        # Skeletons.
        (
            "invalid-skeleton-reference.h5",
            minimal,
            _manifest(lambda m: _entry(m, "jabs.pose.points").update(skeleton="jabs.nowhere")),
        ),
        (
            "invalid-keypoint-without-skeleton.h5",
            minimal,
            _manifest(lambda m: _entry(m, "jabs.pose.points").pop("skeleton")),
        ),
        (
            "invalid-skeleton-edge.h5",
            minimal,
            _manifest(lambda m: m["skeletons"]["jabs.mouse12"]["edges"].append([0, 12])),
        ),
        # Sparse components.
        (
            "invalid-sample-without-sparse.h5",
            sparse,
            _manifest(lambda m: _entry(m, counts).pop("sparse")),
        ),
        (
            "invalid-sparse-index-order.h5",
            sparse,
            _arrays(**{"/jabs/dynamic_objects/fecal_boli/frame_index": lambda a: a[::-1]}),
        ),
        (
            "invalid-sparse-index-range.h5",
            sparse,
            _arrays(
                **{
                    "/jabs/dynamic_objects/fecal_boli/frame_index": lambda a: np.append(
                        a[:-1], FRAMES
                    )
                }
            ),
        ),
        (
            "invalid-sparse-index-dtype.h5",
            sparse,
            _in_order(
                _replace_dataset(
                    "/jabs/dynamic_objects/fecal_boli/frame_index",
                    np.array([1, 3, 6], dtype=np.float32),
                ),
                _manifest(lambda m: _entry(m, index).update(dtype="float32")),
            ),
        ),
        (
            "invalid-sparse-index-length.h5",
            sparse,
            _in_order(
                _replace_dataset(
                    "/jabs/dynamic_objects/fecal_boli/counts", np.array([1, 2], dtype=np.uint32)
                ),
                _manifest(lambda m: _entry(m, counts).update(shape=[2])),
            ),
        ),
        # Ragged offsets.
        (
            "invalid-ragged-group-terminal.h5",
            ragged,
            _arrays(**{f"{RAGGED}/contour_offsets": lambda a: np.append(a[:-1], a[-1] + 1)}),
        ),
        (
            "invalid-ragged-group-length.h5",
            ragged,
            _in_order(
                _replace_dataset(f"{RAGGED}/external_flag", np.zeros(10, dtype=bool)),
                _manifest(lambda m: _entry(m, f"{SEG}.external_flag").update(shape=[10])),
            ),
        ),
        (
            "invalid-ragged-instance-count.h5",
            ragged,
            lambda path: _with_offsets(
                path, f"{RAGGED}/instance_offsets", lambda a: np.delete(a, 1)
            ),
        ),
        (
            "invalid-ragged-order.h5",
            ragged,
            lambda path: _with_offsets(path, f"{RAGGED}/instance_offsets", _swap_first_rise),
        ),
        (
            "invalid-ragged-signed-offsets.h5",
            ragged,
            lambda path: _with_offsets(
                path, f"{RAGGED}/instance_offsets", lambda a: a.astype(np.int64)
            ),
        ),
        # Unsigned but narrower than the uint64 the encoding requires: a
        # reader that builds its index as uint64 refuses it.
        (
            "invalid-ragged-uint32-offsets.h5",
            ragged,
            lambda path: _with_offsets(
                path, f"{RAGGED}/contour_offsets", lambda a: a.astype(np.uint32)
            ),
        ),
        (
            "invalid-ragged-pairing.h5",
            ragged,
            # Both offsets copied, so the flags name a pair of their own rather
            # than sharing one dataset of the contours' pair, which
            # ragged_offsets_pairing refuses.
            _in_order(
                lambda path: _copy_dataset(
                    path, f"{RAGGED}/contour_offsets", f"{RAGGED}/flag_offsets"
                ),
                lambda path: _copy_dataset(
                    path, f"{RAGGED}/instance_offsets", f"{RAGGED}/flag_instances"
                ),
                _manifest(
                    lambda m: _entry(m, f"{SEG}.external_flag")["encoding"].update(
                        group_offsets=f"{RAGGED}/flag_offsets",
                        instance_offsets=f"{RAGGED}/flag_instances",
                    )
                ),
            ),
        ),
        # A second leaf alternate on the contours names the ragged alternate's
        # group_offsets with instance_offsets of its own: one offsets dataset
        # with two partners.
        (
            "invalid-ragged-offsets-pairing.h5",
            alternate,
            _in_order(
                lambda path: _copy_dataset(
                    path, f"{RAGGED_ALT}/instance_offsets", f"{RAGGED_ALT}/instance_offsets_b"
                ),
                lambda path: _copy_dataset(path, f"{RAGGED_ALT}/contours", f"{RAGGED_ALT}/copy"),
                _manifest(
                    lambda m: _entry(m, f"{SEG}.contours")["alternates"].append(
                        {
                            **_first_alternate(m, f"{SEG}.contours"),
                            "name": "copy",
                            "path": f"{RAGGED_ALT}/copy",
                            "encoding": {
                                **_first_alternate(m, f"{SEG}.contours")["encoding"],
                                "instance_offsets": f"{RAGGED_ALT}/instance_offsets_b",
                            },
                        }
                    )
                ),
            ),
        ),
        # Alternates.
        (
            "invalid-alternate-missing.h5",
            alternate,
            lambda path: _delete(path, f"{RAGGED_ALT}/external_flag"),
        ),
        (
            "invalid-alternate-duplicate.h5",
            alternate,
            _manifest(
                lambda m: _entry(m, f"{SEG}.contours")["alternates"].append(
                    dict(_first_alternate(m, f"{SEG}.contours"))
                )
            ),
        ),
        # RLE.
        (
            "invalid-rle-dimensions.h5",
            rle,
            _manifest(lambda m: m["video"].update(width=None, height=None)),
        ),
        (
            "invalid-rle-coverage.h5",
            rle,
            _arrays(**{"/org.example.lab/mask": lambda a: np.append(a[:-1], a[-1] + 1)}),
        ),
        (
            "invalid-rle-offsets.h5",
            rle,
            lambda path: _with_offsets(
                path, "/org.example.lab/mask_offsets", lambda a: a.astype(np.int64)
            ),
        ),
        (
            "invalid-rle-shape.h5",
            rle,
            _in_order(
                lambda path: _with_offsets(
                    path, "/org.example.lab/mask", lambda a: a.reshape(-1, 2)
                ),
                _manifest(
                    lambda m: _entry(m, "org.example.lab.mask").update(
                        axes=["run", "pair"], shape=[20, 2]
                    )
                ),
            ),
        ),
        # Attachments.
        (
            "invalid-attachment-missing.h5",
            minimal,
            _manifest(lambda m: m.update(attachments=[{"path": "/attachments/notes"}])),
        ),
        (
            "invalid-attachment-outside.h5",
            minimal,
            _in_order(
                _add_dataset("/notes", NOTES),
                _manifest(lambda m: m.update(attachments=[{"path": "/notes"}])),
            ),
        ),
        (
            "invalid-attachment-twice.h5",
            minimal,
            _in_order(
                _add_dataset("/attachments/notes", NOTES),
                _manifest(
                    lambda m: m.update(
                        attachments=[
                            {"path": "/attachments/notes"},
                            {"path": "/attachments/notes"},
                        ]
                    )
                ),
            ),
        ),
    ]


def _with_offsets(path: Path, dataset: str, fn: ArrayEdit) -> None:
    """Replace a dataset with ``fn`` of its contents, which may change its shape or dtype."""
    with h5py.File(path, "r+") as h5:
        _recreate(h5, dataset, fn(h5[dataset][()]))


def _copy_dataset(path: Path, source: str, target: str) -> None:
    """Copy one dataset's contents to a new path."""
    with h5py.File(path, "r+") as h5:
        h5.create_dataset(target, data=h5[source][()])


def _delete(path: Path, dataset: str) -> None:
    """Delete one dataset."""
    with h5py.File(path, "r+") as h5:
        del h5[dataset]


def _swap_first_rise(offsets: npt.NDArray[np.generic]) -> npt.NDArray[np.generic]:
    """Swap one rising pair of neighbors, so the array decreases once.

    The second rise, not the first: swapping at the first can move the leading
    0 and break a different rule.
    """
    out = offsets.copy()
    i = int(np.flatnonzero(np.diff(out) > 0)[1])
    out[i], out[i + 1] = out[i + 1], out[i]
    return out


def _write_invalid_corpus(full: Path) -> None:
    """Write every fixture in the invalid corpus beside this script."""
    for name, source, mutate in _invalid_corpus(full):
        target = HERE / name
        if source is not None:
            shutil.copy(source, target)
        mutate(target)


def main() -> None:
    """Write every fixture."""
    write_pose_file(valid_minimal(), HERE / "valid-minimal.h5", created=CREATED)
    write_pose_file(valid_full(), HERE / "valid-full.h5", created=CREATED)
    write_pose_file(valid_sparse(), HERE / "valid-sparse.h5", created=CREATED)
    _write_ragged_baseline(HERE / "valid-ragged.h5")
    _write_ragged_alternate(HERE / "valid-ragged-alternate.h5")
    _write_rle(HERE / "valid-rle.h5")

    # Per-contour flags declared at the leaf level: their payload is one value
    # per contour, so the shared group_offsets cannot also end at its length.
    # The bug the ragged level exists to prevent (#445).
    shutil.copy(HERE / "valid-ragged.h5", HERE / "invalid-ragged-level.h5")
    _rewrite_manifest(
        HERE / "invalid-ragged-level.h5",
        lambda m: _entry(m, "jabs.segmentation.external_flag")["encoding"].update(level="leaf"),
    )

    # An alternate that passes every structural check and still disagrees with
    # its baseline: one contour point moved. Only the opt-in deep check, which
    # decodes both, can see it.
    shutil.copy(HERE / "valid-ragged-alternate.h5", HERE / "invalid-alternate-disagrees.h5")
    with h5py.File(HERE / "invalid-alternate-disagrees.h5", "r+") as h5:
        h5["/jabs/segmentation/ragged/contours"][0, 0] += 1

    # The invalid fixtures are copies of a clean base, taken BEFORE the
    # undeclared attachment goes in: injecting it first gave every invalid
    # fixture an unrelated attachment_undeclared warning.
    base = HERE / ".base.h5"
    write_pose_file(valid_full(), base, created=CREATED)

    # The embedding axis names no dimension, so a wrong length there breaks
    # this rule and no other.
    shutil.copy(base, HERE / "invalid-shape-mismatch.h5")
    _rewrite_manifest(
        HERE / "invalid-shape-mismatch.h5",
        lambda m: _entry(m, "jabs.identity.centers").update(shape=[IDENTITIES, 5]),
    )

    shutil.copy(base, HERE / "invalid-missing-payload.h5")
    with h5py.File(HERE / "invalid-missing-payload.h5", "r+") as h5:
        del h5["/jabs/pose/confidence"]

    shutil.copy(base, HERE / "invalid-dangling-mask.h5")
    _rewrite_manifest(
        HERE / "invalid-dangling-mask.h5",
        lambda m: _entry(m, "jabs.pose.points").update(
            missing={"policy": "mask", "mask": "jabs.pose.no_such_mask"}
        ),
    )

    shutil.copy(base, HERE / "invalid-keypoint-axis.h5")

    def shrink_skeleton(manifest: dict) -> None:
        manifest["skeletons"]["jabs.mouse12"]["body_parts"] = ["NOSE", "TIP_TAIL"]
        manifest["skeletons"]["jabs.mouse12"]["edges"] = [[0, 1]]

    _rewrite_manifest(HERE / "invalid-keypoint-axis.h5", shrink_skeleton)

    _write_invalid_corpus(base)

    # An attachment nobody declared, on the valid-full fixture only, so the
    # specified warning has coverage.
    with h5py.File(HERE / "valid-full.h5", "r+") as h5:
        h5.create_dataset(
            "/attachments/notes", data=np.frombuffer(b"an opaque payload", dtype=np.uint8)
        )
    base.unlink()

    with h5py.File(HERE / "invalid-not-a-pose-file.h5", "w") as h5:
        group = h5.create_group("poseest")
        group.attrs["version"] = np.array([6, 0], dtype=np.uint16)
        group.create_dataset("points", data=np.zeros((FRAMES, 1, 12, 2), dtype=np.uint16))

    for fixture in sorted(HERE.glob("*.h5")):
        print(f"{fixture.name:34s} {fixture.stat().st_size / 1024:7.1f} KiB")


if __name__ == "__main__":
    main()
