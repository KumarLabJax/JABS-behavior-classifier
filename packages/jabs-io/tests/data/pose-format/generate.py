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
from pathlib import Path

import h5py
import numpy as np

from jabs.io.internal.pose_file.types import (
    Component,
    HistoryEntry,
    PoseFile,
    Provenance,
    ProvenanceRecord,
    Skeleton,
    VideoInfo,
)
from jabs.io.internal.pose_file.writer import write_pose_file

HERE = Path(__file__).parent

FRAMES = 8
SLOTS = 3
IDENTITIES = 2

MOUSE12 = Skeleton(
    body_parts=(
        "NOSE",
        "LEFT_EAR",
        "RIGHT_EAR",
        "BASE_NECK",
        "LEFT_FRONT_PAW",
        "RIGHT_FRONT_PAW",
        "CENTER_SPINE",
        "LEFT_REAR_PAW",
        "RIGHT_REAR_PAW",
        "BASE_TAIL",
        "MID_TAIL",
        "TIP_TAIL",
    ),
    edges=(
        (4, 6),
        (6, 5),
        (7, 9),
        (9, 8),
        (0, 3),
        (3, 6),
        (6, 9),
        (9, 10),
        (10, 11),
        (1, 0),
        (0, 2),
    ),
    description="JABS 12-keypoint mouse skeleton",
)

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
    point_valid = confidence > 0.3
    slot_occupied = point_valid.any(axis=2)
    points[~point_valid] = np.nan
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
            provenance="jabs.pose",
        ),
        Component(
            id="jabs.pose.point_valid",
            axes=("frame", "slot", "keypoint"),
            data=point_valid,
            missing={"policy": "none"},
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


def main() -> None:
    """Write every fixture."""
    write_pose_file(valid_minimal(), HERE / "valid-minimal.h5")
    write_pose_file(valid_full(), HERE / "valid-full.h5")
    write_pose_file(valid_sparse(), HERE / "valid-sparse.h5")

    # An attachment nobody declared. The writer cannot produce one yet, so it
    # is added here; the corpus keeps it because a warning is the specified
    # response and that path deserves coverage.
    with h5py.File(HERE / "valid-full.h5", "r+") as h5:
        h5.create_dataset(
            "/attachments/notes", data=np.frombuffer(b"an opaque payload", dtype=np.uint8)
        )

    base = HERE / "valid-full.h5"

    shutil.copy(base, HERE / "invalid-shape-mismatch.h5")
    _rewrite_manifest(
        HERE / "invalid-shape-mismatch.h5",
        lambda m: _entry(m, "jabs.pose.points").update(shape=[99, SLOTS, 12, 2]),
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

    with h5py.File(HERE / "invalid-not-a-pose-file.h5", "w") as h5:
        group = h5.create_group("poseest")
        group.attrs["version"] = np.array([6, 0], dtype=np.uint16)
        group.create_dataset("points", data=np.zeros((FRAMES, 1, 12, 2), dtype=np.uint16))

    for fixture in sorted(HERE.glob("*.h5")):
        print(f"{fixture.name:34s} {fixture.stat().st_size / 1024:7.1f} KiB")


if __name__ == "__main__":
    main()
