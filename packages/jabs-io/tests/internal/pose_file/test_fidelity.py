"""A reader must not return wrong data, and a round trip must not lose data.

Every failure mode here was silent: plausible shapes, no exception, no finding.
"""

import json

import h5py
import numpy as np
import pytest

from jabs.io.internal.pose_file import (
    Attachment,
    Component,
    PoseFile,
    PoseFileError,
    read_component,
    read_pose_file,
    validate,
    write_pose_file,
)


def _rewrite_manifest(path, mutate):
    """Replace a written file's manifest, leaving its arrays untouched."""
    with h5py.File(path, "r+") as h5:
        manifest = json.loads(h5["manifest"][()])
        mutate(manifest)
        del h5["manifest"]
        h5.create_dataset(
            "manifest", data=json.dumps(manifest), dtype=h5py.string_dtype(encoding="utf-8")
        )


@pytest.fixture
def written(tmp_path, sample_pose_file):
    """A written copy of the sample pose file."""
    path = tmp_path / "a_pose.h5"
    write_pose_file(sample_pose_file, path)
    return path


# --- the frame axis is wherever the manifest says it is ---------------------


def test_frame_window_uses_the_declared_frame_axis(tmp_path, sample_pose_file):
    """Nothing requires the frame axis to be axis 0."""
    slot_major = Component(
        id="org.example.lab.slot_major",
        axes=("slot", "frame"),
        data=np.arange(3 * 10, dtype=np.float32).reshape(3, 10),
        missing={"policy": "nan"},
        units="unitless",
    )
    path = tmp_path / "slot_major.h5"
    write_pose_file(
        PoseFile(
            dimensions={"frame": 10, "slot": 3, "identity": 3},
            video=sample_pose_file.video.__class__(frame_count=10, width=8, height=8, fps=30.0),
            skeletons={},
            components=(slot_major,),
        ),
        path,
    )
    window = read_component(path, "org.example.lab.slot_major", frames=slice(0, 2))
    full = read_component(path, "org.example.lab.slot_major")
    np.testing.assert_array_equal(window, full[:, 0:2])
    assert window.shape == (3, 2)


# --- the reader refuses what it cannot interpret ----------------------------


def test_read_refuses_a_manifest_the_schema_rejects(written):
    """parse_manifest documents that it takes a validated document."""
    _rewrite_manifest(written, lambda m: m.pop("video"))
    with pytest.raises(PoseFileError, match="manifest"):
        read_pose_file(written)


def test_read_refuses_a_shape_disagreement(written):
    """Re-deriving shape from the array silently absorbed the manifest's lie."""
    _rewrite_manifest(written, lambda m: m["components"][0].update(shape=[99, 2, 12, 2]))
    with pytest.raises(PoseFileError, match="shape"):
        read_pose_file(written)


def test_read_refuses_a_keypoint_axis_disagreement(written):
    """Two body-part names against a 12-wide axis mislabels every keypoint."""

    def shrink(manifest):
        manifest["skeletons"]["jabs.mouse12"]["body_parts"] = ["NOSE", "TIP_TAIL"]
        manifest["skeletons"]["jabs.mouse12"]["edges"] = [[0, 1]]

    _rewrite_manifest(written, shrink)
    with pytest.raises(PoseFileError, match="keypoint"):
        read_pose_file(written)


def test_read_refuses_a_path_that_is_a_group(written):
    """A declared path must resolve to a dataset, not a group."""
    _rewrite_manifest(written, lambda m: m["components"][0].update(path="/jabs/pose"))
    with pytest.raises(PoseFileError, match="dataset"):
        read_pose_file(written)


# --- encodings are preserved or refused, never relabeled -------------------


def test_writer_refuses_an_encoding_it_cannot_write(tmp_path, sample_pose_file):
    """Only dense is implemented, so anything else must fail loudly."""
    points = sample_pose_file.component("jabs.pose.points")
    ragged = Component(
        id="jabs.segmentation.contours",
        axes=("point", "coord"),
        data=np.zeros((5, 2), dtype=np.int32),
        missing={"policy": "none"},
        units="pixel",
        coord_order="xy",
        encoding={
            "kind": "ragged",
            "group_offsets": "/jabs/segmentation/contour_offsets",
            "instance_offsets": "/jabs/segmentation/instance_offsets",
        },
    )
    with pytest.raises(NotImplementedError, match="ragged"):
        write_pose_file(
            PoseFile(
                dimensions=sample_pose_file.dimensions,
                video=sample_pose_file.video,
                skeletons=sample_pose_file.skeletons,
                components=(points, ragged),
                provenance=sample_pose_file.provenance,
            ),
            tmp_path / "ragged.h5",
        )


def test_dense_encoding_round_trips(written):
    """The declaration survives, rather than being re-asserted on write."""
    back = read_pose_file(written)
    assert back.component("jabs.pose.points").encoding == {"kind": "dense"}


def test_read_component_refuses_a_non_dense_payload(written):
    """Handing back RLE run values as if they were coordinates is worse than failing."""
    _rewrite_manifest(
        written,
        lambda m: m["components"][0].update(
            encoding={"kind": "rle", "instance_offsets": "/jabs/pose/offsets"}
        ),
    )
    with pytest.raises(NotImplementedError, match="rle"):
        read_component(written, "jabs.pose.points")


# --- a round trip loses nothing --------------------------------------------


def test_attachments_survive_a_round_trip(tmp_path, sample_pose_file):
    """The ADR calls silently dropping an attachment a specification violation."""
    payload = np.frombuffer(b"an opaque payload", dtype=np.uint8)
    original = PoseFile(
        dimensions=sample_pose_file.dimensions,
        video=sample_pose_file.video,
        skeletons=sample_pose_file.skeletons,
        components=sample_pose_file.components,
        provenance=sample_pose_file.provenance,
        attachments=(
            Attachment(
                path="/attachments/notes",
                data=payload,
                description="Notes nobody but the producer understands.",
                content_type="text/plain",
            ),
        ),
    )
    first = tmp_path / "first.h5"
    write_pose_file(original, first)
    assert [f for f in validate(first) if f.severity == "error"] == []
    # A declared attachment is not an undeclared one.
    assert not any(f.check == "attachment_undeclared" for f in validate(first))

    back = read_pose_file(first)
    assert len(back.attachments) == 1
    np.testing.assert_array_equal(back.attachments[0].data, payload)
    assert back.attachments[0].content_type == "text/plain"

    second = tmp_path / "second.h5"
    write_pose_file(back, second)
    with h5py.File(second, "r") as h5:
        assert "/attachments/notes" in h5
        manifest = json.loads(h5["manifest"][()])
    assert manifest["attachments"][0]["path"] == "/attachments/notes"


def test_extra_survives_a_round_trip(tmp_path, sample_pose_file):
    """`extra` is the manifest's own forward-compatibility mechanism."""
    points = sample_pose_file.component("jabs.pose.points")
    annotated = Component(
        id=points.id,
        axes=points.axes,
        data=points.data,
        missing=points.missing,
        units=points.units,
        coord_order=points.coord_order,
        skeleton=points.skeleton,
        provenance=points.provenance,
        extra={"org.example.lab": {"calibration": "2026-09-01"}},
    )
    original = PoseFile(
        dimensions=sample_pose_file.dimensions,
        video=sample_pose_file.video,
        skeletons=sample_pose_file.skeletons,
        components=(annotated,),
        provenance=sample_pose_file.provenance,
        extra={"org.example.lab": {"protocol": "open-field"}},
    )
    path = tmp_path / "extra.h5"
    write_pose_file(original, path)
    back = read_pose_file(path)
    assert back.extra == {"org.example.lab": {"protocol": "open-field"}}
    assert back.component("jabs.pose.points").extra == {
        "org.example.lab": {"calibration": "2026-09-01"}
    }


def test_a_declared_path_survives_a_round_trip(tmp_path, sample_pose_file):
    """A payload stored somewhere other than its id-derived path stays put."""
    path = tmp_path / "relocated.h5"
    write_pose_file(sample_pose_file, path)
    _rewrite_manifest(
        path,
        lambda m: next(c for c in m["components"] if c["id"] == "jabs.pose.slot_occupied").update(
            path="/somewhere/else/occupancy"
        ),
    )
    with h5py.File(path, "r+") as h5:
        h5.move("/jabs/pose/slot_occupied", "/somewhere/else/occupancy")
    assert [f for f in validate(path) if f.severity == "error"] == []

    back = read_pose_file(path)
    assert back.component("jabs.pose.slot_occupied").path == "/somewhere/else/occupancy"
    rewritten = tmp_path / "rewritten.h5"
    write_pose_file(back, rewritten)
    with h5py.File(rewritten, "r") as h5:
        assert "/somewhere/else/occupancy" in h5
