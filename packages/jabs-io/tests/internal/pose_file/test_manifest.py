"""Tests for manifest construction and parsing."""

from jabs.io.internal.pose_file.manifest import build_manifest, build_provenance, parse_manifest
from jabs.io.internal.pose_file.schema import validate_manifest, validate_provenance


def test_built_manifest_validates(sample_pose_file):
    """What we build must satisfy the schema we ship."""
    assert validate_manifest(build_manifest(sample_pose_file)) == []


def test_built_provenance_validates(sample_pose_file):
    """The provenance document likewise."""
    assert validate_provenance(build_provenance(sample_pose_file.provenance)) == []


def test_built_manifest_records_shape_and_dtype(sample_pose_file):
    """A component entry describes the dataset a reader will find."""
    manifest = build_manifest(sample_pose_file)
    entry = next(c for c in manifest["components"] if c["id"] == "jabs.pose.points")
    assert entry["shape"] == [4, 2, 12, 2]
    assert entry["dtype"] == "float32"
    assert entry["path"] == "/jabs/pose/points"
    assert entry["encoding"] == {"kind": "dense"}


def test_layouts_are_recorded_when_supplied(sample_pose_file):
    """The writer's storage decision is what the manifest reports."""
    layouts = {"jabs.pose.points": {"storage": "contiguous", "compression": "none"}}
    manifest = build_manifest(sample_pose_file, layouts=layouts)
    entry = next(c for c in manifest["components"] if c["id"] == "jabs.pose.points")
    assert entry["layout"] == {"storage": "contiguous", "compression": "none"}
    other = next(c for c in manifest["components"] if c["id"] == "jabs.pose.confidence")
    assert "layout" not in other


def test_unknown_video_dimensions_are_explicit_nulls(sample_pose_file_no_dimensions):
    """An explicit unknown is recoverable; a plausible default is not."""
    manifest = build_manifest(sample_pose_file_no_dimensions)
    assert manifest["video"]["width"] is None
    assert manifest["video"]["height"] is None
    assert validate_manifest(manifest) == []


def test_optional_component_fields_are_omitted_not_nulled(sample_pose_file):
    """Absent declarations are absent, since the schema forbids nulls there."""
    manifest = build_manifest(sample_pose_file)
    entry = next(c for c in manifest["components"] if c["id"] == "jabs.pose.slot_occupied")
    for absent in ("units", "coord_order", "skeleton", "sparse", "description", "layout"):
        assert absent not in entry


def test_parse_round_trips_dimensions_and_skeletons(sample_pose_file):
    """Parsing recovers the typed view the reader needs."""
    parsed = parse_manifest(build_manifest(sample_pose_file))
    assert parsed.dimensions == {"frame": 4, "slot": 2, "identity": 2}
    assert parsed.video.fps == 30.0
    assert parsed.video.frame_count == 4
    assert parsed.skeletons["jabs.mouse12"].body_parts[0] == "NOSE"
    assert len(parsed.skeletons["jabs.mouse12"].edges) == 11
    assert parsed.skeletons["jabs.mouse12"].edges[0] == (4, 6)
    assert {c["id"] for c in parsed.component_specs} == {
        "jabs.pose.points",
        "jabs.pose.confidence",
        "jabs.pose.point_valid",
        "jabs.pose.slot_occupied",
    }
