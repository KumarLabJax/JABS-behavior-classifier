import pytest
from intervaltree import Interval

from jabs.project.timeline_annotations import MAX_TAG_LEN, TimelineAnnotations


class DummyPose:
    """Dummy pose class with identity mapping function."""

    def identity_index_to_display(self, idx):
        """Dummy mapping function."""
        return f"ID-{idx}"


def test_add_and_len_and_getitem():
    """Basic add, len, getitem functionality."""
    ta = TimelineAnnotations()
    ta.add_annotation(
        TimelineAnnotations.Annotation(
            start=10,
            end=20,
            tag="foo",
            color="#abcdef",
            description="d1",
            identity_index=1,
            display_identity="ID-1",
        )
    )
    ta.add_annotation(TimelineAnnotations.Annotation(start=30, end=40, tag="bar", color="#123456"))

    # len should be number of intervals
    assert len(ta) == 2

    # getitem pass-through should return set of Interval
    result = ta[
        10:21
    ]  # Annotation class is inclusive of end, but IntervalTree is exclusive so add 1
    assert isinstance(result, set)
    assert any(isinstance(iv, Interval) and iv.begin == 10 and iv.end == 21 for iv in result)


def test_find_and_exists_and_remove():
    """Test find, exists, and remove by exact key."""
    ta = TimelineAnnotations()
    # Add two overlapping intervals with different tags/identity
    ta.add_annotation(
        TimelineAnnotations.Annotation(
            start=0, end=5, tag="tag1", color="#000000", identity_index=None
        )
    )
    ta.add_annotation(
        TimelineAnnotations.Annotation(
            start=0, end=5, tag="tag1", color="#111111", identity_index=2
        )
    )

    # find by exact key should return correct count
    matches_none = ta.find_matching_intervals(start=0, end=5, tag="tag1", identity_index=None)
    matches_id2 = ta.find_matching_intervals(start=0, end=5, tag="tag1", identity_index=2)
    assert len(matches_none) == 1
    assert len(matches_id2) == 1

    # exists
    assert ta.annotation_exists(start=0, end=5, tag="tag1", identity_index=None)
    assert ta.annotation_exists(start=0, end=5, tag="tag1", identity_index=2)
    assert not ta.annotation_exists(start=0, end=5, tag="tag2", identity_index=None)

    # remove by key should only remove that one
    removed = ta.remove_annotation_by_key(start=0, end=5, tag="tag1", identity_index=None)
    assert removed == 1
    assert len(ta) == 1
    assert not ta.annotation_exists(start=0, end=5, tag="tag1", identity_index=None)
    assert ta.annotation_exists(start=0, end=5, tag="tag1", identity_index=2)


def test_serialize_roundtrip_basic():
    """Test serialize and load roundtrip with basic required fields."""
    ta = TimelineAnnotations()
    ta.add_annotation(
        TimelineAnnotations.Annotation(
            start=5, end=7, tag="tag1", color="#aaaaaa", description="desc"
        )
    )

    data = ta.serialize()
    # Expected one dict with inclusive end
    assert isinstance(data, list)
    assert len(data) == 1
    entry = data[0]
    assert entry["start"] == 5
    assert entry["end"] == 7
    assert entry["tag"] == "tag1"
    assert entry["color"] == "#aaaaaa"
    assert entry["description"] == "desc"

    # Now load from data should rebuild an equivalent tree
    rebuilt = TimelineAnnotations.load(data)
    assert len(rebuilt) == 1
    # using same inclusive selection to hit the interval
    assert rebuilt.annotation_exists(start=5, end=7, tag="tag1", identity_index=None)


def test_load_with_pose_mapping_and_optional_fields():
    """Test load with optional fields and pose identity mapping."""
    data = [
        {"start": 1, "end": 3, "tag": "tag1", "color": "#ff0000", "identity": 4},
        {"start": 10, "end": 10, "tag": "tag2", "color": "#00ff00", "description": "chair"},
    ]
    rebuilt = TimelineAnnotations.load(data, pose=DummyPose())
    assert len(rebuilt) == 2

    # Check that identity/display_identity are stored
    intervals = list(rebuilt._tree)
    stored = [iv.data for iv in intervals]
    run = next(x for x in stored if x["tag"] == "tag1")
    sit = next(x for x in stored if x["tag"] == "tag2")
    assert run["identity"] == 4
    assert run["display_identity"] == "ID-4"
    assert "description" in sit and sit["description"] == "chair"


def test_load_skips_invalid_entries():
    """Test that load skips entries with missing required fields or invalid tags."""
    # Missing required fields
    data = [
        {"start": 0, "end": 1, "color": "#fff"},  # missing tag
        {"start": 0, "tag": "x", "color": "#fff"},  # missing end
        {"end": 1, "tag": "x", "color": "#fff"},  # missing start
    ]
    rebuilt = TimelineAnnotations.load(data)
    assert len(rebuilt) == 0

    # Invalid tag characters should be skipped
    data2 = [
        {"start": 0, "end": 1, "tag": "bad tag!", "color": "#fff"},
        {"start": 0, "end": 1, "tag": "good_tag-1", "color": "#fff"},
    ]
    rebuilt2 = TimelineAnnotations.load(data2)
    assert len(rebuilt2) == 1
    assert rebuilt2.annotation_exists(start=0, end=1, tag="good_tag-1", identity_index=None)


@pytest.mark.parametrize(
    "tag, ok",
    [
        ("a", True),
        ("a" * MAX_TAG_LEN, True),
        ("a" * (MAX_TAG_LEN + 1), False),
    ],
)
def test_tag_length(tag, ok):
    """Test that tags exceeding max length are skipped."""
    data = [{"start": 0, "end": 0, "tag": tag, "color": "#000"}]
    rebuilt = TimelineAnnotations.load(data)
    assert len(rebuilt) == 1 if ok else len(rebuilt) == 0
