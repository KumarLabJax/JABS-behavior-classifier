"""Tests for the project-directory annotation store."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from jabs.io.annotations import (
    UNVERSIONED,
    LocalAnnotationStore,
    read_document,
    write_document,
)

_VIDEO = "video1.avi"
_DOCUMENT = {"version": 1, "file": _VIDEO, "num_frames": 100, "labels": {}}


@pytest.fixture
def annotations_dir(tmp_path: Path) -> Path:
    """An existing annotations directory, as a real project would have."""
    path = tmp_path / "jabs" / "annotations"
    path.mkdir(parents=True)
    return path


# LocalAnnotationStore


def test_document_path_is_the_video_stem_in_the_annotations_dir(annotations_dir: Path) -> None:
    """Documents keep the on-disk naming JABS projects have always used."""
    store = LocalAnnotationStore(annotations_dir)

    assert store.document_path(_VIDEO) == annotations_dir / "video1.json"


@pytest.mark.parametrize(
    ("video_name", "expected"),
    [
        ("video1.avi", "video1.json"),
        ("video1.mp4", "video1.json"),
        ("video1", "video1.json"),
        ("my.video.avi", "my.video.json"),
        ("/abs/path/video1.avi", "video1.json"),
    ],
    ids=["avi", "mp4", "no-suffix", "dotted-name", "full-path"],
)
def test_document_path_name_derivation(
    annotations_dir: Path, video_name: str, expected: str
) -> None:
    """The document filename is the video filename with a .json suffix."""
    store = LocalAnnotationStore(annotations_dir)

    assert store.document_path(video_name).name == expected


def test_documents_are_reported_unversioned(annotations_dir: Path) -> None:
    """A project directory has no version history to report."""
    store = LocalAnnotationStore(annotations_dir)

    assert store.save_document(_VIDEO, _DOCUMENT) == UNVERSIONED

    loaded = store.load_document(_VIDEO)
    assert loaded is not None
    assert loaded.version == UNVERSIONED


def test_save_ignores_a_stale_base_version(annotations_dir: Path) -> None:
    """Without version tracking there is nothing to reject a stale write."""
    store = LocalAnnotationStore(annotations_dir)
    store.save_document(_VIDEO, _DOCUMENT)

    replacement = {**_DOCUMENT, "num_frames": 500}
    store.save_document(_VIDEO, replacement, base_version=99)

    loaded = store.load_document(_VIDEO)
    assert loaded is not None
    assert loaded.content == replacement


def test_save_creates_a_missing_annotations_directory(tmp_path: Path) -> None:
    """A store can be pointed at a directory that does not exist yet."""
    store = LocalAnnotationStore(tmp_path / "jabs" / "annotations")

    store.save_document(_VIDEO, _DOCUMENT)

    assert store.document_path(_VIDEO).exists()


def test_save_leaves_no_temporary_file_behind(annotations_dir: Path) -> None:
    """The atomic write cleans up after itself."""
    store = LocalAnnotationStore(annotations_dir)

    store.save_document(_VIDEO, _DOCUMENT)

    assert sorted(p.name for p in annotations_dir.iterdir()) == ["video1.json"]


def test_delete_document_unlinks_the_file(annotations_dir: Path) -> None:
    """Deleting through the store removes the annotation file from disk."""
    store = LocalAnnotationStore(annotations_dir)
    store.save_document(_VIDEO, _DOCUMENT)

    assert store.delete_document(_VIDEO) is True
    assert not (annotations_dir / "video1.json").exists()


def test_concurrent_writers_do_not_share_a_temporary_file(annotations_dir: Path) -> None:
    """Two interleaved writes cannot mix into one published document.

    A fixed ``<video>.json.tmp`` would let a second writer scribble into the
    first writer's staging file, publishing a mixture of the two. Each write
    gets its own temporary, so the rename decides and one write lands whole.
    """
    store = LocalAnnotationStore(annotations_dir)
    first = {**_DOCUMENT, "num_frames": 111}
    second = {**_DOCUMENT, "num_frames": 222}

    real_open = Path.open
    seen: list[Path] = []

    def recording_open(self: Path, *args: object, **kwargs: object):
        """Record every temporary a write stages through."""
        if self.name.endswith(".tmp"):
            seen.append(self)
            # stage the competing write while this one holds its temporary open
            if len(seen) == 1:
                store.save_document(_VIDEO, second)
        return real_open(self, *args, **kwargs)

    with mock.patch.object(Path, "open", recording_open):
        store.save_document(_VIDEO, first)

    assert len(seen) == 2
    assert seen[0] != seen[1]

    loaded = store.load_document(_VIDEO)
    assert loaded is not None
    # the outer write renamed last, so it is the one that survives - whole
    assert loaded.content == first


def test_a_failed_write_leaves_no_temporary_behind(annotations_dir: Path) -> None:
    """Unique temporaries would otherwise pile up on every failed save."""
    store = LocalAnnotationStore(annotations_dir)

    with (
        mock.patch("json.dump", side_effect=OSError("disk full")),
        pytest.raises(OSError, match="disk full"),
    ):
        store.save_document(_VIDEO, _DOCUMENT)

    assert list(annotations_dir.iterdir()) == []


def test_a_failed_write_leaves_the_previous_document_intact(annotations_dir: Path) -> None:
    """A save that fails partway does not destroy what was already there."""
    store = LocalAnnotationStore(annotations_dir)
    store.save_document(_VIDEO, _DOCUMENT)

    with (
        mock.patch("json.dump", side_effect=OSError("disk full")),
        pytest.raises(OSError, match="disk full"),
    ):
        store.save_document(_VIDEO, {**_DOCUMENT, "num_frames": 999})

    loaded = store.load_document(_VIDEO)
    assert loaded is not None
    assert loaded.content == _DOCUMENT


def test_load_raises_on_a_corrupt_document(annotations_dir: Path) -> None:
    """A malformed annotation file is surfaced, not silently treated as absent."""
    store = LocalAnnotationStore(annotations_dir)
    store.document_path(_VIDEO).write_text("{not json")

    with pytest.raises(json.JSONDecodeError):
        store.load_document(_VIDEO)


def test_annotations_dir_is_exposed(annotations_dir: Path) -> None:
    """The store reports the directory it was built over."""
    assert LocalAnnotationStore(annotations_dir).annotations_dir == annotations_dir


# read_document / write_document


def test_read_document_returns_none_for_a_missing_file(tmp_path: Path) -> None:
    """Workers treat an absent annotation file as "this video is unlabeled"."""
    assert read_document(tmp_path / "absent.json") is None


def test_write_then_read_document_round_trips(tmp_path: Path) -> None:
    """The path-level helpers are each other's inverse."""
    path = tmp_path / "video1.json"

    write_document(path, _DOCUMENT)

    assert read_document(path) == _DOCUMENT


def test_write_document_is_readable_as_plain_json(tmp_path: Path) -> None:
    """The on-disk format stays ordinary JSON that other tools can read."""
    path = tmp_path / "video1.json"

    write_document(path, _DOCUMENT)

    assert json.loads(path.read_text()) == _DOCUMENT


def test_write_document_replaces_existing_content(tmp_path: Path) -> None:
    """A rewrite does not leave any of the previous document behind."""
    path = tmp_path / "video1.json"
    write_document(path, {**_DOCUMENT, "labels": {"0": {"grooming": []}}})

    write_document(path, _DOCUMENT)

    assert read_document(path) == _DOCUMENT
