"""Tests for the project-directory annotation store."""

from __future__ import annotations

import json
from pathlib import Path

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
