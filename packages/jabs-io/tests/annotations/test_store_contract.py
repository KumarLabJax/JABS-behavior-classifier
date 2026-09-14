"""Behavior every :class:`AnnotationStore` implementation must have.

These tests are written against the interface, not any one backend, so a new
implementation (a Hub-backed store, say) can be covered by adding it to the
``store`` fixture's parameters rather than by writing a second suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jabs.io.annotations import AnnotationStore, LocalAnnotationStore

_VIDEO = "video1.avi"


def _document(num_frames: int = 1000, behavior: str = "grooming") -> dict:
    """Return a serialized annotation document in the shape VideoLabels produces."""
    return {
        "version": 1,
        "file": _VIDEO,
        "num_frames": num_frames,
        "labels": {"0": {behavior: [{"start": 25, "end": 50, "present": True}]}},
        "unfragmented_labels": {"0": {behavior: [{"start": 25, "end": 60, "present": True}]}},
        "metadata": {"project": {}, "video": {}},
        "labeler": "tester",
    }


@pytest.fixture(params=["local"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> AnnotationStore:
    """An empty store of each implementation under test."""
    if request.param == "local":
        return LocalAnnotationStore(tmp_path / "jabs" / "annotations")
    raise AssertionError(f"unhandled store implementation: {request.param}")


def test_load_document_returns_none_when_absent(store: AnnotationStore) -> None:
    """A video that was never labeled has no document."""
    assert store.load_document(_VIDEO) is None


def test_has_document_false_when_absent(store: AnnotationStore) -> None:
    """has_document reports False before anything is saved."""
    assert store.has_document(_VIDEO) is False


def test_save_then_load_round_trips(store: AnnotationStore) -> None:
    """A saved document reads back with its content intact."""
    document = _document()

    store.save_document(_VIDEO, document)
    loaded = store.load_document(_VIDEO)

    assert loaded is not None
    assert loaded.content == document


def test_has_document_true_after_save(store: AnnotationStore) -> None:
    """has_document reports True once a document is saved."""
    store.save_document(_VIDEO, _document())

    assert store.has_document(_VIDEO) is True


def test_load_reports_the_version_save_returned(store: AnnotationStore) -> None:
    """The version a save reports is the version a subsequent load reports."""
    version = store.save_document(_VIDEO, _document())
    loaded = store.load_document(_VIDEO)

    assert loaded is not None
    assert loaded.version == version


def test_save_replaces_the_whole_document(store: AnnotationStore) -> None:
    """Saving is a whole-document write, not a merge into what is there."""
    store.save_document(_VIDEO, _document(behavior="grooming"))
    replacement = _document(num_frames=500, behavior="walking")

    store.save_document(_VIDEO, replacement)
    loaded = store.load_document(_VIDEO)

    assert loaded is not None
    assert loaded.content == replacement


def test_save_accepts_the_version_it_last_returned(store: AnnotationStore) -> None:
    """Writing back at the version just read is the non-conflicting case."""
    version = store.save_document(_VIDEO, _document())

    new_version = store.save_document(_VIDEO, _document(num_frames=500), base_version=version)
    loaded = store.load_document(_VIDEO)

    assert loaded is not None
    assert loaded.version == new_version


def test_videos_have_independent_documents(store: AnnotationStore) -> None:
    """Saving one video's labels leaves another video's alone."""
    first = _document(behavior="grooming")
    second = _document(behavior="walking")

    store.save_document("video1.avi", first)
    store.save_document("video2.mp4", second)

    loaded_first = store.load_document("video1.avi")
    loaded_second = store.load_document("video2.mp4")
    assert loaded_first is not None
    assert loaded_second is not None
    assert loaded_first.content == first
    assert loaded_second.content == second


def test_video_name_is_addressed_by_filename_not_path(store: AnnotationStore) -> None:
    """A video path and its bare filename address the same document."""
    store.save_document(_VIDEO, _document())

    assert store.load_document(f"/some/other/place/{_VIDEO}") is not None
    assert store.has_document(f"/some/other/place/{_VIDEO}") is True


def test_ensure_local_matches_document_path(store: AnnotationStore) -> None:
    """Hydrating a document does not move it away from its reported path."""
    store.save_document(_VIDEO, _document())

    assert store.ensure_local(_VIDEO) == store.document_path(_VIDEO)


def test_ensure_local_yields_a_readable_file(store: AnnotationStore) -> None:
    """After a save, the document is readable from the filesystem by path.

    This is what the feature-extraction workers rely on: they are handed a path
    and never touch the store.
    """
    store.save_document(_VIDEO, _document())

    assert store.ensure_local(_VIDEO).exists()


def test_document_path_does_not_create_anything(store: AnnotationStore) -> None:
    """Asking where a document would live is not a write."""
    path = store.document_path(_VIDEO)

    assert not path.exists()
    assert store.has_document(_VIDEO) is False
