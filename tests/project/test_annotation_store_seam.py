"""Every annotation read and write in a project goes through its store.

The point of the store seam is that the backing of a project's label documents
can be swapped without touching the call sites. These tests substitute an
in-memory store for the default local one and assert the project actually
consults it - if any caller goes back to opening ``jabs/annotations/*.json``
directly, the substituted store stops being asked and the relevant test fails.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from jabs.io.annotations import (
    UNVERSIONED,
    AnnotationDocument,
    AnnotationDocumentContent,
    AnnotationStore,
    LocalAnnotationStore,
    annotation_filename,
)
from jabs.project import Project
from jabs.project.project_paths import ProjectPaths
from jabs.project.settings_manager import SettingsManager
from jabs.project.video_manager import VideoManager

_VIDEO = "video_a.avi"


class RecordingStore(AnnotationStore):
    """In-memory store that records the calls made against it.

    Args:
        cache_dir: Directory the store reports documents as living in.
            Nothing is written there; it only gives ``document_path`` somewhere
            plausible to point.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir
        self.documents: dict[str, AnnotationDocumentContent] = {}
        self.loaded: list[str] = []
        self.saved: list[str] = []
        self.hydrated: list[str] = []
        self.deleted: list[str] = []

    def document_path(self, video_name: str) -> Path:
        """Report where this video's document would be cached."""
        return self._cache_dir / annotation_filename(video_name)

    def ensure_local(self, video_name: str) -> Path:
        """Record the hydration request and report the document path."""
        self.hydrated.append(video_name)
        return self.document_path(video_name)

    def delete_document(self, video_name: str) -> bool:
        """Record the deletion and drop the in-memory document."""
        self.deleted.append(video_name)
        return self.documents.pop(annotation_filename(video_name), None) is not None

    def has_document(self, video_name: str) -> bool:
        """Report whether this store holds a document for the video."""
        return annotation_filename(video_name) in self.documents

    def load_document(self, video_name: str) -> AnnotationDocument | None:
        """Record the read and return the in-memory document, if any."""
        self.loaded.append(video_name)
        content = self.documents.get(annotation_filename(video_name))
        return None if content is None else AnnotationDocument(content, UNVERSIONED)

    def save_document(
        self,
        video_name: str,
        document: AnnotationDocumentContent,
        base_version: int | None = None,
    ) -> int:
        """Record the write and keep the document in memory."""
        self.saved.append(video_name)
        self.documents[annotation_filename(video_name)] = document
        return UNVERSIONED


def _document(behavior: str = "Walk") -> AnnotationDocumentContent:
    """Return a serialized annotation document with one labeled block."""
    return {
        "version": 1,
        "file": _VIDEO,
        "num_frames": 100,
        "labels": {"0": {behavior: [{"start": 0, "end": 4, "present": True}]}},
        "unfragmented_labels": {"0": {behavior: [{"start": 0, "end": 9, "present": True}]}},
    }


@pytest.fixture
def project(tmp_path: Path) -> Project:
    """A project with no videos, enough to exercise the annotation seam."""
    return Project(
        tmp_path,
        enable_video_check=False,
        enable_session_tracker=False,
        validate_project_dir=False,
    )


@pytest.fixture
def store(project: Project, tmp_path: Path) -> RecordingStore:
    """A recording store substituted for the project's local one."""
    recording = RecordingStore(tmp_path / "cache")
    project._annotation_store = recording
    project._video_manager._annotation_store = recording
    return recording


def test_project_builds_a_local_store_by_default(project: Project, tmp_path: Path) -> None:
    """A project directory is backed by the local store, as it always was."""
    assert isinstance(project.annotation_store, LocalAnnotationStore)
    assert project.annotation_store.document_path(_VIDEO) == (
        project.project_paths.annotations_dir / "video_a.json"
    )


def test_video_manager_shares_the_project_store(project: Project) -> None:
    """One store serves the whole project, not one per component."""
    assert project.video_manager.annotation_store is project.annotation_store


def test_video_manager_requires_an_explicit_store(tmp_path: Path) -> None:
    """A VideoManager cannot be built without saying which store it reads from.

    The store is deliberately not defaulted: silently constructing a second one
    would let a caller that forgot to pass the project's store read local JSON
    files while appearing to work.
    """
    paths = ProjectPaths(base_path=tmp_path)
    paths.create_directories(validate=False)

    with pytest.raises(TypeError, match="annotation_store"):
        VideoManager(paths, SettingsManager(paths), enable_video_check=False, scan_results={})


def test_video_manager_uses_the_store_it_is_given(tmp_path: Path) -> None:
    """The store passed in is the one the manager reports and reads from."""
    paths = ProjectPaths(base_path=tmp_path)
    paths.create_directories(validate=False)
    given = LocalAnnotationStore(tmp_path / "elsewhere")
    manager = VideoManager(
        paths,
        SettingsManager(paths),
        enable_video_check=False,
        scan_results={},
        annotation_store=given,
    )

    assert manager.annotation_store is given


def test_save_annotations_writes_through_the_store(
    project: Project, store: RecordingStore
) -> None:
    """The one write path hands its document to the store."""
    labels = MagicMock()
    labels.filename = _VIDEO
    labels.as_dict.return_value = _document()

    project.save_annotations(labels, MagicMock())

    assert store.saved == [_VIDEO]
    assert store.documents[annotation_filename(_VIDEO)]["labels"] == _document()["labels"]


def test_save_annotations_stamps_the_labeler(project: Project, store: RecordingStore) -> None:
    """The labeler is still stamped onto the document that reaches the store."""
    labels = MagicMock()
    labels.filename = _VIDEO
    labels.as_dict.return_value = _document()

    project.save_annotations(labels, MagicMock())

    assert store.documents[annotation_filename(_VIDEO)]["labeler"] == project.labeler


def test_load_counts_reads_through_the_store(project: Project, store: RecordingStore) -> None:
    """Label counts come from the store rather than from a direct file read."""
    store.documents[annotation_filename(_VIDEO)] = _document()

    counts = project.load_counts(_VIDEO, "Walk")

    assert store.loaded == [_VIDEO]
    assert counts[0]["fragmented_frame_counts"] == (5, 0)
    assert counts[0]["unfragmented_frame_counts"] == (10, 0)


def test_load_counts_is_empty_when_the_store_has_no_document(
    project: Project, store: RecordingStore
) -> None:
    """An unlabeled video reports no counts, without touching the filesystem."""
    assert project.load_counts(_VIDEO, "Walk") == {}
    assert store.loaded == [_VIDEO]


def test_derived_file_paths_take_the_annotation_path_from_the_store(
    project: Project, store: RecordingStore
) -> None:
    """The annotation entry in the derived-file list is the store's path."""
    store.documents[annotation_filename(_VIDEO)] = _document()

    paths = project.get_derived_file_paths(_VIDEO)

    assert store.document_path(_VIDEO) in paths


def test_derived_file_paths_omit_an_absent_annotation(
    project: Project, store: RecordingStore
) -> None:
    """A video the store has no document for contributes no annotation path."""
    assert store.document_path(_VIDEO) not in project.get_derived_file_paths(_VIDEO)


def test_feature_load_jobs_hydrate_the_document_first(
    project: Project, store: RecordingStore
) -> None:
    """Worker jobs get a hydrated path, so the child process never needs the store."""
    project._video_manager = MagicMock()
    project._video_manager.video_path.return_value = Path("video_a.avi")
    project._video_manager.get_cached_pose_path.return_value = Path("video_a_pose_est_v6.h5")

    job = project._build_feature_load_job_base(_VIDEO, {})

    assert store.hydrated == [_VIDEO]
    assert job["annotations_path"] == store.document_path(_VIDEO)
