"""Annotation document storage backed by a local project directory."""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path

from .base import (
    UNVERSIONED,
    AnnotationDocument,
    AnnotationDocumentContent,
    AnnotationStore,
    annotation_filename,
)

logger = logging.getLogger(__name__)


def read_document(path: Path) -> AnnotationDocumentContent | None:
    """Read a serialized annotation document from a JSON file.

    Provided for the feature-extraction worker processes, which are handed a
    plain path rather than a store (see
    :meth:`~jabs.io.annotations.base.AnnotationStore.ensure_local`). Code that
    has a store should call its ``load_document`` instead.

    Args:
        path: Path to the annotation JSON file.

    Returns:
        The parsed document, or ``None`` if the file does not exist.

    Raises:
        json.JSONDecodeError: If the file exists but is not valid JSON.
        OSError: If the file exists but cannot be read.
    """
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def write_document(path: Path, document: AnnotationDocumentContent) -> None:
    """Write a serialized annotation document to a JSON file.

    The document goes to a sibling temporary file which is then renamed over the
    destination, so a reader never sees a half-written document and a failed
    write leaves the previous one intact. Missing parent directories are
    created.

    The temporary name is **unique per write**, not a fixed ``<video>.json.tmp``.
    Two writers against the same project directory - a GUI session alongside
    ``jabs-cli merge`` or ``update_labels``, or simply two GUI instances - would
    otherwise share one temporary file and could interleave into a published
    document that is a mixture of both. With distinct temporaries the rename
    decides, so one write wins whole.

    The rename is deliberately **not** fsynced. A process crash cannot corrupt
    the document, but a power loss or kernel panic can still lose the most
    recent save, because the rename may reach disk before the data blocks do.
    The GUI writes this file on every label edit, so an fsync would put disk
    latency directly in the labeling path; durability against power loss is not
    worth that here.

    Args:
        path: Path to write the annotation JSON file to.
        document: Serialized document, as produced by ``VideoLabels.as_dict()``.

    Raises:
        OSError: If the file cannot be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # os.open/Path.open rather than tempfile.mkstemp: mkstemp forces 0600, which
    # would make annotation files unreadable to other members of a lab sharing a
    # project directory. A normal create respects the user's umask, as this has
    # always done.
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with tmp.open("w") as f:
            json.dump(document, f, indent=2)
        tmp.replace(path)
    except BaseException:
        # a unique temporary would otherwise accumulate in the project directory
        # on every failed write
        tmp.unlink(missing_ok=True)
        raise


class LocalAnnotationStore(AnnotationStore):
    """Annotation documents stored as JSON files in a project directory.

    This is the behavior JABS has always had: one
    ``<annotations_dir>/<video>.json`` file per labeled video, rewritten in full
    on every save. The directory has no version history, so every document is
    reported at :data:`~jabs.io.annotations.base.UNVERSIONED` and
    ``base_version`` is ignored.

    Args:
        annotations_dir: Directory holding the project's annotation JSON files,
            normally ``<project>/jabs/annotations``. It does not need to exist
            yet; it is created on the first save.
    """

    def __init__(self, annotations_dir: Path) -> None:
        self._annotations_dir = Path(annotations_dir)

    @property
    def annotations_dir(self) -> Path:
        """Directory holding this store's annotation JSON files."""
        return self._annotations_dir

    def document_path(self, video_name: str) -> Path:
        """Return the path a video's annotation document occupies.

        Args:
            video_name: Video filename the document belongs to.
        """
        return self._annotations_dir / annotation_filename(video_name)

    def ensure_local(self, video_name: str) -> Path:
        """Return the document path; a local store has nothing to fetch.

        Args:
            video_name: Video filename the document belongs to.
        """
        return self.document_path(video_name)

    def delete_document(self, video_name: str) -> bool:
        """Delete a video's annotation file.

        Args:
            video_name: Video filename the document belongs to.

        Returns:
            ``True`` if a file was deleted, ``False`` if none existed.

        Raises:
            OSError: If the file exists but cannot be deleted.
        """
        path = self.document_path(video_name)
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        logger.debug("Deleted annotation document %s", path)
        return True

    def has_document(self, video_name: str) -> bool:
        """Return whether an annotation file exists for a video.

        Args:
            video_name: Video filename to check.
        """
        return self.document_path(video_name).exists()

    def load_document(self, video_name: str) -> AnnotationDocument | None:
        """Read a video's annotation document from disk.

        Args:
            video_name: Video filename the document belongs to.

        Returns:
            The document at :data:`~jabs.io.annotations.base.UNVERSIONED`, or
            ``None`` if no annotation file exists for this video.

        Raises:
            json.JSONDecodeError: If the annotation file is not valid JSON.
            OSError: If the annotation file cannot be read.
        """
        content = read_document(self.document_path(video_name))
        if content is None:
            return None
        return AnnotationDocument(content, UNVERSIONED)

    def save_document(
        self,
        video_name: str,
        document: AnnotationDocumentContent,
        base_version: int | None = None,
    ) -> int:
        """Write a video's annotation document to disk atomically.

        Args:
            video_name: Video filename the document belongs to.
            document: Serialized document, as produced by
                ``VideoLabels.as_dict()``.
            base_version: Ignored. A project directory keeps no version history,
                so there is nothing to check the caller's base version against.

        Returns:
            :data:`~jabs.io.annotations.base.UNVERSIONED`.

        Raises:
            OSError: If the annotation file cannot be written.
        """
        path = self.document_path(video_name)
        write_document(path, document)
        logger.debug("Wrote annotation document %s", path)
        return UNVERSIONED
