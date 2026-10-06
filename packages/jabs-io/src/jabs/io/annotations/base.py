"""Abstract interface for a JABS project's behavior annotation documents.

A JABS project keeps the behavior labels for each video in a single JSON
document (``jabs/annotations/<video>.json``). This module defines the storage
contract over those documents so that no other part of JABS opens one directly.
:class:`~jabs.io.annotations.local.LocalAnnotationStore` backs a project
directory on disk; a Hub-backed store can cache and sync the same documents
without any caller changing.

The unit of storage is the **serialized document** - the dict that
``VideoLabels.as_dict()`` produces and ``VideoLabels.load()`` consumes - rather
than a ``VideoLabels`` object. Keeping the interface at the dict level is what
lets it live in ``jabs.io``: building a ``VideoLabels`` requires pose data,
which belongs to layers above this one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, NamedTuple, TypeAlias

AnnotationDocumentContent: TypeAlias = dict[str, Any]

UNVERSIONED = 0
"""Version reported by stores that do not track document versions.

A local project directory has no version history: the file on disk is whatever
was written last. Local stores report this constant so callers can pass a
version around uniformly, and a versioned store can be substituted later without
the call sites changing shape.
"""


class AnnotationDocument(NamedTuple):
    """A serialized annotation document together with its store version.

    Attributes:
        content: The document as stored, in the format produced by
            ``VideoLabels.as_dict()``.
        version: The version the store holds this document at, or
            :data:`UNVERSIONED` for stores without version tracking.
    """

    content: AnnotationDocumentContent
    version: int


def annotation_filename(video_name: str) -> str:
    """Return the document filename a video's annotations are stored under.

    The name is derived from the video filename, so ``video1.avi`` and
    ``video1.mp4`` in the same project would collide. That has always been true
    of JABS projects on disk and is not introduced here.

    Args:
        video_name: Video filename. A path is accepted and reduced to its final
            component.

    Returns:
        The document filename, e.g. ``"video1.json"``.
    """
    return Path(Path(video_name).name).with_suffix(".json").name


class AnnotationStore(ABC):
    """Storage for a project's per-video behavior annotation documents.

    Implementations are keyed by video filename. A store does not assume a
    document exists for every video in the project: a video that has never been
    labeled simply has no document.
    """

    @abstractmethod
    def load_document(self, video_name: str) -> AnnotationDocument | None:
        """Load the annotation document for a video.

        Args:
            video_name: Video filename the document belongs to.

        Returns:
            The document and its version, or ``None`` if the store holds no
            document for this video.
        """

    @abstractmethod
    def save_document(
        self,
        video_name: str,
        document: AnnotationDocumentContent,
        base_version: int | None = None,
    ) -> int:
        """Write the annotation document for a video.

        Args:
            video_name: Video filename the document belongs to.
            document: Serialized document, as produced by
                ``VideoLabels.as_dict()``.
            base_version: The version the caller last read, for stores that
                detect concurrent writes. Stores without version tracking
                ignore it.

        Returns:
            The version the document is now stored at, or :data:`UNVERSIONED`
            for stores without version tracking.
        """

    @abstractmethod
    def delete_document(self, video_name: str) -> bool:
        """Remove a video's annotation document from the store.

        Deleting through the store rather than unlinking
        :meth:`document_path` is what makes the removal reach the authority.
        For a remote store, unlinking the cached file would leave the
        authoritative document in place, and the next sync would bring the
        labels back.

        Args:
            video_name: Video filename the document belongs to.

        Returns:
            ``True`` if a document was removed, ``False`` if the store held
            none for this video.
        """

    @abstractmethod
    def has_document(self, video_name: str) -> bool:
        """Return whether the store holds an annotation document for a video.

        Args:
            video_name: Video filename to check.
        """

    @abstractmethod
    def document_path(self, video_name: str) -> Path:
        """Return the local path a video's annotation document occupies.

        The path is where the document lives (for a local project) or would be
        cached (for a remote one). It is returned without any I/O, so it may not
        exist. Callers that need the bytes to be present must use
        :meth:`ensure_local` instead.

        Args:
            video_name: Video filename the document belongs to.
        """

    @abstractmethod
    def ensure_local(self, video_name: str) -> Path:
        """Make a video's annotation document readable from the filesystem.

        This is the seam for code that cannot call back into the store - most
        importantly the feature-extraction worker processes, which are handed
        plain paths and must stay free of any network access.

        A remote store fetches the document into its cache; a local store has
        nothing to do. Either way the return value is the same path
        :meth:`document_path` reports, and it still will not exist if the store
        holds no document for this video.

        Args:
            video_name: Video filename the document belongs to.

        Returns:
            Path to the document on the local filesystem.
        """
