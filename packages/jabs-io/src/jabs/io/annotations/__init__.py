"""Storage for a JABS project's behavior annotation documents.

Every read and write of a project's ``jabs/annotations/<video>.json`` documents
goes through an :class:`AnnotationStore`, so the storage backing them can change
without the rest of JABS changing. :class:`LocalAnnotationStore` is the
project-directory implementation and is the only one today.
"""

from .base import (
    UNVERSIONED,
    AnnotationDocument,
    AnnotationDocumentContent,
    AnnotationStore,
    annotation_filename,
)
from .local import LocalAnnotationStore, read_document, write_document

__all__ = [
    "UNVERSIONED",
    "AnnotationDocument",
    "AnnotationDocumentContent",
    "AnnotationStore",
    "LocalAnnotationStore",
    "annotation_filename",
    "read_document",
    "write_document",
]
