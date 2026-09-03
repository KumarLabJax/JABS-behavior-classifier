"""Loading and validation of the pose file JSON Schemas.

The manifest and provenance documents are JSON, and these schemas are the
definition of their validity (ADR 0002). They ship as package data so that a
reader never depends on the repository layout, and they are extracted verbatim
from the ADR so the specification and the implementation cannot drift.

Nothing here inspects ``schema_revision``: the format is additive-only and a
reader asks what a file contains, never how old it is.
"""

import json
from importlib import resources

import jsonschema

FORMAT_ID = "jabs.pose-file"
SCHEMA_REVISION = 1


def _load(name: str) -> dict:
    """Load one schema from package data.

    Args:
        name: File name within the package's ``schemas`` directory.

    Returns:
        The parsed schema.
    """
    text = resources.files("jabs.io.internal.pose_file").joinpath("schemas", name).read_text()
    return json.loads(text)


MANIFEST_SCHEMA = _load("manifest-1.json")
PROVENANCE_SCHEMA = _load("provenance-1.json")


def _errors(schema: dict, instance: dict) -> list[str]:
    """Validate an instance and render its errors as readable strings.

    Args:
        schema: The schema to validate against.
        instance: The document to validate.

    Returns:
        One string per error, each prefixed with the failing document path.
    """
    validator = jsonschema.Draft202012Validator(schema)
    return [
        f"{'/'.join(str(part) for part in error.path) or '<root>'}: {error.message}"
        for error in sorted(validator.iter_errors(instance), key=lambda e: list(e.path))
    ]


def validate_manifest(manifest: dict) -> list[str]:
    """Validate a manifest document.

    Args:
        manifest: The parsed contents of the file's ``/manifest`` dataset.

    Returns:
        Human-readable error strings; empty when the manifest is valid.
    """
    return _errors(MANIFEST_SCHEMA, manifest)


def validate_provenance(provenance: dict) -> list[str]:
    """Validate a provenance document.

    Args:
        provenance: The parsed contents of the file's ``/provenance`` dataset.

    Returns:
        Human-readable error strings; empty when the document is valid.
    """
    return _errors(PROVENANCE_SCHEMA, provenance)
