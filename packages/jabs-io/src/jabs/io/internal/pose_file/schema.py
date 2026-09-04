"""Loading and validation of the pose file JSON Schemas.

The manifest and provenance documents are JSON, and these schemas are the
definition of their validity (ADR 0002). They ship as package data so that a
reader never depends on the repository layout, and they are extracted verbatim
from the ADR so the specification and the implementation cannot drift.

Nothing here inspects ``schema_revision``: the format is additive-only and a
reader asks what a file contains, never how old it is.
"""

import json
from datetime import datetime
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


# The shipped schemas declare `format: "date-time"`, and jsonschema treats
# format as an annotation unless a checker is supplied -- so without this the
# schemas assert a constraint the helpers ignore. Registered locally rather
# than pulling in jsonschema's optional format extras.
FORMAT_CHECKER = jsonschema.FormatChecker()


@FORMAT_CHECKER.checks("date-time", raises=ValueError)
def _is_date_time(value: object) -> bool:
    """Whether a value is an ISO 8601 / RFC 3339 timestamp.

    Args:
        value: The candidate. Non-strings are not this format's business.

    Returns:
        True when the value parses as a timestamp.
    """
    if not isinstance(value, str):
        return True
    # fromisoformat only learned to accept a trailing Z in 3.11.
    datetime.fromisoformat(value.replace("Z", "+00:00"))
    return True


def iter_errors(schema: dict, instance: object) -> list[jsonschema.ValidationError]:
    """Validate an instance, returning the raw errors.

    Callers that need to treat some failures differently from others -- a
    malformed timestamp is worth reporting without abandoning every structural
    check below it -- need the validator keyword, not just a message.

    Args:
        schema: The schema to validate against.
        instance: The document to validate.

    Returns:
        The errors, ordered by document path.
    """
    validator = jsonschema.Draft202012Validator(schema, format_checker=FORMAT_CHECKER)
    return sorted(validator.iter_errors(instance), key=lambda e: list(e.path))


def _errors(schema: dict, instance: dict) -> list[str]:
    """Validate an instance and render its errors as readable strings.

    Args:
        schema: The schema to validate against.
        instance: The document to validate.

    Returns:
        One string per error, each prefixed with the failing document path.
    """
    return [
        f"{'/'.join(str(part) for part in error.path) or '<root>'}: {error.message}"
        for error in iter_errors(schema, instance)
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
