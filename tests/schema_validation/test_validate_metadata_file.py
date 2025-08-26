"""test validating a project metadata file against the schema

The metadata file is a JSON file that can be passed to the jabs-init command
to inject metadata into a new project.
"""

import pytest
from jsonschema.exceptions import ValidationError

from jabs.schema.metadata import validate_metadata


def test_validate_metadata_valid():
    """Test that a valid metadata dictionary passes validation."""
    valid_metadata = {
        "metadata": {
            "description": "Example project",
            "type": "ground truth",
            "created_by": "someone",
        },
        "videos": {
            "video1.mp4": {
                "metadata": {
                    "org_id": 123,
                    "cage_id": 1234,
                    "timestamp": "2023-01-01T12:00:00Z",
                    "duration": 600,
                }
            }
        },
    }
    # Should not raise
    validate_metadata(valid_metadata)


def test_validate_metadata_invalid():
    """Test that an invalid metadata dictionary raises a ValidationError."""
    invalid_metadata = {"bad_top_level_field": "foobar"}
    with pytest.raises(ValidationError):
        validate_metadata(invalid_metadata)


def test_reserved_key():
    """Test that using a reserved key raises a ValidationError."""
    invalid_metadata = {"metadata": {"nwb": "should not be used"}}
    with pytest.raises(ValidationError):
        validate_metadata(invalid_metadata)
