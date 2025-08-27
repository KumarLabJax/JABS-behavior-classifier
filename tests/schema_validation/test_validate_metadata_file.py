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
        "project": {
            "description": "Example project",
            "ground_truth": True,
            "created_by": "someone",
        },
        "videos": {
            "video1.mp4": {
                "org_id": 123,
                "cage_id": 1234,
                "timestamp": "2023-01-01T12:00:00Z",
                "duration": 600,
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


def test_invalid_nwb_key():
    """Test that NWB with an invalid key raises a ValidationError."""
    invalid_metadata = {"project": {"nwb": {"invalid_key": "value"}}}
    with pytest.raises(ValidationError):
        validate_metadata(invalid_metadata)


def test_nwb_general_invalid_type():
    """Test that invalid general field type raises a ValidationError."""
    invalid_metadata = {
        "project": {
            "nwb": {
                "session_description": "A session",
                "identifier": "ID123",
                "session_start_time": "2023-01-01T12:00:00Z",
                "general": {
                    "institution": 123,  # Should be a string
                },
            }
        }
    }
    with pytest.raises(ValidationError):
        validate_metadata(invalid_metadata)


def test_nwb_general_valid():
    """Test that a valid NWB general field passes validation."""
    valid_metadata = {
        "project": {
            "nwb": {
                "session_description": "A session",
                "identifier": "ID123",
                "session_start_time": "2023-01-01T12:00:00Z",
                "general": {
                    "institution": "My Institution",
                    "lab": "My Lab",
                    "experimenter": ["Dr. A", "Dr. B"],
                    "keywords": ["test", "nwb"],
                    "experiment_description": "Testing NWB metadata",
                },
            }
        },
        "videos": {
            "video1.mp4": {
                "org_id": 123,
                "cage_id": 1234,
                "nwb": {
                    "session_description": "Video session",
                    "identifier": "VID123",
                    "session_start_time": "2023-01-01T12:00:00Z",
                    "general": {
                        "institution": "My Institution",
                    },
                },
            },
        },
    }
    # Should not raise
    validate_metadata(valid_metadata)
