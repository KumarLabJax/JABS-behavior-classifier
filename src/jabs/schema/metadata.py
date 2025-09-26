from jsonschema import validate

schema = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "urn:jax.org:schemas:jabs:project-metadata:v1",
    "title": "JABS Project Metadata Schema",
    "type": "object",
    "properties": {
        "project": {"$ref": "#/$defs/project"},
        "videos": {
            "type": "object",
            "additionalProperties": {"$ref": "#/$defs/video"},
            "propertyNames": {"pattern": "^[^/]{1,251}\\.(avi|mp4)$"},
        },
    },
    "required": [],
    "additionalProperties": False,
    "$defs": {
        "primitive": {"anyOf": [{"type": "string"}, {"type": "number"}, {"type": "boolean"}]},
        "project": {
            "type": "object",
            "properties": {"nwb": {"$ref": "#/$defs/nwb"}},
            "additionalProperties": {"$ref": "#/$defs/primitive"},
        },
        "video": {
            "type": "object",
            "properties": {"nwb": {"$ref": "#/$defs/nwb"}},
            "required": [],
            "additionalProperties": {"$ref": "#/$defs/primitive"},
        },
        "nwb": {
            "type": "object",
            "properties": {
                "nwb_version": {"type": "string"},
                "session_description": {"type": "string", "minLength": 1},
                "identifier": {"type": "string", "minLength": 1},
                "session_start_time": {"type": "string", "format": "date-time"},
                "file_create_date": {
                    "type": "array",
                    "items": {"type": "string", "format": "date-time"},
                    "minItems": 1,
                },
                "general": {"$ref": "#/$defs/general"},
                "analysis": {
                    "description": "Lab-specific and custom analysis results. Free-form by design.",
                    "type": "object",
                    "additionalProperties": True,
                },
            },
            "required": [],
            "additionalProperties": False,
        },
        "general": {
            "type": "object",
            "properties": {
                "institution": {"type": "string"},
                "lab": {"type": "string"},
                "experimenter": {"type": "array", "items": {"type": "string"}},
                "keywords": {"type": "array", "items": {"type": "string"}},
                "experiment_description": {"type": "string"},
                "data_collection": {"type": "string"},
                "notes": {"type": "string"},
                "pharmacology": {"type": "string"},
                "protocol": {"type": "string"},
                "slices": {"type": "string"},
                "related_publications": {"type": "array", "items": {"type": "string"}},
                "session_id": {"type": "string"},
                "subject": {"$ref": "#/$defs/Subject"},
                "devices": {"type": "array", "items": {"$ref": "#/$defs/Device"}},
            },
            "additionalProperties": False,
        },
        "Subject": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "subject_id": {"type": "string"},
                "description": {"type": "string"},
                "species": {"type": "string"},
                "strain": {"type": "string"},
                "genotype": {"type": "string"},
                "sex": {"type": "string"},
                "age": {"type": "string"},
                "age_reference": {"type": "string", "enum": ["birth", "gestational"]},
                "date_of_birth": {"type": "string", "format": "date-time"},
                "weight": {"type": "string"},
            },
        },
        "Device": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "manufacturer": {"type": "string"},
                "model_number": {"type": "string"},
                "model_name": {"type": "string"},
                "serial_number": {"type": "string"},
            },
            "required": ["name"],
        },
    },
}


def validate_metadata(data: dict) -> None:
    """Validate project metadata dictionary against the schema.

    Args:
        data: The metadata project dictionary to validate.

    Raises:
        ValidationError: If the data does not conform to the schema.
    """
    validate(instance=data, schema=schema)
