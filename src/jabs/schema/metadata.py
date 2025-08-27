from jsonschema import validate

schema = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "urn:jax.org:schemas:jabs:project-metadata:v1",
    "type": "object",
    "properties": {
        "project": {
            "type": "object",
            # reserve "nwb" for future use
            "propertyNames": {"not": {"const": "nwb"}},
            "additionalProperties": {
                "type": [
                    "string",
                    "number",
                    "boolean",
                ]
            },
        },
        "videos": {
            "type": "object",
            "additionalProperties": {"$ref": "#/$defs/video"},
            # restrict video names (keys) to filenames with .avi or .mp4 extension
            "propertyNames": {"pattern": r"^[^/]{1,251}\.(avi|mp4)$"},
        },
    },
    "additionalProperties": False,
    "$defs": {
        "video": {
            "type": "object",
            "properties": {
                "metadata": {
                    "type": "object",
                    # reserve "nwb" for future use
                    "propertyNames": {"not": {"const": "nwb"}},
                    "additionalProperties": {
                        "type": [
                            "string",
                            "number",
                            "boolean",
                        ]
                    },
                },
            },
            "additionalProperties": False,
        }
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
