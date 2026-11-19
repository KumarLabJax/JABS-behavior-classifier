"""The one error every legacy conversion refusal raises."""


class LegacyConversionError(Exception):
    """Raised when a legacy pose file cannot be converted, naming why."""
