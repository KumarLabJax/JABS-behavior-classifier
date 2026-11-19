"""The one error every legacy conversion refusal raises."""

import logging

logger = logging.getLogger(__name__)


class LegacyConversionError(Exception):
    """Raised when a legacy pose file cannot be converted, naming why."""
