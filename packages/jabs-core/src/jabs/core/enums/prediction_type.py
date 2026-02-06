import enum


class PredictionType(enum.IntEnum):
    """Enum describing a prediction type.

    RAW: Raw model output.
    POSTPROCESSED: Post-processed model output.
    """

    RAW = enum.auto()
    POSTPROCESSED = enum.auto()
