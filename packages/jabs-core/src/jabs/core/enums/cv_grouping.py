"""Cross-validation grouping strategy enum and filename-pattern helpers."""

import re
from enum import Enum


class CrossValidationGroupingStrategy(str, Enum):
    """Cross-validation grouping type for the project.

    Inheriting from str allows for easy serialization to/from JSON (the enum will
    automatically be serialized using the enum value).
    """

    INDIVIDUAL = "Individual Animal"
    VIDEO = "Video"
    FILENAME_PATTERN = "Filename Pattern"


DEFAULT_CV_GROUPING_STRATEGY = CrossValidationGroupingStrategy.INDIVIDUAL


def compile_grouping_regex(regex: str) -> re.Pattern[str]:
    """Compile a filename-pattern cross-validation grouping regular expression.

    Args:
        regex: Regular expression used to extract a grouping key from a video
            filename.

    Returns:
        The compiled regular expression pattern.

    Raises:
        ValueError: If ``regex`` is empty or not a valid regular expression.
    """
    if not regex:
        raise ValueError("Filename pattern grouping requires a non-empty regular expression")
    try:
        return re.compile(regex)
    except re.error as e:
        raise ValueError(f"Invalid filename grouping pattern: {e}") from e


def filename_group_key(video_name: str, pattern: re.Pattern[str]) -> str:
    """Extract a cross-validation grouping key from a video filename.

    The pattern is applied with :meth:`re.Pattern.search`, so it matches anywhere
    in ``video_name``. If the pattern defines a capturing group and it matched, the
    first captured group is used as the key (so a pattern that captures the digits
    in ``cage_1234.mp4`` yields ``"1234"``); otherwise the full matched text is used
    (a pattern matching the whole ``cage_1234`` token yields ``"cage_1234"``).
    Videos that do not match the pattern are placed in their own group, keyed by the
    filename itself.

    Args:
        video_name: Video filename to extract a grouping key from.
        pattern: Compiled regular expression (see :func:`compile_grouping_regex`).

    Returns:
        The grouping key string. All videos that yield the same key are placed in
        the same cross-validation group.
    """
    match = pattern.search(video_name)
    if match is None:
        # No match: the video becomes its own group (keyed by its unique filename).
        return video_name
    if pattern.groups >= 1 and match.group(1) is not None:
        return match.group(1)
    return match.group(0)
