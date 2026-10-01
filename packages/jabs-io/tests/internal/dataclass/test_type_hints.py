"""Tests for the type-annotation helpers shared by the dataclass adapters."""

from datetime import datetime

import numpy as np
import pytest

from jabs.io.internal.dataclass.type_hints import is_datetime_type, unwrap_optional

# ---------------------------------------------------------------------------
# unwrap_optional
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        (int, int),
        (int | None, int),
        (np.ndarray | None, np.ndarray),
        (list[str] | None, list[str]),
        (int | str, int | str),
        (int | str | None, int | str | None),
        (type(None), type(None)),
    ],
    ids=[
        "plain",
        "optional-int",
        "optional-ndarray",
        "optional-parameterized",
        "two-member-union",
        "three-member-union",
        "none-type",
    ],
)
def test_unwrap_optional(annotation: object, expected: object) -> None:
    """unwrap_optional strips None from an optional of exactly one type."""
    assert unwrap_optional(annotation) == expected


# ---------------------------------------------------------------------------
# is_datetime_type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        (datetime, True),
        (datetime | None, True),
        (str, False),
        (int | None, False),
        (str | datetime, True),
        (float, False),
    ],
    ids=["datetime", "optional-datetime", "str", "optional-int", "union-with-datetime", "float"],
)
def test_is_datetime_type(annotation: object, expected: bool) -> None:
    """is_datetime_type detects datetime and Optional[datetime] annotations."""
    assert is_datetime_type(annotation) is expected


def test_is_datetime_type_rejects_string_annotation() -> None:
    """A PEP 563 string annotation is not recognized, as documented."""
    assert is_datetime_type("datetime") is False
