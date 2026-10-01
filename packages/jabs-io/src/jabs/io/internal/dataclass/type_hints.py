"""Type-annotation introspection shared by the dataclass storage adapters.

The JSON, Parquet, and HDF5 dataclass adapters all have to look at a field's
type annotation to decide how to encode or decode its value, and they all have
to cope with the same two spellings of an optional field (``Optional[X]`` and
``X | None``). That introspection lives here so each format adapter is left
holding only its own encoding rules.
"""

import types
from datetime import datetime
from typing import Any, Union, get_args, get_origin


def _is_union(tp: Any) -> bool:
    """Return True if ``tp`` is a union, written either as ``Union[X, Y]`` or ``X | Y``."""
    origin = get_origin(tp)
    return origin is Union or origin is types.UnionType


def unwrap_optional(tp: Any) -> Any:
    """Extract ``X`` from ``X | None``.

    Args:
        tp: Type annotation to unwrap.

    Returns:
        The single non-``None`` member of an optional annotation, or ``tp``
        unchanged when it is not an optional of exactly one type.
    """
    if _is_union(tp):
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return tp


def is_datetime_type(field_type: Any) -> bool:
    """Check if a type annotation represents datetime or Optional[datetime].

    Args:
        field_type: Type annotation to test, as found on ``dataclasses.Field.type``.

    Returns:
        True for ``datetime`` itself and for any union that includes ``datetime``.

    Note:
        Only resolved annotation objects are recognized. A dataclass defined in a
        module that uses ``from __future__ import annotations`` carries *string*
        annotations in ``dataclasses.fields()``, and this returns False for those,
        so its datetime fields are stored and returned unconverted. No dataclass
        JABS persists is declared that way today.
    """
    if field_type is datetime:
        return True
    if _is_union(field_type):
        return datetime in get_args(field_type)
    return False
