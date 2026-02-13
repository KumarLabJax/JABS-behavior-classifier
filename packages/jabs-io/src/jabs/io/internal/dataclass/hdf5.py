"""Generic HDF5 adapter for arbitrary dataclasses."""

import json
import types
from dataclasses import fields, is_dataclass
from typing import Any, Union, get_args, get_origin

import h5py
import numpy as np

from jabs.core.enums import StorageFormat
from jabs.io.base import HDF5Adapter
from jabs.io.registry import register_adapter

# ---------------------------------------------------------------------------
# Type-annotation helpers
# ---------------------------------------------------------------------------


def _unwrap_optional(tp):
    """Extract ``X`` from ``X | None``; returns *tp* unchanged otherwise."""
    origin = get_origin(tp)
    if origin is Union or origin is types.UnionType:
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return tp


def _is_ndarray_type(tp) -> bool:
    return _unwrap_optional(tp) is np.ndarray


def _is_dict_type(tp) -> bool:
    origin = get_origin(_unwrap_optional(tp))
    return origin is dict or _unwrap_optional(tp) is dict


def _is_list_type(tp) -> bool:
    origin = get_origin(_unwrap_optional(tp))
    return origin is list or _unwrap_optional(tp) is list


def _get_list_element_type(tp):
    """Return the element type of ``list[X]``, or ``None``."""
    inner = _unwrap_optional(tp)
    args = get_args(inner)
    if args:
        return args[0]
    return None


def _is_dataclass_type(tp) -> bool:
    return is_dataclass(_unwrap_optional(tp))


def _coerce_scalar(val, tp):
    """Convert h5py numpy scalars to native Python types."""
    unwrapped = _unwrap_optional(tp)
    if unwrapped is str:
        if isinstance(val, bytes):
            return val.decode("utf-8")
        return str(val)
    if unwrapped is int:
        return int(val)
    if unwrapped is float:
        return float(val)
    if unwrapped is bool:
        return bool(val)
    # Fallback: try to convert numpy generics
    if isinstance(val, np.generic):
        return val.item()
    return val


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


@register_adapter(StorageFormat.HDF5, priority=5)
class DataclassHDF5Adapter(HDF5Adapter):
    """HDF5 adapter for arbitrary dataclasses.

    Stores numpy arrays as datasets, nested dataclasses as subgroups,
    scalars as attributes, and dicts/empty-lists as JSON-encoded attributes.
    """

    @classmethod
    def can_handle(cls, data_type):  # noqa: D102
        return is_dataclass(data_type)

    # -- write --------------------------------------------------------------

    def _write_one(self, data, group) -> None:
        for f in fields(data):
            val = getattr(data, f.name)
            if val is None:
                continue

            if isinstance(val, np.ndarray):
                group.create_dataset(f.name, data=val)
            elif is_dataclass(val):
                sub = group.create_group(f.name)
                self._write_one(val, sub)
            elif isinstance(val, dict):
                group.attrs[f.name] = json.dumps(val)
            elif isinstance(val, list):
                if len(val) == 0:
                    group.attrs[f.name] = json.dumps([])
                elif isinstance(val[0], str):
                    group.create_dataset(f.name, data=val, dtype=h5py.string_dtype())
                else:
                    group.create_dataset(f.name, data=np.array(val))
            else:
                # scalar: str, int, float, bool
                group.attrs[f.name] = val

    # -- read ---------------------------------------------------------------

    def _read_one(self, group, data_type: type | None = None):
        if data_type is None:
            return self._read_as_dict(group)

        kwargs: dict[str, Any] = {}
        for f in fields(data_type):
            tp = f.type
            name = f.name

            if name in group.attrs:
                raw = group.attrs[name]
                if _is_dict_type(tp) or _is_list_type(tp):
                    kwargs[name] = json.loads(raw)
                else:
                    kwargs[name] = _coerce_scalar(raw, tp)
            elif name in group:
                child = group[name]
                if isinstance(child, h5py.Group):
                    # It's a subgroup → nested dataclass
                    nested_type = _unwrap_optional(tp)
                    kwargs[name] = self._read_one(child, nested_type)
                else:
                    # It's a dataset
                    if _is_ndarray_type(tp):
                        kwargs[name] = child[()]
                    elif _is_list_type(tp):
                        elem_type = _get_list_element_type(tp)
                        if elem_type is str:
                            kwargs[name] = [
                                v.decode("utf-8") if isinstance(v, bytes) else v for v in child[()]
                            ]
                        else:
                            kwargs[name] = child[()].tolist()
                    else:
                        kwargs[name] = child[()]
            # else: field not present in file → omit, rely on dataclass default

        return data_type(**kwargs)

    @staticmethod
    def _read_as_dict(group) -> dict:
        """Read an HDF5 group into a plain dict (no data_type)."""
        result: dict[str, Any] = {}
        for key in group.attrs:
            result[key] = group.attrs[key]
        for key in group:
            child = group[key]
            if isinstance(child, h5py.Group):
                result[key] = DataclassHDF5Adapter._read_as_dict(child)
            else:
                arr = child[()]
                # Convert numpy scalars/arrays to Python types
                if isinstance(arr, np.ndarray):
                    result[key] = arr
                elif isinstance(arr, np.generic | bytes):
                    if isinstance(arr, bytes):
                        result[key] = arr.decode("utf-8")
                    else:
                        result[key] = arr.item()
                else:
                    result[key] = arr
        return result
