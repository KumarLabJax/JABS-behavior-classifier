import types
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime
from typing import Union, get_args, get_origin

from jabs.core.enums import StorageFormat
from jabs.io.base import JSONAdapter
from jabs.io.registry import register_adapter


@register_adapter(StorageFormat.JSON)
class DataclassJSONAdapter(JSONAdapter):
    """JSON adapter for arbitrary dataclasses.

    Handles serialization via `asdict()` and deserialization via
    `data_type(**dict)`. Works for dataclasses with JSON-serializable
    fields (primitives, lists, dicts, nested dataclasses with same
    constraints). Non-standard types like numpy arrays and datetimes
    are handled by the base class `_json_default` fallback.
    """

    @classmethod
    def can_handle(cls, data_type):  # noqa: D102
        return is_dataclass(data_type)

    @staticmethod
    def _is_datetime_type(field_type) -> bool:
        """Check if a type annotation represents datetime or Optional[datetime]."""
        if field_type is datetime:
            return True
        origin = get_origin(field_type)
        if origin is Union or origin is types.UnionType:
            return datetime in get_args(field_type)
        return False

    def _encode_one(self, data) -> dict:
        return asdict(data)

    def _decode_one(self, data: dict, data_type: type | None = None):
        if data_type is None:
            return data
        for f in fields(data_type):
            val = data.get(f.name)
            if self._is_datetime_type(f.type) and isinstance(val, str):
                data[f.name] = datetime.fromisoformat(val)
        return data_type(**data)
