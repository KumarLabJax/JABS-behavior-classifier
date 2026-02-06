import json
from dataclasses import asdict, is_dataclass

import numpy as np

from jabs.io.backends.json.backend import JSON_ADAPTERS, T


@JSON_ADAPTERS.register()
class DataclassAdapter:
    """Generic JSON adapter for simple dataclasses.

    Handles serialization via `asdict()` and deserialization via
    `data_type(**json_dict)`. Works for dataclasses with JSON-serializable
    fields (primitives, lists, dicts, nested dataclasses with same constraints).

    Example:
        >>> @dataclass
        ... class Config:
        ...     name: str
        ...     value: int
        >>> adapter = DataclassAdapter(Config)
        >>> adapter.to_json(Config(name="test", value=42))
        '{"name": "test", "value": 42}'
    """

    @classmethod
    def can_handle(cls, data_type):
        """Check if this adapter can handle the given data type."""
        return is_dataclass(data_type)

    @staticmethod
    def _json_default(obj):
        # Handle numpy arrays / scalars
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):  # numpy scalar types (e.g., np.int64)
            return obj.item()

        # Handle nested dataclasses (extra-safe)
        if is_dataclass(obj):
            return asdict(obj)

        # Let JSON raise a helpful error for truly unsupported types
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    def to_json(self, data: T) -> str:
        """Serialize a dataclass instance to a JSON string."""
        return json.dumps(asdict(data), default=self._json_default)

    def from_json(self, json_str: str) -> T:
        """Deserialize a JSON string to a dataclass instance."""
        return self.data_type(**json.loads(json_str))
