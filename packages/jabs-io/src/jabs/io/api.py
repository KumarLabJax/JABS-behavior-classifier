import contextlib
from pathlib import Path
from typing import Any, TypeVar

from jabs.io.backends.json import JsonBackend
from jabs.io.backends.parquet import ParquetBackend
from jabs.io.registry import BACKEND_REGISTRY

T = TypeVar("T")

BACKEND_REGISTRY.register(JsonBackend())

with contextlib.suppress(ImportError):
    BACKEND_REGISTRY.register(ParquetBackend())


def load(path: Path, data_type: type[T], **kwargs) -> T:
    """Load data from a file using the appropriate backend."""
    path = Path(path)
    backend = BACKEND_REGISTRY.get_backend(path)
    return backend.load(path, data_type, **kwargs)


def save(data: Any, path: Path, **kwargs) -> None:
    """Save data to a file using the appropriate backend."""
    path = Path(path)
    backend = BACKEND_REGISTRY.get_backend(path)
    backend.save(data, path, **kwargs)
