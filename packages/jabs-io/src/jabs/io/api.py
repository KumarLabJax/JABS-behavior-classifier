from pathlib import Path
from typing import Any

from jabs.io.base import DomainType
from jabs.io.registry import get_adapter, get_domain_type, get_storage_format


def load(path: str | Path, data_type: type[DomainType], **kwargs) -> DomainType:
    """Load data from a file using the appropriate backend."""
    path = Path(path)
    storage_format = get_storage_format(path)
    adapter = get_adapter(storage_format, data_type)
    if adapter is None:
        raise ValueError(
            f"No adapter registered for storage format {storage_format!r} and domain type {data_type!r}"
        )
    return adapter.read(path, **kwargs)


def save(data: Any, path: str | Path, **kwargs) -> None:
    """Save data to a file using the appropriate backend."""
    path = Path(path)
    storage_format = get_storage_format(path)
    domain_type = get_domain_type(data)
    adapter = get_adapter(storage_format, domain_type)
    if adapter is None:
        raise ValueError(
            f"No adapter registered for storage format {storage_format!r} and domain type {domain_type!r}"
        )
    return adapter.write(data, path, **kwargs)
