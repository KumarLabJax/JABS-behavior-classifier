from pathlib import Path
from typing import Any

from jabs.io.base import DomainType
from jabs.io.registry import get_adapter, get_domain_type, get_storage_format


def load(path: str | Path, data_type: type[DomainType], **kwargs) -> DomainType:
    """Load data from a file using the appropriate backend."""
    path = Path(path)
    return get_adapter(get_storage_format(path), data_type).read(path, **kwargs)


def save(data: Any, path: str | Path, **kwargs) -> None:
    """Save data to a file using the appropriate backend."""
    path = Path(path)
    return get_adapter(get_storage_format(path), get_domain_type(data)).write(data, path, **kwargs)
