"""JSON IO backend for JABS."""

from .adapters import DataclassAdapter
from .backend import JsonBackend

__all__ = [
    "JsonBackend",
]
