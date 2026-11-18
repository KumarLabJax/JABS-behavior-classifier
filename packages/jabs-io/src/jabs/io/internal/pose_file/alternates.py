"""Alternate encodings of a component's data (ADR 0002, "Alternate encodings").

An alternate is a derived copy of a component's data in another encoding,
under a name, so a reader with different access needs can read the same data
without decoding the baseline. This revision defines one encoding for them,
``ragged``: a payload indexed by two CSR offset arrays. Alternates of one name
on different components are one encoding and name the same two offset
datasets, so the offsets are their own object, a :class:`RaggedIndex`, which
those alternates share.

``types`` refers to :class:`Alternate` only for annotation, so this module can
import from it without a cycle.
"""

import re
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from jabs.io.internal.pose_file.blocks import ComponentData
from jabs.io.internal.pose_file.types import DTYPE_NAMES, dtype_name

# The schema's alternateName: one lowercase segment, no dots, because the name
# is a key a reader looks an encoding up by rather than a namespaced id.
ALTERNATE_NAME_RE = re.compile(r"^[a-z0-9]([a-z0-9_-]*[a-z0-9])?$")

# What a ragged payload holds: every leaf item, or one value per group.
RAGGED_LEVELS = ("leaf", "group")

_OFFSET_DTYPE = np.dtype("uint64")


def alternate_key(component_id: str, name: str) -> str:
    """The key that names one component's alternate.

    Used to look up an alternate's layout and in history entries'
    ``synthesized`` and ``dropped`` lists, where ADR 0002 writes it this way.

    Args:
        component_id: The component that carries the alternate.
        name: The alternate's name.

    Returns:
        ``<component id>:alternate:<name>``.
    """
    return f"{component_id}:alternate:{name}"


def _check_path(owner: str, path: str) -> None:
    """Refuse a path that is not absolute, as the schema's hdf5Path is."""
    if not path.startswith("/"):
        raise ValueError(f"{owner}: path must be absolute, got {path!r}")


@dataclass(frozen=True, eq=False)
class RaggedIndex:
    """The two CSR offset arrays a ``ragged`` payload is decoded through.

    ``instance_offsets`` holds group indices, one entry per ``(frame, slot)``
    in row-major order plus a trailing one; ``group_offsets`` holds payload
    indices, one entry per group plus a trailing one. Every alternate of one
    name shares a single instance, which is how the writer knows to write the
    offsets once.

    ``eq`` is disabled because the offsets are arrays; identity is what
    sharing means here.

    Attributes:
        group_offsets_path: Absolute HDF5 path of ``group_offsets``.
        instance_offsets_path: Absolute HDF5 path of ``instance_offsets``.
        group_offsets: One-dimensional ``uint64`` offsets into the leaf payload.
        instance_offsets: One-dimensional ``uint64`` offsets into the groups.
    """

    group_offsets_path: str
    instance_offsets_path: str
    group_offsets: ComponentData
    instance_offsets: ComponentData

    def __post_init__(self) -> None:
        """Validate the paths and that both offsets are 1-D ``uint64``."""
        for name in ("group_offsets", "instance_offsets"):
            path = getattr(self, f"{name}_path")
            _check_path(f"ragged {name}", path)
            # Shape and dtype only: a block source's values are not in memory,
            # and reading them would consume the blocks the writer needs.
            offsets = getattr(self, name)
            if offsets.ndim != 1 or offsets.dtype != _OFFSET_DTYPE:
                raise ValueError(
                    f"ragged {name} at {path}: must be one-dimensional uint64, got "
                    f"{offsets.dtype.name}{tuple(offsets.shape)}"
                )
        if self.group_offsets_path == self.instance_offsets_path:
            raise ValueError(
                f"ragged offsets: group_offsets and instance_offsets share the path "
                f"{self.group_offsets_path}"
            )


@dataclass(frozen=True, eq=False)
class Alternate:
    """A derived copy of a component's data in the ``ragged`` encoding.

    An alternate carries every declaration a component does except ``id``,
    which it shares with its component, and ``skeleton`` and ``provenance``,
    which are the component's. ``eq`` is disabled because the payload is an
    array.

    Attributes:
        name: The encoding's name, e.g. ``ragged``; one lowercase segment.
        path: Absolute HDF5 path of the payload, distinct from every other
            payload in the file.
        axes: Axis name per payload dimension, e.g. ``("point", "coord")``.
        data: The payload: an in-memory array or a
            :class:`~jabs.io.internal.pose_file.blocks.BlockSource`.
        missing: The missing-value policy, as for a component.
        index: The offsets the payload is decoded through, shared by every
            alternate of this name.
        level: ``"leaf"`` when the payload holds every leaf item, ``"group"``
            when it holds one value per group. Declared because no length
            check can tell them apart when every group holds one leaf.
        units: Physical units, required with a ``coord`` axis.
        coord_order: ``"xy"`` or ``"yx"``, required with a ``coord`` axis.
        description: Optional human-readable description.
        layout: Declared HDF5 storage; the writer's layout overrides it.
        extra: Namespaced producer metadata.
    """

    name: str
    path: str
    axes: tuple[str, ...]
    data: ComponentData
    missing: dict
    index: RaggedIndex
    level: str = "leaf"
    units: str | None = None
    coord_order: str | None = None
    description: str | None = None
    layout: dict | None = None
    extra: dict | None = None

    def __post_init__(self) -> None:
        """Validate the name, level, path, axis declarations and dtype."""
        if not ALTERNATE_NAME_RE.fullmatch(self.name):
            raise ValueError(
                f"invalid alternate name {self.name!r}: expected one lowercase segment of "
                "letters, digits, '_' or '-'"
            )
        label = f"alternate {self.name!r}"
        if self.level not in RAGGED_LEVELS:
            raise ValueError(f"{label}: level must be one of {RAGGED_LEVELS}, got {self.level!r}")
        _check_path(label, self.path)
        if len(self.axes) != self.data.ndim:
            raise ValueError(
                f"{label}: declared {len(self.axes)} axes {tuple(self.axes)} but the payload "
                f"has {self.data.ndim} dimensions {self.data.shape}"
            )
        if "coord" in self.axes:
            if self.units is None or self.coord_order is None:
                raise ValueError(
                    f"{label}: an alternate with a coord axis must declare units and coord_order"
                )
            pair = self.data.shape[list(self.axes).index("coord")]
            if pair != 2:
                raise ValueError(
                    f"{label}: coord axis is {pair} wide; coord_order describes two values "
                    f"({self.coord_order!r}) and nothing else"
                )
        if self.dtype not in DTYPE_NAMES:
            raise ValueError(f"{label}: unsupported dtype {self.dtype!r}")
        if self.missing.get("policy") == "nan" and self.dtype not in ("float32", "float64"):
            raise ValueError(
                f"{label}: a nan missing-policy needs a float payload, not {self.dtype!r}"
            )
        # As for a component: frozen freezes the bindings, not the objects, so
        # without this a caller could mutate what was validated.
        object.__setattr__(self, "axes", tuple(self.axes))
        object.__setattr__(self, "missing", MappingProxyType(dict(self.missing)))
        if self.extra is not None:
            object.__setattr__(self, "extra", MappingProxyType(dict(self.extra)))

    @property
    def dtype(self) -> str:
        """The payload's dtype name, as the manifest records it."""
        return dtype_name(self.data)
