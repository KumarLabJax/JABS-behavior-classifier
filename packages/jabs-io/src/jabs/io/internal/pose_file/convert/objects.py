"""Legacy static and dynamic objects as components.

Objects are small, so they are read whole and held in memory. Every object
takes ADR 0002's uniform ``object, point, coord`` shape in ``(x, y)``,
replacing the legacy per-object shapes and coordinate orders: ``corners`` is
stored ``(x, y)``, ``lixit`` and ``food_hopper`` ``(y, x)``
(``pose_est_v5.py:11-14``), and a dynamic object's points follow their
``axis_order`` attribute, ``"yx"`` when it is absent (``pose_est_v7.py:15-19``).
"""

import logging
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt

from jabs.io.internal.pose_file.convert.errors import LegacyConversionError
from jabs.io.internal.pose_file.convert.inspect import LegacyLayout
from jabs.io.internal.pose_file.reader import attr_text
from jabs.io.internal.pose_file.types import Component

logger = logging.getLogger(__name__)

# Static objects stored (y, x), which are flipped.
_FLIPPED_STATIC = frozenset({"lixit", "food_hopper"})

# ADR 0002 has each object's ordering stated in its component's description.
_STATIC_DESCRIPTIONS = {
    "corners": "Arena corners, one point each; their order is not guaranteed.",
    "lixit": "Lixits: one point (the tip) or three (tip, left side, right side) each.",
    "food_hopper": "Food hopper corners, one point each, ordered to form a valid polygon.",
}


def object_components(path: Path, layout: LegacyLayout) -> list[Component]:
    """Return the legacy file's static and dynamic objects as components.

    Each static object becomes ``jabs.static_objects.<name>``. Each dynamic
    object becomes ``jabs.dynamic_objects.<name>.frame_index``, ``.points``
    and ``.counts``, on a ``sample`` axis indexed by ``.frame_index``; 3-D
    legacy points get one point per object.

    Args:
        path: The legacy file.
        layout: What ``inspect_legacy`` found in it; its ``static_objects``
            already leave out an empty lixit.

    Returns:
        The components, static objects first, each kind in name order.

    Raises:
        LegacyConversionError: If a static object is not ``(O, 2)`` or
            ``(O, P, 2)``, or a dynamic object's ``sample_indices`` or
            ``counts`` are not integers in the uint32 range.
    """
    components: list[Component] = []
    with h5py.File(path, "r") as h5:
        for name in layout.static_objects:
            components.append(_static(path, name, h5["static_objects"][name]))
        for name in layout.dynamic_objects:
            components.extend(_dynamic(path, name, h5["dynamic_objects"][name]))
    return components


def _static(path: Path, name: str, dataset: h5py.Dataset) -> Component:
    """Return one static object as an ``(object, point, coord)`` component."""
    data = dataset[...].astype(np.float32)
    if data.ndim == 2:
        data = data[:, None, :]
    if data.ndim != 3 or data.shape[-1] != 2:
        raise LegacyConversionError(
            f"{path}: static_objects/{name} has shape {dataset.shape}; expected (O, 2) or (O, P, 2)"
        )
    if name in _FLIPPED_STATIC:
        data = data[..., ::-1]
    return Component(
        id=f"jabs.static_objects.{name}",
        axes=("object", "point", "coord"),
        data=np.ascontiguousarray(data),
        missing={"policy": "none"},
        units="pixel",
        coord_order="xy",
        description=_STATIC_DESCRIPTIONS.get(name),
    )


def _uint32(path: Path, dataset: h5py.Dataset) -> npt.NDArray[np.uint32]:
    """Read an index dataset as uint32, refusing anything the cast would change."""
    name = dataset.name.lstrip("/")
    data = dataset[...]
    if data.dtype.kind not in "iu":
        raise LegacyConversionError(f"{path}: {name} has dtype {data.dtype}; expected integers")
    limit = np.iinfo(np.uint32).max
    bad = data[(data < 0) | (data > limit)] if data.size else data[:0]
    if bad.size:
        raise LegacyConversionError(
            f"{path}: {name} holds {bad.flat[0]}, outside the uint32 range 0..{limit}"
        )
    return data.astype(np.uint32)


def _dynamic(path: Path, name: str, group: h5py.Group) -> list[Component]:
    """Return one dynamic object's ``frame_index``, ``points`` and ``counts`` components."""
    prefix = f"jabs.dynamic_objects.{name}"
    index, counts = f"{prefix}.frame_index", f"{prefix}.counts"
    # Checked before the uint32 cast, which would wrap a negative or wide value.
    frame_index = _uint32(path, group["sample_indices"])
    count_data = _uint32(path, group["counts"])
    points: npt.NDArray[np.float32] = group["points"][...].astype(np.float32)
    if points.ndim == 3:
        points = points[:, :, None, :]
    # Absent means "yx"; inspection refused anything but "xy" and "yx".
    if attr_text(group["points"].attrs.get("axis_order", "yx")) == "yx":
        points = points[..., ::-1]
    return [
        Component(
            id=index,
            axes=("sample",),
            data=frame_index,
            missing={"policy": "none"},
            units="frame",
            # Its own index, so a clip tool filters it by the rule it applies
            # to everything else (ADR 0002).
            sparse_index=index,
        ),
        Component(
            id=f"{prefix}.points",
            axes=("sample", "object", "point", "coord"),
            data=np.ascontiguousarray(points),
            missing={"policy": "length", "length": counts},
            units="pixel",
            coord_order="xy",
            sparse_index=index,
        ),
        Component(
            id=counts,
            axes=("sample",),
            data=count_data,
            missing={"policy": "none"},
            units="unitless",
            sparse_index=index,
        ),
    ]
