"""Component data supplied in frame blocks rather than as one array.

A converter that reads a long video's pose a block at a time should not have
to assemble the whole array in memory just so the writer can write it. A
:class:`BlockSource` declares the shape and dtype up front, which is all that
construction-time validation reads, and hands the writer its rows in blocks
along the first axis.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

# numpy dtype kinds the format calls "string": object (h5py variable length),
# bytes and unicode. Defined here rather than in types, which imports this
# module.
_STRING_KINDS = "OSU"


@dataclass(frozen=True, eq=False)
class BlockSource:
    """An array declared by shape and dtype and produced in blocks.

    ``blocks`` is a callable rather than an iterable so that nothing can
    consume the blocks by accident: validation reads only ``shape``, ``ndim``,
    ``size`` and ``dtype``, and the writer calls ``blocks()`` once. Blocks whose
    row count is a multiple of the dataset's chunk rows avoid recompressing
    chunks split across blocks.

    Attributes:
        shape: The full array's shape.
        dtype: The full array's dtype; coerced with ``np.dtype``. String dtypes
            are refused: string components are written from memory only.
        blocks: Returns ``(start, block)`` pairs, where ``block`` holds the
            rows starting at ``start`` along the first axis.
    """

    shape: tuple[int, ...]
    dtype: np.dtype
    blocks: Callable[[], Iterable[tuple[int, npt.NDArray[np.generic]]]]

    def __post_init__(self) -> None:
        """Coerce the shape and dtype, and refuse a string dtype."""
        object.__setattr__(self, "shape", tuple(int(n) for n in self.shape))
        object.__setattr__(self, "dtype", np.dtype(self.dtype))
        if self.dtype.kind in _STRING_KINDS:
            raise ValueError(
                f"a block source cannot carry string data (dtype {self.dtype.str!r}); "
                "string components must be supplied as in-memory arrays"
            )

    @property
    def ndim(self) -> int:
        """The number of dimensions."""
        return len(self.shape)

    @property
    def size(self) -> int:
        """The number of elements."""
        return int(np.prod(self.shape, dtype=np.int64))


ComponentData: TypeAlias = npt.NDArray[np.generic] | BlockSource
