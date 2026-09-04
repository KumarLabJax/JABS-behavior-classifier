"""Registry adapter for :class:`PoseFile`.

Without this, ``jabs.io.save(pose_file, path)`` resolves to the polymorphic
``DataclassHDF5Adapter`` — ``PoseFile`` is a dataclass, so it matches — which
cannot encode numpy object arrays and truncates the destination before finding
out. Registering at priority 10 puts the real codec ahead of it, following
``internal/prediction/hdf5.py``'s precedent.
"""

from pathlib import Path

from jabs.core.enums import StorageFormat
from jabs.io.base import Adapter
from jabs.io.internal.pose_file.reader import read_pose_file
from jabs.io.internal.pose_file.types import PoseFile
from jabs.io.internal.pose_file.writer import write_pose_file
from jabs.io.registry import register_adapter


@register_adapter(StorageFormat.HDF5, PoseFile, priority=10)
class PoseFileHDF5Adapter(Adapter):
    """Reads and writes revision-1 pose files through the registry.

    Lists are not supported: a pose file describes exactly one video, and the
    base class's ``_item_N`` subgroup convention would produce something no
    other implementation of the specification could read.
    """

    @classmethod
    def can_handle(cls, data_type: type) -> bool:  # noqa: D102
        return data_type is PoseFile

    def write(self, data: PoseFile, path: str | Path, **kwargs) -> None:
        """Write one pose file.

        Args:
            data: The pose file to write.
            path: Destination path.
            **kwargs: Unused; accepted for interface compatibility.

        Raises:
            TypeError: If ``data`` is a list rather than a single pose file.
        """
        if isinstance(data, list):
            raise TypeError("a pose file describes one video; write each PoseFile to its own path")
        write_pose_file(data, path)

    def read(self, path: str | Path, data_type: type | None = None) -> PoseFile:
        """Read one pose file.

        Args:
            path: The file to read.
            data_type: Unused; accepted for interface compatibility.

        Returns:
            The file's contents.
        """
        return read_pose_file(path)
