"""Reader + discovery for V-JEPA embedding sidecars (format v1).

A sidecar is an HDF5 file written by the pbs-vjepa sidecar generator holding, per
identity, a dense per-frame embedding matrix and a coverage mask. It sits next to
the pose file as ``<video_stem>.vjepa.h5``. This module locates and reads it; it is
the JABS-side of the frozen sidecar contract and never imports torch or pbs_vjepa.
"""

from pathlib import Path

import h5py
import numpy as np

from jabs.core.utils import pose_file_stem

SUPPORTED_SIDECAR_FORMAT_VERSION = 1


class EmbeddingSidecarError(Exception):
    """Raised when a sidecar is missing an identity or has an unsupported format."""


def sidecar_path_for_pose(pose_file: Path) -> Path:
    """Return the sidecar path for a pose file: ``<dir>/<video_stem>.vjepa.h5``."""
    pose_file = Path(pose_file)
    return pose_file.parent / f"{pose_file_stem(pose_file)}.vjepa.h5"


def sidecar_exists(pose_file: Path) -> bool:
    """True if an embedding sidecar exists next to ``pose_file``."""
    return sidecar_path_for_pose(pose_file).is_file()


def read_provenance_hash(pose_file: Path) -> str:
    """Return the sidecar's provenance hash, or ``""`` if no sidecar exists.

    Used to key the feature cache on sidecar state so a stale or changed sidecar
    forces a recompute. Validates the sidecar's format version on the way (so a
    bad sidecar is caught even on a cache-hit path, where the sidecar is otherwise
    never opened).
    """
    path = sidecar_path_for_pose(pose_file)
    if not path.is_file():
        return ""
    with h5py.File(path, "r") as f:
        version = int(f.attrs["format_version"])
        if version != SUPPORTED_SIDECAR_FORMAT_VERSION:
            raise EmbeddingSidecarError(
                f"sidecar {path} format_version {version}, "
                f"expected {SUPPORTED_SIDECAR_FORMAT_VERSION}"
            )
        return str(f.attrs["provenance_hash"])


class EmbeddingInfo:
    """Per-identity embedding block loaded from a sidecar, uncovered frames NaN-filled."""

    def __init__(self, sidecar_path: Path, identity: int) -> None:
        with h5py.File(Path(sidecar_path), "r") as f:
            version = int(f.attrs["format_version"])
            if version != SUPPORTED_SIDECAR_FORMAT_VERSION:
                raise EmbeddingSidecarError(
                    f"sidecar {sidecar_path} format_version {version}, "
                    f"expected {SUPPORTED_SIDECAR_FORMAT_VERSION}"
                )
            self.embed_dim = int(f.attrs["embed_dim"])
            self.num_frames = int(f.attrs["num_frames"])
            key = str(identity)
            if key not in f["identities"]:
                raise EmbeddingSidecarError(f"identity {identity} not in sidecar {sidecar_path}")
            ig = f["identities"][key]
            emb = ig["embedding"][()].astype(np.float32)
            cov = ig["coverage"][()].astype(np.uint8)

        # uncovered frames -> NaN so JABS clean_features treats them as missing
        emb[cov == 0] = np.nan
        self.frame_embeddings = emb
        self.column_names = [f"emb_{j:04d}" for j in range(self.embed_dim)]
