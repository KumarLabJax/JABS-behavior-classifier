"""Shared fixtures for embedding feature tests."""

from pathlib import Path

import h5py
import numpy as np
import pytest


def write_test_sidecar(
    path: Path,
    *,
    num_frames: int,
    embed_dim: int,
    identities: dict[int, tuple[np.ndarray, np.ndarray]],
    format_version: int = 1,
    provenance_hash: str = "deadbeef0000",
) -> None:
    """Write a minimal format-v1 sidecar for tests (mirrors the frozen contract)."""
    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = format_version
        f.attrs["provenance_hash"] = provenance_hash
        f.attrs["fps"] = 30.0
        f.attrs["num_frames"] = num_frames
        f.attrs["embed_dim"] = embed_dim
        grp = f.create_group("identities")
        for ident, (emb, cov) in identities.items():
            ig = grp.create_group(str(ident))
            ig.create_dataset("embedding", data=emb.astype(np.float32))
            ig.create_dataset("coverage", data=cov.astype(np.uint8))


@pytest.fixture
def sidecar_factory(tmp_path):
    """Return a callable that writes a sidecar under ``tmp_path`` and returns its path."""

    def _make(name="video.vjepa.h5", **kwargs):
        path = tmp_path / name
        write_test_sidecar(path, **kwargs)
        return path

    return _make


@pytest.fixture
def sidecar_writer():
    """Return the :func:`write_test_sidecar` helper for writing a sidecar at an explicit path."""
    return write_test_sidecar
