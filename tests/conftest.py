"""Shared test configuration for the application test suite."""

import os
from collections.abc import Callable, Iterator
from pathlib import Path

import h5py
import numpy as np
import pytest

from jabs.pose_estimation import conversion_cache

# Render Qt widgets offscreen so UI tests don't open windows and steal focus. Set before
# any test module creates a QApplication; override with e.g. QT_QPA_PLATFORM=cocoa to
# watch the widgets.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_SAMPLE_POSE_V4 = Path(__file__).parent / "data" / "sample_pose_est_v4.h5"


@pytest.fixture(scope="session", autouse=True)
def _per_user_conversion_cache(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Keep conversions opened without a cache_dir out of the real per-user cache.

    Tests that patch ``conversion_cache.user_cache_dir`` themselves still win.
    A ``JABS_CACHE_DIR`` set in the developer's environment would override the
    redirect, so it is cleared.
    """
    user_cache = tmp_path_factory.mktemp("user_cache")
    with pytest.MonkeyPatch.context() as patch:
        patch.delenv(conversion_cache.CACHE_DIR_ENV, raising=False)
        patch.setattr(conversion_cache, "user_cache_dir", lambda _app: str(user_cache))
        yield


def _write_single_mouse_v4(path: Path, frames: int = 200) -> Path:
    """Write the v4 sample's first frames keeping only embed id 1; return ``path``.

    The other instances are masked, so the file has one identity in a format
    that can hold several mice (its masked instances become tail slots).
    """
    with h5py.File(_SAMPLE_POSE_V4, "r") as source, h5py.File(path, "w") as h5:
        pose = h5.create_group("poseest")
        for key, value in source["poseest"].attrs.items():
            pose.attrs[key] = value
        embed = source["poseest/instance_embed_id"][:frames]
        keep = embed == 1
        for name, dataset in source["poseest"].items():
            data = dataset[:1] if name == "instance_id_center" else dataset[:frames]
            if name == "id_mask":
                data = ~keep
            elif name == "instance_embed_id":
                data = np.where(keep, embed, 0).astype(embed.dtype)
            pose.create_dataset(name, data=data)
    return path


@pytest.fixture
def single_mouse_v4() -> Callable[[Path], Path]:
    """Return a function writing a 200-frame, one-identity ``pose_est_v4`` file to a path."""
    return _write_single_mouse_v4
