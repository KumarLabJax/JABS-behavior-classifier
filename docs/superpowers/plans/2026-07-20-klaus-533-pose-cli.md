# KLAUS-533 Pose-Estimation CLI + Image Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wrap `jabs-vision`'s library-only HRNet pose inference in a `jabs-pose` CLI that reads a video and writes a valid `*_pose_est_v2.h5`, backed by a forward-looking pose writer in `jabs-io`, and ship a GPU container image.

**Architecture:** A new `PoseHDF5Adapter` in `jabs-io` (registered for `(HDF5, PoseData)`, selected via the generic `jabs.io.save(..., legacy=JabsPoseVersion.V2)`) writes the legacy v2 layout from the canonical `PoseData` model. `jabs-vision` gains an `imageio[ffmpeg]` frame reader and a Click CLI that composes `load_pose_model(strict=True)` + `predict_single_pose`, assembles a `PoseData`, and calls `save`. Coordinates flow inference `(y,x)` → canonical `PoseData` `(x,y)` → on-disk v2 `(y,x)`. A GPU `Dockerfile.pose` (built in CI, published later) runs the CLI.

**Tech Stack:** Python 3.10+, uv workspace, PyTorch (HRNet), h5py, imageio[ffmpeg], Click, pytest.

## Global Constraints

- **Python floor 3.10** (`jabs-vision`, `jabs-io`, `jabs-core` all `requires-python = ">=3.10,<3.15"`); CI runs tests on 3.13, wheels build on 3.10.
- **Ruff**: line length 99, target py310. Google-style docstrings on every module/class/public function; one-liner on `_`-prefixed helpers. American spelling. Never use `–` (EN DASH) in docstrings — use `-`.
- **Type hints on all new code**: modern syntax (`list[str]`, `X | None`), `npt.NDArray[np.float64]` for arrays, `pathlib.Path` for paths.
- **Logging**: `logger = logging.getLogger(__name__)` per module; lazy `%` formatting; never `print()`.
- **Dependencies via `uv` only** (`uv add --package <pkg> ...`); run everything with `uv run`. Never edit `uv.lock` by hand.
- **Version lockstep**: all sub-package versions must equal the root version (`0.46.1`) — CI enforces via `dev/sync-versions.sh`. This work adds no version bump (that happens at release).
- **Tests**: pytest (plain functions), `monkeypatch` + stdlib `unittest.mock` only (no `pytest-mock`). Mirror source paths; per-package `tests/` dirs run separately.
- **Coordinate conventions**: canonical `PoseData.points` are `(x, y)`; legacy v2 on-disk `poseest/points` are `(y, x)`, `uint16`; `poseest/confidence` is `float32`; `poseest.attrs["version"] = np.asarray([2, 0], uint16)`. 12 keypoints, order `NOSE=0 … TIP_TAIL=11`.

## Execution notes (inline vs. infra)

- **Tasks 1–6 are CPU-only, TDD, executable inline** in this session. They fully cover structural correctness (format, dtypes, axis order) via the round-trip test (Task 6).
- **Task 7** (Dockerfile + CI job) is written inline; the actual image build/GPU run needs infra.
- **Task 8** (spike + dev-Batch acceptance) requires the real gait-model weights + a GPU and is run manually on dev-Batch. It validates real-weight concerns (strict checkpoint key match, `HEAD_ARCH` string) that do **not** affect the code structure built in Tasks 1–7, so building inline first is low-risk.

## File Structure

| Package | File | Responsibility |
|---------|------|----------------|
| jabs-core | `src/jabs/core/enums/pose_version.py` (create) | `JabsPoseVersion` IntEnum |
| jabs-core | `src/jabs/core/enums/__init__.py` (modify) | export `JabsPoseVersion` |
| jabs-core | `src/jabs/core/types/pose.py` (modify) | first-class `confidence` field on `PoseData` |
| jabs-io | `src/jabs/io/internal/pose/hdf5.py` (create) | `PoseHDF5Adapter` (writes legacy v2) |
| jabs-io | `src/jabs/io/internal/pose/__init__.py` (modify) | import `PoseHDF5Adapter` so it registers |
| jabs-vision | `src/jabs/vision/io/__init__.py`, `io/frames.py` (create) | `read_frames`, `video_fps` |
| jabs-vision | `src/jabs/vision/cli/__init__.py`, `cli/pose.py` (create) | `run_pose_inference`, `pose_command`, `main` |
| jabs-vision | `pyproject.toml` (modify) | `[project.scripts]` + deps (`click`, `imageio[ffmpeg]`, `h5py`) |
| root | `tests/io/test_pose_v2_roundtrip.py` (create) | write via adapter → read via monolith `PoseEstimationV2` |
| repo | `containers/docker/Dockerfile.pose` (create) | GPU image for `jabs-pose` |
| repo | `.github/workflows/build-pose-image.yml` (create) | CI build-only job |

---

### Task 1: `JabsPoseVersion` enum (jabs-core)

**Files:**
- Create: `packages/jabs-core/src/jabs/core/enums/pose_version.py`
- Modify: `packages/jabs-core/src/jabs/core/enums/__init__.py`
- Test: `packages/jabs-core/tests/test_enums.py`

**Interfaces:**
- Produces: `JabsPoseVersion(IntEnum)` with members `V2 = 2`, `V3 = 3`; importable as `from jabs.core.enums import JabsPoseVersion`.

- [ ] **Step 1: Write the failing test** — append to `packages/jabs-core/tests/test_enums.py`:

```python
from jabs.core.enums import JabsPoseVersion


def test_jabs_pose_version_is_int_aligned():
    assert JabsPoseVersion.V2 == 2
    assert JabsPoseVersion.V3 == 3
    assert int(JabsPoseVersion.V2) == 2


def test_jabs_pose_version_ordered():
    # IntEnum members compare numerically, matching the legacy integer majors.
    assert JabsPoseVersion.V2 < JabsPoseVersion.V3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/jabs-core/tests/test_enums.py -k jabs_pose_version -v`
Expected: FAIL — `ImportError: cannot import name 'JabsPoseVersion'`.

- [ ] **Step 3: Create the enum** — `packages/jabs-core/src/jabs/core/enums/pose_version.py`:

```python
"""Enum for legacy JABS pose-file format versions."""

from enum import IntEnum


class JabsPoseVersion(IntEnum):
    """Legacy JABS pose-file format major versions (the ``pose_est_vN`` convention).

    The ``pose_est_vN`` convention is planned for deprecation in favor of a future
    backwards-compatible pose format. Until then this enum names the legacy layouts;
    it is used as the ``legacy=`` selector when writing pose files. Members are aligned
    to the historical integer majors (2-8) so numeric comparisons keep working.
    """

    V2 = 2
    V3 = 3
```

- [ ] **Step 4: Export it** — in `packages/jabs-core/src/jabs/core/enums/__init__.py`, add the import (keep alphabetical grouping) and the `__all__` entry:

```python
from .pose_version import JabsPoseVersion
```

Add `"JabsPoseVersion",` to the `__all__` list (place before `"Method"`).

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest packages/jabs-core/tests/test_enums.py -k jabs_pose_version -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Lint + commit**

```bash
uv run ruff check --fix packages/jabs-core && uv run ruff format packages/jabs-core
git add packages/jabs-core/src/jabs/core/enums/pose_version.py \
        packages/jabs-core/src/jabs/core/enums/__init__.py \
        packages/jabs-core/tests/test_enums.py
git commit -m "feat(jabs-core): add JabsPoseVersion enum for legacy pose formats"
```

---

### Task 2: First-class `confidence` field on `PoseData` (jabs-core)

**Files:**
- Modify: `packages/jabs-core/src/jabs/core/types/pose.py:60-73` (fields) and `:75-132` (`__post_init__`)
- Test: `packages/jabs-core/tests/test_pose_data.py` (create)

**Interfaces:**
- Consumes: nothing new.
- Produces: `PoseData(..., confidence: npt.NDArray[np.float32] | None = None)`. When present, `confidence.shape == point_mask.shape == (num_identities, num_frames, num_keypoints)`; otherwise `ValueError`. Backward compatible (defaults to `None`).

- [ ] **Step 1: Write the failing test** — create `packages/jabs-core/tests/test_pose_data.py`:

```python
"""Tests for the PoseData dataclass, focused on the confidence field."""

import numpy as np
import pytest

from jabs.core.types import PoseData


def _kwargs(num_idents=1, num_frames=3, num_kp=12):
    return {
        "points": np.zeros((num_idents, num_frames, num_kp, 2), dtype=np.float64),
        "point_mask": np.ones((num_idents, num_frames, num_kp), dtype=bool),
        "identity_mask": np.ones((num_idents, num_frames), dtype=bool),
        "body_parts": [f"kp{i}" for i in range(num_kp)],
        "edges": [],
        "fps": 30,
    }


def test_confidence_defaults_to_none():
    pose = PoseData(**_kwargs())
    assert pose.confidence is None


def test_confidence_accepts_matching_shape():
    kw = _kwargs()
    kw["confidence"] = np.full((1, 3, 12), 0.9, dtype=np.float32)
    pose = PoseData(**kw)
    assert pose.confidence.shape == (1, 3, 12)


def test_confidence_wrong_shape_raises():
    kw = _kwargs()
    kw["confidence"] = np.full((1, 3, 11), 0.9, dtype=np.float32)
    with pytest.raises(ValueError, match="confidence shape"):
        PoseData(**kw)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/jabs-core/tests/test_pose_data.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'confidence'`.

- [ ] **Step 3: Add the field** — in `packages/jabs-core/src/jabs/core/types/pose.py`, add the field after `bounding_boxes` (line 67):

```python
    confidence: np.ndarray | None = None
```

Add to the class docstring `Attributes:` block (after the `bounding_boxes` entry):

```
        confidence: Optional per-keypoint confidence scores, shape
            (num_identities, num_frames, num_keypoints).  None when unknown.
```

- [ ] **Step 4: Add validation** — in `__post_init__`, after the `bounding_boxes` check (line 114) and before the `edges` check, add:

```python
        if self.confidence is not None and self.confidence.shape != (
            num_idents,
            num_frames,
            num_keypoints,
        ):
            raise ValueError(
                f"confidence shape {self.confidence.shape} must match points "
                f"dimensions {(num_idents, num_frames, num_keypoints)}"
            )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest packages/jabs-core/tests/test_pose_data.py -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Verify no regression in jabs-core + jabs-io (NWB unaffected)**

Run: `uv run pytest packages/jabs-core/tests packages/jabs-io/tests -q`
Expected: PASS (all existing tests green; `confidence=None` default keeps every existing constructor and the NWB read path working).

- [ ] **Step 7: Lint + commit**

```bash
uv run ruff check --fix packages/jabs-core && uv run ruff format packages/jabs-core
git add packages/jabs-core/src/jabs/core/types/pose.py \
        packages/jabs-core/tests/test_pose_data.py
git commit -m "feat(jabs-core): add optional confidence field to PoseData"
```

---

### Task 3: `PoseHDF5Adapter` — legacy v2 writer (jabs-io)

**Files:**
- Create: `packages/jabs-io/src/jabs/io/internal/pose/hdf5.py`
- Modify: `packages/jabs-io/src/jabs/io/internal/pose/__init__.py`
- Test: `packages/jabs-io/tests/internal/pose/test_hdf5.py` (create)

**Interfaces:**
- Consumes: `PoseData` (with the Task 2 `confidence` field), `JabsPoseVersion` (Task 1), `StorageFormat`, `register_adapter`, `HDF5Adapter`.
- Produces: `PoseHDF5Adapter` registered `(StorageFormat.HDF5, PoseData, priority=10)`, with `write(data: PoseData, path, *, legacy: JabsPoseVersion = JabsPoseVersion.V2, **kwargs)`. Reachable via `jabs.io.save(pose_data, path, legacy=JabsPoseVersion.V2)`. `read` raises `NotImplementedError`.

- [ ] **Step 1: Write the failing tests** — create `packages/jabs-io/tests/internal/pose/test_hdf5.py`:

```python
"""Tests for the legacy v2 PoseData HDF5 writer."""

import h5py
import numpy as np
import pytest

from jabs.core.enums import JabsPoseVersion, StorageFormat
from jabs.core.types import PoseData
from jabs.io import save
from jabs.io.internal.pose.hdf5 import PoseHDF5Adapter
from jabs.io.registry import get_adapter


def _make_pose(num_frames=3, num_idents=1):
    n_kp = 12
    # distinct (x, y) per (frame, keypoint): x = f*100 + k, y = f*100 + k + 1
    points = np.zeros((num_idents, num_frames, n_kp, 2), dtype=np.float64)
    for f in range(num_frames):
        for k in range(n_kp):
            points[0, f, k] = [f * 100 + k, f * 100 + k + 1]
    return PoseData(
        points=points,
        point_mask=np.ones((num_idents, num_frames, n_kp), dtype=bool),
        identity_mask=np.ones((num_idents, num_frames), dtype=bool),
        body_parts=[f"kp{i}" for i in range(n_kp)],
        edges=[],
        fps=30,
        confidence=np.full((num_idents, num_frames, n_kp), 0.9, dtype=np.float32),
        metadata={"config": "gait-model.yaml", "model": "gait-model.pth"},
    )


@pytest.fixture
def adapter():
    return PoseHDF5Adapter()


def test_writes_v2_layout(adapter, tmp_path):
    out = tmp_path / "x_pose_est_v2.h5"
    adapter.write(_make_pose(), out, legacy=JabsPoseVersion.V2)

    with h5py.File(out, "r") as h5:
        pose = h5["poseest"]
        assert pose.attrs["version"].tolist() == [2, 0]
        assert pose["points"].dtype == np.uint16
        assert pose["points"].shape == (3, 12, 2)
        assert pose["confidence"].dtype == np.float32
        assert pose["confidence"].shape == (3, 12)
        # canonical (x, y) written as on-disk (y, x): frame 0, kp 0 was (0, 1) -> (1, 0)
        assert pose["points"][0, 0].tolist() == [1, 0]
        assert pose["points"].attrs["config"] == "gait-model.yaml"
        assert pose["points"].attrs["model"] == "gait-model.pth"


def test_save_dispatches_to_pose_adapter(tmp_path):
    out = tmp_path / "y_pose_est_v2.h5"
    save(_make_pose(), out, legacy=JabsPoseVersion.V2)
    with h5py.File(out, "r") as h5:
        assert "poseest/points" in h5


def test_registry_resolves_pose_hdf5_adapter():
    resolved = get_adapter(StorageFormat.HDF5, PoseData)
    assert isinstance(resolved, PoseHDF5Adapter)


def test_can_handle_truth_table():
    assert PoseHDF5Adapter.can_handle(PoseData) is True
    assert PoseHDF5Adapter.can_handle(dict) is False


def test_unsupported_legacy_version_raises(adapter, tmp_path):
    with pytest.raises(ValueError, match="Unsupported legacy pose version"):
        adapter.write(_make_pose(), tmp_path / "z.h5", legacy=JabsPoseVersion.V3)


def test_multi_identity_raises(adapter, tmp_path):
    with pytest.raises(ValueError, match="single-identity"):
        adapter.write(_make_pose(num_idents=2), tmp_path / "z.h5")


def test_missing_confidence_raises(adapter, tmp_path):
    pose = _make_pose()
    object.__setattr__(pose, "confidence", None)  # frozen dataclass
    with pytest.raises(ValueError, match="requires confidence"):
        adapter.write(pose, tmp_path / "z.h5")


def test_read_not_implemented(adapter, tmp_path):
    out = tmp_path / "x_pose_est_v2.h5"
    adapter.write(_make_pose(), out)
    with pytest.raises(NotImplementedError):
        adapter.read(out)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/jabs-io/tests/internal/pose/test_hdf5.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'jabs.io.internal.pose.hdf5'`.

- [ ] **Step 3: Create the adapter** — `packages/jabs-io/src/jabs/io/internal/pose/hdf5.py`:

```python
"""HDF5 adapter for PoseData using the legacy ``pose_est_vN`` layout.

Legacy v2 layout::

    / (root)
      poseest/ (group)
        attrs: version = uint16[2] = [2, 0]
        points (dataset, uint16, shape n_frames x 12 x 2, order (y, x))
          attrs: config (str), model (str)
        confidence (dataset, float32, shape n_frames x 12)

Only single-identity v2 is supported today; the ``legacy=`` selector reserves
space for further legacy versions and the future backwards-compatible default.
"""

import logging
from pathlib import Path

import h5py
import numpy as np

from jabs.core.enums import JabsPoseVersion, StorageFormat
from jabs.core.types.pose import PoseData
from jabs.io.base import HDF5Adapter
from jabs.io.registry import register_adapter

logger = logging.getLogger(__name__)


@register_adapter(StorageFormat.HDF5, PoseData, priority=10)
class PoseHDF5Adapter(HDF5Adapter):
    """Write PoseData to a legacy ``pose_est_vN`` HDF5 file.

    Overrides ``write`` directly (like PredictionHDF5Adapter) so it can branch on
    the ``legacy`` version selector. ``read`` is not implemented in this increment;
    the legacy readers live in the ``jabs.pose_estimation`` monolith.
    """

    @classmethod
    def can_handle(cls, data_type: type) -> bool:  # noqa: D102
        return data_type is PoseData

    def _write_one(self, data, group) -> None:
        raise NotImplementedError("Use write() directly for PoseHDF5Adapter")

    def _read_one(self, group, data_type=None):
        raise NotImplementedError("Use read() directly for PoseHDF5Adapter")

    def write(  # noqa: D102
        self,
        data: PoseData,
        path: str | Path,
        *,
        legacy: JabsPoseVersion = JabsPoseVersion.V2,
        **kwargs,
    ) -> None:
        if legacy is not JabsPoseVersion.V2:
            raise ValueError(f"Unsupported legacy pose version: {legacy!r} (only V2 is supported)")
        with h5py.File(path, "w") as h5:
            self._write_v2(data, h5)

    def read(self, path: str | Path, data_type: type | None = None, **kwargs):  # noqa: D102
        raise NotImplementedError(
            "Reading pose HDF5 into PoseData is not implemented; use "
            "jabs.pose_estimation.PoseEstimationV2 for legacy v2 reads."
        )

    @staticmethod
    def _write_v2(data: PoseData, h5: h5py.File) -> None:
        """Write a single-identity PoseData into an open HDF5 file as v2."""
        if data.points.shape[0] != 1:
            raise ValueError(
                f"Legacy v2 is single-identity; got {data.points.shape[0]} identities"
            )
        if data.confidence is None:
            raise ValueError("Legacy v2 requires confidence; PoseData.confidence is None")

        # canonical (x, y) -> on-disk (y, x); float -> uint16
        points_yx = np.flip(data.points[0], axis=-1).astype(np.uint16)
        confidence = data.confidence[0].astype(np.float32)

        pose_grp = h5.require_group("poseest")
        points_ds = pose_grp.create_dataset("points", data=points_yx)
        points_ds.attrs["config"] = str(data.metadata.get("config", ""))
        points_ds.attrs["model"] = str(data.metadata.get("model", ""))
        pose_grp.create_dataset("confidence", data=confidence)
        pose_grp.attrs["version"] = np.asarray([2, 0], dtype=np.uint16)
```

- [ ] **Step 4: Register on import** — replace `packages/jabs-io/src/jabs/io/internal/pose/__init__.py` with:

```python
"""Pose estimation adapters (NWB requires the [nwb] extra)."""

from jabs.io.internal.pose.hdf5 import PoseHDF5Adapter
from jabs.io.internal.pose.nwb import PoseNWBAdapter

__all__ = ["PoseHDF5Adapter", "PoseNWBAdapter"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest packages/jabs-io/tests/internal/pose/test_hdf5.py -v`
Expected: PASS (8 passed).

- [ ] **Step 6: Lint + commit**

```bash
uv run ruff check --fix packages/jabs-io && uv run ruff format packages/jabs-io
git add packages/jabs-io/src/jabs/io/internal/pose/hdf5.py \
        packages/jabs-io/src/jabs/io/internal/pose/__init__.py \
        packages/jabs-io/tests/internal/pose/test_hdf5.py
git commit -m "feat(jabs-io): add PoseHDF5Adapter writing legacy pose_est_v2"
```

---

### Task 4: `read_frames` video reader (jabs-vision)

**Files:**
- Create: `packages/jabs-vision/src/jabs/vision/io/__init__.py`
- Create: `packages/jabs-vision/src/jabs/vision/io/frames.py`
- Test: `packages/jabs-vision/tests/io/__init__.py` (empty), `packages/jabs-vision/tests/io/test_frames.py`
- Modify: `packages/jabs-vision/pyproject.toml` (add `imageio[ffmpeg]` dep — done in Step 0)

**Interfaces:**
- Produces: `read_frames(video: Path) -> Iterator[npt.NDArray[np.uint8]]` yielding `(H, W, 3)` RGB uint8 frames; `video_fps(video: Path) -> int` returning rounded fps (default 30 if unknown).

- [ ] **Step 0: Add the dependency**

```bash
uv add --package jabs-vision "imageio[ffmpeg]>=2.31.6"
```
Expected: `pyproject.toml` gains `imageio[ffmpeg]` under `[project.dependencies]`; `uv.lock` updates.

- [ ] **Step 1: Write the failing test** — create `packages/jabs-vision/tests/io/__init__.py` (empty) and `packages/jabs-vision/tests/io/test_frames.py`:

```python
"""Tests for the imageio-based frame reader."""

from unittest import mock

import numpy as np

import jabs.vision.io.frames as frames_mod
from jabs.vision.io import read_frames, video_fps


class _FakeReader:
    def __init__(self, frames, meta):
        self._frames = frames
        self._meta = meta
        self.closed = False

    def iter_data(self):
        yield from self._frames

    def get_meta_data(self):
        return self._meta

    def close(self):
        self.closed = True


def test_read_frames_yields_all_frames(monkeypatch):
    fake_frames = [np.zeros((4, 4, 3), dtype=np.uint8), np.ones((4, 4, 3), dtype=np.uint8)]
    reader = _FakeReader(fake_frames, {"fps": 30.0})
    monkeypatch.setattr(frames_mod.imageio, "get_reader", mock.Mock(return_value=reader))

    out = list(read_frames("video.mp4"))

    assert len(out) == 2
    assert out[0].shape == (4, 4, 3)
    assert reader.closed is True


def test_video_fps_rounds(monkeypatch):
    reader = _FakeReader([], {"fps": 29.97})
    monkeypatch.setattr(frames_mod.imageio, "get_reader", mock.Mock(return_value=reader))
    assert video_fps("video.mp4") == 30


def test_video_fps_defaults_to_30(monkeypatch):
    reader = _FakeReader([], {})  # no fps key
    monkeypatch.setattr(frames_mod.imageio, "get_reader", mock.Mock(return_value=reader))
    assert video_fps("video.mp4") == 30
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/jabs-vision/tests/io/test_frames.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'jabs.vision.io'`.

- [ ] **Step 3: Create the reader** — `packages/jabs-vision/src/jabs/vision/io/frames.py`:

```python
"""Video frame reading for pose inference (imageio + ffmpeg).

imageio[ffmpeg] is used deliberately to match the reference decoder (mtr), so
decoded pixels - and therefore argmax coordinates - are consistent with the
legacy pipeline.
"""

import logging
from collections.abc import Iterator
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

_DEFAULT_FPS = 30


def read_frames(video: str | Path) -> Iterator[npt.NDArray[np.uint8]]:
    """Yield ``(H, W, 3)`` RGB uint8 frames from a video.

    Args:
        video: Path to the input video.

    Yields:
        Successive RGB frames as uint8 arrays.
    """
    logger.info("reading frames from %s", video)
    reader = imageio.get_reader(str(video), format="ffmpeg")
    try:
        yield from reader.iter_data()
    finally:
        reader.close()


def video_fps(video: str | Path) -> int:
    """Return the video frame rate rounded to an int (default 30 if unknown).

    Args:
        video: Path to the input video.

    Returns:
        Frames per second, rounded to the nearest integer.
    """
    reader = imageio.get_reader(str(video), format="ffmpeg")
    try:
        fps = reader.get_meta_data().get("fps")
    finally:
        reader.close()
    if fps is None:
        logger.warning("no fps in %s metadata; defaulting to %d", video, _DEFAULT_FPS)
        return _DEFAULT_FPS
    return int(round(fps))
```

- [ ] **Step 4: Create the package init** — `packages/jabs-vision/src/jabs/vision/io/__init__.py`:

```python
"""Video I/O for jabs-vision inference."""

from jabs.vision.io.frames import read_frames, video_fps

__all__ = ["read_frames", "video_fps"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest packages/jabs-vision/tests/io/test_frames.py -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Lint + commit**

```bash
uv run ruff check --fix packages/jabs-vision && uv run ruff format packages/jabs-vision
git add packages/jabs-vision/src/jabs/vision/io/ packages/jabs-vision/tests/io/ \
        packages/jabs-vision/pyproject.toml uv.lock
git commit -m "feat(jabs-vision): add imageio-based video frame reader"
```

---

### Task 5: Pose CLI (jabs-vision)

**Files:**
- Create: `packages/jabs-vision/src/jabs/vision/cli/__init__.py`, `cli/pose.py`
- Modify: `packages/jabs-vision/pyproject.toml` (deps + `[project.scripts]`)
- Test: `packages/jabs-vision/tests/cli/__init__.py` (empty), `packages/jabs-vision/tests/cli/test_pose.py`

**Interfaces:**
- Consumes: `read_frames`, `video_fps` (Task 4); `load_pose_model`, `predict_single_pose` (`jabs.vision.hrnet_msfork`); `PoseData`, `JabsPoseVersion` (jabs-core); `jabs.io.save` (Task 3).
- Produces: `run_pose_inference(*, video, out, config, checkpoint, batch_size=1, device=None) -> None`; `pose_command` (Click); `main()`. Entry point `jabs-pose = "jabs.vision.cli:main"`.

- [ ] **Step 0: Add dependencies + entry point**

```bash
uv add --package jabs-vision "click>=8.2.1" "h5py>=3.15.1"
```
Then manually add to `packages/jabs-vision/pyproject.toml` (after `[project.urls]`, before `[build-system]`):

```toml
[project.scripts]
jabs-pose = "jabs.vision.cli:main"
```
Run `uv lock` and `uv sync` to realize the script.
Expected: `pyproject.toml` has `click`, `h5py`, `imageio[ffmpeg]` in dependencies and the `jabs-pose` script; `uv.lock` updated.

- [ ] **Step 1: Write the failing test** — create `packages/jabs-vision/tests/cli/__init__.py` (empty) and `packages/jabs-vision/tests/cli/test_pose.py`:

```python
"""Tests for the jabs-pose CLI core (CPU, no real weights)."""

from unittest import mock

import h5py
import numpy as np
import torch

import jabs.vision.cli.pose as pose_cli
from jabs.vision.cli.pose import run_pose_inference


class _DummyPoseModel(torch.nn.Module):
    """Returns a fixed (B, 12, 4, 4) heatmap so argmax is deterministic."""

    def forward(self, x):  # noqa: D102
        batch = x.shape[0]
        hm = torch.zeros(batch, 12, 4, 4)
        # peak at (row=1, col=2) for every keypoint
        hm[:, :, 1, 2] = 1.0
        return hm


def test_run_pose_inference_writes_v2(monkeypatch, tmp_path):
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)]
    monkeypatch.setattr(pose_cli, "load_pose_model", mock.Mock(return_value=(_DummyPoseModel(), None)))
    monkeypatch.setattr(pose_cli, "read_frames", mock.Mock(return_value=iter(frames)))
    monkeypatch.setattr(pose_cli, "video_fps", mock.Mock(return_value=30))

    out = tmp_path / "clip_pose_est_v2.h5"
    run_pose_inference(
        video=tmp_path / "clip.mp4",
        out=out,
        config=tmp_path / "gait.yaml",
        checkpoint=tmp_path / "gait.pth",
    )

    with h5py.File(out, "r") as h5:
        assert h5["poseest"].attrs["version"].tolist() == [2, 0]
        assert h5["poseest/points"].shape == (3, 12, 2)
        assert h5["poseest/confidence"].shape == (3, 12)
        # dummy peak (row=1, col=2) -> stored on-disk as (y, x) = (1, 2)
        assert h5["poseest/points"][0, 0].tolist() == [1, 2]


def test_run_pose_inference_loads_strict(monkeypatch, tmp_path):
    loader = mock.Mock(return_value=(_DummyPoseModel(), None))
    monkeypatch.setattr(pose_cli, "load_pose_model", loader)
    monkeypatch.setattr(pose_cli, "read_frames", mock.Mock(return_value=iter([np.zeros((8, 8, 3), np.uint8)])))
    monkeypatch.setattr(pose_cli, "video_fps", mock.Mock(return_value=30))

    run_pose_inference(
        video=tmp_path / "c.mp4", out=tmp_path / "c_pose_est_v2.h5",
        config=tmp_path / "g.yaml", checkpoint=tmp_path / "g.pth",
    )
    # strict checkpoint loading is enforced (guards silent partial loads)
    assert loader.call_args.kwargs["strict"] is True


def test_run_pose_inference_empty_video_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(pose_cli, "load_pose_model", mock.Mock(return_value=(_DummyPoseModel(), None)))
    monkeypatch.setattr(pose_cli, "read_frames", mock.Mock(return_value=iter([])))
    monkeypatch.setattr(pose_cli, "video_fps", mock.Mock(return_value=30))

    import pytest
    with pytest.raises(ValueError, match="no frames"):
        run_pose_inference(
            video=tmp_path / "c.mp4", out=tmp_path / "c_pose_est_v2.h5",
            config=tmp_path / "g.yaml", checkpoint=tmp_path / "g.pth",
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/jabs-vision/tests/cli/test_pose.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'jabs.vision.cli'`.

- [ ] **Step 3: Create the CLI** — `packages/jabs-vision/src/jabs/vision/cli/pose.py`:

```python
"""``jabs-pose`` CLI: video -> single-mouse pose_est_v2.h5."""

import logging
from pathlib import Path

import click
import numpy as np
import numpy.typing as npt

from jabs.core.abstract.pose_est import PoseEstimation
from jabs.core.enums import JabsPoseVersion
from jabs.core.types import PoseData
from jabs.io import save
from jabs.vision.cli._logging import configure_logging
from jabs.vision.hrnet_msfork import load_pose_model, predict_single_pose
from jabs.vision.io import read_frames, video_fps

logger = logging.getLogger(__name__)

_BODY_PARTS = [kp.name.lower() for kp in PoseEstimation.KeypointIndex]


def _skeleton_edges() -> list[tuple[int, int]]:
    """Build keypoint edge pairs from the canonical connected segments."""
    edges: list[tuple[int, int]] = []
    for segment in PoseEstimation.FULL_CONNECTED_SEGMENTS:
        for a, b in zip(segment[:-1], segment[1:]):
            edges.append((int(a), int(b)))
    return edges


_SKELETON_EDGES = _skeleton_edges()


def _build_pose_data(
    pose_yx: npt.NDArray[np.uint16],
    confidence: npt.NDArray[np.float32],
    *,
    fps: int,
    config: Path,
    checkpoint: Path,
) -> PoseData:
    """Assemble a canonical PoseData from inference output.

    Args:
        pose_yx: Inference coordinates, shape (n_frames, 12, 2), order (y, x).
        confidence: Per-keypoint confidence, shape (n_frames, 12).
        fps: Video frame rate.
        config: HRNet config path (recorded for provenance).
        checkpoint: Checkpoint path (recorded for provenance).

    Returns:
        A single-identity PoseData with points in (x, y) order.
    """
    points_xy = np.flip(pose_yx.astype(np.float64), axis=-1)[np.newaxis, ...]
    conf = confidence.astype(np.float32)[np.newaxis, ...]
    n_frames = points_xy.shape[1]
    n_kp = points_xy.shape[2]
    return PoseData(
        points=points_xy,
        point_mask=np.ones((1, n_frames, n_kp), dtype=bool),
        identity_mask=np.ones((1, n_frames), dtype=bool),
        body_parts=_BODY_PARTS,
        edges=_SKELETON_EDGES,
        fps=fps,
        confidence=conf,
        metadata={"config": Path(config).name, "model": Path(checkpoint).name},
    )


def run_pose_inference(
    *,
    video: Path,
    out: Path,
    config: Path,
    checkpoint: Path,
    batch_size: int = 1,
    device: str | None = None,
) -> None:
    """Run pose inference on a video and write a pose_est_v2.h5.

    Args:
        video: Input video path.
        out: Output pose HDF5 path.
        config: HRNet YAML config path.
        checkpoint: HRNet checkpoint (.pth) path.
        batch_size: Frames per inference batch.
        device: Torch device ("cuda"/"cpu"), or None to auto-select.

    Raises:
        ValueError: If the video yields no frames.
        RuntimeError: If the checkpoint does not match the model (strict load).
    """
    logger.info("loading model config=%s checkpoint=%s", config, checkpoint)
    model, _ = load_pose_model(config, checkpoint, device=device, strict=True)

    result = predict_single_pose(
        read_frames(video), model, batch_size=batch_size, device=device
    )
    if result.pose.shape[0] == 0:
        raise ValueError(f"inference produced no frames for video {video}")

    pose_data = _build_pose_data(
        result.pose, result.confidence, fps=video_fps(video), config=config, checkpoint=checkpoint
    )
    save(pose_data, out, legacy=JabsPoseVersion.V2)
    logger.info("wrote %d frames to %s", result.pose.shape[0], out)


@click.command(name="pose", context_settings={"max_content_width": 120})
@click.option("--video", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out", required=True, type=click.Path(dir_okay=False, path_type=Path))
@click.option("--config", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--checkpoint", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option("--batch-size", default=1, show_default=True, type=int)
@click.option(
    "--device", default="auto", show_default=True, type=click.Choice(["auto", "cuda", "cpu"])
)
def pose_command(
    video: Path, out: Path, config: Path, checkpoint: Path, batch_size: int, device: str
) -> None:
    """Run single-mouse HRNet pose inference and write a pose_est_v2.h5."""
    configure_logging()
    resolved_device = None if device == "auto" else device
    run_pose_inference(
        video=video,
        out=out,
        config=config,
        checkpoint=checkpoint,
        batch_size=batch_size,
        device=resolved_device,
    )


def main() -> None:
    """Entry point for the ``jabs-pose`` console script."""
    pose_command()
```

- [ ] **Step 4: Create a tiny logging helper** — `packages/jabs-vision/src/jabs/vision/cli/_logging.py`:

```python
"""Logging configuration for jabs-vision CLIs."""

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging for CLI runs if not already configured.

    Args:
        level: Logging level to apply.
    """
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
```

- [ ] **Step 5: Create the package init** — `packages/jabs-vision/src/jabs/vision/cli/__init__.py`:

```python
"""jabs-vision command line interface."""

from jabs.vision.cli.pose import main, pose_command, run_pose_inference

__all__ = ["main", "pose_command", "run_pose_inference"]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest packages/jabs-vision/tests/cli/test_pose.py -v`
Expected: PASS (3 passed).

- [ ] **Step 7: Verify the entry point is installed**

Run: `uv run jabs-pose --help`
Expected: Click help text listing `--video`, `--out`, `--config`, `--checkpoint`, `--batch-size`, `--device`.

- [ ] **Step 8: Lint + commit**

```bash
uv run ruff check --fix packages/jabs-vision && uv run ruff format packages/jabs-vision
git add packages/jabs-vision/src/jabs/vision/cli/ packages/jabs-vision/tests/cli/ \
        packages/jabs-vision/pyproject.toml uv.lock
git commit -m "feat(jabs-vision): add jabs-pose CLI (video -> pose_est_v2.h5)"
```

---

### Task 6: Round-trip integration test (root)

**Files:**
- Create: `tests/io/__init__.py` (if missing), `tests/io/test_pose_v2_roundtrip.py`

**Interfaces:**
- Consumes: `jabs.io.save`, `PoseData`, `JabsPoseVersion` (Task 3); `jabs.pose_estimation.PoseEstimationV2` (monolith reader that flips `(y,x)->(x,y)` and thresholds confidence at `MINIMUM_CONFIDENCE=0.3`).

- [ ] **Step 1: Write the failing test** — create `tests/io/__init__.py` (empty if not present) and `tests/io/test_pose_v2_roundtrip.py`:

```python
"""End-to-end axis/format contract: write via jabs-io, read via the monolith v2 reader."""

import numpy as np

from jabs.core.enums import JabsPoseVersion
from jabs.core.types import PoseData
from jabs.io import save
from jabs.pose_estimation.pose_est_v2 import PoseEstimationV2


def _make_pose(num_frames=4):
    n_kp = 12
    points = np.zeros((1, num_frames, n_kp, 2), dtype=np.float64)
    for f in range(num_frames):
        for k in range(n_kp):
            points[0, f, k] = [f * 10 + k, f * 10 + k + 3]  # distinct (x, y)
    return PoseData(
        points=points,
        point_mask=np.ones((1, num_frames, n_kp), dtype=bool),
        identity_mask=np.ones((1, num_frames), dtype=bool),
        body_parts=[f"kp{i}" for i in range(n_kp)],
        edges=[],
        fps=30,
        confidence=np.full((1, num_frames, n_kp), 0.9, dtype=np.float32),
    )


def test_v2_write_read_roundtrip_preserves_xy(tmp_path):
    pose = _make_pose()
    out = tmp_path / "clip_pose_est_v2.h5"
    save(pose, out, legacy=JabsPoseVersion.V2)

    reader = PoseEstimationV2(out, fps=30)
    points, point_mask = reader.get_identity_poses(0)

    # Monolith reads back (x, y) after its (y,x)->(x,y) flip; the double flip is identity.
    np.testing.assert_array_equal(points, pose.points[0])
    # confidence 0.9 > MINIMUM_CONFIDENCE (0.3) => all keypoints marked valid.
    assert point_mask.all()
    assert reader.format_major_version == 2


def test_v2_low_confidence_masked_out(tmp_path):
    pose = _make_pose(num_frames=2)
    # frozen dataclass: replace confidence with a below-threshold array
    object.__setattr__(pose, "confidence", np.full((1, 2, 12), 0.1, dtype=np.float32))
    out = tmp_path / "clip_pose_est_v2.h5"
    save(pose, out, legacy=JabsPoseVersion.V2)

    reader = PoseEstimationV2(out, fps=30)
    _, point_mask = reader.get_identity_poses(0)
    assert not point_mask.any()  # 0.1 < 0.3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/io/test_pose_v2_roundtrip.py -v`
Expected: FAIL initially only if run before Tasks 2-3 are merged; after Tasks 1-5 it should be run to confirm. If PoseEstimationV2 import fails, ensure the root project is synced (`uv sync`).

- [ ] **Step 3: No implementation needed** — this test exercises code from Tasks 1-3. If it fails, the failure is a real contract bug (axis flip or dtype); fix in the Task 3 adapter, not here.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/io/test_pose_v2_roundtrip.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add tests/io/__init__.py tests/io/test_pose_v2_roundtrip.py
git commit -m "test: v2 pose write/read round-trip contract (jabs-io <-> monolith)"
```

---

### Task 7: GPU container image + CI build job

**Files:**
- Create: `containers/docker/Dockerfile.pose`
- Create: `.github/workflows/build-pose-image.yml`

**Interfaces:**
- Produces: a buildable image whose `ENTRYPOINT` is `jabs-pose`; a CI job that builds it (no push).

- [ ] **Step 1: Write the Dockerfile** — `containers/docker/Dockerfile.pose` (follows mtr's slim-Python + CUDA-via-PyTorch-wheel pattern; no full `nvidia/cuda` base, no TensorFlow):

```dockerfile
# syntax=docker/dockerfile:1
# jabs-pose GPU image. CUDA runtime is bundled by the PyTorch wheels
# (installed from the CUDA wheel index), following mouse-tracking-runtime's pattern.
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    UV_TORCH_BACKEND=auto

# ffmpeg for imageio video decoding
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Workspace metadata + the packages the pose CLI needs.
COPY pyproject.toml uv.lock README.md ./
COPY packages/jabs-core ./packages/jabs-core
COPY packages/jabs-io ./packages/jabs-io
COPY packages/jabs-vision ./packages/jabs-vision

# Install jabs-vision (with the HRNet extra) and its workspace deps, incl. h5py.
RUN uv pip install --system \
      "./packages/jabs-core" \
      "./packages/jabs-io[h5py]" \
      "./packages/jabs-vision[hrnet_msfork]"

ENTRYPOINT ["jabs-pose"]
CMD ["--help"]
```

- [ ] **Step 2: Validate the Dockerfile builds (if Docker is available)**

Run: `docker build -f containers/docker/Dockerfile.pose -t jabs-pose:dev .`
Expected: build succeeds; `docker run --rm jabs-pose:dev --help` prints the CLI help.
(If Docker/GPU is unavailable in this environment, skip execution — the build is exercised by CI in Step 3 and the real GPU run in Task 8.)

- [ ] **Step 3: Write the CI build job** — `.github/workflows/build-pose-image.yml`:

```yaml
name: Build pose image

on:
  pull_request:
    paths:
      - "packages/jabs-vision/**"
      - "packages/jabs-io/**"
      - "packages/jabs-core/**"
      - "containers/docker/Dockerfile.pose"
      - ".github/workflows/build-pose-image.yml"
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      - name: Build pose image (no push)
        uses: docker/build-push-action@v5
        with:
          context: .
          file: containers/docker/Dockerfile.pose
          push: false
          tags: jabs-pose:ci
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

- [ ] **Step 4: Commit**

```bash
git add containers/docker/Dockerfile.pose .github/workflows/build-pose-image.yml
git commit -m "build(jabs-vision): add jabs-pose GPU Dockerfile + CI build job"
```

---

### Task 8: Spike + dev-Batch acceptance (manual gate — real weights + GPU)

**Not executable inline** (needs the gait-model weights + a GPU on dev-Batch). Run when infra + weights are available. This validates the two real-weight risks and the AC.

- [ ] **Step 1: Confirm the checkpoint loads strict** — on a GPU host with the real config/checkpoint:

```bash
uv run python -c "
from jabs.vision.hrnet_msfork import load_pose_model
m, cfg = load_pose_model('gait-model.yaml', 'gait-model.pth', device='cuda', strict=True)
print('HEAD_ARCH', cfg.MODEL.EXTRA.get('HEAD_ARCH'))
print('loaded OK')
"
```
Expected: prints a `HEAD_ARCH` the fork recognizes (a `CONV_TRANS_UPSCALE_*` variant) and `loaded OK` with **no** missing/unexpected-key error. If it raises, the wrapper/model-fork compatibility must be resolved before relying on real output.

- [ ] **Step 2: Run the CLI end-to-end on a reference video (GPU)**

```bash
uv run jabs-pose --video reference.mp4 --out reference_pose_est_v2.h5 \
  --config gait-model.yaml --checkpoint gait-model.pth --device cuda
```
Expected: exits 0; `reference_pose_est_v2.h5` exists with `poseest/points` (uint16) + `poseest/confidence` (float32) + `version=[2,0]`.

- [ ] **Step 3: Human overlay inspection** — render keypoints on the video and eyeball tracking:

```bash
uv run python -c "
import numpy as np
from jabs.vision.io import read_frames
from jabs.vision.hrnet_msfork import load_pose_model, predict_single_pose
from jabs.vision.hrnet_msfork.render import ...  # use existing render_pose_fn hook
"
```
Use `predict_single_pose(..., render='overlay.mp4', render_pose_fn=<fn>)` and confirm keypoints track the mouse (catches scale/axis errors). Document the result on KLAUS-533.

- [ ] **Step 4: Build + run the image on dev-Batch** — build `Dockerfile.pose`, run the CLI in-container on the dev-Batch GPU infra (KLAUS-519). Confirm a valid `pose_est_v2.h5` is produced. Record the image digest for the follow-up AR-publish ticket.

---

## Self-Review

**Spec coverage:**
- CLI (video → v2) — Task 5. ✅
- `PoseHDF5Adapter` + `legacy=` selector via `save` — Task 3. ✅
- `JabsPoseVersion` enum — Task 1. ✅
- First-class `PoseData.confidence` — Task 2. ✅
- imageio[ffmpeg] frame reader — Task 4. ✅
- Strict checkpoint loading — Task 5 (`strict=True`) + Task 8 (real-weight validation). ✅
- Structural validity + axis contract test — Task 6. ✅
- Human overlay inspection — Task 8. ✅
- GPU Dockerfile (mtr pattern) + build-only CI — Task 7. ✅
- AR publish deferred — noted (Task 8 records digest for the follow-up ticket). ✅
- GPU-free module tests (IMPL-7) — Tasks 1-6 all CPU. ✅

**Placeholder scan:** Task 8 Step 3 uses `...` intentionally — it is a manual, weights-dependent inspection whose exact render wiring depends on the reference video; every inline-executable step (Tasks 1-7) has complete code. No TBD/TODO in executable steps.

**Type consistency:** `run_pose_inference` signature matches its call in `pose_command`; `read_frames`/`video_fps` names match Task 4; `save(pose_data, out, legacy=JabsPoseVersion.V2)` matches the Task 3 adapter `write` signature; `PoseData(..., confidence=...)` matches the Task 2 field.
```
