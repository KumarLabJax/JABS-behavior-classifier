# KLAUS-533 — Pose-Estimation CLI + Image (Increment 1)

| Field | Value |
|-------|-------|
| **Jira** | [KLAUS-533](https://jacksonlaboratory.atlassian.net/browse/KLAUS-533) (Story) |
| **Epic** | [KLAUS-531](https://jacksonlaboratory.atlassian.net/browse/KLAUS-531) — ADR-0011 Pipeline Rewrite |
| **ADR** | `jabs-hub/docs/adr/0011-pipeline-rewrite-approach.md` (increment 1) |
| **Repo** | `JABS-behavior-classifier` (`v0.46.1`) |
| **Date** | 2026-07-20 |
| **Status** | Draft — pending user review |

## 1. Summary

Increment 1 of the ADR-0011 pipeline rewrite. `jabs-vision` today has HRNet
single-mouse pose inference but is **library-only**: no CLI, no video reader, and no
pose-file writer. This ticket wires that inference into a **CLI** that reads a video and
writes a valid `*_pose_est_v2.h5`, adds a **forward-looking pose writer** to `jabs-io`,
and ships a self-contained **GPU container image** (built here; published to Artifact
Registry by a separate ticket).

The pipeline pose module (jabs-hub increment 1, "P1") is **blocked by** this ticket and
will invoke the CLI as a digest-pinned image. mtr's `infer single_pose` +
`utils/writers.py` are a **read-only porting reference** — no runtime or code dependency
on mtr.

## 2. Scope

### In scope
- A `jabs-vision` CLI (`jabs-pose`) — video → `*_pose_est_v2.h5`.
- An `imageio[ffmpeg]`-based frame reader in `jabs-vision`.
- A **`PoseHDF5Adapter`** in `jabs-io` writing the legacy v2 layout, plus a
  `JabsPoseVersion` enum and a first-class `confidence` field on `PoseData` in `jabs-core`.
- A self-contained GPU **Dockerfile** for the pose image, built in CI.
- GPU-free module tests (IMPL-7).

### Out of scope (explicit)
- **Artifact Registry publish workflow + CD identity** — a separate ticket (this repo has
  no AR pipeline today; only an orphaned Docker Hub reusable workflow + PyPI wheels). This
  ticket produces a buildable image and a CI build job only.
- **Numeric output-equivalence vs. v0.2.5** — deferred to IMPL-9 / ADR-0011 OQ5. Our bar is
  structural validity + human overlay inspection (ADR-0011 per-increment gate).
- Model weights — remain **external**, supplied at runtime (config-as-data). No weights are
  baked into the image or committed.
- pose v3–v8 writing, segmentation, matching/identity — later increments.

## 3. Context & verified findings

### 3.1 jabs-vision inference (the thing we wrap)
- Entry point to reuse **unchanged**: `predict_single_pose_from_model_files(input_iter, *,
  config_path, checkpoint_path=None, batch_size=1, device=None)` —
  `packages/jabs-vision/src/jabs/vision/hrnet_msfork/model_loader.py:88`. It loads the model
  (YACS config + `.pt` checkpoint) then calls `predict_single_pose`.
- `predict_single_pose` (`hrnet_msfork/single_pose.py:110`) consumes an **iterator of
  `(H,W,3)` uint8 frames** (not a path), returns `SinglePoseInferenceResult` with
  `.pose (N,K,2) uint16` in **(row=y, col=x)** and `.confidence (N,K) float32`.
- Preprocessing (`preprocess.py`) and decode (`decode.py`) are **byte-for-byte faithful
  ports of mtr's `utils/hrnet.py`**: `/255`, HWC→CHW, mean `0.45`/std `0.225`, **no
  resize**; raw `argmax_2d_torch` → `(row,col)`, no sub-pixel. Verified by inspection.
- The fork's `pose_hrnet.py` implements the `CONV_TRANS_UPSCALE_*` heads the gait-model uses
  (`pose_hrnet.py:360,416`) — the head that upsamples the heatmap to input resolution, so
  argmax coordinates are already in **pixel space** (no scaling needed). Verified.
- No CLI/`__main__`; no `[project.scripts]`; `jabs-vision` is **not** a root-app dependency
  (torch is deliberately kept out of the GUI app). Deps: `torch>=2.9.1`, `torchvision`,
  `jabs-io`, `jabs-core`; `hrnet_msfork` extra pulls `yacs`.

### 3.2 mtr reference (read-only)
- `infer single_pose` → `infer_single_pose_pytorch` reads the video with
  **`imageio.get_reader(...).iter_data()`** (HWC RGB uint8), runs HRNet, writes via
  `write_pose_v2_data`.
- v2 on-disk layout (to reproduce): group `poseest` with attr
  `version = np.asarray([2,0], uint16)`; dataset `poseest/points` uint16 `[N,12,2]` in
  **(y,x)** with string attrs `config` and `model`; dataset `poseest/confidence` float32
  `[N,12]`. 12 keypoints, order `NOSE=0 … TIP_TAIL=11`. No confidence threshold applied at
  v2 (thresholding is a later v2→v3 concern @ 0.3).
- GPU is mandatory in the mtr path; weights come from an external model dir.

### 3.3 jabs-io conventions (where the writer lives)
- Public surface is **generic `load(path, data_type)` / `save(data, path, **kwargs)`**
  (`packages/jabs-io/src/jabs/io/api.py`), dispatched by extension → `StorageFormat` and by
  domain type through an **adapter registry** (`jabs/io/registry.py`,
  `@register_adapter(format, domain_type, priority)`). Adapters implement `read`/`write`;
  `HDF5Adapter` (`jabs/io/base.py`) guards the optional `h5py` import in `__init__`.
- Closest templates: `PredictionHDF5Adapter` (`internal/prediction/hdf5.py`) — embeds an
  integer file-version constant, overrides `write` directly, opens `h5py.File(path,"a")`,
  uses `_write_dataset`/`_write_string_dataset`/`_write_optional_attr` helpers; and
  `PoseNWBAdapter` (`internal/pose/nwb.py`) — the only existing `PoseData` adapter,
  **branches `write` on a kwarg** (`multisubject=`), the exact precedent for a `legacy=`
  selector.
- Canonical model `PoseData` (`packages/jabs-core/src/jabs/core/types/pose.py:34`), a frozen
  dataclass: `points (num_identities, num_frames, num_keypoints, 2)`, boolean `point_mask`,
  `identity_mask`, `body_parts`, `edges`, `fps`, `cm_per_pixel`, … , `metadata: dict[str,
  Any]`. **No confidence field; no format-version field.** Canonical `points` are **(x,y)**
  (the monolith `PoseEstimationV2` flips on-disk (y,x)→(x,y) at `pose_est_v2.py:38`).
- Keypoint enum to reuse: `PoseEstimation.KeypointIndex` (`IntEnum`, 12 members) in
  `packages/jabs-core/src/jabs/core/abstract/pose_est.py:58`.
- No pose-version enum exists anywhere; version is a filename-derived `int` (majors 2–8).
- `h5py` is an **optional extra** (`jabs-io[h5py]`); numpy + jabs-core are hard deps;
  models are stdlib dataclasses (no pydantic). Adapters raise builtin `ValueError` /
  `ImportError`.

## 4. Component architecture

```
video.mp4
   │  jabs.vision.io.read_frames()  (imageio[ffmpeg]) → Iterator[(H,W,3) uint8, RGB]
   ▼
jabs.vision.hrnet_msfork.predict_single_pose_from_model_files(iter, config, checkpoint)
   │  → pose (N,12,2) uint16 (y,x),  confidence (N,12) float32
   ▼
jabs.vision.cli  (Click)  builds a PoseData  (points→(x,y), confidence as first-class field)
   ▼
jabs.io.save(pose_data, out_path, legacy=JabsPoseVersion.V2)   [PoseHDF5Adapter]
   ▼
out_pose_est_v2.h5   (poseest/points uint16 (y,x) + attrs, poseest/confidence f32, version [2,0])
```

### 4.1 New/changed locations
| Package | Path | What |
|---------|------|------|
| jabs-core | `src/jabs/core/enums/pose_version.py` (+ `enums/__init__.py`) | `JabsPoseVersion` enum (V2..V8, int-aligned) |
| jabs-core | `src/jabs/core/types/pose.py` | add first-class `confidence` field to `PoseData` (see §5.4) |
| jabs-io | `src/jabs/io/internal/pose/hdf5.py` (+ `pose/__init__.py`) | `PoseHDF5Adapter` (writes v2 layout) |
| jabs-vision | `src/jabs/vision/io/frames.py` | `read_frames(video) -> Iterator` (imageio) |
| jabs-vision | `src/jabs/vision/cli/__init__.py`, `cli/pose.py` | Click CLI + `main()` |
| jabs-vision | `pyproject.toml` | `[project.scripts] jabs-pose=…`; deps `click`, `imageio[ffmpeg]`, `jabs-io[h5py]` |
| repo | `containers/docker/Dockerfile.pose` | GPU image for the pose CLI |
| repo | `.github/workflows/…` | CI build job for the pose image (build only) |

`jabs-vision` becomes the **first sub-package to declare its own `[project.scripts]`** — the
entry point installs only in the jabs-vision image, keeping torch out of the root app.

## 5. Detailed design

### 5.1 `JabsPoseVersion` enum (jabs-core)
`IntEnum` aligned to the existing integer majors so numeric comparisons keep working:

```python
class JabsPoseVersion(IntEnum):
    """Legacy JABS pose-file format major versions (`pose_est_vN`)."""
    V2 = 2
    V3 = 3
    # … V4–V8 reserved for later increments
```
Placed in `jabs.core.enums` beside `StorageFormat` and re-exported. Its docstring notes the
`pose_est_vN` convention is **planned for deprecation** in favor of a future
backwards-compatible format; `legacy=` is the selector that will name the old formats once
the new default exists.

### 5.2 `PoseHDF5Adapter` (jabs-io)
- Registered `@register_adapter(StorageFormat.HDF5, PoseData, priority=10)`, mirroring
  `PoseNWBAdapter`. Coexists with `PredictionHDF5Adapter` (also `.h5`) because dispatch is by
  `(format, domain_type)` and `PoseData != BehaviorPrediction`.
- Subclasses `HDF5Adapter`; guards `h5py` in `__init__` with the install-hint idiom.
- **`write(self, data: PoseData, path, *, legacy: JabsPoseVersion = JabsPoseVersion.V2,
  **kwargs)`** branches on `legacy`. **V2 is the only supported value in this ticket**; any
  other raises `ValueError` (documented `Raises:`). This is the forward-looking seam: when
  the backwards-compatible format lands it becomes the default and `legacy=` selects the old
  layouts.
- `_write_v2(data, h5)` reproduces the mtr layout exactly:
  - single identity only (`data.points.shape[0] == 1`, else `ValueError`);
  - `poseest/points` = `points[0]` converted **(x,y)→(y,x)** as `uint16`, with string attrs
    `config`/`model` sourced from `data.metadata` (provenance: config + checkpoint identity);
  - `poseest/confidence` = `data.confidence[0]` as float32 `[N,12]` (see §5.4);
  - `poseest.attrs["version"] = np.asarray([2,0], np.uint16)`.
- `read` is **not implemented in this ticket** (raises `NotImplementedError` with a pointer
  to the monolith `PoseEstimationV2` reader) — writing is all increment 1 needs, and adding a
  `PoseData` HDF5 reader is a larger, separable effort. `can_handle` returns
  `data_type is PoseData`.

### 5.3 Public surface
The **only** surface is the idiomatic generic `jabs.io.save(pose_data, "x_pose_est_v2.h5",
legacy=JabsPoseVersion.V2)` — kwargs forward to the adapter, exactly as `save` already
forwards `multisubject=` to the NWB adapter. No `write_pose` convenience wrapper is added:
per review, we defer named per-domain functions until a concrete use-case justifies breaking
from the established generic `load`/`save` convention.

### 5.4 Confidence — first-class field on `PoseData` (resolved)
`PoseData` has no confidence field today (validity is a boolean `point_mask`), but v2 stores
real float `poseest/confidence` and downstream v2→v3 promotion thresholds it. **Resolved
(review decision): add a first-class field to the canonical model** — the cleaner, more
discoverable representation:

```python
confidence: npt.NDArray[np.float32] | None = None
    # shape (num_identities, num_frames, num_keypoints); None when unknown
```

- Added to `PoseData` in `packages/jabs-core/src/jabs/core/types/pose.py` with `__post_init__`
  validation: when present, shape must equal `point_mask`'s shape.
- **Backward compatible:** defaults to `None`, so every existing constructor call and the NWB
  read path (which does not set it) keep working; the NWB adapter is unaffected (it continues
  to synthesize/consume via `point_mask`). Its equality/`__post_init__` handling must treat
  the array field like the other optional ndarray fields.
- The CLI populates `confidence` directly from inference output; `point_mask` is set all-True
  (raw inference emits a coordinate for every keypoint — masking/thresholding is a later
  concern). `PoseHDF5Adapter._write_v2` writes `poseest/confidence` from `data.confidence`,
  and raises `ValueError` if it is `None` (v2 requires confidence).

### 5.5 CLI contract (locked — P1 calls this)
```
jabs-pose --video <in.mp4> --out <in_pose_est_v2.h5>
          --config <hrnet.yaml> --checkpoint <gait-model.pth>
          [--batch-size N=1] [--device auto|cuda|cpu] [--out-video <overlay.mp4>]
```
- Click (matches `jabs-cli`; Typer is mtr-legacy). `main()` is the `[project.scripts]` target.
- `--config`/`--checkpoint` are explicit paths (config-as-data; no hardcoded model dir).
  They may be local paths resolved from staged (e.g. `gs://`) inputs by the pipeline.
- `--device auto` → cuda if available else cpu (matches jabs-vision's resolver); real runs
  use cuda, CPU exists for tests.
- Writes provenance attrs (`config`, `model`) derived from the config/checkpoint identity.
- `--out-video` uses jabs-vision's existing `render`/`render_pose_fn` path for the human
  sanity overlay.

### 5.6 Frame reader (jabs-vision)
`read_frames(video: Path) -> Iterator[np.ndarray]` over `imageio.get_reader(video,
"ffmpeg").iter_data()`, yielding HWC **RGB** uint8 — the exact iterator
`predict_single_pose` expects. **imageio[ffmpeg] is chosen deliberately** to match mtr's
decode path (a different decoder, e.g. the monolith's OpenCV `VideoCapture` which yields
BGR, could shift pixels/axes and muddy the future IMPL-9 equivalence effort). `fps` for
`PoseData` comes from the reader metadata, default 30.

## 6. Error handling

- **Strict checkpoint loading (correctness-critical).** The wrapped loader uses
  `strict=False`, which silently tolerates a partially-loaded checkpoint → structurally
  valid but garbage output. The CLI loads with **`strict=True`** (or asserts zero
  missing/unexpected keys) and **fails loudly** on mismatch. Core of the §8.1 spike.
- **Unknown `HEAD_ARCH`** in the config → the fork already raises; surface as a clear CLI
  error.
- Missing/unreadable video, config, or checkpoint → clear message + non-zero exit.
- **Empty / zero-frame video** → explicit failure, **not** a silent zero-row h5.
- Multi-identity `PoseData` into the v2 writer → `ValueError` (v2 is single-mouse).
- Adapter follows jabs-io convention: builtin `ValueError` for bad data, `ImportError` (with
  install hint) for missing `h5py`.

## 7. Container

- New `containers/docker/Dockerfile.pose`, following **mtr's proven pose-image pattern**
  (`mouse-tracking-runtime/Dockerfile` + its `vm/tf-pytoch/Dockerfile` base): a **slim
  Python base** with the **CUDA runtime bundled via the PyTorch wheel index** (mtr uses
  `--index-url https://download.pytorch.org/whl/cu126`) — **not** a full `nvidia/cuda` base
  image — plus apt `ffmpeg`, and `NVIDIA_VISIBLE_DEVICES=all` /
  `NVIDIA_DRIVER_CAPABILITIES=compute,utility`. jabs-vision needs no TensorFlow layer (mtr's
  base carries one; we drop it).
- Built with `uv` like the active repo Dockerfile, but installs `packages/jabs-vision` with
  the `hrnet_msfork` extra + `imageio[ffmpeg]` + `jabs-io[h5py]` and the workspace deps
  (`jabs-io`, `jabs-core`). `ENTRYPOINT ["jabs-pose"]`.
- Distinct from the active `containers/docker/Dockerfile` (which installs only the root
  `src/` and no workspace packages).
- No weights baked in (config-as-data). The exact CUDA wheel tag matching torch≥2.9.1 and the
  dev-Batch GPU driver is a spike checkpoint (§8.1); mtr's `cu126` is the starting reference.
- CI: a **build-only** job (validates the image builds). Push to Artifact Registry by digest
  is the separate CD ticket.

## 8. Testing (GPU-free — IMPL-7)

### 8.1 Spike (do first; gates the rest) — needs real weights on GPU (dev-Batch)
1. The gait-model `HEAD_ARCH` string is one the fork recognizes.
2. The 2019 gait checkpoint loads into the fork with **zero missing/unexpected keys**
   (strict).
3. Overlay render on a reference video — keypoints track the mouse (catches scale/axis
   errors a bare "h5 opens" check misses).
4. torch≥2.9.1 + CUDA run cleanly on the dev-Batch GPU.

If (1)/(2) fail, the "wrap `predict_single_pose_from_model_files`" approach needs
rework before further build — hence spike-first.

### 8.2 Automated unit tests (CPU, no weights)
- `jabs-io`: `PoseHDF5Adapter` — write a `PoseData` → reopen with `h5py`, assert
  `poseest/points` dtype uint16 / shape `[N,12,2]` / (y,x) order, `poseest/confidence`
  float32, `poseest.attrs["version"] == [2,0]`, `config`/`model` attrs. Registry integration
  (`get_adapter(HDF5, PoseData)` resolves it). `can_handle` truth table. `legacy != V2` and
  multi-identity raise. Follows `tests/internal/prediction/test_hdf5.py` patterns
  (`tmp_path`, adapter fixture, `_make_pose(...)` builder).
- `jabs-vision`: frame reader (mock imageio) yields correct shape/dtype/RGB; CLI wiring with
  the existing `DummyPoseModel` pattern (CPU) → produces a file; `PoseData` construction
  (axis flip (y,x)→(x,y), confidence as first-class field) is correct.

### 8.3 Round-trip integration test (CPU) — validates the axis contract end-to-end
Write a v2 file from dummy inference output via the adapter, **read it back with the
monolith `PoseEstimationV2`** (`src/jabs/pose_estimation/pose_est_v2.py`, which flips
(y,x)→(x,y)); assert coordinates and confidences survive the double flip. This proves both
structural validity and the axis convention without a GPU.

### 8.4 Manual acceptance (dev-Batch, real weights)
Run the published image on a GPU: `jabs-pose --video X --out X_pose_est_v2.h5 --config …
--checkpoint …` produces a valid `pose_est_v2.h5`; overlay eyeballed. Satisfies the AC and
the ADR-0011 per-increment human-inspection gate.

## 9. Task breakdown (spike-first)

1. **Spike** (§8.1) — gate before building on the wrapper.
2. **jabs-core:** `JabsPoseVersion` enum + export; first-class `confidence` field on
   `PoseData` (+ `__post_init__` validation, equality handling) + tests.
3. **jabs-io:** `PoseHDF5Adapter` (`_write_v2`) + registration (exposed via generic `save`) +
   tests (§8.2).
4. **jabs-vision:** `read_frames` + tests.
5. **jabs-vision:** Click CLI (`jabs-pose`), `PoseData` assembly, `[project.scripts]`, deps +
   tests.
6. **Round-trip integration test** (§8.3).
7. **`Dockerfile.pose`** + CI build-only job.
8. **Manual dev-Batch acceptance** (§8.4).

All package versions stay in lockstep at the root version (CI enforces
`dev/sync-versions.sh`).

## 10. Acceptance criteria (from KLAUS-533)

- [ ] `jabs-pose --video X --out X_pose_est_v2.h5 --config … --checkpoint …` runs in the
  published image on a GPU (dev-Batch).
- [ ] Output is a valid `pose_est_v2.h5` (structure + human overlay inspection).
- [ ] Image is buildable and *pinnable* by digest (AR push is the separate ticket).
- [ ] Module-level tests run GPU-free via stub/fixtures.

## 11. Decisions & remaining unknowns

- **Confidence representation (§5.4) — RESOLVED:** first-class `confidence` field on
  `PoseData`.
- **Public surface (§5.3) — RESOLVED:** generic `save` only; no `write_pose` wrapper.
- **CUDA base image / torch-CUDA pin (§7) — resolved in the §8.1 spike** against the
  dev-Batch GPU; mtr's slim-Python + PyTorch-`cu126`-wheel pattern is the starting reference.
```
