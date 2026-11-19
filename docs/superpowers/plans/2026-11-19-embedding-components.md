# Embedding Components Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a unified pose file carry named per-instance and whole-frame embeddings
(`jabs.embedding.<name>.instance|frame`) with a validated descriptor, float16 storage and a
producer-selectable layout, and record the decision as ADR 0003.

**Architecture:** A new `embeddings.py` module in `jabs.io.internal.pose_file` owns the descriptor
type, id parsing, canonical JSON and fingerprinting. `Component` gains two optional fields
(`embedding`, `requested_layout`); the manifest builder, reader, JSON Schema, writer layout policy
and validator learn about them. Conformance fixtures and ADR 0003 close it out.

**Tech Stack:** Python 3.10+, numpy, h5py, jsonschema, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-11-19-embedding-components-design.md` (read it first; this
plan argues from it).

All paths below are relative to the repo root of the worktree
`/Users/aberger/Projects/KumarLab/jabs-pose-file-embeddings` (branch `feat/pose-file-embeddings`,
based on PR #596). `PF` = `packages/jabs-io/src/jabs/io/internal/pose_file`,
`TPF` = `packages/jabs-io/tests/internal/pose_file`,
`GEN` = `packages/jabs-io/tests/data/pose-format/generate.py`.

## Global Constraints

- Python target 3.10; ruff line length 99; Google-style docstrings on every module, class and
  public function; full type hints (`npt.NDArray[...]`, `X | None`); no EN DASH in docstrings.
- `logging.getLogger(__name__)`, lazy `%s` formatting; never `print()` in library code (the fixture
  generator's existing `print` stays).
- Tests are plain pytest functions; mocking via `monkeypatch`/`unittest.mock` only.
- Run everything with `uv run`; never edit `uv.lock`.
- `schema_revision` stays **1**.
- Component id shape: exactly `jabs.embedding.<name>.(instance|frame)`, `<name>` matching
  `[a-z0-9]([a-z0-9_-]*[a-z0-9])?`.
- Axes: instance `("frame", "slot", "feature")`, frame `("frame", "feature")`. `feature` is never a
  `dimensions` key.
- Embedding dtype ∈ {`float16`, `float32`}; missing policy `{"policy": "nan"}`.
- Fingerprint = `"sha256:" + sha256(canonical_json(descriptor minus name, kind, fingerprint)).hexdigest()`,
  canonical JSON = `json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")`;
  descriptor contains no floats.
- `model.weights` and `fingerprint` match `^sha256:[0-9a-f]{64}$`.
- Embedding default layout: chunked along frame, other axes whole, `compression: "none"`, frame
  chunk = `max(1, min(F, 1_048_576 // bytes_per_frame))`.
- Requested layouts honored only for `jabs.embedding.*` and non-`jabs` ids; allowed compression
  `none | gzip (opts 0-9) | lzf (opts None)`; contiguous + compression is an error; chunk extents
  clamped to `max(1, axis_length)`.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Streaming producers** — an embedding `Component` whose `data` is a `BlockSource` (how SPAL
   will write) must construct, write, and validate exactly like an in-memory one. Test in Task 3.
2. **Short and empty files** — F = 0 and F smaller than the default/requested frame chunk must
   write (contiguous for zero-size, clamped chunk otherwise) and the manifest must record what is
   on disk. Test in Task 4.
3. **Very wide vectors** — when one frame's row exceeds 1 MiB (e.g. S=8, D=65536 float32) the
   default frame chunk is 1, not 0. Test in Task 4.
4. **float16 NaN** — the deep checks must detect NaN in float16 payloads (no silent upcast bugs)
   and read in bounded frame blocks. Test in Task 6.
5. **Round trip of a mixed file** — read-then-write of a file with pose, an embedding with a
   requested gzip layout and a foreign component keeps the embedding's and foreign layouts and the
   descriptor, while `jabs.pose.*` stays contiguous. Test in Task 4.

---

### Task 0: Baseline

- [ ] **Step 1:** `cd` to the worktree root and run `uv sync --all-packages --all-extras --all-groups`.
- [ ] **Step 2:** Run `uv run pytest packages/jabs-io/tests -q`. Expected: all pass (record the
  count). If anything fails before you change code, stop and report it.

---

### Task 1: float16 dtype

**Files:**
- Modify: `PF/types.py` (`DTYPE_NAMES`; the `nan`-policy check in `Component.__post_init__`)
- Modify: `PF/schemas/manifest-1.json` (`$defs/dtype` enum; the `nan` conditional's dtype enum)
- Modify: `TPF/test_conformance.py` (`SCHEMA_MESSAGES["invalid-nan-policy.h5"]`)
- Test: `TPF/test_types.py`, `TPF/test_writer.py`

**Interfaces:**
- Produces: `"float16"` is a legal manifest dtype everywhere; `nan` policy allowed on
  float16/float32/float64.

- [ ] **Step 1: Failing tests**
  - `test_types.py::test_float16_component_accepts_nan_policy` — `Component(id="org.example.lab.x", axes=("frame",), data=np.zeros(4, np.float16), missing={"policy": "nan"})` constructs; `.dtype == "float16"`.
  - `test_writer.py::test_float16_round_trip` — add that component to `sample_pose_file` (via `dataclasses.replace(pf, components=pf.components + (c,))`), write, `validate()` has no errors, `read_component(...)` returns dtype `float16` with equal values (including a NaN at index 1, compared with `np.testing.assert_array_equal`, which treats NaN==NaN).
- [ ] **Step 2:** `uv run pytest TPF/test_types.py TPF/test_writer.py -q -k float16` → FAIL.
- [ ] **Step 3:** Add `"float16"` to `DTYPE_NAMES`, to the schema dtype enum, and to both nan
  checks (code tuple → `("float16", "float32", "float64")`, schema conditional enum likewise).
  Update the `SCHEMA_MESSAGES` entry to `"dtype: 'bool' is not one of ['float16', 'float32', 'float64']"`.
- [ ] **Step 4:** `uv run pytest packages/jabs-io/tests -q` → PASS.
- [ ] **Step 5:** Commit `feat(pose-file): admit float16 components`.

---

### Task 2: Embedding descriptor module

**Files:**
- Create: `PF/embeddings.py`
- Test: `TPF/test_embeddings.py`

**Interfaces:**
- Produces (all in `jabs.io.internal.pose_file.embeddings`):
  - `EMBEDDING_PREFIX: str = "jabs.embedding."`
  - `EMBEDDING_KINDS: frozenset[str] = frozenset({"instance", "frame"})`
  - `EMBEDDING_AXES: dict[str, tuple[str, ...]] = {"instance": ("frame", "slot", "feature"), "frame": ("frame", "feature")}`
  - `EMBEDDING_DTYPES: frozenset[str] = frozenset({"float16", "float32"})`
  - `parse_embedding_id(component_id: str) -> tuple[str, str] | None` — `(name, kind)` for a well-formed embedding id, else `None` (also `None` for ids outside the prefix).
  - `is_embedding_id(component_id: str) -> bool` — `component_id.startswith(EMBEDDING_PREFIX)` (well-formed or not).
  - `@dataclass(frozen=True) EmbeddingModel(name: str, weights: str)`
  - `@dataclass(frozen=True) TemporalWindow(before: int, after: int)`
  - `@dataclass(frozen=True) Projection(kind: str, input_dim: int, basis: str)`
  - `@dataclass(frozen=True) EmbeddingDescriptor(name: str, kind: str, dim: int, model: EmbeddingModel, tap: str, pooling: str, window: TemporalWindow, normalization: str = "none", projection: Projection | None = None)` with
    - `__post_init__` raising `ValueError` on: bad `name` segment, `kind` not in `EMBEDDING_KINDS`, `dim < 1`, window ints < 0, `model.weights` not matching `^sha256:[0-9a-f]{64}$`, empty `tap`/`pooling`/`normalization`/`model.name`, `projection.input_dim < 1`.
    - `fingerprint` (property) `-> str`
    - `to_json() -> dict` — the manifest form, **including** `"fingerprint"` and `"projection": None` when absent.
    - `@classmethod from_json(cls, data: dict) -> EmbeddingDescriptor` — ignores the stored `fingerprint` (the validator compares it).
  - `canonical_json(obj: object) -> bytes`
  - `compute_fingerprint(descriptor_json: dict) -> str` — operates on the manifest dict form, dropping `name`, `kind`, `fingerprint` keys if present (so the validator can call it on raw manifest JSON).
  - `PAIRED_FIELDS: tuple[str, ...] = ("dim", "model", "tap", "window", "normalization", "projection")` — what `<name>.instance` and `<name>.frame` must share.

- [ ] **Step 1: Failing tests** (`test_embeddings.py`):
  - `test_fingerprint_test_vector` — descriptor `name="trunk", kind="instance", dim=4, model=EmbeddingModel("vit-carry-spal", "sha256:" + "0"*64), tap="trunk.final", pooling="mask_mean", window=TemporalWindow(9, 9)` → `.fingerprint == "sha256:df9397378684bc85bac5df57e6137f5f69fb0dd3abdc99b33d0e2570c5c17696"`; and `canonical_json` of its fingerprint input equals `b'{"dim":4,"model":{"name":"vit-carry-spal","weights":"sha256:000...0"},"normalization":"none","pooling":"mask_mean","projection":null,"tap":"trunk.final","window":{"after":9,"before":9}}'` (64 zeros).
  - `test_fingerprint_ignores_name_and_kind` — same descriptor with `name="other", kind="frame"` has the same fingerprint.
  - `test_fingerprint_changes_with_each_paired_field` — parametrize over replacing dim, model.weights, tap, pooling, window.before, normalization, projection → fingerprint differs.
  - `test_canonical_json_is_key_order_independent_and_utf8` — `canonical_json({"b": 1, "a": "é"}) == canonical_json({"a": "é", "b": 1}) == '{"a":"é","b":1}'.encode()`.
  - `test_json_round_trip` — `from_json(d.to_json()) == d`; `to_json()["fingerprint"] == d.fingerprint`; `compute_fingerprint(d.to_json()) == d.fingerprint`.
  - `test_parse_embedding_id` — parametrize: `jabs.embedding.trunk.instance → ("trunk","instance")`, `jabs.embedding.blk-31.frame → ("blk-31","frame")`, `jabs.embedding.trunk → None`, `jabs.embedding.trunk.extra → None`, `jabs.embedding.a.b.instance → None`, `jabs.pose.points → None`.
  - `test_descriptor_rejects` — parametrize the `__post_init__` ValueError cases above.
- [ ] **Step 2:** `uv run pytest TPF/test_embeddings.py -q` → FAIL (import error).
- [ ] **Step 3:** Implement `PF/embeddings.py` per the interfaces. Descriptor strings/ints only.
- [ ] **Step 4:** Same command → PASS.
- [ ] **Step 5:** Commit `feat(pose-file): embedding descriptor and fingerprint`.

---

### Task 3: `Component.embedding`, manifest, schema, reader

**Files:**
- Modify: `PF/types.py` (`Component`: new field + `__post_init__` rules; docstring)
- Modify: `PF/manifest.py` (`_component_entry` writes `"embedding"`)
- Modify: `PF/reader.py` (`_component_from` parses it)
- Modify: `PF/schemas/manifest-1.json` (`$defs/embedding`; component property; conditional)
- Test: `TPF/test_construction_invariants.py`, `TPF/test_schema.py`, `TPF/test_embeddings.py`, `TPF/conftest.py`

**Interfaces:**
- Consumes: Task 2's `EmbeddingDescriptor`, `parse_embedding_id`, `is_embedding_id`, `EMBEDDING_AXES`, `EMBEDDING_DTYPES`.
- Produces:
  - `Component.embedding: EmbeddingDescriptor | None = None` (declared after `alternates`).
  - `conftest.build_embedding_component(name: str = "trunk", kind: str = "instance", *, frames: int = 4, slots: int = 2, dim: int = 8, dtype: str = "float16", tap: str = "trunk.final") -> Component` and `conftest.build_pose_file_with_embeddings() -> PoseFile` (sample pose file + `trunk.instance` + `trunk.frame`, rows NaN where `jabs.pose.slot_occupied` is false). Later tasks use both.

- [ ] **Step 1: Failing tests**
  - `test_construction_invariants.py::test_embedding_component_rules` — parametrize, each raising `ValueError`: id `jabs.embedding.trunk.extra`; embedding id without descriptor; descriptor on `org.example.lab.x`; descriptor name `blk31` on id `...trunk.instance`; kind mismatch; axes `("frame","slot","embedding")`; dim 8 vs payload last axis 7; dtype float64; missing `{"policy":"none"}`.
  - `test_embeddings.py::test_embedding_round_trip` — write `build_pose_file_with_embeddings()`, `validate` → no errors, `read_pose_file(...).component("jabs.embedding.trunk.instance").embedding == original descriptor`; manifest entry's `embedding["fingerprint"]` equals descriptor fingerprint.
  - `test_embeddings.py::test_embedding_from_block_source` — same file but the instance component's data is a `BlockSource` yielding two blocks of float16 rows; writes, validates, reads back equal (Review Focus 1).
  - `test_schema.py::test_schema_requires_descriptor_on_embedding_ids` and `::test_schema_forbids_descriptor_elsewhere` — via `validate_manifest` on dicts built from `minimal_manifest`.
- [ ] **Step 2:** Run those tests → FAIL.
- [ ] **Step 3:** Implement:
  - `Component.__post_init__`: if `is_embedding_id(id)`: `parse_embedding_id` must succeed, `embedding` required, name/kind equal parsed, `axes == EMBEDDING_AXES[kind]`, `data.shape[-1] == embedding.dim`, `dtype in EMBEDDING_DTYPES`, `missing == {"policy": "nan"}`. Else `embedding` must be `None`.
  - Schema: `$defs/embedding` — object, `additionalProperties: false`, all ten keys required (`name, kind, dim, model, tap, pooling, window, normalization, projection, fingerprint`); `name` pattern `^[a-z0-9]([a-z0-9_-]*[a-z0-9])?$`; `kind` enum; `dim` integer ≥ 1; `model` {`name` string minLength 1, `weights` pattern `^sha256:[0-9a-f]{64}$`}, no extras; `tap`, `pooling`, `normalization` strings minLength 1; `window` {`before`,`after` integers ≥ 0}, no extras; `projection` `oneOf [null, {kind string, input_dim int ≥1, basis string}]` all required, no extras; `fingerprint` pattern `^sha256:[0-9a-f]{64}$`. Component property `"embedding": {"$ref": "#/$defs/embedding"}`. Add to the component `allOf`: `{"if": {"required": ["id"], "properties": {"id": {"pattern": "^jabs\\.embedding\\."}}}, "then": {"required": ["embedding"]}, "else": {"not": {"required": ["embedding"]}}}`.
  - Manifest: `entry["embedding"] = component.embedding.to_json()` when set. Reader: `embedding=EmbeddingDescriptor.from_json(spec["embedding"]) if "embedding" in spec else None`.
- [ ] **Step 4:** `uv run pytest packages/jabs-io/tests -q` → PASS.
- [ ] **Step 5:** Commit `feat(pose-file): embedding components carry a validated descriptor`.

---

### Task 4: Requested layouts and the embedding layout policy

**Files:**
- Modify: `PF/types.py` (`Component.requested_layout` + validation)
- Modify: `PF/writer.py` (`_layout_for`, `_create_dataset`; constants)
- Modify: `PF/reader.py` (`_component_from` fills `requested_layout`)
- Test: `TPF/test_layout.py`

**Interfaces:**
- Consumes: Task 2 `is_embedding_id`; Task 3 conftest builders.
- Produces:
  - `Component.requested_layout: dict | None = None` (after `embedding`).
  - `writer._layout_for(component: Component) -> dict` is the **single source of truth**; `_create_dataset` derives `chunks`/`compression`/`compression_opts` from it (no second policy path).
  - `writer._EMBEDDING_CHUNK_BYTES: int = 1_048_576`.

- [ ] **Step 1: Failing tests** (`test_layout.py`):
  - `test_embedding_default_layout_is_chunked_uncompressed` — F=4,S=2,D=8 float16 instance → manifest layout `{"storage":"chunked","chunks":[4,2,8],"compression":"none"}`; h5py dataset `.chunks == (4,2,8)`, `.compression is None`.
  - `test_embedding_default_chunk_targets_one_mebibyte` — F=1000,S=4,D=1664 float16 (13312 B/frame) → frame chunk `78`.
  - `test_wide_rows_get_one_frame_chunks` — F=3,S=8,D=65536 float32 → chunks `[1,8,65536]` (Review Focus 3).
  - `test_zero_frame_embedding_is_contiguous` — F=0 → layout `{"storage":"contiguous","compression":"none"}`, writes and validates (Review Focus 2).
  - `test_requested_layout_is_honored_and_clamped` — request `{"storage":"chunked","chunks":[500,2,8],"compression":"gzip","compression_opts":4}` on F=4 → recorded/actual chunks `[4,2,8]`, gzip 4.
  - `test_requested_layout_rejected_on_jabs_pose` — `requested_layout` on `jabs.pose.points` → `ValueError`; also rejected on `jabs.segmentation.contour_count`.
  - `test_requested_layout_validation` — parametrize ValueError: contiguous+gzip; compression `szip`; gzip opts 10; lzf with opts 1; chunks rank mismatch; chunk extent 0.
  - `test_round_trip_keeps_embedding_and_foreign_layouts` — file with pose + embedding requested gzip 4 + foreign component `org.example.lab.x` requested `{"storage":"contiguous","compression":"none"}`; read then write to a second path; second manifest has the same layouts for those two and `jabs.pose.points` is contiguous (Review Focus 5).
  - `test_foreign_without_request_keeps_default_policy` — a foreign component with no request still gets today's `(300, ...)` gzip-1 policy on first write.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement. Policy order in `_layout_for`: zero-size → contiguous/none; `requested_layout` → validated+clamped request; embedding id → embedding default; keypoint-scale prefix → contiguous; else existing chunked gzip-1. `Component.__post_init__` validates `requested_layout` shape (keys ⊆ `storage, chunks, compression, compression_opts`) and the rules in Global Constraints, and rejects it on `jabs.*` ids that are not embedding ids. Reader: `requested_layout=spec.get("layout") if (is_embedding_id(id) or not id.startswith("jabs.")) else None`. Record `compression_opts` only when compression is `gzip`.
- [ ] **Step 4:** `uv run pytest packages/jabs-io/tests -q` → PASS.
- [ ] **Step 5:** Commit `feat(pose-file): producer-selectable layout for embedding and foreign components`.

---

### Task 5: Structural validator rules and their fixtures

**Files:**
- Modify: `PF/validate.py` (new `_check_embeddings(manifest: dict, findings: list[Finding]) -> None`, called from `validate()` after `_check_ragged_pairing`)
- Modify: `GEN` (new `valid_embedding()` builder + invalid entries)
- Modify: `TPF/test_conformance.py` (`VALID`, `INVALID`, `SCHEMA_MESSAGES`)
- Create (generated): `packages/jabs-io/tests/data/pose-format/valid-embedding.h5`, `invalid-embedding-*.h5`

**Interfaces:**
- Consumes: Task 2 `parse_embedding_id`, `is_embedding_id`, `compute_fingerprint`, `PAIRED_FIELDS`, `EMBEDDING_AXES`, `EMBEDDING_DTYPES`.
- Produces: error check names `embedding_id_shape`, `embedding_axes`, `embedding_descriptor_matches`, `embedding_storage`, `embedding_fingerprint`, `embedding_pair_consistent`.

Rules (per embedding-prefixed component; when the id does not parse, report only `embedding_id_shape` for it and skip its other rules and its pairing):
- `embedding_id_shape`: `parse_embedding_id` fails.
- `embedding_axes`: axes ≠ `EMBEDDING_AXES[kind]`.
- `embedding_descriptor_matches`: descriptor `name`/`kind` ≠ id, or `dim` ≠ last entry of `shape`.
- `embedding_storage`: dtype ∉ `EMBEDDING_DTYPES` or missing ≠ `{"policy": "nan"}`.
- `embedding_fingerprint`: `compute_fingerprint(descriptor) != descriptor["fingerprint"]`.
- `embedding_pair_consistent`: both kinds present for a name and any `PAIRED_FIELDS` value differs.

- [ ] **Step 1: Fixtures.** In `GEN` add `valid_embedding() -> PoseFile`: `valid_minimal()`'s components plus `jabs.pose.slot_occupied` if absent, `jabs.embedding.trunk.instance` (float16, D=8), `jabs.embedding.trunk.frame` (float16, D=8, pooling `grid_mean`), `jabs.embedding.blk31.instance` (float32, D=4, tap `trunk.block31`); per-instance rows all-NaN where `slot_occupied` is false and at least one occupied all-NaN row; each with a provenance record. Write it in `main()` as `valid-embedding.h5`. Add invalid fixtures as `_invalid_corpus` entries sourced from `valid-embedding.h5`, each a manifest mutation that **recomputes the fingerprint** unless the fixture is about the fingerprint:
  | fixture | mutation | expected |
  |---|---|---|
  | `invalid-embedding-id.h5` | trunk.frame id → `jabs.embedding.trunk.misc` | `embedding_id_shape` |
  | `invalid-embedding-axes.h5` | trunk.frame axes → `["frame","embedding"]` | `embedding_axes` |
  | `invalid-embedding-dim.h5` | blk31.instance descriptor dim → 5 | `embedding_descriptor_matches` |
  | `invalid-embedding-dtype.h5` | blk31.instance: replace dataset with float64, declare `float64` | `embedding_storage` |
  | `invalid-embedding-fingerprint.h5` | trunk.instance fingerprint → `"sha256:" + "f"*64` | `embedding_fingerprint` |
  | `invalid-embedding-pair.h5` | trunk.frame tap → `trunk.other` | `embedding_pair_consistent` |
  | `invalid-embedding-descriptor-missing.h5` | pop trunk.instance `embedding` | `manifest_schema` |
  | `invalid-embedding-descriptor-elsewhere.h5` | copy trunk.instance descriptor onto `jabs.pose.points` | `manifest_schema` |
- [ ] **Step 2: Failing tests.** Add the fixture names to `VALID`/`INVALID`; add `SCHEMA_MESSAGES` entries: missing → `"'embedding' is a required property"`; elsewhere → pin the exact substring jsonschema reports (run once, copy it). Run `uv run python packages/jabs-io/tests/data/pose-format/generate.py` then `uv run pytest TPF/test_conformance.py -q` → FAIL (checks not implemented; `test_every_error_check_has_an_invalid_fixture` passes only after Step 3).
- [ ] **Step 3:** Implement `_check_embeddings` with the six rules; each `Finding(ERROR, "<literal name>", msg)` with a literal check name (so the conformance AST scan sees it).
- [ ] **Step 4:** Regenerate fixtures; `uv run pytest packages/jabs-io/tests -q` → PASS.
- [ ] **Step 5:** Commit `feat(pose-file): validate embedding components`, including the regenerated `.h5` fixtures.

---

### Task 6: Deep checks for embedding rows

**Files:**
- Modify: `PF/validate.py` (new `_check_embedding_rows(h5: h5py.File, manifest: dict, findings: list[Finding]) -> None`, called under `if deep:`; update `validate()` docstring)
- Modify: `GEN`, `TPF/test_conformance.py` (`DEEP_INVALID`)
- Test: `TPF/test_deep_check.py`

**Interfaces:**
- Produces: error checks `embedding_rows_whole`, `embedding_unoccupied_nan`; module constant `_DEEP_FRAME_BLOCK: int = 1024`.

Rules: read each well-formed embedding dataset in frame blocks of `_DEEP_FRAME_BLOCK`; per row (last axis = feature) `isnan` count must be 0 or D → else `embedding_rows_whole` (one finding per component, message names the first offending frame/slot). For `.instance` components, when `jabs.pose.slot_occupied` exists, rows where it is false must be all-NaN → else `embedding_unoccupied_nan`. Skip silently when `slot_occupied` is absent.

- [ ] **Step 1: Fixtures + failing tests.** In `GEN` main: copy `valid-embedding.h5` to `invalid-embedding-partial-nan.h5` and set one element of an occupied, non-NaN trunk.instance row to NaN; copy to `invalid-embedding-unoccupied.h5` and fill one unoccupied blk31 row with zeros. Add both to `DEEP_INVALID`. In `test_deep_check.py`: `test_deep_check_reads_float16_in_blocks` — monkeypatch `validate._DEEP_FRAME_BLOCK` to 3, write a 10-frame float16 file with a partial-NaN row at frame 7 → `embedding_rows_whole` found (Review Focus 4); `test_instance_embedding_without_slot_occupied_is_not_checked` — file without `jabs.pose.slot_occupied` passes deep.
- [ ] **Step 2:** Regenerate, run `uv run pytest TPF/test_conformance.py TPF/test_deep_check.py -q` → FAIL.
- [ ] **Step 3:** Implement `_check_embedding_rows`.
- [ ] **Step 4:** `uv run pytest packages/jabs-io/tests -q` → PASS.
- [ ] **Step 5:** Commit `feat(pose-file): deep-check embedding rows`.

---

### Task 7: `list_embeddings` and public exports

**Files:**
- Modify: `PF/embeddings.py` (add `EmbeddingInfo`, `list_embeddings`)
- Modify: `PF/__init__.py` (exports + module docstring mention)
- Test: `TPF/test_embeddings.py`, `TPF/test_public_api.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) EmbeddingInfo(name: str, kind: str, dim: int, fingerprint: str, component_id: str)`
  - `list_embeddings(manifest: dict) -> list[EmbeddingInfo]` — from a raw manifest (`read_manifest` output); well-formed embedding ids only, sorted by `(name, kind)`.
  - Exported from `jabs.io.internal.pose_file`: `EmbeddingDescriptor`, `EmbeddingModel`, `TemporalWindow`, `Projection`, `EmbeddingInfo`, `list_embeddings`, `compute_fingerprint`.

- [ ] **Step 1: Failing tests** — `test_list_embeddings_from_fixture`: `list_embeddings(read_manifest(DATA/"valid-embedding.h5"))` returns three infos `[("blk31","instance",4,...),("trunk","frame",8,...),("trunk","instance",8,...)]` and the two trunk fingerprints differ (pooling differs). Extend `test_public_names_are_importable` with the seven names.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** full jabs-io suite → PASS.
- [ ] **Step 5:** Commit `feat(pose-file): list a file's embeddings`.

---

### Task 8: ADR 0003 and the pointer in ADR 0002

**Files:**
- Create: `docs/development/adr/0003-embedding-components.md`
- Modify: `docs/development/adr/0002-unified-pose-file-format.md` (one-line pointers only: under Design goal 1 and at the top of "Component catalog": "Extended by ADR 0003 (embedding components).")

**Interfaces:** consumes everything above; the ADR must agree with the code.

- [ ] **Step 1:** Write ADR 0003 (Status: Proposed) following 0002's section layout: Context (why: behavior classifiers, Seizure false-positive test; the spike is not carried forward), Decision, Specification (catalog table, `feature` axis, float16, the `embedding` manifest field table, fingerprint + **normative canonical JSON**, `window` as upper bound with the SPAL `{9, 9}` example, row/NaN semantics incl. tail slots, pairing rule, requested-layout rules and the embedding-scale storage row, scope clarification of 0002 goal 1, clipping note), Validation (rows for the eight new checks, two marked deep), Consequences, Alternatives considered (0002-in-place, lab namespace), Open questions (a clip tool; promoting the pose-file API out of `internal`). Include an example manifest entry using the Task 2 test-vector descriptor and fingerprint.
- [ ] **Step 2:** Verify every check name in ADR 0003's validation table appears in `validate.py`: `for c in embedding_id_shape embedding_axes embedding_descriptor_matches embedding_storage embedding_fingerprint embedding_pair_consistent embedding_rows_whole embedding_unoccupied_nan; do grep -q "\"$c\"" packages/jabs-io/src/jabs/io/internal/pose_file/validate.py || echo MISSING $c; done` → no output.
- [ ] **Step 3:** `uv run ruff check . && uv run ruff format --check .` and `uv run pytest packages/jabs-io/tests -q` → clean / PASS.
- [ ] **Step 4:** Commit `docs: ADR 0003, embedding components`.
