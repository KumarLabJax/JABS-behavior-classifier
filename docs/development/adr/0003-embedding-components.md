# ADR 0003: Embedding Components

## Status

Proposed

Extends [ADR 0002](0002-unified-pose-file-format.md) (Unified Pose File Format), which is itself
still Proposed. Everything here is additive: a file with no `jabs.embedding.*` component is read,
written and validated exactly as ADR 0002 describes. `schema_revision` stays `1`; no unified-format
file has been released.

## Context

JABS behavior classifiers are built on pose and segmentation features. We want to test whether
learned visual embeddings, per mouse and per whole frame, improve them. The first test is the
Seizure false-positive dataset, where pose+segmentation classifiers produce too many false
positives. The comparison is three arms: CNN SPAL (pose+seg), ViT SPAL (pose+seg), and ViT SPAL
(pose+seg+embeddings).

The embeddings come from the pose model itself. ViT-Carry-SPAL (JABS-pose, `spal-video-vit`)
already computes a per-frame visual representation while it infers keypoints; writing it out is
cheap at inference time and expensive to reproduce afterwards. So the natural home is the same
per-video pose file, written in the same run, under the same provenance.

An earlier `embedding-feature-group` spike stored embeddings from a different model in a separate
`.vjepa.h5` sidecar. It was a spike only: it is not a comparison arm, and its sidecar format is
not carried forward. Its useful parts (a feature group, a provenance-keyed cache, AUPRC in
cross-validation) belong to the JABS side of this work, not to the file format.

ADR 0002 has no place for these arrays. Three things in it stand in the way:

- **The name `embedding` is taken.** The `embedding` axis and `jabs.identity.*` mean *identity*
  embedding. Reusing them for visual features would re-meaning an existing name, which goal 4
  forbids.
- **Nothing says what an embedding is.** Two files' vectors can be compared only if the same
  model, weights, tap point, pooling, temporal window, normalization and projection produced them.
  ADR 0002's free-form `extra` and provenance `parameters` could hold that, but nothing validates
  them, and a consumer would end up trusting a producer-chosen name.
- **The size is new.** A ViT-G embedding is 1,664 values. At 3-4 slots and ~108k frames an hour,
  a float32 per-instance embedding is 2.2-2.9 GB per video-hour, two orders of magnitude above
  keypoint scale. ADR 0002 has no `float16` and lets no producer choose a layout.

## Decision

**Embeddings are first-class catalog components under the reserved `jabs` namespace**, in a new
`jabs.embedding` section, each carrying a typed, validated descriptor. Its fingerprint, together
with the component's kind, decides comparability. Concretely, this ADR adds to ADR 0002:

- a catalog section, `jabs.embedding`, with a per-instance and a whole-frame kind, several named
  embeddings per file;
- an axis, `feature`;
- a dtype, `float16`, legal everywhere;
- a manifest field, `embedding` (the descriptor), required on embedding components and forbidden
  on all others;
- producer-selectable storage layout for embedding and foreign components, with an embedding-scale
  default;
- a scope clarification of ADR 0002 design goal 1;
- eight validation checks, two of them opt-in deep checks.

## Specification

Revision 1, as ADR 0002. All rules below are additions to it.

### Scope clarification (ADR 0002 design goal 1)

ADR 0002 goal 1 says "Labels, features, predictions and classifiers stay where they are." This ADR
reads it as follows: a per-frame representation **produced by a model at inference time from
video** is inference output and belongs in the pose file. Features **computed by JABS** from pose
file contents stay out. An embedding is the former: it comes from the pose model's run over the
video, not from anything JABS derives.

### Axes

One axis is added to ADR 0002's vocabulary:

| axis | meaning |
|---|---|
| `feature` | dimension of a learned feature vector (see `jabs.embedding`) |

`feature` is **not** a `dimensions` key. Different embeddings in one file may have different
widths, so its length is per component: the descriptor's `dim`. The existing `embedding` axis keeps
its meaning, identity-embedding dimension, and is not used by embedding components.

### dtype: `float16`

`float16` is added to the dtype enum (`manifest-1.json` `$defs/dtype`, `DTYPE_NAMES` in
`types.py`) and to the `nan` missing-policy conditional, which now allows `float16`, `float32` and
`float64`. It is generally legal, for any component or alternate, not only for embeddings. A
reader returns float16 as float16; upcasting is the consumer's choice.

### Component catalog: `jabs.embedding`

Shapes use `F` = frames, `S` = slots, `D` = the embedding's feature width (its descriptor's `dim`).

| id | path | axes | shape | dtype | missing |
|---|---|---|---|---|---|
| `jabs.embedding.<name>.instance` | `/jabs/embedding/<name>/instance` | frame, slot, feature | F×S×D | float16 \| float32 | `nan` |
| `jabs.embedding.<name>.frame` | `/jabs/embedding/<name>/frame` | frame, feature | F×D | float16 \| float32 | `nan` |

- **`<name>`** is one id segment, `[a-z0-9]([a-z0-9_-]*[a-z0-9])?`, chosen by the producer (`trunk`,
  `blk31`). It is a label, unique per file, and nothing more: comparability is decided by the
  kind and the descriptor's fingerprint, never by the name.
- **The id shape is exact.** Every id under `jabs.embedding.` must be
  `jabs.embedding.<name>.instance` or `jabs.embedding.<name>.frame`. `jabs.embedding.trunk`,
  `jabs.embedding.trunk.extra` and `jabs.embedding.a.b.instance` are invalid.
- **Either kind may appear without the other**, and a file may hold several named embeddings.
- **Row semantics.** Row *t* (and, per instance, slot *k*) is the embedding *for frame t*. A row is
  either a whole vector or **all-NaN**; all-NaN means the producer has no embedding for it. That
  covers every per-instance row where `jabs.pose.slot_occupied` is false, and also rows where a
  slot is occupied but no embedding could be made (an empty mask after pooling, say). **A
  partially-NaN row is invalid.** The missing policy is `nan`, not `mask` → `slot_occupied`,
  precisely so that "occupied but no embedding" is expressible.
- **Tail slots** (`[identity, slot)`, unassigned instances) get embeddings like any other slot, so
  identity resolution stays a two-way door (ADR 0002 goal 8).
- Embeddings are identity-*indexed* by slot, as pose is. They carry no identity semantics
  themselves; `jabs.identity.*` remains the identity embedding.

A `.instance` embedding is legal in a file without `jabs.pose.slot_occupied`. Nothing then says
which slots are empty, and the deep check's unoccupied-slot rule is skipped.

### Manifest field: `embedding` (the descriptor)

A new optional component property. A schema conditional makes it **required** on every component
whose id starts with `jabs.embedding.` and **forbids** it on every other component. Every field is
required; `projection` may be `null`.

| field | type | rule |
|---|---|---|
| `name` | string | one id segment; equals the `<name>` segment of the id |
| `kind` | `"instance"` \| `"frame"` | equals the id's last segment |
| `dim` | integer ≥ 1 | equals the length of the `feature` axis (the last axis of `shape`) |
| `model.name` | non-empty string | model identifier |
| `model.weights` | string, `^sha256:[0-9a-f]{64}$` | content identifier of the exact weights: lowercase hex SHA-256 of the checkpoint file |
| `tap` | non-empty string | producer-defined name of where in the model the vector comes from |
| `pooling` | non-empty string | how spatial positions were reduced; registered values `mask_mean`, `bbox_mean`, `grid_mean`, `token`; others allowed |
| `window` | `{before, after}`, integers ≥ 0 | upper bound on the temporal receptive field (below) |
| `normalization` | non-empty string | `none`, `l2`, or producer-defined |
| `projection` | `null` or `{kind, input_dim, basis}` | a post-tap reduction, e.g. `{"kind": "pca", "input_dim": 1664, "basis": "sha256:<hex>"}`; `kind` and `basis` non-empty; `input_dim` ≥ 1; `dim` is then the output width |
| `fingerprint` | string, `^sha256:[0-9a-f]{64}$` | as below |

#### Fingerprint and canonical JSON (normative)

```
fingerprint = "sha256:" + lowercase_hex(SHA-256(canonical_json(descriptor minus name, kind, fingerprint)))
```

**The comparability key is the pair `(kind, fingerprint)`.** Two embeddings are interchangeable
for training and prediction **only when both their `kind` and their `fingerprint` are equal**;
names are labels only. `name` is excluded from the fingerprint so that the same vectors under
another name compare equal. `kind` is excluded too, which is why a consumer must compare it
alongside the fingerprint: an `.instance` and a `.frame` embedding normally differ in `pooling`
and so in fingerprint, but nothing requires that, and a producer may record the same `pooling`
for both.

The descriptor may contain only objects, arrays, strings, integers, booleans and `null`: **no
floating-point numbers**, and an integer field holds a JSON integer written without a decimal
point (`9`, never `9.0`; JSON Schema accepts `9.0` as an integer, so `validate()` reports it as
`embedding_descriptor_types`). Under that restriction the canonical form is RFC 8785 (JCS), which
for these values is exactly Python's form below. The equivalence holds because every key is a
fixed ASCII key from the schema (so Python's code-point sort and JCS's UTF-16 sort agree) and
every integer is at most 2^53 - 1, a bound the schema enforces as each integer field's `maximum`
(so it serializes identically as an IEEE 754 double):

```python
json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
```

That is: keys sorted at every level, no whitespace, non-ASCII written as UTF-8 rather than
escaped, and an absent projection written as `null`. A producer writing with `h5py` and the
standard `json` module can reproduce it without `jabs-io`.

**Test vector.** The descriptor `name="trunk"`, `kind="instance"`, `dim=4`,
`model={"name": "vit-carry-spal", "weights": "sha256:" + 64 × "0"}`, `tap="trunk.final"`,
`pooling="mask_mean"`, `window={"before": 9, "after": 9}`, `normalization="none"`,
`projection=null` has the fingerprint input

```
{"dim":4,"model":{"name":"vit-carry-spal","weights":"sha256:0000000000000000000000000000000000000000000000000000000000000000"},"normalization":"none","pooling":"mask_mean","projection":null,"tap":"trunk.final","window":{"after":9,"before":9}}
```

and the fingerprint

```
sha256:df9397378684bc85bac5df57e6137f5f69fb0dd3abdc99b33d0e2570c5c17696
```

(`test_fingerprint_test_vector` in `packages/jabs-io/tests/internal/pose_file/test_embeddings.py`
pins both.)

#### `window` is an upper bound

`window.before` (and, separately, `window.after`) is the **maximum over all frames** of how many
frames before (after) *t* the vector for *t* can depend on through the model's clip. It is a
nominal bound, not a per-frame value, so two producers with the same model and clip schedule
record the same number.

For SPAL, which runs 12-frame clips at stride 8, the vector for an interior frame (frames 2-9 of
its clip) can see up to 9 frames before it and up to 9 after. The ends of a video set the
maximum, though: the first clip also emits its frames 0-1, so frame 0 can see 11 frames after it,
and the final, right-aligned clip lets the last frame see up to 11 before it. SPAL's window is
therefore `{"before": 11, "after": 11}`. A producer computes it from its clip schedule rather than
from the interior frames alone. (The examples in this ADR use `{9, 9}`; they are the test vector's
values, not a statement about SPAL.)

**Carry and recurrent models.** For ViT-Carry-SPAL, frame *t*'s vector can also depend, weakly, on
earlier clips through the carried prior. `window` records the nominal clip receptive field only;
recurrent dependence is the producer's to describe in provenance `parameters`. Keeping `window` a
simple, comparable pair of integers is deliberate.

#### Pairing rule

When both `jabs.embedding.<name>.instance` and `jabs.embedding.<name>.frame` exist, their
descriptors may differ only in `kind`, `pooling` and `fingerprint`. `dim`, `model`, `tap`,
`window`, `normalization` and `projection` must be equal. A name means one tap; a different tap
needs a different name.

#### Descriptor and provenance

Each embedding component should carry a provenance record, as every component should. The
descriptor is the normative, validated subset of what produced the vectors; provenance holds
everything else (git commit, inference parameters, carry details).

### Example manifest entry

A per-instance float16 embedding using the test-vector descriptor, as `jabs-io` writes it for a
1,800-frame, 4-slot file:

```json
{ "id": "jabs.embedding.trunk.instance", "path": "/jabs/embedding/trunk/instance",
  "axes": ["frame", "slot", "feature"],
  "dtype": "float16", "shape": [1800, 4, 4],
  "encoding": { "kind": "dense" },
  "missing": { "policy": "nan" },
  "layout": { "storage": "chunked", "chunks": [1800, 4, 4], "compression": "none" },
  "embedding": {
    "name": "trunk",
    "kind": "instance",
    "dim": 4,
    "model": {
      "name": "vit-carry-spal",
      "weights": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
    },
    "tap": "trunk.final",
    "pooling": "mask_mean",
    "window": { "before": 9, "after": 9 },
    "normalization": "none",
    "projection": null,
    "fingerprint": "sha256:df9397378684bc85bac5df57e6137f5f69fb0dd3abdc99b33d0e2570c5c17696"
  },
  "provenance": "jabs.embedding.trunk" }
```

The chunk covers all 1,800 frames because the default targets ~1 MiB and a frame here is only
32 bytes. A real ViT-G embedding (1,664 features, 4 slots, float16, 13,312 bytes a frame) gets
78-frame chunks.

### Storage layout: producer-selectable

ADR 0002 lets the writer decide every layout from the component id. This ADR lets a producer
choose, for some components.

- **Who may request.** `jabs-io`'s `Component` gains an optional `requested_layout` (same
  vocabulary as the manifest `layout`: `storage`, `chunks`, `compression`, `compression_opts`). It
  is honored **only for `jabs.embedding.*` and foreign (non-`jabs`) components**. A request on any
  other `jabs.*` component is a `ValueError`: `jabs.pose.*` stays contiguous for the Hub's
  byte-range overlay, and segmentation keeps its policy. `Component.layout` keeps its ADR 0002
  meaning (what a file declared) and is never applied.
- **What may be requested.** The only keys are `storage`, `chunks`, `compression` and
  `compression_opts`. `storage` is `contiguous` or `chunked`. `compression` is `none`, `gzip` or `lzf`. `gzip` requires `compression_opts`, an integer 0-9;
  `none` and `lzf` take none. `chunked` requires `chunks`, one integer ≥ 1 per axis; `contiguous`
  forbids `chunks` and forbids compression, since HDF5 filters need chunking. Anything else is a
  `ValueError` at construction.
- **How it is applied.** The writer applies a valid request and records the result as the
  manifest `layout`. Each chunk extent is clamped to its axis length (minimum 1), so a frame chunk
  larger than short or clipped data shrinks to fit; the manifest records the clamped value. An
  empty payload is stored contiguous whatever was requested, as ADR 0002 already does.
- **Round trip.** On read, `jabs-io` turns a recorded `layout` into the component's
  `requested_layout` for embedding and foreign components **only if it passes the rules above**.
  A recorded layout the writer could not honor (a foreign producer's `szip`, say, which the schema
  admits) is dropped with a debug log, and the writer's policy applies on the next write. All other
  `jabs.*` components are re-laid-out by policy exactly as before, so a file without embedding or
  foreign components round-trips unchanged.

**The round trip freezes a layout.** Because the request travels with the component, a
read-then-write keeps the recorded layout even if the caller later replaces the data with a
different frame count or dtype. Chunk extents are still clamped to the new shape, but they are not
recomputed: a 78-frame chunk chosen for float16 stays 78 frames after an upcast to float32 (about
2 MiB), and a chunk clamped down for a short clip stays small when the data grows. A caller that
wants the default back passes `requested_layout=None`. Whether read-back should freeze layouts at
all is an open question below.

ADR 0002's storage table gains one row:

| component scale | layout | rationale |
|---|---|---|
| embedding-scale — `jabs.embedding.*` | **chunked along `frame`, every other axis whole, uncompressed**, frame chunk `max(1, min(F, 1,048,576 // bytes_per_frame))` (≈1 MiB) unless the producer requests otherwise | learned float vectors compress poorly, so gzip costs CPU for little; ≈1 MiB chunks keep the chunk count low over an hour-long recording while a frame window reads little more than it asked for |

### Clipping

No new rules. Every embedding component is dense with `frame` as axis 0, so ADR 0002's generic
frame-axis slice clips it. **The descriptor is unchanged by clipping**: it describes the model, not
the frames. `jabs-io` has no clip tool yet; embeddings are clip-ready by construction.

### Reading

`read_component(path, "jabs.embedding.trunk.instance", frames=slice(a, b))` works as for any dense
component. `list_embeddings(manifest)` returns one `EmbeddingInfo` (name, kind, dim, fingerprint,
component id) per well-formed embedding component, sorted by `(name, kind)`, so consumers need not
parse ids. It reports the stored fingerprint; checking it is `validate()`'s job.

### Validation

`validate()` gains these error checks, named as the validator reports them:

| check | rule | severity |
|---|---|---|
| `embedding_id_shape` | an id under `jabs.embedding.` is exactly `jabs.embedding.<name>.instance` or `jabs.embedding.<name>.frame`; a component that fails reports only this, since its name and kind are unknown | error |
| `embedding_axes` | axes are exactly `[frame, slot, feature]` (instance) or `[frame, feature]` (frame) | error |
| `embedding_descriptor_matches` | the descriptor's `name` and `kind` equal the id's, and `dim` equals the feature-axis length | error |
| `embedding_storage` | dtype is `float16` or `float32`, and `missing` is exactly `{"policy": "nan"}` | error |
| `embedding_descriptor_types` | `dim`, `window.before`, `window.after` and `projection.input_dim` hold no float (even an integral one such as `9.0`) or bool | error |
| `embedding_fingerprint` | the stored `fingerprint` equals the one recomputed from the descriptor; not checked for a component that fails `embedding_descriptor_types`, which has no canonical form | error |
| `embedding_pair_consistent` | when `<name>.instance` and `<name>.frame` both exist, their `dim`, `model`, `tap`, `window`, `normalization` and `projection` are equal | error |
| `embedding_rows_whole` | **opt-in deep check:** no row is partially NaN | error when requested |
| `embedding_unoccupied_nan` | **opt-in deep check:** when `jabs.pose.slot_occupied` exists, every per-instance row where it is false is all-NaN | error when requested |

The schema carries the rest and reports through ADR 0002's existing `manifest_schema` check: the
descriptor is required on `jabs.embedding.*` ids and forbidden elsewhere, its fields have the types
and patterns in the table above, and `float16` passes the dtype and `nan`-policy rules.

**The deep checks read every embedding row**, which costs as much as reading the payload, so they
run only under `validate(path, deep=True)`, alongside ADR 0002's alternate-decode check. Each
embedding is read in pieces bounded by the same byte budget as ADR 0002's alternate deep check
(payload plus its NaN mask): whole frames while they fit, and a single frame a slice of the
feature axis at a time when it alone exceeds the budget. A multi-gigabyte embedding, or a file
whose declared shape is hostile, never forces a large allocation. Each rule reports at most one finding per component,
naming the first offending row. A block that cannot be read (a corrupt chunk) is reported as
`file_readable` and ends that component's deep check; `validate()` still never raises on a
malformed file. A component the structural checks found unindexable (missing payload, wrong rank,
non-float dtype) is skipped by the deep checks rather than reported twice.

`jabs-io` enforces every structural rule above at construction, before anything is written:
`EmbeddingDescriptor` the descriptor's types (`embedding_descriptor_types`), `Component` the
per-component rules (`embedding_id_shape`, `embedding_axes`, `embedding_descriptor_matches`,
`embedding_storage`; `embedding_fingerprint` holds by construction, since the writer computes the
fingerprint), and `PoseFile` the pairing rule (`embedding_pair_consistent`). The deep checks do
not run there. The writer runs `validate()` on the finished file before publishing it, which
re-checks all of these, and adds the two deep checks when it is called with `deep=True`.

**Conformance fixtures** follow ADR 0002's one-file-per-error-check pattern: `valid-embedding.h5`
(two named embeddings: `trunk` with float16 instance and frame kinds, `blk31` with a float32
instance kind; NaN rows on unoccupied slots) and one
invalid file per rule above, plus one each for a missing descriptor and a descriptor on a
non-embedding component.

## Consequences

### Positive

- Embeddings are a first-class, validated JABS input: a consumer can decide whether two files'
  embeddings are interchangeable from `(kind, fingerprint)` alone, without trusting names.
- One pose file per video still holds all inference output; embeddings inherit provenance,
  validation, round trips and lazy frame-window reads from the generic tooling.
- float16 halves embedding storage, to roughly 1.1-1.4 GB per video-hour for a 3-4 slot ViT-G
  per-instance embedding.
- A producer, or a foreign component's author, can choose a layout suited to its access pattern
  without changing the writer.

### Negative and trade-offs

- Two documents to read for the pose format instead of one.
- Files with embeddings are large, and the default leaves them uncompressed. That is deliberate,
  but it is disk the producer pays for.
- The read-back rule freezes a layout across later data changes (above), which can leave a
  replaced payload with chunks sized for its predecessor.
- A foreign component's on-disk layout now survives a round trip, where before it was re-laid-out
  by policy. That is intended (it is the foreign producer's choice) but it is a behavior change
  for files with foreign components.
- `window` deliberately does not capture recurrent dependence; two carry models with different
  carry behavior and the same clip schedule can share a fingerprint unless they differ elsewhere
  (they will normally differ in `model.weights`).

### Risks

- **Producer-defined strings decide comparability.** `tap`, `pooling` and `normalization` are free
  strings. Two producers who spell the same tap differently get different fingerprints (safe,
  but no reuse); one producer who reuses a tap name for a different tap gets a false match.
- **The canonical form depends on the no-float rule.** A future descriptor field that needs a real
  number must be encoded as a string or an integer, or the canonical-JSON rule must be revisited.

## Alternatives considered

**Amend ADR 0002 in place.** One document, but it reopens a large ADR mid-review and couples
acceptance of the pose format to decisions about embeddings. A separate ADR lets 0002 be reviewed
and accepted on its own.

**Lab-namespace foreign components (`org.jax.*`), no catalog entry.** Needs almost nothing from the
specification: only float16 would change. Rejected: JABS would depend on names the specification
does not define, which is the "optional everything" drift ADR 0002 warns about, and the
descriptor would live in unvalidated `extra`. Embeddings are meant to be a first-class JABS input.

**Reuse the `embedding` axis and `jabs.identity` namespace.** Rejected: it re-meanings an existing
name (ADR 0002 goal 4).

**Descriptor in `extra` or provenance `parameters`.** Rejected: nothing validates either, so a
consumer could not rely on it, and comparability would fall back to the name.

## Open questions

1. **A clip tool.** `jabs-io` has no clip tool. Embeddings need no new clipping rules, but nothing
   exercises the generic frame slice on them until one exists.
2. **Promoting the pose-file API out of `internal`.** Everything here lives in
   `jabs.io.internal.pose_file`. JABS adopting the format and ViT SPAL writing it will both depend
   on it; when to promote it to a public `jabs.io` API is open.
3. **Should read-back freeze layouts?** Carrying a recorded layout forward preserves a producer's
   choice across a round trip, at the cost described under Storage layout when the data changes
   shape or dtype. The alternatives are to freeze only when the payload is unchanged, or to never
   carry layouts forward and require the caller to restate them.

## References

- [ADR 0002: Unified Pose File Format](0002-unified-pose-file-format.md)
- `docs/superpowers/specs/2026-11-19-embedding-components-design.md` — the design this ADR records
- `packages/jabs-io/src/jabs/io/internal/pose_file/embeddings.py` — descriptor, fingerprint,
  `list_embeddings`
- `packages/jabs-io/src/jabs/io/internal/pose_file/types.py` — `requested_layout` rules
  (`requested_layout_problems`)
- `packages/jabs-io/src/jabs/io/internal/pose_file/validate.py` — the checks above
- RFC 8785, JSON Canonicalization Scheme (JCS)
