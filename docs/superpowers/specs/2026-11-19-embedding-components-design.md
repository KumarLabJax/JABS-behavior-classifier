# Embedding components in the unified pose file — design

Date: 2026-11-19
Status: design approved in conversation (approach); document pending review

## Why

We want JABS behavior classifiers to use learned visual embeddings — per mouse and per whole
frame — alongside pose and segmentation, to beat pose+segmentation alone. The first test is the
Seizure false-positive dataset, where pose+segmentation classifiers produce too many false
positives. The embeddings will come from the ViT-Carry-SPAL pose model (JABS-pose,
`spal-video-vit`), which will write one unified pose file (ADR 0002) per video holding keypoints,
segmentation and embeddings. The comparison is CNN SPAL (pose+seg) vs ViT SPAL (pose+seg) vs
ViT SPAL (pose+seg+embeddings).

The earlier `embedding-feature-group` spike (a separate `.vjepa.h5` sidecar from pbs-vjepa) was
a spike only; it is not a comparison arm and its sidecar format is not carried forward.

This document is **project 1 of 5**. The full sequence, agreed in conversation:

1. **Embedding components in the format** (this document): ADR 0003 + `jabs-io` support.
2. **JABS adopts the unified format** (ADR 0002 option "B": consumers rewritten onto the new
   model, legacy version branching removed, existing projects converted). Critical path.
3. **ViT SPAL writes the unified file**, including embeddings, via `jabs-io`. CNN SPAL files reach
   the new format through the legacy converter (PR #598). Runs in parallel with 2.
4. **JABS embedding features** on top of 2 (reusing the spike's good parts: feature group,
   provenance-keyed cache, AUPRC in CV).
5. **Seizure false-positive experiment**, three arms as above.

## Goals

- A file can carry **several named embeddings**, each with an optional **per-instance**
  component (frame × slot × D) and an optional **whole-frame** component (frame × D).
- Every embedding carries a **typed, validated descriptor** sufficient for a consumer to decide
  whether two files' embeddings are interchangeable (same model, weights, tap, pooling, window,
  normalization/projection), without trusting the producer-chosen name.
- Embeddings can be stored compactly (**float16**) and the producer can choose their **storage
  layout and compression**.
- Embedding components work with the generic tooling `jabs-io` already has: validation,
  read-then-write round trips, lazy frame-window reads. (Clipping is not implemented in `jabs-io`
  yet; embeddings are clip-ready by construction — `frame` is axis 0 — but no clip tool is built
  here. *(Claude's call.)*)

## Non-goals

- How JABS turns embeddings into features (project 4).
- How SPAL computes embeddings (tap points, pooling implementation) (project 3).
- JABS reading the unified format at all (project 2).
- A promoted public `jabs.io` API for the pose file (still `jabs.io.internal.pose_file`; revisit in
  project 2/3).
- Sparse (sample-axis) or ragged embeddings. Dense only.

## Anti-goals

- **Re-meaning existing names.** The `embedding` axis and `jabs.identity.*` mean identity
  embedding (ADR 0002 goal 4). Behavior/visual embeddings must not reuse them.
- **Name-keyed trust.** A consumer must never treat two embeddings as the same because they share
  `<name>`; the descriptor decides.
- **Free-form-only metadata.** Model/tap/pooling/window must not live only in `extra` or
  provenance `parameters`, where nothing validates them.
- **Breaking ADR 0002 for files without embeddings.** Everything here is additive; a file with no
  `jabs.embedding.*` components is unaffected.

## Constraints

- ADR 0002 is still *Proposed*; no unified-format files have been released.
- ADR 0002 rules: `jabs.*` reserved for spec-defined components; additive only; readers never
  branch on `schema_revision`; absence stated (NaN/mask/length), never sentinels.
- Component ids map to paths by dots (`types.py` `Component.path`); `COMPONENT_ID_RE` already
  admits `jabs.embedding.<name>.instance`.
- The manifest schema sets `additionalProperties: false` on components, so a new field is a schema
  change.
- Any axis whose name equals a `dimensions` key is length-checked against it.
- Scale: ViT-G D = 1664; 3–4 slots; ~108k frames/hour at 30 fps. float32 per-instance ≈ 2.2–2.9
  GB/hour; float16 halves that.
- Implementation builds on PR #596 (`feat/pose-file-alternates-and-streaming`), whose writer
  (BlockSource streaming, layouts) this work modifies; the SPAL producer (project 3) needs that
  streaming anyway.

## Approaches considered

1. **New ADR 0003 "Embedding components", extending 0002** — *chosen.* Adds a catalog section, an
   axis, a dtype, a manifest field, a layout override and a scope clarification, in its own
   decision record. 0002 can be reviewed/accepted independently; cost is two documents to read.
2. **Amend ADR 0002 in place.** One document, but reopens a large ADR mid-review and couples
   pose-format acceptance to embedding decisions.
3. **Lab-namespace (`org.jax.*`) foreign components, no catalog entry.** Only float16 would change.
   Rejected: JABS would depend on names the spec does not define — the "optional everything"
   drift ADR 0002 warns about — contrary to making embeddings a first-class JABS input.

## Decisions (made in conversation)

- Embeddings live under the reserved `jabs.*` namespace as catalog components (JABS gives them
  meaning; any model may produce them).
- Two kinds: per-instance (frame × slot × D) and whole-frame (frame × D).
- Multiple embeddings per file, distinguished by a producer-chosen name.
- New axis rather than reusing `embedding`.
- Typed, validated descriptor (not `extra`).
- float16 added; producer-selectable layout/compression.
- Scope clarification: per-frame representations a model produces at inference time are inference
  output; features JABS computes remain out of scope.
- `schema_revision` stays **1** (nothing released yet). *(Claude's call, stated in conversation.)*

## The design

### Deliverables

- `docs/development/adr/0003-embedding-components.md` (Status: Proposed), referencing 0002.
- `jabs-io` changes in `packages/jabs-io/src/jabs/io/internal/pose_file/` plus
  `schemas/manifest-1.json`, conformance fixtures and tests.
- A short pointer added to ADR 0002 (catalog intro and goal 1) saying 0003 extends it — the only
  edit to 0002.

### Catalog: `jabs.embedding`

| id | path | axes | shape | dtype | missing |
|---|---|---|---|---|---|
| `jabs.embedding.<name>.instance` | `/jabs/embedding/<name>/instance` | frame, slot, feature | F×S×D | float16 \| float32 | `nan` |
| `jabs.embedding.<name>.frame` | `/jabs/embedding/<name>/frame` | frame, feature | F×D | float16 \| float32 | `nan` |

- `<name>`: one id segment (`[a-z0-9]([a-z0-9_-]*[a-z0-9])?`), producer-chosen, e.g. `trunk`,
  `blk31`. Unique per file. Other ids under `jabs.embedding.` are invalid (e.g.
  `jabs.embedding.trunk.extra`, `jabs.embedding.trunk`).
- Either kind may appear without the other.
- **Row semantics.** Row *t* is the embedding *for frame t*. A row is all-NaN when the producer has
  no embedding for it: per-instance rows wherever `jabs.pose.slot_occupied` is false, and also
  where a slot is occupied but no embedding could be made (e.g. empty mask after pooling). A
  partially-NaN row is invalid. Missing policy is `nan` (not mask→`slot_occupied`) precisely
  because "occupied but no embedding" must be expressible.
- **Tail slots** (`[identity, slot)`, unassigned instances) get embeddings like any other slot, so
  identity resolution stays a two-way door.
- Embeddings are identity-*indexed* by slot like pose; they carry no identity semantics themselves.

### New axis: `feature`

`feature` — "dimension of a learned feature vector (see `jabs.embedding`)". It is **not** a
`dimensions` key: different embeddings in one file may have different D, so its length is
per-component (the descriptor's `dim`, below).

### dtype: `float16`

Added to the dtype enum (ADR, `manifest-1.json` `$defs/dtype`, `types.py` `DTYPE_NAMES`) and to
the `nan` policy conditional (now float16/float32/float64). Generally legal, not embedding-only.

### Manifest field: `embedding` (the descriptor)

A new optional component property, **required** (schema conditional + validator) on every
`jabs.embedding.*` component and **forbidden** on all others:

```json
"embedding": {
  "name": "trunk",
  "kind": "instance",
  "dim": 1664,
  "model": {
    "name": "vit-carry-spal",
    "weights": "sha256:<hex>"
  },
  "tap": "trunk.final",
  "pooling": "mask_mean",
  "window": {"before": 5, "after": 6},
  "normalization": "none",
  "projection": null,
  "fingerprint": "sha256:<hex>"
}
```

| field | type | rule |
|---|---|---|
| `name` | string | equals the `<name>` segment of the id |
| `kind` | `"instance"` \| `"frame"` | equals the id's last segment |
| `dim` | int ≥ 1 | equals the length of the `feature` axis |
| `model.name` | string | model identifier |
| `model.weights` | string | content identifier of the exact weights, `sha256:<hex>` of the checkpoint file |
| `tap` | string | producer-defined name of where in the model the vector comes from |
| `pooling` | string | how spatial positions were reduced; registered values `mask_mean`, `bbox_mean`, `grid_mean`, `token`; others allowed |
| `window` | `{before, after}` ints ≥ 0 | frames before/after *t* whose pixels the vector for *t* may depend on through the model's clip (nominal temporal receptive field) |
| `normalization` | string | `none`, `l2`, or producer-defined |
| `projection` | null or object `{kind, input_dim, basis}` | applied post-tap reduction (e.g. `{"kind":"pca","input_dim":1664,"basis":"sha256:<hex>"}`); `dim` is then the output dim |
| `fingerprint` | string | `sha256:` of the canonical JSON (sorted keys, no whitespace) of the descriptor **excluding** `name`, `kind` and `fingerprint` |

Validator rules (new):

- descriptor present iff id is `jabs.embedding.*`; id shape is exactly
  `jabs.embedding.<name>.(instance|frame)`.
- axes exactly `[frame, slot, feature]` (instance) or `[frame, feature]` (frame).
- `name`/`kind`/`dim` agree with the id and shape.
- dtype ∈ {float16, float32}; missing policy `nan`.
- `fingerprint` recomputes.
- If both `<name>.instance` and `<name>.frame` exist, their descriptors may differ only in `kind`,
  `pooling` and `fingerprint` (same model, weights, tap, window, normalization, projection, dim).
  *(Claude's call: a name means one tap; a different tap needs a different name.)*
- Deep check (opt-in, `deep=True`): no partially-NaN rows; and, **when `jabs.pose.slot_occupied`
  exists**, every per-instance row where it is false is all-NaN (error). A `.instance` embedding
  is legal in a file without `jabs.pose.slot_occupied`; the check is then skipped. *(Claude's
  call.)*

Semantics of the fingerprint: it is the consumer's comparability key. Two embeddings with equal
fingerprints are interchangeable for training/prediction; names are labels only. Pooling is in the
fingerprint, so an instance and a frame embedding never share one.

**Carry/recurrent models.** For ViT-Carry-SPAL, frame *t*'s vector can also depend (weakly) on
earlier windows through the carried prior. `window` records the nominal clip receptive field only;
recurrent dependence is the producer's to describe in provenance `parameters`. *(Claude's call:
keep `window` a simple, comparable number; don't model unbounded recurrence.)*

Provenance: each embedding component should carry a provenance record (as every component should);
the descriptor is the normative, validated subset; provenance holds everything else (git commit,
inference parameters, carry details).

### Storage layout: producer-selectable

*(Claude's calls in this section, settled after builder review.)*

- `Component` gains a **new** optional field `requested_layout` (`storage`, `chunks`,
  `compression`, `compression_opts`, same vocabulary as the manifest `layout` field). The existing
  `Component.layout` keeps its current meaning (what the file actually has; the writer's choice
  wins).
- A request is honored **only for `jabs.embedding.*` and foreign (non-`jabs`) components**. A
  request on any other `jabs.*` component is a `ValueError` — `jabs.pose.*` stays contiguous for
  the Hub's byte-range overlay, and segmentation keeps its policy.
- When honored, the writer applies it and records the result in the manifest `layout`. A frame
  chunk larger than the frame axis (short or clipped data) is clamped to the axis length, minimum
  1; the manifest records the clamped value.
- Round trip: for `jabs.embedding.*` and foreign components only, the reader populates
  `requested_layout` from the manifest `layout`, so read-then-write keeps their layout. All other
  components are re-laid-out by policy exactly as today, so files without embeddings behave
  unchanged. No manifest schema change for this.
- Validation of a requested layout: any legal HDF5 chunk shape is accepted (rank matches, each
  extent ≥ 1); contiguous + compression is rejected (HDF5 requires chunking for filters). Allowed
  compression: `none`, `gzip` (opts 0–9), `lzf`.
- **Default for `jabs.embedding.*`**: chunked along `frame`, all other axes whole, **uncompressed**
  (learned float vectors compress poorly; gzip costs CPU for little), frame chunk chosen so a chunk
  is ≈1 MiB (≥ 1 frame). ADR 0003 adds an "embedding-scale" row to ADR 0002's storage table
  stating this.
- The id-prefix rule stays the default for everything else.

### Scope clarification (ADR 0003 text, amends 0002 goal 1)

0002 goal 1 says "Labels, features, predictions and classifiers stay where they are." 0003 states:
a per-frame representation produced **by a model at inference time from video** is inference output
and belongs in the pose file; features **computed by JABS** from pose file contents stay out.

### Clipping, round-trip, reading

- Clipping (when a clip tool exists): no new rules; the generic frame-axis slice applies. ADR 0003
  states the descriptor is unchanged by clipping (it describes the model, not the frames).
- Read-then-write must preserve components, descriptors and (for embedding/foreign components)
  layouts.
- `read_component(path, "jabs.embedding.trunk.instance", frames=slice(...))` works as for any
  dense component; float16 comes back as float16 (consumer upcasts).
- A small helper `list_embeddings(manifest) -> list[EmbeddingInfo]` (name, kind, dim, fingerprint,
  component id) so consumers don't parse ids. *(Claude's call.)*

### Conformance fixtures

Add valid fixtures (one instance + one frame embedding, float16; two named embeddings) and one
invalid fixture per new rule, following the existing one-fixture-per-error-check pattern
(`7c92024b`).

## Left to the builder

- Exact Python types (`EmbeddingDescriptor` dataclass shape, where it lives in `types.py`).
- Canonical-JSON details for the fingerprint beyond "sorted keys, compact separators, UTF-8".
- Chunk-size arithmetic for the ≈1 MiB default.
- ADR 0003 prose, examples and an example manifest snippet.
- Whether the validator's id-shape rule lives in the schema (regex/pattern) or code, as long as both
  `validate()` and the writer enforce it.
- Rebasing onto `main` once PRs #596/#598 merge.
