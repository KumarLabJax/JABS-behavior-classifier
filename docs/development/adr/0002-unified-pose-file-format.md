# ADR 0002: Unified Pose File Format

## Status

Proposed

## Context

The pose format has been extended seven times. Each extension was reasonable in isolation; the
accumulated result is that every consumer must know which version it is reading, and the version is
encoded in the *filename*. Reading a pose file today means dispatching to one of seven classes
(`pose_est_v2.py` … `pose_est_v8.py`), each of which knows a slightly different dialect of the same
idea.

The dialect differences are not cosmetic:

- **Coordinate order is per-dataset.** `poseest/points` is (y, x); `static_objects/corners` is
  (x, y); `poseest/seg_data` is (x, y); `dynamic_objects/fecal_boli/points` is (y, x). The
  producer's own specification says of dynamic objects that "x,y or y,x sorting may be different
  per-static object."
- **Padding collides with data.** `poseest/points` pads with `0`, which is a valid pixel
  coordinate. Validity is therefore recovered from `confidence > 0`, and then re-derived a second
  time by each consumer applying its own threshold — JABS uses `MINIMUM_CONFIDENCE = 0.3`.
- **One integer carries three meanings.** In `poseest/instance_embed_id`, `0` means invalid,
  `1…N` means identity, and `> N` means "valid instance we could not assign to an identity."
- **The coordinate space is not recorded.** No version stores frame width or height. A pose file
  cannot state the pixel space its own coordinates live in, so it cannot be validated, rendered or
  sanity-checked without the video beside it.

### The tell: one transform, written three times

Because slots in `points` are detector order rather than identity, every consumer that wants
per-animal data performs the same scatter-and-flip. That code exists independently in three
repositories:

| Location | What it does |
|---|---|
| `src/jabs/pose_estimation/pose_est_v4.py` | scatters by `instance_embed_id`, flips (y,x)→(x,y), NaNs sub-threshold points |
| `src/jabs/pose_estimation/pose_est_v4.py` (`_cache_poses`) | writes the result to a second HDF5 file per video |
| `JABS-postprocess/.../analysis_utils/clip_utils.py` | reimplements the same scatter and flip |

JABS additionally maintains an entire per-video cache file whose only purpose is to memoize that
transform, plus `jabs/cache/pose_attribute_cache.json` to memoize the small attributes that a
project-load scan would otherwise reread from every pose file.

### What the files actually contain

Measured from a one-hour single-mouse `pose_est_v6.h5` (108,150 frames, 33 MB on disk):

| dataset | stored | uncompressed | read by |
|---|---:|---:|---|
| `poseest/points` | 5.19 MB | 5.19 MB | everything |
| `poseest/confidence` | 5.19 MB | 5.19 MB | everything |
| `poseest/id_mask` | 0.11 MB | 0.11 MB | everything (v4+) |
| `poseest/instance_embed_id` | 0.43 MB | 0.43 MB | everything (v4+) |
| `poseest/instance_id_center` | <0.01 MB | <0.01 MB | JABS (shape only); JABS-postprocess (data) |
| `poseest/identity_embeds` | 0.43 MB | 0.43 MB | JABS-postprocess (`attrs["network"]`) |
| `poseest/instance_track_id` | 0.43 MB | 0.43 MB | clip utilities (v3 path) |
| `poseest/instance_embedding` | 5.19 MB | 5.19 MB | nothing outside producer tests |
| `poseest/seg_data` | 16.52 MB | **1,184 MB** | segmentation features |

Two observations shaped this design. `instance_embedding` is 5.19 MB per mouse-hour that no
consumer reads — but `identity_embeds` and `instance_id_center`, which look equally vestigial from
inside JABS, are read by `JABS-postprocess` for cross-video identity linking. A redesign that
trusted one repository's usage would have deleted them.

And `seg_data` expands 72× on read, because the array's shape must accommodate the single longest
contour in the whole video:

| file | real data | dense uncompressed | stored |
|---|---:|---:|---:|
| v6 · 1 mouse · 1 hour | 7.3% | 1,184 MB | 16.5 MB |
| v8 · 3 mice · 60 seconds | 0.82% | 1,123 MB | 3.4 MB |

### Identity slots are already aligned — the format just never promised it

| file | version | slots | valid instances in the wrong slot |
|---|---|---:|---|
| 1-mouse 1 hr (JABS-pose) | v6 | 1 | 0 / 108,150 |
| 3-mouse clips ×3 (JABS-pose) | v8 | 3 | 0 / ~3,600 each |
| multi-mouse test file | v6 | 5 | 1,102 / 2,444 |
| multi-mouse samples ×2 | v5 | 5 | 10,914 / 14,393 |

Current inference recycles instance ids (`"recycle_instance_ids": true` in v8
`model_metadata_json`), so slot *k* already holds identity *k*. Older multi-animal files do not.
Because the format never guaranteed it, every consumer scatters defensively — and against v5/v6
files it genuinely must.

### Requirements

Hard requirements, agreed with the JABS tech lead:

1. Readable on HPC with `h5py` alone — no service, no special runtime.
2. Partial / range reads over cloud object storage, for the in-browser pose overlay.
3. One file per video.
4. Efficient whole-file sequential reads for training and feature extraction.

Nice to have: a community-standard archival form (satisfied by an NWB converter, lossy accepted —
see `docs/development/jabs-nwb-format.md`), and enough self-description that a collaborator
without JABS can find the keypoints.

Requirements 2 and 4 pull in opposite directions and together decide the chunking policy.
Requirement 1 plus "one file" rules out a directory-based store such as Zarr.

## Decision

**One format, one reader, forever additive.** A file declares its contents in a manifest; a reader
asks what a file *contains*, never how old it is. Data JABS does not understand is a first-class
citizen with declared axes, not an escape hatch.

### Design goals

1. **Scope is the per-video pose file.** Inference output only, read-only once written. Labels,
   features, predictions and classifiers stay where they are.
2. **One HDF5 file**, readable with `h5py` and nothing else.
3. **Identified by root attributes, never by filename.** `<video>_pose.h5` is a recommended
   convention that carries no meaning.
4. **Additive-only, with a declared contents manifest.** Components may be added; never removed,
   renamed or re-meaninged. `schema_revision` is diagnostic — **no reader may branch on it**.
5. **Wrong data is corrected by regenerating files.** Files are read-only and reproducible from
   video, so a semantic error is fixed by re-running rather than by mutation or version-gating.
   Deprecate-in-place stays available case by case.
6. **Two tiers of extension.** First-class components declare axes, dtype, units and missing-value
   policy in the manifest. Tier-1 *attachments* declare nothing; any tool that transforms the file
   preserves them verbatim and records the transformation, and never silently drops them.
7. **`jabs.*` is reserved; everyone else uses reverse-DNS.** JABS's own optional data —
   segmentation, static objects, dynamic objects — are components like any other.
8. **Keypoints are frame-major with identity-aligned slots.** Slot *k* means identity *k*. Instances
   the identity step could not assign live in tail slots, so the pre-identity view survives and
   identity resolution never becomes a one-way door.
9. **Identity ids are slot indices, zero-based, with no sentinel.** Presence is a mask.
10. **The skeleton is declared in the file, and required.** Arbitrary keypoint sets are legal;
    multiple skeletons in one file are not precluded.
11. **Absence is stated, never encoded in-band.** Floats use `NaN`; everything else uses an explicit
    mask or an explicit length. Sentinels are banned. Keypoint validity is *both* raw confidence and
    a producer mask, with the threshold recorded.
12. **Each component kind has one baseline encoding.** Readers must implement the baseline and may
    implement others; producers are strongly encouraged to write it and may override per run.
13. **Keypoint-scale data is uncompressed and chunked along frames**; segmentation-scale data is
    chunked and compressed.
14. **The file knows its video and its time base**, including frame width and height.
15. **Provenance is per-component, with an append-only history** that records what a converter had
    to invent.
16. **A valid file is a well-formed manifest.** Every component is optional; consumers declare their
    own requirements. `jabs-io` owns the specification, the validator and the conformance fixtures.

## Specification

Revision 1. All paths are absolute HDF5 paths. All JSON documents are UTF-8.

### File identification

Root group attributes:

| attribute | HDF5 type | value |
|---|---|---|
| `jabs_format` | variable-length UTF-8 string scalar | `"jabs.pose-file"` |
| `schema_revision` | `int32` scalar | `1` |

A reader identifies the format by `jabs_format` alone. `schema_revision` is recorded for
diagnostics and provenance; **branching on it is a specification violation**. Legacy files are
distinguishable in the same header read by the presence of `poseest/version` and the absence of
`jabs_format`.

### Top-level layout

```
/                                   attrs: jabs_format, schema_revision
/manifest                           scalar UTF-8 string — contents declaration (JSON)
/provenance                         scalar UTF-8 string — provenance + history (JSON)
/jabs/pose/…                        ┐
/jabs/identity/…                    │
/jabs/segmentation/…                │ first-class components
/jabs/static_objects/…              │
/jabs/dynamic_objects/…             │
/jabs/time/…                        ┘
/org.example.thing/…                foreign first-class components (reverse-DNS)
/attachments/…                      tier-1 opaque payloads
```

### Component identifiers and namespaces

A component id is a dot-separated lowercase path with at least two segments:

```
^[a-z0-9]([a-z0-9_-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9_-]*[a-z0-9])?)+$
```

The first segment is the namespace root. `jabs` is reserved for components defined by this
specification. Any other producer **must** use a reverse-DNS root of at least two segments
(`org.jax.gait.stride_length`, `edu.example.lab.whisker_angle`). This rule is what makes
"add a component" a safe operation for a party that has never spoken to the JABS maintainers.

### The manifest

`/manifest` is a scalar dataset holding one JSON object.

#### JSON Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://jabs.jax.org/schema/pose-file/manifest/1",
  "title": "JABS pose file manifest",
  "type": "object",
  "required": ["format", "schema_revision", "dimensions", "video", "components"],
  "additionalProperties": false,
  "properties": {
    "format": { "const": "jabs.pose-file" },
    "schema_revision": { "type": "integer", "minimum": 1 },
    "created": { "type": "string", "format": "date-time" },
    "dimensions": {
      "type": "object",
      "required": ["frame", "slot", "identity"],
      "properties": {
        "frame": { "type": "integer", "minimum": 0 },
        "slot": { "type": "integer", "minimum": 0 },
        "identity": { "type": "integer", "minimum": 0 }
      },
      "additionalProperties": { "type": "integer", "minimum": 0 }
    },
    "video": { "$ref": "#/$defs/video" },
    "skeletons": {
      "type": "object",
      "propertyNames": { "$ref": "#/$defs/componentId" },
      "additionalProperties": { "$ref": "#/$defs/skeleton" }
    },
    "components": { "type": "array", "items": { "$ref": "#/$defs/component" } },
    "attachments": { "type": "array", "items": { "$ref": "#/$defs/attachment" } },
    "extra": { "type": "object" }
  },

  "$defs": {
    "componentId": {
      "type": "string",
      "pattern": "^[a-z0-9]([a-z0-9_-]*[a-z0-9])?(\\.[a-z0-9]([a-z0-9_-]*[a-z0-9])?)+$"
    },
    "hdf5Path": { "type": "string", "pattern": "^/[^\\0]*$" },
    "dtype": {
      "enum": ["float32", "float64", "int8", "int16", "int32", "int64",
               "uint8", "uint16", "uint32", "uint64", "bool", "string"]
    },

    "video": {
      "type": "object",
      "required": ["frame_count", "width", "height", "fps"],
      "additionalProperties": false,
      "properties": {
        "frame_count": { "type": "integer", "minimum": 0 },
        "width":  { "type": ["integer", "null"], "minimum": 1 },
        "height": { "type": ["integer", "null"], "minimum": 1 },
        "fps": { "type": ["number", "null"], "exclusiveMinimum": 0 },
        "cm_per_pixel": { "type": ["number", "null"], "exclusiveMinimum": 0 },
        "cm_per_pixel_source": {
          "enum": ["corner_detection", "manually_set", "default_alignment", null]
        },
        "start_time": { "type": ["string", "null"], "format": "date-time" },
        "filename": { "type": ["string", "null"] },
        "content_hash": { "type": ["string", "null"] },
        "clip_of": {
          "type": "object",
          "required": ["frame_offset"],
          "additionalProperties": false,
          "properties": {
            "frame_offset": { "type": "integer", "minimum": 0 },
            "content_hash": { "type": ["string", "null"] },
            "filename": { "type": ["string", "null"] }
          }
        }
      }
    },

    "skeleton": {
      "type": "object",
      "required": ["body_parts", "edges"],
      "additionalProperties": false,
      "properties": {
        "body_parts": {
          "type": "array", "minItems": 1,
          "items": { "type": "string", "minLength": 1 }
        },
        "edges": {
          "type": "array",
          "items": {
            "type": "array", "minItems": 2, "maxItems": 2,
            "items": { "type": "integer", "minimum": 0 }
          }
        },
        "description": { "type": "string" }
      }
    },

    "encoding": {
      "oneOf": [
        {
          "type": "object", "required": ["kind"], "additionalProperties": false,
          "properties": { "kind": { "const": "dense" } }
        },
        {
          "type": "object",
          "required": ["kind", "group_offsets", "instance_offsets"],
          "additionalProperties": false,
          "properties": {
            "kind": { "const": "ragged" },
            "group_offsets": { "$ref": "#/$defs/hdf5Path" },
            "instance_offsets": { "$ref": "#/$defs/hdf5Path" }
          }
        },
        {
          "type": "object",
          "required": ["kind", "instance_offsets"],
          "additionalProperties": false,
          "properties": {
            "kind": { "const": "rle" },
            "instance_offsets": { "$ref": "#/$defs/hdf5Path" },
            "order": { "enum": ["column-major", "row-major"] }
          }
        }
      ]
    },

    "missing": {
      "oneOf": [
        { "type": "object", "required": ["policy"], "additionalProperties": false,
          "properties": { "policy": { "enum": ["none", "nan"] } } },
        { "type": "object", "required": ["policy", "mask"], "additionalProperties": false,
          "properties": { "policy": { "const": "mask" },
                          "mask": { "$ref": "#/$defs/componentId" } } },
        { "type": "object", "required": ["policy", "length"], "additionalProperties": false,
          "properties": { "policy": { "const": "length" },
                          "length": { "$ref": "#/$defs/componentId" } } }
      ]
    },

    "sparse": {
      "type": "object",
      "required": ["index"],
      "additionalProperties": false,
      "properties": { "index": { "$ref": "#/$defs/componentId" } }
    },

    "layout": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "storage": { "enum": ["contiguous", "chunked"] },
        "chunks": { "type": "array", "items": { "type": "integer", "minimum": 1 } },
        "compression": { "enum": ["none", "gzip", "lzf", "szip"] },
        "compression_opts": { "type": ["integer", "null"] }
      }
    },

    "component": {
      "type": "object",
      "required": ["id", "path", "axes", "dtype", "shape", "encoding", "missing"],
      "additionalProperties": false,
      "allOf": [
        {
          "if": {
            "required": ["axes"],
            "properties": { "axes": { "contains": { "const": "coord" } } }
          },
          "then": { "required": ["units", "coord_order"] }
        },
        {
          "if": {
            "required": ["axes"],
            "properties": { "axes": { "contains": { "const": "sample" } } }
          },
          "then": { "required": ["sparse"] }
        }
      ],
      "properties": {
        "id": { "$ref": "#/$defs/componentId" },
        "path": { "$ref": "#/$defs/hdf5Path" },
        "axes": { "type": "array", "minItems": 1, "items": { "type": "string" } },
        "dtype": { "$ref": "#/$defs/dtype" },
        "shape": { "type": "array", "items": { "type": "integer", "minimum": 0 } },
        "units": { "enum": ["pixel", "cm", "second", "frame", "radian", "unitless"] },
        "coord_order": { "enum": ["xy", "yx"] },
        "encoding": { "$ref": "#/$defs/encoding" },
        "missing": { "$ref": "#/$defs/missing" },
        "sparse": { "$ref": "#/$defs/sparse" },
        "layout": { "$ref": "#/$defs/layout" },
        "skeleton": { "$ref": "#/$defs/componentId" },
        "provenance": { "type": "string" },
        "description": { "type": "string" },
        "extra": { "type": "object" }
      }
    },

    "attachment": {
      "type": "object",
      "required": ["path"],
      "additionalProperties": false,
      "properties": {
        "path": { "$ref": "#/$defs/hdf5Path" },
        "description": { "type": "string" },
        "content_type": { "type": "string" }
      }
    }
  }
}
```

`extra` exists on the manifest root and on every component so that a producer can attach
namespaced metadata without the manifest failing validation. It is the manifest's own extension
point, and it is why `additionalProperties` can safely be `false` everywhere else.

#### Axes

`axes` names each dimension of the array in order. The axis vocabulary defined by this revision:

| axis | meaning |
|---|---|
| `frame` | video frame index; an axis named `frame` **always** has length `dimensions.frame` |
| `sample` | index into a sparse component's own samples; its mapping to frames is `sparse.index` |
| `slot` | instance slot; slot *k* is identity *k* for *k* < `dimensions.identity` |
| `identity` | assigned identity, used by per-identity arrays that have no frame axis |
| `keypoint` | index into the referenced skeleton's `body_parts` |
| `coord` | coordinate pair, ordered by `coord_order` |
| `point` | a point within a contour or polygon |
| `contour` | a contour within an instance |
| `object` | an instance of a static or dynamic object |
| `corner` | the two corners of a bounding box |
| `embedding` | identity-embedding dimension |
| `run` | a run in an RLE encoding |

A foreign component may introduce its own axis names. Tooling that subsets a frame range operates
on any component whose `axes` contains `frame` (slice the axis) or `sample` (filter the samples
whose `sparse.index` value falls in range, and rewrite the index), regardless of namespace — which
is what makes extensions survive clipping.

`frame` and `sample` are deliberately distinct. A sparse component's data axis is `sample`, and the
index component that maps samples to frame numbers has axes `["sample"]` with `units: "frame"` — it
holds frame *numbers*, not per-frame values, so slicing it by a frame range would be meaningless.

### Dimensions

| name | meaning |
|---|---|
| `frame` | number of video frames |
| `slot` | number of instance slots, ≥ `identity` |
| `identity` | number of assigned long-term identities |

Slots `[0, identity)` are the assigned identities, in identity order. Slots
`[identity, slot)` hold **unassigned instances** — detections that pose inference produced but
identity resolution could not assign. There is no sentinel and no separate "unassigned" marker: a
slot's meaning is its index, and whether anything is in it on a given frame is
`jabs.pose.slot_occupied`.

The occupancy mask is named for the slot rather than the identity on purpose. A tail slot never
holds an identity, so a mask called `identity_present` would be asserting something false about
exactly the slots that motivated keeping them. **`slot_occupied` is defined as: pose inference
placed an instance in this slot on this frame.** It is a statement about detection, not about
quality — it says nothing about how many keypoints were localized or how confident they were.

### Skeletons

```json
"skeletons": {
  "jabs.mouse12": {
    "description": "JABS 12-keypoint mouse skeleton",
    "body_parts": [
      "NOSE", "LEFT_EAR", "RIGHT_EAR", "BASE_NECK",
      "LEFT_FRONT_PAW", "RIGHT_FRONT_PAW", "CENTER_SPINE",
      "LEFT_REAR_PAW", "RIGHT_REAR_PAW",
      "BASE_TAIL", "MID_TAIL", "TIP_TAIL"
    ],
    "edges": [
      [4, 6], [6, 5],
      [7, 9], [9, 8],
      [0, 3], [3, 6], [6, 9], [9, 10], [10, 11],
      [1, 0], [0, 2]
    ]
  }
}
```

The 11 edges are `PoseEstimation.FULL_CONNECTED_SEGMENTS` expanded from polylines into pairs.
Rendering an edge if and only if both endpoints are valid produces output identical to today's
`gen_line_fragments`, which exists solely to split a polyline at missing keypoints — so that helper
becomes unnecessary.

Multiple skeletons are legal in one file, and each keypoint component names the one it uses.
`PoseEstimation.NVSN_CONNECTED_SEGMENTS` — the reduced skeleton for the Envision Hydra model —
is the case that already exists.

### Encodings

Each component kind declares a **baseline** encoding. A conforming reader must implement the
baseline for every `jabs.*` component it claims to support, and may implement the alternatives. A
producer should write the baseline unless explicitly configured otherwise for a run. `jabs-io`
implements all defined encodings and normalizes them on read, so the plurality is invisible to
anyone using the reference library.

| kind | baseline for | notes |
|---|---|---|
| `dense` | every component defined here | a plain N-dimensional array |
| `ragged` | — | two-level CSR, see below |
| `rle` | — | run-length masks; loses contour hierarchy |

**The payload always lives at the component's `path`**, whatever the encoding. `dtype` and `shape`
describe that dataset. An encoding object names only its *index* datasets, never the payload — so
there is exactly one place a validator looks for a component's data.

**Ragged.** The payload at `path` holds all leaf items concatenated. `group_offsets` is a CSR index
into the payload; `instance_offsets` is a CSR index into `group_offsets`, in row-major
`(frame, slot)` order. Decoding, precisely:

```
group g          spans payload[group_offsets[g] : group_offsets[g + 1]]
instance (f, s)  owns groups g for g in [instance_offsets[f*S + s], instance_offsets[f*S + s + 1])
```

Note the index spaces: `instance_offsets` holds **group indices**, and `group_offsets` holds
**payload indices**. Applying `group_offsets[...]` to an instance's group range would yield payload
offsets, not the groups themselves — a formula that reads naturally and decodes wrongly, so it is
spelled out here. Both offset arrays are `uint64` and carry one trailing element:
`len(group_offsets) == num_groups + 1` and `len(instance_offsets) == F*S + 1`. No cap on group
count or group length exists.

**RLE.** The payload at `path` holds concatenated run lengths in the given `order` (default
`column-major`, the COCO convention); `instance_offsets` is a `uint64` CSR index into it, in
row-major `(frame, slot)` order, with `len(instance_offsets) == F*S + 1`. Mask dimensions come from
`video.width` / `video.height`, which must therefore be non-null for a file using this encoding.

### Component catalog

Shapes use `F` = frames, `S` = slots, `I` = identities, `K` = keypoints, `E` = embedding
dimension. Every component below is optional.

#### `jabs.pose`

| id | path | axes | shape | dtype | units | missing |
|---|---|---|---|---|---|---|
| `jabs.pose.points` | `/jabs/pose/points` | frame, slot, keypoint, coord | F×S×K×2 | float32 | pixel | `nan` |
| `jabs.pose.confidence` | `/jabs/pose/confidence` | frame, slot, keypoint | F×S×K | float32 | unitless | `none` |
| `jabs.pose.point_valid` | `/jabs/pose/point_valid` | frame, slot, keypoint | F×S×K | bool | — | `none` |
| `jabs.pose.slot_occupied` | `/jabs/pose/slot_occupied` | frame, slot | F×S | bool | — | `none` |
| `jabs.pose.slot_usable` | `/jabs/pose/slot_usable` | frame, slot | F×S | bool | — | `none` |
| `jabs.pose.bbox` | `/jabs/pose/bbox` | frame, slot, corner, coord | F×S×2×2 | float32 | pixel | mask → `slot_occupied` |
| `jabs.pose.tracklet_id` | `/jabs/pose/tracklet_id` | frame, slot | F×S | uint32 | — | mask → `slot_occupied` |

`points` uses `coord_order: "xy"` and carries a `skeleton` reference. `point_valid` is the
producer's recommendation; the threshold that produced it is recorded in that component's
provenance as `parameters.confidence_threshold`. A consumer may ignore the mask and re-threshold
`confidence` itself — that is the entire reason both are present.

`tracklet_id` values are gap-free intervals: a tracklet occupies a contiguous run of frames with no
breaks. This invariant is **normative** here rather than a footnote in a producer document, because
JABS depends on it.

**`slot_usable` is a declared quality gate, and it is not `slot_occupied`.** JABS today computes
`identity_mask` as `sum(point_mask[..., :-2]) >= 3` (`pose_est_v4.py:159-172`) — at least three
keypoints above `MINIMUM_CONFIDENCE`, excluding `MID_TAIL` and `TIP_TAIL`, because the convex hull
needs three points. That is a *consumer requirement*, not a property of the data: it becomes
`_frame_valid` in `features.py:335`, gates centroid velocity, lixit and every window operation, and
forces labels to `NONE` where false (`parallel_workers.py:352`), which decides what is trainable.

Three frame states make the difference concrete — an instance is present, and JABS still treats the
frame as unusable:

| frame state | `slot_occupied` | `slot_usable` |
|---|---|---|
| 2 confident body keypoints | true | false |
| 5 body keypoints, all confidence 0.2 | true | false |
| tail keypoints only, high confidence | true | false |

So the two masks are both present and neither substitutes for the other. `slot_usable` is optional;
its rule is declared in provenance as `parameters` — `confidence_threshold`, `min_valid_keypoints`
and `excluded_keypoints` — which is the same argument this ADR makes about `MINIMUM_CONFIDENCE`
being silently re-derived by every consumer. A consumer that disagrees with the producer's rule can
recompute from `confidence` and knows exactly what it is departing from. A consumer reading a file
where it is absent derives it itself.

#### `jabs.identity`

| id | path | axes | shape | dtype | notes |
|---|---|---|---|---|---|
| `jabs.identity.embeddings` | `/jabs/identity/embeddings` | frame, slot, embedding | F×S×E | float32 | per-instance identity embedding |
| `jabs.identity.centers` | `/jabs/identity/centers` | identity, embedding | I×E | float32 | cluster center per identity; used for cross-video linking |
| `jabs.identity.external_ids` | `/jabs/identity/external_ids` | identity | I | string | optional display names |

`jabs.identity.centers` replaces `poseest/instance_id_center`. It is the component
`JABS-postprocess` reads, and its length — not a separate field — gives the identity count, which
is also stated as `dimensions.identity`.

#### `jabs.segmentation`

Dense baseline:

| id | path | axes | shape | dtype | missing |
|---|---|---|---|---|---|
| `jabs.segmentation.contours` | `/jabs/segmentation/contours` | frame, slot, contour, point, coord | F×S×C×P×2 | int32 | `length` → `contour_length` |
| `jabs.segmentation.contour_count` | `/jabs/segmentation/contour_count` | frame, slot | F×S | uint32 | `none` |
| `jabs.segmentation.contour_length` | `/jabs/segmentation/contour_length` | frame, slot, contour | F×S×C | uint32 | `none` |
| `jabs.segmentation.external_flag` | `/jabs/segmentation/external_flag` | frame, slot, contour | F×S×C | bool | `none` |

Note what changed even in the dense case: validity comes from `contour_count` and
`contour_length`, **not** from `-1` padding. Padding bytes become unspecified rather than
meaningful, which is what brings the dense encoding into line with design goal 11.

Under the ragged encoding, `jabs.segmentation.contours` has `axes: ["point", "coord"]` and its
`path` holds the concatenated contour points; `group_offsets` is
`/jabs/segmentation/contour_offsets` and `instance_offsets` is
`/jabs/segmentation/instance_offsets`. `external_flag` becomes a flat `contour`-axis array.
`contour_count` and `contour_length` are then derivable from the offsets and must be omitted.

There is no `instance_seg_id` and no `longterm_seg_id`. Segmentation is indexed by the same
`(frame, slot)` as pose, so the pose↔segmentation link **is** the slot, and the matching step those
two datasets existed to record disappears.

#### `jabs.static_objects`

`jabs.static_objects.<name>` at `/jabs/static_objects/<name>`, axes `object, point, coord`, shape
O×P×2, `float32`, `pixel`, `coord_order: "xy"`. Names defined by this revision: `corners`
(O=4, P=1), `lixit` (P=1 or 3), `food_hopper` (O=4, P=1). Others may be added without a schema
change.

The uniform `object, point, coord` shape replaces today's per-object shapes and per-object
coordinate orders. `corners` carries no ordering guarantee; `food_hopper` corners are ordered to
form a valid polygon — both stated in the component's `description`.

#### `jabs.dynamic_objects`

Objects predicted on a subset of frames. `jabs.dynamic_objects.<name>.*` at
`/jabs/dynamic_objects/<name>/`:

| id suffix | axes | shape | dtype | units | notes |
|---|---|---|---|---|---|
| `.frame_index` | sample | M | uint32 | frame | the frames on which a prediction was made, strictly increasing |
| `.points` | sample, object, point, coord | M×O×P×2 | float32 | pixel | `sparse.index` → `.frame_index` |
| `.counts` | sample | M | uint32 | unitless | valid object count per sample |

All three use the `sample` axis, not `frame`, and `.points` / `.counts` declare
`sparse: {"index": "<name>.frame_index"}`. `.frame_index` is itself a `sample`-axis component whose
`units` are `frame` — it holds frame numbers. Keeping these axes distinct is what stops a generic
clip tool from slicing an index array as though it were one value per video frame. This generalizes
`dynamic_objects/*/sample_indices` into the mechanism any component may use, which is how the
format expresses "not predicted" without padding.

`fecal_boli` (P=1) is defined by this revision.

#### `jabs.time`

| id | path | axes | shape | dtype | units |
|---|---|---|---|---|---|
| `jabs.time.timestamps` | `/jabs/time/timestamps` | frame | F | float64 | second |

Seconds from the start of the video. Frame index remains the canonical time axis; timestamps exist
so a producer that knows frames were dropped can say so, instead of every consumer computing
`frame / fps` and being quietly wrong. Absolute wall-clock start, when known, is
`video.start_time`.

### Example manifest

```json
{
  "format": "jabs.pose-file",
  "schema_revision": 1,
  "created": "2026-08-26T14:02:11Z",
  "dimensions": { "frame": 108150, "slot": 4, "identity": 3 },
  "video": {
    "frame_count": 108150,
    "width": 800,
    "height": 800,
    "fps": 30.0,
    "cm_per_pixel": 0.13082914,
    "cm_per_pixel_source": "default_alignment",
    "start_time": "2026-03-06T18:00:04Z",
    "filename": "cage_7412.2026-03-06.18.00.mp4",
    "content_hash": "blake2b:9f3c1d…"
  },
  "skeletons": {
    "jabs.mouse12": {
      "description": "JABS 12-keypoint mouse skeleton",
      "body_parts": ["NOSE", "LEFT_EAR", "RIGHT_EAR", "BASE_NECK",
                     "LEFT_FRONT_PAW", "RIGHT_FRONT_PAW", "CENTER_SPINE",
                     "LEFT_REAR_PAW", "RIGHT_REAR_PAW",
                     "BASE_TAIL", "MID_TAIL", "TIP_TAIL"],
      "edges": [[4,6],[6,5],[7,9],[9,8],[0,3],[3,6],[6,9],[9,10],[10,11],[1,0],[0,2]]
    }
  },
  "components": [
    { "id": "jabs.pose.points", "path": "/jabs/pose/points",
      "axes": ["frame", "slot", "keypoint", "coord"],
      "dtype": "float32", "shape": [108150, 4, 12, 2],
      "units": "pixel", "coord_order": "xy",
      "encoding": { "kind": "dense" },
      "missing": { "policy": "nan" },
      "layout": { "storage": "contiguous", "compression": "none" },
      "skeleton": "jabs.mouse12",
      "provenance": "jabs.pose" },

    { "id": "jabs.pose.confidence", "path": "/jabs/pose/confidence",
      "axes": ["frame", "slot", "keypoint"],
      "dtype": "float32", "shape": [108150, 4, 12], "units": "unitless",
      "encoding": { "kind": "dense" }, "missing": { "policy": "none" },
      "provenance": "jabs.pose" },

    { "id": "jabs.pose.point_valid", "path": "/jabs/pose/point_valid",
      "axes": ["frame", "slot", "keypoint"],
      "dtype": "bool", "shape": [108150, 4, 12],
      "encoding": { "kind": "dense" }, "missing": { "policy": "none" },
      "provenance": "jabs.pose" },

    { "id": "jabs.pose.slot_occupied", "path": "/jabs/pose/slot_occupied",
      "axes": ["frame", "slot"],
      "dtype": "bool", "shape": [108150, 4],
      "encoding": { "kind": "dense" }, "missing": { "policy": "none" },
      "provenance": "jabs.pose" },

    { "id": "jabs.pose.slot_usable", "path": "/jabs/pose/slot_usable",
      "axes": ["frame", "slot"],
      "dtype": "bool", "shape": [108150, 4],
      "encoding": { "kind": "dense" }, "missing": { "policy": "none" },
      "provenance": "jabs.pose.slot_usable" },

    { "id": "jabs.identity.centers", "path": "/jabs/identity/centers",
      "axes": ["identity", "embedding"],
      "dtype": "float32", "shape": [3, 1],
      "encoding": { "kind": "dense" }, "missing": { "policy": "none" },
      "provenance": "jabs.identity" }
  ],
  "attachments": []
}
```

### Provenance

`/provenance` is a scalar dataset holding one JSON object with two members: `records`, keyed by an
identifier that components reference, and `history`, an ordered list.

```json
{
  "records": {
    "jabs.pose": {
      "producer": "JABS-pose",
      "version": "2026-06-04_17203557_7",
      "created": "2026-03-07T02:14:55Z",
      "model": {
        "name": "spal_seg_and_pose_multihead_effv2s_fpn_12kp",
        "uri": "mlflow://jabs/models/spal-effv2s/17",
        "git_commit": "3f0163f52a169806068bdf08edfa9abbfa6e9968"
      },
      "parameters": { "confidence_threshold": 0.3, "max_instances": 3 },
      "extra": {}
    },
    "jabs.identity": {
      "producer": "JABS-pose",
      "version": "2026-06-04_17203557_7",
      "created": "2026-03-07T02:31:02Z",
      "algorithm": { "name": "embedding-cluster", "tracklet_stitch": "greedy" },
      "parameters": { "recycle_instance_ids": true }
    },
    "jabs.pose.slot_usable": {
      "producer": "jabs-io",
      "version": "0.9.0",
      "created": "2026-08-26T14:02:11Z",
      "algorithm": { "name": "min-confident-keypoints" },
      "parameters": {
        "confidence_threshold": 0.3,
        "min_valid_keypoints": 3,
        "excluded_keypoints": ["MID_TAIL", "TIP_TAIL"]
      }
    }
  },
  "history": [
    { "operation": "infer", "tool": "JABS-pose", "version": "1.4.0",
      "time": "2026-03-07T02:14:55Z" },
    { "operation": "convert", "tool": "jabs-io", "version": "0.9.0",
      "time": "2026-08-26T14:02:11Z",
      "source": { "format": "pose_est_v6", "content_hash": "blake2b:7ab2…" },
      "synthesized": ["skeletons.jabs.mouse12", "jabs.pose.point_valid",
                      "video.width", "video.height"] }
  ]
}
```

#### JSON Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://jabs.jax.org/schema/pose-file/provenance/1",
  "title": "JABS pose file provenance",
  "type": "object",
  "required": ["records", "history"],
  "additionalProperties": false,
  "properties": {
    "records": {
      "type": "object",
      "additionalProperties": { "$ref": "#/$defs/record" }
    },
    "history": {
      "type": "array",
      "items": { "$ref": "#/$defs/historyEntry" }
    },
    "extra": { "type": "object" }
  },

  "$defs": {
    "reference": {
      "type": "object",
      "required": ["name"],
      "properties": {
        "name": { "type": "string", "minLength": 1 },
        "uri": { "type": "string" },
        "git_commit": { "type": "string", "pattern": "^[0-9a-f]{7,40}$" },
        "version": { "type": "string" }
      },
      "additionalProperties": true
    },
    "record": {
      "type": "object",
      "required": ["producer", "version", "created"],
      "additionalProperties": false,
      "properties": {
        "producer": { "type": "string", "minLength": 1 },
        "version": { "type": "string", "minLength": 1 },
        "created": { "type": "string", "format": "date-time" },
        "model": { "$ref": "#/$defs/reference" },
        "algorithm": { "$ref": "#/$defs/reference" },
        "parameters": { "type": "object" },
        "extra": { "type": "object" }
      }
    },
    "historyEntry": {
      "type": "object",
      "required": ["operation", "tool", "version", "time"],
      "additionalProperties": false,
      "properties": {
        "operation": { "enum": ["infer", "convert", "clip", "merge", "annotate"] },
        "tool": { "type": "string", "minLength": 1 },
        "version": { "type": "string", "minLength": 1 },
        "time": { "type": "string", "format": "date-time" },
        "source": {
          "type": "object",
          "properties": {
            "format": { "type": "string" },
            "content_hash": { "type": ["string", "null"] },
            "filename": { "type": ["string", "null"] }
          },
          "additionalProperties": false
        },
        "synthesized": { "type": "array", "items": { "type": "string" } },
        "dropped": { "type": "array", "items": { "type": "string" } },
        "notes": { "type": "string" },
        "extra": { "type": "object" }
      },
      "allOf": [
        {
          "if": { "properties": { "operation": { "const": "convert" } },
                  "required": ["operation"] },
          "then": { "required": ["source", "synthesized"] }
        }
      ]
    }
  }
}
```

`model.uri` should be resolvable — an MLflow run or registry reference — but is never required. The
`convert` conditional is what makes design goal 5 enforceable rather than aspirational: a converted
file cannot validate without saying where it came from and what was invented.

`synthesized` is **required on any `convert` entry** and lists what the converter had to invent
rather than read. This is what keeps design goal 5 honest: a converted file must be
distinguishable from a natively produced one, or a skeleton the converter assumed and a validity
mask it derived by thresholding both read as ground truth.

The history is written once at creation and never edited. A tool producing a *derived* file
appends its own entry to the copy.

### Attachments

`/attachments/<name>` may hold anything. An attachment declares no axes, so no tool can subset it
correctly. Therefore:

- A tool that copies or transforms a file **must** carry attachments through verbatim.
- A tool that subsets a file **must** record the transformation in `history`, so a consumer can see
  that an attachment may no longer correspond to the file's frame range.
- Silently dropping an attachment is a specification violation. Dropping is irrecoverable;
  preserving with a recorded caveat is not.

Anything that needs to survive subsetting correctly should be a first-class component instead.

### Validation

`jabs-io` ships `validate(path) -> list[Finding]`. Checks:

| check | severity |
|---|---|
| root attributes present, `jabs_format` correct | error |
| `/manifest` parses and validates against the JSON Schema | error |
| `/provenance` parses and validates against the provenance JSON Schema | error |
| every component `path` exists in the file and holds that component's payload | error |
| declared `dtype` and `shape` match the dataset at `path` | error |
| `len(axes) == len(shape)` | error |
| a `mask` or `length` reference resolves to an existing component with a compatible shape | error |
| a component with a `sample` axis declares `sparse`, and vice versa | error |
| `sparse.index` resolves to a 1-D `sample`-axis component, strictly increasing, within `[0, video.frame_count)` | error |
| `sparse.index` length equals the `sample`-axis length of every component referencing it | error |
| ragged `group_offsets` is non-decreasing, starts at 0, ends at `shape[0]` of the payload, length `num_groups + 1` | error |
| ragged/RLE `instance_offsets` is non-decreasing, starts at 0, has length `frame*slot + 1`, and ends at `len(group_offsets) - 1` (ragged) or `shape[0]` of the payload (RLE) | error |
| every component `provenance` id resolves to a record in `/provenance` | error |
| a component whose `axes` contain `coord` declares `units` and `coord_order` | error |
| an RLE component's file has non-null `video.width` and `video.height` | error |
| `skeleton` references resolve; every edge index `< len(body_parts)` | error |
| every keypoint component's `keypoint` axis length equals its skeleton's `body_parts` length | error |
| `dimensions.identity <= dimensions.slot` | error |
| component ids are unique and namespace-well-formed | error |
| a non-`jabs` namespace has a reverse-DNS root of ≥2 segments | error |
| `video.width` / `video.height` non-null | warning |
| declared `layout` matches the dataset's actual HDF5 storage and filters | warning |
| an `/attachments` member has no manifest entry | warning |

**Conformance fixtures ship with the specification** in `packages/jabs-io/tests/data/pose-format/`:
a minimal valid file, one file per defined encoding, one with a sparse component, one with a
foreign component and an attachment, and a set of deliberately invalid files, one per error check
above. Three repositories have independently reimplemented the identity scatter; fixtures are the
cheapest available defense against a fourth.

## Consequences

### Storage layout and chunking

| component scale | layout | rationale |
|---|---|---|
| keypoint-scale — points, confidence, masks, identity, bbox | **contiguous, uncompressed** where the frame count is known at write time; otherwise chunked along `frame` in time-sized chunks (≈1,800 frames), uncompressed | contiguous storage is the only layout under which a frame range really is one byte range; these arrays are ≤ ~48 MB at the worst realistic size, so compression buys little and costs the access pattern |
| segmentation-scale | chunked **and compressed** | the padding is what makes it enormous, and gzip is what removes it |

This is the one place where the format is constrained by a decision made outside this repository.
The JABS Hub's in-browser pose overlay reads a frame window as a byte range; that works today only
because `points` and `confidence` happen to be stored contiguous and uncompressed. The policy turns
that accident into a stated intent.

**What chunking does and does not guarantee.** HDF5 chunks are individually located and need not be
adjacent or in order on disk, so a frame window over a *chunked* dataset is **not** one byte range.
It touches `ceil(w / c) + 1` chunks in the worst case for a window of `w` frames and a chunk of `c`,
plus chunk-index metadata — bounded read amplification, not a single range. Only *contiguous*
storage gives the single-range property, which is why it is the recommendation for keypoint-scale
components whenever the frame count is known when the file is written, as it is for a completed
inference run.

A component's `layout` field records `storage`, `chunks`, `compression` and `compression_opts`, so a
reader can tell which case it is in before deciding how to read. That field is advisory metadata
about the HDF5 layout, and `validate()` warns when it disagrees with the file.

### What disappears

- `pose_attribute_cache.json` — every attribute it memoizes (frame count, identity count, static
  object names, lixit keypoint count, the `cm_per_pixel` flag) is in the manifest, readable in one
  small read without opening any array.
- The identity scatter in `pose_est_v4.py`, the cache writer, and `clip_utils.py`.
- The per-video `*_cache.h5` file, **with a caveat**. It holds three datasets: `points`,
  `point_mask` and `identity_mask`. The first two become free. `identity_mask` is a JABS policy no
  producer computes, so it is either declared as `jabs.pose.slot_usable` or derived at load — and
  deriving it is cheap: `point_valid[..., :-2].sum(-1) >= 3` over a 108,150-frame 3-identity mask
  measures **3.2 ms**, against **629 ms** for the current `np.vectorize` + `fromfunction`
  implementation (199×, identical output). The cache existed for the scatter; this array never
  needed it.
- `poseest/instance_seg_id` and `poseest/longterm_seg_id`.
- `poseest/instance_count`, now the row-sum of `slot_occupied` — matching that field's documented
  meaning ("instances containing at least one non-zero keypoint confidence"). Note this is *not*
  the count JABS uses anywhere; JABS counts usable identities, which is `slot_usable`.
- `gen_line_fragments`, replaced by edges drawn when both endpoints are valid.
- Filename version parsing, and the `*_pose_est_v*.h5` glob in `ProjectPaths`.
- `poseest/instance_embedding`, which no consumer reads.

### Conversion

| source | conversion |
|---|---|
| v4–v8 | scatter to identity-aligned slots once; flip (y,x)→(x,y); derive `point_valid` at confidence > 0.3; write `jabs.mouse12` explicitly; carry embeddings, centers, tracklets and bbox forward; segmentation as the dense baseline with `contour_count` / `contour_length` computed from the `-1` padding |
| v2 | single animal → one slot; `slot_occupied` derived from confidence > 0 |
| v3 | tracklets only: `tracklet_id` populated, `dimensions.identity = 0`, no long-term identity. The file is valid and honest; JABS declines it for identity-requiring work |

Every conversion writes a `convert` history entry naming what it synthesized.

**Frame dimensions** are the awkward case: required for the file to be self-contained, and present
in no legacy file. The converter reads them from the video when it can; when it cannot, it writes
`null` and records why. An explicit unknown is recoverable; a plausible default is not.

**NWB export** remains a converter, lossy by design — it drops segmentation, as the current one
does — and records what it dropped.

### Positive

- One reader. No consumer asks how old a file is, only what it contains.
- A pose file is interpretable on its own: coordinate space, skeleton, scale and source video are
  all in the file.
- Foreign data can be subset, validated and carried correctly by tools that know nothing about its
  meaning.
- Adding a component is data, not a format revision, and requires no reader change anywhere.
- Arbitrary keypoint sets and multiple skeletons are supported without a format change, which the
  Envision Hydra model already needs.

### Negative and trade-offs

- Every consumer must be rewritten once and every existing file converted. There is no incremental
  path; that is the cost of ending version dispatch.
- Per-identity whole-video reads become strided rather than contiguous. Cheap on HPC, where that
  access pattern lives; wasteful over a network.
- Permitting multiple encodings reintroduces a variant axis. The baseline rule contains it, but
  "baseline plus optional" is a promise that needs enforcing in review.
- Uncompressed keypoint arrays make files larger on disk than they need to be, deliberately.
- The manifest is a second source of truth about shapes and dtypes and can disagree with the
  arrays. `validate()` exists because of this.

### Risks

- **The identity-alignment guarantee rests on a small sample** — three v8 clips from one model, and
  a v6 file with a single slot. It is specified as a requirement on producers, not as an observed
  property, but a producer that violates it silently corrupts identity.
- **Converters become load-bearing.** Every historical file's fidelity depends on one
  implementation, including the parts it must synthesize.
- **Optional-everything can drift into unusable-in-practice** if producers omit components that
  consumers assume. Named profiles are the escape hatch if that happens.

## Alternatives considered

**Keep versioning, promise compatibility within a major version.** The familiar option, and the one
that reproduces the current situation the first time someone bumps. Seven versions is the evidence.

**Zarr or another chunked store.** Better cloud semantics and native partial reads, rejected on the
single-file and `h5py`-only requirements.

**NWB with a JABS extension as the native form.** Attractive for archival and interoperability, and
an `ndx-pose` adapter already exists. Rejected as the *native* form: segmentation does not survive,
the dependency footprint is heavy for a file read by shell scripts on a cluster, and the archival
requirement is a nice-to-have that a converter satisfies.

**Identity-major keypoint layout** (`[identity][frame][keypoint][coord]`). Matches what every
consumer builds in memory and makes per-identity reads contiguous. Rejected: a time window costs N
ranges, and unassigned instances have no lane — retaining them needs a duplicate component costing
up to +31 MB per 3-mouse hour, and dropping them makes identity resolution a one-way door.

**Ragged segmentation as the baseline.** Measurement showed it does not reduce stored size — gzip
already compresses runs of `-1` to almost nothing, so ragged's ~87 MB of real int32 points for a
mouse-hour is *larger* than the 16.5 MB the padded array occupies. Its real gains are a 13× smaller
uncompressed footprint, per-frame random access, and the removal of shape-baked caps. Permitted,
not mandated.

**Bake validity into `points` and drop `confidence`.** Smallest files, simplest readers. Rejected:
it makes one threshold permanent and destroys the ability to re-evaluate it.

## Open questions

1. **Do the recording devices drop frames in practice?** If they do, `jabs.time.timestamps` should
   probably become required for natively produced files — and the current format has been hiding a
   data-integrity issue rather than recording it.
2. **Is the v3 conversion policy right?** A tracklets-only file with no long-term identity is
   honest, but it may be more useful to refuse conversion than to produce files most tooling will
   reject downstream.
3. **Frame dimensions for archived poses whose video is gone.** Explicit `null` is specified;
   whether that is acceptable, or whether conversion should simply require the video, is a call for
   whoever runs the migration.
4. **Should the validity masks be bit-packed?** One byte per keypoint per frame is ~3.9 MB per
   mouse-hour; bit-packed it is ~0.5 MB. Worth the reader complexity, or not?
5. **How lossy may the NWB converter be, and is DANDI publication a goal?** The answer decides
   whether the loss must be enumerated in this specification or merely logged.
6. **Does identity re-resolution tooling exist, or is it planned?** Retaining unassigned instances
   in tail slots is justified by it. If nothing will ever re-run identity assignment, that is dead
   weight by the same standard this ADR applies to `instance_embedding`.
7. **Should producers be required to write `jabs.pose.slot_usable`, or is deriving it at load the
   expected path?** It is optional here, which means two consumers can still disagree about which
   frames are usable — the very problem the component exists to fix. Requiring it pushes a JABS
   consumer policy onto every producer, including foreign ones.
8. **Is `slot_usable` the right name?** The review proposed `identity_usable`; this draft renamed it
   for consistency with `slot_occupied`, since the mask is slot-indexed and tail slots hold no
   identity. The rule it encodes is really "enough confident non-tail keypoints to compute shape
   features", which neither name says.
9. **Does `jabs.identity.embeddings` need its network name preserved as a first-class field?**
   `JABS-postprocess` reads `identity_embeds.attrs["network"]`; this ADR puts it in provenance as
   `model.name`, which is a rename that consumer will have to follow.

## References

- `docs/pose/file_format.md` in `mouse-tracking-runtime` — the current (v7) producer specification
- `docs/development/jabs-nwb-format.md` — the existing NWB export
- `src/jabs/pose_estimation/pose_est_v2.py` … `pose_est_v8.py` — the seven readers this replaces
- `packages/jabs-core/src/jabs/core/abstract/pose_est.py` — `KeypointIndex`,
  `FULL_CONNECTED_SEGMENTS`, `NVSN_CONNECTED_SEGMENTS`, `MINIMUM_CONFIDENCE`
- `packages/jabs-core/src/jabs/core/types/pose.py` — the `PoseData` prototype, which anticipated
  much of this model and is reviewed here as a candidate rather than treated as a baseline
- `JABS-postprocess/src/jabs_postprocess/utils/project_utils.py` — cross-video identity linking,
  the consumer of `instance_id_center` and `identity_embeds`

**Measurement provenance.** v6 figures come from a one-hour single-mouse `pose_est_v6.h5`
(108,150 frames) inspected with `h5py`. v8 dtype, layout and identity-alignment figures come from
three 3-animal 1,800-frame clips in one seizure dataset, produced by one model. Segmentation
occupancy is extrapolated from 500 sampled frames (v6) and 20 (v8); the ragged estimate counts
contour points only and excludes the offset arrays. Three-animal hour figures are extrapolated from
measured per-frame sizes rather than measured directly.
