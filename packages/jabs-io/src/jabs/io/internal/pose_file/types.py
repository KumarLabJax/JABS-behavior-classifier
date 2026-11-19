"""Domain types for the JABS pose file format (ADR 0002).

These types mirror the *file*: a :class:`PoseFile` is a set of
:class:`Component` objects, each carrying its array alongside the declarations
the manifest records — axes, units, coordinate order and missing-value policy.
That is deliberately different from :class:`jabs.core.types.pose.PoseData`,
which is the science-facing view (identity-major, no slots, no provenance).
Mapping between the two belongs with the JABS integration, not here.

Validation lives in ``__post_init__`` so an invalid file cannot be constructed
in memory, let alone written. The writer relies on this: it builds and
validates before opening the file, because ``h5py.File(path, "w")`` truncates
at open time.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from jabs.io.internal.pose_file.blocks import _STRING_KINDS, ComponentData
from jabs.io.internal.pose_file.embeddings import (
    EMBEDDING_AXES,
    EMBEDDING_DTYPES,
    EmbeddingDescriptor,
    is_embedding_id,
    parse_embedding_id,
)

if TYPE_CHECKING:
    # Annotation only: alternates imports this module for its dtype rules.
    from jabs.io.internal.pose_file.alternates import Alternate

# A dot-separated lowercase path of at least two segments. The first segment is
# the namespace root; `jabs` is reserved by the specification.
COMPONENT_ID_RE = re.compile(
    r"^[a-z0-9]([a-z0-9_-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9_-]*[a-z0-9])?)+$"
)

# The dtypes the manifest schema admits. numpy reports booleans as "bool", and
# every string representation h5py can hand back -- object, bytes, unicode --
# is declared as "string", which is the name the schema uses.
DTYPE_NAMES = frozenset(
    {
        "float16",
        "float32",
        "float64",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "bool",
        "string",
    }
)

RESERVED_NAMESPACE = "jabs"

_LAYOUT_KEYS = frozenset({"storage", "chunks", "compression", "compression_opts"})
_LAYOUT_COMPRESSIONS = ("none", "gzip", "lzf")
_GZIP_LEVELS = range(10)


def accepts_requested_layout(component_id: str) -> bool:
    """Return whether a producer may choose a component's storage layout.

    The writer owns the layout of every ``jabs`` component except embeddings,
    whose size makes the trade-off the producer's to make.

    Args:
        component_id: Any component id.

    Returns:
        True for an embedding id or an id outside the ``jabs`` namespace.
    """
    return is_embedding_id(component_id) or component_id.split(".")[0] != RESERVED_NAMESPACE


def requested_layout_problems(layout: Mapping, ndim: int) -> list[str]:
    """List why the writer could not honor a requested storage layout.

    The single statement of the rules: ``Component`` raises on any problem,
    and the reader treats a manifest layout with problems as no request.

    Args:
        layout: The requested layout.
        ndim: The rank of the payload it would apply to.

    Returns:
        One message per problem; empty when the writer can honor the request.
    """
    if not isinstance(layout, Mapping):
        return [f"must be a mapping, not {type(layout).__name__}"]
    problems = [f"unknown key {key!r}" for key in sorted(set(layout) - _LAYOUT_KEYS)]
    storage = layout.get("storage")
    compression = layout.get("compression", "none")
    opts = layout.get("compression_opts")
    chunks = layout.get("chunks")
    if storage not in ("contiguous", "chunked"):
        problems.append(f"storage must be 'contiguous' or 'chunked', not {storage!r}")
    if compression not in _LAYOUT_COMPRESSIONS:
        problems.append(f"compression must be one of {_LAYOUT_COMPRESSIONS}, not {compression!r}")
    if compression == "gzip":
        if isinstance(opts, bool) or not isinstance(opts, int) or opts not in _GZIP_LEVELS:
            problems.append(f"gzip needs an integer compression_opts 0-9, not {opts!r}")
    elif opts is not None:
        problems.append(f"{compression!r} compression takes no compression_opts, got {opts!r}")
    if storage == "contiguous":
        if compression != "none":
            problems.append(f"contiguous storage cannot be compressed with {compression!r}")
        if chunks is not None:
            problems.append("contiguous storage takes no chunks")
    elif storage == "chunked":
        if not isinstance(chunks, list | tuple) or len(chunks) != ndim:
            problems.append(f"chunked storage needs {ndim} chunk extents, not {chunks!r}")
        elif any(isinstance(c, bool) or not isinstance(c, int) or c < 1 for c in chunks):
            problems.append(f"chunk extents must be integers of at least 1, not {list(chunks)}")
    return problems


def dtype_name(data: ComponentData) -> str:
    """Return a payload's dtype name, as the manifest records it.

    Args:
        data: An in-memory array or a block source.

    Returns:
        ``"string"`` for every string representation, else numpy's name.
    """
    if data.dtype.kind in _STRING_KINDS:
        return "string"
    return data.dtype.name


@dataclass(frozen=True)
class Skeleton:
    """A named keypoint set and the edges that connect it.

    Edges are pairs rather than polylines. Drawing an edge only when both of
    its endpoints are valid gives the same picture as splitting a polyline at
    missing keypoints, which is what JABS's ``gen_line_fragments`` exists to do.

    Attributes:
        body_parts: Keypoint names, in index order.
        edges: Pairs of indexes into ``body_parts``.
        description: Optional human-readable description.
    """

    body_parts: tuple[str, ...]
    edges: tuple[tuple[int, int], ...] = ()
    description: str | None = None

    def __post_init__(self) -> None:
        """Validate that every edge names a keypoint this skeleton has."""
        if not self.body_parts:
            raise ValueError("a skeleton must have at least one body part")
        limit = len(self.body_parts)
        bad = [e for e in self.edges if e[0] >= limit or e[1] >= limit or e[0] < 0 or e[1] < 0]
        if bad:
            raise ValueError(f"skeleton edge out of range for {limit} body parts: {bad}")


@dataclass(frozen=True)
class VideoInfo:
    """What the file knows about the video it describes.

    ``width`` and ``height`` are the coordinate space the pixel coordinates
    live in. They are nullable because no legacy pose format recorded them, and
    an explicit unknown is recoverable where a plausible default is not.

    Attributes:
        frame_count: Number of video frames.
        width: Frame width in pixels, or None when unknown.
        height: Frame height in pixels, or None when unknown.
        fps: Frame rate, or None when unknown.
        cm_per_pixel: Scale factor, or None when unknown.
        cm_per_pixel_source: How the scale was determined.
        start_time: Wall-clock start of the video, ISO 8601.
        filename: Advisory source video file name.
        content_hash: Authoritative link to the source video.
        clip_of: Source identity and ``frame_offset`` when this file is a clip.
    """

    frame_count: int
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    cm_per_pixel: float | None = None
    cm_per_pixel_source: str | None = None
    start_time: str | None = None
    filename: str | None = None
    content_hash: str | None = None
    clip_of: dict | None = None


@dataclass(frozen=True, eq=False)
class Attachment:
    """A tier-1 opaque payload that rides along in the file.

    An attachment declares no axes, so no tool can subset it correctly. The
    specification therefore requires that a tool transforming a file carry
    attachments through verbatim and record the transformation — dropping one
    silently is a specification violation.

    Attributes:
        path: Absolute HDF5 path, conventionally under ``/attachments/``.
        data: The payload.
        description: Optional human-readable description.
        content_type: Optional media type, for a consumer that knows the
            producer's convention.
    """

    path: str
    data: np.ndarray
    description: str | None = None
    content_type: str | None = None

    def __post_init__(self) -> None:
        """Validate that the path is absolute."""
        if not self.path.startswith("/"):
            raise ValueError(f"attachment path must be absolute, got {self.path!r}")


@dataclass(frozen=True, eq=False)
class Component:
    """One declared array in a pose file.

    ``eq`` is disabled because the payload is a numpy array, and dataclass
    equality on arrays returns an array rather than a bool.

    Attributes:
        id: Namespaced component id, e.g. ``jabs.pose.points``.
        axes: Axis name per array dimension, e.g. ``("frame", "slot")``.
        data: The payload: an in-memory array, or a
            :class:`~jabs.io.internal.pose_file.blocks.BlockSource` that
            declares its shape and dtype and yields the rows in blocks.
        missing: The missing-value policy — ``{"policy": "none" | "nan"}``,
            ``{"policy": "mask", "mask": <id>}`` or
            ``{"policy": "length", "length": <id>}``.
        units: Physical units, required when the component has a ``coord`` axis.
        coord_order: ``"xy"`` or ``"yx"``, required with a ``coord`` axis.
        skeleton: Skeleton id, for components with a ``keypoint`` axis.
        provenance: Key into the file's provenance records.
        sparse_index: Component id mapping ``sample`` positions to frame
            numbers. Required with a ``sample`` axis, and forbidden without one.
        description: Optional human-readable description.
        layout: Declared HDF5 storage, e.g.
            ``{"storage": "contiguous", "compression": "none"}``.
        encoding: How the payload is encoded. Only ``{"kind": "dense"}`` is
            implemented; a declaration that is preserved rather than re-asserted
            on write, so a ragged or RLE file cannot be silently relabeled.
        extra: Namespaced producer metadata. The manifest's own extension
            point, so it must survive a round trip.
        stored_path: Where the payload actually lives, when that differs from
            the path derived from the id. Preserved so a read-write cycle does
            not relocate a third party's data.
        alternates: Derived copies of the data in another encoding, each under
            a name that appears at most once per component. They are derived
            from ``data``, so per ADR 0002 a caller that changes a component's
            baseline data must drop or regenerate its alternates, e.g.
            ``dataclasses.replace(c, data=..., alternates=())``. Nothing here
            compares them with the data, and the default ``deep=False`` write
            does not catch a stale one.
        embedding: The descriptor of an embedding component. Required for an
            id under ``jabs.embedding.``, which must be exactly
            ``jabs.embedding.<name>.(instance|frame)`` with axes, feature
            width, dtype and missing policy matching the descriptor, and
            forbidden on every other id.
        requested_layout: The HDF5 storage the producer asks the writer to
            use, in the manifest's layout form. Honored only for embedding ids
            and ids outside ``jabs``; chunk extents are clamped to the axis
            lengths. None leaves the writer's policy in charge. ``layout``, by
            contrast, records what a file declared and is never applied.
    """

    id: str
    axes: tuple[str, ...]
    data: ComponentData
    missing: dict
    units: str | None = None
    coord_order: str | None = None
    skeleton: str | None = None
    provenance: str | None = None
    sparse_index: str | None = None
    description: str | None = None
    layout: dict | None = None
    encoding: dict = field(default_factory=lambda: {"kind": "dense"})
    extra: dict | None = None
    stored_path: str | None = None
    alternates: "tuple[Alternate, ...]" = ()
    embedding: EmbeddingDescriptor | None = None
    requested_layout: dict | None = None

    def __post_init__(self) -> None:
        """Validate the id, axes, payload dtype, alternates, embedding and requested layout."""
        if not COMPONENT_ID_RE.fullmatch(self.id):
            raise ValueError(
                f"invalid component id {self.id!r}: expected lowercase, dot-separated, "
                "at least two segments"
            )
        segments = self.id.split(".")
        if segments[0] != RESERVED_NAMESPACE and len(segments) < 3:
            raise ValueError(
                f"invalid component id {self.id!r}: a non-jabs namespace must use a "
                "reverse-DNS root of at least two segments"
            )
        if len(self.axes) != self.data.ndim:
            raise ValueError(
                f"{self.id}: declared {len(self.axes)} axes {self.axes} but the payload has "
                f"{self.data.ndim} dimensions {self.data.shape}"
            )
        if "coord" in self.axes:
            if self.units is None or self.coord_order is None:
                raise ValueError(
                    f"{self.id}: a component with a coord axis must declare units and coord_order"
                )
            pair = self.data.shape[self.axes.index("coord")]
            if pair != 2:
                raise ValueError(
                    f"{self.id}: coord axis is {pair} wide; coord_order describes two values "
                    f"({self.coord_order!r}) and nothing else"
                )
        if "keypoint" in self.axes and self.skeleton is None:
            raise ValueError(f"{self.id}: a component with a keypoint axis must name its skeleton")
        if self.missing.get("policy") == "nan" and self.dtype not in (
            "float16",
            "float32",
            "float64",
        ):
            raise ValueError(
                f"{self.id}: a nan missing-policy needs a float payload, not {self.dtype!r} -- "
                "NaN is reserved for floats and no other dtype can represent it"
            )
        if self.encoding.get("kind") == "rle" and self.dtype not in (
            "uint8",
            "uint16",
            "uint32",
            "uint64",
        ):
            raise ValueError(
                f"{self.id}: an RLE payload holds run lengths and must be an unsigned "
                f"integer, not {self.dtype!r}"
            )
        if "sample" in self.axes and self.sparse_index is None:
            raise ValueError(
                f"{self.id}: a component with a sample axis must declare sparse_index"
            )
        if self.sparse_index is not None and "sample" not in self.axes:
            raise ValueError(f"{self.id}: sparse_index declared but there is no sample axis")
        if self.dtype not in DTYPE_NAMES:
            raise ValueError(f"{self.id}: unsupported dtype {self.dtype!r}")
        names = [alternate.name for alternate in self.alternates]
        repeated = sorted({n for n in names if names.count(n) > 1})
        if repeated:
            raise ValueError(
                f"{self.id}: alternates {repeated} are declared more than once; a name "
                "identifies one encoding"
            )
        self._check_embedding()
        self._check_requested_layout()
        # Frozen only freezes the attribute bindings. Without this a caller
        # keeps a live reference to the dicts that were validated, and can
        # mutate them past every check above.
        object.__setattr__(self, "axes", tuple(self.axes))
        object.__setattr__(self, "missing", MappingProxyType(dict(self.missing)))
        object.__setattr__(self, "encoding", MappingProxyType(dict(self.encoding)))
        object.__setattr__(self, "alternates", tuple(self.alternates))
        if self.requested_layout is not None:
            frozen = dict(self.requested_layout)
            if "chunks" in frozen:
                frozen["chunks"] = tuple(frozen["chunks"])
            object.__setattr__(self, "requested_layout", MappingProxyType(frozen))

    def _check_requested_layout(self) -> None:
        """Validate a requested layout against the id and the payload's rank.

        Raises:
            ValueError: If the id does not admit a request, or the writer could
                not honor it.
        """
        if self.requested_layout is None:
            return
        if not accepts_requested_layout(self.id):
            raise ValueError(
                f"{self.id}: requested_layout is honored only for jabs.embedding.* and "
                "non-jabs components; the writer owns every other jabs layout"
            )
        problems = requested_layout_problems(self.requested_layout, self.data.ndim)
        if problems:
            raise ValueError(f"{self.id}: invalid requested_layout -- " + "; ".join(problems))

    def _check_embedding(self) -> None:
        """Validate the embedding descriptor against the id and payload.

        Raises:
            ValueError: If an embedding id is malformed, lacks a descriptor or
                disagrees with it, or a non-embedding id carries one.
        """
        if not is_embedding_id(self.id):
            if self.embedding is not None:
                raise ValueError(
                    f"{self.id}: only jabs.embedding.<name>.<kind> components carry an "
                    "embedding descriptor"
                )
            return
        parsed = parse_embedding_id(self.id)
        if parsed is None:
            raise ValueError(
                f"invalid embedding component id {self.id!r}: expected "
                "jabs.embedding.<name>.instance or jabs.embedding.<name>.frame"
            )
        descriptor = self.embedding
        if descriptor is None:
            raise ValueError(f"{self.id}: an embedding component must carry its descriptor")
        name, kind = parsed
        if (descriptor.name, descriptor.kind) != (name, kind):
            raise ValueError(
                f"{self.id}: descriptor names {descriptor.name!r} of kind {descriptor.kind!r}, "
                f"but the id says {name!r} of kind {kind!r}"
            )
        if self.axes != EMBEDDING_AXES[kind]:
            raise ValueError(
                f"{self.id}: a {kind} embedding has axes {EMBEDDING_AXES[kind]}, not {self.axes}"
            )
        if self.data.shape[-1] != descriptor.dim:
            raise ValueError(
                f"{self.id}: feature axis is {self.data.shape[-1]} wide but the descriptor's "
                f"dim is {descriptor.dim}"
            )
        if self.dtype not in EMBEDDING_DTYPES:
            raise ValueError(
                f"{self.id}: an embedding must be one of {sorted(EMBEDDING_DTYPES)}, "
                f"not {self.dtype!r}"
            )
        if dict(self.missing) != {"policy": "nan"}:
            raise ValueError(
                f"{self.id}: an embedding's missing policy must be {{'policy': 'nan'}}, "
                f"not {dict(self.missing)}"
            )

    @property
    def path(self) -> str:
        """The absolute HDF5 path this component's payload lives at."""
        if self.stored_path is not None:
            return self.stored_path
        segments = self.id.split(".")
        if segments[0] == RESERVED_NAMESPACE:
            root, rest = segments[0], segments[1:]
        else:
            # Reverse-DNS: everything up to the local name is one path element,
            # so org.jax.gait.stride_length -> /org.jax.gait/stride_length.
            root, rest = ".".join(segments[:-1]), segments[-1:]
        return "/" + "/".join([root, *rest])

    @property
    def dtype(self) -> str:
        """The payload's dtype name, as the manifest records it."""
        return dtype_name(self.data)


@dataclass(frozen=True)
class ProvenanceRecord:
    """What produced one component.

    Attributes:
        producer: Producing software.
        version: Producer version.
        created: ISO 8601 creation time.
        model: Model reference, ideally resolvable.
        algorithm: Algorithm reference.
        parameters: The policy the producer applied, e.g. a confidence
            threshold. Declaring it is what lets a consumer disagree knowingly.
        extra: Namespaced producer metadata, preserved across a round trip.
    """

    producer: str
    version: str
    created: str
    model: dict | None = None
    algorithm: dict | None = None
    parameters: dict | None = None
    extra: dict | None = None


@dataclass(frozen=True)
class HistoryEntry:
    """One operation applied to the file.

    Attributes:
        operation: One of ``infer``, ``convert``, ``clip``, ``merge``,
            ``annotate``.
        tool: The tool that performed it.
        version: The tool's version.
        time: ISO 8601 timestamp.
        source: Where the input came from, required for ``convert``.
        synthesized: What the operation invented rather than read, required for
            ``convert`` — without it a converted file is indistinguishable from
            a natively produced one.
        dropped: What the operation discarded.
        notes: Free text.
        extra: Namespaced metadata, preserved across a round trip.
    """

    operation: str
    tool: str
    version: str
    time: str
    source: dict | None = None
    synthesized: tuple[str, ...] | None = None
    dropped: tuple[str, ...] | None = None
    notes: str | None = None
    extra: dict | None = None


@dataclass(frozen=True)
class Provenance:
    """Per-component records plus the file's append-only history.

    Attributes:
        records: Provenance records, keyed by the name components reference.
        history: Operations applied to the file, in order.
        extra: Namespaced document-level metadata, preserved across a round
            trip.
    """

    records: dict[str, ProvenanceRecord] = field(default_factory=dict)
    history: tuple[HistoryEntry, ...] = ()
    extra: dict | None = None


@dataclass(frozen=True, eq=False)
class PoseFile:
    """The full contents of one pose file.

    Attributes:
        dimensions: Named dimension sizes; ``frame``, ``slot`` and ``identity``
            are required.
        video: What the file knows about its video.
        skeletons: Skeletons by id.
        components: The file's components.
        provenance: Provenance records and history.
        attachments: Tier-1 opaque payloads, carried verbatim.
        extra: Namespaced file-level metadata.
    """

    dimensions: dict[str, int]
    video: VideoInfo
    skeletons: dict[str, Skeleton] = field(default_factory=dict)
    components: tuple[Component, ...] = ()
    provenance: Provenance = field(default_factory=Provenance)
    attachments: tuple[Attachment, ...] = ()
    extra: dict | None = None

    def __post_init__(self) -> None:
        """Validate dimensions and every cross-reference between components."""
        for required in ("frame", "slot", "identity"):
            if required not in self.dimensions:
                raise ValueError(f"dimensions must declare {required!r}")
        if self.dimensions["frame"] != self.video.frame_count:
            raise ValueError(
                f"dimensions.frame ({self.dimensions['frame']}) disagrees with "
                f"video.frame_count ({self.video.frame_count})"
            )
        if self.dimensions["identity"] > self.dimensions["slot"]:
            raise ValueError(
                f"dimensions.identity ({self.dimensions['identity']}) exceeds "
                f"dimensions.slot ({self.dimensions['slot']})"
            )

        by_id = {c.id: c for c in self.components}
        ids = [c.id for c in self.components]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate component ids: {duplicates}")

        known = set(ids)
        for component in self.components:
            if component.skeleton is not None and component.skeleton not in self.skeletons:
                raise ValueError(
                    f"{component.id}: references unknown skeleton {component.skeleton!r}"
                )
            if (
                component.provenance is not None
                and component.provenance not in self.provenance.records
            ):
                raise ValueError(
                    f"{component.id}: references unknown provenance record "
                    f"{component.provenance!r}"
                )
            if component.sparse_index is not None and component.sparse_index not in known:
                raise ValueError(
                    f"{component.id}: sparse_index references unknown component "
                    f"{component.sparse_index!r}"
                )
            mask = component.missing.get("mask") or component.missing.get("length")
            if mask is not None and mask not in known:
                raise ValueError(f"{component.id}: missing policy references unknown {mask!r}")
            self._check_axis_lengths(component)
            self._check_skeleton_width(component)
            self._check_sparse_reference(component, by_id)
            self._check_missing_reference(component, by_id)
        self._check_alternates()

        self._freeze()

    def _check_axis_lengths(self, component: Component) -> None:
        """Validate a component's axes against the file's dimensions.

        The specification says an axis named ``frame`` always has length
        ``dimensions.frame``, and the same reasoning applies to every axis the
        file gives a size. Checked at construction so the writer cannot emit a
        file that the validator would then reject.

        Args:
            component: The component to check.

        Raises:
            ValueError: If a named axis disagrees with its dimension.
        """
        for axis, length in zip(component.axes, component.data.shape, strict=True):
            declared = self.dimensions.get(axis)
            if declared is not None and length != declared:
                raise ValueError(
                    f"{component.id}: {axis} axis is {length} but dimensions.{axis} is {declared}"
                )

    def _check_skeleton_width(self, component: Component) -> None:
        """Validate a keypoint axis against the skeleton it names.

        Args:
            component: The component to check.

        Raises:
            ValueError: If the skeleton cannot label the keypoint axis.
        """
        if component.skeleton is None or "keypoint" not in component.axes:
            return
        width = component.data.shape[component.axes.index("keypoint")]
        body_parts = len(self.skeletons[component.skeleton].body_parts)
        if width != body_parts:
            raise ValueError(
                f"{component.id}: keypoint axis is {width} wide but skeleton "
                f"{component.skeleton!r} names {body_parts} body parts"
            )

    def _check_sparse_reference(self, component: Component, by_id: dict) -> None:
        """Validate a sparse index's shape and axes.

        Args:
            component: The referencing component.
            by_id: Every component, keyed by id.

        Raises:
            ValueError: If the index is not a one-dimensional sample-axis
                component of the same length as the referencing sample axis.
        """
        if component.sparse_index is None:
            return
        index = by_id[component.sparse_index]
        if index.axes != ("sample",):
            raise ValueError(
                f"{component.id}: sparse index {index.id!r} has axes {index.axes}, expected a "
                "one-dimensional sample axis"
            )
        samples = component.data.shape[component.axes.index("sample")]
        if samples != index.data.shape[0]:
            raise ValueError(
                f"{component.id}: sample axis is {samples} but index {index.id!r} has "
                f"{index.data.shape[0]} entries"
            )

    def _check_missing_reference(self, component: Component, by_id: dict) -> None:
        """Validate that a mask or length reference can align with its target.

        Args:
            component: The component whose policy is checked.
            by_id: Every component, keyed by id.

        Raises:
            ValueError: If the reference's axes are not the target's leading
                axes.
        """
        policy = component.missing.get("policy")
        if policy not in ("mask", "length"):
            return
        other = by_id[component.missing[policy]]
        if other.axes != component.axes[: len(other.axes)]:
            raise ValueError(
                f"{component.id}: {policy} reference {other.id!r} has axes {other.axes}, which "
                f"are not the leading axes of {component.axes}"
            )
        if other.data.shape != component.data.shape[: other.data.ndim]:
            raise ValueError(
                f"{component.id}: {policy} reference {other.id!r} has shape "
                f"{other.data.shape}, which does not align with {component.data.shape}"
            )
        # Alignment is not interpretability: a boolean mask says present or
        # absent, and a length says how many, so a float mask or a negative
        # length cannot be applied however well its shape lines up.
        if policy == "mask" and other.data.dtype.kind != "b":
            raise ValueError(
                f"{component.id}: mask reference {other.id!r} has dtype "
                f"{other.data.dtype.name!r}; a mask must be boolean"
            )
        if policy == "length":
            if other.data.dtype.kind not in "iu":
                raise ValueError(
                    f"{component.id}: length reference {other.id!r} has dtype "
                    f"{other.data.dtype.name!r}; a length must be an integer"
                )
            bound = (
                component.data.shape[other.data.ndim]
                if other.data.ndim < component.data.ndim
                else 0
            )
            # A block source's values are not in memory, and reading them here
            # would consume the blocks the writer needs; the validator checks
            # the written values instead.
            if (
                isinstance(other.data, np.ndarray)
                and other.data.size
                and (other.data.min() < 0 or other.data.max() > bound)
            ):
                raise ValueError(
                    f"{component.id}: length reference {other.id!r} has values outside "
                    f"[0, {bound}], the axis it bounds"
                )

    def _check_alternates(self) -> None:
        """Validate alternates across components.

        An alternate's payload path must differ from every component's and
        every other alternate's, or one dataset would be claimed twice. And
        alternates of one name are one encoding (ADR 0002): they must name the
        same two offset paths, or the name would mean two things, and those
        naming the same offset paths must share one
        :class:`~jabs.io.internal.pose_file.alternates.RaggedIndex`, or the
        writer would have two different arrays to write at one path.
        Parent/child path collisions, and two names' different indexes at one
        pair of paths, are the writer's checks.

        Raises:
            ValueError: If a path is claimed twice, alternates of one name name
                different offset paths, or one name's offsets are declared by
                two different indexes.
        """
        # Imported here: alternates imports this module at load time.
        from jabs.io.internal.pose_file.alternates import alternate_key

        owners = {component.path: component.id for component in self.components}
        indexes: dict[tuple[str, str, str], object] = {}
        first: dict[str, tuple[str, tuple[str, str]]] = {}
        for component in self.components:
            for alternate in component.alternates:
                label = alternate_key(component.id, alternate.name)
                other = owners.get(alternate.path)
                if other is not None:
                    raise ValueError(f"{label}: path {alternate.path} is already {other}'s")
                owners[alternate.path] = label
                index = alternate.index
                offsets = (index.group_offsets_path, index.instance_offsets_path)
                other_label, other_offsets = first.setdefault(alternate.name, (label, offsets))
                if offsets != other_offsets:
                    raise ValueError(
                        f"{label}: names offsets {offsets}, but {other_label} names "
                        f"{other_offsets}; alternates of one name are one encoding and share "
                        "their offsets"
                    )
                key = (alternate.name, *offsets)
                shared = indexes.setdefault(key, index)
                if shared is not index:
                    raise ValueError(
                        f"{label}: alternates named {alternate.name!r} declare offsets at "
                        f"{key[1]} and {key[2]} through different RaggedIndex objects; "
                        "they must share one"
                    )

    def _freeze(self) -> None:
        """Replace the validated mappings with read-only views.

        Frozen dataclasses freeze attribute bindings, not the objects bound.
        Without this, ``pose.dimensions["identity"] = 99`` sails past
        ``__post_init__`` and is then written to disk.
        """
        object.__setattr__(self, "dimensions", MappingProxyType(dict(self.dimensions)))
        object.__setattr__(self, "skeletons", MappingProxyType(dict(self.skeletons)))
        object.__setattr__(self, "components", tuple(self.components))
        object.__setattr__(self, "attachments", tuple(self.attachments))

    def component(self, component_id: str) -> Component:
        """Return one component by id.

        Args:
            component_id: The component's namespaced id.

        Returns:
            The matching component.

        Raises:
            KeyError: If this file has no such component.
        """
        for candidate in self.components:
            if candidate.id == component_id:
                return candidate
        raise KeyError(f"no component {component_id!r} in this pose file")
