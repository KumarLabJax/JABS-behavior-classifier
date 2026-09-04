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
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

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

# numpy dtype kinds that the format calls "string": object (h5py variable
# length), bytes and unicode.
_STRING_KINDS = "OSU"

RESERVED_NAMESPACE = "jabs"


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
        data: The payload.
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
    """

    id: str
    axes: tuple[str, ...]
    data: np.ndarray
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

    def __post_init__(self) -> None:
        """Validate the id, the axis declarations and the payload dtype."""
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
        if "coord" in self.axes and (self.units is None or self.coord_order is None):
            raise ValueError(
                f"{self.id}: a component with a coord axis must declare units and coord_order"
            )
        if "sample" in self.axes and self.sparse_index is None:
            raise ValueError(
                f"{self.id}: a component with a sample axis must declare sparse_index"
            )
        if self.sparse_index is not None and "sample" not in self.axes:
            raise ValueError(f"{self.id}: sparse_index declared but there is no sample axis")
        if self.dtype not in DTYPE_NAMES:
            raise ValueError(f"{self.id}: unsupported dtype {self.dtype!r}")
        # Frozen only freezes the attribute bindings. Without this a caller
        # keeps a live reference to the dicts that were validated, and can
        # mutate them past every check above.
        object.__setattr__(self, "axes", tuple(self.axes))
        object.__setattr__(self, "missing", MappingProxyType(dict(self.missing)))
        object.__setattr__(self, "encoding", MappingProxyType(dict(self.encoding)))

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
        if self.data.dtype.kind in _STRING_KINDS:
            return "string"
        return self.data.dtype.name


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
    """

    producer: str
    version: str
    created: str
    model: dict | None = None
    algorithm: dict | None = None
    parameters: dict | None = None


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
    """

    operation: str
    tool: str
    version: str
    time: str
    source: dict | None = None
    synthesized: tuple[str, ...] | None = None
    dropped: tuple[str, ...] | None = None
    notes: str | None = None


@dataclass(frozen=True)
class Provenance:
    """Per-component records plus the file's append-only history.

    Attributes:
        records: Provenance records, keyed by the name components reference.
        history: Operations applied to the file, in order.
    """

    records: dict[str, ProvenanceRecord] = field(default_factory=dict)
    history: tuple[HistoryEntry, ...] = ()


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
        if self.dimensions["identity"] > self.dimensions["slot"]:
            raise ValueError(
                f"dimensions.identity ({self.dimensions['identity']}) exceeds "
                f"dimensions.slot ({self.dimensions['slot']})"
            )

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

        self._freeze()

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
