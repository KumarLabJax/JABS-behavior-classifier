"""Embedding component descriptors for the JABS pose file format (ADR 0002).

An embedding component carries a learned feature vector per frame (``frame``
kind) or per frame and slot (``instance`` kind) under the id
``jabs.embedding.<name>.<kind>``. Its manifest entry records an
:class:`EmbeddingDescriptor` saying which model produced it and how, plus a
fingerprint of everything that determines the vectors' meaning. Two
embeddings with equal fingerprints are directly comparable.

The fingerprint is the SHA-256 of the canonical JSON of the descriptor's
manifest form minus ``name``, ``kind`` and ``fingerprint``. The descriptor
holds only strings, ints and nested objects of them, never floats, so its
canonical form is unambiguous.
"""

import hashlib
import json
import operator
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

EMBEDDING_PREFIX: str = "jabs.embedding."
EMBEDDING_KINDS: frozenset[str] = frozenset({"instance", "frame"})
EMBEDDING_AXES: dict[str, tuple[str, ...]] = {
    "instance": ("frame", "slot", "feature"),
    "frame": ("frame", "feature"),
}
EMBEDDING_DTYPES: frozenset[str] = frozenset({"float16", "float32"})

# What an embedding's `.instance` and `.frame` components must agree on.
PAIRED_FIELDS: tuple[str, ...] = ("dim", "model", "tap", "window", "normalization", "projection")

# One component id segment; the same rule COMPONENT_ID_RE in types.py applies
# per segment. Kept local because types.py will import this module.
_NAME_RE = re.compile(r"[a-z0-9]([a-z0-9_-]*[a-z0-9])?")
_EMBEDDING_ID_RE = re.compile(
    re.escape(EMBEDDING_PREFIX) + r"(?P<name>" + _NAME_RE.pattern + r")\.(?P<kind>instance|frame)"
)
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}")

# Manifest keys that identify an embedding rather than describe its vectors.
_UNFINGERPRINTED = frozenset({"name", "kind", "fingerprint"})


def parse_embedding_id(component_id: str) -> tuple[str, str] | None:
    """Split a well-formed embedding component id into its name and kind.

    Args:
        component_id: Any component id.

    Returns:
        ``(name, kind)`` for an id of exactly the form
        ``jabs.embedding.<name>.(instance|frame)``, otherwise ``None``.
    """
    match = _EMBEDDING_ID_RE.fullmatch(component_id)
    if match is None:
        return None
    return match["name"], match["kind"]


def is_embedding_id(component_id: str) -> bool:
    """Return whether an id lies in the embedding namespace, well-formed or not.

    Args:
        component_id: Any component id.

    Returns:
        True when the id starts with ``jabs.embedding.``.
    """
    return component_id.startswith(EMBEDDING_PREFIX)


def canonical_json(obj: object) -> bytes:
    """Serialize a JSON value to its canonical bytes.

    Keys are sorted, no whitespace is emitted and non-ASCII characters are
    written as UTF-8 rather than escaped.

    Args:
        obj: A JSON-serializable value.

    Returns:
        The canonical UTF-8 encoding.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def compute_fingerprint(descriptor_json: dict[str, Any]) -> str:
    """Compute the fingerprint of a descriptor in its manifest form.

    The ``name``, ``kind`` and ``fingerprint`` keys are dropped if present, so
    raw manifest JSON can be passed directly.

    Args:
        descriptor_json: The descriptor as recorded in the manifest.

    Returns:
        ``"sha256:"`` followed by the hex digest of the canonical JSON.
    """
    fingerprinted = {k: v for k, v in descriptor_json.items() if k not in _UNFINGERPRINTED}
    return "sha256:" + hashlib.sha256(canonical_json(fingerprinted)).hexdigest()


def _require_non_empty(value: str, label: str) -> None:
    """Raise ValueError when a descriptor string is empty."""
    if not value:
        raise ValueError(f"embedding {label} must be a non-empty string")


def _require_str(value: object, label: str) -> None:
    """Raise ValueError when a descriptor string field holds something else."""
    if not isinstance(value, str):
        raise ValueError(f"embedding {label} must be a string, got {type(value).__name__}")


def _as_int(value: object, label: str) -> int:
    """Return a descriptor integer as a plain int.

    NumPy integers are converted; anything else that is not exactly an int
    (a float, even an integral one, or a bool) is rejected, because it would
    serialize differently and change the fingerprint.

    Args:
        value: The field's value.
        label: The field's name, for the error message.

    Returns:
        The value as an ``int``.

    Raises:
        ValueError: If the value is not an integer, or is a bool.
    """
    if isinstance(value, np.integer):
        value = operator.index(value)
    if type(value) is not int:
        raise ValueError(f"embedding {label} must be an int, got {type(value).__name__}")
    return value


@dataclass(frozen=True)
class EmbeddingModel:
    """The model that produced an embedding.

    Attributes:
        name: Human-readable model name.
        weights: ``sha256:<hex>`` digest of the model weights.
    """

    name: str
    weights: str


@dataclass(frozen=True)
class TemporalWindow:
    """The largest temporal context, in frames, any vector may draw on.

    Attributes:
        before: Frames before the current frame.
        after: Frames after the current frame.
    """

    before: int
    after: int


@dataclass(frozen=True)
class Projection:
    """A projection applied to the tapped features before storage.

    Attributes:
        kind: The projection method, e.g. ``"pca"``.
        input_dim: Feature width before projection.
        basis: Identifier of the projection basis.
    """

    kind: str
    input_dim: int
    basis: str


@dataclass(frozen=True)
class EmbeddingDescriptor:
    """The manifest description of one embedding component.

    Attributes:
        name: The ``<name>`` segment of ``jabs.embedding.<name>.<kind>``.
        kind: ``"instance"`` or ``"frame"``.
        dim: Length of the feature axis.
        model: The producing model.
        tap: Where in the model the features were taken.
        pooling: How features were pooled into one vector.
        window: The temporal context bound.
        normalization: Normalization applied to stored vectors.
        projection: Projection applied before storage, if any.
    """

    name: str
    kind: str
    dim: int
    model: EmbeddingModel
    tap: str
    pooling: str
    window: TemporalWindow
    normalization: str = "none"
    projection: Projection | None = None

    def __post_init__(self) -> None:
        """Validate the descriptor.

        NumPy integers are stored as plain ints.

        Raises:
            ValueError: If any field has the wrong type, is out of range or is
                malformed.
        """
        self._check_types()
        if not _NAME_RE.fullmatch(self.name):
            raise ValueError(f"embedding name {self.name!r} is not a valid id segment")
        if self.kind not in EMBEDDING_KINDS:
            raise ValueError(
                f"embedding kind must be one of {sorted(EMBEDDING_KINDS)}, got {self.kind!r}"
            )
        if self.dim < 1:
            raise ValueError(f"embedding dim must be >= 1, got {self.dim}")
        _require_non_empty(self.model.name, "model.name")
        if not _SHA256_RE.fullmatch(self.model.weights):
            raise ValueError(
                f"embedding model.weights must match sha256:<64 hex>, got {self.model.weights!r}"
            )
        if self.window.before < 0 or self.window.after < 0:
            raise ValueError(
                "embedding window extents must be >= 0, got "
                f"({self.window.before}, {self.window.after})"
            )
        if self.projection is not None and self.projection.input_dim < 1:
            raise ValueError(
                f"embedding projection.input_dim must be >= 1, got {self.projection.input_dim}"
            )
        _require_non_empty(self.tap, "tap")
        _require_non_empty(self.pooling, "pooling")
        _require_non_empty(self.normalization, "normalization")

    def _check_types(self) -> None:
        """Require real strings and ints, converting NumPy integers in place.

        Raises:
            ValueError: If a string field is not a string, or an integer field
                is a float, a bool or not a number.
        """
        projection = self.projection
        strings = {
            "name": self.name,
            "kind": self.kind,
            "model.name": self.model.name,
            "model.weights": self.model.weights,
            "tap": self.tap,
            "pooling": self.pooling,
            "normalization": self.normalization,
        }
        if projection is not None:
            strings |= {"projection.kind": projection.kind, "projection.basis": projection.basis}
        for label, value in strings.items():
            _require_str(value, label)

        # Frozen, so coerced values are set through object.__setattr__.
        object.__setattr__(self, "dim", _as_int(self.dim, "dim"))
        window = TemporalWindow(
            _as_int(self.window.before, "window.before"),
            _as_int(self.window.after, "window.after"),
        )
        object.__setattr__(self, "window", window)
        if projection is not None:
            projection = Projection(
                projection.kind,
                _as_int(projection.input_dim, "projection.input_dim"),
                projection.basis,
            )
            object.__setattr__(self, "projection", projection)

    @property
    def fingerprint(self) -> str:
        """The ``sha256:<hex>`` fingerprint of the fingerprinted fields."""
        return compute_fingerprint(self._fields_json())

    def _fields_json(self) -> dict[str, Any]:
        """Return the manifest form without the fingerprint."""
        projection = self.projection
        return {
            "name": self.name,
            "kind": self.kind,
            "dim": self.dim,
            "model": {"name": self.model.name, "weights": self.model.weights},
            "tap": self.tap,
            "pooling": self.pooling,
            "window": {"before": self.window.before, "after": self.window.after},
            "normalization": self.normalization,
            "projection": None
            if projection is None
            else {
                "kind": projection.kind,
                "input_dim": projection.input_dim,
                "basis": projection.basis,
            },
        }

    def to_json(self) -> dict[str, Any]:
        """Return the manifest form, including the fingerprint.

        Returns:
            A JSON-serializable dict; an absent projection is recorded as None.
        """
        data = self._fields_json()
        data["fingerprint"] = compute_fingerprint(data)
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "EmbeddingDescriptor":
        """Build a descriptor from its manifest form.

        Any stored ``fingerprint`` is ignored; comparing it with the computed
        one is the validator's job. Every key the schema requires is indexed
        directly, so call this only on schema-validated manifest data.

        Args:
            data: The descriptor as recorded in the manifest.

        Returns:
            The descriptor.

        Raises:
            ValueError: If a field is out of range or malformed.
        """
        projection = data["projection"]
        return cls(
            name=data["name"],
            kind=data["kind"],
            dim=data["dim"],
            model=EmbeddingModel(**data["model"]),
            tap=data["tap"],
            pooling=data["pooling"],
            window=TemporalWindow(**data["window"]),
            normalization=data["normalization"],
            projection=None if projection is None else Projection(**projection),
        )


@dataclass(frozen=True)
class EmbeddingInfo:
    """A summary of one embedding component, as a file's manifest declares it.

    Attributes:
        name: The ``<name>`` segment of the component id.
        kind: ``"instance"`` or ``"frame"``.
        dim: Length of the feature axis, from the descriptor.
        fingerprint: The fingerprint the manifest records.
        component_id: The full component id.
    """

    name: str
    kind: str
    dim: int
    fingerprint: str
    component_id: str


def list_embeddings(manifest: dict[str, Any]) -> list[EmbeddingInfo]:
    """List the embeddings a manifest declares.

    Name and kind come from the component id; dim and fingerprint from its
    descriptor. The fingerprint is the stored one, not recomputed: checking it
    is the validator's job. A ``jabs.embedding.*`` id that does not parse is
    not listed.

    Args:
        manifest: A schema-validated manifest, as ``read_manifest`` returns it.

    Returns:
        One entry per well-formed embedding component, sorted by ``(name, kind)``.
    """
    infos = []
    for spec in manifest["components"]:
        parsed = parse_embedding_id(spec["id"])
        if parsed is None:
            continue
        name, kind = parsed
        descriptor = spec["embedding"]
        infos.append(
            EmbeddingInfo(
                name=name,
                kind=kind,
                dim=descriptor["dim"],
                fingerprint=descriptor["fingerprint"],
                component_id=spec["id"],
            )
        )
    return sorted(infos, key=lambda info: (info.name, info.kind))
