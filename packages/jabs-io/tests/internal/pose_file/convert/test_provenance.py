"""The convert entry's helpers turn legacy attributes into strict JSON, and name the dialect."""

import base64
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from jabs.io.internal.pose_file.convert import convert_legacy, inspect_legacy
from jabs.io.internal.pose_file.convert.provenance import (
    _json_value,
    legacy_scale,
    source_format,
)

from .conftest import legacy

# Not UTF-8: 0xff and 0xfe never start a UTF-8 sequence.
RAW = b"\xff\xfeabc"
MARKED = {"base64": base64.b64encode(RAW).decode("ascii")}


def _not_json(token: str) -> None:
    """Refuse the NaN and infinity tokens strict JSON parsers reject."""
    raise ValueError(f"{token} is not strict JSON")


def _strict(value: object) -> object:
    """Round-trip a value through strict JSON, as a Go or JavaScript consumer would read it."""
    text = json.dumps(value, allow_nan=False, ensure_ascii=False)
    # Encoding raises on a lone surrogate, which is not a Unicode scalar value.
    return json.loads(text.encode("utf-8"), parse_constant=_not_json)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(b"cm", "cm", id="utf8-bytes"),
        pytest.param(np.bytes_("µm".encode()), "µm", id="utf8-np-bytes"),
        pytest.param(RAW, MARKED, id="non-utf8-bytes"),
        pytest.param(np.bytes_(RAW), MARKED, id="non-utf8-np-bytes"),
        # h5py hands back a variable-length string's undecodable bytes as
        # surrogate escapes.
        pytest.param(RAW.decode("utf-8", "surrogateescape"), MARKED, id="surrogate-escaped-str"),
        pytest.param(
            "a\ud800",
            {"base64": base64.b64encode("a\ud800".encode("utf-8", "surrogatepass")).decode()},
            id="lone-surrogate-str",
        ),
        pytest.param(np.array([RAW, b"ok"], dtype="S5"), [MARKED, "ok"], id="fixed-bytes-array"),
        pytest.param(np.array([b"ok"], dtype=object), ["ok"], id="object-bytes-array"),
        pytest.param("plain", "plain", id="str"),
        pytest.param(np.float32(np.nan), "NaN", id="nan"),
        pytest.param(np.array([np.inf, -np.inf, 1.5]), ["Infinity", "-Infinity", 1.5], id="inf"),
        pytest.param(np.int64(7), 7, id="int"),
        pytest.param(np.bool_(True), True, id="bool"),
        pytest.param(np.array(2.5), 2.5, id="zero-d"),
        pytest.param(np.arange(4).reshape(2, 2), [[0, 1], [2, 3]], id="2d"),
        pytest.param(h5py.Empty("f4"), None, id="empty"),
    ],
)
def test_json_value(value: object, expected: object) -> None:
    """Each attribute kind gets a strict JSON form; undecodable bytes are kept as base64."""
    converted = _json_value(value, None)

    assert converted == expected
    assert _strict(converted) == expected


def test_json_value_base64_keeps_the_bytes() -> None:
    """The base64 marker decodes to exactly the stored bytes."""
    marked = _json_value(RAW.decode("utf-8", "surrogateescape"), None)

    assert isinstance(marked, dict)
    assert base64.b64decode(marked["base64"]) == RAW


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(np.zeros(2, dtype=[("a", "i4"), ("b", "f4")]), id="compound-array"),
        pytest.param(np.zeros(1, dtype=[("a", "i4")])[0], id="compound-scalar"),
        pytest.param(object(), id="object"),
    ],
)
def test_json_value_refuses_kinds_without_json(value: object) -> None:
    """A value with no JSON form raises TypeError, which the caller names."""
    with pytest.raises(TypeError, match="no JSON form"):
        _json_value(value, None)


def test_json_value_reference_needs_its_file(tmp_path: Path) -> None:
    """A non-null reference cannot be named without the file it points into."""
    with h5py.File(tmp_path / "ref.h5", "w") as h5:
        group = h5.create_group("target")
        assert _json_value(group.ref, h5) == "<reference:/target>"
        assert _json_value(h5py.Reference(), h5) == "<reference:null>"
        with pytest.raises(TypeError, match="without its file"):
            _json_value(group.ref, None)


@pytest.mark.parametrize(
    ("values", "expected", "noted"),
    [
        pytest.param({}, (None, None), (), id="absent"),
        pytest.param(
            {"cm_per_pixel": np.float32(0.5), "cm_per_pixel_source": "manually_set"},
            (0.5, "manually_set"),
            (),
            id="usable",
        ),
        pytest.param(
            {"cm_per_pixel": np.array([0.25]), "cm_per_pixel_source": b"corner_detection"},
            (0.25, "corner_detection"),
            (),
            id="one-element-array-and-bytes-source",
        ),
        pytest.param({"cm_per_pixel": np.nan}, (None, None), ("not a positive number",), id="nan"),
        pytest.param({"cm_per_pixel": 0.0}, (None, None), ("not a positive number",), id="zero"),
        pytest.param(
            {"cm_per_pixel": np.bytes_(RAW)},
            (None, None),
            ("not a positive number",),
            id="undecodable-bytes",
        ),
        pytest.param(
            {"cm_per_pixel": 0.5, "cm_per_pixel_source": "eyeballed"},
            (0.5, None),
            ("not a recognized source",),
            id="unknown-source",
        ),
        pytest.param(
            {"cm_per_pixel": -1.0, "cm_per_pixel_source": "manually_set"},
            (None, None),
            ("not a positive number", "means nothing without it"),
            id="source-without-scale",
        ),
    ],
)
def test_legacy_scale(
    tmp_path: Path,
    values: dict[str, object],
    expected: tuple[object, object],
    noted: tuple[str, ...],
) -> None:
    """A usable scale passes through; an unusable one is null, with a note per value."""
    with h5py.File(tmp_path / "attrs.h5", "w") as h5:
        attrs = h5.create_group("poseest").attrs
        for name, value in values.items():
            attrs[name] = value
        scale, source, notes = legacy_scale(attrs)

    assert (scale, source) == (pytest.approx(expected[0]) if expected[0] else None, expected[1])
    assert len(notes) == len(noted)
    for note, phrase in zip(notes, noted, strict=True):
        assert phrase in note


@pytest.mark.parametrize(
    ("name", "dialect"),
    [
        ("v8-with-centers.h5", "pose_est_v8"),
        ("v7-dynamic-objects.h5", "pose_est_v7"),
    ],
)
def test_source_format(name: str, dialect: str) -> None:
    """The dialect names the version's major part."""
    assert source_format(inspect_legacy(legacy(name))) == dialect


def test_source_format_without_a_version(tmp_path: Path) -> None:
    """Without a version the dialect is just ``pose_est``."""
    src = tmp_path / "unversioned.h5"
    shutil.copyfile(legacy("v8-with-centers.h5"), src)
    with h5py.File(src, "r+") as h5:
        del h5["poseest"].attrs["version"]

    assert source_format(inspect_legacy(src)) == "pose_est"


def _vlen(attrs: h5py.AttributeManager, name: str, raw: bytes) -> None:
    """Store ``raw`` as a variable-length string attribute, whatever its encoding."""
    attrs.create(name, np.array(raw, dtype=object), dtype=h5py.string_dtype("utf-8"))


def test_undecodable_attributes_round_trip(tmp_path: Path) -> None:
    """Undecodable attribute bytes, fixed or variable length, come back exactly from the history.

    ``model_metadata_json`` takes the same path, and an array of it is kept
    whole rather than cut to its first element.
    """
    src = tmp_path / "bytes.h5"
    shutil.copyfile(legacy("v8-with-centers.h5"), src)
    with h5py.File(src, "r+") as h5:
        attrs = h5["poseest"].attrs
        attrs["fixed"] = np.bytes_(RAW)
        _vlen(attrs, "variable", RAW)
        attrs["model_metadata_json"] = np.array([b'{"a": 1}', RAW], dtype="S8")
    out = tmp_path / "out.h5"
    convert_legacy(src, out)

    with h5py.File(out, "r") as h5:
        document = json.loads(h5["provenance"][()], parse_constant=_not_json)
    # Strict: no NaN tokens, and no lone surrogates.
    (entry, *_) = _strict(document)["history"]
    extra = entry["extra"]
    raw = extra["legacy_attributes"]["poseest"]
    assert base64.b64decode(raw["fixed"]["base64"]) == RAW
    assert base64.b64decode(raw["variable"]["base64"]) == RAW
    assert extra["model_metadata_json"] == ['{"a": 1}', MARKED]


def test_model_metadata_json_text_is_kept(tmp_path: Path) -> None:
    """A UTF-8 ``model_metadata_json`` is kept as its text."""
    src = tmp_path / "metadata.h5"
    shutil.copyfile(legacy("v8-with-centers.h5"), src)
    with h5py.File(src, "r+") as h5:
        _vlen(h5["poseest"].attrs, "model_metadata_json", b'{"model": "hrnet"}')
    out = tmp_path / "out.h5"
    convert_legacy(src, out)

    with h5py.File(out, "r") as h5:
        (entry, *_) = json.loads(h5["provenance"][()])["history"]
    assert entry["extra"]["model_metadata_json"] == '{"model": "hrnet"}'
