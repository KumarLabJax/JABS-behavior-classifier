"""Alternates and their ragged index: construction rules and manifest entries."""

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from jabs.io.internal.pose_file import Alternate, RaggedIndex
from jabs.io.internal.pose_file.blocks import BlockSource
from jabs.io.internal.pose_file.manifest import build_manifest
from jabs.io.internal.pose_file.schema import validate_manifest
from jabs.io.internal.pose_file.types import Component, PoseFile, VideoInfo, alternate_key

from .conftest import ragged_segmentation

PREFIX = "/jabs/segmentation/ragged"
FRAMES, SLOTS = 4, 2


def _index(seg: dict[str, npt.NDArray[np.generic]], prefix: str = PREFIX) -> RaggedIndex:
    """The shared offsets of the segmentation's ragged alternates."""
    return RaggedIndex(
        group_offsets_path=f"{prefix}/contour_offsets",
        instance_offsets_path=f"{prefix}/instance_offsets",
        group_offsets=seg["contour_offsets"],
        instance_offsets=seg["instance_offsets"],
    )


def _contours_alternate(
    seg: dict[str, npt.NDArray[np.generic]], index: RaggedIndex, **overrides: Any
) -> Alternate:
    """The leaf-level ragged alternate of the contours."""
    fields: dict[str, Any] = {
        "name": "ragged",
        "path": f"{PREFIX}/contours",
        "axes": ("point", "coord"),
        "data": seg["points"],
        "missing": {"policy": "none"},
        "index": index,
        "level": "leaf",
        "units": "pixel",
        "coord_order": "xy",
    }
    fields.update(overrides)
    return Alternate(**fields)


def _flags_alternate(
    seg: dict[str, npt.NDArray[np.generic]], index: RaggedIndex, **overrides: Any
) -> Alternate:
    """The group-level ragged alternate of the external flags."""
    fields: dict[str, Any] = {
        "name": "ragged",
        "path": f"{PREFIX}/external_flag",
        "axes": ("contour",),
        "data": seg["flags"],
        "missing": {"policy": "none"},
        "index": index,
        "level": "group",
    }
    fields.update(overrides)
    return Alternate(**fields)


def _segmentation_file(
    seg: dict[str, npt.NDArray[np.generic]],
    contours_alternates: tuple[Alternate, ...] = (),
    flag_alternates: tuple[Alternate, ...] = (),
) -> PoseFile:
    """Dense segmentation, with the given alternates on contours and flags."""
    return PoseFile(
        dimensions={"frame": FRAMES, "slot": SLOTS, "identity": SLOTS},
        video=VideoInfo(frame_count=FRAMES, width=800, height=800, fps=30.0),
        components=(
            Component(
                id="jabs.segmentation.contours",
                axes=("frame", "slot", "contour", "point", "coord"),
                data=seg["contours"],
                missing={"policy": "length", "length": "jabs.segmentation.contour_length"},
                units="pixel",
                coord_order="xy",
                alternates=contours_alternates,
            ),
            Component(
                id="jabs.segmentation.contour_count",
                axes=("frame", "slot"),
                data=seg["contour_count"],
                missing={"policy": "none"},
            ),
            Component(
                id="jabs.segmentation.contour_length",
                axes=("frame", "slot", "contour"),
                data=seg["contour_length"],
                missing={"policy": "none"},
            ),
            Component(
                id="jabs.segmentation.external_flag",
                axes=("frame", "slot", "contour"),
                data=seg["external_flag"],
                missing={"policy": "none"},
                alternates=flag_alternates,
            ),
        ),
    )


def _no_blocks() -> Iterable[tuple[int, npt.NDArray[np.generic]]]:
    """Yield nothing."""
    return iter(())


def test_ragged_segmentation_decodes_to_its_dense_form() -> None:
    """The shared helper's ragged form follows the ADR's decoding rule."""
    seg = ragged_segmentation()
    go, io = seg["contour_offsets"], seg["instance_offsets"]
    assert len(io) == FRAMES * SLOTS + 1
    assert int(go[-1]) == len(seg["points"])
    assert int(io[-1]) == len(go) - 1 == len(seg["flags"])
    for f in range(FRAMES):
        for s in range(SLOTS):
            groups = range(int(io[f * SLOTS + s]), int(io[f * SLOTS + s + 1]))
            assert len(groups) == seg["contour_count"][f, s]
            for c, g in enumerate(groups):
                points = seg["points"][int(go[g]) : int(go[g + 1])]
                length = int(seg["contour_length"][f, s, c])
                np.testing.assert_array_equal(points, seg["contours"][f, s, c, :length])
                assert seg["flags"][g] == seg["external_flag"][f, s, c]
    # Non-trivial: some instances own no contours, some own two.
    assert 0 in seg["contour_count"] and 2 in seg["contour_count"]


def _bad_name(seg: dict, index: RaggedIndex) -> Alternate:
    """A name with an uppercase letter."""
    return _contours_alternate(seg, index, name="Ragged")


def _bad_level(seg: dict, index: RaggedIndex) -> Alternate:
    """A level that is neither leaf nor group."""
    return _contours_alternate(seg, index, level="middle")


def _rank_mismatch(seg: dict, index: RaggedIndex) -> Alternate:
    """One axis for a two-dimensional payload."""
    return _contours_alternate(seg, index, axes=("point",))


def _coord_without_units(seg: dict, index: RaggedIndex) -> Alternate:
    """A coord axis without units."""
    return _contours_alternate(seg, index, units=None)


def _coord_without_order(seg: dict, index: RaggedIndex) -> Alternate:
    """A coord axis without coord_order."""
    return _contours_alternate(seg, index, coord_order=None)


def _coord_three_wide(seg: dict, index: RaggedIndex) -> Alternate:
    """A coord axis three values wide."""
    return _contours_alternate(seg, index, data=np.zeros((5, 3), dtype=np.int32))


def _offsets_not_uint64(seg: dict, index: RaggedIndex) -> RaggedIndex:
    """Signed offsets."""
    return RaggedIndex(
        group_offsets_path=index.group_offsets_path,
        instance_offsets_path=index.instance_offsets_path,
        group_offsets=seg["contour_offsets"].astype(np.int64),
        instance_offsets=seg["instance_offsets"],
    )


def _offsets_not_1d(seg: dict, index: RaggedIndex) -> RaggedIndex:
    """Two-dimensional offsets."""
    return RaggedIndex(
        group_offsets_path=index.group_offsets_path,
        instance_offsets_path=index.instance_offsets_path,
        group_offsets=seg["contour_offsets"],
        instance_offsets=BlockSource((3, 3), "uint64", _no_blocks),
    )


def _string_payload(seg: dict, index: RaggedIndex) -> Alternate:
    """Text, which no defined ragged alternate holds."""
    return _flags_alternate(seg, index, data=np.array(["a", "b"]), axes=("contour",))


@pytest.mark.parametrize(
    ("build", "match"),
    [
        (_bad_name, "invalid alternate name"),
        (_bad_level, "level must be one of"),
        (_rank_mismatch, "declared 1 axes"),
        (_coord_without_units, "must declare units and coord_order"),
        (_coord_without_order, "must declare units and coord_order"),
        (_coord_three_wide, "coord axis is 3 wide"),
        (_offsets_not_uint64, "group_offsets .* must be one-dimensional uint64"),
        (_offsets_not_1d, "instance_offsets .* must be one-dimensional uint64"),
        (_string_payload, "cannot hold string data"),
    ],
)
def test_alternate_validates_its_declarations(
    build: Callable[[dict, RaggedIndex], object], match: str
) -> None:
    """Each malformed declaration is refused at construction, for its own reason."""
    seg = ragged_segmentation()
    index = _index(seg)
    with pytest.raises(ValueError, match=match):
        build(seg, index)


def test_an_alternate_freezes_what_it_validated() -> None:
    """Axes, missing and extra cannot be mutated past validation."""
    seg = ragged_segmentation()
    alternate = _contours_alternate(
        seg, _index(seg), axes=["point", "coord"], extra={"org.example": {"a": 1}}
    )
    assert alternate.axes == ("point", "coord")
    with pytest.raises(TypeError):
        alternate.missing["policy"] = "nan"  # type: ignore[index]
    with pytest.raises(TypeError):
        alternate.extra["org.example"] = {}  # type: ignore[index]
    assert alternate.dtype == "int32"


def test_a_block_source_is_accepted_and_never_read() -> None:
    """Payload and offsets may be block sources; construction reads only shapes."""

    def never_read() -> Iterable[tuple[int, npt.NDArray[np.generic]]]:
        raise AssertionError("construction read a block source")

    index = RaggedIndex(
        group_offsets_path=f"{PREFIX}/contour_offsets",
        instance_offsets_path=f"{PREFIX}/instance_offsets",
        group_offsets=BlockSource((6,), "uint64", never_read),
        instance_offsets=BlockSource((FRAMES * SLOTS + 1,), "uint64", never_read),
    )
    alternate = _contours_alternate(
        ragged_segmentation(), index, data=BlockSource((12, 2), "int32", never_read)
    )
    assert alternate.dtype == "int32"


def test_a_component_refuses_two_alternates_of_one_name() -> None:
    """A name identifies one encoding, so it appears once per component."""
    seg = ragged_segmentation()
    index = _index(seg)
    first = _contours_alternate(seg, index)
    second = _contours_alternate(seg, index, path=f"{PREFIX}/contours_again")
    with pytest.raises(ValueError, match="ragged"):
        _segmentation_file(seg, contours_alternates=(first, second))


def test_a_component_keeps_alternates_given_as_a_generator() -> None:
    """Alternates from a one-shot iterable survive the duplicate-name check, in order."""
    seg = ragged_segmentation()
    index = _index(seg)
    first = _contours_alternate(seg, index)
    second = _contours_alternate(seg, index, name="copy", path=f"{PREFIX}/copy")
    pose = _segmentation_file(seg, contours_alternates=(a for a in (first, second)))  # type: ignore[arg-type]
    assert pose.component("jabs.segmentation.contours").alternates == (first, second)


def test_pose_file_refuses_alternate_path_collisions() -> None:
    """An alternate's path is distinct from every component's and alternate's."""
    seg = ragged_segmentation()
    index = _index(seg)
    on_a_component = _contours_alternate(seg, index, path="/jabs/segmentation/contour_count")
    with pytest.raises(ValueError, match="/jabs/segmentation/contour_count"):
        _segmentation_file(seg, contours_alternates=(on_a_component,))

    contours = _contours_alternate(seg, index)
    flags = _flags_alternate(seg, index, path=f"{PREFIX}/contours")
    with pytest.raises(ValueError, match=f"{PREFIX}/contours"):
        _segmentation_file(seg, contours_alternates=(contours,), flag_alternates=(flags,))


def test_alternates_of_one_name_share_one_index() -> None:
    """Two indexes of one name at the same paths would be written twice."""
    seg = ragged_segmentation()
    contours = _contours_alternate(seg, _index(seg))
    flags = _flags_alternate(seg, _index(seg))
    with pytest.raises(ValueError, match="ragged"):
        _segmentation_file(seg, contours_alternates=(contours,), flag_alternates=(flags,))

    shared = _index(seg)
    pose = _segmentation_file(
        seg,
        contours_alternates=(_contours_alternate(seg, shared),),
        flag_alternates=(_flags_alternate(seg, shared),),
    )
    assert (
        pose.component("jabs.segmentation.contours").alternates[0].index
        is pose.component("jabs.segmentation.external_flag").alternates[0].index
    )


def test_an_alternate_missing_reference_must_be_declared() -> None:
    """A mask or length an alternate names must be a component, as for a component."""
    seg = ragged_segmentation()
    typo = _contours_alternate(
        seg,
        _index(seg),
        missing={"policy": "length", "length": "jabs.segmentation.contour_lenght"},
    )
    with pytest.raises(ValueError, match="contour_lenght"):
        _segmentation_file(seg, contours_alternates=(typo,))


def test_an_alternate_missing_reference_must_align() -> None:
    """A reference whose axes are not the alternate's leading axes is refused."""
    seg = ragged_segmentation()
    misaligned = _contours_alternate(
        seg,
        _index(seg),
        missing={"policy": "length", "length": "jabs.segmentation.contour_length"},
    )
    with pytest.raises(ValueError, match="not the leading axes"):
        _segmentation_file(seg, contours_alternates=(misaligned,))


def test_an_alternate_reference_typo_is_refused_before_any_block_is_read() -> None:
    """Refused at construction, so a streamed payload is never consumed."""

    def never_read() -> Iterable[tuple[int, npt.NDArray[np.generic]]]:
        raise AssertionError("a block source was read")

    seg = ragged_segmentation()
    typo = _contours_alternate(
        seg,
        _index(seg),
        data=BlockSource(seg["points"].shape, seg["points"].dtype, never_read),
        missing={"policy": "mask", "mask": "jabs.segmentation.no_such_mask"},
    )
    with pytest.raises(ValueError, match="no_such_mask"):
        _segmentation_file(seg, contours_alternates=(typo,))


def test_two_names_at_one_offset_path_share_one_index() -> None:
    """Different indexes at one pair of offset paths are refused, whatever their names.

    They are not two datasets but one shared by mistake, so the refusal says
    what to do instead.
    """
    seg = ragged_segmentation()
    index = _index(seg)
    contours = _contours_alternate(seg, index)
    flags = _flags_alternate(seg, index)
    copy = _flags_alternate(seg, _index(seg), name="copy", path=f"{PREFIX}/copy")
    with pytest.raises(ValueError, match="must share one"):
        _segmentation_file(seg, contours_alternates=(contours,), flag_alternates=(flags, copy))


def test_alternate_key_names_a_component_alternate() -> None:
    """The key layouts and history entries use."""
    assert (
        alternate_key("jabs.segmentation.contours", "ragged")
        == "jabs.segmentation.contours:alternate:ragged"
    )


def test_manifest_entry_for_an_alternate() -> None:
    """Leaf and group alternates are declared as ADR 0002 shows them."""
    seg = ragged_segmentation()
    index = _index(seg)
    pose = _segmentation_file(
        seg,
        contours_alternates=(_contours_alternate(seg, index),),
        flag_alternates=(_flags_alternate(seg, index),),
    )
    layout = {"storage": "chunked", "chunks": [65536, 2], "compression": "gzip"}
    manifest = build_manifest(
        pose,
        layouts={alternate_key("jabs.segmentation.contours", "ragged"): layout},
        created="2026-11-18T00:00:00Z",
    )
    assert validate_manifest(manifest) == []
    by_id = {entry["id"]: entry for entry in manifest["components"]}
    encoding_paths = {
        "group_offsets": f"{PREFIX}/contour_offsets",
        "instance_offsets": f"{PREFIX}/instance_offsets",
    }
    assert by_id["jabs.segmentation.contours"]["alternates"] == [
        {
            "name": "ragged",
            "path": f"{PREFIX}/contours",
            "axes": ["point", "coord"],
            "dtype": "int32",
            "shape": [len(seg["points"]), 2],
            "units": "pixel",
            "coord_order": "xy",
            "missing": {"policy": "none"},
            "encoding": {"kind": "ragged", "level": "leaf", **encoding_paths},
            "layout": layout,
        }
    ]
    # The order the conformance fixture's entries have, so a manifest built
    # here serializes the same way.
    contours_entry = by_id["jabs.segmentation.contours"]["alternates"][0]
    assert list(contours_entry) == [
        "name",
        "path",
        "axes",
        "dtype",
        "shape",
        "units",
        "coord_order",
        "encoding",
        "missing",
        "layout",
    ]
    assert list(contours_entry["encoding"]) == [
        "kind",
        "level",
        "group_offsets",
        "instance_offsets",
    ]
    assert by_id["jabs.segmentation.external_flag"]["alternates"] == [
        {
            "name": "ragged",
            "path": f"{PREFIX}/external_flag",
            "axes": ["contour"],
            "dtype": "bool",
            "shape": [len(seg["flags"])],
            "missing": {"policy": "none"},
            "encoding": {"kind": "ragged", "level": "group", **encoding_paths},
        }
    ]


def test_a_component_without_alternates_has_no_alternates_key() -> None:
    """Existing manifests stay byte-identical."""
    manifest = build_manifest(_segmentation_file(ragged_segmentation()))
    assert all("alternates" not in entry for entry in manifest["components"])


# HDF5 folds "a//b" into "a/b" and ignores a trailing slash, so a path spelled
# either way aliases another while comparing unequal to it as a string.
ALIASING_PATHS = [
    pytest.param("/jabs/segmentation//ragged/contours", id="double-slash"),
    pytest.param(f"{PREFIX}/contours/", id="trailing-slash"),
    pytest.param("relative/contours", id="relative"),
]


@pytest.mark.parametrize("path", ALIASING_PATHS)
def test_an_alternate_path_must_be_canonical(path: str) -> None:
    """``/jabs/segmentation//ragged/contours`` beside ``.../ragged/contours`` is one dataset."""
    seg = ragged_segmentation()
    with pytest.raises(ValueError, match="path"):
        _contours_alternate(seg, _index(seg), path=path)


@pytest.mark.parametrize("field", ["group_offsets_path", "instance_offsets_path"])
@pytest.mark.parametrize("path", ALIASING_PATHS)
def test_an_offsets_path_must_be_canonical(field: str, path: str) -> None:
    """The offsets are datasets too, and alias the same way."""
    seg = ragged_segmentation()
    fields = {
        "group_offsets_path": f"{PREFIX}/contour_offsets",
        "instance_offsets_path": f"{PREFIX}/instance_offsets",
        "group_offsets": seg["contour_offsets"],
        "instance_offsets": seg["instance_offsets"],
    }
    with pytest.raises(ValueError, match="path"):
        RaggedIndex(**{**fields, field: path})


@pytest.mark.parametrize(
    "group_path,instance_path",
    [
        pytest.param(f"{PREFIX}/contour_offsets", f"{PREFIX}/instance_offsets_b", id="same-group"),
        pytest.param(
            f"{PREFIX}/group_offsets_b", f"{PREFIX}/instance_offsets", id="same-instance"
        ),
        pytest.param(f"{PREFIX}/instance_offsets", f"{PREFIX}/other", id="crossed-roles"),
    ],
)
def test_an_offsets_dataset_has_one_partner(group_path: str, instance_path: str) -> None:
    """Two indexes sharing one offsets path but not the other are refused.

    The reviewer's reproduction: a second leaf alternate ``copy`` on the
    contours, with the ``ragged`` alternate's group_offsets and its own
    instance_offsets. The writer would list the shared path once per index and
    refuse the file as a path collision, after it had been read clean.
    """
    seg = ragged_segmentation()
    index = _index(seg)
    other = RaggedIndex(
        group_offsets_path=group_path,
        instance_offsets_path=instance_path,
        group_offsets=seg["contour_offsets"],
        instance_offsets=seg["instance_offsets"],
    )
    contours = _contours_alternate(seg, index)
    copy = _contours_alternate(seg, other, name="copy", path=f"{PREFIX}/copy")
    with pytest.raises(ValueError, match="one partner"):
        _segmentation_file(
            seg,
            contours_alternates=(contours, copy),
            flag_alternates=(_flags_alternate(seg, index),),
        )


def _never_read() -> Iterable[tuple[int, npt.NDArray[np.generic]]]:
    raise AssertionError("a block source was read")


def _unread(array: npt.NDArray[np.generic]) -> BlockSource:
    """A block source of the array's shape and dtype that fails if it is read."""
    return BlockSource(array.shape, array.dtype, _never_read)


@pytest.mark.parametrize("streamed", [False, True], ids=["in-memory", "block-source"])
def test_a_group_level_alternate_needs_a_leaf_level_partner(streamed: bool) -> None:
    """The validator's ragged_group_pairing, checked before anything is written.

    Otherwise the write streams every block source and is then refused by the
    post-write validation.
    """
    seg = ragged_segmentation()
    index = _index(seg)
    if streamed:
        index = RaggedIndex(
            group_offsets_path=index.group_offsets_path,
            instance_offsets_path=index.instance_offsets_path,
            group_offsets=_unread(seg["contour_offsets"]),
            instance_offsets=_unread(seg["instance_offsets"]),
        )
    data = _unread(seg["flags"]) if streamed else seg["flags"]
    flags = _flags_alternate(seg, index, data=data)
    with pytest.raises(ValueError, match="leaf-level"):
        _segmentation_file(seg, flag_alternates=(flags,))


@pytest.mark.parametrize("streamed", [False, True], ids=["in-memory", "block-source"])
def test_instance_offsets_cover_every_frame_and_slot(streamed: bool) -> None:
    """``len(instance_offsets) == frame*slot + 1``, from the shape alone."""
    seg = ragged_segmentation()
    short = seg["instance_offsets"][:-1]
    index = RaggedIndex(
        group_offsets_path=f"{PREFIX}/contour_offsets",
        instance_offsets_path=f"{PREFIX}/instance_offsets",
        group_offsets=_unread(seg["contour_offsets"]) if streamed else seg["contour_offsets"],
        instance_offsets=_unread(short) if streamed else short,
    )
    data = _unread(seg["points"]) if streamed else seg["points"]
    with pytest.raises(ValueError, match=r"frame\*slot\+1"):
        _segmentation_file(seg, contours_alternates=(_contours_alternate(seg, index, data=data),))
