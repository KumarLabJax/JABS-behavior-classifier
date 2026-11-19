"""``check_self_contained`` finds every reference to data outside the file."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from jabs.io.internal.pose_file.hdf5_safety import check_self_contained
from jabs.io.internal.pose_file.writer import write_pose_file

from .conftest import OUTSIDE_KINDS, add_outside_reference
from .convert.conftest import SAMPLES, legacy, requires_samples

POSE_FORMAT = Path(__file__).parents[2] / "data" / "pose-format"


@pytest.mark.parametrize("kind", sorted(OUTSIDE_KINDS))
def test_each_outside_reference_is_named(tmp_path: Path, kind: str) -> None:
    """A link or storage outside the file is one problem naming its path and kind."""
    path = tmp_path / "file.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("inside", data=np.zeros(3))
        add_outside_reference(h5, "/group/nested/leak", kind, tmp_path)

    with h5py.File(path, "r") as h5:
        problems = check_self_contained(h5)

    assert len(problems) == 1, problems
    assert problems[0].startswith("/group/nested/leak ")
    assert OUTSIDE_KINDS[kind] in problems[0]


def test_every_problem_is_reported(tmp_path: Path) -> None:
    """The walk goes on past the first problem, so a caller can name them all."""
    path = tmp_path / "file.h5"
    with h5py.File(path, "w") as h5:
        for kind in OUTSIDE_KINDS:
            add_outside_reference(h5, f"/{kind}", kind, tmp_path)

    with h5py.File(path, "r") as h5:
        problems = check_self_contained(h5)

    assert sorted(p.split(" ")[0] for p in problems) == sorted(f"/{k}" for k in OUTSIDE_KINDS)


def test_a_dangling_external_link_is_refused(tmp_path: Path) -> None:
    """Whether the target exists does not matter: the link is never followed."""
    path = tmp_path / "file.h5"
    with h5py.File(path, "w") as h5:
        h5["leak"] = h5py.ExternalLink("/no/such/file.h5", "/data")

    with h5py.File(path, "r") as h5:
        assert check_self_contained(h5) == ["/leak is an external link to /no/such/file.h5:/data"]


def test_a_soft_link_loop_terminates(tmp_path: Path) -> None:
    """A soft link to its own group is refused, not followed round and round."""
    path = tmp_path / "file.h5"
    with h5py.File(path, "w") as h5:
        h5.create_group("a")["again"] = h5py.SoftLink("/a")

    with h5py.File(path, "r") as h5:
        assert check_self_contained(h5) == ["/a/again is a soft link to /a"]


def test_hard_links_are_inside_the_file(tmp_path: Path) -> None:
    """A second name for an object, even one that makes a cycle, is not a problem."""
    path = tmp_path / "file.h5"
    with h5py.File(path, "w") as h5:
        group = h5.create_group("a")
        group["data"] = np.arange(3)
        h5["b"] = group["data"]
        # A cycle: the group holds a hard link to itself.
        group["self"] = group

    with h5py.File(path, "r") as h5:
        assert check_self_contained(h5) == []


def test_a_written_pose_file_is_self_contained(tmp_path: Path, sample_pose_file) -> None:
    """What the writer produces passes."""
    path = tmp_path / "pose.h5"
    write_pose_file(sample_pose_file, path)
    with h5py.File(path, "r") as h5:
        assert check_self_contained(h5) == []


@pytest.mark.parametrize(
    "path",
    [
        *sorted(POSE_FORMAT.glob("valid-*.h5")),
        *sorted(legacy("").glob("*.h5")),
    ],
    ids=lambda p: p.name,
)
def test_the_corpora_are_self_contained(path: Path) -> None:
    """No committed fixture, valid or legacy, holds an outside reference."""
    with h5py.File(path, "r") as h5:
        assert check_self_contained(h5) == []


@requires_samples
@pytest.mark.parametrize(
    "name", ["sample_pose_est_v4.h5", "sample_pose_est_v5.h5", "sample_pose_est_v6.h5"]
)
def test_the_public_samples_are_self_contained(name: str) -> None:
    """Real legacy files pass."""
    with h5py.File(SAMPLES / name, "r") as h5:
        assert check_self_contained(h5) == []


def test_a_hard_link_to_the_root_is_walked_once(tmp_path: Path) -> None:
    """A group holding the root again does not report the root's problems twice."""
    path = tmp_path / "file.h5"
    with h5py.File(path, "w") as h5:
        h5["leak"] = h5py.SoftLink("/elsewhere")
        h5.create_group("a")["root"] = h5["/"]

    with h5py.File(path, "r") as h5:
        assert check_self_contained(h5) == ["/leak is a soft link to /elsewhere"]
