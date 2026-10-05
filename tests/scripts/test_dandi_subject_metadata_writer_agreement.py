"""The DANDI pre-flight check and the NWB writer must agree on what is absent.

Each case here passed validation and then either crashed the write or reached the
archive as the CRITICAL finding the pre-flight exists to prevent, because the
validator treated any falsy value as absent while the writer only filtered
``None``. Both halves now share
:func:`~jabs.io.internal.pose.subject_value_is_absent`, and these tests exercise
the pair together - asserting the validator's verdict alone cannot catch the two
drifting apart again.

Needs the ``nwb`` extra: ``PoseNWBAdapter._make_subject`` builds a pynwb
``Subject``. The validator-only half of these rules lives in
``test_dandi_subject_metadata.py``, which stays runnable without it.
"""

import datetime
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pynwb")
pytest.importorskip("ndx_pose")
pytest.importorskip("ndx_jabs")

from pynwb import NWBHDF5IO

from jabs.core.abstract.pose_est import PoseEstimation
from jabs.core.types.pose import PoseData
from jabs.io.internal.pose import PoseNWBAdapter
from jabs.scripts.cli.dandi_subject_metadata import subject_metadata_problems

VALID = {
    "subject_id": "M123",
    "species": "Mus musculus",
    "sex": "M",
    "age": "P70D",
}


def _single_identity_pose(identity: str, subjects: dict[str, dict]) -> PoseData:
    """Build a minimal one-identity PoseData carrying the given subject metadata."""
    body_parts = [kpt.name for kpt in PoseEstimation.KeypointIndex]
    num_keypoints = len(body_parts)
    return PoseData(
        points=np.zeros((1, 4, num_keypoints, 2)),
        point_mask=np.ones((1, 4, num_keypoints), dtype=bool),
        identity_mask=np.ones((1, 4), dtype=bool),
        body_parts=body_parts,
        edges=[],
        fps=30,
        external_ids=[identity],
        subjects=subjects,
    )


@pytest.mark.parametrize(
    "field",
    ["species", "sex", "age", "subject_id", "date_of_birth", "weight"],
    ids=["species", "sex", "age", "subject_id", "date_of_birth", "weight"],
)
def test_blank_agrees_with_the_writer(field: str) -> None:
    """A blank value is absent to the validator and omitted by the writer alike."""
    meta = {**VALID, field: ""}
    problems = subject_metadata_problems(meta, default_subject_id="subject_1")
    written = PoseNWBAdapter._make_subject(meta)

    if field in ("species", "sex"):
        assert problems == [f"{field} is missing"]
    else:
        # age is unmet only because date_of_birth is absent too; the remaining
        # required fields are present, so a blank optional field is just dropped.
        assert problems == ([] if field != "age" else ["age or date_of_birth is missing"])
    assert getattr(written, field) is None


def test_blank_date_of_birth_does_not_crash_the_writer() -> None:
    """A blank date_of_birth used to raise ValueError inside the write loop."""
    meta = {**VALID, "date_of_birth": ""}

    assert subject_metadata_problems(meta) == []
    assert PoseNWBAdapter._make_subject(meta).date_of_birth is None


def test_blank_age_is_not_written_as_an_empty_age() -> None:
    """Subject(age="") tripped check_subject_age at the archive."""
    meta = {**VALID, "age": "", "date_of_birth": "2024-01-15T00:00:00+00:00"}

    assert subject_metadata_problems(meta) == []
    assert PoseNWBAdapter._make_subject(meta).age is None


@pytest.mark.parametrize("value", ["", None], ids=["blank", "null"])
def test_blank_subject_id_falls_back_like_the_writer(tmp_path: Path, value: str | None) -> None:
    """Both halves fall back to the identity name, so no empty id is written.

    The writer's fallback lives in the per-identity write loop rather than in
    ``_make_subject``, so this writes a real file and reads the Subject back.

    Args:
        tmp_path: Pytest temporary directory the NWB file is written to.
        value: The blank or null ``subject_id`` supplied for the identity.
    """
    meta = {**VALID, "subject_id": value}
    data = _single_identity_pose("mouse_a", {"mouse_a": meta})

    assert subject_metadata_problems(meta, default_subject_id="mouse_a") == []

    PoseNWBAdapter().write(data, tmp_path / "pose.nwb")

    with NWBHDF5IO(str(tmp_path / "pose_mouse_a.nwb"), mode="r") as io:
        assert io.read().subject.subject_id == "mouse_a"


def test_datetime_date_of_birth_survives_the_writer() -> None:
    """The validator accepts a datetime, so the writer has to take one too."""
    dob = datetime.datetime(2024, 1, 15, tzinfo=datetime.timezone.utc)
    meta = {**VALID, "date_of_birth": dob}

    assert subject_metadata_problems(meta) == []
    assert PoseNWBAdapter._make_subject(meta).date_of_birth == dob


def test_naive_datetime_date_of_birth_gets_utc() -> None:
    """A naive datetime is made timezone-aware, matching the string path."""
    meta = {**VALID, "date_of_birth": datetime.datetime(2024, 1, 15)}

    written = PoseNWBAdapter._make_subject(meta)

    assert written.date_of_birth.tzinfo is not None
