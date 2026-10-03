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

import pytest

pytest.importorskip("pynwb")
pytest.importorskip("ndx_pose")

from jabs.io.internal.pose import PoseNWBAdapter
from jabs.scripts.cli.dandi_subject_metadata import subject_metadata_problems

VALID = {
    "subject_id": "M123",
    "species": "Mus musculus",
    "sex": "M",
    "age": "P70D",
}


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
    assert getattr(written, field, None) in (None, "M123", "Mus musculus", "M")


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
def test_blank_subject_id_falls_back_like_the_writer(value: str | None) -> None:
    """Both halves fall back to the identity name, so no empty id is written."""
    meta = {**VALID, "subject_id": value}

    assert subject_metadata_problems(meta, default_subject_id="subject_1") == []
    assert PoseNWBAdapter._make_subject({**meta, "subject_id": "subject_1"}).subject_id == (
        "subject_1"
    )


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
