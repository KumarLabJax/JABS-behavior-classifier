"""Tests for DANDI subject-metadata pre-flight validation."""

import datetime
import logging

import numpy as np
import pytest

from jabs.core.abstract.pose_est import PoseEstimation
from jabs.core.types.pose import PoseData
from jabs.scripts.cli.dandi_subject_metadata import (
    subject_metadata_problems,
    validate_subjects,
)

VALID = {
    "subject_id": "M123",
    "species": "Mus musculus",
    "sex": "M",
    "age": "P70D",
}


def _pose_data(
    num_identities: int = 2,
    external_ids: list[str] | None = None,
    subjects: dict[str, dict] | None = None,
) -> PoseData:
    """Build a minimal PoseData carrying the given identities and subject metadata."""
    body_parts = [kpt.name for kpt in PoseEstimation.KeypointIndex]
    num_keypoints = len(body_parts)
    return PoseData(
        points=np.zeros((num_identities, 4, num_keypoints, 2)),
        point_mask=np.ones((num_identities, 4, num_keypoints), dtype=bool),
        identity_mask=np.ones((num_identities, 4), dtype=bool),
        body_parts=body_parts,
        edges=[],
        fps=30,
        external_ids=external_ids,
        subjects=subjects,
    )


# ---------------------------------------------------------------------------
# subject_metadata_problems - the pure per-identity rules
# ---------------------------------------------------------------------------


def test_valid_metadata_has_no_problems() -> None:
    """Fully specified metadata reports no problems."""
    assert subject_metadata_problems(VALID) == []


def test_date_of_birth_substitutes_for_age() -> None:
    """date_of_birth satisfies the age requirement on its own."""
    meta = {**VALID}
    del meta["age"]
    meta["date_of_birth"] = "2024-01-15T00:00:00+00:00"
    assert subject_metadata_problems(meta) == []


def test_empty_metadata_reports_the_three_dandi_critical_fields() -> None:
    """An identity with no metadata reports exactly what nwbinspector flags CRITICAL."""
    problems = subject_metadata_problems({}, default_subject_id="subject_1")

    assert problems == [
        "species is missing",
        "sex is missing",
        "age or date_of_birth is missing",
    ]


def test_absent_subject_id_is_not_a_problem_when_defaulted() -> None:
    """The writer defaults subject_id, so requiring it in the dict would be a false positive."""
    meta = {k: v for k, v in VALID.items() if k != "subject_id"}

    assert subject_metadata_problems(meta, default_subject_id="mouse_a") == []


def test_defaulted_subject_id_with_slash_is_caught() -> None:
    """An unsanitized external ID becomes the subject_id, so its slashes still matter."""
    meta = {k: v for k, v in VALID.items() if k != "subject_id"}

    problems = subject_metadata_problems(meta, default_subject_id="mouse/a")

    assert len(problems) == 1
    assert "contains '/'" in problems[0]


def test_subject_id_missing_entirely() -> None:
    """With neither a supplied nor a default subject_id, it is reported missing."""
    meta = {k: v for k, v in VALID.items() if k != "subject_id"}

    assert "subject_id is missing" in subject_metadata_problems(meta, default_subject_id="")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("species", None),
        ("species", ""),
        ("species", "   "),
        ("sex", None),
        ("sex", ""),
    ],
    ids=["species-none", "species-empty", "species-blank", "sex-none", "sex-empty"],
)
def test_blank_required_fields_count_as_missing(field: str, value: str | None) -> None:
    """None, empty and whitespace-only values are all treated as missing."""
    problems = subject_metadata_problems({**VALID, field: value})

    assert problems == [f"{field} is missing"]


@pytest.mark.parametrize(
    "species",
    ["Mus musculus", "Rattus norvegicus", "http://purl.obolibrary.org/obo/NCBITaxon_10090"],
    ids=["mouse", "rat", "ncbi-iri"],
)
def test_accepted_species_forms(species: str) -> None:
    """Latin binomials and NCBI taxonomy IRIs are accepted."""
    assert subject_metadata_problems({**VALID, "species": species}) == []


@pytest.mark.parametrize(
    "species",
    ["mouse", "Mus Musculus", "mus musculus", "Mus"],
    ids=["common-name", "capitalized-epithet", "lowercase-genus", "genus-only"],
)
def test_rejected_species_forms(species: str) -> None:
    """Common names and mis-capitalized binomials are rejected."""
    problems = subject_metadata_problems({**VALID, "species": species})

    assert len(problems) == 1
    assert "Latin binomial" in problems[0]


@pytest.mark.parametrize("sex", ["M", "F", "O", "U"], ids=["m", "f", "o", "u"])
def test_accepted_sex_values(sex: str) -> None:
    """The four DANDI sex codes are accepted."""
    assert subject_metadata_problems({**VALID, "sex": sex}) == []


@pytest.mark.parametrize(
    "sex", ["male", "m", "female", "X"], ids=["male", "lower-m", "female", "x"]
)
def test_rejected_sex_values(sex: str) -> None:
    """Sex is case-sensitive and restricted to the four codes."""
    problems = subject_metadata_problems({**VALID, "sex": sex})

    assert len(problems) == 1
    assert "must be one of" in problems[0]


@pytest.mark.parametrize("sex", ["XO", "XX"], ids=["xo", "xx"])
def test_c_elegans_uses_its_own_sex_vocabulary(sex: str) -> None:
    """C. elegans uses XO/XX rather than M/F/O/U."""
    meta = {**VALID, "species": "Caenorhabditis elegans", "sex": sex}

    assert subject_metadata_problems(meta) == []


def test_c_elegans_rejects_the_default_sex_vocabulary() -> None:
    """C. elegans rejects M/F/O/U, which are valid for other species."""
    meta = {**VALID, "species": "Caenorhabditis elegans", "sex": "M"}

    problems = subject_metadata_problems(meta)

    assert len(problems) == 1
    assert "'XO'" in problems[0]


@pytest.mark.parametrize(
    "age",
    ["P70D", "P2Y", "P23W", "PT12H", "P1D/P3D", "P90Y/", "/P3D"],
    ids=["days", "years", "weeks", "hours", "range", "open-upper", "open-lower"],
)
def test_accepted_age_forms(age: str) -> None:
    """ISO 8601 durations and ranges, including open-ended ones, are accepted."""
    assert subject_metadata_problems({**VALID, "age": age}) == []


@pytest.mark.parametrize(
    "age",
    ["70 days", "P70", "P", "70D", "P1D/bogus"],
    ids=["prose", "no-unit", "bare-p", "no-prefix", "bad-range-bound"],
)
def test_rejected_age_forms(age: str) -> None:
    """Prose ages and malformed durations are rejected."""
    problems = subject_metadata_problems({**VALID, "age": age})

    assert len(problems) == 1
    assert "ISO 8601 duration" in problems[0]


def test_unparseable_date_of_birth_is_caught() -> None:
    """A non-ISO date_of_birth is caught here rather than failing mid-write."""
    meta = {**VALID, "date_of_birth": "15/01/2024"}

    problems = subject_metadata_problems(meta)

    assert len(problems) == 1
    assert "ISO 8601 datetime" in problems[0]


def test_datetime_date_of_birth_is_accepted() -> None:
    """An already-parsed datetime is accepted as date_of_birth."""
    meta = {**VALID, "date_of_birth": datetime.datetime(2024, 1, 15, tzinfo=datetime.timezone.utc)}

    assert subject_metadata_problems(meta) == []


@pytest.mark.parametrize(
    "weight", ["25 g", "0.025 kg", "1 mg"], ids=["grams", "kilograms", "milligrams"]
)
def test_accepted_weight_forms(weight: str) -> None:
    """Weight is accepted as '[numeric] [unit]' with a space."""
    assert subject_metadata_problems({**VALID, "weight": weight}) == []


def test_weight_without_a_space_is_rejected() -> None:
    """'25g' is the form the docs used to recommend; nwbinspector flags it CRITICAL."""
    problems = subject_metadata_problems({**VALID, "weight": "25g"})

    assert len(problems) == 1
    assert "[numeric] [unit]" in problems[0]


def test_absent_weight_is_not_a_problem() -> None:
    """Weight is optional, so omitting it is fine."""
    assert subject_metadata_problems(VALID) == []


def test_all_problems_are_reported_together() -> None:
    """Every failed rule is reported, not just the first."""
    meta = {"subject_id": "M1", "species": "mouse", "sex": "male", "age": "70 days"}

    assert len(subject_metadata_problems(meta)) == 3


# ---------------------------------------------------------------------------
# validate_subjects - resolution against a PoseData
# ---------------------------------------------------------------------------


def test_validate_passes_with_metadata_for_every_identity() -> None:
    """Validation passes when every identity has complete metadata."""
    data = _pose_data(subjects={"subject_1": VALID, "subject_2": VALID})

    validate_subjects(data)


def test_validate_raises_without_any_subjects() -> None:
    """With no subjects at all, every identity is reported."""
    data = _pose_data(subjects=None)

    with pytest.raises(ValueError, match="missing or malformed") as exc:
        validate_subjects(data)

    message = str(exc.value)
    assert "subject_1: species is missing" in message
    assert "subject_2: species is missing" in message


def test_validate_names_only_the_failing_identity() -> None:
    """Identities with valid metadata are not named in the error."""
    data = _pose_data(subjects={"subject_1": VALID})

    with pytest.raises(ValueError) as exc:
        validate_subjects(data)

    message = str(exc.value)
    assert "subject_2" in message
    assert "subject_1" not in message.split("\n")[1]


def test_validate_resolves_metadata_by_raw_external_id() -> None:
    """subjects is keyed by the raw external ID, not the sanitized container name."""
    data = _pose_data(num_identities=1, external_ids=["mouse/a"], subjects={"mouse/a": {**VALID}})

    validate_subjects(data)


def test_validate_falls_back_to_the_sanitized_name() -> None:
    """A subjects key matching the sanitized name also resolves."""
    data = _pose_data(num_identities=1, external_ids=["mouse/a"], subjects={"mouse_a": {**VALID}})

    validate_subjects(data)


def test_validate_warns_about_unused_subjects_keys(caplog) -> None:
    """A key matching no identity is the usual cause of an otherwise baffling failure."""
    data = _pose_data(subjects={"subject_1": VALID, "subject_2": VALID, "subject_0": VALID})

    with caplog.at_level(logging.WARNING):
        validate_subjects(data)

    assert "subject_0" in caplog.text
    assert "match no identity" in caplog.text
    assert "subject_1, subject_2" in caplog.text


def test_unused_key_does_not_itself_raise(caplog) -> None:
    """The mismatch is a warning; it only fails via the identity left without metadata."""
    data = _pose_data(subjects={"subject_1": VALID, "typo": VALID})

    with caplog.at_level(logging.WARNING), pytest.raises(ValueError) as exc:
        validate_subjects(data)

    assert "typo" in caplog.text
    assert "subject_2" in str(exc.value)
    assert "typo" not in str(exc.value)


def test_no_warning_when_every_key_matches(caplog) -> None:
    """No unused-key warning is logged when all keys resolve."""
    data = _pose_data(subjects={"subject_1": VALID, "subject_2": VALID})

    with caplog.at_level(logging.WARNING):
        validate_subjects(data)

    assert "match no identity" not in caplog.text
