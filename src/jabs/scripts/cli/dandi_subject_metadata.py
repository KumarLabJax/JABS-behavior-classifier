"""Pre-flight validation of NWB subject metadata against DANDI's requirements.

JABS NWB output is published to the EMBER-DANDI archive, which runs
``nwbinspector`` and refuses any file carrying a ``CRITICAL`` finding. Several of
those findings concern ``Subject`` metadata that cannot be derived from a pose
file at all - species, sex and age are biological facts only the user can supply,
via ``--subjects``. Without them the converter would write a full set of files
that the archive then rejects, so these checks run before anything is written.

The rules mirror ``nwbinspector``'s own subject checks - ``check_subject_age``,
``check_subject_sex``, ``check_subject_id_exists``, ``check_subject_id_no_slashes``,
``check_subject_species_exists``, ``check_subject_species_form`` and
``check_subject_weight`` - all of which DANDI's config elevates to ``CRITICAL``.
They are reimplemented here rather than imported because ``nwbinspector`` is not
a JABS dependency; the regexes below are kept identical to theirs so the two
agree.

This is a pre-flight check, not a replacement: run ``nwbinspector`` against the
converted files for the authoritative verdict before uploading.
"""

import datetime
import logging
import re

from jabs.core.types.pose import PoseData
from jabs.io.internal.pose import resolve_identity_subjects

logger = logging.getLogger(__name__)

# Kept byte-identical to nwbinspector's `species_form_regex`, `duration_regex` and
# `weight_form_regex` so this pre-flight check agrees with the archive's validator.
_SPECIES_FORM = re.compile(
    r"([A-Z][a-z]* [a-z]+)|(http://purl\.obolibrary\.org/obo/NCBITaxon_\d+)"
)
_DURATION = re.compile(
    r"^P(?!$)(\d+(?:\.\d+)?Y)?(\d+(?:\.\d+)?M)?(\d+(?:\.\d+)?W)?(\d+(?:\.\d+)?D)?"
    r"(T(?=\d)(\d+(?:\.\d+)?H)?(\d+(?:\.\d+)?M)?(\d+(?:\.\d+)?S)?)?$"
)
_WEIGHT_FORM = re.compile(r"(?i)^\d+(\.\d+)? (kg|g|mg|ug|μg|ng|pg)$")

# nwbinspector validates sex against a species-dependent vocabulary.
_SEX_VALUES = ("M", "F", "O", "U")
_C_ELEGANS_SEX_VALUES = ("XO", "XX")
_C_ELEGANS_SPECIES = ("Caenorhabditis elegans", "C. elegans")

_DOCS_REFERENCE = "docs/user-guide/nwb-export.md (Subjects JSON format)"


def _is_iso_duration(value: str) -> bool:
    """Return whether a string is an ISO 8601 duration, or a range of two.

    Mirrors ``check_subject_age``, which accepts a plain duration (``"P70D"``) as
    well as a ``/``-separated range with either bound optionally blank
    (``"P1D/P3D"``, ``"P90Y/"``).
    """
    if _DURATION.fullmatch(value):
        return True
    if "/" not in value:
        return False
    lower, _, upper = value.partition("/")
    return all(bound == "" or bool(_DURATION.fullmatch(bound)) for bound in (lower, upper))


def _missing(metadata: dict, field: str) -> bool:
    """Return whether a field is absent, None, or an empty/whitespace-only string."""
    value = metadata.get(field)
    if value is None:
        return True
    return isinstance(value, str) and not value.strip()


def subject_metadata_problems(metadata: dict, *, default_subject_id: str = "") -> list[str]:
    """Return every DANDI subject-metadata problem in one identity's metadata.

    Args:
        metadata: One identity's subject metadata, as supplied via ``--subjects``.
            An empty dict means the identity has no metadata at all.
        default_subject_id: The ``subject_id`` the writer falls back to when
            ``metadata`` supplies none - the identity's raw external ID, or its
            sanitized container name. Checked in place of an absent ``subject_id``
            so this agrees with what is actually written, and so an unsanitized
            external ID containing ``/`` is still caught.

    Returns:
        Human-readable problem descriptions, one per failed requirement, ordered
        so that missing required fields come before malformed optional ones.
        Empty when the metadata satisfies every requirement checked here.
    """
    problems: list[str] = []

    # The writer defaults subject_id rather than leaving it unset, so validate the
    # value that will actually be written.
    subject_id = metadata.get("subject_id") or default_subject_id
    if not str(subject_id).strip():
        problems.append("subject_id is missing")
    elif "/" in str(subject_id):
        problems.append(f"subject_id {subject_id!r} contains '/', which breaks DANDI paths")

    species = metadata.get("species")
    if _missing(metadata, "species"):
        problems.append("species is missing")
    elif not _SPECIES_FORM.fullmatch(str(species)):
        problems.append(
            f"species {species!r} must be Latin binomial (e.g. 'Mus musculus') or an "
            "NCBI taxonomy IRI (e.g. 'http://purl.obolibrary.org/obo/NCBITaxon_10090')"
        )

    sex = metadata.get("sex")
    if _missing(metadata, "sex"):
        problems.append("sex is missing")
    else:
        allowed = _C_ELEGANS_SEX_VALUES if species in _C_ELEGANS_SPECIES else _SEX_VALUES
        if str(sex) not in allowed:
            problems.append(f"sex {sex!r} must be one of {', '.join(repr(v) for v in allowed)}")

    age_missing = _missing(metadata, "age")
    dob_missing = _missing(metadata, "date_of_birth")
    if age_missing and dob_missing:
        problems.append("age or date_of_birth is missing")
    if not age_missing and not _is_iso_duration(str(metadata["age"])):
        problems.append(
            f"age {metadata['age']!r} must be an ISO 8601 duration (e.g. 'P70D', 'P2Y') "
            "or a range (e.g. 'P1D/P3D', 'P90Y/')"
        )
    if not dob_missing:
        dob = metadata["date_of_birth"]
        # _make_subject parses a string date_of_birth with datetime.fromisoformat;
        # catch an unparseable value here rather than mid-write.
        if isinstance(dob, str):
            try:
                datetime.datetime.fromisoformat(dob)
            except ValueError:
                problems.append(
                    f"date_of_birth {dob!r} is not an ISO 8601 datetime "
                    "(e.g. '2024-01-15T00:00:00+00:00')"
                )
        elif not isinstance(dob, datetime.datetime):
            problems.append(
                f"date_of_birth must be an ISO 8601 datetime string, got {type(dob).__name__}"
            )

    weight = metadata.get("weight")
    if weight is not None and isinstance(weight, str) and not _WEIGHT_FORM.fullmatch(weight):
        problems.append(
            f"weight {weight!r} must be '[numeric] [unit]' with a space, e.g. '25 g' or '0.025 kg'"
        )

    return problems


def validate_subjects(data: PoseData) -> None:
    """Validate every identity's subject metadata before any NWB file is written.

    Resolves metadata exactly the way the NWB writer will - so an identity whose
    ``--subjects`` key does not match is seen as having none - and reports every
    problem across every identity at once, rather than failing on the first.

    Unused ``subjects`` keys are logged as a warning rather than raising: a key
    that matches no identity leaves that identity without metadata, which surfaces
    below as a missing-field error, and the warning is what explains why.

    Args:
        data: The pose data about to be written.

    Raises:
        ValueError: If any identity's subject metadata is missing or malformed,
            listing the offending identities and their problems.
    """
    resolved = resolve_identity_subjects(data)

    supplied = set((data.subjects or {}).keys())
    if supplied:
        matched = {key for entry in resolved for key in entry.lookup_keys}
        unused = sorted(supplied - matched)
        if unused:
            logger.warning(
                "These --subjects keys match no identity and were ignored: %s. "
                "Valid identity names for this pose file are: %s",
                ", ".join(unused),
                ", ".join(entry.identity_name for entry in resolved),
            )

    failures = [
        (entry.identity_name, problems)
        for entry in resolved
        if (
            problems := subject_metadata_problems(
                entry.metadata, default_subject_id=entry.lookup_keys[0]
            )
        )
    ]
    if not failures:
        logger.debug("Subject metadata satisfies DANDI requirements for all identities")
        return

    detail = "\n".join(f"  {name}: {'; '.join(problems)}" for name, problems in failures)
    raise ValueError(
        "Subject metadata required by the DANDI archive is missing or malformed:\n"
        f"{detail}\n"
        "Every identity needs subject_id, species, sex, and age or date_of_birth. "
        f"Supply them with --subjects; see {_DOCS_REFERENCE}."
    )
