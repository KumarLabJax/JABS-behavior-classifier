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
from jabs.io.internal.pose import resolve_identity_subjects, subject_value_is_absent

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


# Nominal day-count bounds per ISO 8601 duration unit. Years and months are ranges
# because their real length depends on the calendar date they are measured from.
_UNIT_DAY_BOUNDS: dict[str, tuple[float, float]] = {
    "Y": (365.0, 366.0),
    "M": (28.0, 31.0),
    "W": (7.0, 7.0),
    "D": (1.0, 1.0),
    "TH": (1 / 24, 1 / 24),
    "TM": (1 / 1440, 1 / 1440),
    "TS": (1 / 86400, 1 / 86400),
}


def _is_iso_duration(value: str) -> bool:
    """Return whether a string is an ISO 8601 duration, or a range of two.

    Mirrors ``check_subject_age``, which accepts a plain duration (``"P70D"``) as
    well as a ``/``-separated range with either bound optionally blank - so
    ``"P1D/P3D"``, ``"P90Y/"`` and ``"/P3D"`` are all valid.
    """
    if _DURATION.fullmatch(value):
        return True
    if "/" not in value:
        return False
    lower, _, upper = value.partition("/")
    return all(bound == "" or bool(_DURATION.fullmatch(bound)) for bound in (lower, upper))


def _duration_day_bounds(value: str) -> tuple[float, float] | None:
    """Return the (minimum, maximum) possible length of a duration, in days.

    A duration containing years or months has no single length - ``P1M`` is 28 to
    31 days depending on when it starts - so this returns an interval rather than a
    point. Returns None when the string is not a plain ISO 8601 duration.
    """
    match = _DURATION.fullmatch(value)
    if not match:
        return None
    groups = match.groups()
    # Group order follows _DURATION: Y, M, W, D, (T...), TH, TM, TS. Index 4 is the
    # whole time section, which carries no magnitude of its own.
    units = ("Y", "M", "W", "D", None, "TH", "TM", "TS")
    low = high = 0.0
    for unit, group in zip(units, groups, strict=True):
        if unit is None or group is None:
            continue
        amount = float(group[:-1])
        unit_low, unit_high = _UNIT_DAY_BOUNDS[unit]
        low += amount * unit_low
        high += amount * unit_high
    return low, high


def _age_range_is_reversed(value: str) -> bool:
    """Return whether an age range's bounds are provably not strictly increasing.

    Mirrors ``check_subject_proper_age_range``, which flags ``lower >= upper``, but
    stays conservative where that check uses exact calendar arithmetic: this only
    reports a range whose bounds cannot overlap under any calendar, so a genuinely
    ambiguous pair such as ``"P1M/P30D"`` is left for the archive's validator
    rather than rejected here.
    """
    if "/" not in value:
        return False
    lower_text, _, upper_text = value.partition("/")
    lower = _duration_day_bounds(lower_text)
    upper = _duration_day_bounds(upper_text)
    if lower is None or upper is None:
        return False
    # Shortest the lower bound can be vs. longest the upper bound can be.
    return lower[0] >= upper[1]


def _missing(metadata: dict, field: str) -> bool:
    """Return whether a field is absent, None, or an empty/whitespace-only string.

    Delegates to :func:`~jabs.io.internal.pose.subject_value_is_absent` so this
    agrees with what the writer omits. Keeping one definition matters: when the two
    drift, a blank value passes validation and then either crashes the write or
    reaches the archive as the CRITICAL finding this check exists to prevent.
    """
    return subject_value_is_absent(metadata.get(field))


def subject_metadata_problems(metadata: dict, *, default_subject_id: str = "") -> list[str]:
    """Return every DANDI subject-metadata problem in one identity's metadata.

    Args:
        metadata: One identity's subject metadata, as supplied via ``--subjects``.
            An empty dict means the identity has no metadata at all. A non-dict
            value is reported as a problem rather than raising.
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
    if not isinstance(metadata, dict):
        # The CLI only checks that the subjects JSON is an object at the top level, so
        # a non-object value for one identity reaches here. Report it rather than
        # letting .get() raise an AttributeError the user cannot act on.
        return [f"metadata must be a JSON object, got {type(metadata).__name__}"]

    problems: list[str] = []

    # The writer falls back to the identity name whenever the supplied id is absent
    # - including a blank string - so apply exactly the same rule here. Reporting a
    # blank id instead would reject input that in fact produces a valid Subject.
    supplied_id = metadata.get("subject_id")
    subject_id = default_subject_id if subject_value_is_absent(supplied_id) else supplied_id
    if subject_value_is_absent(subject_id):
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
    if not age_missing:
        age = str(metadata["age"])
        if not _is_iso_duration(age):
            problems.append(
                f"age {age!r} must be an ISO 8601 duration (e.g. 'P70D', 'P2Y') "
                "or a range (e.g. 'P1D/P3D', 'P90Y/')"
            )
        elif _age_range_is_reversed(age):
            problems.append(
                f"age range {age!r} must be strictly increasing - the upper (right) "
                "bound has to be a longer duration than the lower (left) bound"
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
    if not _missing(metadata, "weight"):
        if isinstance(weight, str):
            if not _WEIGHT_FORM.fullmatch(weight):
                problems.append(
                    f"weight {weight!r} must be '[numeric] [unit]' with a space, "
                    "e.g. '25 g' or '0.025 kg'"
                )
        elif not isinstance(weight, float) or isinstance(weight, bool):
            # pynwb's Subject accepts only str or float; an int from JSON raises
            # TypeError during the write rather than failing validation here.
            problems.append(
                f"weight must be a string with units (e.g. '25 g') or a float in "
                f"kilograms, got {type(weight).__name__}"
            )

    return problems


def validate_subjects(data: PoseData) -> None:
    """Validate every identity's subject metadata before any NWB file is written.

    Resolves metadata exactly the way the NWB writer will - so an identity whose
    ``--subjects`` key does not match is seen as having none - and reports every
    problem across every identity at once, rather than failing on the first.

    ``subjects`` keys that no identity reads are logged as a warning rather than
    raising - both keys matching no identity at all, and keys shadowed by a
    higher-precedence one. An unmatched key leaves its identity without metadata,
    which surfaces below as a missing-field error, and the warning explains why.

    Args:
        data: The pose data about to be written.

    Raises:
        ValueError: If any identity's subject metadata is missing or malformed,
            listing the offending identities and their problems.
    """
    resolved = resolve_identity_subjects(data)

    supplied = set((data.subjects or {}).keys())
    if supplied:
        # matched_key, not lookup_keys: when an identity offers both a raw and a
        # sanitized key and the subjects dict has both, only the raw one is read and
        # the other is silently discarded. That shadowed key needs the warning too.
        used = {entry.matched_key for entry in resolved if entry.matched_key is not None}
        unused = sorted(supplied - used)
        if unused:
            logger.warning(
                "These --subjects keys were ignored because no identity reads them: %s. "
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
