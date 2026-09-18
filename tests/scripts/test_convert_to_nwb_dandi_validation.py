"""End-to-end check that converted NWB files clear the DANDI archive's gate.

This is the guarantee KLAUS-706 exists to provide, asserted against the real
validator rather than against our own reimplementation of its rules: the
pre-flight check in :mod:`jabs.scripts.cli.dandi_subject_metadata` mirrors
``nwbinspector``, and a mirror can drift. If nwbinspector adds or re-weights a
subject check, this test fails and the pre-flight rules get updated to match.

Requires the ``nwb`` extra (pynwb, ndx-pose) and the ``test`` group's
``nwbinspector``; skipped when either is unavailable.
"""

from pathlib import Path

import pytest

pytest.importorskip("pynwb")
pytest.importorskip("ndx_pose")
pytest.importorskip("nwbinspector")

from nwbinspector import inspect_nwbfile, load_config

from jabs.scripts.cli.convert_to_nwb import run_conversion

POSE_FILE = Path(__file__).parent.parent / "data" / "sample_pose_est_v6.h5"

# The sample pose file has no external IDs, so identities fall back to subject_N.
SUBJECTS = {
    f"subject_{i}": {
        "subject_id": f"M12{i}",
        "species": "Mus musculus",
        "sex": "M" if i % 2 else "F",
        "age": "P70D",
        "strain": "C57BL/6J",
        "weight": "25 g",
    }
    for i in range(1, 5)
}


def _critical_findings(path: Path) -> list[str]:
    """Return the names of every CRITICAL check the DANDI config reports for a file."""
    config = load_config(filepath_or_keyword="dandi")
    return [
        message.check_function_name
        for message in inspect_nwbfile(nwbfile_path=str(path), config=config)
        if "CRITICAL" in str(message.importance)
    ]


@pytest.fixture(scope="module")
def converted(tmp_path_factory) -> list[Path]:
    """Convert the sample pose file once and return the per-identity outputs."""
    out_dir = tmp_path_factory.mktemp("nwb")
    run_conversion(POSE_FILE, out_dir / "session.nwb", subjects=SUBJECTS)
    return sorted(out_dir.glob("*.nwb"))


def test_conversion_writes_one_file_per_identity(converted: list[Path]) -> None:
    """The sample pose file has four identities, so four files are written."""
    assert [path.name for path in converted] == [
        "session_subject_1.nwb",
        "session_subject_2.nwb",
        "session_subject_3.nwb",
        "session_subject_4.nwb",
    ]


def test_output_has_no_critical_nwbinspector_findings(converted: list[Path]) -> None:
    """A CRITICAL finding is what blocks upload, so there must be none."""
    findings = {path.name: _critical_findings(path) for path in converted}

    assert findings == {name: [] for name in findings}


def test_missing_subject_metadata_is_caught_before_writing(tmp_path) -> None:
    """The pre-flight check must fire rather than leaving the archive to reject it."""
    with pytest.raises(ValueError, match="missing or malformed"):
        run_conversion(POSE_FILE, tmp_path / "session.nwb", subjects=None)

    assert list(tmp_path.glob("*.nwb")) == []
