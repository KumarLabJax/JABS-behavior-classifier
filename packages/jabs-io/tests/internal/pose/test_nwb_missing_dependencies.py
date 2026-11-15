"""Tests for the errors raised when the NWB dependencies are missing or too old.

These need none of the NWB packages, so they run with or without the ``nwb`` extra.
"""

from types import SimpleNamespace

import pytest

from jabs.core.exceptions import MissingOptionalDependencyError
from jabs.io.internal.pose import PoseNWBAdapter, nwb, nwb_video, require_video_link_support


def test_adapter_reports_missing_nwb_packages(monkeypatch) -> None:
    """Without pynwb and friends, the adapter names the extra that provides them."""
    monkeypatch.setattr(nwb, "_NWB_AVAILABLE", False)

    with pytest.raises(MissingOptionalDependencyError, match=r"jabs-io\[nwb\]") as raised:
        PoseNWBAdapter()

    assert raised.value.extra == "nwb"


def test_multisubject_output_reports_missing_extension(monkeypatch) -> None:
    """Without ndx-multisubjects, multisubject output names the extra that provides it."""
    monkeypatch.setattr(nwb, "_MULTISUBJECTS_AVAILABLE", False)

    with pytest.raises(MissingOptionalDependencyError, match="ndx-multisubjects") as raised:
        PoseNWBAdapter._require_multisubjects()

    assert raised.value.extra == "nwb"


def test_video_linking_reports_missing_pynwb(monkeypatch) -> None:
    """Without pynwb, video linking names the extra that provides it."""
    monkeypatch.setattr(nwb_video, "_PYNWB_AVAILABLE", False)

    with pytest.raises(MissingOptionalDependencyError) as raised:
        require_video_link_support()

    assert raised.value.extra == "nwb"


def test_video_linking_with_old_pynwb_is_not_a_missing_dependency(monkeypatch) -> None:
    """An installed but too-old pynwb is a plain ImportError, not a missing extra.

    Installing the extra would not necessarily fix it, so callers must not be steered
    toward that advice.
    """
    monkeypatch.setattr(nwb_video, "_PYNWB_AVAILABLE", True)
    monkeypatch.setattr(nwb_video, "pynwb", SimpleNamespace(__version__="3.1.0"), raising=False)

    with pytest.raises(ImportError, match="pynwb>=4") as raised:
        require_video_link_support()

    assert not isinstance(raised.value, MissingOptionalDependencyError)
