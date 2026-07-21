"""Unit tests for the JabsPoseVersion enum."""

from jabs.core.enums import JabsPoseVersion


def test_jabs_pose_version_is_int_aligned():
    """JabsPoseVersion members are aligned to the legacy integer majors."""
    assert JabsPoseVersion.V2 == 2
    assert JabsPoseVersion.V3 == 3
    assert int(JabsPoseVersion.V2) == 2


def test_jabs_pose_version_ordered():
    """IntEnum members compare numerically, matching the legacy integer majors."""
    assert JabsPoseVersion.V2 < JabsPoseVersion.V3
