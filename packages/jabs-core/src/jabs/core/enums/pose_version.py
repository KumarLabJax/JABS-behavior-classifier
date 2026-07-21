"""Enum for legacy JABS pose-file format versions."""

from enum import IntEnum


class JabsPoseVersion(IntEnum):
    """Legacy JABS pose-file format major versions (the ``pose_est_vN`` convention).

    The ``pose_est_vN`` convention is planned for deprecation in favor of a future
    backwards-compatible pose format. Until then this enum names the legacy layouts;
    it is used as the ``legacy=`` selector when writing pose files. Members are aligned
    to the historical integer majors (2-8) so numeric comparisons keep working.
    """

    V2 = 2
    V3 = 3
