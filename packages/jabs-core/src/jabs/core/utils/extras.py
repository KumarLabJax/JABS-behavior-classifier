"""Report which optional extras of an installed distribution are installed."""

import importlib.metadata
import logging
from dataclasses import dataclass

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtraStatus:
    """Install status of one optional extra.

    Attributes:
        name: Name of the extra, as declared by the distribution.
        required: Names of the packages the extra installs on this Python version and
            platform, including those pulled in through nested extras. Packages whose
            environment markers exclude this platform are left out.
        missing: The subset of ``required`` that is not installed, or is installed at a
            version the extra does not accept.
    """

    name: str
    required: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def installed(self) -> bool:
        """Whether every package the extra needs here is installed."""
        return not self.missing

    @property
    def partially_installed(self) -> bool:
        """Whether some, but not all, of the extra's packages are installed."""
        return bool(self.missing) and len(self.missing) < len(self.required)


def extras_status(distribution: str) -> list[ExtraStatus]:
    """Return the install status of every optional extra a distribution declares.

    Reads the distribution's own metadata, so the result follows whatever extras the
    installed version declares, including environment markers (an extra's package that
    does not apply on this Python version is not required) and nested extras (an extra
    that depends on another package's extra).

    Args:
        distribution: Name of the installed distribution, e.g.
            ``"jabs-behavior-classifier"``.

    Returns:
        One :class:`ExtraStatus` per declared extra, sorted by name. Empty when the
        distribution is not installed or declares no extras.
    """
    try:
        metadata = importlib.metadata.metadata(distribution)
    except importlib.metadata.PackageNotFoundError:
        logger.debug("Distribution %s is not installed; no extras to report", distribution)
        return []

    statuses = []
    for extra in sorted(set(metadata.get_all("Provides-Extra") or [])):
        required = _extra_requirements(distribution, extra, seen=set())
        missing = tuple(name for name, satisfied in required.items() if not satisfied)
        statuses.append(ExtraStatus(name=extra, required=tuple(required), missing=missing))
    return statuses


def _extra_requirements(
    distribution: str, extra: str, seen: set[tuple[str, str]]
) -> dict[str, bool]:
    """Map each package an extra needs on this platform to whether it is satisfied.

    Args:
        distribution: Distribution declaring the extra.
        extra: Name of the extra.
        seen: ``(distribution, extra)`` pairs already visited, so that extras which
            refer to each other do not recurse forever.

    Returns:
        Package name to ``True`` if installed at an accepted version, in the order the
        requirements are declared, followed by the packages of any nested extras.
    """
    key = (canonicalize_name(distribution), extra)
    if key in seen:
        return {}
    seen.add(key)

    try:
        declared = importlib.metadata.requires(distribution) or []
    except importlib.metadata.PackageNotFoundError:
        return {}

    result: dict[str, bool] = {}
    for line in declared:
        requirement = _parse(line)
        if requirement is None or not _enabled_by_extra(requirement, extra):
            continue
        satisfied = _is_satisfied(requirement)
        # a requirement such as ``jabs-io[nwb]`` stands for that package's extra, so
        # report the extra's packages rather than the (possibly core) package itself
        if satisfied and requirement.extras:
            for nested in sorted(requirement.extras):
                result.update(_extra_requirements(requirement.name, nested, seen))
        else:
            result[requirement.name] = satisfied
    return result


def _parse(line: str) -> Requirement | None:
    """Parse one ``Requires-Dist`` entry, or return None if it is malformed."""
    try:
        return Requirement(line)
    except InvalidRequirement:
        logger.debug("Skipping unparsable requirement %r", line)
        return None


def _enabled_by_extra(requirement: Requirement, extra: str) -> bool:
    """Whether a requirement applies here because ``extra`` was requested.

    A requirement belongs to the extra when its marker holds with the extra requested
    but not without it. That leaves out the distribution's core requirements, and the
    extra's packages whose markers exclude this Python version or platform.
    """
    if requirement.marker is None:
        return False
    return requirement.marker.evaluate({"extra": extra}) and not requirement.marker.evaluate(
        {"extra": ""}
    )


def _is_satisfied(requirement: Requirement) -> bool:
    """Whether a requirement's package is installed at a version it accepts."""
    try:
        installed = importlib.metadata.version(requirement.name)
    except importlib.metadata.PackageNotFoundError:
        return False
    return requirement.specifier.contains(installed, prereleases=True)
