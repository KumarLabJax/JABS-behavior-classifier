"""jabs version"""

from jabs.core.utils.version import version_str as new_version_str


def version_str() -> str:
    """Return version string from package metadata or pyproject.toml.

    If jabs-behavior-classifier is an installed package, gets the version from the package metadata. If not installed,
    attempts to read the project's pyproject.toml file to get the version. Returns 'dev' if it's not able to determine
    the version using either rof these methods.
    """
    return new_version_str("jabs-behavior-classifier")
