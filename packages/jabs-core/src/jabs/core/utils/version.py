"""Get version string from package metadata or fall back to pyproject.toml"""

import importlib
import importlib.metadata
import warnings
from pathlib import Path


def version_str(pacakge: str = "jabs-behavior-classifier") -> str:
    """Return version string from package metadata or pyproject.toml.

    If jabs-behavior-classifier is an installed package, gets the version from the package metadata. If not installed,
    attempts to read the project's pyproject.toml file to get the version. Returns 'dev' if it's not able to determine
    the version using either rof these methods.
    """
    try:
        return importlib.metadata.version(pacakge)
    except importlib.metadata.PackageNotFoundError:
        return _toml_version(pacakge)


def _toml_version(pacakge: str):
    warnings.warn(
        f"Could not determine version for '{pacakge}' from package metadata. \n"
        "Falling back to 'jabs-behavior-classifier' pyproject.toml, which may not be "
        "the package whose version you requested. \n"
        "This fallback is DEPRECATED and will be removed in a future release. \n"
        "Install the package (e.g. `uv tool install`, `pipx install`, "
        "`pip install -e .`) for reliable version detection. ",
        category=DeprecationWarning,
        stacklevel=2,
    )

    try:
        import toml

        pyproject_file = (
            Path(__file__).parent.parent.parent.parent.parent.parent.parent / "pyproject.toml"
        )

        data = toml.load(pyproject_file)
        return data["project"]["version"]
    except (ImportError, FileNotFoundError, KeyError, toml.TomlDecodeError):
        return "dev"
