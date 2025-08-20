"""jabs version"""

import importlib.metadata
from pathlib import Path

import toml


def version_str() -> str:
    """Return version string from package metadata or pyproject.toml.

    If jabs-behavior-classifier is an installed package, gets the version from the package metadata. If not installed,
    attempts to read the project's pyproject.toml file to get the version. Returns 'dev' if it's not able to determine
    the version using either rof these methods.
    """
    try:
        return importlib.metadata.version("jabs-behavior-classifier")
    except importlib.metadata.PackageNotFoundError:
        pyproject_file = Path(__file__).parent.parent.parent.parent / "pyproject.toml"
        try:
            data = toml.load(pyproject_file)
            return data["tool"]["poetry"]["version"]
        except (FileNotFoundError, KeyError, toml.TomlDecodeError):
            return "dev"
