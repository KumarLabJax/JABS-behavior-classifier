"""jabs version"""

import importlib.metadata
from pathlib import Path

import toml


def version_str():
    """return version string using the version specified in the pyproject.toml file"""
    try:
        # if the jabs package is installed, return version from package metadata
        return importlib.metadata.version("jabs-behavior-classifier")
    except:  # noqa: E722
        # not installed as a package, assume we're running out of a cloned git repo without running `poetry install`
        # try to get the version from the pyproject.toml file

        # placeholder string in case we can't get version from the pyproject.toml file for some reason
        version = "dev"

        pyproject_file = Path(__file__).parent.parent.parent.parent / "pyproject.toml"
        try:
            data = toml.load(pyproject_file)
            version = data["tool"]["poetry"]["version"]
        except:  # noqa: E722
            pass

        return version
