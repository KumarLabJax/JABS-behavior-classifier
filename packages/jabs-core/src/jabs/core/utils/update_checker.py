"""Utilities for checking PyPI for JABS updates."""

import json
import logging
import urllib.request
from importlib import metadata

from packaging.version import parse as parse_version

# TODO: Consider moving this to jabs.core
from jabs.version import version_str

logger = logging.getLogger(__name__)


def check_for_update() -> tuple[bool, str | None, str]:
    """Check PyPI for newer version of jabs-behavior-classifier.

    Returns:
        tuple: (has_update: bool, latest_version: str | None, current_version: str)
            - has_update: True if a newer version is available
            - latest_version: Latest version string from PyPI, or None if check failed
            - current_version: Current installed version string
    """
    try:
        current_version = version_str()

        with urllib.request.urlopen(
            "https://pypi.org/pypi/jabs-behavior-classifier/json", timeout=5
        ) as response:
            data = json.loads(response.read())
            latest_version = data["info"]["version"]

        has_update = parse_version(latest_version) > parse_version(current_version)
        return has_update, latest_version, current_version
    except Exception as e:
        logger.warning(f"Failed to check for updates: {e}")
        return False, None, version_str()


def is_pypi_install() -> bool:
    """Check if jabs-behavior-classifier was installed from PyPI.

    Returns:
        bool: True if installed via pip from PyPI, False otherwise
    """
    try:
        dist = metadata.distribution("jabs-behavior-classifier")
        # Check if installer was pip
        installer = dist.read_text("INSTALLER")
        return installer is not None and installer.strip() in ("pip", "uv")
    except Exception as e:
        logger.debug(f"Could not determine installation method: {e}")
        return False
