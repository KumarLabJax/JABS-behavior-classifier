"""Logging configuration for jabs-vision CLIs."""

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging for CLI runs if not already configured.

    Args:
        level: Logging level to apply.
    """
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
