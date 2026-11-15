"""Install advice for CLI commands that need an optional JABS extra."""

import click

from jabs.core.exceptions import MissingOptionalDependencyError


def missing_extra_error(error: MissingOptionalDependencyError) -> click.ClickException:
    """Build the CLI error for a command whose optional extra is not installed.

    The library that raised ``error`` suggests installing its own extra, which is
    right for that library's users but not for JABS: an isolated uv or pipx install
    of JABS needs the extra added to JABS itself, or it is lost on the next upgrade.

    Args:
        error: The missing-dependency error raised by the library.

    Returns:
        A ``ClickException`` telling the user which JABS extra to install and where
        to find the command for their install method.
    """
    extra = error.extra
    return click.ClickException(
        f"This command requires the '{extra}' extra, which is not installed.\n"
        f'Install JABS with it, for example: pip install "jabs-behavior-classifier[{extra}]"\n'
        "If you installed JABS with uv or pipx, see the Optional Extras page of the JABS "
        "user guide for the matching command."
    )
