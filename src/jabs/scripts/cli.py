"""The JABS CLI

This module provides a command-line interface to interact with a JABS project.
Initially, it was developed as a collect-all of smaller utilities that didn't
warrant their own scripts, but it might eventually evolve into a more
comprehensive tool for managing JABS projects (possibly integrating
functionality from existing JABS scripts such as `jabs-init` or `jabs-merge`).
"""

from pathlib import Path

import click
from rich.console import Console

from jabs.classifier import Classifier
from jabs.project import Project, export_training_data
from jabs.types import ClassifierType

# find out which classifiers are supported in this environment
CLASSIFIER_CHOICES: list[ClassifierType] = Classifier().classifier_choices()


@click.group(context_settings={"max_content_width": 120})
@click.option("--verbose", is_flag=True, help="Enable verbose output.")
@click.pass_context
def cli(ctx, verbose):
    """JABS CLI."""
    ctx.ensure_object(dict)
    ctx.obj["VERBOSE"] = verbose


@cli.command(name="export-training")
@click.argument(
    "directory",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
)
@click.option(
    "--behavior",
    required=True,
    type=str,
    help="Specify the behavior to export (required).",
)
@click.option(
    "--classifier",
    "classifier",
    default="xgboost",
    type=click.Choice([c.name for c in CLASSIFIER_CHOICES], case_sensitive=False),
    help="Default classifier set in the training file. Default is 'xgboost'.",
)
@click.option(
    "--outfile",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    required=False,
    help=(
        "Optional path to write the exported training data file. "
        "Default is <project_dir>/<behavior>_training_<YYYYMMDD_HHMMSS>.h5"
    ),
)
@click.pass_context
def export_training(ctx, directory: Path, behavior: str, classifier: str, outfile: Path | None):
    """Export training data for a specified behavior and JABS project directory."""
    #    ctx: Click context.
    #    directory (Path): Path to the JABS project directory.
    #    behavior (str): Behavior to export.
    #    classifier (str): Default classifier type set in the training file,
    #        can be overridden by the jabs-classify train command.
    #    outfile (Path | None): Optional path to write the exported training data
    #        file. If not provided, export_training_data will generate a unique
    #        filename in the project directory using .

    if ctx.obj["VERBOSE"]:
        click.echo("Exporting training data with the following parameters:")
        click.echo(f"\tBehavior: {behavior}")
        click.echo(f"\tClassifier type: {classifier}")
        click.echo(f"\tJABS project directory: {directory}")

    classifier = ClassifierType[classifier.upper()]
    jabs_project = Project(directory, enable_session_tracker=False)

    # validate that the behavior exists in the project
    if behavior not in jabs_project.settings["behavior"]:
        raise click.ClickException(f"Behavior '{behavior}' not found in project.")

    console = Console()
    status_text = f"Exporting training data (behavior={behavior}, classifier={classifier.name})"
    with console.status(status_text, spinner="dots"):
        outfile = export_training_data(
            jabs_project,
            behavior,
            jabs_project.feature_manager.min_pose_version,
            classifier,
            out_file=outfile,
        )

    click.echo(f"Exported training data to {outfile}")


@cli.command(name="rename-behavior")
@click.argument(
    "directory",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
)
@click.argument(
    "old-name",
    type=str,
)
@click.argument(
    "new-name",
    type=str,
)
@click.pass_context
def rename_behavior(ctx, directory: Path, old_name: str, new_name: str) -> None:
    """Rename a behavior in a JABS project."""
    # Args:
    #    ctx: Click context.
    #    directory (Path): Path to the JABS project directory.
    #    old_name (str): Current name of the behavior to rename.
    #    new_name (str): New name for the behavior.
    #
    # Raises:
    #    click.ClickException: If the old behavior name does not exist or
    #        the new behavior name already exists.

    if ctx.obj["VERBOSE"]:
        click.echo("Renaming behavior with the following parameters:")
        click.echo(f"\tOld behavior name: {old_name}")
        click.echo(f"\tNew behavior name: {new_name}")
        click.echo(f"\tJABS project directory: {directory}")

    if not Project.is_valid_project_directory(directory):
        raise click.ClickException(f"Invalid JABS project directory: {directory}")

    jabs_project = Project(directory, enable_session_tracker=False)

    # validate that the old behavior exists in the project
    if old_name not in jabs_project.settings["behavior"]:
        raise click.ClickException(f"Behavior '{old_name}' not found in project.")

    # validate that the new behavior does not already exist in the project
    if new_name in jabs_project.settings["behavior"]:
        raise click.ClickException(f"Behavior '{new_name}' already exists in project.")

    console = Console()
    status_text = f"Renaming behavior '{old_name}' to '{new_name}'"
    with console.status(status_text, spinner="dots"):
        jabs_project.rename_behavior(old_name, new_name)


def main():
    """Entry point for the JABS CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
