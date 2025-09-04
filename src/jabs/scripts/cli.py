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
from jabs.utils import FINAL_TRAIN_SEED

# find out which classifiers are supported in this environment
CLASSIFIER_CHOICES: list[ClassifierType] = Classifier().classifier_choices()


@click.group()
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
        exists=True,  # require the directory to exist
        file_okay=False,  # must be a directory
        dir_okay=True,
        path_type=Path,
    ),
)
@click.option(
    "--behavior",
    required=True,
    type=str,
    help="Specify the behavior to use during export (required).",
)
@click.option(
    "--classifier-type",
    "classifier_type",
    required=True,
    type=click.Choice([c.name for c in CLASSIFIER_CHOICES], case_sensitive=False),
    help="Classifier to use. Choices: one of CLASSIFIER_CHOICES.",
)
@click.option(
    "--outfile",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    required=False,
    help="Optional path to write the exported training data file.",
)
@click.pass_context
def export_training(
    ctx, directory: Path, behavior: str, classifier_type: str, outfile: Path | None
):
    """Export training data to the given DIRECTORY with a specified BEHAVIOR."""
    if ctx.obj["VERBOSE"]:
        click.echo("Exporting training data with the following parameters:")
        click.echo(f"\tBehavior: {behavior}")
        click.echo(f"\tClassifier type: {classifier_type}")
        click.echo(f"\tJABS project directory: {directory}")

    classifier_type = ClassifierType[classifier_type.upper()]
    jabs_project = Project(directory, enable_session_tracker=False)

    # validate that the behavior exists in the project
    if behavior not in jabs_project.settings["behavior"]:
        raise click.ClickException(f"Behavior '{behavior}' not found in project.")

    console = Console()
    status_text = (
        f"Exporting training data (behavior={behavior}, classifier={classifier_type.name})"
    )
    with console.status(status_text, spinner="dots"):
        outfile = export_training_data(
            jabs_project,
            behavior,
            jabs_project.feature_manager.min_pose_version,
            classifier_type,
            FINAL_TRAIN_SEED,
            out_file=outfile,
        )

    click.echo(f"Exported training data to {outfile}")


def main():
    """Entry point for the JABS CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
