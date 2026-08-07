"""Merge one JABS project into another.

The source project's videos, pose files, behaviors, and labels are imported into
the destination project, which is modified in place. The source project is left
unchanged.

Where both projects label the same video, identity, and behavior, conflicting
labels are resolved by the selected ``--merge-strategy``:

\b
  * behavior-wins:      Keep the label with the behavior annotation.
  * not-behavior-wins:  Keep the label without the behavior annotation.
  * destination-wins:   Keep the label from the destination project.

Example:

\b
  jabs-cli merge /path/to/destination_project /path/to/source_project --merge-strategy destination-wins
"""

from __future__ import annotations

from pathlib import Path

import click

from jabs.project import Project
from jabs.project.project_merge import MergeStrategy, merge_projects

MERGE_STRATEGY_MAP: dict[str, MergeStrategy] = {
    "behavior-wins": MergeStrategy.BEHAVIOR_WINS,
    "not-behavior-wins": MergeStrategy.NOT_BEHAVIOR_WINS,
    "destination-wins": MergeStrategy.DESTINATION_WINS,
}


@click.command(
    name="merge",
    context_settings={"max_content_width": 120},
    help=__doc__,
)
@click.argument(
    "destination_project",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.argument(
    "source_project",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "--merge-strategy",
    "merge_strategy",
    type=click.Choice(list(MERGE_STRATEGY_MAP), case_sensitive=False),
    required=True,
    help="Strategy to use when merging conflicting labels.",
)
def merge_projects_command(
    destination_project: Path,
    source_project: Path,
    merge_strategy: str,
) -> None:
    """Merge a source JABS project into a destination JABS project."""
    if destination_project.resolve() == source_project.resolve():
        raise click.ClickException("Destination and source projects cannot be the same.")

    if not Project.is_valid_project_directory(destination_project):
        raise click.ClickException(
            f"Destination project {destination_project} is not a valid JABS project."
        )

    if not Project.is_valid_project_directory(source_project):
        raise click.ClickException(f"Source project {source_project} is not a valid JABS project.")

    destination = Project(destination_project)
    source = Project(source_project)

    merge_projects(destination, source, MERGE_STRATEGY_MAP[merge_strategy.lower()])
