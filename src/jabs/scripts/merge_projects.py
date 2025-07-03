import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich_argparse import RichHelpFormatter

from jabs.project import Project
from jabs.project.project_merge import MergeStrategy, merge_projects

MERGE_STRATEGY_MAP: dict[str, tuple[MergeStrategy, str]] = {
    "behavior-wins": (MergeStrategy.BEHAVIOR_WINS, "Keep the label with the behavior annotation."),
    "not-behavior-wins": (
        MergeStrategy.NOT_BEHAVIOR_WINS,
        "Keep the label without the behavior annotation.",
    ),
    "destination-wins": (
        MergeStrategy.DESTINATION_WINS,
        "Keep the label from the destination project.",
    ),
}

max_key_len = max(len(key) for key in MERGE_STRATEGY_MAP)
MERGE_STRATEGY_DETAILS = "About Merge Strategy Options:\n" + "\n".join(
    f"  * {(key + ':').ljust(max_key_len + 1)}  {desc}"
    for key, (_, desc) in MERGE_STRATEGY_MAP.items()
)


def rich_error_exit(message: str) -> None:
    """Print a formatted error message and exit the program."""
    console = Console()
    console.print(f"\n  [bold red]ERROR:[/bold red] {message}\n")
    sys.exit(1)


def main():
    """project merge script entry point."""

    def dir_path(path_str: str) -> Path:
        path = Path(path_str)
        if not path.is_dir():
            raise argparse.ArgumentTypeError(f"{path} is not a valid directory")
        return path

    """Command for merging two JABS projects."""
    parser = argparse.ArgumentParser(
        description="Merge two JABS projects.",
        epilog=Markdown(MERGE_STRATEGY_DETAILS, style="argparse.text"),  # type: ignore
        formatter_class=RichHelpFormatter,
    )
    parser.add_argument(
        "destination_project",
        type=dir_path,
        help="Path to the destination project. This project will be modified by importing videos and labels from the"
        " source project.",
    )
    parser.add_argument("source_project", type=dir_path, help="Path to the source project.")
    parser.add_argument(
        "--merge-strategy",
        choices=MERGE_STRATEGY_MAP.keys(),
        required=True,
        default="destination-wins",
        help="Strategy to use when merging conflicting labels. (default: destination-wins)",
    )
    args = parser.parse_args()

    if args.destination_project == args.source_project:
        rich_error_exit("Destination and source projects cannot be the same.")

    if not (args.destination_project / "jabs").exists():
        rich_error_exit(
            f"Destination project {args.destination_project} is not a valid JABS project."
        )

    if not (args.source_project / "jabs").exists():
        rich_error_exit(f"Source project {args.source_project} is not a valid JABS project.")

    destination = Project(args.destination_project)
    source = Project(args.source_project)

    merge_projects(destination, source, MERGE_STRATEGY_MAP[args.merge_strategy][0])


if __name__ == "__main__":
    main()
