"""Shared Click options for the CLI commands that remap labels onto new poses.

``jabs-cli update-pose`` and ``jabs-cli update-labels`` solve the same problem
from two directions: both carry label blocks from one set of pose files onto
another, and both drive that transfer with a
:class:`~.label_remap_matching.MatchSettings`. The flags that fill those
settings were declared separately in each command, which meant the thresholds,
the metric choice and their defaults were written out three times - once per
command and once more as the dataclass defaults. Nothing kept the three copies
in agreement, so a threshold changed in one place would leave the same flag
meaning different things in the two commands.

Declaring them once here removes that risk: both commands apply
:func:`identity_match_options` and build their settings with
:func:`match_settings_from_options`, and the defaults are read from
``MatchSettings`` so the dataclass stays the single place a default is written.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import click

from .label_remap_matching import MatchMetric, MatchSettings

#: A Click decorator, i.e. what ``click.option(...)`` returns.
_Decorator = Callable[[Any], Any]

F = TypeVar("F", bound=Callable[..., Any])

# The defaults the options advertise. Reading them from the dataclass keeps the
# values the CLI shows and the values used when no CLI is involved identical.
_DEFAULTS = MatchSettings()

# The shared options, in the order they appear in ``--help``.
_MATCH_OPTIONS: tuple[_Decorator, ...] = (
    click.option(
        "--min-iou-thresh",
        "min_iou",
        type=float,
        default=_DEFAULTS.min_bbox_iou,
        show_default=True,
        help="Minimum acceptable median bounding box IoU for a match decided by bounding boxes.",
    ),
    click.option(
        "--min-mask-iou",
        "min_mask_iou",
        type=float,
        default=_DEFAULTS.min_mask_iou,
        show_default=True,
        help=(
            "Minimum acceptable median segmentation mask IoU for a match decided by segmentation."
        ),
    ),
    click.option(
        "--match-by",
        "match_by",
        type=click.Choice([m.value for m in MatchMetric]),
        default=_DEFAULTS.match_by.value,
        show_default=True,
        help=(
            "Which score decides each label block's identity: 'auto' uses segmentation mask IoU "
            "when both identities have masks in the block and bounding box IoU otherwise; "
            "'segmentation' and 'bbox' use only that score."
        ),
    ),
    click.option(
        "--split-blocks/--no-split-blocks",
        "split_blocks",
        default=_DEFAULTS.split_blocks,
        show_default=True,
        help=(
            "Match each label block frame by frame and split it where the best-matching "
            "identity changes, e.g. when one animal is handed from one identity to another "
            "mid-block. --no-split-blocks assigns each whole block to a single identity."
        ),
    ),
    click.option(
        "--min-run-frames",
        "min_run_frames",
        type=click.IntRange(min=1),
        default=_DEFAULTS.min_run_frames,
        show_default=True,
        help="When splitting, fold runs shorter than this many frames into a neighboring run.",
    ),
)


def identity_match_options(command: F) -> F:
    """Add the shared identity-matching options to a Click command.

    The options are applied innermost-first, which is the order stacked
    ``@click.option`` decorators are applied in, so a command that uses this in
    place of the stack keeps the same parameter order in ``--help``.

    Args:
        command: The command function being decorated.

    Returns:
        ``command``, with the shared options attached.
    """
    for option in reversed(_MATCH_OPTIONS):
        command = option(command)
    return command


def match_settings_from_options(
    *,
    min_iou: float,
    min_mask_iou: float,
    match_by: str,
    split_blocks: bool,
    min_run_frames: int,
) -> MatchSettings:
    """Build the matcher settings from the values :func:`identity_match_options` parsed.

    Args:
        min_iou: Value of ``--min-iou-thresh``.
        min_mask_iou: Value of ``--min-mask-iou``.
        match_by: Value of ``--match-by``, as the string Click parsed.
        split_blocks: Value of ``--split-blocks/--no-split-blocks``.
        min_run_frames: Value of ``--min-run-frames``.

    Returns:
        The settings to hand to the identity matcher.
    """
    return MatchSettings(
        min_bbox_iou=min_iou,
        min_mask_iou=min_mask_iou,
        match_by=MatchMetric(match_by),
        split_blocks=split_blocks,
        min_run_frames=min_run_frames,
    )
