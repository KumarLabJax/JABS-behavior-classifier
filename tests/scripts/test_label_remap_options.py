"""Tests for the Click options shared by ``update-pose`` and ``update-labels``.

The point of :mod:`jabs.scripts.cli.label_remap_options` is that the two
commands cannot drift apart, so these tests assert exactly that: the shared
flags expose the same spelling, type and default in both commands, and those
defaults are the ones :class:`MatchSettings` carries.
"""

import click
import pytest

from jabs.scripts.cli.label_remap_matching import MatchMetric, MatchSettings
from jabs.scripts.cli.label_remap_options import (
    identity_match_options,
    match_settings_from_options,
)
from jabs.scripts.cli.update_labels import update_labels_command
from jabs.scripts.cli.update_pose import update_pose_command

SHARED_PARAM_NAMES = (
    "min_iou",
    "min_mask_iou",
    "match_by",
    "split_blocks",
    "min_run_frames",
)


def _params_by_name(command: click.Command) -> dict[str, click.Parameter]:
    """Map a command's parameters by their Python parameter name."""
    return {param.name: param for param in command.params}


def _spec(param: click.Parameter) -> tuple:
    """Summarize the parts of an option that have to agree between commands."""
    return (
        tuple(param.opts),
        tuple(param.secondary_opts),
        param.default,
        repr(param.type),
        getattr(param, "help", None),
        getattr(param, "show_default", None),
    )


def test_decorator_adds_the_shared_options_in_order() -> None:
    """The decorator attaches exactly the shared options, in declaration order."""

    @identity_match_options
    def command(**kwargs) -> None:
        """Stub command body."""

    cmd = click.command()(command)
    assert [param.name for param in cmd.params] == list(SHARED_PARAM_NAMES)


def test_defaults_come_from_match_settings() -> None:
    """Each shared option advertises the matching ``MatchSettings`` default."""

    @identity_match_options
    def command(**kwargs) -> None:
        """Stub command body."""

    params = _params_by_name(click.command()(command))
    defaults = MatchSettings()
    assert params["min_iou"].default == defaults.min_bbox_iou
    assert params["min_mask_iou"].default == defaults.min_mask_iou
    assert params["match_by"].default == defaults.match_by.value
    assert params["split_blocks"].default == defaults.split_blocks
    assert params["min_run_frames"].default == defaults.min_run_frames


@pytest.mark.parametrize("name", SHARED_PARAM_NAMES)
def test_both_commands_expose_the_same_option(name: str) -> None:
    """``update-pose`` and ``update-labels`` declare each shared option identically."""
    pose_param = _params_by_name(update_pose_command)[name]
    labels_param = _params_by_name(update_labels_command)[name]
    assert _spec(pose_param) == _spec(labels_param)


def test_shared_options_keep_their_position_in_both_commands() -> None:
    """The shared options stay contiguous and in the same order in both commands."""
    for command in (update_pose_command, update_labels_command):
        names = [param.name for param in command.params]
        start = names.index(SHARED_PARAM_NAMES[0])
        assert names[start : start + len(SHARED_PARAM_NAMES)] == list(SHARED_PARAM_NAMES)


def test_match_settings_from_options_round_trips_the_defaults() -> None:
    """Passing the advertised defaults back reproduces ``MatchSettings()``."""
    defaults = MatchSettings()
    assert (
        match_settings_from_options(
            min_iou=defaults.min_bbox_iou,
            min_mask_iou=defaults.min_mask_iou,
            match_by=defaults.match_by.value,
            split_blocks=defaults.split_blocks,
            min_run_frames=defaults.min_run_frames,
        )
        == defaults
    )


def test_match_settings_from_options_converts_the_metric_string() -> None:
    """The ``--match-by`` string Click parses becomes a ``MatchMetric``."""
    settings = match_settings_from_options(
        min_iou=0.25,
        min_mask_iou=0.75,
        match_by="segmentation",
        split_blocks=False,
        min_run_frames=7,
    )
    assert settings == MatchSettings(
        min_bbox_iou=0.25,
        min_mask_iou=0.75,
        match_by=MatchMetric.SEGMENTATION,
        split_blocks=False,
        min_run_frames=7,
    )
