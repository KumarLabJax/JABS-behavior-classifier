"""Expansion of a postprocessing config into a grid of concrete configs.

**This list syntax is specific to ``jabs-cli evaluate``.** Everywhere else a
postprocessing config is consumed - ``jabs-cli postprocess``, the GUI,
:class:`~jabs.behavior.postprocessing.PostprocessingPipeline` itself - a
parameter holds exactly one value, and a list is rejected by the stage
constructors. This module expands a swept config down to those single-valued
configs before any of them reaches a pipeline, so that contract is unchanged.

Tuning a pipeline means trying several values for a parameter and comparing the
results. Because a pipeline is a pure function of the predictions it is given,
``evaluate`` can classify once and then apply every combination to the cached
predictions, paying the expensive half of the work once however large the grid.

In an ``evaluate`` config, a parameter holding a list is a sweep axis and a
scalar is a fixed value:

```yaml
- stage_name: BoutStitchingStage
  parameters:
    max_stitch_gap: [15, 30, 45]
- stage_name: BoutDurationFilterStage
  parameters:
    min_duration: 60
```

That expands to three concrete configs. A config that is valid for the other
tools has no lists, so reading it this way expands it to a single point and
changes nothing.

Axes are taken only from **enabled** stages. A list on a disabled stage is left
untouched and never reaches a constructor, because ``PostprocessingPipeline``
drops disabled stages before instantiating them.
"""

from __future__ import annotations

import copy
import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

#: Default ceiling on the number of expanded combinations. No reclassification
#: happens per combination, but each one still costs a full pass of every stage
#: over every identity, so a large grid is slow.
DEFAULT_MAX_COMBINATIONS = 256

#: The list of stage dicts a pipeline is built from.
StageConfig = list[dict[str, Any]]


@dataclass(frozen=True)
class SweepAxis:
    """One parameter being varied, and the values it takes.

    Attributes:
        stage_index: Position of the owning stage in the config list.
        stage_name: Name of the owning stage, e.g. ``"BoutDurationFilterStage"``.
        parameter: Parameter being varied, e.g. ``"min_duration"``.
        values: The values to try, in the order given in the config.
    """

    stage_index: int
    stage_name: str
    parameter: str
    values: tuple[Any, ...]


@dataclass(frozen=True)
class SweepPoint:
    """One concrete combination of axis values, and the config it produces.

    Attributes:
        index: Position in the expanded grid, starting at zero.
        values: The value taken on each axis, parallel to the axis list.
        config: A single-valued stage config, with every swept parameter
            replaced by its value for this point.
    """

    index: int
    values: tuple[Any, ...]
    config: StageConfig


def find_sweep_axes(config: StageConfig) -> list[SweepAxis]:
    """Find the parameters a config varies.

    Args:
        config: List of stage config dicts, as a pipeline is built from.

    Returns:
        One axis per list-valued parameter of an enabled stage, ordered by stage
        position and then by parameter order within that stage. Empty when the
        config varies nothing.

    Raises:
        ValueError: If a swept parameter holds an empty list, which would expand
            to no combinations at all.
    """
    axes: list[SweepAxis] = []
    for index, stage in enumerate(config):
        if not isinstance(stage, dict) or not stage.get("enabled", True):
            continue
        parameters = stage.get("parameters") or {}
        if not isinstance(parameters, dict):
            continue
        for parameter, value in parameters.items():
            if not isinstance(value, list | tuple):
                continue
            if len(value) == 0:
                raise ValueError(
                    f"{stage.get('stage_name', f'stage {index}')}.{parameter} is an empty "
                    "list, so there is nothing to sweep. Give it at least one value."
                )
            axes.append(
                SweepAxis(
                    stage_index=index,
                    stage_name=str(stage.get("stage_name", f"stage {index}")),
                    parameter=parameter,
                    values=tuple(value),
                )
            )
    return axes


def axis_column_names(axes: Sequence[SweepAxis]) -> list[str]:
    """Build short, unambiguous column headings for a set of axes.

    A parameter name alone is used when it is unique across the axes, and is
    otherwise qualified with its stage name, so two stages sweeping the same
    parameter stay distinguishable.

    Args:
        axes: The axes to name, in column order.

    Returns:
        One heading per axis, in the same order.
    """
    counts: dict[str, int] = {}
    for axis in axes:
        counts[axis.parameter] = counts.get(axis.parameter, 0) + 1
    return [
        axis.parameter
        if counts[axis.parameter] == 1
        else f"{axis.stage_name.removesuffix('Stage')}.{axis.parameter}"
        for axis in axes
    ]


def format_point_label(axes: Sequence[SweepAxis], point: SweepPoint) -> str:
    """Describe one combination in a single line.

    Args:
        axes: The axes the point's values correspond to.
        point: The combination to describe.

    Returns:
        e.g. ``"min_duration=60, max_stitch_gap=30"``, or ``"default"`` when the
        config varies nothing.
    """
    if not axes:
        return "default"
    names = axis_column_names(axes)
    return ", ".join(f"{name}={value}" for name, value in zip(names, point.values, strict=True))


def expand_sweep(
    config: StageConfig,
    max_combinations: int = DEFAULT_MAX_COMBINATIONS,
) -> tuple[list[SweepAxis], list[SweepPoint]]:
    """Expand a config into every combination of its swept parameters.

    Args:
        config: List of stage config dicts, possibly with list-valued parameters.
        max_combinations: Ceiling on the size of the expanded grid.

    Returns:
        Tuple of ``(axes, points)``. When the config varies nothing, ``axes`` is
        empty and ``points`` holds a single point carrying the config unchanged,
        so callers need no special case.

    Raises:
        ValueError: If a swept parameter holds an empty list, or the grid would
            exceed ``max_combinations``.
    """
    axes = find_sweep_axes(config)
    if not axes:
        return [], [SweepPoint(index=0, values=(), config=copy.deepcopy(config))]

    total = 1
    for axis in axes:
        total *= len(axis.values)
    if total > max_combinations:
        detail = ", ".join(f"{a.parameter} ({len(a.values)})" for a in axes)
        raise ValueError(
            f"This config expands to {total} combinations ({detail}), above the limit of "
            f"{max_combinations}. Narrow the value lists, or raise the limit with "
            "--max-sweep-combinations."
        )

    points: list[SweepPoint] = []
    # product varies the last axis fastest, which reads naturally down a table
    for index, combination in enumerate(itertools.product(*(a.values for a in axes))):
        concrete = copy.deepcopy(config)
        for axis, value in zip(axes, combination, strict=True):
            concrete[axis.stage_index]["parameters"][axis.parameter] = value
        points.append(SweepPoint(index=index, values=combination, config=concrete))

    return axes, points
