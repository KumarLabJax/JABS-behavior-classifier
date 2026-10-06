"""Tests for expanding an evaluate postprocessing config into a sweep grid."""

import pytest

from jabs.behavior.postprocessing import PostprocessingPipeline
from jabs.scripts.cli.evaluate_sweep import (
    axis_column_names,
    expand_sweep,
    find_sweep_axes,
    format_point_label,
)


def _stage(name: str, enabled: bool = True, **parameters):
    """Build a stage config dict."""
    return {"stage_name": name, "enabled": enabled, "parameters": dict(parameters)}


DURATION = "BoutDurationFilterStage"
STITCH = "BoutStitchingStage"
GAP = "GapInterpolationStage"


# -----------------------------------------------------------------------------
# find_sweep_axes
# -----------------------------------------------------------------------------


def test_no_axes_for_an_all_scalar_config() -> None:
    """A config valid for the other tools varies nothing."""
    assert find_sweep_axes([_stage(DURATION, min_duration=60)]) == []


def test_a_list_parameter_becomes_an_axis() -> None:
    """The list syntax that is specific to evaluate."""
    (axis,) = find_sweep_axes([_stage(DURATION, min_duration=[30, 60, 90])])
    assert axis.stage_name == DURATION
    assert axis.parameter == "min_duration"
    assert axis.values == (30, 60, 90)
    assert axis.stage_index == 0


def test_axes_are_ordered_by_stage_then_parameter() -> None:
    """Column order follows the config, so the table reads like the file."""
    axes = find_sweep_axes(
        [
            _stage(GAP, max_interpolation_gap=[5, 15]),
            _stage(STITCH, max_stitch_gap=[15, 30]),
            _stage(DURATION, min_duration=[30, 60]),
        ]
    )
    assert [a.parameter for a in axes] == [
        "max_interpolation_gap",
        "max_stitch_gap",
        "min_duration",
    ]


def test_a_disabled_stage_contributes_no_axis() -> None:
    """A disabled stage never reaches a constructor, so sweeping it is pointless."""
    config = [
        _stage(DURATION, enabled=False, min_duration=[30, 60, 90]),
        _stage(STITCH, max_stitch_gap=[15, 30]),
    ]
    axes = find_sweep_axes(config)
    assert [a.parameter for a in axes] == ["max_stitch_gap"]


def test_a_tuple_is_also_an_axis() -> None:
    """YAML gives lists, but a programmatic caller may pass a tuple."""
    (axis,) = find_sweep_axes([_stage(DURATION, min_duration=(30, 60))])
    assert axis.values == (30, 60)


def test_a_single_element_list_is_still_an_axis() -> None:
    """It expands to one point, and keeps its column in the table."""
    (axis,) = find_sweep_axes([_stage(DURATION, min_duration=[60])])
    assert axis.values == (60,)


def test_an_empty_list_is_rejected() -> None:
    """Expanding it would yield no combinations at all."""
    with pytest.raises(ValueError, match="empty list"):
        find_sweep_axes([_stage(DURATION, min_duration=[])])


def test_missing_and_none_parameters_are_tolerated() -> None:
    """A stage with no parameters block must not crash axis discovery."""
    assert find_sweep_axes([{"stage_name": DURATION}]) == []
    assert find_sweep_axes([{"stage_name": DURATION, "parameters": None}]) == []


def test_a_non_dict_stage_entry_is_skipped() -> None:
    """Malformed config is the pipeline's to report, not this module's to crash on."""
    assert find_sweep_axes(["nonsense"]) == []  # type: ignore[list-item]


# -----------------------------------------------------------------------------
# expand_sweep
# -----------------------------------------------------------------------------


def test_a_scalar_config_expands_to_one_unchanged_point() -> None:
    """Callers need no special case for the non-sweep path."""
    config = [_stage(DURATION, min_duration=60)]
    axes, points = expand_sweep(config)
    assert axes == []
    assert len(points) == 1
    assert points[0].values == ()
    assert points[0].config == config


def test_expansion_is_the_cartesian_product() -> None:
    """Every combination of the axis values, in config order."""
    config = [_stage(STITCH, max_stitch_gap=[15, 30]), _stage(DURATION, min_duration=[30, 60, 90])]
    axes, points = expand_sweep(config)
    assert len(axes) == 2
    assert len(points) == 6
    assert [p.values for p in points] == [
        (15, 30),
        (15, 60),
        (15, 90),
        (30, 30),
        (30, 60),
        (30, 90),
    ]


def test_each_point_carries_a_single_valued_config() -> None:
    """The whole point: what reaches a pipeline has no lists in it."""
    config = [_stage(STITCH, max_stitch_gap=[15, 30]), _stage(DURATION, min_duration=[30, 60])]
    _, points = expand_sweep(config)
    for point in points:
        for stage in point.config:
            for value in stage["parameters"].values():
                assert not isinstance(value, list | tuple), f"list survived: {stage}"


def test_each_point_builds_a_real_pipeline() -> None:
    """Expanded configs satisfy the shared, single-value-only contract."""
    config = [_stage(STITCH, max_stitch_gap=[15, 30]), _stage(DURATION, min_duration=[30, 60])]
    _, points = expand_sweep(config)
    for point in points:
        pipeline = PostprocessingPipeline(point.config)
        assert [type(s).__name__ for s in pipeline.stages] == [STITCH, DURATION]


def test_expansion_leaves_scalar_parameters_alone() -> None:
    """A fixed parameter keeps its value in every point."""
    config = [_stage(GAP, max_interpolation_gap=15), _stage(DURATION, min_duration=[30, 60])]
    _, points = expand_sweep(config)
    assert all(p.config[0]["parameters"]["max_interpolation_gap"] == 15 for p in points)
    assert [p.config[1]["parameters"]["min_duration"] for p in points] == [30, 60]


def test_expansion_does_not_mutate_the_input_config() -> None:
    """The caller's parsed config is reused for reporting, so it must survive."""
    config = [_stage(DURATION, min_duration=[30, 60])]
    expand_sweep(config)
    assert config[0]["parameters"]["min_duration"] == [30, 60]


def test_points_do_not_share_mutable_state() -> None:
    """A deep copy per point, so editing one cannot bleed into another."""
    config = [_stage(DURATION, min_duration=[30, 60])]
    _, points = expand_sweep(config)
    points[0].config[0]["parameters"]["min_duration"] = 999
    assert points[1].config[0]["parameters"]["min_duration"] == 60


def test_a_disabled_swept_stage_keeps_its_list_but_adds_no_combinations() -> None:
    """The pipeline drops it before instantiating, so the list is harmless."""
    config = [
        _stage(DURATION, enabled=False, min_duration=[30, 60, 90]),
        _stage(STITCH, max_stitch_gap=[15, 30]),
    ]
    axes, points = expand_sweep(config)
    assert len(axes) == 1
    assert len(points) == 2
    # the disabled stage is untouched, and building a pipeline still works
    assert points[0].config[0]["parameters"]["min_duration"] == [30, 60, 90]
    assert [type(s).__name__ for s in PostprocessingPipeline(points[0].config).stages] == [STITCH]


def test_the_combination_ceiling_is_enforced() -> None:
    """Each combination costs a full pass over every identity."""
    config = [
        _stage(DURATION, min_duration=list(range(10))),
        _stage(STITCH, max_stitch_gap=list(range(10))),
    ]
    with pytest.raises(ValueError, match="expands to 100 combinations"):
        expand_sweep(config, max_combinations=99)


def test_the_ceiling_error_names_the_axes_and_the_flag() -> None:
    """The error has to say which axis is large and how to proceed."""
    config = [_stage(DURATION, min_duration=[1, 2, 3])]
    with pytest.raises(ValueError, match=r"min_duration \(3\)"):
        expand_sweep(config, max_combinations=2)
    with pytest.raises(ValueError, match="--max-sweep-combinations"):
        expand_sweep(config, max_combinations=2)


def test_a_grid_exactly_at_the_ceiling_is_allowed() -> None:
    """The limit is inclusive."""
    config = [_stage(DURATION, min_duration=[1, 2, 3])]
    _, points = expand_sweep(config, max_combinations=3)
    assert len(points) == 3


# -----------------------------------------------------------------------------
# labels and column names
# -----------------------------------------------------------------------------


def test_column_names_use_the_bare_parameter_when_unique() -> None:
    """No stage prefix when the parameter name is unambiguous."""
    axes = find_sweep_axes(
        [_stage(STITCH, max_stitch_gap=[15, 30]), _stage(DURATION, min_duration=[30, 60])]
    )
    assert axis_column_names(axes) == ["max_stitch_gap", "min_duration"]


def test_column_names_qualify_a_parameter_two_stages_share() -> None:
    """Contrived today, but the columns must stay distinguishable if it happens."""
    axes = find_sweep_axes(
        [_stage(STITCH, max_stitch_gap=[15, 30]), _stage(DURATION, max_stitch_gap=[1, 2])]
    )
    assert axis_column_names(axes) == [
        "BoutStitching.max_stitch_gap",
        "BoutDurationFilter.max_stitch_gap",
    ]


def test_point_label_lists_each_axis_value() -> None:
    """The label identifies a row in the sweep table."""
    config = [_stage(STITCH, max_stitch_gap=[15, 30]), _stage(DURATION, min_duration=[30, 60])]
    axes, points = expand_sweep(config)
    assert format_point_label(axes, points[0]) == "max_stitch_gap=15, min_duration=30"
    assert format_point_label(axes, points[3]) == "max_stitch_gap=30, min_duration=60"


def test_point_label_for_a_config_that_varies_nothing() -> None:
    """The non-sweep path still needs a label."""
    axes, points = expand_sweep([_stage(DURATION, min_duration=60)])
    assert format_point_label(axes, points[0]) == "default"
