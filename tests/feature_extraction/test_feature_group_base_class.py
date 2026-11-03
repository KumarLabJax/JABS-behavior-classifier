"""Tests for timing instrumentation in the FeatureGroup base class."""

from __future__ import annotations

import typing
from types import SimpleNamespace

import numpy as np

from jabs.feature_extraction.feature_base_class import Feature
from jabs.feature_extraction.feature_group_base_class import FeatureGroup
from jabs.feature_extraction.feature_timing import (
    GROUP_SETUP_FEATURE,
    FeatureTimings,
    TimingKey,
    TimingStage,
)


class _StubFeature(Feature):
    """Feature that returns constant values without touching pose data."""

    _name = "stub"
    _feature_names: typing.ClassVar[list[str]] = ["value"]

    def per_frame(self, identity: int) -> dict[str, np.ndarray]:
        """Return a constant per-frame value."""
        return {"value": np.ones(5)}

    def window(self, identity: int, window_size: int, per_frame_features: dict) -> dict:
        """Return a constant window value."""
        return {"mean": {"value": np.ones(5)}}


class _OtherStubFeature(_StubFeature):
    """Second stub so the group has more than one module."""

    _name = "other_stub"


class _StubGroup(FeatureGroup):
    """Feature group built from stub features."""

    _name = "stub_group"
    _features: typing.ClassVar[dict[str, type[Feature]]] = {
        _StubFeature.name(): _StubFeature,
        _OtherStubFeature.name(): _OtherStubFeature,
    }

    def _init_feature_mods(self, identity: int) -> dict[str, Feature]:
        """Instantiate every enabled stub feature."""
        return {
            name: self._features[name](self._poses, self._pixel_scale)
            for name in self._enabled_features
        }


def _make_group() -> _StubGroup:
    """Build a stub group with minimal fake pose data."""
    return _StubGroup(SimpleNamespace(fps=30), 1.0)


def test_timing_disabled_by_default() -> None:
    """Groups compute features normally with no collector attached."""
    group = _make_group()
    assert group.timings is None
    per_frame = group.per_frame(0)
    assert set(per_frame) == {"stub", "other_stub"}
    assert set(group.window(0, 5, per_frame)) == {"stub", "other_stub"}


def test_per_frame_and_window_record_timings() -> None:
    """Each enabled module is timed per stage, plus shared group setup."""
    group = _make_group()
    timings = FeatureTimings()
    group.timings = timings

    per_frame = group.per_frame(0)
    window = group.window(0, 5, per_frame)

    np.testing.assert_array_equal(per_frame["stub"]["value"], np.ones(5))
    np.testing.assert_array_equal(window["other_stub"]["mean"]["value"], np.ones(5))
    assert set(timings.stats()) == {
        TimingKey("stub_group", GROUP_SETUP_FEATURE, TimingStage.SETUP),
        TimingKey("stub_group", "stub", TimingStage.PER_FRAME),
        TimingKey("stub_group", "other_stub", TimingStage.PER_FRAME),
        TimingKey("stub_group", "stub", TimingStage.WINDOW),
        TimingKey("stub_group", "other_stub", TimingStage.WINDOW),
    }
    setup = timings.stats()[TimingKey("stub_group", GROUP_SETUP_FEATURE, TimingStage.SETUP)]
    assert setup.calls == 2


def test_only_enabled_features_are_timed() -> None:
    """Disabled feature modules produce no timing entries."""
    group = _make_group()
    group._enabled_features = ["stub"]
    group.timings = FeatureTimings()
    group.per_frame(0)
    assert {k.feature for k in group.timings.stats()} == {GROUP_SETUP_FEATURE, "stub"}


def test_timings_can_be_detached() -> None:
    """Setting timings to None stops recording."""
    group = _make_group()
    timings = FeatureTimings()
    group.timings = timings
    group.timings = None
    group.per_frame(0)
    assert len(timings) == 0
