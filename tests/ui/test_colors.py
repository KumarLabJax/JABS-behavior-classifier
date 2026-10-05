"""Tests for multi-class color utilities in jabs.ui.colors."""

from collections.abc import Callable

import numpy as np
import pytest

from jabs.core.constants import MULTICLASS_NONE_BEHAVIOR

try:
    from PySide6.QtGui import QColor

    from jabs.ui.colors import (
        BACKGROUND_COLOR,
        BEHAVIOR_COLOR,
        NOT_BEHAVIOR_COLOR,
        build_multiclass_color_lut,
        is_color_light,
        make_behavior_color_map,
    )

    SKIP_UI_TESTS = False
except ImportError as e:
    SKIP_UI_TESTS = True
    SKIP_REASON = f"Qt/UI dependencies not available: {e}"

pytestmark = pytest.mark.skipif(
    SKIP_UI_TESTS,
    reason=SKIP_REASON if SKIP_UI_TESTS else "",
)


def test_make_behavior_color_map_empty():
    """Empty input returns an empty dict."""
    assert make_behavior_color_map([]) == {}


def test_make_behavior_color_map_keys():
    """Returned dict has one entry per behavior name."""
    result = make_behavior_color_map(["walk", "groom", "rear"])
    assert set(result.keys()) == {"walk", "groom", "rear"}


def test_make_behavior_color_map_distinct():
    """All generated colors are visually distinct from each other."""
    result = make_behavior_color_map(["walk", "groom", "rear", "eat"])
    colors = [c.getRgb()[:3] for c in result.values()]
    assert len(colors) == len(set(colors))


def test_make_behavior_color_map_no_background_collision():
    """Generated colors do not match the background gray."""
    bg = BACKGROUND_COLOR.getRgb()[:3]
    for color in make_behavior_color_map(["walk", "groom"]).values():
        assert color.getRgb()[:3] != bg


def test_make_behavior_color_map_no_not_behavior_collision():
    """Generated colors do not match the not-behavior blue."""
    nb = NOT_BEHAVIOR_COLOR.getRgb()[:3]
    for color in make_behavior_color_map(["walk", "groom"]).values():
        assert color.getRgb()[:3] != nb


def test_make_behavior_color_map_no_behavior_collision():
    """Generated colors do not match the behavior orange."""
    beh = BEHAVIOR_COLOR.getRgb()[:3]
    for color in make_behavior_color_map(["walk", "groom"]).values():
        assert color.getRgb()[:3] != beh


def test_make_behavior_color_map_deterministic():
    """Same input always produces the same colors."""
    a = make_behavior_color_map(["walk", "groom"])
    b = make_behavior_color_map(["walk", "groom"])
    assert {k: v.getRgb() for k, v in a.items()} == {k: v.getRgb() for k, v in b.items()}


def test_make_behavior_color_map_single():
    """Single-behavior input returns a dict with that one key."""
    assert "walk" in make_behavior_color_map(["walk"])


@pytest.mark.parametrize(
    ("behavior_names", "expected_rows"),
    [([], 2), (["walk", "groom"], 4)],
    ids=["no-behaviors", "two-behaviors"],
)
def test_build_multiclass_color_lut_layout(behavior_names: list[str], expected_rows: int) -> None:
    """The LUT has N+2 RGBA rows: background at index 0, the None color at index 1."""
    color_map = make_behavior_color_map(behavior_names)
    lut = build_multiclass_color_lut(behavior_names, color_map)

    assert lut.shape == (expected_rows, 4)
    assert lut.dtype == np.uint8
    assert tuple(lut[0]) == BACKGROUND_COLOR.getRgb()
    assert tuple(lut[1]) == NOT_BEHAVIOR_COLOR.getRgb()


def test_build_multiclass_color_lut_behavior_indices():
    """Behaviors appear at indices 2..N+1 in behavior_names order."""
    color_map = make_behavior_color_map(["walk", "groom"])
    lut = build_multiclass_color_lut(["walk", "groom"], color_map)
    assert tuple(lut[2]) == color_map["walk"].getRgb()
    assert tuple(lut[3]) == color_map["groom"].getRgb()


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: make_behavior_color_map([MULTICLASS_NONE_BEHAVIOR]), "reserved"),
        (lambda: make_behavior_color_map(["walk", "walk"]), "duplicate"),
        (lambda: build_multiclass_color_lut([MULTICLASS_NONE_BEHAVIOR], {}), "reserved"),
        (lambda: build_multiclass_color_lut(["walk", "walk"], {"walk": None}), "duplicate"),
    ],
    ids=[
        "color-map-reserved-name",
        "color-map-duplicate-names",
        "lut-reserved-name",
        "lut-duplicate-names",
    ],
)
def test_invalid_behavior_names_raise_value_error(call: Callable[[], object], match: str) -> None:
    """Both builders reject the reserved None behavior name and duplicate names.

    The calls are lambdas so that nothing needing Qt is evaluated when the
    parametrization is collected.
    """
    with pytest.raises(ValueError, match=match):
        call()


def test_build_multiclass_color_lut_missing_key_raises():
    """Name in behavior_names absent from color_map raises ValueError."""
    with pytest.raises(ValueError, match="missing from color_map"):
        build_multiclass_color_lut(["walk"], {})


def test_is_color_light_dark_color():
    """A dark color is not considered light."""
    assert is_color_light(QColor(20, 20, 20)) is False


def test_is_color_light_light_color():
    """A near-white color is considered light."""
    assert is_color_light(QColor(240, 240, 240)) is True


def test_is_color_light_uses_green_weighting():
    """Saturated green reads as light while saturated blue does not (BT.709)."""
    assert is_color_light(QColor(0, 255, 0)) is True
    assert is_color_light(QColor(0, 0, 255)) is False
