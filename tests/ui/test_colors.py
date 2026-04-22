"""Tests for multi-class color utilities in jabs.ui.colors."""

import numpy as np
import pytest

try:
    from jabs.ui.colors import (
        BACKGROUND_COLOR,
        BEHAVIOR_COLOR,
        NOT_BEHAVIOR_COLOR,
        build_multiclass_color_lut,
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


def test_build_multiclass_color_lut_shape_no_behaviors():
    """LUT with no behaviors has shape (2, 4): background + None."""
    lut = build_multiclass_color_lut([], {})
    assert lut.shape == (2, 4)
    assert lut.dtype == np.uint8


def test_build_multiclass_color_lut_shape_with_behaviors():
    """LUT shape is N+2 rows: background, None, plus one per behavior."""
    color_map = make_behavior_color_map(["walk", "groom"])
    lut = build_multiclass_color_lut(["walk", "groom"], color_map)
    assert lut.shape == (4, 4)


def test_build_multiclass_color_lut_index_0_background():
    """Index 0 maps to the background color."""
    lut = build_multiclass_color_lut([], {})
    assert tuple(lut[0]) == BACKGROUND_COLOR.getRgb()


def test_build_multiclass_color_lut_index_1_not_behavior():
    """Index 1 maps to the not-behavior (None) color."""
    lut = build_multiclass_color_lut([], {})
    assert tuple(lut[1]) == NOT_BEHAVIOR_COLOR.getRgb()


def test_build_multiclass_color_lut_behavior_indices():
    """Behaviors appear at indices 2..N+1 in behavior_names order."""
    color_map = make_behavior_color_map(["walk", "groom"])
    lut = build_multiclass_color_lut(["walk", "groom"], color_map)
    assert tuple(lut[2]) == color_map["walk"].getRgb()
    assert tuple(lut[3]) == color_map["groom"].getRgb()
