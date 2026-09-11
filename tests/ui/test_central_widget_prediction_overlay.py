"""Tests for the prediction overlay CentralWidget hands to the video export."""

from types import SimpleNamespace

import numpy as np
import pytest

from jabs.core.enums import ClassifierMode, PredictionType

try:
    from jabs.ui.main_window.central_widget import CentralWidget

    SKIP_UI_TESTS = False
    SKIP_REASON = None
except ImportError as e:
    SKIP_UI_TESTS = True
    SKIP_REASON = f"Qt/UI dependencies not available: {e}"

pytestmark = pytest.mark.skipif(
    SKIP_UI_TESTS,
    reason=SKIP_REASON if SKIP_UI_TESTS else "",
)

_RAW = [np.array([1, 1, 0, 0], dtype=np.int8)]
_POSTPROCESSED = [np.array([1, 1, 1, 0], dtype=np.int8)]

# Sentinel for "present"; prediction_overlay() only checks these for None.
_PRESENT = object()


def _widget(
    *,
    classifier_mode=None,
    predictions=None,
    postprocessed: bool = False,
    color_lut=None,
    project=_PRESENT,
    loaded_video=_PRESENT,
    pose_est=_PRESENT,
) -> SimpleNamespace:
    """Stand-in exposing what prediction_overlay() reads from self."""
    mode = ClassifierMode.BINARY if classifier_mode is None else classifier_mode
    pose = SimpleNamespace(num_identities=1) if pose_est is _PRESENT else pose_est
    return SimpleNamespace(
        _project=(
            None
            if project is None
            else SimpleNamespace(settings_manager=SimpleNamespace(classifier_mode=mode))
        ),
        _loaded_video=loaded_video,
        _pose_est=pose,
        _predictions={0: _RAW[0]} if predictions is None else predictions,
        _jabs_timeline=SimpleNamespace(multiclass_color_lut=color_lut),
        _controls=SimpleNamespace(behaviors=["Grooming", "Rearing"]),
        _player_widget=SimpleNamespace(num_frames=4),
        _showing_postprocessed_predictions=postprocessed,
        behavior="Grooming",
        _get_prediction_list=lambda: (_POSTPROCESSED if postprocessed else _RAW, []),
        _build_multiclass_overlay_labels=lambda: [np.array([1, 2, 3, 0], dtype=np.int16)],
    )


def test_binary_overlay_carries_the_displayed_predictions() -> None:
    """The export draws exactly what the player's prediction overlay would."""
    overlay = CentralWidget.prediction_overlay(_widget())

    assert overlay is not None
    assert overlay.labels is _RAW
    assert overlay.color_lut is None
    assert overlay.caption == "Grooming predictions (raw)"


def test_binary_overlay_follows_the_postprocessed_choice() -> None:
    """Switching the timeline to post-processed predictions exports those instead."""
    overlay = CentralWidget.prediction_overlay(_widget(postprocessed=True))

    assert overlay.labels is _POSTPROCESSED
    assert overlay.caption == "Grooming predictions (post-processed)"


def test_multiclass_overlay_uses_the_project_color_table() -> None:
    """Multi-class exports carry the same colors and class names as the timeline."""
    lut = np.array(
        [[0, 0, 0, 255], [1, 1, 1, 255], [2, 2, 2, 255], [3, 3, 3, 255]], dtype=np.uint8
    )

    overlay = CentralWidget.prediction_overlay(
        _widget(classifier_mode=ClassifierMode.MULTICLASS, color_lut=lut)
    )

    assert overlay is not None
    assert overlay.color_lut is lut
    assert [name for name, _ in overlay.legend] == [
        "None",
        "Grooming",
        "Rearing",
        "no prediction",
    ]


@pytest.mark.parametrize(
    "widget_kwargs",
    [
        {"predictions": {}},
        {"project": None},
        {"loaded_video": None},
        {"pose_est": None},
        {"classifier_mode": ClassifierMode.MULTICLASS, "color_lut": None},
    ],
    ids=[
        "nothing-classified",
        "no-project",
        "no-video",
        "no-pose",
        "multiclass-without-a-color-table",
    ],
)
def test_no_overlay_when_there_is_nothing_to_draw(widget_kwargs: dict) -> None:
    """Every state with no drawable predictions returns None, not a blank overlay."""
    assert CentralWidget.prediction_overlay(_widget(**widget_kwargs)) is None


@pytest.mark.parametrize(
    ("prediction_type", "keys_match", "expected"),
    [
        (PredictionType.POSTPROCESSED, True, True),
        (PredictionType.POSTPROCESSED, False, False),
        (PredictionType.RAW, True, False),
    ],
    ids=["requested-and-available", "requested-but-incomplete", "not-requested"],
)
def test_postprocessed_flag_requires_the_data_to_exist(
    prediction_type, keys_match: bool, expected: bool
) -> None:
    """Asking for post-processed predictions is not enough; they have to be there.

    The display silently falls back to raw predictions when they are not, and the
    caption has to fall back with it rather than mislabeling the export.
    """
    widget = SimpleNamespace(
        prediction_type=prediction_type,
        _predictions={0: _RAW[0]},
        _predictions_postprocessed={0: _POSTPROCESSED[0]} if keys_match else {},
    )

    assert CentralWidget._showing_postprocessed_predictions.fget(widget) is expected
