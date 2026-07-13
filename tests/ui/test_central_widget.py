"""Tests for CentralWidget helpers that don't require instantiating the widget."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

try:
    from jabs.core.enums import ClassifierMode
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


def _stub_widget(excluded: set[str]) -> SimpleNamespace:
    """Minimal stand-in exposing what _included_counts reads from self."""
    return SimpleNamespace(
        _project=SimpleNamespace(
            settings_manager=SimpleNamespace(is_video_excluded=lambda v: v in excluded)
        )
    )


def test_included_counts_drops_excluded_videos():
    """Excluded videos are removed from the counts used for train-button thresholds."""
    counts = {
        "a.avi": {0: {"fragmented_frame_counts": (30, 30)}},
        "b.avi": {0: {"fragmented_frame_counts": (30, 30)}},
        "c.avi": {0: {"fragmented_frame_counts": (30, 30)}},
    }
    result = CentralWidget._included_counts(_stub_widget({"b.avi"}), counts)

    assert set(result.keys()) == {"a.avi", "c.avi"}
    # surviving entries are passed through unchanged
    assert result["a.avi"] == counts["a.avi"]


def test_included_counts_no_exclusions_returns_all():
    """With nothing excluded, all videos are retained."""
    counts = {"a.avi": {0: {}}, "b.avi": {0: {}}}
    result = CentralWidget._included_counts(_stub_widget(set()), counts)
    assert set(result.keys()) == {"a.avi", "b.avi"}


def test_included_counts_none_returns_empty():
    """None counts (not yet computed) return an empty dict instead of raising."""
    assert CentralWidget._included_counts(_stub_widget(set()), None) == {}


def _bout_stub_widget(counts: dict, excluded: set[str]) -> SimpleNamespace:
    """Stand-in exposing what _included_project_bout_totals reads from self."""
    stub = _stub_widget(excluded)
    stub._counts = counts
    return stub


def test_included_project_bout_totals_excludes_excluded_videos():
    """Bout totals for the report sum only non-excluded videos."""
    counts = {
        "a.avi": {0: {"unfragmented_bout_counts": (3, 2)}},
        "b.avi": {0: {"unfragmented_bout_counts": (10, 10)}},  # excluded
    }
    stub = _bout_stub_widget(counts, {"b.avi"})
    assert CentralWidget._included_project_bout_totals(stub) == (3, 2)


def test_included_project_bout_totals_handles_none_counts():
    """No counts yet -> zero totals (no crash)."""
    stub = _bout_stub_widget(None, set())
    assert CentralWidget._included_project_bout_totals(stub) == (0, 0)


# ---------------------------------------------------------------------------
# Single-video classification (context-menu path)
# ---------------------------------------------------------------------------


def test_set_classify_enabled_updates_control_and_emits():
    """_set_classify_enabled updates the control and emits classify_availability_changed."""
    controls = SimpleNamespace()
    emitted: list[bool] = []
    stub = SimpleNamespace(
        _controls=controls,
        classify_availability_changed=SimpleNamespace(emit=emitted.append),
    )

    CentralWidget._set_classify_enabled(stub, True)

    assert controls.classify_button_enabled is True
    assert emitted == [True]


def test_classify_single_video_warns_when_not_ready(monkeypatch):
    """With no classifier ready, classify_single_video warns and does not start a run."""
    warn = MagicMock()
    monkeypatch.setattr("jabs.ui.main_window.central_widget.MessageDialog.warning", warn)
    stub = SimpleNamespace(
        _controls=SimpleNamespace(classify_button_enabled=False),
        _start_classification=MagicMock(),
    )

    CentralWidget.classify_single_video(stub, "v.avi")

    warn.assert_called_once()
    stub._start_classification.assert_not_called()


def test_classify_single_video_starts_when_ready():
    """When a classifier is ready, classify_single_video starts a single-video run."""
    stub = SimpleNamespace(
        _controls=SimpleNamespace(classify_button_enabled=True),
        _start_classification=MagicMock(),
    )

    CentralWidget.classify_single_video(stub, "v.avi")

    stub._start_classification.assert_called_once_with(["v.avi"])


def test_start_classification_ignored_when_thread_running():
    """A second classification request is ignored while one is already in flight."""
    stub = SimpleNamespace(
        _classify_thread=object(),  # a run is already active
        _player_widget=MagicMock(),
    )

    CentralWidget._start_classification(stub, None)

    # early return: playback is not stopped and no new thread work begins
    stub._player_widget.stop.assert_not_called()


def _completion_stub(targets, loaded_video_name):
    """Build a stub self for _classify_thread_complete with mocked collaborators."""
    return SimpleNamespace(
        _classification_targets=targets,
        _loaded_video=(
            SimpleNamespace(name=loaded_video_name) if loaded_video_name is not None else None
        ),
        _cleanup_progress_dialog=MagicMock(),
        _cleanup_classify_thread=MagicMock(),
        status_message=SimpleNamespace(emit=MagicMock()),
        request_video_selection=SimpleNamespace(emit=MagicMock()),
        _set_prediction_vis=MagicMock(),
        _project=SimpleNamespace(
            settings_manager=SimpleNamespace(classifier_mode=ClassifierMode.BINARY)
        ),
        _predictions={"original": 1},
        _probabilities={},
        _predictions_postprocessed={},
    )


_COMPLETION_OUTPUT = {
    "predictions": {0: "p"},
    "probabilities": {0: "q"},
    "predictions_postprocessed": {0: "r"},
    "class_names": None,
}


def test_classify_complete_refreshes_when_all_videos_classified():
    """Classifying all videos refreshes the display from the completion payload."""
    stub = _completion_stub(targets=None, loaded_video_name="loaded.avi")

    CentralWidget._classify_thread_complete(stub, _COMPLETION_OUTPUT, 1234)

    assert stub._predictions == {0: "p"}
    stub._set_prediction_vis.assert_called_once()
    stub.request_video_selection.emit.assert_not_called()
    assert stub._classification_targets is None


def test_classify_complete_refreshes_when_loaded_video_in_subset():
    """Classifying a subset that includes the loaded video refreshes the display."""
    stub = _completion_stub(targets=["loaded.avi"], loaded_video_name="loaded.avi")

    CentralWidget._classify_thread_complete(stub, _COMPLETION_OUTPUT, 1234)

    assert stub._predictions == {0: "p"}
    stub._set_prediction_vis.assert_called_once()
    stub.request_video_selection.emit.assert_not_called()


def test_classify_complete_autoswitches_to_other_video():
    """Classifying a single non-loaded video switches to it without touching the current view."""
    stub = _completion_stub(targets=["other.avi"], loaded_video_name="loaded.avi")

    CentralWidget._classify_thread_complete(stub, _COMPLETION_OUTPUT, 1234)

    # current predictions are left untouched; we request a switch to the classified video
    assert stub._predictions == {"original": 1}
    stub._set_prediction_vis.assert_not_called()
    stub.request_video_selection.emit.assert_called_once_with("other.avi")
