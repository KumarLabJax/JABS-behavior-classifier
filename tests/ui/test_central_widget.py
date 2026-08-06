"""Tests for CentralWidget helpers that don't require instantiating the widget."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from jabs.classifier import MultiClassClassifier
from jabs.core.constants import MULTICLASS_NONE_BEHAVIOR

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


def test_frame_count_mismatch_message_none_when_equal():
    """Matching video/pose frame counts produce no warning message."""
    assert CentralWidget._frame_count_mismatch_message("v.avi", 100, 100) is None


def test_frame_count_mismatch_message_reports_counts():
    """A mismatch yields a message naming the video and both frame counts."""
    msg = CentralWidget._frame_count_mismatch_message("v.avi", 100, 90)
    assert msg is not None
    assert "v.avi" in msg
    assert "100" in msg
    assert "90" in msg


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


def _feature_check_stub(
    *,
    mode=None,
    window_size=5,
    current_behavior="Walking",
    behaviors=("Walking", "Grooming"),
    classifier=None,
    project_defaults=None,
) -> SimpleNamespace:
    """Build a stub self for the feature cache warning helpers."""
    if mode is None:
        mode = ClassifierMode.BINARY
    return SimpleNamespace(
        _window_size=window_size,
        _classifier=classifier,
        _controls=SimpleNamespace(current_behavior=current_behavior, behaviors=list(behaviors)),
        _project=SimpleNamespace(
            settings_manager=SimpleNamespace(classifier_mode=mode),
            get_project_defaults=lambda: project_defaults or {"window_size": 11},
            video_manager=SimpleNamespace(num_videos=4),
        ),
    )


def test_training_behaviors_binary_uses_current_behavior():
    """Binary training only reads labels for the behavior being trained."""
    stub = _feature_check_stub(current_behavior="Walking")
    assert CentralWidget._training_behaviors(stub) == ["Walking"]


def test_training_behaviors_multiclass_includes_none_class():
    """Multi-class training reads labels for every behavior plus the None class."""
    stub = _feature_check_stub(mode=ClassifierMode.MULTICLASS)
    assert CentralWidget._training_behaviors(stub) == [
        MULTICLASS_NONE_BEHAVIOR,
        "Walking",
        "Grooming",
    ]


def test_feature_window_size_binary_uses_control_value():
    """Binary mode uses the window size shown in the controls."""
    stub = _feature_check_stub(window_size=7)
    assert CentralWidget._feature_window_size(stub) == 7


def test_feature_window_size_multiclass_uses_classifier_settings():
    """Multi-class mode prefers the window size the classifier was configured with."""
    classifier = MagicMock(spec=MultiClassClassifier)
    classifier.project_settings = {"window_size": 30}
    stub = _feature_check_stub(mode=ClassifierMode.MULTICLASS, classifier=classifier)

    assert CentralWidget._feature_window_size(stub) == 30


def test_feature_window_size_multiclass_falls_back_to_project_defaults():
    """Without classifier settings, multi-class mode uses the project defaults."""
    classifier = MagicMock(spec=MultiClassClassifier)
    classifier.project_settings = None
    stub = _feature_check_stub(
        mode=ClassifierMode.MULTICLASS, classifier=classifier, project_defaults={"window_size": 11}
    )

    assert CentralWidget._feature_window_size(stub) == 11


def test_confirm_on_demand_features_skips_dialog_when_all_cached(monkeypatch):
    """Nothing is asked when every needed video already has cached features."""
    confirm = MagicMock(return_value=False)
    monkeypatch.setattr(
        "jabs.ui.main_window.central_widget.MessageDialog.confirm", confirm, raising=True
    )

    assert CentralWidget._confirm_on_demand_features(_feature_check_stub(), [], 5, "training")
    confirm.assert_not_called()


@pytest.mark.parametrize("answer", [True, False], ids=["continue", "cancel"])
def test_confirm_on_demand_features_returns_user_choice(monkeypatch, answer):
    """The user's answer to the warning decides whether the run proceeds."""
    confirm = MagicMock(return_value=answer)
    monkeypatch.setattr(
        "jabs.ui.main_window.central_widget.MessageDialog.confirm", confirm, raising=True
    )

    result = CentralWidget._confirm_on_demand_features(
        _feature_check_stub(), ["a.avi", "b.avi"], 5, "classification"
    )

    assert result is answer
    message = confirm.call_args.kwargs["message"]
    assert "<b>5</b>" in message
    assert "<b>2 videos</b>" in message
    assert "classification" in message
    assert "a.avi" in confirm.call_args.kwargs["details"]


def test_confirm_on_demand_features_suggests_jabs_init(monkeypatch):
    """The warning points at jabs-init, with the window size that is missing."""
    confirm = MagicMock(return_value=True)
    monkeypatch.setattr(
        "jabs.ui.main_window.central_widget.MessageDialog.confirm", confirm, raising=True
    )

    CentralWidget._confirm_on_demand_features(_feature_check_stub(), ["a.avi"], 30, "training")

    message = confirm.call_args.kwargs["message"]
    assert "jabs-init" in message
    assert "jabs-init -w 30" in message
    assert "jabs-features" not in message


def test_confirm_on_demand_features_uses_singular_for_one_video(monkeypatch):
    """A single uncached video reads as "1 video", not "1 videos"."""
    confirm = MagicMock(return_value=True)
    monkeypatch.setattr(
        "jabs.ui.main_window.central_widget.MessageDialog.confirm", confirm, raising=True
    )

    CentralWidget._confirm_on_demand_features(_feature_check_stub(), ["a.avi"], 5, "training")

    assert "<b>1 video</b>" in confirm.call_args.kwargs["message"]


def _gating_stub(confirmed: bool, missing=("a.avi",)) -> SimpleNamespace:
    """Build a stub self for the train/classify feature cache gate."""
    project = SimpleNamespace(
        videos_missing_window_features=MagicMock(return_value=list(missing)),
        labeled_identities=MagicMock(return_value={"a.avi": {0}}),
        settings_manager=SimpleNamespace(classifier_mode=ClassifierMode.BINARY),
    )
    return SimpleNamespace(
        _player_widget=MagicMock(),
        _ensure_classifier_for_mode=MagicMock(),
        _feature_window_size=MagicMock(return_value=5),
        _training_behaviors=MagicMock(return_value=["Walking"]),
        _confirm_on_demand_features=MagicMock(return_value=confirmed),
        _confirm_training_features=MagicMock(return_value=confirmed),
        _project=project,
        _classify_thread=None,
        _training_report_markdown="stale report",
        _classification_targets=None,
        _training_cache_targets=None,
    )


def test_train_aborted_when_the_feature_check_is_declined(monkeypatch):
    """Declining the uncached-features warning stops training before it starts."""
    training_thread = MagicMock()
    monkeypatch.setattr(
        "jabs.ui.main_window.central_widget.TrainingThread", training_thread, raising=True
    )
    stub = _gating_stub(confirmed=False)

    CentralWidget._train_button_clicked(stub)

    training_thread.assert_not_called()
    stub._confirm_training_features.assert_called_once_with(5)
    # the training report from a previous run is left alone since nothing ran
    assert stub._training_report_markdown == "stale report"


def test_training_feature_check_skips_annotation_reads_when_all_cached():
    """A fully cached project costs no annotation I/O to check.

    Working out the labeled identities means reading every annotation file, which
    is pointless when no video is missing features for the window size: a subset of
    those identities cannot be missing either.
    """
    stub = _gating_stub(confirmed=True, missing=())

    assert CentralWidget._confirm_training_features(stub, 5) is True

    stub._project.videos_missing_window_features.assert_called_once_with(5)
    stub._project.labeled_identities.assert_not_called()
    stub._confirm_on_demand_features.assert_not_called()
    assert stub._training_cache_targets == []


def test_training_feature_check_narrows_to_labeled_identities():
    """When something is missing, the check narrows to the labeled identities."""
    stub = _gating_stub(confirmed=True)

    assert CentralWidget._confirm_training_features(stub, 5) is True

    stub._project.labeled_identities.assert_called_once_with(["Walking"])
    assert stub._project.videos_missing_window_features.call_args_list[-1].kwargs == {
        "identities": {"a.avi": {0}}
    }
    stub._confirm_on_demand_features.assert_called_once_with(["a.avi"], 5, "training")
    assert stub._training_cache_targets == ["a.avi"]


def test_training_feature_check_records_nothing_when_declined():
    """Declining the warning leaves no videos recorded, since no run starts."""
    stub = _gating_stub(confirmed=False)

    assert CentralWidget._confirm_training_features(stub, 5) is False

    assert stub._training_cache_targets is None


def test_classify_aborted_when_user_declines_feature_computation(monkeypatch):
    """Declining the uncached-features warning stops classification before it starts."""
    classify_thread = MagicMock()
    monkeypatch.setattr(
        "jabs.ui.main_window.central_widget.ClassifyThread", classify_thread, raising=True
    )
    stub = _gating_stub(confirmed=False)

    CentralWidget._start_classification(stub, ["a.avi"])

    classify_thread.assert_not_called()
    stub._project.videos_missing_window_features.assert_called_once_with(5, videos=["a.avi"])
    assert stub._classification_targets is None


def _cleanup_stub(*, classification_targets=None, training_cache_targets=None):
    """Build a stub self for the thread cleanup handlers."""
    return SimpleNamespace(
        _training_thread=None,
        _classify_thread=None,
        _classification_targets=classification_targets,
        _training_cache_targets=training_cache_targets,
        _project=MagicMock(),
        feature_cache_changed=SimpleNamespace(emit=MagicMock()),
    )


def test_training_cleanup_invalidates_only_the_videos_it_read():
    """Training reads features only for labeled videos, so only those go stale."""
    stub = _cleanup_stub(training_cache_targets=["a.avi", "b.avi"])

    CentralWidget._cleanup_training_thread(stub)

    stub._project.invalidate_feature_cache_status.assert_called_once_with(["a.avi", "b.avi"])
    stub.feature_cache_changed.emit.assert_called_once()
    # the targets are consumed so a later cleanup cannot act on a stale list
    assert stub._training_cache_targets is None


def test_training_cleanup_without_known_targets_invalidates_everything():
    """With no recorded targets (no run started), nothing is assumed to be current."""
    stub = _cleanup_stub()

    CentralWidget._cleanup_training_thread(stub)

    stub._project.invalidate_feature_cache_status.assert_called_once_with(None)


def test_classify_cleanup_invalidates_only_classified_videos():
    """A single-video classification only invalidates that video."""
    stub = _cleanup_stub(classification_targets=["a.avi"])

    CentralWidget._cleanup_classify_thread(stub)

    stub._project.invalidate_feature_cache_status.assert_called_once_with(["a.avi"])
    stub.feature_cache_changed.emit.assert_called_once()


def test_classify_all_cleanup_invalidates_everything():
    """Classifying every video (targets None) invalidates every status."""
    stub = _cleanup_stub(classification_targets=None)

    CentralWidget._cleanup_classify_thread(stub)

    stub._project.invalidate_feature_cache_status.assert_called_once_with(None)


class _StopBeforeThreadStart(Exception):
    """Raised by the patched TrainingThread to end the handler under test early."""


def _train_stub() -> SimpleNamespace:
    """Build a gating stub that can reach the TrainingThread construction.

    Carries the attributes _train_button_clicked reads while building the thread's
    arguments, so the patched TrainingThread is what stops the handler.
    """
    stub = _gating_stub(confirmed=True)
    stub._classifier = MagicMock()
    stub._controls = SimpleNamespace(
        current_behavior="Walking", behaviors=["Walking"], all_kfold=False, kfold_value=1
    )
    stub._included_project_bout_totals = MagicMock(return_value=(5, 5))
    return stub


def test_train_proceeds_once_the_feature_check_passes(monkeypatch):
    """An accepted feature check lets training start."""
    monkeypatch.setattr(
        "jabs.ui.main_window.central_widget.TrainingThread",
        MagicMock(side_effect=_StopBeforeThreadStart),
        raising=True,
    )
    stub = _train_stub()

    # the patched thread stops the handler at the progress dialog setup this stub
    # does not provide, which is past the point of interest
    with pytest.raises(_StopBeforeThreadStart):
        CentralWidget._train_button_clicked(stub)

    stub._confirm_training_features.assert_called_once_with(5)
    assert stub._training_report_markdown is None
