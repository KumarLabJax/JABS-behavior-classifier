"""Tests for CentralWidget helpers that don't require instantiating the widget."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from jabs.classifier import MultiClassClassifier
from jabs.core.constants import MULTICLASS_NONE_BEHAVIOR
from jabs.core.enums import CrossValidationGroupingStrategy, ProjectDistanceUnit
from jabs.project.unusable_report import UnusableNotice

try:
    import jabs.ui.main_window.central_widget as central_widget_module
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


def _stub_widget(excluded: set[str], unusable: frozenset[str] = frozenset()) -> SimpleNamespace:
    """Minimal stand-in exposing what _included_counts reads from self."""
    return SimpleNamespace(
        _project=SimpleNamespace(
            settings_manager=SimpleNamespace(is_video_excluded=lambda v: v in excluded),
            is_trainable=lambda v: v not in excluded and v not in unusable,
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


def test_included_counts_drops_unusable_videos():
    """An unusable video is not trained on, so its labels do not count toward readiness."""
    counts = {"a.avi": {0: {}}, "b.avi": {0: {}}}
    result = CentralWidget._included_counts(_stub_widget(set(), frozenset({"b.avi"})), counts)
    assert set(result.keys()) == {"a.avi"}


def test_included_project_bout_totals_excludes_unusable_videos():
    """Bout totals for the training report leave out unusable videos."""
    counts = {
        "a.avi": {0: {"unfragmented_bout_counts": (3, 2)}},
        "b.avi": {0: {"unfragmented_bout_counts": (10, 10)}},  # unusable
    }
    stub = _stub_widget(set(), frozenset({"b.avi"}))
    stub._counts = counts
    assert CentralWidget._included_project_bout_totals(stub) == (3, 2)


def test_included_counts_no_exclusions_returns_all():
    """With nothing excluded, all videos are retained."""
    counts = {"a.avi": {0: {}}, "b.avi": {0: {}}}
    result = CentralWidget._included_counts(_stub_widget(set()), counts)
    assert set(result.keys()) == {"a.avi", "b.avi"}


def test_included_counts_none_returns_empty():
    """None counts (not yet computed) return an empty dict instead of raising."""
    assert CentralWidget._included_counts(_stub_widget(set()), None) == {}


def _train_button_stub(all_kfold: bool, kfold_value: int) -> SimpleNamespace:
    """Stand-in for a binary project whose videos all fall in one filename-pattern group."""
    return SimpleNamespace(
        _project=SimpleNamespace(
            settings_manager=SimpleNamespace(
                classifier_mode=ClassifierMode.BINARY,
                cv_grouping_strategy=CrossValidationGroupingStrategy.FILENAME_PATTERN,
                cv_grouping_regex=r"cage_(\d+)",
                is_video_excluded=lambda video: False,
            )
        ),
        _counts={
            "cage_1_day1.avi": {0: {"fragmented_frame_counts": (30, 30)}},
            "cage_1_day2.avi": {0: {"fragmented_frame_counts": (30, 30)}},
        },
        _controls=SimpleNamespace(
            all_kfold=all_kfold,
            kfold_value=kfold_value,
            train_button_enabled=None,
        ),
        _included_counts=lambda counts: counts,
        export_training_status_change=SimpleNamespace(emit=MagicMock()),
    )


def test_train_enabled_for_one_group_when_cross_validation_is_off():
    """k=0 trains without a held-out group, so one filename-pattern group is enough."""
    stub = _train_button_stub(all_kfold=False, kfold_value=0)

    CentralWidget.set_train_button_enabled_state(stub)

    assert stub._controls.train_button_enabled is True
    stub.export_training_status_change.emit.assert_called_once_with(True)


def test_train_disabled_for_one_group_when_cross_validation_is_requested():
    """One group cannot be split into train and test sets, so k=1 still blocks training."""
    stub = _train_button_stub(all_kfold=False, kfold_value=1)

    CentralWidget.set_train_button_enabled_state(stub)

    assert stub._controls.train_button_enabled is False


def test_train_disabled_for_one_group_when_all_kfold_is_checked():
    """The all-k-fold checkbox cross-validates over every group, so it needs two.

    The k slider is disabled (and may read zero) while the checkbox is checked, so
    the checkbox, not the slider, decides whether a CV split is required.
    """
    stub = _train_button_stub(all_kfold=True, kfold_value=0)

    CentralWidget.set_train_button_enabled_state(stub)

    assert stub._controls.train_button_enabled is False


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
    """Build a stub self for _classify_thread_complete with mocked collaborators.

    The classify cleanup clears the targets, as the real one does, so a handler that
    reads them after the cleanup sees ``None`` (every video) instead of its own targets.
    """
    stub = SimpleNamespace(
        _classification_targets=targets,
        _loaded_video=(
            SimpleNamespace(name=loaded_video_name) if loaded_video_name is not None else None
        ),
        _cleanup_progress_dialog=MagicMock(),
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
    stub._cleanup_classify_thread = MagicMock(
        side_effect=lambda: setattr(stub, "_classification_targets", None)
    )
    return stub


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
    behavior_settings=None,
) -> SimpleNamespace:
    """Build a stub self for the feature cache warning helpers.

    The settings resolvers are wired to their real implementations so the tests
    exercise the actual resolution chain rather than a mocked shortcut.
    """
    if mode is None:
        mode = ClassifierMode.BINARY
    stub = SimpleNamespace(
        _window_size=window_size,
        _classifier=classifier,
        _controls=SimpleNamespace(current_behavior=current_behavior, behaviors=list(behaviors)),
        _project=SimpleNamespace(
            settings_manager=SimpleNamespace(
                classifier_mode=mode,
                get_behavior=lambda _behavior: behavior_settings or {"window_size": window_size},
            ),
            get_project_defaults=lambda: project_defaults or {"window_size": 11},
            video_manager=SimpleNamespace(num_videos=4),
        ),
    )
    stub._feature_op_settings = lambda: CentralWidget._feature_op_settings(stub)
    stub._feature_cm_units = lambda: CentralWidget._feature_cm_units(stub)
    return stub


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
    """Binary mode uses the window size shown in the controls, not the saved settings."""
    stub = _feature_check_stub(window_size=7, behavior_settings={"window_size": 99})
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


@pytest.mark.parametrize(
    ("videos", "window_size", "action", "answer", "count_text"),
    [
        (["a.avi", "b.avi"], 5, "classification", True, "<b>2 videos</b>"),
        (["a.avi", "b.avi"], 5, "classification", False, "<b>2 videos</b>"),
        (["a.avi"], 30, "training", True, "<b>1 video</b>"),
        (["a.avi"], 5, "training", True, "<b>1 video</b>"),
    ],
    ids=["continue", "cancel", "window-size-in-init-hint", "singular-for-one-video"],
)
def test_confirm_on_demand_features_warning(
    monkeypatch: pytest.MonkeyPatch,
    videos: list[str],
    window_size: int,
    action: str,
    answer: bool,
    count_text: str,
) -> None:
    """The warning returns the user's choice and describes what is missing.

    The user's answer decides whether the run proceeds. The message names the window
    size, the video count (a single uncached video reads as "1 video", not "1 videos"),
    the action, and lists the videos, and it points at jabs-init with the window size
    that is missing.
    """
    confirm = MagicMock(return_value=answer)
    monkeypatch.setattr(
        "jabs.ui.main_window.central_widget.MessageDialog.confirm", confirm, raising=True
    )

    result = CentralWidget._confirm_on_demand_features(
        _feature_check_stub(), videos, window_size, action
    )

    assert result is answer
    message = confirm.call_args.kwargs["message"]
    assert f"<b>{window_size}</b>" in message
    assert count_text in message
    assert action in message
    assert "jabs-init" in message
    assert f"jabs-init -w {window_size}" in message
    assert "jabs-features" not in message
    for video in videos:
        assert video in confirm.call_args.kwargs["details"]


def _gating_stub(confirmed: bool, missing=("a.avi",), cm_units=False) -> SimpleNamespace:
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
        _feature_cm_units=MagicMock(return_value=cm_units),
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

    stub._project.videos_missing_window_features.assert_called_once_with(5, cm_units=False)
    stub._project.labeled_identities.assert_not_called()
    stub._confirm_on_demand_features.assert_not_called()
    assert stub._training_cache_targets == []


def test_training_feature_check_narrows_to_labeled_identities():
    """When something is missing, the check narrows to the labeled identities."""
    stub = _gating_stub(confirmed=True)

    assert CentralWidget._confirm_training_features(stub, 5) is True

    stub._project.labeled_identities.assert_called_once_with(["Walking"])
    assert stub._project.videos_missing_window_features.call_args_list[-1].kwargs == {
        "identities": {"a.avi": {0}},
        "cm_units": False,
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
    stub._project.videos_missing_window_features.assert_called_once_with(
        5, videos=["a.avi"], cm_units=False
    )
    assert stub._classification_targets is None


def _cleanup_stub() -> SimpleNamespace:
    """Build a stub self for the thread cleanup handlers."""
    return SimpleNamespace(
        _training_thread=None,
        _classify_thread=None,
        _classification_targets=None,
        _training_cache_targets=None,
        _project=MagicMock(),
        feature_cache_changed=SimpleNamespace(emit=MagicMock()),
    )


@pytest.mark.parametrize(
    ("cleanup_method", "targets_attr", "targets"),
    [
        ("_cleanup_training_thread", "_training_cache_targets", ["a.avi", "b.avi"]),
        ("_cleanup_training_thread", "_training_cache_targets", None),
        ("_cleanup_classify_thread", "_classification_targets", ["a.avi"]),
        ("_cleanup_classify_thread", "_classification_targets", None),
    ],
    ids=[
        "training-only-videos-it-read",
        "training-without-known-targets",
        "classify-single-video",
        "classify-all-videos",
    ],
)
def test_thread_cleanup_invalidates_only_the_videos_the_run_targeted(
    cleanup_method: str, targets_attr: str, targets: list[str] | None
) -> None:
    """Cleanup drops the cache status of the targeted videos, or of all when none are known.

    Training reads features only for labeled videos, and a single-video classification
    only for that video, so only those go stale. Targets of ``None`` (no run started,
    or every video classified) mean nothing is assumed to be current. The targets are
    consumed, so a canceled or failed run cannot leave stale targets behind for a later
    cleanup to act on.
    """
    stub = _cleanup_stub()
    setattr(stub, targets_attr, targets)

    getattr(CentralWidget, cleanup_method)(stub)

    stub._project.invalidate_feature_cache_status.assert_called_once_with(targets)
    stub.feature_cache_changed.emit.assert_called_once()
    assert getattr(stub, targets_attr) is None


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


def test_feature_op_settings_binary_uses_behavior_settings():
    """Binary mode reads the settings of the behavior being trained."""
    stub = _feature_check_stub(behavior_settings={"window_size": 7, "cm_units": True})

    assert CentralWidget._feature_op_settings(stub) == {"window_size": 7, "cm_units": True}


def test_feature_op_settings_multiclass_prefers_classifier_settings():
    """Multi-class mode uses the bundle the classifier was configured with."""
    classifier = MagicMock(spec=MultiClassClassifier)
    classifier.project_settings = {"window_size": 30, "cm_units": True}
    stub = _feature_check_stub(mode=ClassifierMode.MULTICLASS, classifier=classifier)

    assert CentralWidget._feature_op_settings(stub) == {"window_size": 30, "cm_units": True}


def test_feature_op_settings_multiclass_falls_back_to_defaults():
    """Without classifier settings, multi-class mode uses the project defaults."""
    classifier = MagicMock(spec=MultiClassClassifier)
    classifier.project_settings = None
    stub = _feature_check_stub(
        mode=ClassifierMode.MULTICLASS,
        classifier=classifier,
        project_defaults={"window_size": 11, "cm_units": False},
    )

    assert CentralWidget._feature_op_settings(stub) == {"window_size": 11, "cm_units": False}


@pytest.mark.parametrize(
    ("setting", "expected"),
    [
        (ProjectDistanceUnit.CM, True),
        (ProjectDistanceUnit.PIXEL, False),
        (True, True),
        (False, False),
        (None, False),
    ],
    ids=["enum-cm", "enum-pixel", "bool-true", "bool-false", "missing"],
)
def test_feature_cm_units_matches_the_extractor_truthiness(setting, expected):
    """The unit flag is read the way IdentityFeatures reads it.

    The stored value is a ProjectDistanceUnit (PIXEL is 0), and the extractor tests
    it for truthiness, so this must agree for both enum and bool values.
    """
    behavior_settings = {"window_size": 5}
    if setting is not None:
        behavior_settings["cm_units"] = setting
    stub = _feature_check_stub(behavior_settings=behavior_settings)

    assert CentralWidget._feature_cm_units(stub) is expected


def test_training_feature_check_passes_the_unit_setting(monkeypatch):
    """The gate tells the project which units the run will use."""
    stub = _gating_stub(confirmed=True, cm_units=True)

    CentralWidget._confirm_training_features(stub, 5)

    for call in stub._project.videos_missing_window_features.call_args_list:
        assert call.kwargs["cm_units"] is True


def _behavior_change_stub(mode, prediction_manager) -> SimpleNamespace:
    """Stand-in exposing what _on_behavior_changed() reads from self."""
    return SimpleNamespace(
        _project=SimpleNamespace(
            session_tracker=SimpleNamespace(behavior_selected=MagicMock()),
            settings_manager=SimpleNamespace(classifier_mode=mode, save_project_file=MagicMock()),
            prediction_manager=prediction_manager,
            counts=MagicMock(return_value={}),
        ),
        behavior="Grooming",
        _loaded_video=SimpleNamespace(name="clip.avi"),
        # Predictions left over from before a classifier-mode change.
        _predictions={0: "stale"},
        _probabilities={0: "stale"},
        _predictions_postprocessed={},
        _multiclass_class_names=None,
        _counts={},
        _update_controls_from_project_settings=MagicMock(),
        _load_cached_classifier=MagicMock(),
        _update_label_counts=MagicMock(),
        _set_label_track=MagicMock(),
        _update_label_button_color=MagicMock(),
        set_train_button_enabled_state=MagicMock(),
    )


def test_behavior_change_loads_binary_predictions_for_the_behavior():
    """Binary mode reloads the newly selected behavior's saved predictions."""
    manager = MagicMock()
    manager.load_predictions.return_value = ({0: "binary"}, {0: "prob"}, {})
    stub = _behavior_change_stub(ClassifierMode.BINARY, manager)

    CentralWidget._on_behavior_changed(stub)

    manager.load_predictions.assert_called_once_with("clip.avi", "Grooming")
    manager.load_multiclass_predictions.assert_not_called()
    assert stub._predictions == {0: "binary"}


def test_behavior_change_replaces_predictions_left_over_from_the_other_mode():
    """Switching an open project to multi-class must not keep the binary predictions.

    The mode change reaches this method, and stale binary 0/1 values read as
    multi-class color indices would mis-color the timeline, the label overlay and an
    exported video.
    """
    manager = MagicMock()
    manager.load_multiclass_predictions.return_value = ({}, {}, {}, None)
    stub = _behavior_change_stub(ClassifierMode.MULTICLASS, manager)

    CentralWidget._on_behavior_changed(stub)

    manager.load_multiclass_predictions.assert_called_once_with("clip.avi")
    manager.load_predictions.assert_not_called()
    assert stub._predictions == {}, "stale binary predictions were kept"


def _training_completion_stub(
    cv_warning: str | None, skipped: UnusableNotice | None = None
) -> SimpleNamespace:
    """Stand-in exposing what _training_thread_complete() reads from self."""
    return SimpleNamespace(
        _cleanup_training_thread=MagicMock(),
        _cleanup_progress_dialog=MagicMock(),
        status_message=SimpleNamespace(emit=MagicMock()),
        _set_classify_enabled=MagicMock(),
        _training_cv_warning=cv_warning,
        # no report markdown, so the report-dialog branch is skipped
        _training_report_markdown=None,
        _training_behaviors=lambda: ["Walk"],
        _project=SimpleNamespace(skipped_training_notice=MagicMock(return_value=skipped)),
    )


def test_training_completion_reports_labeled_videos_left_out(monkeypatch):
    """Labeled videos skipped as unusable are named in a dialog when training finishes."""
    warnings = []
    monkeypatch.setattr(
        "jabs.ui.main_window.central_widget.MessageDialog.warning",
        lambda *args, **kwargs: warnings.append((args, kwargs)),
    )
    notice = UnusableNotice("1 video with labels was left out", "b.avi: refused")
    stub = _training_completion_stub(None, notice)

    CentralWidget._training_thread_complete(stub, 1234)

    stub._project.skipped_training_notice.assert_called_once_with(["Walk"])
    assert len(warnings) == 1
    assert warnings[0][1]["message"] == notice.message
    assert warnings[0][1]["details"] == "b.avi: refused"


def test_training_completion_warns_when_cross_validation_was_skipped(monkeypatch):
    """A skipped-CV warning reaches the user as a dialog, not just a status message."""
    warnings = []
    monkeypatch.setattr(
        "jabs.ui.main_window.central_widget.MessageDialog.warning",
        lambda *args, **kwargs: warnings.append((args, kwargs)),
    )
    stub = _training_completion_stub("no group could serve as a test split")

    CentralWidget._training_thread_complete(stub, 1234)

    assert len(warnings) == 1
    _args, kwargs = warnings[0]
    assert kwargs["details"] == "no group could serve as a test split"
    # cleared so a later run does not repeat a stale warning
    assert stub._training_cv_warning is None


def test_training_completion_is_quiet_when_cross_validation_ran(monkeypatch):
    """No warning dialog when there was nothing to warn about."""
    warnings = []
    monkeypatch.setattr(
        "jabs.ui.main_window.central_widget.MessageDialog.warning",
        lambda *args, **kwargs: warnings.append((args, kwargs)),
    )

    CentralWidget._training_thread_complete(_training_completion_stub(None), 1234)

    assert warnings == []


def _search_stub(project, hit_file: str) -> SimpleNamespace:
    """Build a stub self for _on_search_hit_changed with one current search hit."""
    hit = SimpleNamespace(file=hit_file, start_frame=5, identity=None, behavior=None)
    return SimpleNamespace(
        _search_bar_widget=SimpleNamespace(current_search_hit=hit),
        _project=project,
        _loaded_video=None,
        load_video=MagicMock(),
        _player_widget=MagicMock(),
        search_hit_loaded=MagicMock(),
    )


def test_search_hit_in_an_unusable_video_shows_the_reason(monkeypatch):
    """A search hit in an unusable video is not loaded; the reason is shown instead."""
    dialog = MagicMock()
    monkeypatch.setattr(central_widget_module, "MessageDialog", dialog)
    project = MagicMock()
    project.is_usable.return_value = False
    project.unusable_videos = {"bad.avi": "shared tracklet"}
    stub = _search_stub(project, "bad.avi")

    CentralWidget._on_search_hit_changed(stub)

    stub.load_video.assert_not_called()
    stub.search_hit_loaded.emit.assert_not_called()
    assert dialog.warning.call_args.kwargs["details"] == "shared tracklet"


def test_search_hit_whose_pose_fails_to_open_is_reported(monkeypatch):
    """A PoseOpenError loading a search hit's video is reported, not raised from the slot."""
    dialog = MagicMock()
    monkeypatch.setattr(central_widget_module, "MessageDialog", dialog)
    project = MagicMock()
    project.is_usable.return_value = True
    stub = _search_stub(project, "a.avi")
    stub.load_video.side_effect = central_widget_module.PoseOpenError(
        Path("a_pose_est_v8.h5"), "disk full"
    )

    CentralWidget._on_search_hit_changed(stub)

    assert dialog.error.call_args.kwargs["details"] == "disk full"
    stub._player_widget.seek_to_frame.assert_not_called()


def _stale_classifier_stub() -> SimpleNamespace:
    """Stand-in for _load_cached_classifier whose saved classifier must be retrained."""
    error = central_widget_module.ClassifierFeatureVersionError(
        "trained with feature version 19; retrain the classifier", 19, 20
    )
    stub = SimpleNamespace(
        _project=SimpleNamespace(
            load_classifier=MagicMock(side_effect=error),
            settings_manager=SimpleNamespace(classifier_mode=ClassifierMode.BINARY),
        ),
        _classifier=SimpleNamespace(classifier_name="Random Forest"),
        behavior="Walk",
        _unsupported_classifier_warned=set(),
        _ensure_classifier_for_mode=MagicMock(),
        _reset_classifier=MagicMock(),
        _update_classifier_controls=MagicMock(),
        _set_classify_enabled=MagicMock(),
    )
    stub._claim_saved_classifier_warning = lambda: CentralWidget._claim_saved_classifier_warning(
        stub
    )
    stub._warn_stale_classifier = lambda e: CentralWidget._warn_stale_classifier(stub, e)
    return stub


def test_stale_classifier_is_reset_and_the_user_told_to_retrain(monkeypatch):
    """A classifier from another feature version is dropped and a retrain dialog shown once."""
    dialog = MagicMock()
    monkeypatch.setattr(central_widget_module, "MessageDialog", dialog)
    monkeypatch.setattr(
        central_widget_module.QtCore.QTimer, "singleShot", lambda _ms, _ctx, fn: fn()
    )
    stub = _stale_classifier_stub()

    CentralWidget._load_cached_classifier(stub)
    CentralWidget._load_cached_classifier(stub)

    stub._reset_classifier.assert_called()
    stub._set_classify_enabled.assert_called_with(False)
    dialog.warning.assert_called_once()
    assert "retrain" in dialog.warning.call_args.args[2]


def test_set_project_drops_the_previous_projects_video():
    """Opening a project unloads the previous project's video, pose and selection state.

    When none of the new project's videos is usable, no video is loaded to replace it,
    and its leftover pose and enabled select button let the label keys act on labels
    that are gone.
    """
    stub = SimpleNamespace(
        _project=None,
        _labels=MagicMock(),
        _loaded_video=Path("old.avi"),
        _pose_est=MagicMock(num_identities=2),
        _controls=MagicMock(select_button_enabled=True),
        _player_widget=MagicMock(),
        _jabs_timeline=MagicMock(),
        _search_bar_widget=MagicMock(),
        _update_timeline_search_results=MagicMock(),
    )
    stub._set_identities = lambda identities: CentralWidget._set_identities(stub, identities)
    stub._start_selection = lambda pressed: CentralWidget._start_selection(stub, pressed)
    stub._update_select_button_state = lambda: CentralWidget._update_select_button_state(stub)

    CentralWidget.set_project(stub, MagicMock())

    assert stub._labels is None
    assert stub._loaded_video is None
    assert stub._pose_est is None
    stub._player_widget.reset.assert_called_once()
    stub._controls.set_identities.assert_called_with([])
    assert stub._controls.select_button_enabled is False


def _load_video_stub(previous_pose: MagicMock, events: list[str]) -> MagicMock:
    """A CentralWidget stand-in for load_video, recording player loads and pose closes."""
    stub = MagicMock()
    stub._pose_est = previous_pose
    stub._labels = None
    stub._loaded_video = Path("old.avi")
    stub._frame_count_mismatch_message = lambda *_args: None
    stub._player_widget.load_video.side_effect = lambda _path, pose, _labels: events.append(
        f"player:{pose.name}"
    )
    previous_pose.close.side_effect = lambda: events.append("close:old")
    return stub


def test_load_video_closes_the_previous_pose_after_the_new_one_is_shown(monkeypatch):
    """The replaced pose is closed (its file released) once the player holds the new one."""
    events: list[str] = []
    old_pose, new_pose = MagicMock(), MagicMock()
    old_pose.name, new_pose.name = "old", "new"
    stub = _load_video_stub(old_pose, events)
    stub._project.load_pose_est.return_value = new_pose
    monkeypatch.setattr(
        central_widget_module.central_widget_mode,
        "load_video_predictions",
        lambda *_args: ({}, {}, {}, None),
    )

    CentralWidget.load_video(stub, Path("new.avi"))

    assert stub._pose_est is new_pose
    assert events == ["player:new", "close:old"]
    new_pose.close.assert_not_called()


def test_load_video_closes_the_previous_pose_when_the_new_one_fails():
    """A failed load drops the previous pose from the player and closes it."""
    events: list[str] = []
    old_pose = MagicMock()
    stub = _load_video_stub(old_pose, events)
    stub._project.load_pose_est.side_effect = central_widget_module.PoseOpenError(
        Path("new_pose_est_v6.h5"), "cannot read"
    )

    with pytest.raises(central_widget_module.PoseOpenError):
        CentralWidget.load_video(stub, Path("new.avi"))

    assert stub._pose_est is None
    stub._player_widget.reset.assert_called_once()
    old_pose.close.assert_called_once()


def test_set_project_closes_the_previous_projects_pose():
    """The pose set_project drops is closed after the player lets go of it."""
    pose = MagicMock()
    player = MagicMock()
    player.reset.side_effect = lambda: pose.close.assert_not_called()
    stub = SimpleNamespace(
        _project=None,
        _labels=None,
        _loaded_video=None,
        _pose_est=pose,
        _controls=MagicMock(select_button_enabled=False),
        _player_widget=player,
        _jabs_timeline=MagicMock(),
        _search_bar_widget=MagicMock(),
        _update_timeline_search_results=MagicMock(),
    )
    stub._set_identities = lambda identities: CentralWidget._set_identities(stub, identities)
    stub._start_selection = lambda pressed: CentralWidget._start_selection(stub, pressed)
    stub._update_select_button_state = lambda: CentralWidget._update_select_button_state(stub)

    CentralWidget.set_project(stub, MagicMock())

    pose.close.assert_called_once()
