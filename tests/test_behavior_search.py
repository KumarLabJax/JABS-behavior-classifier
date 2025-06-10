from unittest.mock import MagicMock

import numpy as np
import pytest

from jabs.behavior_search.behavior_search_util import (
    LabelBehaviorSearchQuery,
    PredictionBehaviorSearchQuery,
    PredictionSearchKind,
    SearchHit,
    _search_behaviors_gen,
)


def _make_project(
    videos=None,
    annotations=None,
    predictions=None,
    probabilities=None,
    project_settings=None,
):
    """Helper to build a dummy project with video_manager and prediction_manager"""
    # VideoManager mock
    video_manager = MagicMock()
    video_manager.videos = videos or []
    video_manager.load_annotations.side_effect = (
        lambda video: annotations.get(video, None) if annotations else None
    )

    # PredictionManager mock
    prediction_manager = MagicMock()

    def load_predictions(video, behavior):
        preds = predictions.get((video, behavior), {}) if predictions else {}
        probs = probabilities.get((video, behavior), {}) if probabilities else {}
        return preds, probs, None

    prediction_manager.load_predictions.side_effect = load_predictions

    # SettingsManager mock
    settings_manager = MagicMock()
    settings_manager.project_settings = project_settings or {}

    # Project mock
    project = MagicMock()
    project.video_manager = video_manager
    project.prediction_manager = prediction_manager
    project.settings_manager = settings_manager
    return project


def _correct_probs(preds, probs):
    """Move 0.0 - 1.0 probaabilities into 0.5 - 1.0 range based on predictions."""
    probs = probs.copy()
    probs[preds == 0] = 1.0 - probs[preds == 0]

    return probs


# --- LabelBehaviorSearchQuery tests ---


def test_label_query_positive_match():
    """Test positive label query matches correct blocks."""
    annotations = {
        "video1": {
            "labels": {
                "1": {
                    "foo": [
                        {"start": 0, "end": 10, "present": True},
                        {"start": 11, "end": 20, "present": False},
                    ]
                }
            }
        }
    }
    project = _make_project(videos=["video1"], annotations=annotations)
    query = LabelBehaviorSearchQuery(behavior_label="foo", positive=True)
    hits = list(_search_behaviors_gen(project, query))
    assert hits == [
        SearchHit(file="video1", identity="1", behavior="foo", start_frame=0, end_frame=10)
    ]


def test_label_query_negative_match():
    """Test negative label query matches correct blocks."""
    annotations = {
        "video1": {
            "labels": {
                "1": {
                    "foo": [
                        {"start": 0, "end": 10, "present": True},
                        {"start": 11, "end": 20, "present": False},
                    ]
                }
            }
        }
    }
    project = _make_project(videos=["video1"], annotations=annotations)
    query = LabelBehaviorSearchQuery(behavior_label="foo", negative=True)
    hits = list(_search_behaviors_gen(project, query))
    assert hits == [
        SearchHit(file="video1", identity="1", behavior="foo", start_frame=11, end_frame=20)
    ]


def test_label_query_behavior_label_none():
    """Test label query with behavior_label=None matches all behaviors."""
    annotations = {
        "video1": {
            "labels": {
                "1": {
                    "foo": [
                        {"start": 0, "end": 10, "present": True},
                    ],
                    "bar": [
                        {"start": 5, "end": 15, "present": True},
                    ],
                }
            }
        }
    }
    project = _make_project(videos=["video1"], annotations=annotations)
    query = LabelBehaviorSearchQuery(behavior_label=None, positive=True)
    hits = list(_search_behaviors_gen(project, query))
    assert len(hits) == 2
    assert {h.behavior for h in hits} == {"foo", "bar"}


def test_label_query_empty_annotations():
    """Test label query with empty annotations yields nothing."""
    project = _make_project(videos=["video1"], annotations={"video1": {}})
    query = LabelBehaviorSearchQuery(behavior_label="foo", positive=True)
    hits = list(_search_behaviors_gen(project, query))
    assert hits == []


def test_label_query_project_none():
    """Test label query with project=None yields nothing."""
    query = LabelBehaviorSearchQuery(behavior_label="foo", positive=True)
    hits = list(_search_behaviors_gen(None, query))
    assert hits == []


def test_label_query_query_none():
    """Test label query with search_query=None yields nothing."""
    project = _make_project(videos=["video1"])
    hits = list(_search_behaviors_gen(project, None))
    assert hits == []


def test_label_query_prefers_unfragmented_labels():
    """Test that unfragmented_labels are preferred over labels if present."""
    annotations = {
        "video1": {
            "labels": {
                "1": {
                    "foo": [
                        {"start": 0, "end": 10, "present": True},  # Should be ignored
                    ]
                }
            },
            "unfragmented_labels": {
                "1": {
                    "foo": [
                        {"start": 20, "end": 30, "present": True},  # Should be used
                    ]
                }
            },
        }
    }
    project = _make_project(videos=["video1"], annotations=annotations)
    query = LabelBehaviorSearchQuery(behavior_label="foo", positive=True)
    hits = list(_search_behaviors_gen(project, query))
    # Only the unfragmented_labels block should be returned
    assert hits == [
        SearchHit(file="video1", identity="1", behavior="foo", start_frame=20, end_frame=30)
    ]


# --- PredictionBehaviorSearchQuery tests ---


def test_prediction_positive_prediction():
    """Test positive prediction search yields correct intervals."""
    preds = {"1": np.array([1, 1, 0, 1, 1])}
    probs = {"1": _correct_probs(preds["1"], np.array([0.9, 0.8, 0.2, 0.95, 0.99]))}
    predictions = {("video1", "foo"): preds}
    probabilities = {("video1", "foo"): probs}
    project_settings = {"behavior": {"foo": {}}}
    project = _make_project(
        videos=["video1"],
        predictions=predictions,
        probabilities=probabilities,
        project_settings=project_settings,
    )
    query = PredictionBehaviorSearchQuery(
        search_kind=PredictionSearchKind.POSITIVE_PREDICTION,
        behavior_label="foo",
    )
    hits = list(_search_behaviors_gen(project, query))
    # Should yield intervals for [0,1] and [3,4]
    assert hits == [
        SearchHit(file="video1", identity="1", behavior="foo", start_frame=0, end_frame=1),
        SearchHit(file="video1", identity="1", behavior="foo", start_frame=3, end_frame=4),
    ]


def test_prediction_negative_prediction():
    """Test negative prediction search yields correct intervals."""
    preds = {"1": np.array([0, 0, 1, 0, 0])}
    probs = {"1": _correct_probs(preds["1"], np.array([0.1, 0.2, 0.8, 0.05, 0.01]))}
    predictions = {("video1", "foo"): preds}
    probabilities = {("video1", "foo"): probs}
    project_settings = {"behavior": {"foo": {}}}
    project = _make_project(
        videos=["video1"],
        predictions=predictions,
        probabilities=probabilities,
        project_settings=project_settings,
    )
    query = PredictionBehaviorSearchQuery(
        search_kind=PredictionSearchKind.NEGATIVE_PREDICTION,
        behavior_label="foo",
    )
    hits = list(_search_behaviors_gen(project, query))
    # Should yield intervals for [0,1] and [3,4]
    assert hits == [
        SearchHit(file="video1", identity="1", behavior="foo", start_frame=0, end_frame=1),
        SearchHit(file="video1", identity="1", behavior="foo", start_frame=3, end_frame=4),
    ]


def test_prediction_probability_range():
    """Test probability range search yields correct intervals."""
    preds = {"1": np.array([1, 1, 0, 1, 0])}
    probs = {"1": _correct_probs(preds["1"], np.array([0.6, 0.7, 0.4, 0.95, 0.2]))}
    predictions = {("video1", "foo"): preds}
    probabilities = {("video1", "foo"): probs}
    project_settings = {"behavior": {"foo": {}}}
    project = _make_project(
        videos=["video1"],
        predictions=predictions,
        probabilities=probabilities,
        project_settings=project_settings,
    )
    query = PredictionBehaviorSearchQuery(
        search_kind=PredictionSearchKind.PROBABILITY_RANGE,
        behavior_label="foo",
        prob_greater_value=0.5,
    )
    hits = list(_search_behaviors_gen(project, query))
    # Should yield intervals for [0,1] and [3,3]
    assert hits == [
        SearchHit(file="video1", identity="1", behavior="foo", start_frame=0, end_frame=1),
        SearchHit(file="video1", identity="1", behavior="foo", start_frame=3, end_frame=3),
    ]


def test_prediction_min_contiguous_frames():
    """Test min_contiguous_frames filters out short intervals."""
    preds = {"1": np.array([1, 1, 0, 1, 1])}
    probs = {"1": _correct_probs(preds["1"], np.array([0.9, 0.8, 0.2, 0.95, 0.99]))}
    predictions = {("video1", "foo"): preds}
    probabilities = {("video1", "foo"): probs}
    project_settings = {"behavior": {"foo": {}}}
    project = _make_project(
        videos=["video1"],
        predictions=predictions,
        probabilities=probabilities,
        project_settings=project_settings,
    )
    query = PredictionBehaviorSearchQuery(
        search_kind=PredictionSearchKind.POSITIVE_PREDICTION,
        behavior_label="foo",
        min_contiguous_frames=2,
    )
    hits = list(_search_behaviors_gen(project, query))
    # Only [0,1] and [3,4], both length 2, so both should be included
    assert hits == [
        SearchHit(file="video1", identity="1", behavior="foo", start_frame=0, end_frame=1),
        SearchHit(file="video1", identity="1", behavior="foo", start_frame=3, end_frame=4),
    ]
    # Now require 3 frames, should yield nothing
    query2 = PredictionBehaviorSearchQuery(
        search_kind=PredictionSearchKind.POSITIVE_PREDICTION,
        behavior_label="foo",
        min_contiguous_frames=3,
    )
    hits2 = list(_search_behaviors_gen(project, query2))
    assert hits2 == []


def test_prediction_size_mismatch():
    """Test prediction/probabilities size mismatch skips animal."""
    preds = {"1": np.array([1, 1, 0])}
    probs = {"1": np.array([0.9, 0.8])}
    predictions = {("video1", "foo"): preds}
    probabilities = {("video1", "foo"): probs}
    project_settings = {"behavior": {"foo": {}}}
    project = _make_project(
        videos=["video1"],
        predictions=predictions,
        probabilities=probabilities,
        project_settings=project_settings,
    )
    query = PredictionBehaviorSearchQuery(
        search_kind=PredictionSearchKind.POSITIVE_PREDICTION,
        behavior_label="foo",
    )
    hits = list(_search_behaviors_gen(project, query))
    assert hits == []


def test_prediction_empty_predictions():
    """Test empty predictions/probabilities yields nothing."""
    preds = {"1": np.array([])}
    probs = {"1": np.array([])}
    predictions = {("video1", "foo"): preds}
    probabilities = {("video1", "foo"): probs}
    project_settings = {"behavior": {"foo": {}}}
    project = _make_project(
        videos=["video1"],
        predictions=predictions,
        probabilities=probabilities,
        project_settings=project_settings,
    )
    query = PredictionBehaviorSearchQuery(
        search_kind=PredictionSearchKind.POSITIVE_PREDICTION,
        behavior_label="foo",
    )
    hits = list(_search_behaviors_gen(project, query))
    assert hits == []


def test_prediction_behavior_label_none():
    """Test prediction query with behavior_label=None searches all behaviors."""
    preds = {"1": np.array([1, 1, 0])}
    probs = {"1": _correct_probs(preds["1"], np.array([0.9, 0.8, 0.2]))}
    # probs = {"1": np.array([0.9, 0.8, 0.2])}
    predictions = {("video1", "foo"): preds, ("video1", "bar"): preds}
    probabilities = {("video1", "foo"): probs, ("video1", "bar"): probs}
    project_settings = {"behavior": {"foo": {}, "bar": {}}}
    project = _make_project(
        videos=["video1"],
        predictions=predictions,
        probabilities=probabilities,
        project_settings=project_settings,
    )
    query = PredictionBehaviorSearchQuery(
        search_kind=PredictionSearchKind.POSITIVE_PREDICTION,
        behavior_label=None,
    )
    hits = list(_search_behaviors_gen(project, query))
    # Should yield intervals for both behaviors
    assert len(hits) == 2
    assert {h.behavior for h in hits} == {"foo", "bar"}


def test_unknown_query_type():
    """Test unknown query type raises ValueError."""

    class DummyQuery:
        pass

    project = _make_project(videos=["video1"])
    with pytest.raises(ValueError):
        list(_search_behaviors_gen(project, DummyQuery()))
