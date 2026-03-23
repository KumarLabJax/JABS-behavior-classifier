"""Tests for the public metrics SDK surface."""

from __future__ import annotations

from jabs.vision.timm import metrics
from jabs.vision.timm.metrics import evaluation, matching, pck, ranking, reporting, similarity


class TestPublicApi:
    """Smoke tests for the intended public import surface."""

    def test_top_level_exports(self) -> None:
        """Top-level package re-exports the main SDK entrypoints."""
        assert metrics.evaluate_pose is evaluation.evaluate_pose
        assert metrics.evaluate_detection is evaluation.evaluate_detection
        assert metrics.evaluate_pck is evaluation.evaluate_pck
        assert metrics.format_results is reporting.format_results
        assert metrics.compute_oks is similarity.compute_oks
        assert metrics.greedy_match is matching.greedy_match
        assert metrics.compute_pck is pck.compute_pck

    def test_public_submodules_import(self) -> None:
        """Public submodules expose the developer-facing SDK."""
        assert callable(similarity.compute_bbox_iou)
        assert callable(matching.match_image)
        assert callable(ranking.compute_ap_ar)
        assert callable(reporting.format_results)
        assert callable(pck.bbox_diagonal)
