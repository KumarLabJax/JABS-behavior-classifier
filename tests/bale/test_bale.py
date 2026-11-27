"""Unit and integration tests for `jabs.bale`."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import mahalanobis
from sklearn.decomposition import PCA

from jabs.bale import (
    BaleUndoStack,
    apply_bout_edit,
    clear_bout_extractor_cache,
    compute_fast_bout_quantiles,
    compute_within_class_consensus_outliers,
    evaluate_cv_group_impact,
    export_bale_summary_figure,
    find_dual_class_pairing_candidate,
    recompute_embedding_only,
    run_bale,
)
from jabs.behavior_search import CustomBoutListSearchQuery, SearchHit, search_behaviors
from jabs.core.enums import ClassifierMode


def _write_synthetic_parquet_cache(
    identity_dir: Path,
    df: pd.DataFrame,
    *,
    identity: int = 0,
    feature_version: int | None = None,
    pose_hash: str = "synth_pose_hash",
    distance_scale_factor: float | None = None,
) -> None:
    """Write `per_frame.parquet` + matching `metadata.json` via `ParquetFeatureCacheWriter`."""
    from jabs.core.types import FeatureCacheMetadata, PerFrameCacheData
    from jabs.feature_extraction import FEATURE_VERSION
    from jabs.io.feature_cache.parquet import ParquetFeatureCacheWriter

    n_rows = len(df)
    if "_jabs_frame_valid" in df.columns:
        valid_arr = np.asarray(df["_jabs_frame_valid"], dtype=np.uint8)
    else:
        valid_arr = np.ones(n_rows, dtype=np.uint8)
    feats = {
        str(col): np.asarray(df[col], dtype=np.float32)
        for col in df.columns
        if col != "_jabs_frame_valid"
    }
    meta = FeatureCacheMetadata(
        feature_version=int(FEATURE_VERSION if feature_version is None else feature_version),
        identity=int(identity),
        num_frames=n_rows,
        pose_hash=pose_hash,
        distance_scale_factor=distance_scale_factor,
        avg_wall_length=None,
        cached_window_sizes=frozenset(),
    )
    ParquetFeatureCacheWriter().write_per_frame(
        identity_dir,
        meta,
        PerFrameCacheData(frame_valid=valid_arr, features=feats),
    )


def test_fast_bout_quantiles_accuracy() -> None:
    """Verify `compute_fast_bout_quantiles` matches reference order statistics and half_delta."""
    rng = np.random.default_rng(123)
    X = rng.normal(loc=5.0, scale=2.0, size=(40, 6)).astype(np.float32)
    valid_mask = np.ones(40, dtype=bool)
    valid_mask[3] = False  # 1 dropped pose frame

    vec, valid_frac = compute_fast_bout_quantiles(X, valid_mask)
    assert np.isclose(valid_frac, 39.0 / 40.0)
    # 7 stats * 6 features + 2 meta = 44 dimensions
    assert vec.shape == (44,)
    assert np.isclose(vec[-2], np.log2(41.0))
    assert np.isclose(vec[-1], 39.0 / 40.0)

    X_valid = X[valid_mask]
    ref_med = np.median(X_valid, axis=0)
    ref_mean = np.mean(X_valid, axis=0)
    assert np.allclose(vec[0:6], ref_med, atol=1e-4)
    assert np.allclose(vec[24:30], ref_mean, atol=1e-5)


def test_whitened_pca_mahalanobis_exactness() -> None:
    """Verify whitened-PCA Mahalanobis identity matches `scipy.spatial.distance.mahalanobis` (< 1e-11)."""
    rng = np.random.default_rng(42)
    # Full-rank matrix (N=50 > D=8)
    X = rng.normal(size=(50, 8)).astype(np.float64)
    mu = np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False, ddof=1)
    inv_cov = np.linalg.inv(cov)

    scipy_dists = np.array([mahalanobis(row, mu, inv_cov) for row in X])

    pca = PCA(n_components=8, random_state=42)
    scores = pca.fit_transform(X)
    eigvals = pca.explained_variance_  # uses ddof=1
    whitened_dists = np.sqrt(np.sum((scores**2) / eigvals[None, :], axis=1))

    max_abs_err = float(np.max(np.abs(scipy_dists - whitened_dists)))
    assert max_abs_err < 1e-11


def test_within_class_consensus_outliers_detects_spike() -> None:
    """Verify within-class 5-metric consensus outlier detector flags an extreme bout."""
    rng = np.random.default_rng(7)
    X_inliers = rng.normal(0.0, 1.0, size=(39, 12)).astype(np.float32)
    X_outlier = np.full((1, 12), 18.0, dtype=np.float32)
    X = np.vstack([X_inliers, X_outlier])
    classes = ["Grooming"] * 40
    sources = ["labeled"] * 40

    scores, votes, lof, iso, ood = compute_within_class_consensus_outliers(X, classes, sources)
    assert scores.shape == (40,)
    assert int(np.argmax(scores)) == 39
    assert int(votes[39]) >= 4


def test_custom_bout_list_search_query_preserves_rank_order() -> None:
    """Verify `CustomBoutListSearchQuery` returns hits in exact ranked order without sorting by filename."""
    hits = (
        SearchHit(
            file="vid_z.mp4", identity="1", behavior="Grooming", start_frame=200, end_frame=240
        ),
        SearchHit(
            file="vid_a.mp4", identity="0", behavior="Grooming", start_frame=10, end_frame=50
        ),
        SearchHit(
            file="vid_m.mp4", identity="2", behavior="Grooming", start_frame=80, end_frame=110
        ),
    )
    query = CustomBoutListSearchQuery(
        hits=hits,
        description="Ranked Suspect Queue",
        preserve_rank_order=True,
    )
    res = search_behaviors(project=None, search_query=query)
    assert [h.file for h in res] == ["vid_z.mp4", "vid_a.mp4", "vid_m.mp4"]


def _build_synthetic_explorer_project(tmp_path: Path) -> tuple[MagicMock, Path]:
    """Build a synthetic JABS project directory with Parquet features, annotations, and predictions."""
    clear_bout_extractor_cache()
    proj_dir = tmp_path / "synth_jabs_project"
    anno_dir = proj_dir / "jabs" / "annotations"
    feat_dir = proj_dir / "jabs" / "features"
    pred_dir = proj_dir / "jabs" / "predictions"
    anno_dir.mkdir(parents=True)
    feat_dir.mkdir(parents=True)
    pred_dir.mkdir(parents=True)

    rng = np.random.default_rng(2026)
    n_frames = 600
    feat_cols = [
        "nose_speed",
        "front_paws_dist",
        "body_area",
        "spine_curvature",
        "ear_dist",
        "tail_base_speed",
    ]

    videos = [f"cage_{i:02d}.mp4" for i in range(6)]
    counts_by_video: dict[str, dict[int, dict[str, tuple[int, int]]]] = {}

    for v_idx, video_name in enumerate(videos):
        stem = Path(video_name).stem
        unfrag_labels: dict[str, dict[str, list[dict[str, object]]]] = {}
        counts_by_video[video_name] = {}

        for identity in (0, 1, 2):
            # Create synthetic per_frame.parquet
            X_base = rng.normal(0.0, 0.5, size=(n_frames, len(feat_cols))).astype(np.float32)

            # Positive bout: [50, 90] -> high nose_speed & front_paws_dist (~ +4.0)
            X_base[50:91, :3] += 4.5
            # Negative bout: [200, 240] -> low kinematics (~ -4.0)
            X_base[200:241, :3] -= 4.5

            # On video 0, identity 0: add a deliberately MISLABELED bout at [320, 360]
            # (kinematics are strongly Negative -4.5, but labeled present=True!)
            blocks = [
                {"start": 50, "end": 90, "present": True},
                {"start": 200, "end": 240, "present": False},
            ]
            if v_idx == 0 and identity == 0:
                X_base[320:361, :3] -= 5.0
                blocks.append({"start": 320, "end": 360, "present": True})

            # On video 1, identity 0: add a CHIMERIC bout at [400, 460]
            # (first half 400..429 is Negative -4.5, second half 430..460 is Positive +4.5, labeled present=True)
            if v_idx == 1 and identity == 0:
                X_base[400:430, :3] -= 4.8
                X_base[430:461, :3] += 4.8
                blocks.append({"start": 400, "end": 460, "present": True})

            df_dict = {col: X_base[:, j] for j, col in enumerate(feat_cols)}
            df_dict["_jabs_frame_valid"] = np.ones(n_frames, dtype=np.uint8)
            _write_synthetic_parquet_cache(
                feat_dir / stem / str(identity),
                pd.DataFrame(df_dict),
                identity=identity,
            )

            unfrag_labels[str(identity)] = {"Grooming": blocks}
            pos_f = sum(int(b["end"]) - int(b["start"]) + 1 for b in blocks if b["present"])
            neg_f = sum(int(b["end"]) - int(b["start"]) + 1 for b in blocks if not b["present"])
            pos_b = sum(1 for b in blocks if b["present"])
            neg_b = sum(1 for b in blocks if not b["present"])
            counts_by_video[video_name][identity] = {
                "fragmented_frame_counts": (pos_f, neg_f),
                "fragmented_bout_counts": (pos_b, neg_b),
                "unfragmented_frame_counts": (pos_f, neg_f),
                "unfragmented_bout_counts": (pos_b, neg_b),
            }

        anno_doc = {
            "version": 1,
            "file": video_name,
            "num_frames": n_frames,
            "labels": unfrag_labels,
            "unfragmented_labels": unfrag_labels,
        }
        (anno_dir / f"{stem}.json").write_text(json.dumps(anno_doc, indent=2))

    # Build a lightweight mock Project wrapping proj_dir
    project = MagicMock()
    project.dir = proj_dir
    project.project_paths.project_dir = proj_dir
    project.project_paths.annotations_dir = anno_dir
    project.project_paths.feature_dir = feat_dir
    project.project_paths.prediction_dir = pred_dir
    project.settings_manager.classifier_mode = ClassifierMode.BINARY
    project.settings_manager.behavior_names = ["Grooming"]
    project.settings_manager.is_video_excluded.return_value = False
    project.settings_manager.get_behavior.return_value = {
        "social": True,
        "segmentation": True,
        "window_size": 30,
    }
    from jabs.project import VideoLabels

    pose_mock = MagicMock()
    pose_mock.num_frames = n_frames
    pose_mock.external_identities = None
    pose_mock.identity_mask.return_value = np.ones(n_frames, dtype=bool)
    project.load_pose_est.return_value = pose_mock

    def _mock_load_video_labels(v_name: str, _pose: object = None) -> VideoLabels:
        stem = Path(v_name).stem
        doc = json.loads((anno_dir / f"{stem}.json").read_text())
        return VideoLabels.load(doc)

    def _mock_save_annotations(v_labels: VideoLabels, _pose: object = None) -> None:
        stem = Path(v_labels.filename).stem
        out_doc = v_labels.as_dict(pose_mock)
        (anno_dir / f"{stem}.json").write_text(json.dumps(out_doc, indent=2))

    project.video_manager.videos = videos
    project.video_manager.load_video_labels.side_effect = _mock_load_video_labels
    project.save_annotations.side_effect = _mock_save_annotations
    project.counts.return_value = counts_by_video
    return project, proj_dir


def test_end_to_end_explorer_detects_mislabel_and_chimeric_bout(tmp_path: Path) -> None:
    """Verify `run_bale` ranks the intentional mislabel #1 in confusion and detects chimeric slop."""
    project, proj_dir = _build_synthetic_explorer_project(tmp_path)
    report = run_bale(project, "Grooming", space="classifier", compute_tsne=True)

    # 6 videos * 3 mice * 2 bouts + 1 mislabel + 1 chimeric = 38 bouts
    assert len(report.feature_matrix.records) == 38
    assert report.cv_groups_count == 18
    assert report.dual_class_cv_met is True
    assert report.jabs_threshold_met is True

    records = report.feature_matrix.records
    diag = report.diagnostics

    # Locate our intentional mislabeled bout (cage_00.mp4, id=0, 320..360)
    mislabel_idx = next(
        i
        for i, r in enumerate(records)
        if r.video == "cage_00.mp4" and r.identity == 0 and r.start_frame == 320
    )
    # Its confusion score should be high (>= 0.65) and top-ranked among clean bouts
    assert float(diag.confusion_scores[mislabel_idx]) >= 0.65
    assert int(np.argmax(diag.confusion_scores)) == mislabel_idx

    # Locate our intentional chimeric bout (cage_01.mp4, id=0, 400..460)
    chimeric_idx = next(
        i
        for i, r in enumerate(records)
        if r.video == "cage_01.mp4" and r.identity == 0 and r.start_frame == 400
    )
    assert float(diag.transition_scores[chimeric_idx]) >= 0.45
    assert diag.suggested_trim_bounds[chimeric_idx] is not None

    # Test warm recompute (< 150 ms)
    warm_report = recompute_embedding_only(report, space="unsupervised", forced_k=3)
    assert warm_report.embedding.optimal_k == 3
    assert warm_report.timings_ms["warm_recompute_ms"] < 500.0

    # Test PNG + editable SVG export (`svg.fonttype = 'none'`)
    out_png = tmp_path / "summary.png"
    out_svg = tmp_path / "summary.svg"
    export_bale_summary_figure(report, out_png, out_svg)
    assert out_png.exists() and out_png.stat().st_size > 10_000
    assert out_svg.exists() and "<text" in out_svg.read_text()


def test_cv_group_guard_warns_when_dropping_below_dual_class_minimum(tmp_path: Path) -> None:
    """Verify `evaluate_cv_group_impact` flags an edit that drops dual-class tracks below 15 (`Dual-Class CV rule`)."""
    project, _ = _build_synthetic_explorer_project(tmp_path)
    # Keep only 15 dual-class tracks in counts
    raw_counts = project.counts("Grooming")
    trimmed_counts: dict[str, dict[int, dict[str, tuple[int, int]]]] = {}
    kept = 0
    for v, id_map in raw_counts.items():
        trimmed_counts[v] = {}
        for ident, cdict in id_map.items():
            if kept < 15:
                trimmed_counts[v][ident] = cdict
                kept += 1
    project.counts.return_value = trimmed_counts

    impact = evaluate_cv_group_impact(
        project,
        "Grooming",
        video="cage_00.mp4",
        identity=0,
        action="delete",
        start_frame=200,
        end_frame=240,
        current_is_positive=False,
    )
    assert impact.before_groups == 15
    assert impact.after_groups == 14
    assert impact.dual_class_cv_met_before is True
    assert impact.dual_class_cv_met_after is False
    assert impact.safe_to_apply is False
    assert impact.warning_message is not None and "Dual-Class CV rule" in impact.warning_message


def test_apply_bout_edit_and_undo_stack(tmp_path: Path) -> None:
    """Verify `apply_bout_edit` (relabel, trim, split, promote_pred_pos) and `BaleUndoStack`."""
    project, proj_dir = _build_synthetic_explorer_project(tmp_path)
    undo_stack = BaleUndoStack()
    anno_path = proj_dir / "jabs" / "annotations" / "cage_00.json"

    # 1. Flip mislabeled positive bout [320, 360] -> Negative
    apply_bout_edit(
        project,
        "Grooming",
        video="cage_00.mp4",
        identity=0,
        action="flip",
        start_frame=320,
        end_frame=360,
        current_is_positive=True,
        target_is_positive=False,
        undo_stack=undo_stack,
    )
    doc = json.loads(anno_path.read_text())
    bouts_after_flip = doc["unfragmented_labels"]["0"]["Grooming"]
    flipped = next(b for b in bouts_after_flip if b["start"] == 320 and b["end"] == 360)
    assert flipped["present"] is False
    assert undo_stack.can_undo

    # 2. Undo restores `present: true`
    snap_undone = undo_stack.undo(project)
    assert snap_undone is not None and snap_undone.video == "cage_00.mp4"
    doc_restored = json.loads(anno_path.read_text())
    bouts_restored = doc_restored["unfragmented_labels"]["0"]["Grooming"]
    restored = next(b for b in bouts_restored if b["start"] == 320 and b["end"] == 360)
    assert restored["present"] is True

    # 3. Split chimeric bout [400, 460] on cage_01.mp4 at frame 430
    apply_bout_edit(
        project,
        "Grooming",
        video="cage_01.mp4",
        identity=0,
        action="split",
        start_frame=400,
        end_frame=460,
        split_frame=430,
        undo_stack=undo_stack,
    )
    doc_c1 = json.loads((proj_dir / "jabs" / "annotations" / "cage_01.json").read_text())
    bouts_c1 = doc_c1["unfragmented_labels"]["0"]["Grooming"]
    starts = [b["start"] for b in bouts_c1]
    assert 400 in starts and 432 in starts
    # Both `labels` and `unfragmented_labels` stay synchronized and sorted
    assert doc_c1["labels"]["0"]["Grooming"] == doc_c1["unfragmented_labels"]["0"]["Grooming"]


def test_find_dual_class_pairing_candidate_finds_opposite_bout(tmp_path: Path) -> None:
    """Verify `find_dual_class_pairing_candidate` unpacks `pm.load_predictions` and locates an opposite bout."""
    project, proj_dir = _build_synthetic_explorer_project(tmp_path)
    # Create dummy prediction file marker
    (proj_dir / "jabs" / "predictions" / "cage_00.h5").write_bytes(b"stub")

    probs_arr = np.full(600, 0.50, dtype=np.float32)
    # High-confidence negative span at [500..545] separated by >= w+5 (35) frames from existing labels
    probs_arr[500:546] = 0.02
    project.prediction_manager.load_predictions.return_value = (
        {0: np.zeros(600, dtype=np.int32)},
        {0: probs_arr},
        {0: np.zeros(600, dtype=np.int32)},
    )

    cand = find_dual_class_pairing_candidate(
        project,
        video="cage_00.mp4",
        identity=0,
        behavior="Grooming",
        needed_positive=False,
        exclude_span=(50, 90),
    )
    assert cand is not None
    assert cand.is_positive is False
    assert 500 <= cand.start_frame < cand.end_frame <= 545
    assert cand.duration_frames >= 18


def test_scoped_span_undo_preserves_external_timeline_edits(tmp_path: Path) -> None:
    """Verify Undo in BALE restores ONLY `touched_spans` without wiping manual timeline edits."""
    project, proj_dir = _build_synthetic_explorer_project(tmp_path)
    undo_stack = BaleUndoStack()
    anno_path = proj_dir / "jabs" / "annotations" / "cage_00.json"

    # 1. Explorer flips bout [320, 360] on (cage_00.mp4, id=0)
    apply_bout_edit(
        project,
        "Grooming",
        video="cage_00.mp4",
        identity=0,
        action="flip",
        start_frame=320,
        end_frame=360,
        current_is_positive=True,
        target_is_positive=False,
        undo_stack=undo_stack,
    )

    # 2. User manually adds a brand-new bout [510, 540] on the SAME (cage_00.mp4, id=0) track in JABS timeline!
    doc_ext = json.loads(anno_path.read_text())
    doc_ext["unfragmented_labels"]["0"]["Grooming"].append(
        {"start": 510, "end": 540, "present": True}
    )
    doc_ext["labels"]["0"]["Grooming"].append({"start": 510, "end": 540, "present": True})
    anno_path.write_text(json.dumps(doc_ext))

    # 3. User presses Undo in Explorer -> [320, 360] reverts to `present: True`, AND [510, 540] stays intact!
    undo_stack.undo(project)
    doc_after = json.loads(anno_path.read_text())
    bouts = doc_after["unfragmented_labels"]["0"]["Grooming"]
    b320 = next(b for b in bouts if b["start"] == 320 and b["end"] == 360)
    b510 = next((b for b in bouts if b["start"] == 510 and b["end"] == 540), None)
    assert b320["present"] is True
    assert b510 is not None and b510["present"] is True


def test_split_on_predicted_bout_writes_both_halves_and_undoes_cleanly(tmp_path: Path) -> None:
    """Verify `action='split'` with `source_is_predicted=True` promotes both halves around the split gap."""
    project, proj_dir = _build_synthetic_explorer_project(tmp_path)
    undo_stack = BaleUndoStack()
    anno_path = proj_dir / "jabs" / "annotations" / "cage_00.json"

    apply_bout_edit(
        project,
        "Grooming",
        video="cage_00.mp4",
        identity=0,
        action="split",
        start_frame=480,
        end_frame=540,
        split_frame=510,
        target_is_positive=True,
        source_is_predicted=True,
        pair_with_opposite=False,
        undo_stack=undo_stack,
    )
    doc = json.loads(anno_path.read_text())
    bouts = doc["unfragmented_labels"]["0"]["Grooming"]
    left_half = next((b for b in bouts if b["start"] == 480), None)
    right_half = next((b for b in bouts if b["end"] == 540), None)
    assert left_half is not None and left_half["end"] == 508 and left_half["present"] is True
    assert right_half is not None and right_half["start"] == 512 and right_half["present"] is True

    # Undo clears the newly promoted split halves cleanly back to unlabeled
    undo_stack.undo(project)
    doc_undone = json.loads(anno_path.read_text())
    bouts_undone = doc_undone["unfragmented_labels"]["0"]["Grooming"]
    assert all(b["end"] < 480 for b in bouts_undone)


def test_avi_video_extension_resolution_and_multiclass_all_predictions(tmp_path: Path) -> None:
    """Verify `.avi` video names resolve accurately and `All Classes` extracts predictions for every class."""
    from jabs.bale import collect_predicted_bout_records, resolve_project_video_name
    from jabs.core.enums import ClassifierMode

    project, proj_dir = _build_synthetic_explorer_project(tmp_path)
    # Replace cage_00.mp4 in video_manager with cage_00.avi
    avi_videos = ["cage_00.avi", *project.video_manager.videos[1:]]
    project.video_manager.videos = avi_videos
    assert resolve_project_video_name(project, "cage_00", {}) == "cage_00.avi"

    # Multi-class prediction extraction with behavior=None extracts BOTH G.1 and G.2 bouts
    project.settings_manager.classifier_mode = ClassifierMode.MULTICLASS
    project.settings_manager.behavior_names = ["G.1_Paw_Lick", "G.2_Unilateral_Face"]
    (proj_dir / "jabs" / "predictions" / "cage_00.h5").write_bytes(b"stub")

    raw_mc = np.zeros(600, dtype=np.int32)
    post_mc = np.zeros(600, dtype=np.int32)
    probs_mc = np.zeros((600, 3), dtype=np.float32)
    probs_mc[:, 0] = 0.90
    # Class 1 (G.1_Paw_Lick) at [460..495], Class 2 (G.2_Unilateral_Face) at [520..560]
    raw_mc[460:496] = 1
    post_mc[460:496] = 1
    probs_mc[460:496] = [0.10, 0.82, 0.08]
    raw_mc[520:561] = 2
    post_mc[520:561] = 2
    probs_mc[520:561] = [0.10, 0.15, 0.75]

    project.prediction_manager.load_multiclass_predictions.side_effect = lambda vid: (
        (
            {0: raw_mc},
            {0: probs_mc},
            {0: post_mc},
            ["None", "G.1_Paw_Lick", "G.2_Unilateral_Face"],
        )
        if vid == "cage_00.avi"
        else ({}, {}, {}, [])
    )

    pred_recs, _, _ = collect_predicted_bout_records(
        project, behavior=None, max_predicted_bouts=50
    )
    classes_found = {r.class_name for r in pred_recs}
    assert "G.1_Paw_Lick" in classes_found
    assert "G.2_Unilateral_Face" in classes_found
    assert all(r.video == "cage_00.avi" for r in pred_recs)


def test_multi_animal_and_non_contiguous_identity_parity(tmp_path: Path) -> None:
    """Ensure non-contiguous JSON animal keys ('1', '2') and predictions (2, 3) never shift to 0."""
    from jabs.bale import (
        collect_labeled_bout_records,
        collect_predicted_bout_records,
        extract_bout_feature_matrix,
    )

    project, proj_dir = _build_synthetic_explorer_project(tmp_path)
    project.video_manager.videos = ["cage_00.mp4"]
    for p in (proj_dir / "jabs" / "annotations").glob("*.json"):
        if p.name != "cage_00.json":
            p.unlink()
    clear_bout_extractor_cache()
    rng = np.random.default_rng(19)
    # Create 4 animal feature parquets (0, 1, 2, 3) for cage_00 with distinct values per animal
    feat_cols = [f"f{k}" for k in range(4)]
    for aid in range(4):
        p_dir = proj_dir / "jabs" / "features" / "cage_00" / str(aid)
        p_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            {
                c: rng.normal(loc=float(aid) * 10.0, scale=0.3, size=600).astype(np.float32)
                for c in feat_cols
            }
        )
        _write_synthetic_parquet_cache(p_dir, df, identity=aid)

    # Annotate ONLY animals '1' and '2' (omit '0' completely)
    bouts_id1 = [
        {"start": 30, "end": 75, "present": True},
        {"start": 160, "end": 205, "present": False},
    ]
    bouts_id2 = [{"start": 90, "end": 140, "present": False}]
    anno_doc = {
        "num_frames": 600,
        "labels": {"1": {"Grooming": bouts_id1}, "2": {"Grooming": bouts_id2}},
        "unfragmented_labels": {"1": {"Grooming": bouts_id1}, "2": {"Grooming": bouts_id2}},
    }
    (proj_dir / "jabs" / "annotations" / "cage_00.json").write_text(json.dumps(anno_doc))

    labeled_recs = collect_labeled_bout_records(project, "Grooming")
    assert {(r.identity, r.start_frame, r.is_positive) for r in labeled_recs} == {
        (1, 30, True),
        (1, 160, False),
        (2, 90, False),
    }
    # Check features came from the exact <identity>/per_frame.parquet (~10.0 for id 1, ~20.0 for id 2)
    fm = extract_bout_feature_matrix(project, labeled_recs, "Grooming")
    f0_by_id = {
        rec.identity: float(fm.frame_features_by_bout[i][:, 0].mean())
        for i, rec in enumerate(fm.records)
    }
    assert abs(f0_by_id[1] - 10.0) < 1.0
    assert abs(f0_by_id[2] - 20.0) < 1.0

    # Predictions only on animals 2 and 3 (again no 0)
    (proj_dir / "jabs" / "predictions" / "cage_00.h5").write_bytes(b"stub")
    p2_raw = np.zeros(600, dtype=np.int32)
    p2_post = np.zeros(600, dtype=np.int32)
    p2_prob = np.full(600, 0.1, dtype=np.float32)
    p2_raw[250:290] = 1
    p2_post[250:290] = 1
    p2_prob[250:290] = 0.85

    p3_raw = np.zeros(600, dtype=np.int32)
    p3_post = np.zeros(600, dtype=np.int32)
    p3_prob = np.full(600, 0.1, dtype=np.float32)
    p3_raw[320:360] = 1
    p3_post[320:360] = 1
    p3_prob[320:360] = 0.72

    project.prediction_manager.load_predictions.side_effect = lambda _vid, _beh: (
        {2: p2_raw, 3: p3_raw},
        {2: p2_prob, 3: p3_prob},
        {2: p2_post, 3: p3_post},
    )
    pred_recs, _, _ = collect_predicted_bout_records(project, "Grooming", max_predicted_bouts=20)
    assert {(r.identity, r.start_frame) for r in pred_recs} == {(2, 250), (3, 320)}


def test_real_cage_2169_21_04_animal_id_parity_if_available() -> None:
    """Read-only parity check against study_275.cage_2169.2024-11-12.21.04.top annotations & pose."""
    from jabs.bale import collect_labeled_bout_records
    from jabs.pose_estimation import open_pose_file
    from jabs.project.project import Project

    proj_dir = Path("/Users/vkumar/Downloads/PBS jabs_project")
    ann_path = proj_dir / "jabs" / "annotations" / "study_275.cage_2169.2024-11-12.21.04.top.json"
    if not proj_dir.is_dir() or not ann_path.is_file():
        pytest.skip("Local PBS jabs_project dataset not present")

    try:
        doc = json.loads(ann_path.read_text())
    except PermissionError:
        pytest.skip("Local PBS jabs_project dataset not readable in sandbox")
    expected_rearing = {
        (int(id_str), b["start"], b["end"], bool(b["present"]))
        for id_str, bdict in doc["unfragmented_labels"].items()
        for b in bdict.get("Rearing", [])
    }
    assert expected_rearing == {(2, 1670, 1700, False)}

    proj = Project(proj_dir)
    pose_path = proj.video_manager.get_cached_pose_path(
        "study_275.cage_2169.2024-11-12.21.04.top.mp4"
    )
    if pose_path is not None and Path(pose_path).is_file():
        pose = open_pose_file(Path(pose_path))
        assert [pose.identity_index_to_display(i) for i in range(pose.num_identities)] == [
            "0",
            "1",
            "2",
            "3",
        ]

    rearing_recs = collect_labeled_bout_records(proj, "Rearing")
    recs_2104 = [
        (r.identity, r.start_frame, r.end_frame, r.is_positive)
        for r in rearing_recs
        if r.video == "study_275.cage_2169.2024-11-12.21.04.top.mp4"
    ]
    assert set(recs_2104) == expected_rearing

    huddle2_recs = collect_labeled_bout_records(proj, "Huddle2")
    expected_huddle2 = {
        (int(id_str), b["start"], b["end"], bool(b["present"]))
        for id_str, bdict in doc["unfragmented_labels"].items()
        for b in bdict.get("Huddle2", [])
    }
    got_huddle2 = {
        (r.identity, r.start_frame, r.end_frame, r.is_positive)
        for r in huddle2_recs
        if r.video == "study_275.cage_2169.2024-11-12.21.04.top.mp4"
    }
    assert got_huddle2 == expected_huddle2


def _make_mc_prediction_project(tmp_path: Path):
    """Create a minimal Multi-Class JABS mock project with multi-dot video filenames."""
    import json
    from unittest.mock import MagicMock

    import numpy as np
    import pandas as pd

    from jabs.bale.bout_extractor import strip_known_video_suffix
    from jabs.core.enums import ClassifierMode

    proj_dir = tmp_path / "mc_pred_proj"
    for sub in ("annotations", "features", "predictions", "classifiers", "training_logs"):
        (proj_dir / "jabs" / sub).mkdir(parents=True, exist_ok=True)

    videos = ["clip.top.mp4", "clip.21.08.mp4"]
    rng = np.random.default_rng(7)
    n_frames = 180
    for vid in videos:
        (proj_dir / vid).touch()
        stem = strip_known_video_suffix(vid)
        feat_dir = proj_dir / "jabs" / "features" / stem / "0"
        feat_dir.mkdir(parents=True, exist_ok=True)
        mat = rng.normal(0.0, 0.25, size=(n_frames, 6)).astype(np.float32)
        mat[10:25, 0] += 2.5
        mat[120:135, 0] -= 2.5
        df = pd.DataFrame(mat, columns=[f"feat_{k}" for k in range(6)])
        _write_synthetic_parquet_cache(feat_dir, df, identity=0)

        anno_doc = {
            "file": vid,
            "num_frames": n_frames,
            "labels": {
                "0": {
                    "Turn_Left": [{"start": 10, "end": 24, "present": True}],
                    "Turn_Right": [{"start": 120, "end": 134, "present": True}],
                    "None": [{"start": 145, "end": 164, "present": True}],
                }
            },
            "unfragmented_labels": {
                "0": {
                    "Turn_Left": [{"start": 10, "end": 24, "present": True}],
                    "Turn_Right": [{"start": 120, "end": 134, "present": True}],
                    "None": [{"start": 145, "end": 164, "present": True}],
                }
            },
        }
        (proj_dir / "jabs" / "annotations" / f"{stem}.json").write_text(
            json.dumps(anno_doc), encoding="utf-8"
        )

    proj = MagicMock()
    proj.dir = proj_dir
    proj.annotation_dir = proj_dir / "jabs" / "annotations"
    proj.feature_dir = proj_dir / "jabs" / "features"
    proj.classifier_dir = proj_dir / "jabs" / "classifiers"
    proj.project_paths.project_dir = proj_dir
    proj.video_manager.videos = list(videos)
    proj.settings_manager.classifier_mode = ClassifierMode.MULTICLASS
    proj.settings_manager.behavior_names = ["Turn_Left", "Turn_Right"]
    proj.settings_manager.is_video_excluded.return_value = False
    beh_cfg = {"window_size": 10, "balance_labels": False, "symmetric_behavior": False}
    proj.settings_manager.get_behavior.return_value = beh_cfg
    proj.settings_manager.project_settings = {
        "defaults": beh_cfg,
        "settings": {"social": False},
        "behaviors": {"Turn_Left": dict(beh_cfg), "Turn_Right": dict(beh_cfg)},
    }

    def _load_anno(video_name: str) -> dict:
        stem = strip_known_video_suffix(video_name)
        p = proj_dir / "jabs" / "annotations" / f"{stem}.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return {"labels": {}, "unfragmented_labels": {}, "num_frames": n_frames}

    def _load_mc_preds(video_name: str):
        import h5py

        stem = strip_known_video_suffix(video_name)
        p = proj_dir / "jabs" / "predictions" / f"{stem}.h5"
        if not p.exists():
            return {}, {}, {}, None
        preds_d, probs_d, post_d = {}, {}, {}
        class_names = None
        with h5py.File(p, "r") as f:
            raw_cls = f.attrs.get("class_names")
            if raw_cls is not None:
                class_names = [
                    c.decode("utf-8") if isinstance(c, bytes) else str(c) for c in raw_cls
                ]
            for ident_str in f:
                ident = int(ident_str)
                grp = f[ident_str]
                raw_arr = np.asarray(grp["predictions_raw"], dtype=np.int32)
                post_arr = np.asarray(grp["predictions_post_processed"], dtype=np.int32)
                prob_arr = np.asarray(grp["prediction_probabilities"], dtype=np.float32)
                preds_d[ident] = raw_arr
                post_d[ident] = post_arr
                probs_d[ident] = prob_arr
        return preds_d, probs_d, post_d, class_names

    proj.load_annotations.side_effect = _load_anno
    proj.prediction_manager.load_multiclass_predictions.side_effect = _load_mc_preds
    return proj, proj_dir


def test_multidot_stem_resolution_and_synthesize_default_false(tmp_path: Path) -> None:
    """Multi-dot `.top.h5`/`.21.08.h5` resolve intact and empty predictions never synthesize."""
    from jabs.bale import run_bale
    from jabs.bale.bout_extractor import (
        resolve_project_video_name,
        strip_known_video_suffix,
    )
    from jabs.bale.prediction_extractor import collect_predicted_bout_records

    proj, _ = _make_mc_prediction_project(tmp_path)

    assert strip_known_video_suffix("clip.top.h5") == "clip.top"
    assert strip_known_video_suffix("clip.21.08.mp4") == "clip.21.08"
    assert resolve_project_video_name(proj, "clip.top.h5") == "clip.top.mp4"
    assert resolve_project_video_name(proj, "clip.21.08.h5") == "clip.21.08.mp4"

    pred_recs, _, _ = collect_predicted_bout_records(proj, "Turn_Left")
    assert pred_recs == []

    report = run_bale(
        proj,
        "Turn_Left",
        include_predictions=True,
        compute_mds=False,
    )
    assert all(r.source == "labeled" for r in report.feature_matrix.records)
    assert all(r.origin_source == "labeled" for r in report.feature_matrix.records)


def test_collect_predictions_from_multidot_h5_sets_origin_and_excludes_annotated(
    tmp_path: Path,
) -> None:
    """Real multi-dot `.h5` predictions exclude annotated overlaps and keep origin metadata."""
    import h5py
    import numpy as np

    from jabs.bale.prediction_extractor import collect_predicted_bout_records

    proj, proj_dir = _make_mc_prediction_project(tmp_path)
    cls_names = ["None", "Turn_Left", "Turn_Right"]

    # clip.top.h5: 10..24 overlaps human Turn_Left label; 45..64 is Turn_Left; 85..104 is Turn_Right
    top_post = np.zeros(180, dtype=np.int32)
    top_post[10:25] = 1
    top_post[45:65] = 1
    top_post[85:105] = 2
    top_probs = np.zeros((180, 3), dtype=np.float32)
    top_probs[:, 0] = 0.85
    top_probs[10:25, 1] = 0.90
    top_probs[45:65, 1] = 0.62
    top_probs[45:65, 0] = 0.35
    top_probs[85:105, 2] = 0.88
    top_probs[85:105, 0] = 0.10
    with h5py.File(proj_dir / "jabs" / "predictions" / "clip.top.h5", "w") as f:
        f.attrs["class_names"] = cls_names
        g = f.create_group("0")
        g.create_dataset("predictions_raw", data=top_post)
        g.create_dataset("predictions_post_processed", data=top_post)
        g.create_dataset("prediction_probabilities", data=top_probs)

    # clip.21.08.h5: unannotated Turn_Left bout at 45..64
    ts_post = np.zeros(180, dtype=np.int32)
    ts_post[45:65] = 1
    ts_probs = np.zeros((180, 3), dtype=np.float32)
    ts_probs[:, 0] = 0.80
    ts_probs[45:65, 1] = 0.66
    ts_probs[45:65, 0] = 0.30
    with h5py.File(proj_dir / "jabs" / "predictions" / "clip.21.08.h5", "w") as f:
        f.attrs["class_names"] = cls_names
        g = f.create_group("0")
        g.create_dataset("predictions_raw", data=ts_post)
        g.create_dataset("predictions_post_processed", data=ts_post)
        g.create_dataset("prediction_probabilities", data=ts_probs)

    recs, _, _ = collect_predicted_bout_records(proj, "Turn_Left")
    assert len(recs) == 2
    assert {r.video for r in recs} == {"clip.top.mp4", "clip.21.08.mp4"}
    assert all(r.behavior == "Turn_Left" for r in recs)
    assert all(r.origin_source == "predicted" for r in recs)
    assert all(r.origin_behavior == "Turn_Left" for r in recs)
    assert all(not (r.video == "clip.top.mp4" and r.start_frame == 10) for r in recs)

    # Promoting a predicted bout in-session preserves origin_source & origin_behavior
    recs[0].source = "labeled"
    recs[0].behavior = "None"
    recs[0].class_name = "None"
    assert recs[0].origin_source == "predicted"
    assert recs[0].origin_behavior == "Turn_Left"


def test_subtypes_tier1_pos_only_inlier_ward_lens_rf_prox_anova_and_cage_skew() -> None:
    """Verify Tier 1 Sub-Types: lenses, pos_only fit, inlier seeding, RF Ward, ANOVA eta2, skew."""
    from jabs.bale.clustering_and_embedding import (
        compute_clustering_and_embedding,
        select_feature_lens_mask,
    )
    from jabs.bale.models import (
        BoutFeatureMatrix,
        BoutRecord,
        ConfusionAndOutlierResult,
    )

    stat_names = [
        "centroid_velocity_mag_mean",
        "centroid_velocity_mag_q90",
        "shape_descriptor aspect_ratio_q50",
        "pairwise_distances nose tip_tail_mean",
        "centroid_velocity_dir_cos_mean",
        "point_mask nose_mean",
        "huddle_features min torso connected group size_mean",
        "closest_identities 0_mean",
        "point_speeds base_neck_mad",
        "angular_velocity nose neck_q90",
        "hu_moments hu1_q50",
    ]
    assert select_feature_lens_mask(stat_names, "intrinsic").tolist() == [
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
    ]
    assert select_feature_lens_mask(stat_names, "locomotion").tolist() == [
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        False,
    ]
    assert select_feature_lens_mask(stat_names, "posture").tolist() == [
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
    ]
    assert select_feature_lens_mask(stat_names, "social").tolist() == [
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
    ]
    assert all(select_feature_lens_mask(stat_names, "all").tolist())
    assert all(select_feature_lens_mask(stat_names, "classifier_weighted").tolist())

    rng = np.random.default_rng(7)
    n_pos_a, n_pos_b, n_pos_out, n_neg = 9, 4, 1, 8
    n_pos = n_pos_a + n_pos_b + n_pos_out
    n_total = n_pos + n_neg
    d_stats = len(stat_names)
    X = rng.normal(0.0, 0.05, size=(n_total, d_stats)).astype(np.float32)

    records: list[BoutRecord] = []
    # Positive Sub-Type A (9 bouts, low speed, cage_101)
    for idx in range(n_pos_a):
        X[idx, 0] = 1.0 + rng.normal(0.0, 0.05)
        X[idx, 1] = 1.2 + rng.normal(0.0, 0.05)
        X[idx, 8] = 0.5 + rng.normal(0.0, 0.04)
        X[idx, 9] = 0.4 + rng.normal(0.0, 0.04)
        records.append(
            BoutRecord(
                bout_id=idx + 1,
                source="labeled",
                video="study_1.cage_101.2024-11-12.top.mp4",
                identity=0,
                behavior="Turn_Left",
                class_name="Turn_Left",
                is_positive=True,
                start_frame=10 + idx * 30,
                end_frame=25 + idx * 30,
                duration_frames=16,
            )
        )
    # Positive Sub-Type B (4 bouts, high speed, cage_202)
    for offset in range(n_pos_b):
        idx = n_pos_a + offset
        X[idx, 0] = 10.0 + rng.normal(0.0, 0.05)
        X[idx, 1] = 11.0 + rng.normal(0.0, 0.05)
        X[idx, 8] = 6.0 + rng.normal(0.0, 0.04)
        X[idx, 9] = 5.5 + rng.normal(0.0, 0.04)
        records.append(
            BoutRecord(
                bout_id=idx + 1,
                source="labeled",
                video="study_1.cage_202.2024-11-12.top.mp4",
                identity=0,
                behavior="Turn_Left",
                class_name="Turn_Left",
                is_positive=True,
                start_frame=10 + offset * 30,
                end_frame=25 + offset * 30,
                duration_frames=16,
            )
        )
    # 1 extreme corrupted outlier spike in positive cohort (outlier_votes=4)
    idx_out = n_pos_a + n_pos_b
    X[idx_out, 0] = 1.1
    X[idx_out, 2] = 500.0
    X[idx_out, 10] = 400.0
    records.append(
        BoutRecord(
            bout_id=idx_out + 1,
            source="labeled",
            video="study_1.cage_101.2024-11-12.top.mp4",
            identity=0,
            behavior="Turn_Left",
            class_name="Turn_Left",
            is_positive=True,
            start_frame=450,
            end_frame=470,
            duration_frames=21,
        )
    )
    # 8 negative bouts with massive contrast on posture/hu_moments to verify pos_only isolation
    for offset in range(n_neg):
        idx = n_pos + offset
        X[idx, 2] = -80.0 if offset < 4 else 80.0
        X[idx, 10] = -40.0 if offset < 4 else 40.0
        records.append(
            BoutRecord(
                bout_id=idx + 1,
                source="labeled",
                video="study_1.cage_101.2024-11-12.top.mp4",
                identity=0,
                behavior="Turn_Left",
                class_name="None",
                is_positive=False,
                start_frame=600 + offset * 25,
                end_frame=615 + offset * 25,
                duration_frames=16,
            )
        )

    fm = BoutFeatureMatrix(
        records=records,
        X_bout=X,
        stat_names=stat_names,
        base_feature_names=[f"b{i}" for i in range(10)],
        # First 2 columns share base feature 0 to verify ANOVA eta2 deduplication
        stat_to_base_idx=np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int32),
    )

    # Bias classifier weights toward a decoy noise column (`hu_moments hu1_q50`)
    stat_weights = np.full(d_stats, 0.001, dtype=np.float32)
    stat_weights[10] = 0.95

    outlier_votes = np.zeros(n_total, dtype=np.int32)
    outlier_votes[idx_out] = 4

    # Construct an orthogonal RF proximity structure dividing positives [0..6] vs [7..13]
    rf_prox = np.full((n_total, n_total), 0.02, dtype=np.float32)
    rf_prox[np.ix_(range(7), range(7))] = 0.98
    rf_prox[np.ix_(range(7, n_pos), range(7, n_pos))] = 0.98
    rf_prox[np.ix_(range(n_pos, n_total), range(n_pos, n_total))] = 0.95
    np.fill_diagonal(rf_prox, 1.0)

    diag = ConfusionAndOutlierResult(
        confusion_scores=np.zeros(n_total, dtype=np.float32),
        oof_mean_probs=np.full(n_total, 0.5, dtype=np.float32),
        knn_disagreement=np.zeros(n_total, dtype=np.float32),
        ood_scores=np.zeros(n_total, dtype=np.float32),
        transition_scores=np.zeros(n_total, dtype=np.float32),
        suggested_trim_bounds=[None] * n_total,
        suggested_split_frame=[None] * n_total,
        outlier_scores=np.zeros(n_total, dtype=np.float32),
        outlier_votes=outlier_votes,
        lof_scores=np.zeros(n_total, dtype=np.float32),
        isoforest_scores=np.zeros(n_total, dtype=np.float32),
        base_feature_weights=np.full(10, 0.1, dtype=np.float32),
        stat_feature_weights=stat_weights,
        rf_proximity_matrix=rf_prox,
    )

    res = compute_clustering_and_embedding(
        fm,
        diag,
        space="unsupervised",
        subtype_scope="pos_only",
        feature_lens="intrinsic",
        compute_tsne=True,
        compute_mds=True,
    )
    assert len(res.cluster_labels) == n_total
    assert res.subtype_scope == "pos_only"
    assert res.feature_lens == "intrinsic"
    assert sum(res.cluster_counts.values()) == n_pos
    assert res.optimal_k == 2
    assert res.cluster_counts[1] == n_pos_a + n_pos_out
    assert res.cluster_counts[2] == n_pos_b
    assert len(set(res.cluster_labels[:n_pos_a])) == 1
    assert len(set(res.cluster_labels[n_pos_a : n_pos_a + n_pos_b])) == 1
    assert res.cluster_labels[0] != res.cluster_labels[n_pos_a]
    assert res.coords_tsne_2d is not None and res.coords_tsne_2d.shape == (n_total, 2)

    # Unbiased ANOVA eta2 puts true speed separator ahead of high-weight decoy, deduplicated by base
    top_eta2_feats = list(res.cluster_feature_eta2.keys())
    assert top_eta2_feats[0] in ("centroid_velocity_mag_mean", "centroid_velocity_mag_q90")
    assert res.cluster_feature_eta2[top_eta2_feats[0]] > 0.70
    assert not (
        "centroid_velocity_mag_mean" in res.cluster_z_feature_names[:4]
        and "centroid_velocity_mag_q90" in res.cluster_z_feature_names[:4]
    )

    # Cage skew flags batch confounding across cage_101 vs cage_202
    assert res.cage_skew_p is not None and res.cage_skew_p < 0.005
    assert res.cage_skew_cramers_v is not None and res.cage_skew_cramers_v >= 0.35
    assert res.dominant_cage_by_cluster[1][0] == "cage_101"
    assert res.dominant_cage_by_cluster[2][0] == "cage_202"

    # forced_k override works cleanly
    res_k3 = compute_clustering_and_embedding(
        fm,
        diag,
        space="unsupervised",
        subtype_scope="pos_only",
        feature_lens="intrinsic",
        compute_tsne=False,
        compute_mds=False,
        forced_k=3,
    )
    assert res_k3.optimal_k == 3
    assert res_k3.forced_k == 3
    assert len(res_k3.cluster_counts) == 3

    # True RF proximity Ward linkage separates [0..6] vs [7..12] inliers along rf_prox
    res_rf = compute_clustering_and_embedding(
        fm,
        diag,
        space="rf_proximity",
        subtype_scope="pos_only",
        feature_lens="all",
        compute_tsne=False,
        compute_mds=True,
    )
    assert res_rf.optimal_k == 2
    assert res_rf.silhouette > 0.5
    assert len(set(res_rf.cluster_labels[:7])) == 1
    assert len(set(res_rf.cluster_labels[7:13])) == 1
    assert res_rf.cluster_labels[0] != res_rf.cluster_labels[7]


def test_explorer_report_to_dataframe_and_to_dict(tmp_path: Path) -> None:
    """Verify `BaleReport.to_dataframe(exclude_bout_ids=...)` and JSON-safe `to_dict()`."""
    project, _ = _build_synthetic_explorer_project(tmp_path)
    report = run_bale(
        project,
        "Grooming",
        space="unsupervised",
        subtype_scope="pos_only",
        feature_lens="intrinsic",
        compute_tsne=True,
        compute_mds=True,
    )
    df_full = report.to_dataframe()
    assert len(df_full) == len(report.feature_matrix.records)
    expected_cols = {
        "bout_id",
        "source",
        "video",
        "identity",
        "behavior",
        "class_name",
        "is_positive",
        "start_frame",
        "end_frame",
        "duration_frames",
        "valid_pose_fraction",
        "confusion_score",
        "confusion_or_doubt_score",
        "oof_or_pred_prob",
        "knn_disagreement",
        "transition_score",
        "postproc_fragility",
        "outlier_score",
        "outlier_votes_5",
        "ood_score",
        "cluster_id",
        "pc1",
        "pc2",
        "tsne1",
        "tsne2",
        "suggested_trim_start",
        "suggested_trim_end",
        "suggested_split_frame",
        "top_divergent_features",
        "recommended_action",
    }
    assert expected_cols.issubset(df_full.columns)
    assert np.all(np.isfinite(df_full["pc1"].to_numpy()))
    assert np.all(np.isfinite(df_full["tsne1"].to_numpy()))

    excluded_id = int(report.feature_matrix.records[0].bout_id)
    df_filtered = report.to_dataframe(exclude_bout_ids={excluded_id})
    assert len(df_filtered) == len(df_full) - 1
    assert excluded_id not in set(df_filtered["bout_id"].tolist())

    payload = report.to_dict(top_n=5)
    serialized = json.dumps(payload)
    decoded = json.loads(serialized)
    assert decoded["behavior"] == "Grooming"
    assert decoded["space"] == "unsupervised"
    assert decoded["total_bouts"] == len(df_full)
    assert decoded["cv_groups_count"] == report.cv_groups_count
    assert len(decoded["top_confusing_labeled_bouts"]) <= 5


def test_pose_trajectory_loader_and_export_summary_all_projections(tmp_path: Path) -> None:
    """Verify `load_bout_pose_trajectory` coordinate flip, LRU cache, and `pca`/`mds`/`tsne` SVG+PNG export."""
    import h5py

    from jabs.bale.trajectory_and_export import (
        _POSE_LRU_CACHE,
        load_bout_pose_trajectory,
    )
    from jabs.pose_estimation import open_pose_file

    _POSE_LRU_CACHE.clear()
    try:
        n_frames, n_mice = 80, 2
        vname = "study_99.cage_404.2024-11-12.21.04.top.mp4"
        pose_path = tmp_path / vname.replace(".mp4", "_pose_est_v3.h5")

        # Raw HDF5 stores (y, x); PoseEstimationV3 flips -> (x, y)
        pts = np.zeros((n_frames, n_mice, 12, 2), dtype=np.uint16)
        conf = np.full((n_frames, n_mice, 12), 0.95, dtype=np.float32)
        icount = np.full((n_frames,), n_mice, dtype=np.uint8)
        itrack = np.broadcast_to(
            np.arange(1, n_mice + 1, dtype=np.uint32), (n_frames, n_mice)
        ).copy()
        for kp in range(12):
            # Animal 0: y=120+t, x=45
            pts[:, 0, kp, 0] = 120 + np.arange(n_frames, dtype=np.uint16)
            pts[:, 0, kp, 1] = 45
            # Animal 1: y=210, x=300
            pts[:, 1, kp, 0] = 210
            pts[:, 1, kp, 1] = 300
        with h5py.File(pose_path, "w") as hf:
            g = hf.create_group("poseest")
            g.attrs["version"] = np.array([3, 0], dtype=np.uint16)
            g.create_dataset("points", data=pts)
            g.create_dataset("confidence", data=conf)
            g.create_dataset("instance_count", data=icount)
            g.create_dataset("instance_track_id", data=itrack)
            # Raw static lixit coordinates also stored as (y, x) = (33, 77) -> (x=77, y=33)
            hf.create_group("static_objects").create_dataset(
                "lixit", data=np.array([[33, 77]], dtype=np.uint16)
            )

        pose_obj = open_pose_file(pose_path)
        proj_mock = MagicMock()
        proj_mock.dir = tmp_path
        proj_mock.video_manager.video_path.side_effect = lambda v: tmp_path / v
        proj_mock.video_manager.get_cached_pose_path.return_value = pose_path
        proj_mock.load_pose_est.return_value = pose_obj

        traj = load_bout_pose_trajectory(proj_mock, vname, 0, start_frame=10, end_frame=24)
        assert len(traj.frames) == 15
        assert np.allclose(traj.nose_xy[:, 0], 45.0)
        assert np.allclose(traj.nose_xy[0, 1], 130.0)
        assert np.allclose(traj.nose_xy[-1, 1], 144.0)
        assert 1 in traj.partner_centroids_xy
        assert np.allclose(traj.partner_centroids_xy[1][0], [300.0, 210.0])
        assert traj.lixit_xy is not None
        assert np.allclose(traj.lixit_xy[0], [77.0, 33.0])
        assert proj_mock.load_pose_est.call_count == 1

        # Warm call uses LRU cache without re-opening HDF5 via load_pose_est
        traj_warm = load_bout_pose_trajectory(proj_mock, vname, 1, start_frame=5, end_frame=15)
        assert proj_mock.load_pose_est.call_count == 1
        assert np.allclose(traj_warm.spine_xy[0], [300.0, 210.0])
    finally:
        _POSE_LRU_CACHE.clear()

    # Export summary figures under PCA, MDS, and t-SNE projections
    synth_proj, _ = _build_synthetic_explorer_project(tmp_path)
    report = run_bale(
        synth_proj, "Grooming", compute_tsne=True, compute_mds=True, space="classifier"
    )
    for proj_kind in ("pca", "mds", "tsne"):
        png_p = tmp_path / f"exp_{proj_kind}.png"
        svg_p = tmp_path / f"exp_{proj_kind}.svg"
        export_bale_summary_figure(report, png_p, svg_p, projection=proj_kind)
        assert png_p.exists() and png_p.stat().st_size > 10_000
        svg_txt = svg_p.read_text(encoding="utf-8")
        assert "<text" in svg_txt

    # Empty 0-bout report exports cleanly without IndexError or legend warnings
    empty_report = run_bale(synth_proj, "Unannotated_Beh")
    assert empty_report.feature_matrix.records == []
    empty_png = tmp_path / "exp_empty.png"
    empty_svg = tmp_path / "exp_empty.svg"
    export_bale_summary_figure(empty_report, empty_png, empty_svg)
    assert empty_png.exists() and empty_png.stat().st_size > 5_000
    assert "No labeled bouts available" in empty_svg.read_text(encoding="utf-8")


def test_prediction_synthesis_when_h5_missing_multiclass_exclusivity(tmp_path: Path) -> None:
    """Verify `collect_predicted_bout_records(..., synthesize_if_missing=True)` synthesizes mutually exclusive Multi-Class bouts outside annotated intervals."""
    from jabs.bale import collect_predicted_bout_records
    from jabs.core.constants import MULTICLASS_NONE_BEHAVIOR
    from jabs.project import VideoLabels

    clear_bout_extractor_cache()
    proj_dir = tmp_path / "synth_mc_pred_missing"
    anno_dir = proj_dir / "jabs" / "annotations"
    feat_dir = proj_dir / "jabs" / "features"
    pred_dir = proj_dir / "jabs" / "predictions"
    anno_dir.mkdir(parents=True)
    feat_dir.mkdir(parents=True)
    pred_dir.mkdir(parents=True)

    rng = np.random.default_rng(17)
    n_frames = 220
    feat_cols = ["angular_vel", "turn_dir", "speed_fwd", "body_len"]
    videos = [f"mc_{i:02d}.mp4" for i in range(4)]

    for _v_idx, vname in enumerate(videos):
        stem = Path(vname).stem
        unfrag: dict[str, dict[str, list[dict[str, object]]]] = {}
        for ident in (0, 1):
            X = rng.normal(0.0, 0.25, size=(n_frames, len(feat_cols))).astype(np.float32)
            # Labeled intervals: None [0..24], Turn_Left [30..54], Turn_Right [60..84]
            X[30:55, 0] += 5.0
            X[30:55, 1] += 4.5
            X[60:85, 0] -= 5.0
            X[60:85, 1] -= 4.5
            # Unlabeled distinct intervals for synthesis: Turn_Left [110..134], Turn_Right [160..184]
            X[110:135, 0] += 5.2
            X[110:135, 1] += 4.6
            X[160:185, 0] -= 5.2
            X[160:185, 1] -= 4.6

            df_dict = {col: X[:, j] for j, col in enumerate(feat_cols)}
            df_dict["_jabs_frame_valid"] = np.ones(n_frames, dtype=np.uint8)
            _write_synthetic_parquet_cache(
                feat_dir / stem / str(ident),
                pd.DataFrame(df_dict),
                identity=ident,
            )

            unfrag[str(ident)] = {
                MULTICLASS_NONE_BEHAVIOR: [{"start": 0, "end": 24, "present": True}],
                "Turn_Left": [{"start": 30, "end": 54, "present": True}],
                "Turn_Right": [{"start": 60, "end": 84, "present": True}],
            }
        doc = {
            "version": 1,
            "file": vname,
            "num_frames": n_frames,
            "labels": unfrag,
            "unfragmented_labels": unfrag,
        }
        (anno_dir / f"{stem}.json").write_text(json.dumps(doc))

    pose_mock = MagicMock()
    pose_mock.num_frames = n_frames
    pose_mock.external_identities = None
    pose_mock.identity_mask.return_value = np.ones(n_frames, dtype=bool)

    proj = MagicMock()
    proj.dir = proj_dir
    proj.project_paths.project_dir = proj_dir
    proj.project_paths.annotations_dir = anno_dir
    proj.project_paths.feature_dir = feat_dir
    proj.project_paths.prediction_dir = pred_dir
    proj.settings_manager.classifier_mode = ClassifierMode.MULTICLASS
    proj.settings_manager.behavior_names = ["Turn_Left", "Turn_Right"]
    proj.settings_manager.is_video_excluded.return_value = False
    proj.settings_manager.get_behavior.return_value = {
        "social": False,
        "segmentation": False,
        "window_size": 10,
    }
    proj.video_manager.videos = videos
    proj.video_manager.load_video_labels.side_effect = lambda vn, _pose=None: VideoLabels.load(
        json.loads((anno_dir / f"{Path(vn).stem}.json").read_text())
    )
    proj.load_pose_est.return_value = pose_mock
    proj.prediction_manager.load_predictions.side_effect = AssertionError(
        "No HDF5 prediction files exist; should synthesize instead"
    )

    recs_left, probs_left, _ = collect_predicted_bout_records(
        proj, "Turn_Left", synthesize_if_missing=True, max_predicted_bouts=100
    )
    assert len(recs_left) > 0
    assert all(r.class_name == "Turn_Left" and r.behavior == "Turn_Left" for r in recs_left)
    for r in recs_left:
        assert r.bout_id in probs_left
        # Never overlaps annotated [0..84] frames
        assert r.start_frame > 84

    recs_all, _, _ = collect_predicted_bout_records(
        proj, None, synthesize_if_missing=True, max_predicted_bouts=100
    )
    classes_seen = {r.class_name for r in recs_all}
    assert classes_seen == {"Turn_Left", "Turn_Right"}
    # Strictly mutually exclusive per (video, identity, frame)
    by_track_frame: dict[tuple[str, int, int], str] = {}
    for r in recs_all:
        for f in range(r.start_frame, r.end_frame + 1):
            key = (r.video, r.identity, f)
            assert key not in by_track_frame, f"Synthesized overlap at {key}"
            by_track_frame[key] = r.class_name


def test_find_dual_class_pairing_candidate_real_jabs_binary_confidence_semantics(
    tmp_path: Path,
) -> None:
    """Verify binary `pm.load_predictions` confidence (`[0.5, 1.0]`) + `-1` invalid-pose handling."""
    project, proj_dir = _build_synthetic_explorer_project(tmp_path)
    (proj_dir / "jabs" / "predictions" / "cage_00.h5").write_bytes(b"stub")

    # Real JABS binary predictions store predicted class in {-1, 0, 1} and
    # predicted-class confidence max(P(0), P(1)) in [0.5, 1.0] (with 0.0 on -1 invalid frames).
    preds_arr = np.full(600, 0, dtype=np.int8)
    probs_arr = np.full(600, 0.55, dtype=np.float32)

    # Invalid pose window [-1] with 0.0 confidence must be excluded from both classes
    preds_arr[420:460] = -1
    probs_arr[420:460] = 0.0
    # Confident Not-Behavior span [480..515]: pred=0, conf=0.97 -> P(pos)=0.03
    preds_arr[480:516] = 0
    probs_arr[480:516] = 0.97
    # Confident Behavior span [540..575]: pred=1, conf=0.98 -> P(pos)=0.98
    preds_arr[540:576] = 1
    probs_arr[540:576] = 0.98

    project.prediction_manager.load_predictions.return_value = (
        {0: preds_arr},
        {0: probs_arr},
        {0: preds_arr},
    )

    cand_neg = find_dual_class_pairing_candidate(
        project,
        video="cage_00.mp4",
        identity=0,
        behavior="Grooming",
        needed_positive=False,
        exclude_span=(50, 90),
    )
    assert cand_neg is not None
    assert cand_neg.is_positive is False
    assert 480 <= cand_neg.start_frame < cand_neg.end_frame <= 515

    cand_pos = find_dual_class_pairing_candidate(
        project,
        video="cage_00.mp4",
        identity=0,
        behavior="Grooming",
        needed_positive=True,
        exclude_span=(50, 90),
    )
    assert cand_pos is not None
    assert cand_pos.is_positive is True
    assert 540 <= cand_pos.start_frame < cand_pos.end_frame <= 575


def test_label_halo_tagging_and_fair_standalone_quota(tmp_path: Path) -> None:
    """Predicted fragments touching/near labeled bouts are tagged `is_label_halo` and never starve standalone bouts."""
    from jabs.bale import collect_predicted_bout_records

    project, proj_dir = _build_synthetic_explorer_project(tmp_path)
    # Overwrite cage_00.json with two clean manual spans: [100..140] and [200..220]
    anno_path = proj_dir / "jabs" / "annotations" / "cage_00.json"
    doc = json.loads(anno_path.read_text())
    spans = [
        {"start": 100, "end": 140, "present": True},
        {"start": 200, "end": 220, "present": False},
    ]
    doc["labels"]["0"]["Grooming"] = spans
    doc["unfragmented_labels"]["0"]["Grooming"] = spans
    anno_path.write_text(json.dumps(doc))

    (proj_dir / "jabs" / "predictions" / "cage_00.h5").write_bytes(b"stub")
    preds = np.zeros(600, dtype=np.int32)
    probs = np.full(600, 0.05, dtype=np.float32)

    # 1. Predicted bout [92..148] overlaps label [100..140] -> onset [92..99] & offset [141..148] touching halos (low prob = high doubt!)
    preds[92:149] = 1
    probs[92:149] = 0.51
    # 2. Nearby short halo [190..195] (4f gap before [200..220], dur=6 <= 2*margin)
    preds[190:196] = 1
    probs[190:196] = 0.52
    # 3. Standalone bout 1 at [320..360] (high confidence, far from labels)
    preds[320:361] = 1
    probs[320:361] = 0.94
    # 4. Standalone bout 2 at [440..480] (medium confidence, far from labels)
    preds[440:481] = 1
    probs[440:481] = 0.78

    project.prediction_manager.load_predictions.side_effect = lambda vid, _beh: (
        ({0: preds}, {0: probs}, {0: preds}) if vid == "cage_00.mp4" else ({}, {}, {})
    )

    # Even with tight max_predicted_bouts=2, BOTH standalone bouts survive despite low-prob halos!
    recs, _, _ = collect_predicted_bout_records(
        project,
        "Grooming",
        max_predicted_bouts=2,
        halo_margin_frames=10,
        max_halo_bouts=10,
    )
    standalone = [r for r in recs if not r.is_label_halo]
    halos = [r for r in recs if r.is_label_halo]

    assert sorted((r.start_frame, r.end_frame) for r in standalone) == [(320, 360), (440, 480)]
    halo_map = {(r.start_frame, r.end_frame): (r.halo_side, r.halo_distance_frames) for r in halos}
    assert halo_map[(92, 99)] == ("before", 0)
    assert halo_map[(141, 148)] == ("after", 0)
    assert halo_map[(190, 195)] == ("before", 4)


def test_examined_bout_store_persistence_overlap_and_reset(tmp_path: Path) -> None:
    """`ExaminedBoutStore` persists atomically, matches shifted predictions by overlap, and supports Undo/reset."""
    from jabs.bale import BoutRecord, ExaminedBoutStore

    store = ExaminedBoutStore(tmp_path)
    rec_pred = BoutRecord(
        bout_id=0,
        source="predicted",
        video="/some/path/cage_01.top.mp4",
        identity=2,
        behavior="Rearing",
        class_name="Rearing",
        is_positive=True,
        start_frame=400,
        end_frame=440,
        duration_frames=41,
    )
    rec_other = BoutRecord(
        bout_id=1,
        source="labeled",
        video="cage_01.top.mp4",
        identity=2,
        behavior="Jumping",
        class_name="Jumping",
        is_positive=True,
        start_frame=100,
        end_frame=120,
        duration_frames=21,
    )

    store.mark(rec_pred, action="examined", note="✓ Examined (saved)")
    store.mark(rec_other, action="confirmed", note="✓ Kept Jumping")
    assert store.path.is_file()

    # Reload in a brand-new instance (simulating JABS restart)
    reloaded = ExaminedBoutStore(tmp_path)
    assert reloaded.find_match(rec_pred) is not None
    assert reloaded.find_match(rec_other) is not None

    # Shifted predicted bout [403..438] (high overlap) still matches persisted Examined flag
    rec_shifted = BoutRecord(
        bout_id=9,
        source="predicted",
        video="cage_01.top.mp4",
        identity=2,
        behavior="Rearing",
        class_name="Rearing",
        is_positive=True,
        start_frame=403,
        end_frame=438,
        duration_frames=36,
    )
    assert reloaded.find_match(rec_shifted) is not None

    # Disjoint predicted bout [500..540] does NOT match
    rec_far = BoutRecord(
        bout_id=10,
        source="predicted",
        video="cage_01.top.mp4",
        identity=2,
        behavior="Rearing",
        class_name="Rearing",
        is_positive=True,
        start_frame=500,
        end_frame=540,
        duration_frames=41,
    )
    assert reloaded.find_match(rec_far) is None

    # Unmark + Undo restore works cleanly
    removed = reloaded.unmark(rec_shifted)
    assert removed is not None
    assert reloaded.find_match(rec_pred) is None
    reloaded.restore_entry(removed)
    assert reloaded.find_match(rec_pred) is not None

    # Scoped clear removes only Rearing while preserving Jumping
    assert reloaded.clear(behavior="Rearing") == 1
    assert reloaded.find_match(rec_pred) is None
    assert reloaded.find_match(rec_other) is not None
    assert reloaded.clear() == 1
    assert reloaded.count() == 0


def test_video_screener_binary_high_frames_bouts_zero_and_50k_buckets(tmp_path: Path) -> None:
    """Verify zero-Parquet whole-video scanning finds high-frame, high-bout, zero-behavior, and 50k E1-E5 outliers."""
    from jabs.bale import filter_video_records, summarize_project_videos

    proj_dir = tmp_path / "video_screener_bin"
    anno_dir = proj_dir / "jabs" / "annotations"
    pred_dir = proj_dir / "jabs" / "predictions"
    anno_dir.mkdir(parents=True)
    pred_dir.mkdir(parents=True)

    n_frames = 600
    v_high_frames = "study_275.cage_2101.2024-11-12.20.00.top.mp4"
    v_high_bouts = "study_275.cage_2102.2024-11-12.20.01.top.mp4"
    v_low_conf_e3a = "study_275.cage_2103.2024-11-12.20.02.top.mp4"
    v_fragile_e4 = "study_275.cage_2104.2024-11-12.20.03.top.mp4"
    v_zero_nearmiss = "study_275.cage_2105.2024-11-12.20.04.top.mp4"
    v_zero_clean = "study_275.cage_2106.2024-11-12.20.05.top.mp4"
    videos = [
        v_high_frames,
        v_high_bouts,
        v_low_conf_e3a,
        v_fragile_e4,
        v_zero_nearmiss,
        v_zero_clean,
    ]
    for v_name in videos:
        (pred_dir / f"{Path(v_name).stem}.h5").write_bytes(b"stub")

    # Only v_high_frames has human labels (dual-class track 0)
    anno_doc = {
        "version": 1,
        "file": v_high_frames,
        "num_frames": n_frames,
        "labels": {
            "0": {
                "Grooming": [
                    {"start": 100, "end": 180, "present": True},
                    {"start": 10, "end": 60, "present": False},
                ]
            }
        },
        "unfragmented_labels": {
            "0": {
                "Grooming": [
                    {"start": 100, "end": 180, "present": True},
                    {"start": 10, "end": 60, "present": False},
                ]
            }
        },
    }
    (anno_dir / f"{Path(v_high_frames).stem}.json").write_text(json.dumps(anno_doc))

    pred_store: dict[
        str, tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]
    ] = {}

    # 1. High frames + long bout (JABS [0.5, 1.0] predicted-class confidence format)
    r1 = np.zeros(n_frames, dtype=np.int32)
    r1[100:560] = 1
    p1 = np.full(n_frames, 0.95, dtype=np.float32)
    p1[100:560] = 0.94
    pred_store[v_high_frames] = ({0: r1}, {0: p1}, {0: r1.copy()})

    # 2. High bouts (9 distinct 15f bouts)
    r2 = np.zeros(n_frames, dtype=np.int32)
    p2 = np.full(n_frames, 0.96, dtype=np.float32)
    for k in range(9):
        s = 20 + k * 60
        r2[s : s + 15] = 1
        p2[s : s + 15] = 0.93
    pred_store[v_high_bouts] = ({0: r2}, {0: p2}, {0: r2.copy()})

    # 3. E3a low-confidence positive bout (classic [0, 1] positive-class probability schema)
    r3 = np.zeros(n_frames, dtype=np.int32)
    r3[100:130] = 1
    r3[300:330] = 1
    p3 = np.full(n_frames, 0.04, dtype=np.float32)
    p3[100:130] = 0.58  # E3a low-confidence Behavior bout
    p3[300:330] = 0.92
    pred_store[v_low_conf_e3a] = ({0: r3}, {0: p3}, {0: r3.copy()})

    # 4. E4 postprocessing fragility (raw has 100f, postprocessed keeps only 40f -> 60% delta)
    r4_raw = np.zeros(n_frames, dtype=np.int32)
    r4_raw[100:200] = 1
    r4_post = np.zeros(n_frames, dtype=np.int32)
    r4_post[100:140] = 1
    p4 = np.full(n_frames, 0.94, dtype=np.float32)
    p4[100:200] = 0.88
    pred_store[v_fragile_e4] = ({0: r4_raw}, {0: p4}, {0: r4_post})

    # 5. E5 Zero behavior + E3b low-confidence Not-Behavior near-miss span at [200..239]
    r5 = np.zeros(n_frames, dtype=np.int32)
    p5 = np.full(n_frames, 0.96, dtype=np.float32)
    p5[200:240] = 0.56  # predicted-class confidence <= 0.65 on Not-Behavior -> E3b near-miss
    pred_store[v_zero_nearmiss] = ({0: r5}, {0: p5}, {0: r5.copy()})

    # 6. E5 Zero behavior high-confidence clean control
    r6 = np.zeros(n_frames, dtype=np.int32)
    p6 = np.full(n_frames, 0.98, dtype=np.float32)
    pred_store[v_zero_clean] = ({0: r6}, {0: p6}, {0: r6.copy()})

    project = MagicMock()
    project.dir = proj_dir
    project.project_paths.project_dir = proj_dir
    project.project_paths.annotations_dir = anno_dir
    project.project_paths.prediction_dir = pred_dir
    project.settings_manager.classifier_mode = ClassifierMode.BINARY
    project.settings_manager.behavior_names = ["Grooming"]
    project.settings_manager.is_video_excluded.return_value = False
    project.video_manager.videos = videos
    project.prediction_manager.load_predictions.side_effect = lambda v, _b: pred_store.get(
        v, ({}, {}, {})
    )

    rep = summarize_project_videos(project, "Grooming", include_all_multiclass=False)
    assert len(rep.records) == 6
    assert rep.has_any_predictions is True
    assert rep.has_any_labels is True
    by_vid = {r.video: r for r in rep.records}

    # Check exact whole-video counts and 50k tags
    rec1 = by_vid[v_high_frames]
    assert rec1.pred_pos_frames == 460
    assert rec1.pred_bout_count == 1
    assert rec1.pred_max_bout_frames == 460
    assert rec1.has_labels is True
    assert rec1.label_dual_class_tracks == 1
    assert "E1_high_frames" in rec1.outlier_tags
    assert "E2_long_bout" in rec1.outlier_tags

    rec2 = by_vid[v_high_bouts]
    assert rec2.pred_bout_count == 9
    assert "E1_high_bouts" in rec2.outlier_tags

    rec3 = by_vid[v_low_conf_e3a]
    assert rec3.pred_low_conf_pos_bouts == 1
    assert "E3a_low_conf_pos" in rec3.outlier_tags
    assert rec3.focal_event_for("low_conf_pos").start_frame == 100

    rec4 = by_vid[v_fragile_e4]
    assert rec4.pred_postproc_fragility == pytest.approx(0.60, abs=1e-4)
    assert "E4_fragile" in rec4.outlier_tags

    rec5 = by_vid[v_zero_nearmiss]
    assert rec5.pred_pos_frames == 0
    assert rec5.pred_bout_count == 0
    assert rec5.pred_low_conf_none_spans == 1
    assert "E3b_low_conf_none" in rec5.outlier_tags
    assert "E5_zero_control" in rec5.outlier_tags
    assert rec5.focal_event_for("low_conf_none").start_frame == 200

    rec6 = by_vid[v_zero_clean]
    assert rec6.pred_pos_frames == 0
    assert rec6.pred_bout_count == 0
    assert rec6.pred_low_conf_none_spans == 0
    assert "E5_zero_control" in rec6.outlier_tags
    assert rec6.primary_bucket == "E5 Zero (High-Conf)"

    # Check preset filters (high frames, high bouts, zero behavior, 50k buckets, unlabeled_only, labels source)
    assert [r.video for r in filter_video_records(rep.records, preset="high_frames")] == [
        v_high_frames
    ]
    assert [r.video for r in filter_video_records(rep.records, preset="high_bouts")] == [
        v_high_bouts
    ]
    assert [r.video for r in filter_video_records(rep.records, preset="low_conf_pos")] == [
        v_low_conf_e3a
    ]
    assert [r.video for r in filter_video_records(rep.records, preset="low_conf_none")] == [
        v_zero_nearmiss
    ]
    assert [r.video for r in filter_video_records(rep.records, preset="fragile")] == [v_fragile_e4]
    assert {r.video for r in filter_video_records(rep.records, preset="zero_all")} == {
        v_zero_nearmiss,
        v_zero_clean,
    }
    assert [r.video for r in filter_video_records(rep.records, preset="zero_high_conf")] == [
        v_zero_clean
    ]
    assert [r.video for r in filter_video_records(rep.records, preset="labeled_only")] == [
        v_high_frames
    ]
    unlab_with_beh = filter_video_records(rep.records, preset="unlabeled_with_behavior")
    assert {r.video for r in unlab_with_beh} == {v_high_bouts, v_low_conf_e3a, v_fragile_e4}
    assert v_high_frames not in {
        r.video for r in filter_video_records(rep.records, preset="all", unlabeled_only=True)
    }
    # Labels source: 5 of 6 videos have 0 human labels
    zero_lbl = filter_video_records(rep.records, preset="zero_all", source="labels")
    assert len(zero_lbl) == 5

    df = rep.to_dataframe()
    assert len(df) == 6
    assert {
        "video",
        "primary_bucket",
        "pred_pos_frames",
        "pred_bout_count",
        "pred_low_conf_none_spans",
    }.issubset(set(df.columns))


def test_video_screener_multiclass_chiral_asymmetry_and_single_class_scope(tmp_path: Path) -> None:
    """Verify multi-class video screening detects chiral Left/Right skew (E1) and respects single-class filtering."""
    from jabs.bale import filter_video_records, summarize_project_videos

    proj_dir = tmp_path / "video_screener_mc"
    anno_dir = proj_dir / "jabs" / "annotations"
    pred_dir = proj_dir / "jabs" / "predictions"
    anno_dir.mkdir(parents=True)
    pred_dir.mkdir(parents=True)

    n_frames = 500
    class_names = ["None", "Turn_Left", "Turn_Right"]
    v_skew = "study_275.cage_2201.2024-11-12.20.00.top.mp4"
    v_balanced = "study_275.cage_2202.2024-11-12.20.01.top.mp4"
    v_right_only = "study_275.cage_2203.2024-11-12.20.02.top.mp4"
    v_zero = "study_275.cage_2204.2024-11-12.20.03.top.mp4"
    videos = [v_skew, v_balanced, v_right_only, v_zero]
    for v_name in videos:
        (pred_dir / f"{Path(v_name).stem}.h5").write_bytes(b"stub")

    def _make_mc(spans: list[tuple[int, int, int]]) -> tuple[np.ndarray, np.ndarray]:
        cls_arr = np.zeros(n_frames, dtype=np.int32)
        pm = np.zeros((n_frames, 3), dtype=np.float32)
        pm[:, 0] = 0.94
        pm[:, 1] = 0.03
        pm[:, 2] = 0.03
        for s, e, c_idx in spans:
            cls_arr[s:e] = c_idx
            pm[s:e, :] = 0.03
            pm[s:e, c_idx] = 0.94
        return cls_arr, pm

    mc_store = {
        v_skew: _make_mc([(20, 40, 1), (80, 100, 1), (140, 160, 1), (200, 220, 1), (260, 280, 1)]),
        v_balanced: _make_mc([(20, 40, 1), (80, 100, 2), (140, 160, 1), (200, 220, 2)]),
        v_right_only: _make_mc([(50, 80, 2), (180, 210, 2)]),
        v_zero: _make_mc([]),
    }

    def _mock_load_mc(vid: str):
        cls_arr, pm = mc_store[vid]
        return ({0: cls_arr}, {0: pm}, {0: cls_arr.copy()}, class_names)

    project = MagicMock()
    project.dir = proj_dir
    project.project_paths.project_dir = proj_dir
    project.project_paths.annotations_dir = anno_dir
    project.project_paths.prediction_dir = pred_dir
    project.settings_manager.classifier_mode = ClassifierMode.MULTICLASS
    project.settings_manager.behavior_names = ["Turn_Left", "Turn_Right"]
    project.settings_manager.is_video_excluded.return_value = False
    project.video_manager.videos = videos
    project.prediction_manager.load_multiclass_predictions.side_effect = _mock_load_mc

    # All-classes scope: chiral skew detected on v_skew; v_right_only has 2 bouts; only v_zero is zero_all
    rep_all = summarize_project_videos(project, "Turn_Left", include_all_multiclass=True)
    by_all = {r.video: r for r in rep_all.records}
    assert by_all[v_skew].pred_chiral_asymmetry == pytest.approx(1.0)
    assert "E1_chiral_asym" in by_all[v_skew].outlier_tags
    assert by_all[v_balanced].pred_chiral_asymmetry == pytest.approx(0.0)
    assert by_all[v_right_only].pred_bout_count == 2
    assert [r.video for r in filter_video_records(rep_all.records, preset="zero_all")] == [v_zero]

    # Single-class Turn_Left scope: v_right_only has 0 Turn_Left bouts -> joins zero_all alongside v_zero
    rep_left = summarize_project_videos(project, "Turn_Left", include_all_multiclass=False)
    by_left = {r.video: r for r in rep_left.records}
    assert by_left[v_skew].pred_bout_count == 5
    assert by_left[v_right_only].pred_bout_count == 0
    assert {r.video for r in filter_video_records(rep_left.records, preset="zero_all")} == {
        v_right_only,
        v_zero,
    }
    # All-classes mode populates per-class spans for JABS Set1 timeline rendering
    assert "Turn_Left" in by_all[v_skew].pred_by_class[0]
    assert "Turn_Right" in by_all[v_balanced].pred_by_class[0]


def test_video_screener_gap_interpolation_and_e3b_shoulder_separation(tmp_path: Path) -> None:
    """GapInterpolationStage frames (`raw == -1, post == 1`) stay intact and postproc drops never contaminate E3b."""
    from jabs.bale import filter_video_records, summarize_project_videos

    proj_dir = tmp_path / "video_screener_gap_e3b"
    anno_dir = proj_dir / "jabs" / "annotations"
    pred_dir = proj_dir / "jabs" / "predictions"
    anno_dir.mkdir(parents=True)
    pred_dir.mkdir(parents=True)

    n_frames = 400
    v_gap = "study_275.cage_3001.2024-11-12.20.00.top.mp4"
    v_drop = "study_275.cage_3002.2024-11-12.20.01.top.mp4"
    v_shoulder = "study_275.cage_3003.2024-11-12.20.02.top.mp4"
    v_true_e3b = "study_275.cage_3004.2024-11-12.20.03.top.mp4"
    videos = [v_gap, v_drop, v_shoulder, v_true_e3b]
    for v_name in videos:
        (pred_dir / f"{Path(v_name).stem}.h5").write_bytes(b"stub")

    # 1. GapInterpolationStage bridged 10 invalid raw frames (`raw == -1`) inside a continuous post bout [100..159]
    raw_gap = np.zeros(n_frames, dtype=np.int32)
    post_gap = np.zeros(n_frames, dtype=np.int32)
    prob_gap = np.full(n_frames, 0.05, dtype=np.float32)
    raw_gap[100:125] = 1
    raw_gap[125:135] = -1  # Missing pose frames bridged by GapInterpolationStage
    raw_gap[135:160] = 1
    post_gap[100:160] = 1
    prob_gap[100:125] = 0.90
    prob_gap[125:135] = np.nan
    prob_gap[135:160] = 0.90

    # 2. Postproc dropped a raw positive span [150..180] with borderline prob 0.65 -> E4 fragile, NOT E3b!
    raw_drop = np.zeros(n_frames, dtype=np.int32)
    post_drop = np.zeros(n_frames, dtype=np.int32)
    prob_drop = np.full(n_frames, 0.05, dtype=np.float32)
    raw_drop[150:180] = 1
    prob_drop[150:180] = 0.65

    # 3. Uncertain Not-Behavior right on the shoulder (within ±15f) of a positive bout -> NOT E3b!
    raw_sh = np.zeros(n_frames, dtype=np.int32)
    post_sh = np.zeros(n_frames, dtype=np.int32)
    prob_sh = np.full(n_frames, 0.05, dtype=np.float32)
    raw_sh[100:150] = 1
    post_sh[100:150] = 1
    prob_sh[100:150] = 0.92
    prob_sh[150:162] = 0.42  # 12 frames right next to bout end (within 15f shoulder pad)

    # 4. Standalone true E3b near-miss strictly away from any raw/post bout -> E3b!
    raw_e3b = np.zeros(n_frames, dtype=np.int32)
    post_e3b = np.zeros(n_frames, dtype=np.int32)
    prob_e3b = np.full(n_frames, 0.05, dtype=np.float32)
    prob_e3b[220:240] = 0.44

    pred_store = {
        v_gap: (raw_gap, prob_gap, post_gap),
        v_drop: (raw_drop, prob_drop, post_drop),
        v_shoulder: (raw_sh, prob_sh, post_sh),
        v_true_e3b: (raw_e3b, prob_e3b, post_e3b),
    }

    def _mock_load_binary(vid: str, _beh: str):
        r, pr, po = pred_store[vid]
        return ({0: r}, {0: pr}, {0: po})

    project = MagicMock()
    project.dir = proj_dir
    project.project_paths.project_dir = proj_dir
    project.project_paths.annotations_dir = anno_dir
    project.project_paths.prediction_dir = pred_dir
    project.settings_manager.classifier_mode = ClassifierMode.BINARY
    project.settings_manager.behavior_names = ["Huddle2"]
    project.settings_manager.is_video_excluded.return_value = False
    project.video_manager.videos = videos
    project.prediction_manager.load_predictions.side_effect = _mock_load_binary

    rep = summarize_project_videos(project, "Huddle2", use_postprocessed=True)
    by_vid = {r.video: r for r in rep.records}

    # GapInterpolationStage check: all 60 frames & 1 unsplit bout preserved, finite mean prob
    assert by_vid[v_gap].pred_pos_frames == 60
    assert by_vid[v_gap].pred_bout_count == 1
    assert by_vid[v_gap].pred_max_bout_frames == 60
    assert by_vid[v_gap].pred_mean_prob == pytest.approx(0.90, abs=1e-4)

    # E3b hygiene: postproc-dropped raw positive & bout shoulder do NOT trigger E3b
    assert by_vid[v_drop].pred_low_conf_none_spans == 0
    assert "E4_fragile" in by_vid[v_drop].outlier_tags
    assert "E3b_low_conf_none" not in by_vid[v_drop].outlier_tags
    assert by_vid[v_shoulder].pred_low_conf_none_spans == 0
    assert by_vid[v_true_e3b].pred_low_conf_none_spans == 1
    assert [r.video for r in filter_video_records(rep.records, preset="low_conf_none")] == [
        v_true_e3b
    ]


def test_video_screener_nan_probs_unlabeled_source_labels_guard_and_edge_cases(
    tmp_path: Path,
) -> None:
    """Cover NaN/Inf probs, `predictions_unlabeled` & `labels` guards, excluded videos, 0-frame tracks, and non-zero None index."""
    from jabs.bale import (
        classify_video_outliers,
        filter_video_records,
        summarize_project_videos,
    )

    proj_dir = tmp_path / "video_screener_edges"
    anno_dir = proj_dir / "jabs" / "annotations"
    pred_dir = proj_dir / "jabs" / "predictions"
    anno_dir.mkdir(parents=True)
    pred_dir.mkdir(parents=True)

    v_nan = "study_275.cage_4001.2024-11-12.20.00.top.mp4"
    v_flicker = "study_275.cage_4002.2024-11-12.20.01.top.mp4"
    v_empty_trk = "study_275.cage_4003.2024-11-12.20.02.top.mp4"
    v_disagree_lowpose = "study_275.cage_4004.2024-11-12.20.03.top.mp4"
    v_excluded = "study_275.cage_4005.2024-11-12.20.04.top.mp4"
    videos = [v_nan, v_flicker, v_empty_trk, v_disagree_lowpose, v_excluded]
    for v_name in videos:
        (pred_dir / f"{Path(v_name).stem}.h5").write_bytes(b"stub")

    # Put human labels on v_disagree_lowpose (positive on 0..100 while model predicts 0) and on v_flicker (full coverage)
    with open(anno_dir / f"{Path(v_disagree_lowpose).stem}.json", "w") as f:
        json.dump(
            {
                "num_frames": 200,
                "labels": {"0": {"Grooming": [{"start": 10, "end": 80, "present": True}]}},
                "unfragmented_labels": {
                    "0": {"Grooming": [{"start": 10, "end": 80, "present": True}]}
                },
            },
            f,
        )
    with open(anno_dir / f"{Path(v_flicker).stem}.json", "w") as f:
        json.dump(
            {
                "num_frames": 200,
                "labels": {"0": {"Grooming": [{"start": 0, "end": 199, "present": True}]}},
                "unfragmented_labels": {
                    "0": {"Grooming": [{"start": 0, "end": 199, "present": True}]}
                },
            },
            f,
        )

    # NaN probs on binary track must NOT trigger E3b
    raw_nan = np.zeros(200, dtype=np.int32)
    prob_nan = np.full(200, np.nan, dtype=np.float32)

    # 1-frame flicker on v_flicker (fully covered by human labels!)
    raw_flk = np.zeros(200, dtype=np.int32)
    raw_flk[50] = 1
    prob_flk = np.full(200, 0.05, dtype=np.float32)
    prob_flk[50] = 0.88

    # Low pose validity (< 0.75) + human-vs-model disagreement on v_disagree_lowpose
    raw_dis = np.zeros(200, dtype=np.int32)
    raw_dis[100:200] = -1  # 50% pose invalid
    prob_dis = np.full(200, 0.08, dtype=np.float32)
    prob_dis[100:200] = np.nan

    def _mock_load_binary(vid: str, _beh: str):
        if vid == v_nan:
            return ({0: raw_nan}, {0: prob_nan}, {0: raw_nan.copy()})
        if vid == v_flicker:
            return ({0: raw_flk}, {0: prob_flk}, {0: raw_flk.copy()})
        if vid == v_empty_trk:
            return (
                {0: np.zeros(0, dtype=np.int32)},
                {0: np.zeros(0, dtype=np.float32)},
                {0: np.zeros(0, dtype=np.int32)},
            )
        return ({0: raw_dis}, {0: prob_dis}, {0: raw_dis.copy()})

    project = MagicMock()
    project.dir = proj_dir
    project.project_paths.project_dir = proj_dir
    project.project_paths.annotations_dir = anno_dir
    project.project_paths.prediction_dir = pred_dir
    project.settings_manager.classifier_mode = ClassifierMode.BINARY
    project.settings_manager.behavior_names = ["Grooming"]
    project.settings_manager.is_video_excluded.side_effect = lambda v: v == v_excluded
    project.video_manager.videos = videos
    project.prediction_manager.load_predictions.side_effect = _mock_load_binary

    rep = summarize_project_videos(project, "Grooming")
    by_vid = {r.video: r for r in rep.records}
    assert v_excluded not in by_vid
    assert len(rep.records) == 4
    assert by_vid[v_nan].pred_low_conf_none_spans == 0
    assert by_vid[v_empty_trk].has_predictions is False
    # 1-frame flicker keeps finite mean prob via fallback
    assert by_vid[v_flicker].pred_mean_prob == pytest.approx(0.88, abs=1e-4)
    # Disagree + low pose validity trigger E4 tags
    assert by_vid[v_disagree_lowpose].pred_label_disagree_rate >= 0.35
    assert by_vid[v_disagree_lowpose].pose_valid_frac == pytest.approx(0.50)
    assert "E4_pred_label_disagree" in by_vid[v_disagree_lowpose].outlier_tags
    assert "E4_low_pose_valid" in by_vid[v_disagree_lowpose].outlier_tags

    # `predictions_unlabeled` must NOT falsely tag a fully-labeled video with pred_pos_frames > 0 as E5_zero_control
    classify_video_outliers(rep.records, source="predictions_unlabeled")
    assert by_vid[v_flicker].unlabeled_pred_frames == 0
    assert by_vid[v_flicker].active_max_bout("predictions_unlabeled") == 0
    assert "E5_zero_control" not in by_vid[v_flicker].outlier_tags

    # `source="labels"` never leaks prediction E3a/E3b/E4/zero_high_conf presets
    classify_video_outliers(rep.records, source="labels")
    assert filter_video_records(rep.records, preset="low_conf_pos", source="labels") == []
    assert filter_video_records(rep.records, preset="low_conf_none", source="labels") == []
    assert filter_video_records(rep.records, preset="fragile", source="labels") == []
    assert filter_video_records(rep.records, preset="zero_high_conf", source="labels") == []

    # Tiny cohort guard (n = 0, 1) never crashes
    assert classify_video_outliers([], source="predictions")["n_videos"] == 0.0
    assert classify_video_outliers([by_vid[v_nan]], source="predictions")["n_videos"] == 1.0

    # Multi-class with non-zero "None" class index + NaN rows
    mc_proj = MagicMock()
    mc_proj.dir = proj_dir
    mc_proj.project_paths.project_dir = proj_dir
    mc_proj.project_paths.annotations_dir = anno_dir
    mc_proj.project_paths.prediction_dir = pred_dir
    mc_proj.settings_manager.classifier_mode = ClassifierMode.MULTICLASS
    mc_proj.settings_manager.behavior_names = ["Turn_Left", "Turn_Right"]
    mc_proj.settings_manager.is_video_excluded.return_value = False
    mc_proj.video_manager.videos = [v_nan]
    cls_arr = np.full(100, 1, dtype=np.int32)  # index 1 == "None"
    cls_arr[20:40] = 0  # index 0 == "Turn_Left"
    pm = np.zeros((100, 3), dtype=np.float32)
    pm[:, 1] = 0.92
    pm[20:40, :] = [0.90, 0.05, 0.05]
    pm[60:80, :] = np.nan  # NaN rows must be skipped cleanly, not trigger E3b
    mc_proj.prediction_manager.load_multiclass_predictions.return_value = (
        {0: cls_arr},
        {0: pm},
        {0: cls_arr.copy()},
        ["Turn_Left", "None", "Turn_Right"],
    )
    mc_rep = summarize_project_videos(mc_proj, "Turn_Left", include_all_multiclass=True)
    assert mc_rep.records[0].pred_pos_frames == 20
    assert mc_rep.records[0].pred_low_conf_none_spans == 0
    assert mc_rep.records[0].pred_by_class[0]["Turn_Left"] == [(20, 39)]


def test_extract_bout_spatial_localization_v4_h5_and_cache(tmp_path: Path) -> None:
    """Verify unified `Pose` spatial extraction, static landmarks, warm LRU cache, and NaN safety."""
    import h5py

    from jabs.bale import clear_spatial_cache, extract_bout_spatial_locations
    from jabs.bale.models import BoutRecord
    from jabs.pose_estimation import open_pose_file

    clear_spatial_cache()
    proj_dir = tmp_path / "spatial_proj"
    cache_dir = proj_dir / "jabs" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Synthetic v3 HDF5 with raw (y, x) keypoints + static landmarks, opened via unified `open_pose_file`
    pose_path = proj_dir / "vid_a_pose_est_v3.h5"
    n_frames, n_mice, n_kpts = 60, 2, 12
    pts_yx = np.zeros((n_frames, n_mice, n_kpts, 2), dtype=np.uint16)
    conf = np.full((n_frames, n_mice, n_kpts), 0.9, dtype=np.float32)
    icount = np.full((n_frames,), n_mice, dtype=np.uint8)
    itrack = np.broadcast_to(np.arange(1, n_mice + 1, dtype=np.uint32), (n_frames, n_mice)).copy()

    # Identity 0 at (x=150, y=320) -> stored in raw H5 as (y=320, x=150)
    pts_yx[:, 0, :, 0] = 320
    pts_yx[:, 0, :, 1] = 150
    # Nose slightly offset (x=170, y=310) -> stored as (y=310, x=170)
    pts_yx[:, 0, 0, 0] = 310
    pts_yx[:, 0, 0, 1] = 170
    # Frame window [40..59] for identity 0 has low confidence -> masked out (NaN bout)
    conf[40:, 0, :] = 0.05

    # Identity 1 at (x=420, y=180)
    pts_yx[:, 1, :, 0] = 180
    pts_yx[:, 1, :, 1] = 420

    with h5py.File(pose_path, "w") as hf:
        grp = hf.create_group("poseest")
        grp.attrs["version"] = np.array([3, 0], dtype=np.uint16)
        grp.attrs["cm_per_pixel"] = 0.05
        grp.create_dataset("points", data=pts_yx)
        grp.create_dataset("confidence", data=conf)
        grp.create_dataset("instance_count", data=icount)
        grp.create_dataset("instance_track_id", data=itrack)
        so = hf.create_group("static_objects")
        # corners are stored (x, y); lixit/food_hopper are stored (y, x) in raw HDF5
        so.create_dataset(
            "corners",
            data=np.array([[100, 100], [500, 100], [100, 400], [500, 400]], dtype=np.float32),
        )
        so.create_dataset("lixit", data=np.array([[90, 260]], dtype=np.uint16))  # y=90, x=260

    project = MagicMock()
    project.dir = proj_dir
    project._paths.cache_dir = cache_dir
    project.project_paths.cache_dir = cache_dir
    project.video_manager.get_cached_pose_path.return_value = pose_path
    project.video_manager.video_path.return_value = proj_dir / "vid_a.mp4"
    project.load_pose_est.side_effect = lambda _vp: open_pose_file(pose_path, cache_dir=cache_dir)

    records = [
        BoutRecord(
            bout_id=0,
            source="labeled",
            video="vid_a.mp4",
            identity=0,
            behavior="Grooming",
            class_name="Grooming",
            is_positive=True,
            start_frame=5,
            end_frame=20,
            duration_frames=16,
        ),
        BoutRecord(
            bout_id=1,
            source="labeled",
            video="vid_a.mp4",
            identity=1,
            behavior="Grooming",
            class_name="Not-Grooming",
            is_positive=False,
            start_frame=10,
            end_frame=30,
            duration_frames=21,
        ),
        BoutRecord(
            bout_id=2,
            source="labeled",
            video="vid_a.mp4",
            identity=0,
            behavior="Grooming",
            class_name="Grooming",
            is_positive=True,
            start_frame=42,
            end_frame=55,
            duration_frames=14,
        ),
    ]

    sp = extract_bout_spatial_locations(project, records)
    assert sp.normalized is False
    assert sp.coords_xy.shape == (3, 2)
    assert list(sp.valid_mask) == [True, True, False]
    # Median of (nose=170, 9 other body keypoints=150) => x = 152.0, y = 319.0
    assert sp.coords_xy[0, 0] == pytest.approx(152.0, abs=1e-4)
    assert sp.coords_xy[0, 1] == pytest.approx(319.0, abs=1e-4)
    assert sp.nose_xy[0, 0] == pytest.approx(170.0, abs=1e-4)
    assert sp.nose_xy[0, 1] == pytest.approx(310.0, abs=1e-4)
    assert sp.coords_xy[1, 0] == pytest.approx(420.0, abs=1e-4)
    assert sp.coords_xy[1, 1] == pytest.approx(180.0, abs=1e-4)
    assert not np.isfinite(sp.coords_xy[2]).any()
    assert sp.cm_per_pixel == pytest.approx(0.05)
    assert "corners" in sp.static_objects
    assert "lixit" in sp.static_objects
    assert sp.static_objects["lixit"][0, 0] == pytest.approx(260.0)
    assert sp.static_objects["lixit"][0, 1] == pytest.approx(90.0)
    assert project.load_pose_est.call_count == 1

    # Second call hits warm spatial LRU cache cleanly without reopening pose
    sp_warm = extract_bout_spatial_locations(project, records)
    assert project.load_pose_est.call_count == 1
    assert np.allclose(sp_warm.coords_xy[:2], sp.coords_xy[:2])


def test_spatial_mixed_resolution_normalization_and_export(tmp_path: Path) -> None:
    """Verify mixed video resolutions normalize to [0, 1], populate DataFrame columns, and export PNG+SVG."""
    import cv2

    from jabs.bale import clear_spatial_cache

    clear_spatial_cache()
    project, proj_dir = _build_synthetic_explorer_project(tmp_path)

    # Create two synthetic videos with different resolutions (800x800 and 1600x1200)
    v0 = proj_dir / "cage_00.mp4"
    v1 = proj_dir / "cage_01.mp4"
    for path, w, h in ((v0, 800, 800), (v1, 1600, 1200)):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, 15.0, (w, h))
        frame = np.full((h, w, 3), 128, dtype=np.uint8)
        writer.write(frame)
        writer.release()

    def _mock_pose(vid_path: Path):
        pose = MagicMock()
        pose.identities = [0, 1, 2]
        pose.num_frames = 600
        pose.cm_per_pixel = None
        pose.static_objects = {}
        name = Path(vid_path).name
        xy = (200.0, 400.0) if name == "cage_00.mp4" else (800.0, 600.0)
        pts = np.tile(np.array(xy, dtype=np.float32), (600, 12, 1))
        msk = np.ones((600, 12), dtype=bool)
        pose.get_identity_poses.return_value = (pts, msk)
        return pose

    project.video_manager.videos = [v0, v1, *[proj_dir / f"cage_0{i}.mp4" for i in range(2, 6)]]
    project.video_manager.video_path.side_effect = lambda v: proj_dir / Path(v).name
    project.video_manager.get_cached_pose_path.side_effect = lambda v: (
        proj_dir / f"{Path(v).stem}_missing.h5"
    )
    project.load_pose_est.side_effect = _mock_pose

    rep = run_bale(project, "Grooming", compute_spatial=True)
    assert rep.spatial is not None
    assert rep.spatial.normalized is True
    assert rep.spatial.frame_width == 1.0
    assert rep.spatial.frame_height == 1.0
    assert "spatial_ms" in rep.timings_ms

    df = rep.to_dataframe()
    assert "spatial_x" in df.columns
    assert "spatial_y" in df.columns
    row0 = df[df["video"] == "cage_00.mp4"].iloc[0]
    row1 = df[df["video"] == "cage_01.mp4"].iloc[0]
    assert row0["spatial_x"] == pytest.approx(200.0 / 800.0, abs=1e-4)
    assert row0["spatial_y"] == pytest.approx(400.0 / 800.0, abs=1e-4)
    assert row1["spatial_x"] == pytest.approx(800.0 / 1600.0, abs=1e-4)
    assert row1["spatial_y"] == pytest.approx(600.0 / 1200.0, abs=1e-4)

    out_png = tmp_path / "spatial_export.png"
    png_p, svg_p = export_bale_summary_figure(rep, out_png, projection="spatial")
    assert png_p.is_file() and png_p.stat().st_size > 5_000
    assert svg_p.is_file() and "<text" in svg_p.read_text(encoding="utf-8")

    # C1 regression: recomputing embedding (e.g. Mode 3 space switch) preserves cached spatial result
    rep_unsupervised = recompute_embedding_only(rep, space="unsupervised")
    assert rep_unsupervised.spatial is rep.spatial


def test_spatial_cache_v5_min_keypoints_truncated_and_mp4_header(tmp_path: Path) -> None:
    """Verify unified `Pose` `min_valid_keypoints=3`, truncated frames NaN safety, and MP4 header parser."""
    import cv2
    import h5py

    from jabs.bale import clear_spatial_cache, extract_bout_spatial_locations
    from jabs.bale.models import BoutRecord
    from jabs.bale.spatial_localization import _read_mp4_dimensions
    from jabs.pose_estimation import open_pose_file

    clear_spatial_cache()
    proj_dir = tmp_path / "tier2_proj"
    cache_dir = proj_dir / "jabs" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # M3 regression: ISO-BMFF header parser matches cv2.VideoCapture dimensions
    mp4_path = proj_dir / "vid_b.mp4"
    writer = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (800, 600))
    writer.write(np.full((600, 800, 3), 64, dtype=np.uint8))
    writer.release()
    assert _read_mp4_dimensions(mp4_path) == (800.0, 600.0)

    # <=2 valid keypoints -> NaN, truncated frames -> NaN through unified `Pose`
    pose_path = proj_dir / "vid_b_pose_est_v3.h5"
    n_id, n_frames, n_kpts = 1, 40, 12
    pts_yx = np.full((n_frames, n_id, n_kpts, 2), np.array([310, 240], dtype=np.uint16))
    conf = np.full((n_frames, n_id, n_kpts), 0.95, dtype=np.float32)
    # Frames 20..39 only have 2 valid keypoints (< _MIN_VALID_KEYPOINTS=3) -> must become NaN
    conf[20:40, :, 2:] = 0.05
    icount = np.full((n_frames,), n_id, dtype=np.uint8)
    itrack = np.ones((n_frames, n_id), dtype=np.uint32)

    with h5py.File(pose_path, "w") as hf:
        grp = hf.create_group("poseest")
        grp.attrs["version"] = np.array([3, 0], dtype=np.uint16)
        grp.attrs["cm_per_pixel"] = 0.05
        grp.create_dataset("points", data=pts_yx)
        grp.create_dataset("confidence", data=conf)
        grp.create_dataset("instance_count", data=icount)
        grp.create_dataset("instance_track_id", data=itrack)

    project = MagicMock()
    project.dir = proj_dir
    project._paths.cache_dir = cache_dir
    project.project_paths.cache_dir = cache_dir
    project.video_manager.get_cached_pose_path.return_value = pose_path
    project.video_manager.video_path.return_value = mp4_path
    project.load_pose_est.side_effect = lambda _vp: open_pose_file(pose_path, cache_dir=cache_dir)

    records = [
        BoutRecord(
            bout_id=0,
            source="labeled",
            video="vid_b.mp4",
            identity=0,
            behavior="Grooming",
            class_name="Grooming",
            is_positive=True,
            start_frame=2,
            end_frame=15,
            duration_frames=14,
        ),
        BoutRecord(
            bout_id=1,
            source="labeled",
            video="vid_b.mp4",
            identity=0,
            behavior="Grooming",
            class_name="Not-Grooming",
            is_positive=False,
            start_frame=22,
            end_frame=35,
            duration_frames=14,
        ),
        BoutRecord(
            bout_id=2,
            source="labeled",
            video="vid_b.mp4",
            identity=0,
            behavior="Grooming",
            class_name="Grooming",
            is_positive=True,
            start_frame=100,
            end_frame=120,
            duration_frames=21,
        ),
    ]

    sp = extract_bout_spatial_locations(project, records)
    assert list(sp.valid_mask) == [True, False, False]
    assert sp.coords_xy[0, 0] == pytest.approx(240.0, abs=1e-4)
    assert sp.coords_xy[0, 1] == pytest.approx(310.0, abs=1e-4)
    assert not np.isfinite(sp.coords_xy[1]).any()
    assert not np.isfinite(sp.coords_xy[2]).any()


def test_perf_sliced_pose_and_annotation_proxy_bypass_full_pose_load(tmp_path: Path) -> None:
    """Verify unified `load_pose_cached` + `_POSE_META_CACHE` keep `LRU<=4` and reuse loaded poses."""
    import h5py

    from jabs.bale.trajectory_and_export import (
        _POSE_LRU_MAX,
        clear_pose_cache,
        load_annotation_pose_proxy,
        load_bout_pose_trajectory,
    )
    from jabs.pose_estimation import open_pose_file

    assert _POSE_LRU_MAX == 4
    clear_pose_cache()

    proj_dir = tmp_path / "sliced_pose_proj"
    cache_dir = proj_dir / "jabs" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    pose_path = proj_dir / "vid_a_pose_est_v3.h5"

    n_frames, n_mice, n_kpts = 50, 2, 12
    # Raw pose stores (y, x) coordinates; unified Pose exposes (x, y)
    pts_yx = np.zeros((n_frames, n_mice, n_kpts, 2), dtype=np.uint16)
    pts_yx[:, 0, :, 0] = 120  # y
    pts_yx[:, 0, :, 1] = 340  # x
    pts_yx[:, 1, :, 0] = 220  # y
    pts_yx[:, 1, :, 1] = 440  # x
    conf = np.full((n_frames, n_mice, n_kpts), 0.9, dtype=np.float32)
    icount = np.full((n_frames,), n_mice, dtype=np.uint8)
    itrack = np.broadcast_to(np.arange(1, n_mice + 1, dtype=np.uint32), (n_frames, n_mice)).copy()

    with h5py.File(pose_path, "w") as hf:
        grp = hf.create_group("poseest")
        grp.attrs["version"] = np.array([3, 0], dtype=np.uint16)
        grp.create_dataset("points", data=pts_yx)
        grp.create_dataset("confidence", data=conf)
        grp.create_dataset("instance_count", data=icount)
        grp.create_dataset("instance_track_id", data=itrack)

    class _CountingPoseProject:
        def __init__(self, d: Path) -> None:
            self.dir = d
            self.video_manager = _VM(d)
            self.load_calls = 0

        def load_pose_est(self, _video_path: Path):
            self.load_calls += 1
            return open_pose_file(pose_path, cache_dir=cache_dir)

    class _VM:
        def __init__(self, d: Path) -> None:
            self._d = d

        def get_cached_pose_path(self, _video: str) -> Path:
            return self._d / "vid_a_pose_est_v3.h5"

        def video_path(self, video: str) -> Path:
            return self._d / video

    proj = _CountingPoseProject(proj_dir)
    traj = load_bout_pose_trajectory(proj, "vid_a.mp4", identity=0, start_frame=10, end_frame=24)  # type: ignore[arg-type]
    assert proj.load_calls == 1
    assert len(traj.frames) == 15
    assert traj.nose_xy[:, 0].mean() == pytest.approx(340.0, abs=1e-4)
    assert traj.nose_xy[:, 1].mean() == pytest.approx(120.0, abs=1e-4)
    assert 1 in traj.partner_centroids_xy
    assert traj.partner_centroids_xy[1][:, 0].mean() == pytest.approx(440.0, abs=1e-4)

    # Warm hits reuse cached Pose / proxy without re-calling `load_pose_est`
    proxy = load_annotation_pose_proxy(proj, "vid_a.mp4")  # type: ignore[arg-type]
    assert proj.load_calls == 1
    assert proxy.num_frames == n_frames
    assert bool(proxy.identity_mask(0)[15]) is True
    assert proxy.identity_index_to_display(1) == "1"

    clear_pose_cache()


def test_perf_lazy_mds_byte_capped_cache_rf_prox_and_cancel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C5 + C7 + M3 + M6 regressions: lazy MDS, byte-capped track cache, sparse RF prox, cancel_event."""
    import threading

    import jabs.bale.bout_extractor as be
    from jabs.bale.spatial_localization import extract_bout_spatial_locations
    from jabs.bale.video_screener import summarize_project_videos

    clear_bout_extractor_cache()
    project, _ = _build_synthetic_explorer_project(tmp_path)

    # C5 + C7 + M3: compute_mds=False skips MDS, frame_features_by_bout cleared, rf_proximity valid
    rep = run_bale(
        project, "Grooming", include_predictions=False, space="classifier", compute_mds=False
    )
    assert rep.embedding.coords_mds_2d is None
    assert len(rep.feature_matrix.frame_features_by_bout) == 0

    rf_prox = rep.diagnostics.rf_proximity_matrix
    assert rf_prox is not None
    assert np.allclose(rf_prox, rf_prox.T, atol=1e-6)
    assert np.allclose(np.diag(rf_prox), 1.0, atol=1e-6)
    assert float(rf_prox.min()) >= -1e-6
    assert float(rf_prox.max()) <= 1.0 + 1e-6

    # Computing MDS once populates coords_mds_2d, and subsequent non-MDS switch preserves it
    rep_mds = recompute_embedding_only(rep, space="classifier", compute_mds=True)
    assert rep_mds.embedding.coords_mds_2d is not None
    rep_keep = recompute_embedding_only(rep_mds, space="classifier", compute_mds=False)
    assert rep_keep.embedding.coords_mds_2d is rep_mds.embedding.coords_mds_2d

    # C7: Byte cap evicts old tracks when _MAX_TRACK_CACHE_BYTES is small, clear resets to 0
    monkeypatch.setattr(be, "_MAX_TRACK_CACHE_BYTES", 50_000)
    clear_bout_extractor_cache()
    for i in range(4):
        be.load_track_per_frame_array(project.dir, f"cage_0{i}.mp4", 0)
        assert be._TRACK_CACHE_BYTES <= 50_000
    clear_bout_extractor_cache()
    assert be._TRACK_CACHE_BYTES == 0

    # M6: Pre-set cancel_event aborts run_bale, spatial localization, and video screener cleanly
    ev = threading.Event()
    ev.set()
    with pytest.raises(RuntimeError, match="cancelled"):
        run_bale(project, "Grooming", include_predictions=False, cancel_event=ev)
    with pytest.raises(RuntimeError, match="cancelled"):
        extract_bout_spatial_locations(project, rep.feature_matrix.records, cancel_event=ev)
    with pytest.raises(RuntimeError, match="cancelled"):
        summarize_project_videos(project, "Grooming", cancel_event=ev)


def test_pr608_review_hdf5_feature_cache_frame_valid_static_objects_and_invalidation(
    tmp_path: Path,
) -> None:
    """PR #608 review (#2, #5, #7): canonical HDF5 cache path, frame_valid, static_objects, per-id invalidation."""
    import jabs.bale.bout_extractor as be
    from jabs.bale import (
        collect_labeled_bout_records,
        extract_bout_feature_matrix,
        invalidate_track_cache,
    )
    from jabs.core.types import FeatureCacheMetadata, PerFrameCacheData
    from jabs.feature_extraction.features import FEATURE_VERSION
    from jabs.io.feature_cache.hdf5 import HDF5FeatureCacheWriter

    clear_bout_extractor_cache()
    proj_dir = tmp_path / "hdf5_proj"
    anno_dir = proj_dir / "jabs" / "annotations"
    feat_dir = proj_dir / "jabs" / "features"
    anno_dir.mkdir(parents=True)
    feat_dir.mkdir(parents=True)

    video_name = "study_275.cage_h5.2024-11-12.21.00.top.mp4"
    stem = "study_275.cage_h5.2024-11-12.21.00.top"
    n_frames = 80
    writer = HDF5FeatureCacheWriter()
    for ident in (0, 1):
        id_dir = feat_dir / stem / str(ident)
        frame_valid = np.ones(n_frames, dtype=np.uint8)
        frame_valid[15:20] = 0  # half of bout [10, 29] is invalid!
        feats = {
            "nose_speed": np.full(n_frames, 3.0 + ident, dtype=np.float32),
            "closest_distances dist_to_nearest": np.full(n_frames, 12.0, dtype=np.float32),
            "hu_moments hu1": np.full(n_frames, 0.5, dtype=np.float32),
            "lixit_distances nose_to_lixit": np.full(n_frames, 4.0, dtype=np.float32),
        }
        meta = FeatureCacheMetadata(
            feature_version=FEATURE_VERSION,
            identity=ident,
            num_frames=n_frames,
            pose_hash="abc123",
            distance_scale_factor=None,
            avg_wall_length=None,
            cached_window_sizes=frozenset(),
        )
        writer.write_per_frame(
            id_dir,
            meta,
            PerFrameCacheData(frame_valid=frame_valid, features=feats),
        )

    anno_doc = {
        "version": 1,
        "file": video_name,
        "num_frames": n_frames,
        "labels": {
            "0": {
                "Grooming": [
                    {"start": 10, "end": 29, "present": True},
                    {"start": 40, "end": 59, "present": False},
                ]
            },
            "1": {
                "Grooming": [
                    {"start": 10, "end": 29, "present": True},
                ]
            },
        },
        "unfragmented_labels": {
            "0": {
                "Grooming": [
                    {"start": 10, "end": 29, "present": True},
                    {"start": 40, "end": 59, "present": False},
                ]
            },
            "1": {
                "Grooming": [
                    {"start": 10, "end": 29, "present": True},
                ]
            },
        },
    }
    (anno_dir / f"{stem}.json").write_text(json.dumps(anno_doc))

    project = MagicMock()
    project.dir = proj_dir
    project.project_paths.project_dir = proj_dir
    project.project_paths.feature_dir = feat_dir
    project.project_paths.annotations_dir = anno_dir
    project.project_paths.prediction_dir = proj_dir / "jabs" / "predictions"
    from jabs.project import VideoLabels

    project.video_manager.videos = [video_name]
    project.video_manager.load_video_labels.side_effect = lambda _v, _pose=None: VideoLabels.load(
        json.loads((anno_dir / f"{stem}.json").read_text())
    )
    project.settings_manager.is_video_excluded.return_value = False
    project.settings_manager.classifier_mode = ClassifierMode.BINARY
    project.settings_manager.behavior_names = ["Grooming"]
    project.settings_manager.project_settings = {"defaults": {"window_size": 10}}
    project.settings_manager.get_behavior.return_value = {
        "window_size": 10,
        "social": False,
        "segmentation": False,
        "static_objects": {"lixit": False},
    }

    recs = collect_labeled_bout_records(project, "Grooming")
    assert len(recs) == 3

    fm = extract_bout_feature_matrix(project, recs, "Grooming")
    # Social, segmentation, and disabled lixit features are gated out; only nose_speed remains
    assert fm.base_feature_names == ["nose_speed"]
    # Bout [10, 29] has 15/20 valid frames (0.75), proving frame_valid was read from HDF5
    assert fm.records[0].valid_pose_fraction == pytest.approx(0.75, abs=1e-5)
    assert float(fm.X_bout[0, -1]) == pytest.approx(0.75, abs=1e-5)

    # Comment #5: per-identity invalidation removes identity=0 only and leaves identity=1 cached
    keys_before = {k[1] for k in be._TRACK_ARRAY_CACHE}
    assert keys_before == {0, 1}
    invalidate_track_cache(project, video_name, identity=0)
    keys_after = {k[1] for k in be._TRACK_ARRAY_CACHE}
    assert keys_after == {1}
    clear_bout_extractor_cache()


def test_pr608_review_relabel_guard_pairing_uses_pose_proxy(tmp_path: Path) -> None:
    """PR #608 review (#9): `find_dual_class_pairing_candidate` passes pose proxy to `load_video_labels`."""
    from jabs.bale import find_dual_class_pairing_candidate, run_bale

    project, proj_dir = _build_synthetic_explorer_project(tmp_path)
    rep = run_bale(project, "Grooming", include_predictions=False, space="classifier")

    pred_path = proj_dir / "jabs" / "predictions" / "cage_00.h5"
    pred_path.write_bytes(b"stub")
    probs = np.full(600, 0.05, dtype=np.float32)
    zeros = np.zeros(600, dtype=np.int32)
    project.prediction_manager.load_predictions.return_value = (
        {0: zeros},
        {0: probs},
        {0: zeros},
    )

    pose_calls: list[object] = []
    orig_load = project.video_manager.load_video_labels

    def _recording_load(video: str, pose: object = None):
        pose_calls.append(pose)
        return orig_load(video)

    project.video_manager.load_video_labels = _recording_load
    find_dual_class_pairing_candidate(
        project,
        "cage_00.mp4",
        0,
        "Grooming",
        needed_positive=False,
        exclude_span=(50, 90),
    )
    assert len(pose_calls) == 1
    assert pose_calls[0] is not None
    # Verify rep stays unused warning-free
    assert rep.behavior == "Grooming"


def test_pr608_round2_core_bugs_and_warnings(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify PR #608 round-2 core fixes: MDS scope guard, spatial byte budget, unreadable dims warn, HDF5 warn."""
    import logging

    import cv2

    import jabs.bale.bout_extractor as be
    import jabs.bale.spatial_localization as sl
    from jabs.bale import clear_spatial_cache, extract_bout_spatial_locations
    from jabs.bale.models import BoutRecord
    from jabs.core.types import FeatureCacheMetadata, PerFrameCacheData
    from jabs.feature_extraction.features import FEATURE_VERSION
    from jabs.io.feature_cache.hdf5 import HDF5FeatureCacheReader, HDF5FeatureCacheWriter

    clear_bout_extractor_cache()
    clear_spatial_cache()

    # 1. Stale MDS must NOT be reused across subtype_scope / feature_lens / space / target_behavior changes
    project, _ = _build_synthetic_explorer_project(tmp_path)
    rep_mds = run_bale(
        project,
        "Grooming",
        include_predictions=False,
        space="classifier",
        subtype_scope="pos_only",
        compute_mds=True,
    )
    assert rep_mds.embedding.coords_mds_2d is not None
    rep_pca_changed_scope = recompute_embedding_only(
        rep_mds,
        space="classifier",
        subtype_scope="all_labeled",
        compute_mds=False,
    )
    assert rep_pca_changed_scope.embedding.coords_mds_2d is None
    rep_pca_changed_lens = recompute_embedding_only(
        rep_mds,
        space="classifier",
        subtype_scope="pos_only",
        feature_lens="locomotion",
        compute_mds=False,
    )
    assert rep_pca_changed_lens.embedding.coords_mds_2d is None
    rep_pca_same = recompute_embedding_only(
        rep_mds,
        space="classifier",
        subtype_scope="pos_only",
        compute_mds=False,
    )
    assert rep_pca_same.embedding.coords_mds_2d is rep_mds.embedding.coords_mds_2d

    # 2. Byte-budgeted `_SPATIAL_VIDEO_CACHE` evicts LRU entries once budget is exceeded
    monkeypatch.setattr(sl, "_MAX_SPATIAL_CACHE_BYTES", 1024)
    e1 = sl._CachedVideoSpatial(
        s_offset=0,
        n_frames_total=40,
        body_xy=np.zeros((1, 40, 2), dtype=np.float32),
        nose_xy=np.zeros((1, 40, 2), dtype=np.float32),
    )
    e2 = sl._CachedVideoSpatial(
        s_offset=0,
        n_frames_total=40,
        body_xy=np.ones((1, 40, 2), dtype=np.float32),
        nose_xy=np.ones((1, 40, 2), dtype=np.float32),
    )
    assert e1.nbytes == 640
    sl._spatial_cache_put(("k1", 1, 1), e1)
    sl._spatial_cache_put(("k2", 1, 1), e2)
    assert len(sl._SPATIAL_VIDEO_CACHE) == 1
    assert sl._SPATIAL_CACHE_BYTES == 640
    assert sl._spatial_cache_get(("k1", 1, 1)) is None
    assert sl._spatial_cache_get(("k2", 1, 1)) is e2
    clear_spatial_cache()
    assert sl._SPATIAL_CACHE_BYTES == 0

    # 3. Normalized mode skips & warns on videos with unreadable dimensions
    mix_dir = tmp_path / "mix_dims_proj"
    mix_dir.mkdir(parents=True)
    for name, w, h in (("v0.mp4", 800, 800), ("v1.mp4", 1600, 1200)):
        writer = cv2.VideoWriter(
            str(mix_dir / name), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (w, h)
        )
        writer.write(np.full((h, w, 3), 100, dtype=np.uint8))
        writer.release()

    def _pose_stub(_vp: Path):
        p = MagicMock()
        p.identities = [0]
        p.num_frames = 30
        p.cm_per_pixel = None
        p.static_objects = {}
        pts = np.tile(np.array([200.0, 300.0], dtype=np.float32), (30, 12, 1))
        msk = np.ones((30, 12), dtype=bool)
        p.get_identity_poses.return_value = (pts, msk)
        return p

    mix_proj = MagicMock()
    mix_proj.dir = mix_dir
    mix_proj.video_manager.video_path.side_effect = lambda v: mix_dir / Path(v).name
    mix_proj.video_manager.get_cached_pose_path.side_effect = lambda v: mix_dir / f"{v}.missing.h5"
    mix_proj.load_pose_est.side_effect = _pose_stub

    mix_recs = [
        BoutRecord(
            bout_id=0,
            source="labeled",
            video="v0.mp4",
            identity=0,
            behavior="Grooming",
            class_name="Grooming",
            is_positive=True,
            start_frame=1,
            end_frame=10,
            duration_frames=10,
        ),
        BoutRecord(
            bout_id=1,
            source="labeled",
            video="v1.mp4",
            identity=0,
            behavior="Grooming",
            class_name="Grooming",
            is_positive=True,
            start_frame=1,
            end_frame=10,
            duration_frames=10,
        ),
        BoutRecord(
            bout_id=2,
            source="labeled",
            video="v_unreadable.mp4",
            identity=0,
            behavior="Grooming",
            class_name="Grooming",
            is_positive=True,
            start_frame=1,
            end_frame=10,
            duration_frames=10,
        ),
    ]
    with caplog.at_level(logging.WARNING):
        sp_mix = extract_bout_spatial_locations(mix_proj, mix_recs, include_reference_frame=False)
    assert sp_mix.normalized is True
    assert list(sp_mix.valid_mask) == [True, True, False]
    assert not np.isfinite(sp_mix.coords_xy[2]).any()
    assert any("unreadable video dimensions" in r.message for r in caplog.records)

    # 4. HDF5FeatureCacheReader failure logs a warning and treats cache as missing (`None`)
    h5_proj_dir = tmp_path / "h5_warn_proj"
    feat_dir = h5_proj_dir / "jabs" / "features"
    id_dir = feat_dir / "vid_warn" / "0"
    HDF5FeatureCacheWriter().write_per_frame(
        id_dir,
        FeatureCacheMetadata(
            feature_version=FEATURE_VERSION,
            identity=0,
            num_frames=20,
            pose_hash="h1",
            distance_scale_factor=None,
            avg_wall_length=None,
            cached_window_sizes=frozenset(),
        ),
        PerFrameCacheData(
            frame_valid=np.ones(20, dtype=np.uint8),
            features={"nose_speed": np.ones(20, dtype=np.float32)},
        ),
    )
    monkeypatch.setattr(
        HDF5FeatureCacheReader,
        "read_per_frame",
        lambda *_a, **_kw: (_ for _ in ()).throw(ValueError("simulated cache mismatch")),
    )
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        loaded = be.load_track_per_frame_array(
            h5_proj_dir,
            "vid_warn.mp4",
            0,
            use_social=False,
            use_segmentation=False,
            feature_dir=feat_dir,
        )
    assert loaded is None
    assert any("HDF5FeatureCacheReader validation failed" in r.message for r in caplog.records)
    clear_bout_extractor_cache()


def test_pr608_round3_jabs_threshold_cache_validation_store_session_group_and_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """PR #608 Round 3: CV threshold parity, Parquet/HDF5 cache validation, incremental ExaminedBoutStore, session_tracker, shared grouping, logging."""
    import logging

    import jabs.bale.bout_extractor as be
    from jabs.bale import (
        BoutRecord,
        ExaminedBoutStore,
        compute_clustering_and_embedding,
        count_dual_class_cv_groups,
        summarize_cv_track_counts,
        summarize_project_videos,
    )
    from jabs.bale.models import BoutFeatureMatrix, ConfusionAndOutlierResult
    from jabs.core.enums import CrossValidationGroupingStrategy
    from jabs.core.types import FeatureCacheMetadata, PerFrameCacheData
    from jabs.feature_extraction import FEATURE_VERSION
    from jabs.io.feature_cache.hdf5 import HDF5FeatureCacheWriter

    clear_bout_extractor_cache()

    # 1. Threshold parity (#4721974836): 12+12 frames fails (0 groups), 20+20 passes (2 groups);
    #    VIDEO & FILENAME_PATTERN match Classifier.count_label_threshold / label_threshold_met
    small_counts = {("v1.mp4", 0): (12, 12), ("v2.mp4", 0): (12, 12)}
    assert summarize_cv_track_counts(small_counts) == (0, False, False)

    exact_counts = {("v1.mp4", 0): (20, 20), ("v2.mp4", 0): (20, 20)}
    assert summarize_cv_track_counts(exact_counts) == (2, True, False)

    # Disjoint on one track (20 pos on id=0, 20 neg on id=1) fails INDIVIDUAL (0) but meets VIDEO (2)
    split_counts = {
        ("cage_1_a.mp4", 0): (20, 0),
        ("cage_1_a.mp4", 1): (0, 20),
        ("cage_2_a.mp4", 0): (20, 0),
        ("cage_2_a.mp4", 1): (0, 20),
    }
    assert summarize_cv_track_counts(
        split_counts, cv_grouping_strategy=CrossValidationGroupingStrategy.INDIVIDUAL
    ) == (0, False, False)
    assert summarize_cv_track_counts(
        split_counts, cv_grouping_strategy=CrossValidationGroupingStrategy.VIDEO
    ) == (2, True, False)

    # Multi-class count_dual_class_cv_groups delegates to MultiClassClassifier with FILENAME_PATTERN
    mc_proj = MagicMock()
    mc_proj.settings_manager.classifier_mode = ClassifierMode.MULTICLASS
    mc_proj.settings_manager.behavior_names = ["Turn_Left", "Turn_Right"]
    mc_proj.settings_manager.cv_grouping_strategy = (
        CrossValidationGroupingStrategy.FILENAME_PATTERN
    )
    mc_proj.settings_manager.cv_grouping_regex = r"(cage_\d+)"
    mc_proj.settings_manager.is_video_excluded.return_value = False
    mc_proj.video_manager.videos = ["cage_101_a.mp4", "cage_202_a.mp4"]
    mc_counts = {
        "None": {
            "cage_101_a.mp4": {0: {"fragmented_frame_counts": (25, 0)}},
            "cage_202_a.mp4": {0: {"fragmented_frame_counts": (25, 0)}},
        },
        "Turn_Left": {
            "cage_101_a.mp4": {0: {"fragmented_frame_counts": (25, 0)}},
            "cage_202_a.mp4": {0: {"fragmented_frame_counts": (25, 0)}},
        },
        "Turn_Right": {
            "cage_101_a.mp4": {0: {"fragmented_frame_counts": (25, 0)}},
            "cage_202_a.mp4": {0: {"fragmented_frame_counts": (25, 0)}},
        },
    }
    mc_proj.counts.side_effect = lambda beh: mc_counts[beh]
    g_cnt, j_met, _adr_met, _ = count_dual_class_cv_groups(mc_proj, "Turn_Left")
    assert g_cnt == 2 and j_met is True

    # 2. Parquet & HDF5 feature cache validation (#4721974846):
    #    stale FEATURE_VERSION, pose_hash mismatch, distance_scale_factor mismatch -> None
    val_dir = tmp_path / "val_cache_proj"
    f_dir = val_dir / "jabs" / "features"
    pq_dir = f_dir / "pq_vid" / "0"
    _write_synthetic_parquet_cache(
        pq_dir,
        pd.DataFrame({"nose_speed": np.ones(20, dtype=np.float32)}),
        identity=0,
        feature_version=FEATURE_VERSION - 1,
        pose_hash="old_hash",
    )
    with caplog.at_level(logging.WARNING):
        assert (
            be.load_track_per_frame_array(
                val_dir,
                "pq_vid.mp4",
                0,
                use_social=False,
                use_segmentation=False,
                feature_dir=f_dir,
            )
            is None
        )

    _write_synthetic_parquet_cache(
        pq_dir,
        pd.DataFrame({"nose_speed": np.ones(20, dtype=np.float32)}),
        identity=0,
        feature_version=FEATURE_VERSION,
        pose_hash="hash_A",
        distance_scale_factor=0.25,
    )
    assert (
        be.load_track_per_frame_array(
            val_dir,
            "pq_vid.mp4",
            0,
            use_social=False,
            use_segmentation=False,
            feature_dir=f_dir,
            expected_pose_hash="hash_B",
            validate_pose_params=True,
        )
        is None
    )
    assert (
        be.load_track_per_frame_array(
            val_dir,
            "pq_vid.mp4",
            0,
            use_social=False,
            use_segmentation=False,
            feature_dir=f_dir,
            expected_pose_hash="hash_A",
            expected_distance_scale_factor=None,
            validate_pose_params=True,
        )
        is None
    )
    assert (
        be.load_track_per_frame_array(
            val_dir,
            "pq_vid.mp4",
            0,
            use_social=False,
            use_segmentation=False,
            feature_dir=f_dir,
            expected_pose_hash="hash_A",
            expected_distance_scale_factor=0.25,
            validate_pose_params=True,
        )
        is not None
    )

    # HDF5 version mismatch returns None cleanly
    h5_dir = f_dir / "h5_stale" / "0"
    HDF5FeatureCacheWriter().write_per_frame(
        h5_dir,
        FeatureCacheMetadata(
            feature_version=FEATURE_VERSION - 1,
            identity=0,
            num_frames=20,
            pose_hash="h_stale",
            distance_scale_factor=None,
            avg_wall_length=None,
            cached_window_sizes=frozenset(),
        ),
        PerFrameCacheData(
            frame_valid=np.ones(20, dtype=np.uint8),
            features={"nose_speed": np.ones(20, dtype=np.float32)},
        ),
    )
    assert (
        be.load_track_per_frame_array(
            val_dir,
            "h5_stale.mp4",
            0,
            use_social=False,
            use_segmentation=False,
            feature_dir=f_dir,
        )
        is None
    )

    # 3. ExaminedBoutStore incremental index + OSError guard (#4721974859)
    store_proj = tmp_path / "store_proj"
    store = ExaminedBoutStore(store_proj)
    r1 = BoutRecord(0, "labeled", "v1.mp4", 0, "Grooming", "Grooming", True, 10, 30, 21)
    r2 = BoutRecord(1, "predicted", "v1.mp4", 0, "Grooming", "Grooming", True, 50, 75, 26)
    r3 = BoutRecord(2, "predicted", "v1.mp4", 0, "Grooming", "Grooming", True, 90, 115, 26)
    monkeypatch.setattr(
        store,
        "_rebuild_index",
        lambda: (_ for _ in ()).throw(
            AssertionError("_rebuild_index should not run in mark/mark_many")
        ),
    )
    store.mark(r1, persist=False)
    assert store.find_match(r1) is not None
    store.mark(r2, persist=False)
    assert store.find_match(r2) is not None
    store.unmark(r1, persist=False)
    assert store.find_match(r1) is None

    monkeypatch.undo()
    rebuild_calls = 0
    orig_rebuild = store._rebuild_index

    def _counting_rebuild() -> None:
        nonlocal rebuild_calls
        rebuild_calls += 1
        orig_rebuild()

    monkeypatch.setattr(store, "_rebuild_index", _counting_rebuild)
    store.mark_many([r1, r2, r3])
    assert rebuild_calls == 1
    assert store.find_match(r3) is not None

    monkeypatch.setattr(be, "extract_video_group_id", be.extract_video_group_id)
    monkeypatch.setattr("os.replace", lambda *_a: (_ for _ in ()).throw(PermissionError("ro")))
    with caplog.at_level(logging.WARNING):
        ok_save = store.save()
    assert ok_save is False
    monkeypatch.undo()

    # 4. session_tracker calls & no cache invalidation on edit/undo (#4721974863, #4721974869)
    edit_proj, _ = _build_synthetic_explorer_project(tmp_path / "edit_session_proj")
    # Pre-warm track cache for cage_00, id=0 and confirm apply_bout_edit + undo don't evict it
    warmed = be.load_track_per_frame_array(edit_proj.dir, "cage_00.mp4", 0)
    assert warmed is not None
    cache_len_before = len(be._TRACK_ARRAY_CACHE)
    undo_stack = BaleUndoStack()
    apply_bout_edit(
        edit_proj,
        "Grooming",
        video="cage_00.mp4",
        identity=0,
        action="flip",
        start_frame=320,
        end_frame=360,
        current_is_positive=True,
        target_is_positive=False,
        undo_stack=undo_stack,
    )
    assert len(be._TRACK_ARRAY_CACHE) == cache_len_before
    edit_proj.session_tracker.label_created.assert_called_with(
        Path("cage_00.mp4"), 0, "Grooming", False, 320, 360
    )
    edit_proj.session_tracker.label_created.reset_mock()
    undo_stack.undo(edit_proj)
    assert len(be._TRACK_ARRAY_CACHE) == cache_len_before
    edit_proj.session_tracker.label_created.assert_called_with(
        Path("cage_00.mp4"), 0, "Grooming", True, 320, 360
    )

    # 5. Shared grouping helper across video_screener and clustering_and_embedding (#4721974873)
    grp_re = r"(grp_[AB])"
    grp_recs = [
        BoutRecord(
            i,
            "labeled",
            f"exp_grp_{'A' if i < 6 else 'B'}_{i}.mp4",
            0,
            "G",
            "G",
            i % 2 == 0,
            10,
            35,
            26,
        )
        for i in range(12)
    ]
    X_grp = np.vstack(
        [
            np.vstack(
                [np.ones((6, 4), dtype=np.float32) * 3.0, np.ones((6, 4), dtype=np.float32) * -3.0]
            )
        ]
    )
    fm_grp = BoutFeatureMatrix(
        records=grp_recs,
        X_bout=X_grp,
        stat_names=["f0", "f1", "f2", "f3"],
        base_feature_names=["f0"],
        stat_to_base_idx=np.zeros(4, dtype=np.int32),
    )
    diag_grp = ConfusionAndOutlierResult(
        confusion_scores=np.zeros(12, dtype=np.float32),
        oof_mean_probs=np.linspace(0.1, 0.9, 12, dtype=np.float32),
        knn_disagreement=np.zeros(12, dtype=np.float32),
        ood_scores=np.zeros(12, dtype=np.float32),
        transition_scores=np.zeros(12, dtype=np.float32),
        suggested_trim_bounds=[None] * 12,
        suggested_split_frame=[None] * 12,
        outlier_scores=np.zeros(12, dtype=np.float32),
        outlier_votes=np.zeros(12, dtype=np.int32),
        lof_scores=np.zeros(12, dtype=np.float32),
        isoforest_scores=np.zeros(12, dtype=np.float32),
        base_feature_weights=np.ones(1, dtype=np.float32),
        stat_feature_weights=np.ones(4, dtype=np.float32),
    )
    emb_grp = compute_clustering_and_embedding(
        fm_grp,
        diag_grp,
        forced_k=2,
        compute_tsne=False,
        compute_mds=False,
        cv_grouping_regex=grp_re,
    )
    dom_groups = {g for g, _frac in emb_grp.dominant_cage_by_cluster.values()}
    assert dom_groups == {"grp_A", "grp_B"}

    scr_proj = MagicMock()
    scr_proj.project_paths.project_dir = tmp_path / "scr_grp_proj"
    (tmp_path / "scr_grp_proj" / "jabs" / "annotations").mkdir(parents=True, exist_ok=True)
    scr_proj.settings_manager.classifier_mode = ClassifierMode.BINARY
    scr_proj.settings_manager.behavior_names = ["Grooming"]
    scr_proj.settings_manager.cv_grouping_regex = grp_re
    scr_proj.settings_manager.is_video_excluded.return_value = False
    scr_proj.video_manager.videos = ["exp_grp_A_0.mp4", "exp_grp_B_6.mp4"]
    scr_rep = summarize_project_videos(scr_proj, "Grooming")
    assert [r.cage for r in scr_rep.records] == ["grp_A", "grp_B"]

    # 6. relabel_guard warning log on prediction load failure (#4721974880)
    pred_fail_proj, pred_fail_dir = _build_synthetic_explorer_project(tmp_path / "pred_log_proj")
    (pred_fail_dir / "jabs" / "predictions" / "cage_00.h5").write_bytes(b"stub")
    pred_fail_proj.prediction_manager.load_predictions.side_effect = OSError("corrupt h5")
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        cand = find_dual_class_pairing_candidate(
            pred_fail_proj, "cage_00.mp4", 0, "Grooming", needed_positive=False
        )
    assert cand is None
    assert any("Failed to load predictions" in r.message for r in caplog.records)
    clear_bout_extractor_cache()


def test_pr608_round4_stale_and_missing_cache_counts_and_mc_video_refresh(
    tmp_path: Path,
) -> None:
    """PR #608 Round 4 (#4723578504, #4723578509): count stale vs missing cache tracks; O(1)-video multi-class refresh."""
    from jabs.bale import (
        collect_multiclass_counts_by_behavior,
        refresh_multiclass_video_counts,
        summarize_multiclass_counts_by_behavior,
    )
    from jabs.core.constants import MULTICLASS_NONE_BEHAVIOR
    from jabs.core.enums import CrossValidationGroupingStrategy
    from jabs.feature_extraction import FEATURE_VERSION

    clear_bout_extractor_cache()

    # 1. Stale vs missing track counts (#4723578509)
    proj, proj_dir = _build_synthetic_explorer_project(tmp_path / "stale_counts_proj")
    feat_root = proj_dir / "jabs" / "features"
    # Make cage_00/0 stale (FEATURE_VERSION - 1), cage_01/0 incomplete (metadata removed), cage_02/0 missing
    stale_df = pd.DataFrame(
        {
            col: np.ones(600, dtype=np.float32)
            for col in (
                "nose_speed",
                "front_paws_dist",
                "body_area",
                "spine_curvature",
                "ear_dist",
                "tail_base_speed",
            )
        }
        | {"_jabs_frame_valid": np.ones(600, dtype=np.uint8)}
    )
    _write_synthetic_parquet_cache(
        feat_root / "cage_00" / "0",
        stale_df,
        identity=0,
        feature_version=FEATURE_VERSION - 1,
    )
    (feat_root / "cage_01" / "0" / "metadata.json").unlink()
    import shutil

    shutil.rmtree(feat_root / "cage_02" / "0")

    rep = run_bale(proj, "Grooming", include_predictions=False, compute_mds=False)
    assert rep.feature_matrix.skipped_stale_cache_tracks == 2
    assert rep.feature_matrix.skipped_missing_cache_tracks == 1
    assert rep.feature_matrix.skipped_feature_cache_tracks == 3
    rep_dict = rep.to_dict()
    assert rep_dict["skipped_stale_cache_tracks"] == 2
    assert rep_dict["skipped_missing_cache_tracks"] == 1

    # 2. Cached multi-class per-behavior counts + single-video refresh (#4723578504)
    mc_proj = MagicMock()
    mc_proj.settings_manager.classifier_mode = ClassifierMode.MULTICLASS
    mc_proj.settings_manager.behavior_names = ["Turn_Left", "Turn_Right"]
    mc_proj.settings_manager.cv_grouping_strategy = CrossValidationGroupingStrategy.INDIVIDUAL
    mc_proj.settings_manager.cv_grouping_regex = ""
    mc_proj.settings_manager.is_video_excluded.return_value = False
    mc_proj.video_manager.videos = ["v0.mp4", "v1.mp4", "v2.mp4"]

    counts_map = {
        MULTICLASS_NONE_BEHAVIOR: {
            "v0.mp4": {0: {"fragmented_frame_counts": (25, 0)}},
            "v1.mp4": {0: {"fragmented_frame_counts": (25, 0)}},
            "v2.mp4": {0: {"fragmented_frame_counts": (25, 0)}},
        },
        "Turn_Left": {
            "v0.mp4": {0: {"fragmented_frame_counts": (25, 0)}},
            "v1.mp4": {0: {"fragmented_frame_counts": (25, 0)}},
            "v2.mp4": {0: {"fragmented_frame_counts": (25, 0)}},
        },
        "Turn_Right": {
            "v0.mp4": {0: {"fragmented_frame_counts": (0, 0)}},
            "v1.mp4": {0: {"fragmented_frame_counts": (25, 0)}},
            "v2.mp4": {0: {"fragmented_frame_counts": (25, 0)}},
        },
    }
    mc_proj.counts.side_effect = lambda beh: counts_map[beh]
    mc_proj.load_counts.side_effect = lambda vid, beh: counts_map[beh].get(vid, {})

    cached = collect_multiclass_counts_by_behavior(mc_proj)
    mc_proj.counts.reset_mock()
    mc_proj.load_counts.reset_mock()

    # Mutate ONLY v0.mp4 so it drops from 2 classes (None + Turn_Left) to 1 class (None only)
    counts_map["Turn_Left"]["v0.mp4"] = {0: {"fragmented_frame_counts": (0, 0)}}
    assert refresh_multiclass_video_counts(mc_proj, cached, {"v0.mp4"}) is True
    assert mc_proj.counts.call_count == 0
    assert {call.args[0] for call in mc_proj.load_counts.call_args_list} == {"v0.mp4"}

    g_cnt, j_met, _adr_met, _ = summarize_multiclass_counts_by_behavior(
        mc_proj, "Turn_Left", cached
    )
    assert g_cnt == 2 and j_met is True
    clear_bout_extractor_cache()
