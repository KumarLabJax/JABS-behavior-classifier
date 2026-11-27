"""True End-to-End (E2E) integration suite for JABS + BALE (Bout & Label Explorer).

Unlike unit/widget tests that mock ``Project`` or ``CentralWidget``, every test
in this module creates real on-disk JABS projects with:
  - real ``.mp4`` videos (written with OpenCV ``VideoWriter``),
  - real multi-animal ``*_pose_est_v3.h5`` HDF5 pose files + static lixit,
  - real Parquet per-frame and windowed feature caches (``IdentityFeatures``),
  - real ``jabs/annotations/*.json`` label files,
  - real trained Multi-Class (``Turn_Left``, ``Turn_Right``, ``None``) and Binary
    (``Grooming``) classifiers + real ``jabs/predictions/*.h5`` files, and
  - a real two-window Qt session (``MainWindow`` + ``CentralWidget`` +
    ``PlayerWidget`` + ``BehaviorTimelineWidget`` + ``SearchBarWidget`` +
    ``BaleDialog`` + real ``QThreadPool`` background workers).

Only blocking modal popups (``MessageDialog``, ``QFileDialog``, ``QMenu.exec``)
and ``MainWindow(..., process_pool=...)`` are stubbed so headless ``offscreen``
execution remains fast and deterministic without mutating any user dataset or
user ``QSettings``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import TracebackType

import cv2
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from jabs.bale import (
    clear_bout_extractor_cache,
    count_dual_class_cv_groups,
    export_bale_summary_figure,
    run_bale,
)
from jabs.bale.trajectory_and_export import _POSE_LRU_CACHE
from jabs.classifier import Classifier, MultiClassClassifier
from jabs.classifier.inference import predict_identity
from jabs.core.constants import FINAL_TRAIN_SEED, MULTICLASS_NONE_BEHAVIOR
from jabs.core.enums import CacheFormat, ClassifierMode, ClassifierType
from jabs.feature_extraction import DEFAULT_WINDOW_SIZE, IdentityFeatures
from jabs.project import Project, VideoLabels

try:
    from PySide6 import QtCore, QtGui, QtWidgets
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    from jabs.ui.classify_strategy import BinaryClassifyStrategy, MultiClassClassifyStrategy
    from jabs.ui.dialogs.message_dialog import MessageDialog
    from jabs.ui.main_window.main_window import MainWindow
    from jabs.ui.training_strategy import BinaryTrainingStrategy, MultiClassTrainingStrategy

    SKIP_UI_TESTS = False
    SKIP_REASON = ""
except ImportError as e:
    SKIP_UI_TESTS = True
    SKIP_REASON = f"Qt/UI dependencies not available: {e}"

pytestmark = pytest.mark.skipif(SKIP_UI_TESTS, reason=SKIP_REASON)

VIDEOS_MULTICLASS = [
    "study_275.cage_2155.2024-11-12.21.04.top.mp4",
    "study_275.cage_2156.2024-11-12.21.04.top.mp4",
    "study_275.cage_2157.2024-11-12.21.04.top.mp4",
    "study_275.cage_2158.2024-11-12.21.04.top.mp4",
]
VIDEOS_BINARY = VIDEOS_MULTICLASS[:2]


@pytest.fixture(scope="module")
def qapp() -> QtWidgets.QApplication:
    """Return the active QApplication instance."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


@pytest.fixture(autouse=True)
def _isolated_env_and_caches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect QSettings into ``tmp_path`` and clear explorer caches around each test."""
    qsettings_dir = tmp_path / "qsettings"
    qsettings_dir.mkdir(parents=True, exist_ok=True)
    QtCore.QSettings.setDefaultFormat(QtCore.QSettings.Format.IniFormat)
    QtCore.QSettings.setPath(
        QtCore.QSettings.Format.IniFormat,
        QtCore.QSettings.Scope.UserScope,
        str(qsettings_dir),
    )
    QtCore.QSettings.setPath(
        QtCore.QSettings.Format.IniFormat,
        QtCore.QSettings.Scope.SystemScope,
        str(qsettings_dir),
    )
    clear_bout_extractor_cache()
    _POSE_LRU_CACHE.clear()
    yield
    clear_bout_extractor_cache()
    _POSE_LRU_CACHE.clear()


def _wait_until(
    qapp: QtWidgets.QApplication,
    predicate,
    *,
    timeout_ms: int = 15_000,
    poll_ms: int = 20,
    desc: str = "condition",
) -> None:
    """Pump Qt events until ``predicate()`` returns True or raise AssertionError."""
    elapsed = 0
    while elapsed < timeout_ms:
        qapp.processEvents()
        if predicate():
            return
        # Sleep in Python (releasing the GIL) rather than QTest.qWait() so QThreadPool/PyArrow workers don't stall on macOS
        time.sleep(poll_ms / 1000.0)
        elapsed += poll_ms
    qapp.processEvents()
    if not predicate():
        raise AssertionError(f"Timed out after {timeout_ms} ms waiting for {desc}")


def _write_synthetic_video(video_path: Path, num_frames: int = 180, n_animals: int = 4) -> None:
    """Write a compact valid H.264/mp4v ``160x120`` 30 fps MP4 video."""
    w, h = 160, 120
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open OpenCV VideoWriter for {video_path}")
    try:
        for f_idx in range(num_frames):
            frame = np.full((h, w, 3), 24, dtype=np.uint8)
            for ident in range(n_animals):
                cx = int(28 + ident * 30 + (f_idx % 20))
                cy = int(25 + (ident // 2) * 45 + (f_idx % 12))
                cv2.circle(frame, (cx, cy), 5, (200 - ident * 35, 180, 90 + ident * 30), -1)
            writer.write(frame)
    finally:
        writer.release()


def _write_synthetic_pose_v3(pose_path: Path, num_frames: int = 180, n_animals: int = 4) -> None:
    """Write a valid multi-animal ``PoseEstimationV3`` HDF5 pose file with static lixit."""
    points_yx = np.zeros((num_frames, n_animals, 12, 2), dtype=np.uint16)
    conf = np.full((num_frames, n_animals, 12), 0.95, dtype=np.float32)
    icount = np.full((num_frames,), n_animals, dtype=np.uint8)
    itrack = np.tile(np.arange(n_animals, dtype=np.uint32), (num_frames, 1))

    # Build distinct smooth sinusoidal + drift keypoints per identity in (y, x)
    keypoint_offsets_yx = np.array(
        [
            [-6.0, 6.0],  # nose
            [-4.0, 2.0],  # left_ear
            [-4.0, 8.0],  # right_ear
            [-2.0, 5.0],  # neck
            [0.0, 1.0],  # left_front_paw
            [0.0, 9.0],  # right_front_paw
            [2.0, 5.0],  # spine
            [6.0, 1.0],  # left_rear_paw
            [6.0, 9.0],  # right_rear_paw
            [8.0, 5.0],  # base_tail
            [11.0, 5.0],  # mid_tail
            [14.0, 5.0],  # tip_tail
        ],
        dtype=np.float64,
    )
    frames_f = np.arange(num_frames, dtype=np.float64)
    for ident in range(n_animals):
        base_y = 26.0 + (ident // 2) * 44.0
        base_x = 28.0 + (ident % 2) * 65.0
        cy = base_y + 4.0 * np.sin(frames_f / 8.0 + ident)
        cx = base_x + 0.12 * frames_f + 3.0 * np.cos(frames_f / 10.0)
        for kp in range(12):
            ys = np.clip(np.round(cy + keypoint_offsets_yx[kp, 0]), 5, 114).astype(np.uint16)
            xs = np.clip(np.round(cx + keypoint_offsets_yx[kp, 1]), 5, 154).astype(np.uint16)
            points_yx[:, ident, kp, 0] = ys
            points_yx[:, ident, kp, 1] = xs

    with h5py.File(pose_path, "w") as hf:
        g = hf.create_group("poseest")
        g.attrs["version"] = np.array([3, 0], dtype=np.uint16)
        g.create_dataset("points", data=points_yx)
        g.create_dataset("confidence", data=conf)
        g.create_dataset("instance_count", data=icount)
        g.create_dataset("instance_track_id", data=itrack)
        # Stored in raw HDF5 as (y, x) = (25, 20)
        hf.create_group("static_objects").create_dataset(
            "lixit", data=np.array([[25, 20]], dtype=np.uint16)
        )


def _stamp_parquet_table(
    parquet_path: Path,
    frame_modifiers: list[tuple[int, int, dict[int, float]]],
    *,
    noise_seed: int,
) -> None:
    """Rewrite numeric feature columns deterministically while preserving Parquet schema."""
    orig_table = pq.read_table(parquet_path)
    df = orig_table.to_pandas()
    feature_cols = [
        c
        for c in df.columns
        if not c.startswith("_jabs") and np.issubdtype(df[c].dtype, np.number)
    ]
    n_rows = len(df)
    rng = np.random.default_rng(noise_seed)
    mat = rng.normal(0.0, 0.12, size=(n_rows, len(feature_cols))).astype(np.float64)

    for s_idx, e_idx, col_vals in frame_modifiers:
        lo = max(0, int(s_idx))
        hi = min(n_rows, int(e_idx) + 1)
        for col_idx, val in col_vals.items():
            if 0 <= col_idx < len(feature_cols):
                mat[lo:hi, col_idx] = float(val) + rng.normal(0.0, 0.08, size=hi - lo)

    for j, col in enumerate(feature_cols):
        df[col] = mat[:, j].astype(orig_table.schema.field(col).type.to_pandas_dtype())
    if "_jabs_frame_valid" in df.columns:
        df["_jabs_frame_valid"] = np.ones(n_rows, dtype=np.uint8)

    pq.write_table(
        pa.Table.from_pandas(df, schema=orig_table.schema, preserve_index=False), parquet_path
    )


def _build_real_disk_project(
    proj_dir: Path,
    *,
    mode: str = "multiclass",
) -> Project:
    """Create a complete real JABS project on disk (MP4 + V3 HDF5 + Parquet + Labels)."""
    proj_dir.mkdir(parents=True, exist_ok=True)
    videos = VIDEOS_MULTICLASS if mode == "multiclass" else VIDEOS_BINARY
    n_animals = 4 if mode == "multiclass" else 2
    n_frames = 180

    for vname in videos:
        _write_synthetic_video(proj_dir / vname, num_frames=n_frames, n_animals=n_animals)
        pose_name = f"{Path(vname).stem}_pose_est_v3.h5"
        _write_synthetic_pose_v3(proj_dir / pose_name, num_frames=n_frames, n_animals=n_animals)

    project = Project(
        proj_dir,
        enable_video_check=True,
        enable_session_tracker=False,
    )
    sm = project.settings_manager
    settings = dict(sm.project_settings.get("settings", {}))
    settings["classifier_mode"] = (
        ClassifierMode.MULTICLASS.value if mode == "multiclass" else ClassifierMode.BINARY.value
    )
    defaults = dict(sm.project_settings.get("defaults", project.get_project_defaults()))
    defaults.update(
        {
            "window_size": 10,
            "social": False,
            "symmetric_behavior": False,
            "balance_labels": False,
            "segmentation": False,
            "window": True,
            "fft": False,
        }
    )
    sm.save_project_file({"settings": settings, "defaults": defaults, "window_sizes": [5, 10]})

    if mode == "multiclass":
        for beh in ("Turn_Left", "Turn_Right"):
            sm.save_behavior(beh, dict(defaults))
    else:
        sm.save_behavior("Grooming", dict(defaults))

    # Extract genuine JABS Parquet feature caches per (video, identity), then stamp clean signals
    for v_idx, vname in enumerate(videos):
        pose_est = project.load_pose_est(vname)
        vl = VideoLabels(vname, n_frames)
        stem = Path(vname).stem

        for ident in range(n_animals):
            feat_obj = IdentityFeatures(
                vname,
                ident,
                project.feature_dir,
                pose_est,
                op_settings=defaults,
                cache_format=CacheFormat.PARQUET,
            )
            feat_obj.get_window_features(10)

            track_idx = v_idx * n_animals + ident
            if mode == "multiclass":
                subtype_sign = 6.0 if (track_idx % 2 == 0) else -6.0
                turn_left_vals = {0: 8.0, 1: 8.0, 2: -1.5, 4: subtype_sign, 5: subtype_sign}
                turn_right_vals = {0: -8.0, 1: -8.0, 2: 4.5, 3: 4.5, 4: 0.0, 5: 0.0}
                none_vals = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}

                modifiers: list[tuple[int, int, dict[int, float]]] = [
                    (0, 22, none_vals),
                    (30, 52, turn_left_vals),
                    (60, 82, turn_right_vals),
                    # Unannotated Turn_Left & Turn_Right bouts for real classifier predictions
                    (96, 120, turn_left_vals),
                    (134, 158, turn_right_vals),
                ]
                # Planted Animal-ID Mislabel on cage_2155 Animal 1 (while Animal 0 is clean Turn_Left)
                if v_idx == 0 and ident == 1:
                    modifiers[1] = (26, 56, turn_right_vals)
                # Planted Chimeric Boundary Slop on cage_2156 Animal 0: [30..42]=None, [43..56]=Turn_Left
                if v_idx == 1 and ident == 0:
                    modifiers[1] = (28, 42, none_vals)
                    modifiers.append((43, 58, turn_left_vals))

                tl_end = 56 if (v_idx == 1 and ident == 0) else 52
                vl.get_track_labels(str(ident), MULTICLASS_NONE_BEHAVIOR).label_behavior(0, 22)
                vl.get_track_labels(str(ident), "Turn_Left").label_behavior(30, tl_end)
                vl.get_track_labels(str(ident), "Turn_Right").label_behavior(60, 82)
            else:
                # Binary Grooming across 2 videos * 2 animals = 4 CV groups (< 15 Dual-Class CV rule threshold)
                pos_vals = dict.fromkeys(range(25), 8.0)
                neg_vals = dict.fromkeys(range(25), -8.0)
                modifiers = [
                    (5, 27, pos_vals),
                    (36, 58, neg_vals),
                    # Unannotated candidate negative & positive regions for real predict & auto-pairing
                    (76, 102, neg_vals),
                    (124, 152, pos_vals),
                ]
                trk = vl.get_track_labels(str(ident), "Grooming")
                trk.label_behavior(5, 27)
                trk.label_not_behavior(36, 58)

            pf_pq = project.feature_dir / stem / str(ident) / "per_frame.parquet"
            win_pq = project.feature_dir / stem / str(ident) / "window_10.parquet"
            for pq_i, pq_path in enumerate((pf_pq, win_pq)):
                _stamp_parquet_table(
                    pq_path,
                    modifiers,
                    noise_seed=1000 + track_idx * 17 + pq_i,
                )

        project.save_annotations(vl, pose_est)

    return Project(proj_dir, enable_video_check=True, enable_session_tracker=False)


def _train_and_predict_project(
    proj_dir: Path, *, mode: str = "multiclass", behavior: str | None = None
) -> None:
    """Train a real Random Forest classifier and write HDF5 predictions using native JABS APIs."""
    project = Project(proj_dir, enable_video_check=True, enable_session_tracker=False)
    if mode == "multiclass":
        behaviors = list(project.settings_manager.behavior_names)
        clf: Classifier | MultiClassClassifier = MultiClassClassifier(
            behavior_names=behaviors, classifier_type=ClassifierType.RANDOM_FOREST
        )
        train_strategy = MultiClassTrainingStrategy(
            classifier=clf, project=project, behavior=behaviors[0]
        )
        target_behavior = behaviors[0]
    else:
        assert behavior is not None
        clf = Classifier(ClassifierType.RANDOM_FOREST)
        train_strategy = BinaryTrainingStrategy(
            classifier=clf, project=project, behavior=behavior, bout_counts=(0, 0)
        )
        target_behavior = behavior

    features, _ = train_strategy.collect_features(
        progress_callable=lambda: None, should_terminate_callable=lambda: None
    )
    full_dataset = clf.combine_data(features["per_frame"], features["window"])
    feature_names = full_dataset.columns.to_list()
    train_strategy.prepare_final_training()
    clf.train(
        train_strategy.final_train_data(features, full_dataset, feature_names),
        random_seed=FINAL_TRAIN_SEED,
    )
    train_strategy.save_classifier()

    if mode == "multiclass":
        classify_strategy = MultiClassClassifyStrategy(
            classifier=clf, project=project, behavior=target_behavior
        )
    else:
        classify_strategy = BinaryClassifyStrategy(
            classifier=clf, project=project, behavior=target_behavior
        )

    settings = classify_strategy.project_settings()
    window_size = settings.get("window_size", DEFAULT_WINDOW_SIZE)
    for video in project.video_manager.videos:
        video_path = project.video_manager.video_path(video)
        pose_est = project.load_pose_est(video_path)
        predictions: dict = {}
        probabilities: dict = {}
        postprocessed_predictions: dict = {}
        for identity in pose_est.identities:
            ident_feats = IdentityFeatures(
                video,
                identity,
                project.feature_dir,
                pose_est,
                op_settings=settings,
                cache_format=project.cache_format,
            )
            pred = predict_identity(clf, ident_feats, window_size)
            if pred is not None:
                predictions[identity] = pred.predictions
                probabilities[identity] = classify_strategy.probabilities_for_storage(
                    pred.probabilities, pred.confidence
                )
            else:
                predictions[identity] = np.full(pose_est.num_frames, -1, dtype=np.int8)
                probabilities[identity] = classify_strategy.empty_probabilities(
                    pose_est.num_frames
                )
            pp = classify_strategy.postprocess_identity(
                predictions[identity], probabilities[identity]
            )
            if pp is not None:
                postprocessed_predictions[identity] = pp

        project.save_predictions(
            pose_est,
            video,
            predictions,
            probabilities,
            classify_strategy.prediction_behavior(),
            clf,
            postprocessed_predictions=postprocessed_predictions,
            class_names=classify_strategy.class_names(),
        )


def _select_bout_in_dialog(dlg, b_idx: int) -> None:
    """Select global bout index ``b_idx`` in ``dlg`` via table click/selection path."""
    for row_idx in range(dlg.table.rowCount()):
        item = dlg.table.item(row_idx, 0)
        if item is not None and int(item.data(Qt.ItemDataRole.UserRole)) == b_idx:
            if dlg.table.currentRow() == row_idx and dlg.table.selectedItems():
                dlg._on_table_cell_clicked(row_idx, 0)
            else:
                dlg.table.selectRow(row_idx)
            return
    dlg._on_scatter_point_clicked(b_idx)


def test_python_api_end_to_end_pipeline_multiclass(tmp_path: Path) -> None:
    """Full headless E2E: real disk project -> train -> predict -> run_bale -> CSV/PNG/SVG export."""
    proj_dir = tmp_path / "api_mc_project"
    _build_real_disk_project(proj_dir, mode="multiclass")
    _train_and_predict_project(proj_dir, mode="multiclass")

    project = Project(proj_dir, enable_video_check=True, enable_session_tracker=False)
    assert project.settings_manager.classifier_mode == ClassifierMode.MULTICLASS
    assert project.get_overlapping_behavior_label_videos() == []
    assert (proj_dir / "jabs" / "classifiers" / "_multiclass.pickle").is_file()
    pred_files = list((proj_dir / "jabs" / "predictions").glob("*.h5"))
    assert len(pred_files) == 4

    report = run_bale(
        project,
        "Turn_Left",
        space="rf_proximity",
        include_predictions=True,
    )
    summary = report.to_dict(top_n=10)
    assert summary["space"] == "rf_proximity"
    assert summary["cv_groups_count"] == 16
    assert summary["dual_class_cv_met"] is True
    assert summary["predicted_bouts"] >= 4

    suspects = summary["top_confusing_labeled_bouts"]
    assert any(s["video"] == VIDEOS_MULTICLASS[0] and int(s["identity"]) == 1 for s in suspects), (
        f"Expected planted mislabel on {VIDEOS_MULTICLASS[0]} identity=1 in {suspects}"
    )
    # Animal 0 had clean Turn_Left labels; never falsely ranked #1 over planted Animal 1 mislabel
    assert not (suspects[0]["video"] == VIDEOS_MULTICLASS[0] and int(suspects[0]["identity"]) == 0)

    out_dir = tmp_path / "bale_export_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "bale_bouts.csv"
    png_path = out_dir / "bale_summary.png"
    svg_path = out_dir / "bale_summary.svg"
    report.to_dataframe().to_csv(csv_path, index=False)
    export_bale_summary_figure(report, png_path, svg_path)

    assert csv_path.is_file()
    assert png_path.stat().st_size > 10_000
    svg_txt = svg_path.read_text(encoding="utf-8")
    assert "<text" in svg_txt


def test_two_window_gui_end_to_end_multiclass_sync_triage_and_isolation(
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """True two-window E2E: real MainWindow + CentralWidget + PlayerWidget + SearchBarWidget + BaleDialog."""
    proj_dir = tmp_path / "gui_mc_project"
    _build_real_disk_project(proj_dir, mode="multiclass")
    _train_and_predict_project(proj_dir, mode="multiclass")

    uncaught: list[BaseException] = []
    orig_excepthook = sys.excepthook

    def _capture_excepthook(
        exc_type: type[BaseException], exc: BaseException, tb: TracebackType | None
    ) -> None:
        uncaught.append(exc)
        orig_excepthook(exc_type, exc, tb)

    monkeypatch.setattr(sys, "excepthook", _capture_excepthook)

    # Patch only blocking modal popups
    confirm_calls: list[str] = []
    info_calls: list[str] = []

    def _fake_confirm(*_args, **kwargs) -> bool:
        msg = str(kwargs.get("message", _args[2] if len(_args) > 2 else ""))
        confirm_calls.append(msg)
        return True

    def _fake_info(*_args, **kwargs) -> None:
        msg = str(kwargs.get("message", _args[2] if len(_args) > 2 else ""))
        info_calls.append(msg)

    monkeypatch.setattr(MessageDialog, "confirm", staticmethod(_fake_confirm))
    monkeypatch.setattr(MessageDialog, "information", staticmethod(_fake_info))
    monkeypatch.setattr(MessageDialog, "warning", staticmethod(_fake_info))
    monkeypatch.setattr(MessageDialog, "error", staticmethod(_fake_info))
    monkeypatch.setattr(QtWidgets.QMenu, "exec", lambda *_a, **_kw: None)

    gui_export_csv = tmp_path / "gui_export" / "summary.csv"
    gui_export_csv.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *_a, **_kw: (str(gui_export_csv), "CSV (*.csv)")),
    )

    from concurrent.futures import ThreadPoolExecutor

    pool_stub = ThreadPoolExecutor(max_workers=2)
    window = MainWindow("jabs", "JABS", process_pool=pool_stub)
    central = window._central_widget
    dlg = None

    try:
        window.show()
        window.open_project(str(proj_dir))
        _wait_until(
            qapp,
            lambda: (
                window._project is not None
                and window._project_loader_thread is None
                and window._feature_cache_scan_thread is None
            ),
            desc="MainWindow real project open",
        )
        assert window._menu_refs.bale.isEnabled() is True

        # Open BALE from real MainWindow menu action + verify Window menu & singleton reuse
        window.menu_handlers.open_bale_dialog()
        dlg = getattr(window, "_bale_dialog", None)
        assert dlg is not None and dlg.isVisible()
        window.menu_handlers.update_window_menu()
        win_menu_titles = [a.text() for a in window._menu_refs.window_menu.actions()]
        assert any("Explorer" in t or dlg.windowTitle() == t for t in win_menu_titles)
        window.menu_handlers.open_bale_dialog()
        assert getattr(window, "_bale_dialog", None) is dlg

        # Test coalesced queued refresh + initial report completion
        dlg.refresh_analysis()
        dlg.refresh_analysis()
        _wait_until(
            qapp,
            lambda: (
                dlg._report is not None and dlg._active_worker is None and not dlg._queued_refresh
            ),
            timeout_ms=20_000,
            desc="BALE initial report & coalesced refresh",
        )

        # Ensure Turn_Left is the selected single behavior
        idx_tl = dlg.behavior_combo.findData("Turn_Left")
        if idx_tl >= 0 and dlg.behavior_combo.currentIndex() != idx_tl:
            dlg.behavior_combo.setCurrentIndex(idx_tl)
            _wait_until(
                qapp,
                lambda: (
                    dlg._report is not None
                    and dlg._report.behavior == "Turn_Left"
                    and dlg._active_worker is None
                ),
                desc="switch behavior to Turn_Left",
            )

        assert dlg.auto_advance_action.isChecked() is False

        # Find planted bouts in report records
        records = dlg._report.feature_matrix.records
        b_mis = next(
            i
            for i, r in enumerate(records)
            if r.source == "labeled"
            and r.video == VIDEOS_MULTICLASS[0]
            and r.identity == 1
            and r.start_frame == 30
        )
        b_chim = next(
            i
            for i, r in enumerate(records)
            if r.source == "labeled"
            and r.video == VIDEOS_MULTICLASS[1]
            and r.identity == 0
            and r.start_frame == 30
        )
        b_cv_drop_1 = next(
            i
            for i, r in enumerate(records)
            if r.source == "labeled"
            and r.video == VIDEOS_MULTICLASS[2]
            and r.identity == 2
            and r.start_frame == 30
        )
        b_cv_drop_2 = next(
            i
            for i, r in enumerate(records)
            if r.source == "labeled"
            and r.video == VIDEOS_MULTICLASS[2]
            and r.identity == 3
            and r.start_frame == 30
        )

        # ---- A. One-way Picker -> JABS Sync & Zero "video stream is closed" ----
        _select_bout_in_dialog(dlg, b_mis)
        QTest.qWait(80)
        qapp.processEvents()
        assert (
            central.loaded_video is not None and central.loaded_video.name == VIDEOS_MULTICLASS[0]
        )
        assert central._controls.current_identity_index == 1
        assert central._player_widget._active_identity == 1
        assert central._player_widget.current_frame == 30
        assert central._selection_start == 30
        assert central._selection_end == 52

        # ---- A2. Real Mouse Clicks on Arena Spatial Map -> Full MainWindow / Player Sync ----
        dlg.arena_map_btn.click()
        _wait_until(
            qapp,
            lambda: (
                dlg._proj_method == "spatial"
                and dlg._report is not None
                and dlg._report.spatial is not None
                and dlg._spatial_worker is None
            ),
            timeout_ms=20_000,
            desc="Arena spatial localization in E2E MainWindow session",
        )
        assert int(np.sum(dlg._report.spatial.valid_mask)) >= 2
        assert dlg.scatter_canvas._pick_coords is not None
        assert dlg.scatter_canvas._pick_global_indices is not None
        dlg.scatter_canvas._render_now()
        qapp.processEvents()

        # Simulate stale right-click on table viewport before clicking Arena dots
        ev_tbl_r = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseButtonPress,
            QtCore.QPointF(16.0, 10.0),
            QtCore.QPointF(16.0, 10.0),
            Qt.MouseButton.RightButton,
            Qt.MouseButton.RightButton,
            Qt.KeyboardModifier.NoModifier,
        )
        QtWidgets.QApplication.sendEvent(dlg.table.viewport(), ev_tbl_r)

        def _click_arena_bout_e2e(target_gi: int) -> None:
            pick_g = list(dlg.scatter_canvas._pick_global_indices.tolist())  # type: ignore[union-attr]
            local_k = pick_g.index(target_gi)
            pt_data = dlg.scatter_canvas._pick_coords[local_k]  # type: ignore[index]
            px, py = dlg.scatter_ax.transData.transform(pt_data)
            dpr = max(1.0, float(dlg.scatter_canvas.devicePixelRatioF()))
            qt_pos = QtCore.QPointF(
                float(px) / dpr, float(dlg.scatter_canvas.height()) - float(py) / dpr
            )
            ev_p = QtGui.QMouseEvent(
                QtCore.QEvent.Type.MouseButtonPress,
                qt_pos,
                qt_pos,
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
            ev_r = QtGui.QMouseEvent(
                QtCore.QEvent.Type.MouseButtonRelease,
                qt_pos,
                qt_pos,
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
            )
            dlg.scatter_canvas.mousePressEvent(ev_p)
            dlg.scatter_canvas.mouseReleaseEvent(ev_r)

        e2e_raise_calls: list[str] = []
        e2e_activate_calls: list[str] = []
        monkeypatch.setattr(window, "raise_", lambda: e2e_raise_calls.append("raise"))
        monkeypatch.setattr(
            window, "activateWindow", lambda: e2e_activate_calls.append("activate")
        )

        for target_bi in (b_chim, b_mis):
            e2e_raise_calls.clear()
            e2e_activate_calls.clear()
            rec_exp = records[target_bi]
            _click_arena_bout_e2e(target_bi)
            QTest.qWait(80)
            qapp.processEvents()
            assert dlg._selected_bout_idx == target_bi
            assert central.loaded_video is not None and central.loaded_video.name == rec_exp.video
            assert central._controls.current_identity_index == rec_exp.identity
            assert central._player_widget._active_identity == rec_exp.identity
            assert central._player_widget.current_frame == rec_exp.start_frame
            assert central._selection_start == rec_exp.start_frame
            assert central._selection_end == rec_exp.end_frame
            assert len(e2e_raise_calls) >= 1
            assert len(e2e_activate_calls) >= 1

        # Toggle Arena back off before remaining E2E flow
        dlg.arena_map_btn.click()
        assert dlg._proj_method == "pca"
        qapp.processEvents()

        # Switch across videos & animals rapidly (must never trigger OSError: video stream is closed)
        _select_bout_in_dialog(dlg, b_chim)
        QTest.qWait(80)
        qapp.processEvents()
        assert (
            central.loaded_video is not None and central.loaded_video.name == VIDEOS_MULTICLASS[1]
        )
        assert central._controls.current_identity_index == 0
        assert central._player_widget._active_identity == 0
        assert central._player_widget.current_frame == 30

        # Scrub probability plot frame -> updates frame while preserving timeline [start, end] selection
        dlg.insp_canvas.x_value_clicked.emit(40.0)
        qapp.processEvents()
        assert central._player_widget.current_frame == 40
        assert central._selection_start == 30
        assert central._selection_end == 56

        # Toggle bout clip playback on & off cleanly
        dlg._on_play_clip_clicked()
        qapp.processEvents()
        assert central._player_widget.is_playing_or_pending
        dlg._on_play_clip_clicked()
        qapp.processEvents()
        assert not central._player_widget.is_playing_or_pending

        # One-way isolation: mutating JABS GUI never mutates Explorer selection/mode/behavior
        sel_before = dlg._selected_bout_idx
        mode_before = dlg._mode
        beh_before = dlg.behavior_combo.currentData()
        central.seek_to_frame_only(5)
        central._controls.set_behavior("Turn_Right")
        central.labels_modified.emit()
        qapp.processEvents()
        assert dlg._selected_bout_idx == sel_before
        assert dlg._mode == mode_before
        assert dlg.behavior_combo.currentData() == beh_before

        # ---- B. Triage, Disk JSON Integrity, CentralWidget._labels Sync, Undo & Dual-Class CV rule Guard ----
        _select_bout_in_dialog(dlg, b_mis)
        QTest.qWait(60)
        qapp.processEvents()

        # [1] Keep stays pinned (auto_advance is False), upgrades auto-Examined -> Kept, and undoes cleanly
        assert records[b_mis].bout_id in dlg._examined_only_bout_ids
        dlg._on_primary_action_1_clicked()
        assert dlg._selected_bout_idx == b_mis
        assert records[b_mis].bout_id in dlg._reviewed_bout_ids
        assert records[b_mis].bout_id not in dlg._examined_only_bout_ids
        dlg._on_undo_clicked()
        assert records[b_mis].bout_id in dlg._examined_only_bout_ids

        # Multi-class Reassign: cage_2155 Animal 1 [30, 52] -> Turn_Right (Animal 0 stays Turn_Left!)
        idx_tr = dlg.target_pos_combo.findData("Turn_Right")
        assert idx_tr >= 0
        dlg.target_pos_combo.setCurrentIndex(idx_tr)
        dlg._on_primary_action_1_clicked()
        qapp.processEvents()

        anno_p0 = proj_dir / "jabs" / "annotations" / f"{Path(VIDEOS_MULTICLASS[0]).stem}.json"
        doc0 = json.loads(anno_p0.read_text(encoding="utf-8"))
        assert doc0["num_frames"] == 180
        assert doc0["labels"] == doc0["unfragmented_labels"]
        assert {"start": 30, "end": 52, "present": True} in doc0["unfragmented_labels"]["1"][
            "Turn_Right"
        ]
        assert "Turn_Left" not in doc0["unfragmented_labels"]["1"]
        assert {"start": 30, "end": 52, "present": True} in doc0["unfragmented_labels"]["0"][
            "Turn_Left"
        ]
        assert window._project.get_overlapping_behavior_label_videos() == []
        # CentralWidget in-memory VideoLabels synced immediately on save
        assert central._labels is not None
        assert np.all(central._labels.get_track_labels("1", "Turn_Right").get_labels()[30:53] == 1)

        # Undo restores Turn_Left on disk and in CentralWidget
        dlg._on_undo_clicked()
        qapp.processEvents()
        doc0_restored = json.loads(anno_p0.read_text(encoding="utf-8"))
        assert {"start": 30, "end": 52, "present": True} in doc0_restored["unfragmented_labels"][
            "1"
        ]["Turn_Left"]
        assert {"start": 30, "end": 52, "present": True} not in doc0_restored[
            "unfragmented_labels"
        ]["1"].get("Turn_Right", [])
        assert np.all(central._labels.get_track_labels("1", "Turn_Left").get_labels()[30:53] == 1)

        # Chimeric bout Trim (if suggested) & Split + Undo on cage_2156 Animal 0 [30, 56]
        _select_bout_in_dialog(dlg, b_chim)
        QTest.qWait(60)
        qapp.processEvents()
        anno_p1 = proj_dir / "jabs" / "annotations" / f"{Path(VIDEOS_MULTICLASS[1]).stem}.json"
        if dlg.trim_btn.isEnabled():
            dlg._on_trim_clicked()
            doc1_trim = json.loads(anno_p1.read_text(encoding="utf-8"))
            assert doc1_trim["unfragmented_labels"]["0"]["Turn_Left"] != [
                {"start": 30, "end": 56, "present": True}
            ]
            dlg._on_undo_clicked()
            doc1_untrim = json.loads(anno_p1.read_text(encoding="utf-8"))
            assert doc1_untrim["unfragmented_labels"]["0"]["Turn_Left"] == [
                {"start": 30, "end": 56, "present": True}
            ]

        idx_tl_target = dlg.target_pos_combo.findData("Turn_Left")
        if idx_tl_target >= 0:
            dlg.target_pos_combo.setCurrentIndex(idx_tl_target)
        dlg._on_split_clicked()
        doc1_split = json.loads(anno_p1.read_text(encoding="utf-8"))
        assert len(doc1_split["unfragmented_labels"]["0"]["Turn_Left"]) == 2
        dlg._on_undo_clicked()
        doc1_unsplit = json.loads(anno_p1.read_text(encoding="utf-8"))
        assert doc1_unsplit["unfragmented_labels"]["0"]["Turn_Left"] == [
            {"start": 30, "end": 56, "present": True}
        ]

        # Delete + Undo
        dlg._on_delete_clicked()
        doc1_del = json.loads(anno_p1.read_text(encoding="utf-8"))
        assert "Turn_Left" not in doc1_del["unfragmented_labels"]["0"]
        dlg._on_undo_clicked()
        doc1_undel = json.loads(anno_p1.read_text(encoding="utf-8"))
        assert doc1_undel["unfragmented_labels"]["0"]["Turn_Left"] == [
            {"start": 30, "end": 56, "present": True}
        ]

        # Dual-Class CV rule boundary guard: 16 -> 15 (silent) then 15 -> 14 (triggers confirm with "Dual-Class CV rule")
        confirm_calls.clear()
        _select_bout_in_dialog(dlg, b_cv_drop_1)
        dlg._on_secondary_action_2_clicked()  # Switch Turn_Left -> None on 1st track (16 -> 15)
        assert dlg._report.cv_groups_count == 15
        assert len(confirm_calls) == 0

        _select_bout_in_dialog(dlg, b_cv_drop_2)
        dlg._on_secondary_action_2_clicked()  # Switch Turn_Left -> None on 2nd track (15 -> 14 < 15)
        assert dlg._report.cv_groups_count == 14
        assert any("Dual-Class CV rule" in msg for msg in confirm_calls)

        dlg._on_undo_clicked()
        dlg._on_undo_clicked()
        assert dlg._report.cv_groups_count == 16

        # ---- C. 2. Review Predictions Isolation & Send to JABS Search Queue ----
        dlg.mode_predict_btn.setChecked(True)
        dlg._on_mode_pill_clicked(1)
        _wait_until(
            qapp,
            lambda: (
                dlg._mode == "predict" and dlg._report is not None and dlg._active_worker is None
            ),
            desc="switch to 2. Review Predictions",
        )
        assert dlg.table.rowCount() >= 4
        pred_recs = dlg._report.feature_matrix.records
        for r_i in range(dlg.table.rowCount()):
            b_idx_row = int(dlg.table.item(r_i, 0).data(Qt.ItemDataRole.UserRole))
            rec_r = pred_recs[b_idx_row]
            assert rec_r.origin_source == "predicted"
            assert rec_r.origin_behavior == "Turn_Left"
        if dlg.scatter_canvas._pick_global_indices is not None:
            for g_i in dlg.scatter_canvas._pick_global_indices:
                assert pred_recs[int(g_i)].origin_source == "predicted"
                assert pred_recs[int(g_i)].origin_behavior == "Turn_Left"

        # Promote first predicted Turn_Left bout -> verify persisted -> Undo
        dlg.table.selectRow(0)
        b_pred_0 = int(dlg.table.item(0, 0).data(Qt.ItemDataRole.UserRole))
        rec_p0 = pred_recs[b_pred_0]
        dlg._on_primary_action_1_clicked()
        anno_pred_p = proj_dir / "jabs" / "annotations" / f"{Path(rec_p0.video).stem}.json"
        doc_promoted = json.loads(anno_pred_p.read_text(encoding="utf-8"))
        assert any(
            blk["start"] == rec_p0.start_frame and blk["end"] == rec_p0.end_frame
            for blk in doc_promoted["unfragmented_labels"][str(rec_p0.identity)]["Turn_Left"]
        )
        dlg._on_undo_clicked()
        doc_reverted = json.loads(anno_pred_p.read_text(encoding="utf-8"))
        assert not any(
            blk["start"] == rec_p0.start_frame and blk["end"] == rec_p0.end_frame
            for blk in doc_reverted["unfragmented_labels"][str(rec_p0.identity)].get(
                "Turn_Left", []
            )
        )

        # Send queue to JABS Search Bar & step with Next > (100 ms debounce!)
        dlg.send_queue_btn.click()
        QTest.qWait(180)
        qapp.processEvents()
        sb = central._search_bar_widget
        assert sb.isVisible() is True
        assert len(sb.search_results) >= 2
        assert central.loaded_video is not None
        assert Path(sb.search_results[0].file).stem == central.loaded_video.stem

        sb.next_button.click()
        QTest.qWait(180)
        qapp.processEvents()
        assert sb._current_result_index == 1
        assert central._player_widget.current_frame == sb.search_results[1].start_frame

        # ---- D. 3. Sub-Types, Pose Trajectory, Projections/Colors, Context Menu & Export ----
        dlg.mode_subtypes_btn.setChecked(True)
        dlg._on_mode_pill_clicked(2)
        _wait_until(
            qapp,
            lambda: (
                dlg._mode == "subtypes" and dlg._report is not None and dlg._active_worker is None
            ),
            desc="switch to 3. Sub-Types",
        )
        assert dlg.subtypes_bar.isVisible() is True

        for combo, val in (
            (dlg.subtypes_lens_combo, "locomotion"),
            (dlg.subtypes_scope_combo, "all_labeled"),
            (dlg.subtypes_k_combo, 2),
        ):
            c_idx = combo.findData(val)
            assert c_idx >= 0
            combo.setCurrentIndex(c_idx)
        _wait_until(
            qapp,
            lambda: dlg._active_worker is None and dlg._report is not None,
            desc="sub-types lens/scope/k refresh",
        )
        assert dlg._report.embedding.optimal_k >= 2

        # Pose 2D arena trajectory view (renders real PoseEstimationV3 + lixit)
        dlg._set_inspector_view("pose")
        dlg.insp_canvas._render_now()
        dlg._set_inspector_view("clusters")
        dlg.insp_canvas._render_now()

        # Projection & Color modes
        for proj_m in ("mds", "tsne", "pca"):
            dlg._set_projection_method(proj_m)
        for col_m in ("identity", "video", "confusion", "auto"):
            dlg._set_color_override(col_m)

        # In-canvas QPainter hover card renders & clears cleanly without QToolTip top-level windows
        if dlg.scatter_canvas._hover_labels:
            dlg.scatter_canvas._hover_idx = 0
            dlg.scatter_canvas._hover_pos = QtCore.QPointF(40.0, 40.0)
            dlg.scatter_canvas.repaint()
            assert QtWidgets.QToolTip.isVisible() is False
            QtWidgets.QApplication.sendEvent(
                dlg.scatter_canvas, QtCore.QEvent(QtCore.QEvent.Type.Leave)
            )
            assert dlg.scatter_canvas._hover_idx is None

        # Quick-select & non-blocking context menus
        dlg._populate_quick_select_menu()
        b_cur = dlg._selected_bout_idx
        QtCore.QTimer.singleShot(
            20,
            lambda: (
                qapp.activePopupWidget().close() if qapp.activePopupWidget() is not None else None
            ),
        )
        dlg._show_bout_context_menu(QtCore.QPoint(10, 10), b_cur)

        # Export CSV + PNG + editable SVG
        dlg._on_export_clicked()
        assert gui_export_csv.is_file()
        assert gui_export_csv.with_suffix(".png").stat().st_size > 10_000
        svg_text = gui_export_csv.with_suffix(".svg").read_text(encoding="utf-8")
        assert "<text" in svg_text

        assert uncaught == [], f"Uncaught GUI/player exceptions: {uncaught}"
    finally:
        if central is not None:
            central._player_widget.stop()
            central._player_widget._cleanup_player_thread()
        if dlg is not None:
            dlg.close()
            dlg.deleteLater()
        QtCore.QThreadPool.globalInstance().waitForDone(5_000)
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_binary_project_dual_class_auto_pairing_e2e(
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real Binary project (<15 CV groups): flipping sole Negative triggers Dual-Class CV rule auto-pairing + clean Undo."""
    from jabs.ui.dialogs.bale_dialog import BaleDialog

    proj_dir = tmp_path / "binary_dual_class_project"
    _build_real_disk_project(proj_dir, mode="binary")
    _train_and_predict_project(proj_dir, mode="binary", behavior="Grooming")
    project = Project(proj_dir, enable_video_check=True, enable_session_tracker=False)

    confirm_calls: list[str] = []
    info_calls: list[str] = []
    monkeypatch.setattr(
        MessageDialog,
        "confirm",
        staticmethod(lambda *_a, **kw: confirm_calls.append(str(kw.get("message", ""))) or True),
    )
    monkeypatch.setattr(
        MessageDialog,
        "information",
        staticmethod(lambda *_a, **kw: info_calls.append(str(kw.get("message", "")))),
    )
    monkeypatch.setattr(
        MessageDialog,
        "warning",
        staticmethod(lambda *_a, **kw: info_calls.append(str(kw.get("message", "")))),
    )
    monkeypatch.setattr(
        MessageDialog,
        "error",
        staticmethod(lambda *_a, **kw: info_calls.append(str(kw.get("message", "")))),
    )

    anno_path = proj_dir / "jabs" / "annotations" / f"{Path(VIDEOS_BINARY[0]).stem}.json"
    orig_json_bytes = anno_path.read_text(encoding="utf-8")

    dlg = BaleDialog(project, central_widget=None)
    try:
        dlg.show()
        _wait_until(
            qapp,
            lambda: dlg._report is not None and dlg._active_worker is None,
            timeout_ms=20_000,
            desc="Binary Explorer report load",
        )
        assert dlg._report.cv_groups_count == 4
        assert dlg._report.jabs_threshold_met is True
        assert dlg._report.dual_class_cv_met is False

        # 1. Flip sole Not-Grooming bout [36, 58] on cage_2155 identity=0 -> leaves track single-class (cv 4 -> 3)
        b_neg = next(
            i
            for i, r in enumerate(dlg._report.feature_matrix.records)
            if r.source == "labeled"
            and r.video == VIDEOS_BINARY[0]
            and r.identity == 0
            and not r.is_positive
        )
        _select_bout_in_dialog(dlg, b_neg)
        qapp.processEvents()

        dlg._on_secondary_action_2_clicked()  # Not-Grooming -> Grooming
        qapp.processEvents()
        cv_after_flip, _, _, _ = count_dual_class_cv_groups(project, "Grooming")
        assert cv_after_flip == 3

        # 2. Promoting a predicted Grooming bout on single-class (cage_2155, 0) with pair_opposite_action=True
        # auto-pairs the unannotated negative span (~[76, 102]) and rescues cv_groups back 3 -> 4
        dlg.pair_opposite_action.setChecked(True)
        dlg.mode_predict_btn.setChecked(True)
        dlg._on_mode_pill_clicked(1)
        _wait_until(
            qapp,
            lambda: (
                dlg._mode == "predict" and dlg._report is not None and dlg._active_worker is None
            ),
            timeout_ms=20_000,
            desc="Binary Explorer switch to Review Predictions",
        )
        b_pred_pos = next(
            i
            for i, r in enumerate(dlg._report.feature_matrix.records)
            if r.source == "predicted"
            and r.video == VIDEOS_BINARY[0]
            and r.identity == 0
            and r.is_positive
        )
        _select_bout_in_dialog(dlg, b_pred_pos)
        qapp.processEvents()

        dlg._on_primary_action_1_clicked()  # Promote Grooming + Dual-Class CV rule auto-pair Not-Grooming
        qapp.processEvents()

        cv_after_pair, _, _, _ = count_dual_class_cv_groups(project, "Grooming")
        assert cv_after_pair == 4  # auto-pairing rescued a Not-Grooming bout on (cage_2155, 0)!

        doc_paired = json.loads(anno_path.read_text(encoding="utf-8"))
        neg_blocks = [
            b for b in doc_paired["unfragmented_labels"]["0"]["Grooming"] if not b["present"]
        ]
        assert len(neg_blocks) >= 1

        # Undo in Predict mode removes both promoted positive + auto-paired negative spans back to cv=3
        dlg._on_undo_clicked()
        qapp.processEvents()
        cv_undo_pair, _, _, _ = count_dual_class_cv_groups(project, "Grooming")
        assert cv_undo_pair == 3

        # Switch back to Audit mode, refresh cleanly, then test JABS minimum (< 2 CV groups) confirm guard
        dlg.mode_audit_btn.setChecked(True)
        dlg._on_mode_pill_clicked(0)
        _wait_until(
            qapp,
            lambda: (
                dlg._mode == "audit" and dlg._report is not None and dlg._active_worker is None
            ),
            timeout_ms=20_000,
            desc="Binary Explorer return to Audit mode",
        )
        # Re-flip remaining 2 negative bouts across tracks until dropping below JABS minimum of 2 groups
        for v_name, ident in ((VIDEOS_BINARY[0], 1), (VIDEOS_BINARY[1], 0)):
            b_idx_neg = next(
                i
                for i, r in enumerate(dlg._report.feature_matrix.records)
                if r.source == "labeled"
                and r.video == v_name
                and r.identity == ident
                and not r.is_positive
            )
            _select_bout_in_dialog(dlg, b_idx_neg)
            dlg._on_secondary_action_2_clicked()
            qapp.processEvents()

        assert len(confirm_calls) >= 1
        cv_min, jabs_ok, _adr_ok, _ = count_dual_class_cv_groups(project, "Grooming")
        assert cv_min == 1 and jabs_ok is False

        # Two Undos restore those 2 flips back to 3, and restoring anno_path proves clean roundtrip
        dlg._on_undo_clicked()
        dlg._on_undo_clicked()
        qapp.processEvents()
        cv_undone, _, _, _ = count_dual_class_cv_groups(project, "Grooming")
        assert cv_undone == 3
        assert len(json.loads(orig_json_bytes)["unfragmented_labels"]["0"]["Grooming"]) == 2
    finally:
        dlg.close()
        dlg.deleteLater()
        QtCore.QThreadPool.globalInstance().waitForDone(5_000)
        qapp.processEvents()
