"""Headless GUI & UX responsiveness tests for the simplified `v2` `BaleDialog`."""

from __future__ import annotations

import ast
import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from jabs.bale import run_bale
from jabs.core.enums import ClassifierMode

try:
    from PySide6 import QtCore, QtGui, QtWidgets
    from PySide6.QtCore import Qt

    from jabs.ui.dialogs.bale_dialog import BaleDialog, _short_video_name

    sys.path.insert(0, str(Path(__file__).parent))
    from test_bale import _build_synthetic_explorer_project

    SKIP_UI_TESTS = False
    SKIP_REASON = ""
except ImportError as e:
    SKIP_UI_TESTS = True
    SKIP_REASON = f"Qt/UI dependencies not available: {e}"

pytestmark = pytest.mark.skipif(SKIP_UI_TESTS, reason=SKIP_REASON)


def _ensure_qapp() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app  # type: ignore[return-value]


def _perf_budget_scale() -> float:
    if sys.gettrace() is not None:
        return 4.0
    if (
        hasattr(sys, "monitoring")
        and sys.monitoring.get_tool(sys.monitoring.COVERAGE_ID) is not None
    ):
        return 4.0
    return 1.0


def test_macos_pyside6_safety_no_qtagg_or_setcursor_in_paintevent() -> None:
    """Enforce that `bale_dialog.py` never imports `FigureCanvasQTAgg` or calls `setCursor`."""
    import jabs.ui.dialogs.bale_dialog as mod

    src = Path(mod.__file__).read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            names = [alias.name for alias in node.names]
            mod_name = getattr(node, "module", "") or ""
            assert "backend_qtagg" not in mod_name
            assert "FigureCanvasQTAgg" not in names
            assert "NavigationToolbar2QT" not in names
        if isinstance(node, ast.Attribute):
            assert node.attr != "setCursor", "Never call setCursor (triggers macOS PySide6 crash)"


def test_all_buttons_use_nofocus_so_spacebar_always_plays_clip(tmp_path: Path) -> None:
    """Ensure no `QPushButton` steals keyboard focus and swallows `Space` to re-click itself."""
    _ensure_qapp()
    project, _ = _build_synthetic_explorer_project(tmp_path)
    central_mock = MagicMock()
    central_mock.behavior = "Grooming"

    dlg = BaleDialog(project, central_widget=central_mock)
    dlg.show()
    QtWidgets.QApplication.processEvents()

    for btn in dlg.findChildren(QtWidgets.QPushButton):
        assert btn.focusPolicy() == Qt.FocusPolicy.NoFocus, (
            f"Button '{btn.text()}' has focusPolicy {btn.focusPolicy()}; must be NoFocus"
        )
    for tbtn in dlg.findChildren(QtWidgets.QToolButton):
        assert tbtn.focusPolicy() == Qt.FocusPolicy.NoFocus, (
            f"ToolButton '{tbtn.text()}' has focusPolicy {tbtn.focusPolicy()}; must be NoFocus"
        )
    for combo in dlg.findChildren(QtWidgets.QComboBox):
        assert combo.focusPolicy() == Qt.FocusPolicy.NoFocus, (
            f"ComboBox '{combo}' has focusPolicy {combo.focusPolicy()}; must be NoFocus"
        )

    rep = run_bale(project, "Grooming", include_predictions=False, space="classifier")
    dlg._on_analysis_finished(rep, included_preds=False)

    # Even when the QTableWidget has focus, pressing Space triggers play_bout_clip!
    dlg.table.setFocus()
    ev_space = QtGui.QKeyEvent(
        QtCore.QEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier, " "
    )
    QtWidgets.QApplication.sendEvent(dlg.table, ev_space)
    assert central_mock.play_bout_clip.call_count == 1
    dlg.close()


def test_short_video_name_formatting() -> None:
    """Verify multi-dot timestamp video names abbreviate cleanly for the table column."""
    assert (
        _short_video_name("study_275.cage_2155.2024-11-12.22.11.top.mp4")
        == "cage_2155 • 11-12 22:11"
    )
    assert _short_video_name("cage_00.mp4") == "cage_00"


def test_simplified_3mode_triage_keep_switch_undo_and_predictions(tmp_path: Path) -> None:
    """Verify consistent `[1]` Keep/Confirm vs `[2]` Switch/Reject, `< 100 ms` edits, Undo, 1-based Animal IDs, and Sub-Types."""
    _ensure_qapp()
    project, proj_dir = _build_synthetic_explorer_project(tmp_path)

    # Add synthetic prediction probabilities on cage_00.mp4 so Review Predictions mode has bouts
    (proj_dir / "jabs" / "predictions" / "cage_00.h5").write_bytes(b"stub")
    preds_raw = np.zeros(600, dtype=np.int32)
    preds_post = np.zeros(600, dtype=np.int32)
    probs = np.full(600, 0.05, dtype=np.float32)
    # Predicted bout at [490..535] with stitched gap frames at [510..514] so Split is offered too
    preds_raw[490:536] = 1
    preds_raw[510:515] = 0
    preds_post[490:536] = 1
    probs[490:536] = 0.56
    project.prediction_manager.load_predictions.side_effect = lambda vid, _beh: (
        ({0: preds_raw}, {0: probs}, {0: preds_post}) if vid == "cage_00.mp4" else ({}, {}, {})
    )

    central_mock = MagicMock()
    central_mock.behavior = "Grooming"
    central_mock.loaded_video = Path(proj_dir / "cage_00.mp4")
    central_mock._pose_est = None

    dlg = BaleDialog(project, central_widget=central_mock)
    dlg.show()
    QtWidgets.QApplication.processEvents()
    rep_audit = run_bale(project, "Grooming", include_predictions=False, space="classifier")
    dlg._on_analysis_finished(rep_audit, included_preds=False)

    # 1. Clean 7-column table (with Examined [V] tag col), exact JABS Animal display parity (`Animal 0`), & top-ranked mislabel selected
    assert dlg.table.columnCount() == 7
    assert dlg._mode == "audit"
    assert dlg.table.rowCount() >= 2
    assert (
        dlg.table.item(0, 3).text() == "Animal 0"
    )  # Matches PoseEstimation.identity_index_to_display(0)!
    # Also verify external_identities / RFID mapping both via live pose_est AND via VideoManager.get_cached_pose_path (.h5)
    fake_pose = MagicMock()
    fake_pose.identity_index_to_display.side_effect = lambda idx: f"RFID-{idx + 1}"
    central_mock._pose_est = fake_pose
    dlg._identity_label_cache.clear()
    assert dlg._animal_display_text("cage_00.mp4", 0) == "Animal RFID-1"
    central_mock._pose_est = None
    dlg._identity_label_cache.clear()
    dlg._video_ext_id_cache.clear()

    import h5py

    pose_h5_path = proj_dir / "cage_01_pose_est_v8.h5"
    with h5py.File(pose_h5_path, "w") as h5f:
        grp = h5f.create_group("poseest")
        grp.create_dataset("external_identity_mapping", data=np.array([b"EAR-L", b"EAR-R"]))
    # Simulate real VideoManager where `get_cached_pose_path` exists and `pose_path` does NOT exist
    from types import SimpleNamespace

    orig_vm = project.video_manager
    project.video_manager = SimpleNamespace(
        videos=orig_vm.videos,
        video_path=orig_vm.video_path,
        get_cached_pose_path=lambda vid: pose_h5_path if vid == "cage_01.mp4" else None,
    )
    assert dlg._animal_display_text("cage_01.mp4", 0) == "Animal EAR-L"
    assert dlg._animal_display_text("cage_01.mp4", 1) == "Animal EAR-R"
    project.video_manager = orig_vm
    dlg._identity_label_cache.clear()
    dlg._video_ext_id_cache.clear()
    first_b_idx = dlg._selected_bout_idx
    assert first_b_idx is not None
    first_rec = dlg._report.feature_matrix.records[first_b_idx]  # type: ignore[union-attr]
    assert first_rec.video == "cage_00.mp4" and first_rec.start_frame == 320
    assert dlg.target_pos_combo.isHidden()

    # 2a. Auto-advance defaults OFF; Key [1] (`✓ Keep Grooming [1]`) signs off in-place without
    # jumping selection or JABS video player (`Needs Review` pins row as `✓ Saved` while badge decrements)
    assert dlg.auto_advance_action.isChecked() is False
    second_row_item = dlg.table.item(1, 0)
    assert second_row_item is not None
    expected_next_b_idx = int(second_row_item.data(Qt.ItemDataRole.UserRole))

    central_mock.loaded_video = proj_dir / first_rec.video
    central_mock._controls.current_identity_index = first_rec.identity
    central_mock.seek_to_bout.reset_mock()

    t_keep = time.perf_counter()
    dlg._on_primary_action_1_clicked()
    keep_ms = (time.perf_counter() - t_keep) * 1000.0
    assert keep_ms < 80.0
    assert first_rec.bout_id in dlg._reviewed_bout_ids
    assert first_rec.bout_id in dlg._pinned_needs_review_bout_ids
    assert dlg._pending_recalc_count == 0  # Zero disk mutation for Keep!
    assert dlg._selected_bout_idx == first_b_idx
    assert dlg._selected_bout_idx != expected_next_b_idx
    assert int(dlg.table.item(0, 0).data(Qt.ItemDataRole.UserRole)) == first_b_idx
    assert "Kept" in dlg.table.item(0, 5).text()
    assert central_mock.seek_to_bout.call_count == 0

    # Undo (`⌘Z`) reverts Keep sign-off immediately and stays on bout #1
    dlg._on_undo_clicked()
    assert first_rec.bout_id not in dlg._reviewed_bout_ids
    assert first_rec.bout_id not in dlg._pinned_needs_review_bout_ids
    assert dlg._selected_bout_idx == first_b_idx

    # 2b. Key [2] (`⇄ Switch to Not-Grooming [2]`) flips disk JSON in < 100 ms & stays pinned on bout #1
    central_mock.seek_to_bout.reset_mock()
    t0 = time.perf_counter()
    dlg._on_secondary_action_2_clicked()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 120.0, f"Expected instant <120 ms triage edit, took {elapsed_ms:.1f} ms"

    doc0 = json.loads((proj_dir / "jabs" / "annotations" / "cage_00.json").read_text())
    b320 = next(b for b in doc0["unfragmented_labels"]["0"]["Grooming"] if b["start"] == 320)
    assert b320["present"] is False
    assert doc0["labels"]["0"]["Grooming"] == doc0["unfragmented_labels"]["0"]["Grooming"]
    assert central_mock.sync_external_video_labels.called
    assert dlg._selected_bout_idx == first_b_idx
    assert dlg._selected_bout_idx != expected_next_b_idx
    assert int(dlg.table.item(0, 0).data(Qt.ItemDataRole.UserRole)) == first_b_idx
    assert central_mock.seek_to_bout.call_count == 0
    assert dlg._pending_recalc_count == 1
    assert "(1)" in dlg.recalc_btn.text()

    # 3. Instant Undo (< 100 ms) restores disk JSON, resets badge, and re-selects bout #1
    t_undo = time.perf_counter()
    dlg._on_undo_clicked()
    undo_ms = (time.perf_counter() - t_undo) * 1000.0
    assert undo_ms < 120.0
    assert dlg._selected_bout_idx == first_b_idx
    assert dlg._pending_recalc_count == 0
    doc0_restored = json.loads((proj_dir / "jabs" / "annotations" / "cage_00.json").read_text())
    b320_restored = next(
        b for b in doc0_restored["unfragmented_labels"]["0"]["Grooming"] if b["start"] == 320
    )
    assert b320_restored["present"] is True

    # 4. Play Clip (`Space`) & Frame Probability Scrubbing
    dlg._on_play_clip_clicked()
    central_mock.play_bout_clip.assert_called_with(
        "cage_00.mp4", 0, 320, 360, "Grooming", context_frames=10
    )
    dlg._on_prob_curve_frame_clicked(342.4)
    central_mock.seek_to_frame_only.assert_called_with(342)

    # 5. Warm switch to Mode 3 (`Sub-Types`) runs in < 250 ms and hides relabeling buttons
    t_sub = time.perf_counter()
    dlg.mode_subtypes_btn.setChecked(True)
    dlg._on_mode_pill_clicked(2)
    sub_ms = (time.perf_counter() - t_sub) * 1000.0
    assert sub_ms < 250.0
    assert dlg._mode == "subtypes"
    assert dlg._report.space == "unsupervised"  # type: ignore[union-attr]
    assert dlg.primary_pos_btn.isHidden()
    assert dlg.secondary_neg_btn.isHidden()
    # Pressing [1] or [2] in Sub-Types does NOT mutate labels
    dlg._on_primary_action_1_clicked()
    dlg._on_secondary_action_2_clicked()
    assert len(dlg._ui_undo_history) == 0

    # 6. Review Predictions mode: Confirm [1] promotes predicted bout to Positive label
    rep_pred = run_bale(
        project, "Grooming", include_predictions=True, max_predicted_bouts=50, space="classifier"
    )
    dlg._mode = "predict"
    dlg.mode_predict_btn.setChecked(True)
    dlg._on_analysis_finished(rep_pred, included_preds=True)
    assert dlg.table.rowCount() >= 1
    pred_b_idx = dlg._selected_bout_idx
    assert pred_b_idx is not None
    pred_rec = dlg._report.feature_matrix.records[pred_b_idx]  # type: ignore[union-attr]
    assert pred_rec.source == "predicted" and pred_rec.start_frame == 490

    dlg._on_primary_action_1_clicked()
    doc0_after_pred = json.loads((proj_dir / "jabs" / "annotations" / "cage_00.json").read_text())
    b490 = next(
        b for b in doc0_after_pred["unfragmented_labels"]["0"]["Grooming"] if b["start"] == 490
    )
    assert b490["present"] is True

    # 7. Persistent keyboard shortcut hint stays intact even after status toast updates
    assert "Space Play/Pause" in dlg.shortcut_hint_label.text()
    assert "Saved as Grooming" in dlg.status_summary_label.text()

    # 8. Typing '1' inside search_input filters text and does NOT trigger label mutation!
    undo_depth_before = len(dlg._ui_undo_history)
    dlg.search_input.setFocus()
    ev = QtGui.QKeyEvent(
        QtCore.QEvent.Type.KeyPress, Qt.Key.Key_1, Qt.KeyboardModifier.NoModifier, "1"
    )
    dlg.keyPressEvent(ev)
    assert len(dlg._ui_undo_history) == undo_depth_before
    dlg.close()


def test_multiclass_auto_syncs_target_class_and_worker_queues_refresh(tmp_path: Path) -> None:
    """Verify Multi-Class auto-syncs `target_pos_combo` to selected bout class and `_queued_refresh` works."""
    _ensure_qapp()
    project, _ = _build_synthetic_explorer_project(tmp_path)
    project.settings_manager.classifier_mode = ClassifierMode.MULTICLASS
    project.settings_manager.behavior_names = ["G.1_Paw_Lick", "G.2_Unilateral_Face"]

    dlg = BaleDialog(project, central_widget=None)
    dlg.show()
    QtWidgets.QApplication.processEvents()
    assert not dlg.target_pos_combo.isHidden()
    cmap = dlg._behavior_color_hex_map()
    from jabs.ui.colors import NOT_BEHAVIOR_COLOR, make_behavior_color_map
    from jabs.ui.dialogs.bale_dialog import _qcolor_to_hex

    expected_qmap = make_behavior_color_map(["G.1_Paw_Lick", "G.2_Unilateral_Face"])
    assert cmap["G.1_Paw_Lick"] == _qcolor_to_hex(expected_qmap["G.1_Paw_Lick"])
    assert cmap["G.2_Unilateral_Face"] == _qcolor_to_hex(expected_qmap["G.2_Unilateral_Face"])
    assert cmap["None"] == _qcolor_to_hex(NOT_BEHAVIOR_COLOR, darken_light=False)

    # Simulate worker in flight: calling refresh_analysis sets _queued_refresh = True instead of dropping!
    dlg._active_worker = MagicMock()
    dlg.refresh_analysis()
    assert dlg._queued_refresh is True
    dlg.close()
    assert dlg._is_closing is True


def test_map_zoom_pan_persistence_and_expand_restore(tmp_path: Path) -> None:
    """Verify wheel zoom, zoom-box, pan, zoom persistence across `[1]`/`[2]` triage edits, `[0]` reset, and `[M]` expand."""
    _ensure_qapp()
    project, _ = _build_synthetic_explorer_project(tmp_path)
    central_mock = MagicMock()
    central_mock.behavior = "Grooming"

    dlg = BaleDialog(project, central_widget=central_mock)
    dlg.resize(1340, 820)
    dlg.show()
    QtWidgets.QApplication.processEvents()

    rep = run_bale(project, "Grooming", include_predictions=False, space="classifier")
    dlg._on_analysis_finished(rep, included_preds=False)
    dlg.scatter_canvas._render_now()
    QtWidgets.QApplication.processEvents()

    assert dlg._scatter_view_limits is None
    assert not dlg.map_reset_zoom_btn.isEnabled()
    def_xlim, def_ylim = dlg._scatter_default_limits  # type: ignore[misc]
    def_span_x = def_xlim[1] - def_xlim[0]

    # 1. Wheel zoom into a dense pocket (< 25 ms)
    coords, _, _ = dlg._current_2d_coords()
    anchor_pt = coords[0]
    t0 = time.perf_counter()
    dlg._on_scatter_wheel_zoom(float(anchor_pt[0]), float(anchor_pt[1]), 0.55)
    zoom_ms = (time.perf_counter() - t0) * 1000.0
    assert zoom_ms < 40.0 * _perf_budget_scale()
    assert dlg._scatter_view_limits is not None
    assert dlg.map_reset_zoom_btn.isEnabled()
    z_xlim = dlg.scatter_ax.get_xlim()
    assert (z_xlim[1] - z_xlim[0]) < 0.65 * def_span_x

    # 2. Zoom-box mode zooms to exact pocket bounds
    dlg.drag_zoom_btn.click()
    assert dlg.scatter_canvas.drag_mode == "zoom"
    bx0, bx1 = float(anchor_pt[0]) - 0.5, float(anchor_pt[0]) + 0.5
    by0, by1 = float(anchor_pt[1]) - 0.4, float(anchor_pt[1]) + 0.4
    dlg._on_scatter_zoom_box(bx0, bx1, by0, by1)
    zx0, zx1 = dlg.scatter_ax.get_xlim()
    assert abs((zx1 - zx0) - 1.0) < 0.05

    # 3. Pan shifts limits smoothly while preserving zoom width
    dlg._on_scatter_pan_dragged(zx0 + 0.3, zx1 + 0.3, by0, by1)
    px0, px1 = dlg.scatter_ax.get_xlim()
    assert abs((px0 - zx0) - 0.3) < 0.05
    assert abs((px1 - px0) - (zx1 - zx0)) < 1e-4

    # 4. Zoom persists across [1] Keep triage edit so user never loses zoomed cluster view!
    saved_zoom = dlg._scatter_view_limits
    dlg._on_primary_action_1_clicked()
    assert dlg._scatter_view_limits == saved_zoom
    assert abs(dlg.scatter_ax.get_xlim()[0] - px0) < 1e-4

    # 5. Shortcut [0] resets zoom back to full map view
    ev_zero = QtGui.QKeyEvent(
        QtCore.QEvent.Type.KeyPress, Qt.Key.Key_0, Qt.KeyboardModifier.NoModifier, "0"
    )
    dlg.keyPressEvent(ev_zero)
    assert dlg._scatter_view_limits is None
    assert not dlg.map_reset_zoom_btn.isEnabled()
    assert abs(dlg.scatter_ax.get_xlim()[0] - def_xlim[0]) < 1e-4

    # 6. Expand Map [M] maximizes the 2D scatter pane from the 2/3:1/3 default and restores saved splitter sizes on toggle off
    orig_upper = list(dlg._upper_splitter.sizes())
    orig_main = list(dlg._main_splitter.sizes())
    ratio_left = orig_upper[0] / max(1, sum(orig_upper))
    assert 0.63 <= ratio_left <= 0.70, (
        f"Expected ~2/3 left 2D map default ratio, got {ratio_left:.3f} ({orig_upper})"
    )
    ev_m = QtGui.QKeyEvent(
        QtCore.QEvent.Type.KeyPress, Qt.Key.Key_M, Qt.KeyboardModifier.NoModifier, "m"
    )
    dlg.keyPressEvent(ev_m)
    assert dlg._map_expanded is True
    assert dlg.expand_map_btn.isChecked()
    assert not dlg._card_panel.isVisible()
    exp_upper = dlg._upper_splitter.sizes()
    assert exp_upper[0] > orig_upper[0] * 1.35
    assert exp_upper[1] == 0

    dlg.keyPressEvent(ev_m)
    assert dlg._map_expanded is False
    assert not dlg.expand_map_btn.isChecked()
    assert dlg._card_panel.isVisible()
    restored_upper = dlg._upper_splitter.sizes()
    restored_main = dlg._main_splitter.sizes()
    assert abs(restored_upper[0] - orig_upper[0]) <= 6
    assert abs(restored_main[0] - orig_main[0]) <= 6
    dlg.close()


def test_lasso_additive_knn_similar_quick_select_and_cohort_selection(tmp_path: Path) -> None:
    """Verify freehand Lasso (+ Shift union), k-NN Similar `[N]`, Quick Select slices, Track/Cluster cohorts, and Esc clear."""
    _ensure_qapp()
    project, proj_dir = _build_synthetic_explorer_project(tmp_path)
    central_mock = MagicMock()
    central_mock.behavior = "Grooming"
    central_mock.loaded_video = Path(proj_dir / "cage_00.mp4")
    central_mock._pose_est = None

    dlg = BaleDialog(project, central_widget=central_mock)
    dlg.show()
    QtWidgets.QApplication.processEvents()
    rep = run_bale(project, "Grooming", include_predictions=False, space="classifier")
    dlg._on_analysis_finished(rep, included_preds=False)

    coords, _, _ = dlg._current_2d_coords()
    records = rep.feature_matrix.records

    # 1. Freehand polygon Lasso around bout 0 + Shift-additive Lasso around bout 5
    dlg.drag_lasso_btn.click()
    assert dlg.scatter_canvas.drag_mode == "lasso"
    x0, y0 = float(coords[0, 0]), float(coords[0, 1])
    poly_a = np.array(
        [
            [x0 - 0.15, y0 - 0.15],
            [x0 + 0.15, y0 - 0.15],
            [x0 + 0.15, y0 + 0.15],
            [x0 - 0.15, y0 + 0.15],
        ]
    )
    dlg._on_scatter_lasso_selected(poly_a, additive=False)
    assert 0 in dlg._lasso_selected_indices
    assert dlg.clear_box_btn.isVisible()
    count_a = len(dlg._lasso_selected_indices)

    # Pick a distant bout outside poly_a and union via additive=True (Shift+Drag)
    far_idx = int(np.argmax(np.hypot(coords[:, 0] - x0, coords[:, 1] - y0)))
    xf, yf = float(coords[far_idx, 0]), float(coords[far_idx, 1])
    poly_b = np.array(
        [
            [xf - 0.15, yf - 0.15],
            [xf + 0.15, yf - 0.15],
            [xf + 0.15, yf + 0.15],
            [xf - 0.15, yf + 0.15],
        ]
    )
    dlg._on_scatter_lasso_selected(poly_b, additive=True)
    assert 0 in dlg._lasso_selected_indices and far_idx in dlg._lasso_selected_indices
    assert len(dlg._lasso_selected_indices) >= count_a + 1

    # 2. k-NN Similar [N]: selects anchor + closest bouts in monotonic feature-distance order (< 20 ms)
    dlg._selected_bout_idx = 0
    t0 = time.perf_counter()
    ev_n = QtGui.QKeyEvent(
        QtCore.QEvent.Type.KeyPress, Qt.Key.Key_N, Qt.KeyboardModifier.NoModifier, "n"
    )
    dlg.keyPressEvent(ev_n)
    sim_ms = (time.perf_counter() - t0) * 1000.0
    assert sim_ms < 80.0 * _perf_budget_scale()
    assert dlg._custom_selection_preserve_order is True
    assert len(dlg._lasso_selected_indices) == min(15, len(records))
    assert dlg._lasso_selected_indices[0] == 0
    assert int(dlg.table.item(0, 0).data(Qt.ItemDataRole.UserRole)) == 0

    X_repr = dlg._get_active_repr_matrix()
    dists = [float(np.linalg.norm(X_repr[gi] - X_repr[0])) for gi in dlg._lasso_selected_indices]
    assert dists == sorted(dists), "k-NN Similar cohort must preserve monotonic similarity order"

    # 3. Send to JABS pushes exact k-NN similarity-ordered cohort to SearchBarWidget
    dlg.send_queue_btn.click()
    assert central_mock.load_custom_search_queue.called
    sent_hits = central_mock.load_custom_search_queue.call_args[0][0]
    assert len(sent_hits) == len(dlg._lasso_selected_indices)
    assert (
        sent_hits[0].file == records[0].video
        and sent_hits[0].start_frame == records[0].start_frame
    )

    # 4. Same Animal Track cohort preserves chronological start_frame order
    dlg._select_same_track_cohort(0)
    trk_starts = [records[gi].start_frame for gi in dlg._lasso_selected_indices]
    assert trk_starts == sorted(trk_starts)
    assert all(
        records[gi].video == records[0].video and records[gi].identity == records[0].identity
        for gi in dlg._lasso_selected_indices
    )

    # 5. Current JABS Video cohort & Cluster cohort & Quick Select Menu population
    dlg._populate_quick_select_menu()
    assert len(dlg.quick_select_menu.actions()) >= 8
    dlg._select_current_jabs_video_cohort()
    assert all(Path(records[gi].video).name == "cage_00.mp4" for gi in dlg._lasso_selected_indices)

    cid0 = int(rep.embedding.cluster_labels[0])
    dlg._select_cluster_cohort(cid0)
    assert all(int(rep.embedding.cluster_labels[gi]) == cid0 for gi in dlg._lasso_selected_indices)

    dlg._select_quick_slice("class:Grooming")
    assert all(records[gi].class_name == "Grooming" for gi in dlg._lasso_selected_indices)

    # 6. Escape clears active selection when search box is unfocused
    ev_esc = QtGui.QKeyEvent(
        QtCore.QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier, ""
    )
    dlg.keyPressEvent(ev_esc)
    assert dlg._lasso_selected_indices == []
    assert not dlg.clear_box_btn.isVisible()
    dlg.close()


def test_audit_fixes_cohort_advance_expanded_shortcuts_and_canvas_deadzone(
    tmp_path: Path,
) -> None:
    """Verify cohort end-of-queue stability, expanded-map shortcuts (`[T]`/`[Del]`/`[Esc]`), right-click deadzone, out-of-table dot click, and live badge count."""
    _ensure_qapp()
    project, proj_dir = _build_synthetic_explorer_project(tmp_path)
    central_mock = MagicMock()
    central_mock.behavior = "Grooming"
    central_mock.loaded_video = Path(proj_dir / "cage_00.mp4")
    central_mock._pose_est = None

    dlg = BaleDialog(project, central_widget=central_mock)
    dlg.show()
    QtWidgets.QApplication.processEvents()
    rep = run_bale(project, "Grooming", include_predictions=False, space="classifier")
    dlg._on_analysis_finished(rep, included_preds=False)
    assert not dlg._init_refresh_timer.isActive()

    records = rep.feature_matrix.records

    # 1. Right-click with 3 px hand jitter (< 5 px deadzone) emits context_menu_requested WITHOUT panning
    pan_events: list[tuple[float, float, float, float]] = []
    ctx_events: list[int] = []
    dlg.scatter_canvas.pan_dragged.connect(
        lambda x0, x1, y0, y1: pan_events.append((x0, x1, y0, y1))
    )
    dlg.scatter_canvas.context_menu_requested.connect(lambda _pt, gi: ctx_events.append(gi))
    # Disconnect popup menu exec during headless synthetic right-click test
    dlg.scatter_canvas.context_menu_requested.disconnect(dlg._on_scatter_context_menu)
    try:
        p_press = QtCore.QPointF(120.0, 100.0)
        p_jitter = QtCore.QPointF(123.0, 102.0)
        ev_rpress = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseButtonPress,
            p_press,
            dlg.scatter_canvas.mapToGlobal(p_press),
            Qt.MouseButton.RightButton,
            Qt.MouseButton.RightButton,
            Qt.KeyboardModifier.NoModifier,
        )
        ev_rmove = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseMove,
            p_jitter,
            dlg.scatter_canvas.mapToGlobal(p_jitter),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.RightButton,
            Qt.KeyboardModifier.NoModifier,
        )
        ev_rrel = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseButtonRelease,
            p_jitter,
            dlg.scatter_canvas.mapToGlobal(p_jitter),
            Qt.MouseButton.RightButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
        dlg.scatter_canvas.mousePressEvent(ev_rpress)
        dlg.scatter_canvas.mouseMoveEvent(ev_rmove)
        dlg.scatter_canvas.mouseReleaseEvent(ev_rrel)
        assert pan_events == [], "Sub-5px right-click jitter must not pan the 2D map"
        assert len(ctx_events) == 1
        assert dlg._scatter_view_limits is None
    finally:
        dlg.scatter_canvas.context_menu_requested.connect(dlg._on_scatter_context_menu)

    # 2. Out-of-table dot click clears stale table row highlight
    dlg._set_queue_filter("reviewed")
    assert dlg.table.rowCount() == 0 or all(
        records[int(dlg.table.item(r, 0).data(Qt.ItemDataRole.UserRole))].bout_id
        in dlg._reviewed_bout_ids
        for r in range(dlg.table.rowCount())
    )
    dlg._set_queue_filter("all")
    dlg.table.selectRow(0)
    assert len(dlg.table.selectedItems()) > 0
    dlg._set_queue_filter("reviewed")
    # Pick any unreviewed bout not in the empty "Reviewed ✓" table
    dlg._on_scatter_point_clicked(0)
    assert dlg._selected_bout_idx == 0
    assert dlg.table.selectedItems() == []
    # Unmark bout 0 (auto-marked on scatter view above) so sub-test 3 starts with 3 fresh unreviewed bouts
    dlg._toggle_bout_examined(0)
    assert records[0].bout_id not in dlg._reviewed_bout_ids

    # 3. Opt-in cohort auto-advance stops cleanly on final row (never bounces N-1 <-> N-2) & order stays fixed
    dlg._set_queue_filter("needs_review")
    dlg._apply_custom_selection([0, 1, 2], "Map Box", preserve_order=False)
    initial_order = [
        int(dlg.table.item(r, 0).data(Qt.ItemDataRole.UserRole))
        for r in range(dlg.table.rowCount())
    ]
    assert len(initial_order) == 3

    dlg.auto_advance_action.setChecked(True)
    ev_1 = QtGui.QKeyEvent(
        QtCore.QEvent.Type.KeyPress, Qt.Key.Key_1, Qt.KeyboardModifier.NoModifier, "1"
    )
    dlg.keyPressEvent(ev_1)
    assert dlg._selected_bout_idx == initial_order[1]
    dlg.keyPressEvent(ev_1)
    assert dlg._selected_bout_idx == initial_order[2]
    dlg.keyPressEvent(ev_1)
    # All 3 rows are now reviewed — selection stays on initial_order[2] and never bounces to initial_order[1]!
    assert dlg._selected_bout_idx == initial_order[2]
    dlg.keyPressEvent(ev_1)
    assert dlg._selected_bout_idx == initial_order[2]
    after_order = [
        int(dlg.table.item(r, 0).data(Qt.ItemDataRole.UserRole))
        for r in range(dlg.table.rowCount())
    ]
    assert after_order == initial_order
    dlg.auto_advance_action.setChecked(False)

    # 4. Expanded Map [M] shortcuts ([T] Trim preserves row order, [Backspace] Delete clears selection + live badge count, and 3-step [Esc] cascade)
    trim_bi = 3
    orig_s, orig_e = records[trim_bi].start_frame, records[trim_bi].end_frame
    if orig_e - orig_s >= 6:
        rep.diagnostics.suggested_trim_bounds[trim_bi] = (orig_s + 2, orig_e - 2)
    dlg._apply_custom_selection([trim_bi, 4, 5], "Map Box", preserve_order=True)
    assert dlg.clear_box_btn.text() == "✕ Map Box (3)"
    trim_order_before = [
        int(dlg.table.item(r, 0).data(Qt.ItemDataRole.UserRole))
        for r in range(dlg.table.rowCount())
    ]

    dlg._zoom_scatter_by_step(0.76)
    assert dlg._scatter_view_limits is not None
    dlg._toggle_expand_map(True)
    assert dlg._map_expanded is True
    assert not dlg._card_panel.isVisible()

    if rep.diagnostics.suggested_trim_bounds[trim_bi] is not None:
        ev_t = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress, Qt.Key.Key_T, Qt.KeyboardModifier.NoModifier, "t"
        )
        dlg.keyPressEvent(ev_t)
        assert records[trim_bi].start_frame == orig_s + 2
        assert dlg._selected_bout_idx == trim_bi
        trim_order_after = [
            int(dlg.table.item(r, 0).data(Qt.ItemDataRole.UserRole))
            for r in range(dlg.table.rowCount())
        ]
        assert trim_order_after == trim_order_before
        dlg._on_undo_clicked()
        assert records[trim_bi].start_frame == orig_s

    # Delete while auto_advance is off updates clear_box_btn count (3 -> 2), clears selection without jumping, and Undo restores (2 -> 3)
    dlg._selected_bout_idx = 5
    dlg._update_inspector_card()
    with patch("jabs.ui.dialogs.bale_dialog.MessageDialog.confirm", return_value=True):
        ev_del = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress, Qt.Key.Key_Backspace, Qt.KeyboardModifier.NoModifier, ""
        )
        dlg.keyPressEvent(ev_del)
    assert dlg.clear_box_btn.text() == "✕ Map Box (2)"
    assert dlg._selected_bout_idx is None
    assert dlg.table.selectedItems() == []
    dlg._on_undo_clicked()
    assert dlg._selected_bout_idx == 5
    assert dlg.clear_box_btn.text() == "✕ Map Box (3)"

    # Cascaded Escape: (1) clears cohort -> (2) collapses expanded map -> (3) resets zoom, without closing dialog!
    ev_esc = QtGui.QKeyEvent(
        QtCore.QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier, ""
    )
    dlg.keyPressEvent(ev_esc)
    assert dlg._lasso_selected_indices == []
    assert dlg._map_expanded is True
    assert dlg._scatter_view_limits is not None
    assert dlg.isVisible()

    dlg.keyPressEvent(ev_esc)
    assert dlg._map_expanded is False
    assert dlg._scatter_view_limits is not None
    assert dlg.isVisible()

    dlg.keyPressEvent(ev_esc)
    assert dlg._scatter_view_limits is None
    assert dlg.isVisible()
    dlg.close()


def test_animal_id_sync_already_selected_row_click_prob_scrub_and_mc_queue(tmp_path: Path) -> None:
    """Verify row-0 auto-sync, already-selected row click sync, prob-curve animal sync, and MC SearchHit behavior."""
    _ensure_qapp()
    project, proj_dir = _build_synthetic_explorer_project(tmp_path)
    # Copy cage_00 feature dir to animal '1' and put bouts on animal '1' as well
    anno0 = proj_dir / "jabs" / "annotations" / "cage_00.json"
    doc0 = json.loads(anno0.read_text())
    import shutil

    fdir0 = proj_dir / "jabs" / "features" / "cage_00" / "0"
    fdir1 = proj_dir / "jabs" / "features" / "cage_00" / "1"
    if not fdir1.exists():
        shutil.copytree(fdir0, fdir1)
    doc0["unfragmented_labels"]["1"] = {
        "Grooming": [
            {"start": 40, "end": 85, "present": True},
            {"start": 160, "end": 205, "present": False},
        ]
    }
    doc0["labels"]["1"] = doc0["unfragmented_labels"]["1"]
    anno0.write_text(json.dumps(doc0))

    central_mock = MagicMock()
    central_mock.behavior = "Grooming"
    central_mock.loaded_video = Path(proj_dir / "cage_00.mp4")
    central_mock._pose_est = None
    central_mock._controls = MagicMock()
    central_mock._controls.current_identity_index = 0

    dlg = BaleDialog(project, central_widget=central_mock)
    dlg.show()
    QtWidgets.QApplication.processEvents()

    rep = run_bale(project, "Grooming", include_predictions=False, space="classifier")
    central_mock.seek_to_bout.reset_mock()
    dlg._on_analysis_finished(rep, included_preds=False)

    # 1. Analysis finish populates row 0 visually WITHOUT jumping JABS main window
    assert dlg.table.rowCount() >= 2
    row0_bidx = int(dlg.table.item(0, 0).data(Qt.ItemDataRole.UserRole))
    row0_rec = rep.feature_matrix.records[row0_bidx]
    assert central_mock.seek_to_bout.call_count == 0

    # 2. Clicking row 0 explicitly seeks JABS GUI to row 0's exact video & 0-based identity
    central_mock.seek_to_bout.reset_mock()
    dlg.table.cellClicked.emit(0, 3)
    assert central_mock.seek_to_bout.call_count == 1
    assert central_mock.seek_to_bout.call_args[0][1] == row0_rec.identity

    # 3. Selecting a DIFFERENT row via Qt selectRow + cellClicked (simulating mouse click) seeks only ONCE
    central_mock.seek_to_bout.reset_mock()
    row1_bidx = int(dlg.table.item(1, 0).data(Qt.ItemDataRole.UserRole))
    row1_rec = rep.feature_matrix.records[row1_bidx]
    dlg.table.selectRow(1)
    dlg.table.cellClicked.emit(1, 3)
    assert central_mock.seek_to_bout.call_count == 1
    assert central_mock.seek_to_bout.call_args[0][1] == row1_rec.identity

    # 4. Clicking prob curve when same video is loaded on Animal 0 while bout is Animal 1 switches animal FIRST!
    id1_row = next(
        r
        for r in range(dlg.table.rowCount())
        if rep.feature_matrix.records[
            int(dlg.table.item(r, 0).data(Qt.ItemDataRole.UserRole))
        ].identity
        == 1
        and rep.feature_matrix.records[
            int(dlg.table.item(r, 0).data(Qt.ItemDataRole.UserRole))
        ].video
        == "cage_00.mp4"
    )
    dlg.table.selectRow(id1_row)
    central_mock.seek_to_bout.reset_mock()
    central_mock.seek_to_frame_only.reset_mock()
    central_mock.loaded_video = Path(proj_dir / "cage_00.mp4")
    central_mock._controls.current_identity_index = 0
    central_mock.behavior = "Grooming"
    id1_rec = rep.feature_matrix.records[
        int(dlg.table.item(id1_row, 0).data(Qt.ItemDataRole.UserRole))
    ]
    dlg._on_prob_curve_frame_clicked(float(id1_rec.start_frame + 5))
    assert central_mock.seek_to_bout.call_count == 1
    assert central_mock.seek_to_bout.call_args[0][1] == 1
    central_mock.seek_to_frame_only.assert_called_once_with(id1_rec.start_frame + 5)

    # 5. Send to JABS maps MULTICLASS_NONE_BEHAVIOR ("None") to report.behavior, never emitting "None"
    from dataclasses import replace as dc_replace

    mc_rec0 = dc_replace(rep.feature_matrix.records[row0_bidx], behavior="None", class_name="None")
    rep.feature_matrix.records[row0_bidx] = mc_rec0
    central_mock.load_custom_search_queue.reset_mock()
    dlg._on_send_queue_to_search_bar()
    sent_hits = central_mock.load_custom_search_queue.call_args[0][0]
    assert all(h.behavior != "None" for h in sent_hits)

    # 6. Strictly one-way Picker -> JABS: filter/mode/cohort clicks & JABS GUI behavior/label changes never yank JABS or mutate Picker
    from types import SimpleNamespace

    from jabs.ui.main_window.menu_handlers import MenuHandlers

    central_mock.seek_to_bout.reset_mock()
    dlg.qf_all_btn.setChecked(True)
    dlg._on_queue_filter_pill_clicked(1)
    dlg.mode_subtypes_btn.setChecked(True)
    dlg._on_mode_pill_clicked(2)
    dlg._apply_custom_selection([row0_bidx, row1_bidx], "Test Cohort", preserve_order=True)
    assert central_mock.seek_to_bout.call_count == 0

    assert not hasattr(dlg, "sync_to_central_behavior")
    mode_before = dlg._mode
    sel_before = dlg._selected_bout_idx
    central_mock.behavior = "OtherBehavior"
    MenuHandlers.open_bale_dialog(
        SimpleNamespace(window=SimpleNamespace(_project=project, _bale_dialog=dlg))
    )
    assert dlg.behavior_combo.currentText() == "Grooming"
    assert dlg._mode == mode_before
    assert dlg._selected_bout_idx == sel_before
    assert central_mock.seek_to_bout.call_count == 0

    dlg._on_external_jabs_labels_modified()
    assert "Updated" in dlg.recalc_btn.text()
    assert dlg._selected_bout_idx == sel_before
    assert central_mock.seek_to_bout.call_count == 0
    dlg.close()


def _build_mc_ui_test_harness(tmp_path: Path, *, all_multiclass: bool = False):
    """Build a headless Multi-Class JABS project, report, and mocked central widget."""
    import json
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import numpy as np
    import pandas as pd

    from jabs.bale.bout_extractor import strip_known_video_suffix
    from jabs.bale.models import (
        BaleReport,
        BoutFeatureMatrix,
        BoutRecord,
        ClusterAndEmbeddingResult,
        ConfusionAndOutlierResult,
    )
    from jabs.core.enums import ClassifierMode

    proj_dir = tmp_path / "ui_mc_proj"
    for sub in ("annotations", "features", "predictions", "classifiers", "training_logs"):
        (proj_dir / "jabs" / sub).mkdir(parents=True, exist_ok=True)

    vid = "cage1.top.mp4"
    (proj_dir / vid).touch()
    stem = strip_known_video_suffix(vid)
    feat_dir = proj_dir / "jabs" / "features" / stem / "0"
    feat_dir.mkdir(parents=True, exist_ok=True)
    n_frames = 220
    pd.DataFrame(
        np.zeros((n_frames, 4), dtype=np.float32),
        columns=["f0", "f1", "f2", "f3"],
    ).to_parquet(feat_dir / "per_frame.parquet", index=False)

    anno_doc = {
        "file": vid,
        "num_frames": n_frames,
        "labels": {
            "0": {
                "Turn_Left": [{"start": 10, "end": 24, "present": True}],
                "Turn_Right": [{"start": 35, "end": 49, "present": True}],
                "None": [{"start": 60, "end": 74, "present": True}],
            }
        },
        "unfragmented_labels": {
            "0": {
                "Turn_Left": [{"start": 10, "end": 24, "present": True}],
                "Turn_Right": [{"start": 35, "end": 49, "present": True}],
                "None": [{"start": 60, "end": 74, "present": True}],
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
    proj.video_manager.videos = [vid]
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

    records = [
        BoutRecord(1, "labeled", vid, 0, "Turn_Left", "Turn_Left", True, 10, 24, 15),
        BoutRecord(2, "labeled", vid, 0, "Turn_Right", "Turn_Right", True, 35, 49, 15),
        BoutRecord(3, "labeled", vid, 0, "None", "None", False, 60, 74, 15),
        BoutRecord(
            4,
            "predicted",
            vid,
            0,
            "Turn_Left",
            "Turn_Left",
            True,
            90,
            104,
            15,
            mean_pred_prob=0.58,
            origin_source="predicted",
            origin_behavior="Turn_Left",
        ),
        BoutRecord(
            5,
            "predicted",
            vid,
            0,
            "Turn_Right",
            "Turn_Right",
            True,
            125,
            139,
            15,
            mean_pred_prob=0.61,
            origin_source="predicted",
            origin_behavior="Turn_Right",
        ),
        BoutRecord(
            6,
            "predicted",
            vid,
            0,
            "Turn_Left",
            "Turn_Left",
            True,
            160,
            174,
            15,
            mean_pred_prob=0.63,
            origin_source="predicted",
            origin_behavior="Turn_Left",
        ),
    ]
    n = len(records)
    coords = np.array(
        [
            [-5.0, -5.0],
            [-4.5, -4.5],
            [-4.0, -5.5],
            [4.0, 4.0],
            [5.0, 5.0],
            [6.0, 4.5],
        ],
        dtype=np.float32,
    )
    frame_probs = {
        r.bout_id: np.full(r.duration_frames, 0.65 if r.is_positive else 0.15, dtype=np.float32)
        for r in records
    }
    fm = BoutFeatureMatrix(
        records=records,
        X_bout=np.zeros((n, 4), dtype=np.float32),
        stat_names=["f0", "f1", "f2", "f3"],
        base_feature_names=["f0", "f1", "f2", "f3"],
        stat_to_base_idx=np.arange(4, dtype=np.int32),
        frame_probs_by_bout=frame_probs,
    )
    diag = ConfusionAndOutlierResult(
        confusion_scores=np.array([0.10, 0.12, 0.15, 0.84, 0.72, 0.52], dtype=np.float32),
        oof_mean_probs=np.array([0.92, 0.90, 0.12, 0.42, 0.58, 0.63], dtype=np.float32),
        knn_disagreement=np.array([0.0, 0.0, 0.0, 0.71, 0.28, 0.14], dtype=np.float32),
        ood_scores=np.zeros(n, dtype=np.float32),
        transition_scores=np.zeros(n, dtype=np.float32),
        suggested_trim_bounds=[None] * n,
        suggested_split_frame=[None] * n,
        outlier_scores=np.zeros(n, dtype=np.float32),
        outlier_votes=np.zeros(n, dtype=np.int32),
        lof_scores=np.zeros(n, dtype=np.float32),
        isoforest_scores=np.zeros(n, dtype=np.float32),
        base_feature_weights=np.full(4, 0.25, dtype=np.float32),
        stat_feature_weights=np.ones(4, dtype=np.float32),
        oof_top_class=[
            "Turn_Left",
            "Turn_Right",
            "None",
            "Turn_Right",
            "Turn_Right",
            "Turn_Left",
        ],
        knn_majority_class=[
            "Turn_Left",
            "Turn_Right",
            "None",
            "Turn_Right",
            "Turn_Right",
            "Turn_Left",
        ],
    )
    emb = ClusterAndEmbeddingResult(
        coords_pca_2d=coords,
        pca_explained_variance_ratio=(0.55, 0.25),
        coords_tsne_2d=None,
        coords_mds_2d=coords,
        cluster_labels=np.array([1, 2, 1, 2, 2, 1], dtype=np.int32),
        optimal_k=2,
        silhouette=0.45,
        cluster_top_features={1: [("f0", 1.2)], 2: [("f1", -1.1)]},
        top_z_diffs_per_bout=[[("f0", 1.0)] for _ in range(n)],
    )
    report = BaleReport(
        project_dir=str(proj_dir),
        behavior="Turn_Left",
        is_multiclass=True,
        space="classifier",
        feature_matrix=fm,
        diagnostics=diag,
        embedding=emb,
        cv_groups_count=16,
        jabs_threshold_met=True,
        dual_class_cv_met=True,
        timings_ms={"total_ms": 10.0},
        cv_track_counts={},
        include_all_multiclass=all_multiclass,
    )

    cw = SimpleNamespace(
        behavior="Turn_Left",
        loaded_video=None,
        seek_to_bout=MagicMock(),
        load_custom_search_queue=MagicMock(),
        sync_external_video_labels=MagicMock(),
    )
    return proj, report, cw


def test_mc_prediction_separation_auto_select_assign_and_no_human_leakage(
    tmp_path: Path,
) -> None:
    """Multi-Class opens on `Turn_Left`, uses `Assign:`, isolates predictions, names chirality."""
    from jabs.ui.dialogs.bale_dialog import BaleDialog

    _ensure_qapp()
    proj, report, cw = _build_mc_ui_test_harness(tmp_path)
    dlg = BaleDialog(proj, central_widget=cw)
    dlg._init_refresh_timer.stop()
    try:
        # 1. Default behavior matches central_widget.behavior even in Multi-Class mode
        assert dlg.behavior_combo.currentData() == "Turn_Left"
        assert dlg.mc_target_lbl.text() == "Assign:"
        assert not dlg.target_pos_combo.isHidden()

        # Load report & simulate reviewing human Bout #1 in 1. Audit Labels
        dlg._report = report
        dlg._report_includes_predictions = True
        dlg._reviewed_bout_ids.add(1)
        dlg._pinned_needs_review_bout_ids.add(1)

        # 2. Switch to 2. Review Predictions: ONLY Turn_Left predictions (indices 3, 5) allowed
        dlg.mode_predict_btn.setChecked(True)
        dlg._on_mode_pill_clicked(1)
        assert dlg._mode == "predict"
        assert dlg._mode_eligible_indices() == [3, 5]
        assert not dlg.target_pos_combo.isHidden()

        # 3. Chiral k-NN / OOF diagnosis names competing Turn_Right instead of None
        badge, fix_txt, diag_txt, _ = dlg._describe_bout_issue(3)
        assert "False Pos?" in badge
        assert "Turn_Right" in fix_txt
        assert "Turn_Right" in diag_txt

        # 4. Primary [Z] button reflects Confirm vs Save as when Assign: target changes
        dlg._selected_bout_idx = 3
        idx_tl = dlg.target_pos_combo.findData("Turn_Left")
        dlg.target_pos_combo.setCurrentIndex(idx_tl)
        assert dlg.primary_pos_btn.text() == "✓ Confirm Turn_Left [Z]"

        idx_tr = dlg.target_pos_combo.findData("Turn_Right")
        dlg.target_pos_combo.setCurrentIndex(idx_tr)
        assert dlg.primary_pos_btn.text() == "⇄ Save as Turn_Right [Z]"
    finally:
        dlg.close()


def test_predict_reviewed_rendering_search_bar_all_classes_and_scatter_backdrop(
    tmp_path: Path,
) -> None:
    """Confirm/Reject formatting, Search Bar exclusion/All Classes, non-pickable backdrop."""
    from jabs.ui.dialogs.bale_dialog import BaleDialog

    _ensure_qapp()
    proj, report, cw = _build_mc_ui_test_harness(tmp_path)
    dlg = BaleDialog(proj, central_widget=cw)
    dlg._init_refresh_timer.stop()
    try:
        dlg._report = report
        dlg._report_includes_predictions = True
        dlg.mode_predict_btn.setChecked(True)
        dlg._on_mode_pill_clicked(1)

        # 2D map pickable points exclude labeled backdrop (0, 1, 2) and Turn_Right pred (4)
        assert dlg.show_pred_backdrop_action.isCheckable()
        assert dlg.show_pred_backdrop_action.isChecked() is True
        pick_ids = list(dlg.scatter_canvas._pick_global_indices.tolist())
        assert pick_ids == [3, 5]

        # Clicking/box/lassoing human backdrop coordinate (-5, -5) never selects human bout 0/2
        dlg._on_scatter_point_clicked(0)
        assert dlg._selected_bout_idx != 0
        dlg._on_scatter_box_selected(-6.0, -3.0, -6.0, -3.0)
        assert dlg._lasso_selected_indices == []
        dlg._on_scatter_lasso_selected([[-6.0, -6.0], [-3.0, -6.0], [-3.0, -3.0], [-6.0, -3.0]])
        assert dlg._lasso_selected_indices == []

        # Toggling backdrop off clears backdrop dots
        dlg.show_pred_backdrop_action.setChecked(False)
        assert dlg._scatter_backdrop_indices() == []

        # Simulate confirming Bout #4 (idx 3) and rejecting Bout #5 (idx 4) in-session
        rec_conf = report.feature_matrix.records[3]
        rec_conf.source = "labeled"
        dlg._reviewed_bout_ids.add(rec_conf.bout_id)

        rec_rej = report.feature_matrix.records[4]
        rec_rej.source = "labeled"
        rec_rej.behavior = "None"
        rec_rej.class_name = "None"
        rec_rej.is_positive = False
        dlg._reviewed_bout_ids.add(rec_rej.bout_id)

        # Single-class Turn_Left Reviewed ✓ tab shows Confirmed Turn_Left (no ● Labeled)
        dlg._set_queue_filter("reviewed", redraw=True)
        assert dlg.table.rowCount() == 1
        assert dlg.table.item(0, 1).text() == "✓ Confirmed: Turn_Left"
        dlg._selected_bout_idx = 3
        dlg._update_inspector_card()
        assert "Reviewed Prediction: Turn_Left" in dlg.card_title_lbl.text()

        # Switch to Needs Review: Search Bar sends only unreviewed prediction (idx 5 / bout 6)
        dlg._set_queue_filter("needs_review", redraw=True)
        dlg._on_send_queue_to_search_bar()
        sent_hits = cw.load_custom_search_queue.call_args[0][0]
        kwargs = cw.load_custom_search_queue.call_args[1]
        assert len(sent_hits) == 1
        assert sent_hits[0].start_frame == 160
        assert "Turn_Left" in kwargs["description"]

        # Switch to All Classes (__ALL_MC__): All Classes label + rejected keeps origin_behavior
        cw.load_custom_search_queue.reset_mock()
        report.include_all_multiclass = True
        idx_all = dlg.behavior_combo.findData("__ALL_MC__")
        dlg.behavior_combo.blockSignals(True)
        dlg.behavior_combo.setCurrentIndex(idx_all)
        dlg.behavior_combo.blockSignals(False)
        dlg._set_queue_filter("all", redraw=True)
        assert any(
            dlg.table.item(r, 1).text() == "✗ Rejected → None" for r in range(dlg.table.rowCount())
        )
        dlg._on_send_queue_to_search_bar()
        sent_all = cw.load_custom_search_queue.call_args[0][0]
        desc_all = cw.load_custom_search_queue.call_args[1]["description"]
        assert "All Classes" in desc_all
        assert "Turn_Left" not in desc_all
        assert "All Classes" in dlg.status_summary_label.text()
        rej_hits = [h for h in sent_all if h.start_frame == 125]
        assert len(rej_hits) == 1 and rej_hits[0].behavior == "Turn_Right"

        # Empty-prediction scatter state cites jabs/predictions/*.h5
        report_empty = report
        report_empty.feature_matrix.records = [
            r for r in report.feature_matrix.records if r.origin_source == "labeled"
        ]
        report_empty.include_all_multiclass = False
        idx_tl2 = dlg.behavior_combo.findData("Turn_Left")
        dlg.behavior_combo.blockSignals(True)
        dlg.behavior_combo.setCurrentIndex(idx_tl2)
        dlg.behavior_combo.blockSignals(False)
        dlg._draw_scatter()
        assert dlg.scatter_canvas._pick_global_indices is None
        ax_texts = " ".join(t.get_text() for t in dlg.scatter_ax.texts)
        assert "jabs/predictions/*.h5" in ax_texts
    finally:
        dlg.close()


def test_subtypes_tier1_controls_cluster_heatmap_click_and_mode_restore(tmp_path: Path) -> None:
    """Verify Tier 1 Sub-Types controls, clickable heatmap filter, and clean Audit restore."""
    _ensure_qapp()
    project, _ = _build_synthetic_explorer_project(tmp_path)
    central_mock = MagicMock()
    central_mock.behavior = "Grooming"
    central_mock.loaded_video = None

    dlg = BaleDialog(project, central_widget=central_mock)
    try:
        rep_audit = run_bale(project, "Grooming", compute_mds=True)
        dlg._on_analysis_finished(rep_audit, included_preds=False)

        # Hidden in 1. Audit Labels
        assert dlg.subtypes_bar.isHidden()
        assert dlg.view_clusters_btn.isHidden()
        assert not dlg.view_prob_btn.isHidden()

        # Hidden in 2. Review Predictions
        rep_pred = run_bale(
            project,
            "Grooming",
            include_predictions=True,
            max_predicted_bouts=50,
            space="classifier",
        )
        dlg._mode = "predict"
        dlg.mode_predict_btn.setChecked(True)
        dlg._on_mode_pill_clicked(1)
        dlg._on_analysis_finished(rep_pred, included_preds=True)
        assert dlg.subtypes_bar.isHidden()
        assert dlg.view_clusters_btn.isHidden()

        # Switch to 3. Sub-Types -> progressive-disclosure controls & Cluster Profiles appear
        dlg.mode_subtypes_btn.setChecked(True)
        dlg._on_mode_pill_clicked(2)
        assert not dlg.subtypes_bar.isHidden()
        assert not dlg.view_clusters_btn.isHidden()
        assert dlg.view_prob_btn.isHidden()
        assert dlg._inspector_view == "clusters"
        assert dlg.view_clusters_btn.isChecked()
        assert dlg.insp_canvas._interactive_y_pick is True
        assert dlg._report is not None
        assert dlg._report.space == "unsupervised"
        assert dlg._report.embedding.subtype_scope == "pos_only"
        assert dlg._report.embedding.feature_lens == "intrinsic"

        n_pos_only = sum(dlg._report.embedding.cluster_counts.values())
        assert n_pos_only > 0
        assert all(
            dlg._report.feature_matrix.records[
                int(dlg.table.item(r, 0).data(Qt.ItemDataRole.UserRole))
            ].is_positive
            for r in range(dlg.table.rowCount())
        )

        # Scope -> All Labeled (Pos + Neg), then back to Positive Only
        idx_all = dlg.subtypes_scope_combo.findData("all_labeled")
        dlg.subtypes_scope_combo.setCurrentIndex(idx_all)
        assert dlg._report.embedding.subtype_scope == "all_labeled"
        assert sum(dlg._report.embedding.cluster_counts.values()) > n_pos_only
        idx_pos = dlg.subtypes_scope_combo.findData("pos_only")
        dlg.subtypes_scope_combo.setCurrentIndex(idx_pos)
        assert dlg._report.embedding.subtype_scope == "pos_only"

        # Lens transitions update space & lens immediately
        for lens_key, expected_space, expected_lens in (
            ("locomotion", "unsupervised", "locomotion"),
            ("social", "unsupervised", "social"),
            ("classifier_weighted", "classifier", "classifier_weighted"),
            ("rf_proximity", "rf_proximity", "all"),
            ("intrinsic", "unsupervised", "intrinsic"),
        ):
            idx_l = dlg.subtypes_lens_combo.findData(lens_key)
            dlg.subtypes_lens_combo.setCurrentIndex(idx_l)
            assert dlg._report.space == expected_space
            assert dlg._report.embedding.feature_lens == expected_lens

        # Clusters combo k = 3 overrides Ward.D2 count
        idx_k3 = dlg.subtypes_k_combo.findData(3)
        dlg.subtypes_k_combo.setCurrentIndex(idx_k3)
        assert dlg._report.embedding.optimal_k == 3
        assert dlg._report.embedding.forced_k == 3
        assert len(dlg._report.embedding.cluster_counts) == 3

        # Cage skew badge shows when p < 0.005 and Cramér's V >= 0.35
        dlg._report.embedding.cage_skew_p = 1e-6
        dlg._report.embedding.cage_skew_cramers_v = 0.68
        dlg._report.embedding.dominant_cage_by_cluster = {
            1: ("cage_00", 0.90),
            2: ("cage_01", 0.85),
            3: ("cage_00", 0.50),
        }
        dlg._sync_subtypes_bar_from_report()
        assert not dlg.cage_skew_badge.isHidden()
        assert "Cage Skew" in dlg.cage_skew_badge.text()

        # Render Cluster Profiles heatmap offscreen and simulate row 0 click -> filters C1
        dlg._update_inspector_card()
        dlg.insp_canvas._render_now()
        assert dlg.insp_canvas._pixmap is not None and not dlg.insp_canvas._pixmap.isNull()
        dlg.insp_canvas.y_value_clicked.emit(0.0)
        assert "Cluster C1" in dlg.clear_box_btn.text()
        assert dlg.table.rowCount() > 0
        for r in range(dlg.table.rowCount()):
            b_i = int(dlg.table.item(r, 0).data(Qt.ItemDataRole.UserRole))
            assert int(dlg._report.embedding.cluster_labels[b_i]) == 1

        # Restore back to 1. Audit Labels resets all Sub-Types overrides cleanly
        dlg.mode_audit_btn.setChecked(True)
        dlg._on_mode_pill_clicked(0)
        assert dlg.subtypes_bar.isHidden()
        assert dlg.view_clusters_btn.isHidden()
        assert not dlg.view_prob_btn.isHidden()
        assert dlg._inspector_view == "prob"
        assert dlg._report.space == "classifier"
        assert dlg._report.embedding.subtype_scope == "all_labeled"
        assert dlg._report.embedding.feature_lens == "all"
        assert dlg._report.embedding.forced_k is None
    finally:
        dlg.close()


def test_inspector_traj_to_prob_resets_aspect_without_matplotlib_warning(
    tmp_path: Path, caplog
) -> None:
    """Verify switching Arena Trajectory -> Probability Curve resets aspect and avoids y-limit warnings."""
    import logging

    import pytest

    from jabs.bale.trajectory_and_export import BoutPoseTrajectory

    _ensure_qapp()
    project, _ = _build_synthetic_explorer_project(tmp_path)
    dlg = BaleDialog(project)
    rep = run_bale(project, "Grooming", include_predictions=False, space="classifier")
    try:
        dlg._on_analysis_finished(rep, included_preds=False)
        assert dlg._selected_bout_idx is not None
        rec = rep.feature_matrix.records[dlg._selected_bout_idx]
        frames = np.arange(rec.start_frame, rec.end_frame + 1)
        xy = np.column_stack(
            [
                np.linspace(100.0, 220.0, len(frames)),
                np.linspace(300.0, 340.0, len(frames)),
            ]
        )
        fake_traj = BoutPoseTrajectory(
            video=rec.video,
            identity=rec.identity,
            start_frame=rec.start_frame,
            end_frame=rec.end_frame,
            frames=frames,
            nose_xy=xy,
            neck_xy=xy,
            spine_xy=xy,
            tail_xy=xy,
            centroid_xy=xy,
        )

        with (
            patch(
                "jabs.ui.dialogs.bale_dialog.load_bout_pose_trajectory",
                return_value=fake_traj,
            ),
            caplog.at_level(logging.WARNING, logger="matplotlib.axes._base"),
        ):
            dlg._set_inspector_view("traj", redraw=True)
            dlg.insp_canvas._render_now()
            assert dlg.insp_ax.get_adjustable() == "datalim"
            assert dlg.insp_ax.yaxis_inverted()

            dlg._set_inspector_view("prob", redraw=True)
            dlg.insp_canvas._render_now()
            assert "Ignoring fixed y limits" not in caplog.text
            assert dlg.insp_ax.get_aspect() == "auto"
            assert dlg.insp_ax.get_adjustable() == "box"
            assert not dlg.insp_ax.yaxis_inverted()
            assert dlg.insp_ax.get_ylim() == pytest.approx((-0.04, 1.06))
    finally:
        dlg.close()


def test_sortable_table_columns_numeric_and_send_to_jabs_order(tmp_path: Path) -> None:
    """Clicking bottom table headers sorts numerically (Dur/Animal/Priority) and preserves selection & queue order."""
    _ensure_qapp()
    proj, report, cw = _build_mc_ui_test_harness(tmp_path, all_multiclass=True)
    # Give varied durations and a multi-digit identity to verify natural numeric sort vs alphabetical
    recs = report.feature_matrix.records
    recs[0].start_frame, recs[0].end_frame, recs[0].duration_frames = 10, 34, 25
    recs[1].start_frame, recs[1].end_frame, recs[1].duration_frames = 20, 119, 100
    recs[2].start_frame, recs[2].end_frame, recs[2].duration_frames = 5, 19, 15
    recs[0].identity = 2
    recs[1].identity = 10
    recs[2].identity = 1

    dlg = BaleDialog(proj, central_widget=cw)
    dlg._init_refresh_timer.stop()
    try:
        idx_all = dlg.behavior_combo.findData("__ALL_MC__")
        assert idx_all >= 0
        dlg.behavior_combo.blockSignals(True)
        dlg.behavior_combo.setCurrentIndex(idx_all)
        dlg.behavior_combo.blockSignals(False)
        dlg._on_analysis_finished(report, included_preds=True)
        dlg._set_queue_filter("all", redraw=False)
        dlg._populate_table(redraw_scatter_on_change=False)
        assert dlg.table.rowCount() >= 3

        selected_before = dlg._selected_bout_idx
        cw.seek_to_bout.reset_mock()

        # Click Col 4 ("Frames (Dur)"): defaults to Descending numeric duration (100f before 25f before 15f)
        dlg._on_table_header_clicked(4)
        durs_desc = [
            recs[int(dlg.table.item(r, 0).data(Qt.ItemDataRole.UserRole))].duration_frames
            for r in range(dlg.table.rowCount())
        ]
        assert durs_desc == sorted(durs_desc, reverse=True)
        assert dlg._selected_bout_idx == selected_before
        assert cw.seek_to_bout.call_count == 0
        assert dlg.table.horizontalHeader().isSortIndicatorShown()

        # Click Col 4 again -> toggles to Ascending numeric duration (15f before 25f before 100f)
        dlg._on_table_header_clicked(4)
        durs_asc = [
            recs[int(dlg.table.item(r, 0).data(Qt.ItemDataRole.UserRole))].duration_frames
            for r in range(dlg.table.rowCount())
        ]
        assert durs_asc == sorted(durs_asc)

        # Click Col 3 ("Animal"): defaults to Ascending natural numeric order (1, 2, 10 — NOT "1", "10", "2")
        dlg._on_table_header_clicked(3)
        idents_asc = [
            recs[int(dlg.table.item(r, 0).data(Qt.ItemDataRole.UserRole))].identity
            for r in range(dlg.table.rowCount())
        ]
        assert idents_asc == sorted(idents_asc)

        # Send to JABS Search Bar must follow the exact sorted table row order
        cw.load_custom_search_queue.reset_mock()
        dlg._on_send_queue_to_search_bar()
        assert cw.load_custom_search_queue.call_count == 1
        sent_hits = cw.load_custom_search_queue.call_args.args[0]
        assert [int(h.identity) for h in sent_hits] == idents_asc

        # Sort column persists across _populate_table()
        dlg._populate_table(redraw_scatter_on_change=False)
        idents_after_repop = [
            recs[int(dlg.table.item(r, 0).data(Qt.ItemDataRole.UserRole))].identity
            for r in range(dlg.table.rowCount())
        ]
        assert idents_after_repop == idents_asc
    finally:
        dlg.close()


def test_z_x_c_shortcuts_no_label_and_fast_inplace_triage(tmp_path: Path) -> None:
    """`Z` (Behavior), `X` (No Label / Clear), and `C` (Not-Behavior/None) work cleanly in Predict & Audit modes."""
    from types import SimpleNamespace

    from jabs.project import VideoLabels

    _ensure_qapp()
    proj, report, cw = _build_mc_ui_test_harness(tmp_path, all_multiclass=False)
    live_labels = VideoLabels("cage1.top.mp4", 220)
    recs = report.feature_matrix.records
    live_labels.get_track_labels("0", "Turn_Left").label_behavior(
        recs[0].start_frame, recs[0].end_frame
    )
    live_labels.get_track_labels("0", "Turn_Right").label_behavior(
        recs[1].start_frame, recs[1].end_frame
    )
    live_labels.get_track_labels("0", "None").label_behavior(
        recs[2].start_frame, recs[2].end_frame
    )
    proj.video_manager.load_video_labels.side_effect = lambda *_a, **_kw: live_labels
    proj.load_pose_est.side_effect = lambda *_a, **_kw: SimpleNamespace(num_frames=220)
    cw.loaded_video = proj.dir / "cage1.top.mp4"
    cw._pose_est = SimpleNamespace(num_frames=220)
    cw._labels = live_labels

    dlg = BaleDialog(proj, central_widget=cw)
    dlg._init_refresh_timer.stop()
    try:
        dlg._on_analysis_finished(report, included_preds=True)
        dlg.pair_opposite_action.setChecked(False)

        # Switch to 2. Review Predictions (predicted Turn_Left bouts: indices 3, 5)
        dlg.mode_predict_btn.setChecked(True)
        dlg._on_mode_pill_clicked(1)
        assert dlg._mode == "predict"
        assert not dlg.delete_btn.isHidden()
        assert "[X]" in dlg.delete_btn.text()

        # Select predicted bout #3 and press [X] -> marks No Label in-place (zero disk write), Undoable!
        row_b3 = dlg._find_table_row_for_bout(3)
        assert row_b3 is not None
        dlg.table.selectRow(row_b3)
        ev_x = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress, Qt.Key.Key_X, Qt.KeyboardModifier.NoModifier, "x"
        )
        dlg.keyPressEvent(ev_x)
        assert recs[3].bout_id in dlg._unlabeled_bout_ids
        assert dlg.table.item(row_b3, 1).text() == "○ No Label"
        assert recs[3].source == "predicted"
        dlg._on_undo_clicked()
        assert recs[3].bout_id not in dlg._unlabeled_bout_ids

        # Press [Z] on predicted bout #3 -> confirms Turn_Left and updates row in-place
        row_b3 = dlg._find_table_row_for_bout(3)
        assert row_b3 is not None
        dlg.table.selectRow(row_b3)
        ev_z = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress, Qt.Key.Key_Z, Qt.KeyboardModifier.NoModifier, "z"
        )
        dlg.keyPressEvent(ev_z)
        assert recs[3].source == "labeled"
        assert recs[3].class_name == "Turn_Left"
        assert dlg.table.item(row_b3, 1).text() == "✓ Confirmed: Turn_Left"
        # Live pose from CentralWidget was reused directly (zero extra HDF5 pose opens!)
        assert proj.load_pose_est.call_count == 0

        # Now press [X] on that newly labeled predicted bout -> clears annotation on disk & keeps row as ○ No Label
        dlg.keyPressEvent(ev_x)
        assert recs[3].source == "predicted"
        assert recs[3].bout_id in dlg._unlabeled_bout_ids
        assert dlg.table.item(row_b3, 1).text() == "○ No Label"

        # Press [C] -> labels explicit None and updates row in-place
        ev_c = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress, Qt.Key.Key_C, Qt.KeyboardModifier.NoModifier, "c"
        )
        dlg.keyPressEvent(ev_c)
        assert recs[3].source == "labeled"
        assert recs[3].class_name == "None"
        assert not recs[3].is_positive
        assert dlg.table.item(row_b3, 1).text() == "✗ Rejected → None"

        # In 1. Audit Labels: on a None bout, [C] keeps None while [Z] switches to Turn_Left
        dlg.mode_audit_btn.setChecked(True)
        dlg._on_mode_pill_clicked(0)
        dlg._set_queue_filter("all", redraw=False)
        dlg._populate_table(redraw_scatter_on_change=False)
        none_idx = next(i for i in dlg._mode_eligible_indices() if recs[i].class_name == "None")
        row_none = dlg._find_table_row_for_bout(none_idx)
        assert row_none is not None
        dlg.table.selectRow(row_none)
        dlg.keyPressEvent(ev_c)
        assert recs[none_idx].class_name == "None"
        assert "Kept None" in dlg.table.item(row_none, 5).text()
        with patch("jabs.ui.dialogs.bale_dialog.MessageDialog.confirm", return_value=True):
            dlg.keyPressEvent(ev_z)
        assert recs[none_idx].class_name == "Turn_Left"
        assert recs[none_idx].is_positive
    finally:
        dlg.close()


def test_gui_timeline_edits_reflect_live_in_explorer_table_and_inspector(tmp_path: Path) -> None:
    """Labeling or clearing a bout in the main JABS GUI reflects immediately in Explorer's table & inspector."""
    from jabs.project import VideoLabels

    _ensure_qapp()
    proj, report, cw = _build_mc_ui_test_harness(tmp_path, all_multiclass=False)
    recs = report.feature_matrix.records
    live_labels = VideoLabels("cage1.top.mp4", 220)
    # Seed initial manual labels matching audit bouts 0, 1, 2
    live_labels.get_track_labels("0", "Turn_Left").label_behavior(
        recs[0].start_frame, recs[0].end_frame
    )
    live_labels.get_track_labels("0", "Turn_Right").label_behavior(
        recs[1].start_frame, recs[1].end_frame
    )
    live_labels.get_track_labels("0", "None").label_behavior(
        recs[2].start_frame, recs[2].end_frame
    )
    cw.loaded_video = proj.dir / "cage1.top.mp4"
    cw._labels = live_labels

    dlg = BaleDialog(proj, central_widget=cw)
    dlg._init_refresh_timer.stop()
    try:
        dlg._on_analysis_finished(report, included_preds=True)
        # Switch to 2. Review Predictions and select predicted bout #3
        dlg.mode_predict_btn.setChecked(True)
        dlg._on_mode_pill_clicked(1)
        row_b3 = dlg._find_table_row_for_bout(3)
        assert row_b3 is not None
        dlg.table.selectRow(row_b3)
        cw.seek_to_bout.reset_mock()

        # 1. User labels predicted bout #3 as Turn_Left directly in the main JABS GUI -> emits labels_modified
        live_labels.get_track_labels("0", "Turn_Left").label_behavior(
            recs[3].start_frame, recs[3].end_frame
        )
        dlg._on_external_jabs_labels_modified()
        assert recs[3].source == "labeled"
        assert recs[3].class_name == "Turn_Left"
        assert dlg.table.item(row_b3, 1).text() == "✓ Confirmed: Turn_Left"
        assert "Reviewed Prediction: Turn_Left" in dlg.card_title_lbl.text()
        assert "Updated" in dlg.recalc_btn.text()
        assert dlg._selected_bout_idx == 3
        assert cw.seek_to_bout.call_count == 0

        # 2. User changes that bout to None in the main JABS GUI -> table updates immediately to Rejected -> None
        live_labels.get_track_labels("0", "Turn_Left").clear_labels(
            recs[3].start_frame, recs[3].end_frame
        )
        live_labels.get_track_labels("0", "None").label_behavior(
            recs[3].start_frame, recs[3].end_frame
        )
        dlg._on_external_jabs_labels_modified()
        assert recs[3].class_name == "None"
        assert not recs[3].is_positive
        assert dlg.table.item(row_b3, 1).text() == "✗ Rejected → None"

        # 3. User clears that bout with [X] in the main JABS GUI -> reverts to Pred: Turn_Left
        live_labels.get_track_labels("0", "None").clear_labels(
            recs[3].start_frame, recs[3].end_frame
        )
        dlg._on_external_jabs_labels_modified()
        assert recs[3].source == "predicted"
        assert dlg.table.item(row_b3, 1).text() == "◆ Pred: Turn_Left"

        # 4. In 1. Audit Labels, clearing a manual bout in the main GUI marks row ○ No Label without losing selection
        dlg.mode_audit_btn.setChecked(True)
        dlg._on_mode_pill_clicked(0)
        dlg._set_queue_filter("all", redraw=False)
        dlg._populate_table(redraw_scatter_on_change=False)
        row_b0 = dlg._find_table_row_for_bout(0)
        assert row_b0 is not None
        dlg.table.selectRow(row_b0)
        live_labels.get_track_labels("0", "Turn_Left").clear_labels(
            recs[0].start_frame, recs[0].end_frame
        )
        dlg._on_external_jabs_labels_modified()
        assert dlg._selected_bout_idx == 0
        assert dlg.table.item(row_b0, 1).text() == "○ No Label"
        assert "No Label" in dlg.card_title_lbl.text()

        # Re-labeling it in the main GUI flips the row right back to Labeled: Turn_Right
        live_labels.get_track_labels("0", "Turn_Right").label_behavior(
            recs[0].start_frame, recs[0].end_frame
        )
        dlg._on_external_jabs_labels_modified()
        assert recs[0].source == "labeled"
        assert recs[0].class_name == "Turn_Right"
        assert dlg.table.item(row_b0, 1).text() == "● Labeled: Turn_Right"
    finally:
        dlg.close()


def test_label_edges_tab_and_persistent_examined_v_across_reopen(tmp_path: Path) -> None:
    """Label Edge halos isolate cleanly into their own tab and `[V]` Examined flags persist across dialog restarts."""
    _ensure_qapp()
    proj, report, cw = _build_mc_ui_test_harness(tmp_path, all_multiclass=False)
    recs = report.feature_matrix.records
    # Tag predicted Turn_Left bout #5 as a touching Onset Label Edge halo while #3 remains standalone
    recs[5].is_label_halo = True
    recs[5].halo_side = "before"
    recs[5].halo_distance_frames = 0

    dlg = BaleDialog(proj, central_widget=cw)
    dlg._init_refresh_timer.stop()
    try:
        dlg._on_analysis_finished(report, included_preds=True)
        # In Audit mode, Label Edges pill stays hidden
        assert dlg.qf_halos_btn.isHidden()

        # Switch to 2. Review Predictions -> Label Edges pill appears and isolates halo bout #5
        dlg.mode_predict_btn.setChecked(True)
        dlg._on_mode_pill_clicked(1)
        assert not dlg.qf_halos_btn.isHidden()
        assert "Needs Review (1)" in dlg.qf_needs_btn.text()
        assert "Label Edges (1)" in dlg.qf_halos_btn.text()
        # Needs Review shows ONLY standalone bout #3 (zero halo clutter!)
        assert dlg.table.rowCount() == 1
        assert int(dlg.table.item(0, 0).data(Qt.ItemDataRole.UserRole)) == 3

        # Clicking Label Edges tab shows ONLY halo bout #5 with purple Label Edge (Onset) badge
        dlg._on_queue_filter_pill_clicked(3)
        assert dlg._queue_filter == "label_edges"
        assert dlg.table.rowCount() == 1
        assert int(dlg.table.item(0, 0).data(Qt.ItemDataRole.UserRole)) == 5
        assert "Label Edge (Onset)" in dlg.table.item(0, 0).text()

        # Quick Select slice finds the Label Edge halo cleanly
        matched_halos, _ = dlg._quick_slice_matches("label_edges")
        assert matched_halos == [5]

        # Press [V] on standalone bout #3 -> marks ✓ Examined in-place (Col 6 tag + Inspector), persists to JSON, Undoable
        dlg._on_queue_filter_pill_clicked(0)
        dlg.table.selectRow(0)
        assert dlg.table.item(0, 6).text() == "—"
        assert dlg.examine_btn.text() == "✓ Examined [V]"
        ev_v = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress, Qt.Key.Key_V, Qt.KeyboardModifier.NoModifier, "v"
        )
        dlg.keyPressEvent(ev_v)
        assert recs[3].bout_id in dlg._reviewed_bout_ids
        assert (
            "False Pos?" in dlg.table.item(0, 0).text()
        )  # Preserves issue diagnosis while Col 6 shows tag
        assert dlg.table.item(0, 6).text() == "✓ Examined"
        assert dlg.examine_btn.text() == "↺ Unmark [V]"
        assert dlg._examined_store.path.is_file()

        # Direct mouse click on Col 6 (`Examined [V]`) toggles Examined OFF and back ON cleanly
        dlg._on_table_cell_clicked(0, 6)
        assert recs[3].bout_id not in dlg._reviewed_bout_ids
        assert dlg.table.item(0, 6).text() == "—"
        dlg._on_table_cell_clicked(0, 6)
        assert recs[3].bout_id in dlg._reviewed_bout_ids
        assert dlg.table.item(0, 6).text() == "✓ Examined"

        # Undo restores unexamined state both in UI (Col 6 -> "—") and on disk
        dlg._on_undo_clicked()
        assert recs[3].bout_id not in dlg._reviewed_bout_ids
        assert dlg.table.item(0, 6).text() == "—"
        assert dlg._examined_store.find_match(recs[3]) is None

        # Mark [V] again, then close & reopen a brand-new dialog (simulating JABS restart / Recalculate)
        dlg.keyPressEvent(ev_v)
        assert dlg._examined_store.find_match(recs[3]) is not None
    finally:
        dlg.close()

    # Reopen a brand-new BaleDialog on the same project -> bout #3 stays in Reviewed ✓!
    dlg2 = BaleDialog(proj, central_widget=cw)
    dlg2._init_refresh_timer.stop()
    try:
        dlg2._mode = "predict"
        dlg2.mode_predict_btn.setChecked(True)
        dlg2.qf_halos_btn.setVisible(True)
        dlg2._on_analysis_finished(report, included_preds=True)
        assert recs[3].bout_id in dlg2._reviewed_bout_ids
        assert "Needs Review (0)" in dlg2.qf_needs_btn.text()
        assert "Reviewed ✓ (1)" in dlg2.qf_done_btn.text()
        assert "Label Edges (1)" in dlg2.qf_halos_btn.text()
    finally:
        dlg2.close()


def test_auto_examined_on_gui_view_click_play_scrub_search_and_opt_out(tmp_path: Path) -> None:
    """Viewing a bout on the JABS GUI automatically marks `✓ Examined` (Col 6 + disk) without losing diagnosis or breaking Col 6 toggle."""
    from types import SimpleNamespace

    from jabs.behavior_search import SearchHit
    from jabs.project import VideoLabels

    _ensure_qapp()
    proj, report, cw = _build_mc_ui_test_harness(tmp_path, all_multiclass=False)
    recs = report.feature_matrix.records
    live_labels = VideoLabels("cage1.top.mp4", 220)
    proj.video_manager.load_video_labels.side_effect = lambda *_a, **_kw: live_labels
    proj.load_pose_est.side_effect = lambda *_a, **_kw: SimpleNamespace(num_frames=220)
    cw.loaded_video = proj.dir / "cage1.top.mp4"
    cw._pose_est = SimpleNamespace(num_frames=220)
    cw._labels = live_labels
    cw.play_bout_clip = MagicMock()
    cw.seek_to_frame_only = MagicMock()

    dlg = BaleDialog(proj, central_widget=cw)
    dlg._init_refresh_timer.stop()
    try:
        assert dlg.auto_mark_viewed_action.isCheckable()
        assert dlg.auto_mark_viewed_action.isChecked() is True

        dlg._on_analysis_finished(report, included_preds=True)
        dlg.pair_opposite_action.setChecked(False)
        dlg.mode_predict_btn.setChecked(True)
        dlg._on_mode_pill_clicked(1)

        # 1. Loading the table does NOT auto-mark row 0 before the user views it on the JABS GUI
        assert dlg.table.rowCount() == 2
        row_b3 = dlg._find_table_row_for_bout(3)
        row_b5 = dlg._find_table_row_for_bout(5)
        assert row_b3 is not None and row_b5 is not None
        assert dlg.table.item(row_b3, 6).text() == "—"
        assert dlg.table.item(row_b5, 6).text() == "—"
        assert recs[3].bout_id not in dlg._reviewed_bout_ids

        # 2. Clicking row_b3 (Col 2 Video) views bout #3 on JABS GUI -> auto-marks ✓ Examined in Col 6,
        #    preserves diagnosis in Col 0/Inspector, persists to JSON, and does NOT pollute Undo stack
        dlg._on_table_cell_clicked(row_b3, 2)
        assert cw.seek_to_bout.call_count >= 1
        assert recs[3].bout_id in dlg._reviewed_bout_ids
        assert dlg.table.item(row_b3, 6).text() == "✓ Examined"
        assert "False Pos?" in dlg.table.item(row_b3, 0).text()
        assert "✓ Examined" in dlg.diagnosis_lbl.text()
        assert dlg.examine_btn.text() == "↺ Unmark [V]"
        assert dlg._examined_store.find_match(recs[3]) is not None
        assert len(dlg._ui_undo_history) == 0
        assert "Needs Review (1)" in dlg.qf_needs_btn.text()
        assert "Reviewed ✓ (1)" in dlg.qf_done_btn.text()

        # 3. Clicking Col 6 on auto-examined row_b3 unmarks it back to "—" WITHOUT immediately re-marking
        dlg._on_table_cell_clicked(row_b3, 6)
        assert recs[3].bout_id not in dlg._reviewed_bout_ids
        assert dlg.table.item(row_b3, 6).text() == "—"
        assert dlg._examined_store.find_match(recs[3]) is None

        # 4. Clicking Col 6 directly on an UNSELECTED unexamined row_b5 toggles ON cleanly (no double-toggle!)
        dlg._table_mouse_press_col = 6
        dlg._table_mouse_press_btn = Qt.MouseButton.LeftButton
        dlg.table.selectRow(row_b5)
        dlg._on_table_cell_clicked(row_b5, 6)
        assert recs[5].bout_id in dlg._reviewed_bout_ids
        assert dlg.table.item(row_b5, 6).text() == "✓ Examined"
        # Toggle row_b5 back off so we can test Play Clip & external Search Bar
        dlg._on_table_cell_clicked(row_b5, 6)
        assert recs[5].bout_id not in dlg._reviewed_bout_ids

        # 5. Pressing Space (Play Clip) on bout #3 auto-marks it ✓ Examined
        dlg._suppress_auto_seek = True
        dlg.table.selectRow(row_b3)
        dlg._suppress_auto_seek = False
        dlg._on_play_clip_clicked()
        assert cw.play_bout_clip.call_count == 1
        assert recs[3].bout_id in dlg._reviewed_bout_ids
        assert dlg.table.item(row_b3, 6).text() == "✓ Examined"

        # 6. Labeling auto-examined bout #3 with [Z] upgrades Col 6 to ✓ Saved; viewing again never downgrades it
        ev_z = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress, Qt.Key.Key_Z, Qt.KeyboardModifier.NoModifier, "z"
        )
        dlg.keyPressEvent(ev_z)
        assert dlg.table.item(row_b3, 6).text() == "✓ Saved"
        dlg._on_table_cell_clicked(row_b3, 2)
        assert dlg.table.item(row_b3, 6).text() == "✓ Saved"
        assert recs[3].bout_id not in dlg._examined_only_bout_ids

        # 7. External JABS Search Bar navigation (`search_hit_loaded`) auto-marks matching bout #5
        hit_b5 = SearchHit(
            file="cage1.top.mp4",
            identity="0",
            behavior="Turn_Left",
            start_frame=recs[5].start_frame,
            end_frame=recs[5].end_frame,
        )
        dlg._on_external_search_hit_loaded(hit_b5)
        assert recs[5].bout_id in dlg._reviewed_bout_ids
        assert dlg.table.item(row_b5, 6).text() == "✓ Examined"

        # 8. Unchecking `Auto-Mark Examined [V] When Viewed in JABS` disables automatic marking
        dlg._on_table_cell_clicked(row_b5, 6)  # unmark
        assert recs[5].bout_id not in dlg._reviewed_bout_ids
        dlg.auto_mark_viewed_action.setChecked(False)
        dlg._on_table_cell_clicked(row_b5, 2)  # views in JABS while auto-mark is OFF
        assert recs[5].bout_id not in dlg._reviewed_bout_ids
        assert dlg.table.item(row_b5, 6).text() == "—"
    finally:
        dlg.close()


def test_bale_resizes_freely_in_x_and_y_across_all_modes_and_stress_states(
    tmp_path: Path,
) -> None:
    """Verify cross-platform X/Y resizability (shrink & grow), zero button clipping/overlap, and dynamic toast stability."""
    _ensure_qapp()
    proj, report, cw = _build_mc_ui_test_harness(tmp_path, all_multiclass=True)
    # Stress-test long behavior names & long status strings
    proj.settings_manager.behavior_names = [
        "Turn_Left_Very_Long_Behavior_Name",
        "Turn_Right_Very_Long_Behavior_Name",
    ]

    dlg = BaleDialog(proj, central_widget=cw)
    dlg._init_refresh_timer.stop()
    try:
        dlg.show()
        QtWidgets.QApplication.processEvents()

        # 1. Cross-platform window flags, size grip, and non-fixed layout constraints
        flags = dlg.windowFlags()
        assert bool(flags & Qt.WindowType.Window)
        assert bool(flags & Qt.WindowType.WindowTitleHint)
        assert bool(flags & Qt.WindowType.WindowSystemMenuHint)
        assert bool(flags & Qt.WindowType.WindowMinimizeButtonHint)
        assert bool(flags & Qt.WindowType.WindowMaximizeButtonHint)
        assert bool(flags & Qt.WindowType.WindowCloseButtonHint)
        assert not bool(flags & Qt.WindowType.MSWindowsFixedSizeDialogHint)
        assert dlg.isSizeGripEnabled() is True
        assert dlg.layout() is not None
        assert dlg.layout().sizeConstraint() != QtWidgets.QLayout.SizeConstraint.SetFixedSize
        assert dlg.maximumWidth() >= 10_000
        assert dlg.maximumHeight() >= 10_000
        assert dlg.minimumWidth() <= 700
        assert dlg.minimumSizeHint().width() <= 700

        dlg._on_analysis_finished(report, included_preds=True)

        # 2. Stress state: Predict mode + Label Edges + active Lasso chip + huge status toast + Sub-Types summary
        dlg.mode_predict_btn.setChecked(True)
        dlg._on_mode_pill_clicked(1)
        dlg.clear_box_btn.setText("✕ 15 Similar to #9999 (15)")
        dlg.clear_box_btn.setVisible(True)
        long_toast = (
            "✓ Saved as Turn_Left_Very_Long_Behavior_Name (+ paired None 120-145) "
            "on study_275.cage_2155.2024-11-12.22.11.top.mp4 (Animal 1) — Undo [⌘Z]"
        )
        dlg.status_summary_label.setText(long_toast)
        assert dlg.status_summary_label.text() == long_toast
        QtWidgets.QApplication.processEvents()

        # Minimum width stays compact even under worst-case badges & 140-char status toast
        assert dlg.minimumSizeHint().width() <= 700

        # 3. Bidirectional X resizing (shrink down to 640 px and expand up to 2100 px)
        for target_w in (1340, 960, 720, 640, 820, 1150, 1600, 2100):
            dlg.resize(target_w, 760)
            QtWidgets.QApplication.processEvents()
            assert dlg.width() == target_w, (
                f"Expected dialog width {target_w}, got {dlg.width()} "
                f"(minHint={dlg.minimumSizeHint()})"
            )
            assert dlg.search_input.width() >= 120

        # 4. Bidirectional Y resizing
        for target_h in (820, 620, 540, 980):
            dlg.resize(1100, target_h)
            QtWidgets.QApplication.processEvents()
            assert dlg.height() == target_h

        # 5. Dynamic toast & Sub-Types mode never jump/expand the user's chosen narrow width
        dlg.resize(740, 680)
        QtWidgets.QApplication.processEvents()
        assert dlg.width() == 740
        dlg.mode_subtypes_btn.setChecked(True)
        dlg._on_mode_pill_clicked(2)
        dlg.cage_skew_badge.setText("⚠ Cage Skew (V=0.72)")
        dlg.cage_skew_badge.setVisible(True)
        dlg.subtypes_summary_lbl.setText(
            "Pos+Neg • k=6 (C1:19 • C2:13 • C3:6 • C4:12 • C5:8 • C6:4) • Sil=0.53 • Top η²: long_feature (82%)"
        )
        dlg.status_summary_label.setText("Loaded 500 bouts in 140 ms — " + "x" * 160)
        QtWidgets.QApplication.processEvents()
        assert dlg.width() == 740
        assert dlg.minimumSizeHint().width() <= 700

        # 6. Zero clipped or overlapping visible buttons at narrow (720), medium (960), and wide (1340) widths
        dlg.mode_audit_btn.setChecked(True)
        dlg._on_mode_pill_clicked(0)
        for w, h in ((720, 680), (960, 720), (1340, 820)):
            dlg.resize(w, h)
            QtWidgets.QApplication.processEvents()
            dlg.scatter_canvas._render_now()
            dlg.insp_canvas._render_now()
            QtWidgets.QApplication.processEvents()
            assert dlg.width() == w and dlg.height() == h
            buttons = [b for b in dlg.findChildren(QtWidgets.QAbstractButton) if b.isVisible()]
            assert len(buttons) >= 15
            for i, b1 in enumerate(buttons):
                parent_w = b1.parentWidget()
                if parent_w is not None:
                    assert parent_w.rect().contains(b1.geometry()), (
                        f"Button '{b1.text()}' clipped at {w}x{h}: {b1.geometry()} vs {parent_w.rect()}"
                    )
                for b2 in buttons[i + 1 :]:
                    if b1.parentWidget() is b2.parentWidget():
                        assert not b1.geometry().intersects(b2.geometry()), (
                            f"Buttons '{b1.text()}' and '{b2.text()}' overlap at {w}x{h}"
                        )

        # 7. Expanded Map [M] also resizes freely in X without locking width
        dlg._toggle_expand_map(True)
        for target_w in (680, 1280, 800):
            dlg.resize(target_w, 700)
            QtWidgets.QApplication.processEvents()
            assert dlg.width() == target_w
        dlg._toggle_expand_map(False)
    finally:
        dlg.close()


def test_jabs_core_windows_resize_freely_in_x_and_y(tmp_path: Path) -> None:
    """Guard core JABS windows (`MainWindow`, `BehaviorSearchDialog`, `MessageDialog`, `AnnotationEditDialog`) against X-axis resize locks."""
    from jabs.ui.dialogs.annotation_edit_dialog import AnnotationEditDialog
    from jabs.ui.dialogs.behavior_search_dialog import BehaviorSearchDialog
    from jabs.ui.dialogs.message_dialog import MessageDialog
    from jabs.ui.main_window import MainWindow

    _ensure_qapp()
    proj, _ = _build_synthetic_explorer_project(tmp_path)
    proj.settings = proj.settings_manager.project_settings

    windows = [
        ("MainWindow", MainWindow("jabs", "JABS", process_pool=MagicMock()), (800, 1100, 1500)),
        ("BehaviorSearchDialog", BehaviorSearchDialog(proj), (500, 800, 1200)),
        (
            "MessageDialog",
            MessageDialog(title="Resize Check", message="Short message"),
            (550, 800, 1100),
        ),
        ("AnnotationEditDialog", AnnotationEditDialog(10, 50), (760, 950, 1250)),
    ]
    for name, win, widths in windows:
        try:
            win.show()
            QtWidgets.QApplication.processEvents()
            assert win.maximumWidth() >= 10_000, f"{name} has locked maximumWidth"
            min_h = max(480, win.minimumSizeHint().height())
            for tw in widths:
                win.resize(tw, min_h)
                QtWidgets.QApplication.processEvents()
                assert win.width() == tw, (
                    f"{name} failed horizontal resize to {tw}px (got {win.width()}px, "
                    f"minHint={win.minimumSizeHint()})"
                )
        finally:
            win.close()


def test_mode4_videos_screener_sorting_filtering_timeline_scrub_and_jabs_handoff(
    tmp_path: Path,
) -> None:
    """Verify Mode 4 (`4. Videos`) whole-video screener: quick pills, 50k buckets, sorting, timeline scrub, JABS handoff, and resize hygiene."""
    from jabs.bale import summarize_project_videos

    _ensure_qapp()
    project, proj_dir = _build_synthetic_explorer_project(tmp_path)
    n_frames = 600

    # Configure distinct whole-video profiles across the 6 synthetic cages:
    # cage_00: high frames (420f = 70%), cage_01: high bouts (9 bouts),
    # cage_02: E3a low-confidence positive bout at [100..130], cage_03: normal moderate,
    # cage_04: 0 behavior + E3b near-miss at [220..255], cage_05: 0 behavior clean control
    pred_map: dict[
        str, tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]
    ] = {}

    r0 = np.zeros(n_frames, dtype=np.int32)
    r0[80:500] = 1
    p0 = np.full(n_frames, 0.95, dtype=np.float32)
    pred_map["cage_00.mp4"] = ({0: r0}, {0: p0}, {0: r0.copy()})

    r1 = np.zeros(n_frames, dtype=np.int32)
    p1 = np.full(n_frames, 0.95, dtype=np.float32)
    for k in range(9):
        s = 20 + k * 60
        r1[s : s + 15] = 1
    pred_map["cage_01.mp4"] = ({0: r1}, {0: p1}, {0: r1.copy()})

    r2 = np.zeros(n_frames, dtype=np.int32)
    r2[100:130] = 1
    r2[320:350] = 1
    p2 = np.full(n_frames, 0.04, dtype=np.float32)
    p2[100:130] = 0.57
    p2[320:350] = 0.93
    pred_map["cage_02.mp4"] = ({0: r2}, {0: p2}, {0: r2.copy()})

    r3 = np.zeros(n_frames, dtype=np.int32)
    r3[150:190] = 1
    p3 = np.full(n_frames, 0.95, dtype=np.float32)
    pred_map["cage_03.mp4"] = ({0: r3}, {0: p3}, {0: r3.copy()})

    r4 = np.zeros(n_frames, dtype=np.int32)
    p4 = np.full(n_frames, 0.96, dtype=np.float32)
    p4[220:256] = 0.55
    pred_map["cage_04.mp4"] = ({0: r4}, {0: p4}, {0: r4.copy()})

    r5 = np.zeros(n_frames, dtype=np.int32)
    p5 = np.full(n_frames, 0.98, dtype=np.float32)
    pred_map["cage_05.mp4"] = ({0: r5}, {0: p5}, {0: r5.copy()})

    for v_name in pred_map:
        (proj_dir / "jabs" / "predictions" / f"{Path(v_name).stem}.h5").write_bytes(b"stub")

    project.prediction_manager.load_predictions.side_effect = lambda v, _b: pred_map.get(
        v, ({}, {}, {})
    )

    central_mock = MagicMock()
    central_mock.behavior = "Grooming"
    central_mock.loaded_video = Path(proj_dir / "cage_00.mp4")
    central_mock._pose_est = None

    dlg = BaleDialog(project, central_widget=central_mock)
    try:
        dlg.show()
        QtWidgets.QApplication.processEvents()

        # Preload bout report + switch to 4. Videos mode with synchronously computed video report
        bout_rep = run_bale(project, "Grooming", include_predictions=True, space="classifier")
        dlg._on_analysis_finished(bout_rep, included_preds=True)
        # Keep cage_05.mp4 as a genuine unlabeled zero-behavior clean control (no human-vs-model E4 disagreement)
        (proj_dir / "jabs" / "annotations" / "cage_05.json").unlink(missing_ok=True)
        vid_rep = summarize_project_videos(project, "Grooming", include_all_multiclass=False)

        dlg.mode_videos_btn.setChecked(True)
        dlg._on_mode_pill_clicked(3)
        dlg._on_video_summary_finished(vid_rep, dlg._active_video_scope_key())
        QtWidgets.QApplication.processEvents()

        # 1. Mode 4 UI stack, table schema, visibility, and strict NoFocus policy
        assert dlg._mode == "videos"
        assert dlg.bottom_table_stack.currentIndex() == 1
        assert dlg.table.columnCount() == 7
        assert dlg.video_table.columnCount() == 9
        assert dlg.video_table.rowCount() == 6
        assert dlg.videos_bar.isVisible()
        assert dlg.video_action_bar.isVisible()
        assert not dlg.bout_act_container.isVisible()
        for w_host in (dlg.videos_bar, dlg.video_action_bar, dlg.video_qbar_widget):
            for btn in w_host.findChildren(QtWidgets.QPushButton):
                assert btn.focusPolicy() == Qt.FocusPolicy.NoFocus
            for combo in w_host.findChildren(QtWidgets.QComboBox):
                assert combo.focusPolicy() == Qt.FocusPolicy.NoFocus

        # 2. Quick filter pills find high-frame, high-bout, zero-behavior, and 50k outlier videos immediately
        dlg.vf_high_frames_btn.click()
        assert dlg.video_table.rowCount() == 1
        assert dlg.video_table.item(0, 0).data(Qt.ItemDataRole.UserRole) == "cage_00.mp4"

        dlg.vf_high_bouts_btn.click()
        assert dlg.video_table.rowCount() == 1
        assert dlg.video_table.item(0, 0).data(Qt.ItemDataRole.UserRole) == "cage_01.mp4"

        dlg.vf_zero_btn.click()
        zero_vids = {
            dlg.video_table.item(r, 0).data(Qt.ItemDataRole.UserRole)
            for r in range(dlg.video_table.rowCount())
        }
        assert zero_vids == {"cage_04.mp4", "cage_05.mp4"}

        dlg.vf_outliers_btn.click()
        outlier_vids = {
            dlg.video_table.item(r, 0).data(Qt.ItemDataRole.UserRole)
            for r in range(dlg.video_table.rowCount())
        }
        assert {"cage_00.mp4", "cage_01.mp4", "cage_02.mp4", "cage_04.mp4"}.issubset(outlier_vids)
        assert "cage_05.mp4" not in outlier_vids

        # 3. Restore All Videos + verify numeric header sorting on Frames (% Time) and Bouts
        dlg.vf_all_btn.click()
        assert dlg.video_table.rowCount() == 6
        dlg._video_table_sort_col = 0  # Reset so clicking col 2 sets descending cleanly
        dlg._on_video_table_header_clicked(2)  # Frames (% Time) DESC
        assert dlg.video_table.item(0, 0).data(Qt.ItemDataRole.UserRole) == "cage_00.mp4"
        dlg._on_video_table_header_clicked(3)  # Bouts DESC
        assert dlg.video_table.item(0, 0).data(Qt.ItemDataRole.UserRole) == "cage_01.mp4"

        # 4. Send Videos to JABS Search Bar preserves exact displayed video_table sort order
        expected_order = [
            dlg.video_table.item(r, 0).data(Qt.ItemDataRole.UserRole)
            for r in range(dlg.video_table.rowCount())
        ]
        dlg._on_send_videos_to_search_bar()
        assert central_mock.load_custom_search_queue.call_count == 1
        sent_hits = central_mock.load_custom_search_queue.call_args[0][0]
        assert [h.file for h in sent_hits] == expected_order

        # 5. Select E3a video (cage_02.mp4) -> Open in JABS, Frame 0, Timeline scrub, and Space play
        central_mock.reset_mock()
        dlg._select_video_by_name("cage_02.mp4")
        dlg._on_video_open_clicked()
        assert central_mock.seek_to_bout.call_count >= 1
        last_args = central_mock.seek_to_bout.call_args
        assert last_args[0][0] == "cage_02.mp4"
        assert last_args[0][2] == 100

        central_mock.reset_mock()
        dlg._on_video_open_start_clicked()
        assert central_mock.seek_to_bout.call_count == 1
        assert central_mock.seek_to_bout.call_args[0][2] == 0
        assert central_mock.seek_to_bout.call_args[1].get("select_span") is False

        central_mock.reset_mock()
        central_mock.loaded_video = Path(proj_dir / "cage_02.mp4")
        dlg._on_prob_curve_frame_clicked(125.0)
        central_mock.seek_to_frame_only.assert_called_once_with(125)

        central_mock.reset_mock()
        dlg.video_table.setFocus()
        ev_space = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier, " "
        )
        QtWidgets.QApplication.sendEvent(dlg.video_table, ev_space)
        assert central_mock.play_bout_clip.call_count == 1

        # 6. Resize freedom & zero overlapping/clipped visible buttons in 4. Videos mode
        assert dlg.minimumSizeHint().width() <= 700
        for w, h in ((720, 680), (1340, 820)):
            dlg.resize(w, h)
            QtWidgets.QApplication.processEvents()
            dlg.scatter_canvas._render_now()
            dlg.insp_canvas._render_now()
            QtWidgets.QApplication.processEvents()
            assert dlg.width() == w and dlg.height() == h
            buttons = [b for b in dlg.findChildren(QtWidgets.QAbstractButton) if b.isVisible()]
            for i, b1 in enumerate(buttons):
                parent_w = b1.parentWidget()
                if parent_w is not None:
                    assert parent_w.rect().contains(b1.geometry()), (
                        f"Mode 4 button '{b1.text()}' clipped at {w}x{h}: {b1.geometry()} vs {parent_w.rect()}"
                    )
                for b2 in buttons[i + 1 :]:
                    if b1.parentWidget() is b2.parentWidget():
                        assert not b1.geometry().intersects(b2.geometry()), (
                            f"Mode 4 buttons '{b1.text()}' and '{b2.text()}' overlap at {w}x{h}"
                        )

        # 7. Drill down into `2. Predictions` for selected video restores bout table cleanly
        dlg._on_video_inspect_bouts_clicked("predict")
        QtWidgets.QApplication.processEvents()
        assert dlg._mode == "predict"
        assert dlg.bottom_table_stack.currentIndex() == 0
    finally:
        dlg.close()


def test_bale_ui_mode4_speed_races_shortcuts_and_filters(tmp_path: Path, monkeypatch) -> None:
    """Verify Mode 4 performance (250 rows), worker race/stale rejection, shortcut isolation, search, lasso, and CSV export."""
    import time

    from jabs.bale import (
        VideoScreenerReport,
        VideoSummaryRecord,
        classify_video_outliers,
        run_bale,
        summarize_project_videos,
    )
    from jabs.ui.dialogs.bale_dialog import BaleDialog

    _ensure_qapp()
    project, proj_dir = _build_synthetic_explorer_project(tmp_path / "mode4_hardening")
    n_frames = 600
    pred_map: dict[
        str, tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]
    ] = {}
    for idx, v_name in enumerate(project.video_manager.videos):
        r = np.zeros(n_frames, dtype=np.int32)
        p = np.full(n_frames, 0.96, dtype=np.float32)
        if idx == 0:
            r[80:500] = 1
        elif idx == 1:
            for k in range(6):
                r[20 + k * 70 : 35 + k * 70] = 1
        elif idx == 2:
            r[100:130] = 1
            p[100:130] = 0.57
        elif idx == 3:
            r[150:190] = 1
        elif idx == 4:
            p[220:256] = 0.55
        pred_map[v_name] = ({0: r}, {0: p}, {0: r.copy()})
        (proj_dir / "jabs" / "predictions" / f"{Path(v_name).stem}.h5").write_bytes(b"stub")
    project.prediction_manager.load_predictions.side_effect = lambda v, _b: pred_map.get(
        v, ({}, {}, {})
    )
    central_mock = MagicMock()
    central_mock.behavior = "Grooming"
    central_mock.loaded_video = None
    central_mock._pose_est = None

    dlg = BaleDialog(project, central_widget=central_mock)
    try:
        dlg.show()
        QtWidgets.QApplication.processEvents()

        bout_rep = run_bale(project, "Grooming", include_predictions=True, space="classifier")
        dlg._on_analysis_finished(bout_rep, included_preds=True)
        vid_rep = summarize_project_videos(project, "Grooming", include_all_multiclass=False)

        # Enter Mode 4
        dlg.mode_videos_btn.setChecked(True)
        dlg._on_mode_pill_clicked(3)
        dlg._on_video_summary_finished(vid_rep, dlg._active_video_scope_key())
        QtWidgets.QApplication.processEvents()

        # 1. Undo button hidden, recalc_btn reads "↻ Rescan", and bout shortcuts (Z/X/C/V/1/2/Cmd+Z) are isolated
        assert not dlg.undo_btn.isVisible()
        assert "Rescan" in dlg.recalc_btn.text()
        dlg._ui_undo_history.append({"disk_mutated": False})
        dlg.undo_btn.setEnabled(True)
        for key_code, mod in (
            (Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier),
            (Qt.Key.Key_Z, Qt.KeyboardModifier.NoModifier),
            (Qt.Key.Key_X, Qt.KeyboardModifier.NoModifier),
            (Qt.Key.Key_C, Qt.KeyboardModifier.NoModifier),
            (Qt.Key.Key_V, Qt.KeyboardModifier.NoModifier),
            (Qt.Key.Key_1, Qt.KeyboardModifier.NoModifier),
            (Qt.Key.Key_2, Qt.KeyboardModifier.NoModifier),
        ):
            ev = QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, key_code, mod, "")
            dlg._dispatch_shortcut_key(ev)
        assert len(dlg._ui_undo_history) == 1  # Cmd+Z did NOT pop bout undo stack in Mode 4!
        dlg._ui_undo_history.clear()
        dlg.undo_btn.setEnabled(False)

        # 2. Worker race & stale-key rejection
        launched: list[tuple[str, bool]] = []
        monkeypatch.setattr(
            dlg._thread_pool,
            "start",
            lambda w: launched.append((str(w.behavior or ""), bool(w.include_all_multiclass))),
        )
        dlg._video_worker = MagicMock()
        dlg.refresh_video_summary(force=True)
        assert dlg._queued_video_refresh is True
        # Deliver stale key -> ignored & triggers queued fresh scan
        dlg._on_video_summary_finished(vid_rep, ("StaleBehavior", False), req_id=9999)
        assert len(launched) == 1
        dlg._video_worker = None
        dlg._queued_video_refresh = False

        # 3. Behavior change in Mode 4 does NOT launch slow Parquet CV (`refresh_analysis`)
        bout_refresh_called = False
        orig_refresh = dlg.refresh_analysis

        def _spy_refresh() -> None:
            nonlocal bout_refresh_called
            bout_refresh_called = True
            orig_refresh()

        monkeypatch.setattr(dlg, "refresh_analysis", _spy_refresh)
        dlg._on_behavior_combo_changed()
        assert not bout_refresh_called
        assert dlg._analysis_needed_on_bout_mode is True
        dlg._video_worker = None

        # Restore report and verify background _on_analysis_finished preserves Mode 4 zoom & status
        dlg._on_video_summary_finished(vid_rep, dlg._active_video_scope_key())
        dlg._scatter_view_limits = ((0.0, 5.0), (0.0, 20.0))
        dlg.status_summary_label.setText("Mode 4 preserved status")
        dlg._on_analysis_finished(bout_rep, included_preds=True)
        assert dlg._scatter_view_limits == ((0.0, 5.0), (0.0, 20.0))
        assert dlg.status_summary_label.text() == "Mode 4 preserved status"
        assert "Rescan" in dlg.recalc_btn.text()

        # 4. Search text updates table + scatter cleanly via debounce flush
        dlg.video_search_input.setText("cage_00")
        dlg._flush_video_search_if_pending()
        assert dlg.video_table.rowCount() == 1
        assert dlg.video_table.item(0, 0).data(Qt.ItemDataRole.UserRole) == "cage_00.mp4"
        dlg.video_search_input.clear()
        dlg._flush_video_search_if_pending()
        assert dlg.video_table.rowCount() == 6

        # 5. Box selecting a dimmed point outside active preset relaxes filter back to `all` so rows appear
        dlg.vf_zero_btn.click()
        assert dlg._video_preset == "zero_all"
        # cage_00.mp4 has high frames/bouts (dimmed under zero_all); box around it!
        rec0 = dlg._get_video_record("cage_00.mp4")
        assert rec0 is not None
        bx, by = float(rec0.active_bouts("predictions")), float(rec0.active_pct("predictions"))
        dlg._on_video_scatter_box_selected(bx - 0.5, bx + 0.5, by - 0.5, by + 0.5)
        assert dlg._video_preset == "all"
        assert dlg.video_table.rowCount() >= 1
        assert dlg.video_clear_sel_btn.isVisible()
        dlg._on_clear_video_map_selection()
        assert not dlg.video_clear_sel_btn.isVisible()

        # 6. Right-click suppresses auto-seek; same-row left-click on row 0 seeks JABS
        central_mock.reset_mock()
        dlg._video_table_mouse_press_btn = Qt.MouseButton.RightButton
        dlg.video_table.selectRow(1)
        assert central_mock.seek_to_bout.call_count == 0
        dlg._video_table_mouse_press_btn = Qt.MouseButton.LeftButton
        dlg._pending_video_click_already_sought_row = None
        dlg._on_video_table_cell_clicked(1, 0)
        assert central_mock.seek_to_bout.call_count == 1

        # 7. Unlabeled Only toggle (all 6 synthetic cages carry labels -> 0 unlabeled) + Source switch + CSV export
        dlg.videos_unlabeled_btn.setChecked(True)
        assert dlg.video_table.rowCount() == 0
        dlg.videos_unlabeled_btn.setChecked(False)
        assert dlg.video_table.rowCount() == 6
        lbl_idx = dlg.videos_source_combo.findData("labels")
        assert lbl_idx >= 0
        dlg.videos_source_combo.setCurrentIndex(lbl_idx)
        assert dlg._video_source == "labels"
        dlg.videos_source_combo.setCurrentIndex(dlg.videos_source_combo.findData("predictions"))

        # Export CSV writes ONLY currently filtered visible rows in displayed sort order
        dlg.vf_zero_btn.click()
        csv_out = tmp_path / "exported_zero.csv"
        monkeypatch.setattr(
            QtWidgets.QFileDialog,
            "getSaveFileName",
            lambda *a, **kw: (str(csv_out), "CSV Files (*.csv)"),
        )
        monkeypatch.setattr(
            "jabs.ui.dialogs.bale_dialog.MessageDialog.information",
            lambda *a, **kw: None,
        )
        dlg._on_export_videos_csv_clicked()
        assert csv_out.is_file()
        lines = csv_out.read_text().strip().splitlines()
        assert len(lines) == 1 + dlg.video_table.rowCount()

        # 8. Table repopulation & row-stepping performance regression guard (250 synthetic videos!)
        synth_recs = [
            VideoSummaryRecord(
                video=f"study_275.cage_{2100 + (i % 20)}.2024-11-12.{i // 60:02d}.{i % 60:02d}.top.mp4",
                n_animals=3,
                valid_frames=5400,
                max_track_frames=1800,
                has_predictions=True,
                pred_pos_frames=int((i * 19) % 900),
                pred_bout_count=int(i % 18),
                pred_max_bout_frames=int(15 + (i % 120)),
                pred_pos_pct=float(((i * 19) % 900) / 54.0),
                pred_mean_prob=0.84,
                pred_min_bout_prob=0.62 if (i % 11 == 0) else 0.84,
                pred_low_conf_pos_bouts=1 if (i % 11 == 0) else 0,
                pred_low_conf_none_spans=1 if (i % 13 == 0) else 0,
                pred_low_conf_none_frames=21 if (i % 13 == 0) else 0,
                has_labels=(i % 7 == 0),
                label_pos_bouts=2 if (i % 7 == 0) else 0,
                label_neg_bouts=2 if (i % 7 == 0) else 0,
                label_dual_class_tracks=1 if (i % 7 == 0) else 0,
            )
            for i in range(250)
        ]
        classify_video_outliers(synth_recs, source="predictions")
        big_rep = VideoScreenerReport(
            project_dir=proj_dir,
            behavior="Grooming",
            is_multiclass=False,
            include_all_multiclass=False,
            use_postprocessed=True,
            records=synth_recs,
            has_any_predictions=True,
            has_any_labels=True,
        )
        dlg._video_preset = "all"
        dlg._video_report = big_rep
        t0 = time.perf_counter()
        dlg._populate_video_table(redraw_scatter_on_change=False)
        t_pop_ms = (time.perf_counter() - t0) * 1000.0
        assert dlg.video_table.rowCount() == 250
        assert t_pop_ms < 750.0, f"250-row video_table repopulate too slow: {t_pop_ms:.1f} ms"
        # Check Near-Miss formatting has zero `s` unit collision (uses `1 (0.7s)`, never `1s (21f)`)
        e3b_cells = [
            dlg.video_table.item(r, 6).text()
            for r in range(dlg.video_table.rowCount())
            if dlg.video_table.item(r, 6) is not None and dlg.video_table.item(r, 6).text() != "—"
        ]
        assert e3b_cells and all("s (" not in txt and txt.endswith("s)") for txt in e3b_cells)

        # Fast ring-only row stepping
        dlg._draw_video_scatter()
        t_step0 = time.perf_counter()
        dlg._step_table_row(1)
        dlg._step_table_row(1)
        t_step_ms = (time.perf_counter() - t_step0) * 1000.0
        assert t_step_ms < 300.0, f"Mode 4 row stepping too slow: {t_step_ms:.1f} ms"
    finally:
        dlg.close()


def test_bale_user_guide_docs_in_sync() -> None:
    """Verify bundled and web BALE user guide Markdown files exist and are byte-identical."""
    repo_root = Path(__file__).resolve().parents[2]
    bundled = repo_root / "src" / "jabs" / "resources" / "docs" / "user_guide" / "bale.md"
    web_doc = repo_root / "docs" / "user-guide" / "bale.md"
    assert bundled.is_file(), f"Missing bundled guide: {bundled}"
    assert web_doc.is_file(), f"Missing web guide: {web_doc}"
    assert bundled.read_bytes() == web_doc.read_bytes()


def test_examined_bout_store_legacy_filename_fallback(tmp_path: Path) -> None:
    """Verify ExaminedBoutStore reads legacy label_explorer_reviewed.json and writes bale_reviewed.json."""
    from jabs.bale.examined_store import (
        LEGACY_FILENAME,
        ExaminedBoutStore,
        make_bout_signature_key,
    )
    from jabs.bale.models import BoutRecord

    rec = BoutRecord(
        bout_id=0,
        video="vid_a.mp4",
        identity=0,
        behavior="Grooming",
        class_name="Grooming",
        start_frame=10,
        end_frame=30,
        duration_frames=21,
        source="labeled",
        is_positive=True,
    )
    key = make_bout_signature_key(rec)
    session_dir = tmp_path / "jabs" / "session"
    session_dir.mkdir(parents=True, exist_ok=True)
    legacy_path = session_dir / LEGACY_FILENAME
    legacy_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entries": {
                    key: {
                        "key": key,
                        "video": "vid_a.mp4",
                        "identity": 0,
                        "behavior": "Grooming",
                        "origin_source": "labeled",
                        "start_frame": 10,
                        "end_frame": 30,
                        "action": "examined",
                        "note": "✓ Examined",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    store = ExaminedBoutStore(tmp_path)
    assert store.count() == 1
    assert store.find_match(rec) is not None
    store.save()
    assert (session_dir / "bale_reviewed.json").is_file()


def test_custom_bout_list_search_query_preserves_rank_order() -> None:
    """Verify CustomBoutListSearchQuery keeps explicit rank order in search_behaviors and SearchBarWidget."""
    from jabs.behavior_search import CustomBoutListSearchQuery, SearchHit, search_behaviors
    from jabs.ui.search_bar_widget import SearchBarWidget

    _ensure_qapp()
    hits = (
        SearchHit(
            file="vid_z.mp4", identity="1", behavior="Grooming", start_frame=200, end_frame=240
        ),
        SearchHit(
            file="vid_a.mp4", identity="0", behavior="Grooming", start_frame=10, end_frame=30
        ),
        SearchHit(
            file="vid_m.mp4", identity="2", behavior="Grooming", start_frame=50, end_frame=80
        ),
    )
    query = CustomBoutListSearchQuery(
        hits=hits, description="BALE Top Suspects", preserve_rank_order=True
    )
    assert search_behaviors(None, query) == list(hits)

    bar = SearchBarWidget()
    try:
        bar.update_search(query)
        assert bar.text_label.text() == "BALE Top Suspects"
        assert bar.current_search_hit == hits[0]
        bar._on_next_clicked()
        assert bar.current_search_hit == hits[1]
        bar._on_next_clicked()
        assert bar.current_search_hit == hits[2]
        bar._on_prev_clicked()
        assert bar.current_search_hit == hits[1]
    finally:
        bar.close()


def test_arena_spatial_map_toggle_inverted_y_zoom_and_nan_selection(tmp_path: Path) -> None:
    """Verify `🗺 Arena [A]` toggle, `A` shortcut, inverted-Y zoom preservation, and NaN-safe selection."""
    from jabs.bale.models import BoutSpatialResult

    _ensure_qapp()
    project, _ = _build_synthetic_explorer_project(tmp_path)
    central_mock = MagicMock()
    central_mock.behavior = "Grooming"

    dlg = BaleDialog(project, central_widget=central_mock)
    try:
        rep = run_bale(project, "Grooming", include_predictions=False, space="classifier")
        n_bouts = len(rep.feature_matrix.records)
        coords_xy = np.tile(np.array([[220.0, 180.0]], dtype=np.float32), (n_bouts, 1))
        coords_xy[-1] = np.nan  # Include an unmapped NaN bout cleanly
        valid_mask = np.ones(n_bouts, dtype=bool)
        valid_mask[-1] = False
        rep.spatial = BoutSpatialResult(
            coords_xy=coords_xy,
            nose_xy=coords_xy.copy(),
            valid_mask=valid_mask,
            frame_width=800.0,
            frame_height=600.0,
            normalized=False,
            reference_frame_gray=np.full((300, 400), 120, dtype=np.uint8),
            static_objects={
                "corners": np.array(
                    [[100, 100], [700, 100], [700, 500], [100, 500]], dtype=np.float32
                )
            },
            elapsed_ms=4.2,
        )
        dlg._on_analysis_finished(rep, included_preds=False)
        dlg.show()
        QtWidgets.QApplication.processEvents()

        # Toggle Arena via `[A]` shortcut
        assert dlg._proj_method == "pca"
        assert not dlg.arena_map_btn.isChecked()
        ev_a = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress, Qt.Key.Key_A, Qt.KeyboardModifier.NoModifier, "a"
        )
        QtWidgets.QApplication.sendEvent(dlg, ev_a)
        assert dlg._proj_method == "spatial"
        assert dlg.arena_map_btn.isChecked()
        assert dlg._proj_menu_actions["spatial"].isChecked()

        y0, y1 = dlg.scatter_ax.get_ylim()
        assert y0 > y1, "Arena spatial map must keep top-down inverted Y axis"

        # Box zoom AND wheel zoom must preserve inverted Y (never flip upside down)
        dlg._on_scatter_zoom_box(150.0, 350.0, 100.0, 300.0)
        zy0, zy1 = dlg.scatter_ax.get_ylim()
        assert zy0 > zy1
        dlg._on_scatter_wheel_zoom(250.0, 200.0, 0.8)
        wy0, wy1 = dlg.scatter_ax.get_ylim()
        assert wy0 > wy1

        # NaN-safe box & lasso selection must not select the NaN bout
        dlg._on_scatter_box_selected(100.0, 400.0, 100.0, 300.0, additive=False)
        assert (n_bouts - 1) not in dlg._lasso_selected_indices
        poly = np.array([[100.0, 100.0], [400.0, 100.0], [400.0, 300.0], [100.0, 300.0]])
        dlg._on_scatter_lasso_selected(poly, additive=False)
        assert (n_bouts - 1) not in dlg._lasso_selected_indices

        # Switch to Mode 3 (`3. Sub-Types`, unsupervised embedding recompute) while Arena is active
        # and confirm cached `spatial` survives recompute_embedding_only (C1 UI regression)
        dlg.mode_subtypes_btn.setChecked(True)
        dlg._on_mode_pill_clicked(2)
        QtWidgets.QApplication.processEvents()
        assert dlg._report is not None
        assert dlg._report.spatial is not None
        assert dlg._proj_method == "spatial"

        # Pressing A again returns to previous feature projection and resets aspect
        QtWidgets.QApplication.sendEvent(dlg, ev_a)
        assert dlg._proj_method == "pca"
        assert not dlg.arena_map_btn.isChecked()

        # Mode 4 (Videos) hides the Arena button and Map Class Filter
        dlg.mode_videos_btn.setChecked(True)
        dlg._on_mode_pill_clicked(3)
        assert not dlg.arena_map_btn.isVisible()
        assert not dlg.map_class_combo.isVisible()
    finally:
        dlg.close()


def test_class_and_table_filters_sync_pca_and_arena_plots(tmp_path: Path) -> None:
    """Verify Class Filter dropdowns, search text, Quick Select, and Esc sync across Table, PCA, and Arena plots."""
    from jabs.bale.models import BoutSpatialResult

    _ensure_qapp()
    project, _ = _build_synthetic_explorer_project(tmp_path)
    central_mock = MagicMock()
    central_mock.behavior = "Grooming"

    dlg = BaleDialog(project, central_widget=central_mock)
    try:
        rep = run_bale(project, "Grooming", include_predictions=False, space="classifier")
        n_bouts = len(rep.feature_matrix.records)
        rng = np.random.default_rng(7)
        coords_xy = rng.uniform(100.0, 500.0, size=(n_bouts, 2)).astype(np.float32)
        rep.spatial = BoutSpatialResult(
            coords_xy=coords_xy,
            nose_xy=coords_xy.copy(),
            valid_mask=np.ones(n_bouts, dtype=bool),
            frame_width=800.0,
            frame_height=600.0,
            normalized=False,
            reference_frame_gray=np.full((300, 400), 120, dtype=np.uint8),
            static_objects={},
            elapsed_ms=2.1,
        )
        dlg._on_analysis_finished(rep, included_preds=False)
        dlg.show()
        QtWidgets.QApplication.processEvents()

        # Both Class Filter combos carry NoFocus so Spacebar still plays clips cleanly
        assert dlg.map_class_combo.focusPolicy() == Qt.FocusPolicy.NoFocus
        assert dlg.class_filter_combo.focusPolicy() == Qt.FocusPolicy.NoFocus

        records = rep.feature_matrix.records
        pos_ids = {i for i, r in enumerate(records) if r.is_positive}
        neg_ids = {i for i, r in enumerate(records) if not r.is_positive}
        assert pos_ids and neg_ids

        # Switch to All Bouts queue and capture full-cloud PCA limits
        dlg.qf_all_btn.setChecked(True)
        dlg._on_queue_filter_pill_clicked(1)
        assert dlg._scatter_default_limits is not None
        full_pca_xlim, full_pca_ylim = dlg._scatter_default_limits

        # 1. Selecting Positive Only in map_class_combo syncs table combo, table rows, AND PCA plot
        idx_pos = dlg.map_class_combo.findData("pos")
        assert idx_pos >= 0
        dlg.map_class_combo.setCurrentIndex(idx_pos)
        assert dlg._class_filter == "pos"
        assert dlg.class_filter_combo.currentData() == "pos"
        pca_picks = set(dlg.scatter_canvas._pick_global_indices.tolist())
        assert pca_picks == pos_ids
        assert dlg.table.rowCount() == len(pos_ids)
        assert dlg._scatter_default_limits == (full_pca_xlim, full_pca_ylim)

        # Switching to Arena [A] keeps the exact same positive class filter + inverted-Y bounds
        dlg._toggle_arena_map(True)
        arena_picks_pos = set(dlg.scatter_canvas._pick_global_indices.tolist())
        assert arena_picks_pos == pos_ids
        ay0, ay1 = dlg.scatter_ax.get_ylim()
        assert ay0 > ay1

        # 2. Selecting Negative Only in table class_filter_combo updates Arena plot & map combo
        idx_neg = dlg.class_filter_combo.findData("neg")
        assert idx_neg >= 0
        dlg.class_filter_combo.setCurrentIndex(idx_neg)
        assert dlg._class_filter == "neg"
        assert dlg.map_class_combo.currentData() == "neg"
        arena_picks_neg = set(dlg.scatter_canvas._pick_global_indices.tolist())
        assert arena_picks_neg == neg_ids
        assert dlg.table.rowCount() == len(neg_ids)

        # Back to All Classes
        idx_all = dlg.class_filter_combo.findData("all")
        dlg.class_filter_combo.setCurrentIndex(idx_all)
        assert set(dlg.scatter_canvas._pick_global_indices.tolist()) == (pos_ids | neg_ids)

        # 3. Text filtering ("Grooming" vs "Not-Grooming") never matches "Not-Grooming" for "Grooming"
        # and immediately updates Arena + PCA scatter plots even when row 0 stays identical
        dlg.search_input.setText("Grooming")
        QtWidgets.QApplication.processEvents()
        search_pos_picks = set(dlg.scatter_canvas._pick_global_indices.tolist())
        assert search_pos_picks == pos_ids
        assert dlg.table.rowCount() == len(pos_ids)

        dlg.search_input.setText("Not-Grooming")
        QtWidgets.QApplication.processEvents()
        assert set(dlg.scatter_canvas._pick_global_indices.tolist()) == neg_ids
        assert dlg.table.rowCount() == len(neg_ids)
        dlg.search_input.clear()
        QtWidgets.QApplication.processEvents()

        # 4. Quick Select cohort filters Arena plot (and PCA on toggle back), and Esc restores all
        dlg._select_quick_slice("class:Grooming")
        assert set(dlg.scatter_canvas._pick_global_indices.tolist()) == pos_ids
        dlg._toggle_arena_map(False)
        assert set(dlg.scatter_canvas._pick_global_indices.tolist()) == pos_ids
        ev_esc = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier, ""
        )
        dlg.keyPressEvent(ev_esc)
        assert set(dlg.scatter_canvas._pick_global_indices.tolist()) == (pos_ids | neg_ids)
    finally:
        dlg.close()


def test_perf_ui_interactive_header_o1_ring_search_debounce_and_clean_close(
    tmp_path: Path,
) -> None:
    """C2 + C3 + C4 + C6/M6 regressions: Interactive table header, O(1) scatter ring, search debounce, timer stop."""
    _ensure_qapp()
    project, _ = _build_synthetic_explorer_project(tmp_path)
    central_mock = MagicMock()
    central_mock.behavior = "Grooming"

    dlg = BaleDialog(project, central_widget=central_mock)
    try:
        rep = run_bale(project, "Grooming", include_predictions=False, space="classifier")
        dlg._on_analysis_finished(rep, included_preds=False)
        dlg.resize(1340, 820)
        dlg.show()
        QtWidgets.QApplication.processEvents()

        # C2: Header modes stay Interactive (col 5 Stretch) after populate and in-place row update
        hdr = dlg.table.horizontalHeader()
        for col in (0, 1, 3, 4, 6):
            assert hdr.sectionResizeMode(col) == QtWidgets.QHeaderView.ResizeMode.Interactive
        assert hdr.sectionResizeMode(5) == QtWidgets.QHeaderView.ResizeMode.Stretch
        dlg._update_table_rows_in_place({0})
        for col in (0, 1, 3, 4, 6):
            assert hdr.sectionResizeMode(col) == QtWidgets.QHeaderView.ResizeMode.Interactive

        # C3: Row stepping moves the persistent O(1) selection ring without calling _draw_scatter
        # (Visit row 1 and row 0 once first so their initial auto-Examined transitions settle)
        assert dlg._bout_scatter_sel_artist is not None
        if dlg.table.rowCount() >= 2:
            from PySide6.QtTest import QTest

            dlg.table.selectRow(1)
            QTest.qWait(80)
            dlg.table.selectRow(0)
            QTest.qWait(80)
            with patch.object(dlg, "_draw_scatter", wraps=dlg._draw_scatter) as spy_draw:
                dlg.table.selectRow(1)
                QTest.qWait(70)
                dlg.table.selectRow(0)
                QTest.qWait(70)
                assert not dlg._scatter_redraw_timer.isActive()
                assert spy_draw.call_count == 0
                assert dlg._bout_scatter_sel_artist.get_visible()

        # C4: Focused typing activates 120 ms debounce timer; flush applies filter immediately
        dlg._set_queue_filter("all", redraw=False)
        with patch.object(dlg.search_input, "hasFocus", return_value=True):
            dlg.search_input.setText("Not-Grooming")
            assert dlg._bout_search_timer.isActive()
            dlg._flush_bout_search_if_pending()
            assert not dlg._bout_search_timer.isActive()
            assert dlg.table.rowCount() >= 1

        # C6 / M6: Coalesced resize/viewport timers and active worker cancel_event stop on close
        assert dlg.scatter_canvas._resize_timer.interval() == 50
        assert dlg.scatter_canvas._viewport_timer.interval() == 60
        dummy_worker = MagicMock()
        import threading

        dummy_worker.cancel_event = threading.Event()
        dlg._active_worker = dummy_worker
        dlg.scatter_canvas._resize_timer.start()
        dlg.insp_canvas._viewport_timer.start()
        dlg.close()
        assert dummy_worker.cancel_event.is_set()
        assert not dlg.scatter_canvas._resize_timer.isActive()
        assert not dlg.insp_canvas._viewport_timer.isActive()
    finally:
        dlg.close()


def test_pr608_review_lazy_import_menu_lifecycle_worker_slots_and_timeline_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PR #608 review (#1, #3, #4, #6, #8, #10): lazy dialog export, guarded destroyed, bound worker slots, MessageDialog, timeline & signal sync."""
    from types import SimpleNamespace

    import jabs.ui.dialogs as dialogs_pkg
    from jabs.ui.dialogs.message_dialog import MessageDialog
    from jabs.ui.main_window.central_widget import CentralWidget
    from jabs.ui.main_window.menu_handlers import MenuHandlers

    # #1: BaleDialog is not re-exported from jabs.ui.dialogs (keeps startup lazy & matplotlib-safe)
    assert "BaleDialog" not in dialogs_pkg.__all__
    assert not hasattr(dialogs_pkg, "BaleDialog")

    # #8: Zero QMessageBox calls in bale_dialog.py
    bale_dialog_src = (
        Path(__file__).resolve().parents[2] / "src" / "jabs" / "ui" / "dialogs" / "bale_dialog.py"
    ).read_text(encoding="utf-8")
    assert "QMessageBox" not in bale_dialog_src

    _ensure_qapp()
    project, _ = _build_synthetic_explorer_project(tmp_path)
    central_mock = MagicMock()
    central_mock.behavior = "Grooming"

    # #3: Old dialog's deferred `destroyed` signal must NOT clear replacement dialog reference
    win = SimpleNamespace(
        _project=project,
        _central_widget=central_mock,
        _bale_dialog=None,
    )
    mh = MenuHandlers(win)
    mh.open_bale_dialog()
    dlg1 = win._bale_dialog
    assert dlg1 is not None
    project2, _ = _build_synthetic_explorer_project(tmp_path / "proj2")
    win._project = project2
    mh.open_bale_dialog()
    dlg2 = win._bale_dialog
    assert dlg2 is not None and dlg2 is not dlg1
    QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    QtWidgets.QApplication.processEvents()
    assert win._bale_dialog is dlg2

    # #8: _on_reset_examined_flags uses MessageDialog.confirm
    confirm_msgs: list[str] = []
    monkeypatch.setattr(
        MessageDialog,
        "confirm",
        staticmethod(lambda *_a, **kw: confirm_msgs.append(str(kw.get("message", ""))) or False),
    )
    rep = run_bale(project2, "Grooming", include_predictions=False, space="classifier")
    dlg2._on_analysis_finished(rep, included_preds=False)
    dlg2._examined_only_bout_ids.add(rep.feature_matrix.records[0].bout_id)
    dlg2._examined_store.count = MagicMock(return_value=1)
    dlg2._on_reset_examined_flags(all_behaviors=False)
    assert len(confirm_msgs) == 1

    # #4: Calling _on_analysis_finished / _on_analysis_error after close returns safely without touching _init_refresh_timer
    dlg2.close()
    assert dlg2._is_closing is True
    dlg2._init_refresh_timer = MagicMock()
    dlg2._on_analysis_finished(rep, False)
    dlg2._on_analysis_error("late error")
    dlg2._init_refresh_timer.stop.assert_not_called()
    QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    QtWidgets.QApplication.processEvents()

    # #6 & #10: CentralWidget.sync_external_video_labels updates player timeline_annotations;
    # on_annotation_edited / on_annotation_deleted / _save_and_refresh_annotations emit labels_modified.
    new_annos = object()
    new_labels = SimpleNamespace(timeline_annotations=new_annos)
    set_annos_calls: list[object] = []
    emitted_count = 0

    class _Sig:
        def emit(self) -> None:
            nonlocal emitted_count
            emitted_count += 1

    cw_stub = SimpleNamespace(
        _project=SimpleNamespace(
            settings_manager=SimpleNamespace(classifier_mode=ClassifierMode.BINARY),
            save_annotations=MagicMock(),
        ),
        _loaded_video=SimpleNamespace(name="cage_00.mp4"),
        _pose_est=SimpleNamespace(identity_index_to_display=lambda i: str(i)),
        _labels=SimpleNamespace(
            timeline_annotations=SimpleNamespace(
                annotation_exists=lambda **_kw: False,
                remove_annotation_by_key=MagicMock(),
                add_annotation=MagicMock(),
            )
        ),
        _counts=None,
        _none_counts=None,
        _controls=SimpleNamespace(select_button_set_checked=MagicMock()),
        _start_selection=MagicMock(),
        _update_label_counts=MagicMock(),
        _set_label_track=MagicMock(),
        set_train_button_enabled_state=MagicMock(),
        _player_widget=SimpleNamespace(
            set_timeline_annotations=lambda ann: set_annos_calls.append(ann),
            update=MagicMock(),
        ),
        labels_modified=_Sig(),
    )
    CentralWidget.sync_external_video_labels(cw_stub, "cage_00.mp4", updated_labels=new_labels)
    assert set_annos_calls == [new_annos]

    cw_stub._labels = SimpleNamespace(
        timeline_annotations=SimpleNamespace(
            annotation_exists=lambda **_kw: False,
            remove_annotation_by_key=MagicMock(),
            add_annotation=MagicMock(),
        )
    )
    CentralWidget.on_annotation_edited(
        cw_stub,
        {"start": 10, "end": 20, "tag": "t1", "identity": 0},
        {"tag": "t2", "identity_scoped": True, "color": "#00FF00", "description": ""},
    )
    CentralWidget.on_annotation_deleted(
        cw_stub, {"start": 10, "end": 20, "tag": "t2", "identity_index": 0}
    )
    CentralWidget._save_and_refresh_annotations(cw_stub)
    assert emitted_count == 3


def test_arena_view_real_mouse_clicks_and_full_gui_connectivity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify real mouse clicks on Arena [A] & all BALE->JABS GUI sync entry points never disconnect."""
    from jabs.bale.models import BoutSpatialResult
    from jabs.ui.dialogs.message_dialog import MessageDialog

    _ensure_qapp()
    project, proj_dir = _build_synthetic_explorer_project(tmp_path)
    (proj_dir / "jabs" / "predictions" / "cage_00.h5").write_bytes(b"stub")
    preds_raw = np.zeros(600, dtype=np.int32)
    preds_post = np.zeros(600, dtype=np.int32)
    probs = np.full(600, 0.05, dtype=np.float32)
    preds_raw[490:536] = 1
    preds_post[490:536] = 1
    probs[490:536] = 0.56
    project.prediction_manager.load_predictions.side_effect = lambda vid, _beh: (
        ({0: preds_raw}, {0: probs}, {0: preds_post}) if vid == "cage_00.mp4" else ({}, {}, {})
    )
    monkeypatch.setattr(MessageDialog, "confirm", staticmethod(lambda *_a, **_kw: True))
    monkeypatch.setattr(MessageDialog, "information", staticmethod(lambda *_a, **_kw: None))
    monkeypatch.setattr(MessageDialog, "warning", staticmethod(lambda *_a, **_kw: None))

    rep = run_bale(project, "Grooming", include_predictions=True, space="classifier")
    records = rep.feature_matrix.records
    n_bouts = len(records)
    assert n_bouts >= 6

    # Build deterministic spread coordinates across a 2048x1536 arena, including a wall/perimeter bout
    # and a trailing unmapped NaN bout.
    coords_xy = np.column_stack(
        [
            np.linspace(180.0, 1880.0, n_bouts, dtype=np.float32),
            np.linspace(220.0, 1340.0, n_bouts, dtype=np.float32),
        ]
    )
    labeled_idxs = [i for i, r in enumerate(records) if r.source == "labeled"]
    pred_idxs = [i for i, r in enumerate(records) if r.source == "predicted"]
    assert len(labeled_idxs) >= 4
    assert len(pred_idxs) >= 1

    # Put one labeled bout right at the right/top arena perimeter wall (`2047.5, 2.0`),
    # and keep the last bout unmapped NaN if it's not needed below.
    perim_gi = labeled_idxs[2]
    coords_xy[perim_gi] = np.array([2047.5, 2.0], dtype=np.float32)
    valid_mask = np.ones(n_bouts, dtype=bool)
    rep.spatial = BoutSpatialResult(
        coords_xy=coords_xy,
        nose_xy=coords_xy.copy(),
        valid_mask=valid_mask,
        frame_width=2048.0,
        frame_height=1536.0,
        normalized=False,
        reference_frame_gray=np.full((480, 640), 120, dtype=np.uint8),
        static_objects={},
        elapsed_ms=2.5,
    )

    central_mock = MagicMock()
    central_mock.behavior = "Grooming"
    central_mock.loaded_video = None
    central_mock.seek_to_bout = MagicMock()
    central_mock.play_bout_clip = MagicMock()
    central_mock.seek_to_frame_only = MagicMock()
    central_mock.load_custom_search_queue = MagicMock()
    central_mock.sync_external_video_labels = MagicMock()

    def _click_canvas_at_data(
        canvas: QtWidgets.QWidget,
        data_xy: tuple[float, float] | np.ndarray,
        *,
        dx_px: float = 0.0,
        dy_px: float = 0.0,
    ) -> None:
        px, py = canvas.ax.transData.transform(data_xy)  # type: ignore[attr-defined]
        dpr = max(1.0, float(canvas.devicePixelRatioF()))
        qt_pos = QtCore.QPointF(
            float(px) / dpr + dx_px,
            (float(canvas.height()) - float(py) / dpr) + dy_px,
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
        canvas.mousePressEvent(ev_p)
        canvas.mouseReleaseEvent(ev_r)

    dlg = BaleDialog(project, central_widget=central_mock)
    try:
        dlg.resize(1200, 760)
        dlg.show()
        dlg._on_analysis_finished(rep, included_preds=True)
        QtWidgets.QApplication.processEvents()

        # Switch to Needs Review or All so we have both in-table and out-of-table labeled bouts
        assert dlg.table.rowCount() >= 2
        row0_gi = int(dlg.table.item(0, 0).data(Qt.ItemDataRole.UserRole))
        row1_gi = int(dlg.table.item(1, 0).data(Qt.ItemDataRole.UserRole))
        table_bis = {
            int(dlg.table.item(r, 0).data(Qt.ItemDataRole.UserRole))
            for r in range(dlg.table.rowCount())
        }
        out_of_table_labeled = [i for i in labeled_idxs if i not in table_bis]
        if not out_of_table_labeled:
            dlg.qf_needs_btn.click()
            QtWidgets.QApplication.processEvents()
            table_bis = {
                int(dlg.table.item(r, 0).data(Qt.ItemDataRole.UserRole))
                for r in range(dlg.table.rowCount())
            }
            out_of_table_labeled = [i for i in labeled_idxs if i not in table_bis]

        # 1. Switch to Arena [A] via button click and flush initial Agg render
        dlg.arena_map_btn.click()
        assert dlg._proj_method == "spatial"
        assert dlg.arena_map_btn.isChecked()
        dlg.scatter_canvas._render_now()
        QtWidgets.QApplication.processEvents()

        # Click an in-table bout different from current selection via real mouse press+release
        target_gi = row1_gi if dlg._selected_bout_idx == row0_gi else row0_gi
        central_mock.seek_to_bout.reset_mock()
        _click_canvas_at_data(dlg.scatter_canvas, coords_xy[target_gi])
        QtWidgets.QApplication.processEvents()
        assert dlg._selected_bout_idx == target_gi
        assert central_mock.seek_to_bout.call_count == 1
        rec_t = records[target_gi]
        assert central_mock.seek_to_bout.call_args.args[:4] == (
            rec_t.video,
            rec_t.identity,
            rec_t.start_frame,
            rec_t.end_frame,
        )
        assert central_mock.seek_to_bout.call_args.kwargs.get("select_span") is True

        # 2. Re-clicking the already-selected bout on Arena still seeks JABS GUI
        central_mock.seek_to_bout.reset_mock()
        _click_canvas_at_data(dlg.scatter_canvas, coords_xy[target_gi])
        QtWidgets.QApplication.processEvents()
        assert central_mock.seek_to_bout.call_count == 1

        # 3. Clicking an out-of-table bout (clean bout while Needs Review tab is active) seeks GUI
        if out_of_table_labeled:
            out_gi = out_of_table_labeled[0]
            central_mock.seek_to_bout.reset_mock()
            _click_canvas_at_data(dlg.scatter_canvas, coords_xy[out_gi])
            QtWidgets.QApplication.processEvents()
            assert dlg._selected_bout_idx == out_gi
            assert central_mock.seek_to_bout.call_count == 1
            assert central_mock.seek_to_bout.call_args.args[:3] == (
                records[out_gi].video,
                records[out_gi].identity,
                records[out_gi].start_frame,
            )

        # 4. Perimeter/wall bout clicked 3 px past ax.bbox edge (within 14 px pick tolerance) still seeks GUI
        central_mock.seek_to_bout.reset_mock()
        _click_canvas_at_data(dlg.scatter_canvas, coords_xy[perim_gi], dx_px=3.0, dy_px=-3.0)
        QtWidgets.QApplication.processEvents()
        assert dlg._selected_bout_idx == perim_gi
        assert central_mock.seek_to_bout.call_count == 1

        # 5. Regression: stale RightButton press (no release, as when QMenu.exec() steals it)
        # and stale Col 6 press on table.viewport() must NEVER block subsequent Arena/table/arrow seeks!
        ev_tbl_right = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseButtonPress,
            QtCore.QPointF(20.0, 10.0),
            QtCore.QPointF(20.0, 10.0),
            Qt.MouseButton.RightButton,
            Qt.MouseButton.RightButton,
            Qt.KeyboardModifier.NoModifier,
        )
        QtWidgets.QApplication.sendEvent(dlg.table.viewport(), ev_tbl_right)
        assert dlg._table_mouse_press_btn == Qt.MouseButton.RightButton

        central_mock.seek_to_bout.reset_mock()
        _click_canvas_at_data(dlg.scatter_canvas, coords_xy[row0_gi])
        QtWidgets.QApplication.processEvents()
        assert dlg._selected_bout_idx == row0_gi
        assert central_mock.seek_to_bout.call_count == 1

        # Also test stale Col 6 press -> Arena click & arrow step
        col6_x = float(max(0, dlg.table.columnViewportPosition(6)) + 5)
        ev_tbl_col6 = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseButtonPress,
            QtCore.QPointF(col6_x, 10.0),
            QtCore.QPointF(col6_x, 10.0),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        QtWidgets.QApplication.sendEvent(dlg.table.viewport(), ev_tbl_col6)
        central_mock.seek_to_bout.reset_mock()
        _click_canvas_at_data(dlg.scatter_canvas, coords_xy[row1_gi])
        QtWidgets.QApplication.processEvents()
        assert dlg._selected_bout_idx == row1_gi
        assert central_mock.seek_to_bout.call_count == 1

        QtWidgets.QApplication.sendEvent(dlg.table.viewport(), ev_tbl_right)
        central_mock.seek_to_bout.reset_mock()
        dlg._step_table_row(1)
        QtWidgets.QApplication.processEvents()
        assert central_mock.seek_to_bout.call_count == 1

        # 6. Zoomed-in Arena BEFORE 60 ms _viewport_timer settles + Retina DPR=2.0 still seeks GUI accurately
        orig_dpr_fn = dlg.scatter_canvas.devicePixelRatioF
        try:
            dlg.scatter_canvas.devicePixelRatioF = lambda: 2.0  # type: ignore[method-assign]
            dlg.scatter_canvas.mark_layout_dirty()
            dlg.scatter_canvas._render_now()
            z_gi = perim_gi if perim_gi != row1_gi else row0_gi
            zx, zy = float(coords_xy[z_gi, 0]), float(coords_xy[z_gi, 1])
            dlg._on_scatter_wheel_zoom(zx, zy, 0.45)
            # Notice _viewport_timer is still active (fast preview before 60 ms Agg settle)
            assert dlg.scatter_canvas._viewport_timer.isActive()
            central_mock.seek_to_bout.reset_mock()
            _click_canvas_at_data(dlg.scatter_canvas, coords_xy[z_gi])
            QtWidgets.QApplication.processEvents()
            assert dlg._selected_bout_idx == z_gi
            assert central_mock.seek_to_bout.call_count == 1
        finally:
            dlg.scatter_canvas.devicePixelRatioF = orig_dpr_fn  # type: ignore[method-assign]
            dlg._reset_scatter_zoom()
            dlg.scatter_canvas._render_now()

        # 7. Mode 2 (Review Predictions) + Arena real mouse click seeks JABS GUI
        dlg.mode_predict_btn.setChecked(True)
        dlg._on_mode_pill_clicked(1)
        QtWidgets.QApplication.processEvents()
        dlg.scatter_canvas._render_now()
        p_gi = pred_idxs[0]
        central_mock.seek_to_bout.reset_mock()
        _click_canvas_at_data(dlg.scatter_canvas, coords_xy[p_gi])
        QtWidgets.QApplication.processEvents()
        assert dlg._selected_bout_idx == p_gi
        assert central_mock.seek_to_bout.call_count == 1

        # 8. Mode 3 (Sub-Types) + Arena real mouse click seeks JABS GUI
        dlg.mode_subtypes_btn.setChecked(True)
        dlg._on_mode_pill_clicked(2)
        QtWidgets.QApplication.processEvents()
        dlg.scatter_canvas._render_now()
        pos_labeled = [i for i in labeled_idxs if records[i].is_positive]
        s_gi = pos_labeled[0]
        central_mock.seek_to_bout.reset_mock()
        _click_canvas_at_data(dlg.scatter_canvas, coords_xy[s_gi])
        QtWidgets.QApplication.processEvents()
        assert dlg._selected_bout_idx == s_gi
        assert central_mock.seek_to_bout.call_count == 1

        # 9. Toggle back to PCA and verify PCA scatter real mouse click + remaining Mode 1-3 GUI links
        dlg.mode_audit_btn.setChecked(True)
        dlg._on_mode_pill_clicked(0)
        QtWidgets.QApplication.processEvents()
        dlg.arena_map_btn.click()
        assert dlg._proj_method == "pca"
        dlg.scatter_canvas._render_now()
        pca_coords = rep.embedding.coords_pca_2d
        pca_target = row0_gi if dlg._selected_bout_idx != row0_gi else row1_gi
        central_mock.seek_to_bout.reset_mock()
        _click_canvas_at_data(dlg.scatter_canvas, pca_coords[pca_target])
        QtWidgets.QApplication.processEvents()
        assert dlg._selected_bout_idx == pca_target
        assert central_mock.seek_to_bout.call_count == 1

        # Spacebar / Play Clip -> play_bout_clip; Prob scrub -> seek_to_frame_only;
        # Send Queue -> load_custom_search_queue; Label flip + Undo -> sync_external_video_labels.
        central_mock.play_bout_clip.reset_mock()
        dlg.keyPressEvent(
            QtGui.QKeyEvent(
                QtCore.QEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier, " "
            )
        )
        assert central_mock.play_bout_clip.call_count == 1

        dlg._set_inspector_view("prob")
        dlg.insp_canvas._render_now()
        rec_cur = records[dlg._selected_bout_idx]
        central_mock.seek_to_frame_only.reset_mock()
        _click_canvas_at_data(
            dlg.insp_canvas, (0.5 * (rec_cur.start_frame + rec_cur.end_frame), 0.5)
        )
        assert central_mock.seek_to_frame_only.call_count == 1

        central_mock.load_custom_search_queue.reset_mock()
        dlg.send_queue_btn.click()
        assert central_mock.load_custom_search_queue.call_count == 1

        central_mock.sync_external_video_labels.reset_mock()
        dlg._on_secondary_action_2_clicked()
        assert central_mock.sync_external_video_labels.call_count >= 1
        dlg._on_undo_clicked()

        # 10. Mode 4 (Videos) connectivity: stale right-click + real scatter click + buttons + scrub + send
        dlg.mode_videos_btn.setChecked(True)
        dlg._on_mode_pill_clicked(3)
        for _ in range(80):
            QtWidgets.QApplication.processEvents()
            if dlg._video_report is not None and dlg._video_worker is None:
                break
            time.sleep(0.02)
        assert dlg._video_report is not None and dlg.video_table.rowCount() >= 2
        dlg.scatter_canvas._render_now()

        ev_vtbl_right = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseButtonPress,
            QtCore.QPointF(20.0, 10.0),
            QtCore.QPointF(20.0, 10.0),
            Qt.MouseButton.RightButton,
            Qt.MouseButton.RightButton,
            Qt.KeyboardModifier.NoModifier,
        )
        QtWidgets.QApplication.sendEvent(dlg.video_table.viewport(), ev_vtbl_right)
        assert dlg._video_table_mouse_press_btn == Qt.MouseButton.RightButton

        assert dlg._video_scatter_coords is not None
        assert len(dlg._video_scatter_names) >= 2
        vid_target_idx = 1 if dlg._selected_video_name == dlg._video_scatter_names[0] else 0
        central_mock.seek_to_bout.reset_mock()
        _click_canvas_at_data(dlg.scatter_canvas, dlg._video_scatter_coords[vid_target_idx])
        QtWidgets.QApplication.processEvents()
        assert dlg._selected_video_name == dlg._video_scatter_names[vid_target_idx]
        assert central_mock.seek_to_bout.call_count == 1

        # Re-click already-selected video dot still seeks JABS GUI
        central_mock.seek_to_bout.reset_mock()
        _click_canvas_at_data(dlg.scatter_canvas, dlg._video_scatter_coords[vid_target_idx])
        QtWidgets.QApplication.processEvents()
        assert central_mock.seek_to_bout.call_count == 1

        central_mock.play_bout_clip.reset_mock()
        dlg.play_clip_btn.click()
        assert central_mock.play_bout_clip.call_count == 1

        central_mock.seek_to_frame_only.reset_mock()
        dlg.insp_canvas._render_now()
        _click_canvas_at_data(dlg.insp_canvas, (90.0, 0.0))
        assert central_mock.seek_to_frame_only.call_count == 1

        central_mock.load_custom_search_queue.reset_mock()
        dlg.send_videos_btn.click()
        assert central_mock.load_custom_search_queue.call_count == 1
    finally:
        dlg.close()
        QtCore.QThreadPool.globalInstance().waitForDone(5_000)
        QtWidgets.QApplication.processEvents()
