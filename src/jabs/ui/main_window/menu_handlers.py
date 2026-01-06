# noinspection PyProtectedMember
"""Menu action handlers for MainWindow.

This module contains all the callback methods for menu actions, extracted from
MainWindow to improve code organization and maintainability.
"""

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6 import QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction

from jabs.project import export_training_data
from jabs.utils import FINAL_TRAIN_SEED

from ..about_dialog import AboutDialog
from ..archive_behavior_dialog import ArchiveBehaviorDialog
from ..behavior_search_dialog import BehaviorSearchDialog
from ..license_dialog import LicenseAgreementDialog
from ..player_widget import PlayerWidget
from ..project_pruning_dialog import ProjectPruningDialog
from ..stacked_timeline_widget import StackedTimelineWidget
from ..user_guide_dialog import UserGuideDialog
from ..util import send_file_to_recycle_bin
from .constants import SESSION_TRACKING_ENABLED_KEY, USE_NATIVE_FILE_DIALOG

if TYPE_CHECKING:
    from .main_window import MainWindow


class MenuHandlers:
    """Handles menu action callbacks for MainWindow.

    This class contains all the menu callback methods, keeping them organized
    separately from the main window initialization and setup code.

    Note: This class intentionally accesses protected members of MainWindow
    as it serves as a helper/companion class for menu handling.
    """

    def __init__(self, main_window: "MainWindow"):
        """Initialize menu handlers.

        Args:
            main_window: The MainWindow instance that owns these handlers
        """
        self.window = main_window

    # ========== File Menu Handlers ==========

    def show_project_open_dialog(self) -> None:
        """Show file dialog to select and open a project directory."""
        if USE_NATIVE_FILE_DIALOG:
            project_path = QtWidgets.QFileDialog.getExistingDirectory(
                self.window, "Select Project Directory"
            )
        else:
            project_path = QtWidgets.QFileDialog.getExistingDirectory(
                self.window,
                "Select Project Directory",
                options=QtWidgets.QFileDialog.Option.DontUseNativeDialog,
            )

        if project_path:
            self.window.open_project(project_path)

    def export_training_data(self) -> None:
        """Export training data for the current classifier."""
        if not self.window._central_widget.classify_button_enabled:
            # Classify button disabled, don't allow exporting training data
            QtWidgets.QMessageBox.warning(
                self.window,
                "Unable to export training data",
                "Classifier has not been trained, or classifier parameters "
                "have changed.\n\n"
                "You must train the classifier before export.",
            )
            return

        try:
            out_path = export_training_data(
                self.window._project,
                self.window._central_widget.behavior,
                self.window._project.feature_manager.min_pose_version,
                self.window._central_widget.classifier_type,
                FINAL_TRAIN_SEED,
            )
            self.window.display_status_message(f"Training data exported: {out_path}", 5000)
        except OSError as e:
            print(f"Unable to export training data: {e}", file=sys.stderr)

    def open_archive_behavior_dialog(self) -> None:
        """Open dialog to archive a behavior and its labels."""
        archive_dialog = ArchiveBehaviorDialog(self.window._central_widget.behaviors, self.window)
        archive_dialog.behavior_archived.connect(self.archive_behavior_callback)
        archive_dialog.exec()

    def archive_behavior_callback(self, behavior: str) -> None:
        """Handle behavior archive completion.

        Args:
            behavior: Name of the archived behavior
        """
        self.window._project.archive_behavior(behavior)
        self.window._central_widget.remove_behavior(behavior)

    def show_project_pruning_dialog(self) -> None:
        """Open dialog to prune videos without labels from the project."""
        prune_dialog = ProjectPruningDialog(self.window._project, parent=self.window)
        if prune_dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            # Get the videos to delete from the dialog
            videos_to_prune = prune_dialog.videos_to_prune

            # Don't let the user remove all videos from the project
            if len(videos_to_prune) == len(self.window._project.video_manager.videos):
                QtWidgets.QMessageBox.critical(
                    self.window,
                    "All Videos Selected",
                    "ERROR: This action would remove all videos from the project.",
                )
                return

            # Create a Set of all files to delete
            files_to_delete = {video.video_path for video in videos_to_prune}
            files_to_delete.update(video.pose_path for video in videos_to_prune)
            files_to_delete.update(video.annotation_path for video in videos_to_prune)

            if files_to_delete:
                self.move_files_to_recycle_bin_with_delete_fallback(files_to_delete)

                # Remove videos from the project video manager
                for video in videos_to_prune:
                    self.window._project.video_manager.remove_video(video.video_path.name)

                # Force the video list to update its contents
                self.window.video_list.set_project(self.window._project)

                self.window.display_status_message(
                    f"Removed {len(files_to_delete)} unlabeled video(s) and pose file(s)",
                    duration=5000,
                )

    def clear_cache(self) -> None:
        """Clear the project's feature cache after user confirmation."""
        app = QtWidgets.QApplication.instance()
        dont_use_native_dialogs = app.testAttribute(
            Qt.ApplicationAttribute.AA_DontUseNativeDialogs
        )

        # if app is currently set to use native dialogs, we will temporarily set it to use Qt dialogs
        # the native style, at least on macOS, is not ideal so we'll force the Qt dialog instead
        if not dont_use_native_dialogs:
            app.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeDialogs, True)

        result = QtWidgets.QMessageBox.warning(
            self.window,
            "Clear Cache",
            "Are you sure you want to clear the project cache?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )

        # restore the original setting
        if not dont_use_native_dialogs:
            app.setAttribute(
                Qt.ApplicationAttribute.AA_DontUseNativeDialogs, dont_use_native_dialogs
            )

        if result == QtWidgets.QMessageBox.StandardButton.Yes:
            self.window._project.clear_cache()
            # need to reload the current video to force the pose file to reload
            if self.window._central_widget.loaded_video:
                self.window._central_widget.load_video(self.window._central_widget.loaded_video)
            self.window.display_status_message("Cache cleared", duration=3000)

    # ========== App Menu Handlers ==========

    def show_about_dialog(self) -> None:
        """Show the About dialog."""
        about_dialog = AboutDialog(
            f"{self.window._app_name_long} ({self.window._app_name})",
            self.window,
        )
        about_dialog.exec()

    def open_user_guide(self) -> None:
        """Show the user guide document in a separate window."""
        if self.window._user_guide_window is None:
            self.window._user_guide_window = UserGuideDialog(
                f"{self.window._app_name_long} ({self.window._app_name})", parent=None
            )
        self.window._user_guide_window.show()

    def show_license_dialog(self) -> None:
        """View the license agreement (JABS→View License Agreement menu action)."""
        dialog = LicenseAgreementDialog(self.window, view_only=True)
        dialog.exec_()

    def on_session_tracking_triggered(self, checked: bool) -> None:
        """Handle session tracking toggle.

        Args:
            checked: Whether session tracking is enabled
        """
        self.window._settings.setValue(SESSION_TRACKING_ENABLED_KEY, checked)
        if self.window._project:
            if checked:
                # if a project is already loaded and user is enabling session tracking,
                # they need to open the project again to enable session tracking.
                QtWidgets.QMessageBox.warning(
                    self.window,
                    "Session Tracking Enabled",
                    "Session Tracking Enabled: Please reopen the project to start tracking.",
                )
            else:
                # if session tracking was just disabled, we stop logging new events
                self.window._project.session_tracker.enabled = False

    # ========== View Menu Handlers ==========

    def set_video_list_visibility(self, checked: bool) -> None:
        """Toggle video playlist visibility.

        Args:
            checked: Whether playlist should be visible
        """
        # Note: if user closes playlist with the X, we get called with checked=False
        # So we'll hide the playlist, which is redundant but harmless
        self.window.video_list.setVisible(checked)

    def set_animal_track_visibility(self, checked: bool) -> None:
        """Toggle track overlay visibility.

        Args:
            checked: Whether track overlay should be visible
        """
        self.window._central_widget.show_track(checked)

    def set_pose_overlay_visibility(self, checked: bool) -> None:
        """Toggle pose overlay visibility.

        Args:
            checked: Whether pose overlay should be visible
        """
        self.window._central_widget.overlay_pose(checked)

    def set_landmark_overlay_visibility(self, checked: bool) -> None:
        """Toggle landmark overlay visibility.

        Args:
            checked: Whether landmark overlay should be visible
        """
        self.window._central_widget.overlay_landmarks(checked)

    def set_segmentation_overlay_visibility(self, checked: bool) -> None:
        """Toggle segmentation overlay visibility.

        Args:
            checked: Whether segmentation overlay should be visible
        """
        self.window._central_widget.overlay_segmentation(checked)

    def show_behavior_search_dialog(self) -> None:
        """Open the behavior search dialog."""
        search_dialog = BehaviorSearchDialog(
            self.window._project,
            self.window,
        )
        if search_dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            search_query = search_dialog.behavior_search_query
            self.window._central_widget.update_behavior_search_query(search_query)

    def on_timeline_view_mode_changed(self) -> None:
        """Handle timeline view mode change (Labels, Predictions, or Both)."""
        if self.window._timeline_labels_preds.isChecked():
            mode = StackedTimelineWidget.ViewMode.LABELS_AND_PREDICTIONS
        elif self.window._timeline_labels.isChecked():
            mode = StackedTimelineWidget.ViewMode.LABELS
        else:
            mode = StackedTimelineWidget.ViewMode.PREDICTIONS

        self.window._central_widget.timeline_view_mode = mode

    def on_timeline_identity_mode_changed(self) -> None:
        """Handle timeline identity mode change (All Animals or Selected Animal)."""
        if self.window._timeline_all_animals.isChecked():
            mode = StackedTimelineWidget.IdentityMode.ALL
        else:
            mode = StackedTimelineWidget.IdentityMode.ACTIVE

        self.window._central_widget.timeline_identity_mode = mode

    def on_label_overlay_mode_changed(self) -> None:
        """Handle label overlay mode change (None, Labels, or Predictions)."""
        if self.window._label_overlay_none.isChecked():
            mode = PlayerWidget.LabelOverlayMode.NONE
        elif self.window._label_overlay_labels.isChecked():
            mode = PlayerWidget.LabelOverlayMode.LABEL
        else:
            mode = PlayerWidget.LabelOverlayMode.PREDICTION

        self.window._central_widget.label_overlay_mode = mode

    # ========== Features Menu Handlers ==========

    def set_use_cm_units(self, checked: bool) -> None:
        """Toggle between CM and pixel units.

        Args:
            checked: Whether to use CM units
        """
        behavior = self.window._central_widget.behavior
        self.window._project.settings_manager.save_behavior(behavior, {"cm_units": checked})

    def set_social_features_enabled(self, checked: bool) -> None:
        """Toggle social features.

        Args:
            checked: Whether social features are enabled
        """
        behavior = self.window._central_widget.behavior
        self.window._project.settings_manager.save_behavior(behavior, {"social": checked})

    def set_window_features_enabled(self, checked: bool) -> None:
        """Toggle window features.

        Args:
            checked: Whether window features are enabled
        """
        behavior = self.window._central_widget.behavior
        self.window._project.settings_manager.save_behavior(behavior, {"window": checked})

    def set_fft_features_enabled(self, checked: bool) -> None:
        """Toggle FFT/signal features.

        Args:
            checked: Whether FFT features are enabled
        """
        behavior = self.window._central_widget.behavior
        self.window._project.settings_manager.save_behavior(behavior, {"fft": checked})

    def set_segmentation_features_enabled(self, checked: bool) -> None:
        """Toggle segmentation features.

        Args:
            checked: Whether segmentation features are enabled
        """
        behavior = self.window._central_widget.behavior
        self.window._project.settings_manager.save_behavior(behavior, {"segmentation": checked})

    def set_static_object_features_enabled(self, checked: bool, landmark_name: str) -> None:
        """Toggle static object (landmark) features.

        Args:
            checked: Whether the feature is enabled
            landmark_name: The name of the landmark feature (e.g., 'corner', 'wall')
        """
        behavior = self.window._central_widget.behavior
        # Get all current static object settings, update just this one, then save all back
        # This is necessary because save_behavior replaces the entire "static_objects" dict
        all_object_settings = self.window._project.settings_manager.get_behavior(behavior).get(
            "static_objects", {}
        )
        all_object_settings[landmark_name] = checked
        self.window._project.settings_manager.save_behavior(
            behavior, {"static_objects": all_object_settings}
        )

    # ========== Window Menu Handlers ==========

    def toggle_zoom(self) -> None:
        """Toggle between normal and maximized window state."""
        if self.window.isMaximized():
            self.window.showNormal()
        else:
            self.window.showMaximized()

    def bring_all_windows_to_front(self) -> None:
        """Bring all JABS windows to the front."""
        for widget in QtWidgets.QApplication.topLevelWidgets():
            if widget.isVisible() and not widget.isMinimized():
                widget.raise_()
                widget.activateWindow()

    def update_window_menu(self) -> None:
        """Update the Window menu with the current list of open windows."""
        # Remove all dynamic window items (everything after the separator)
        actions = self.window._window_menu.actions()
        separator_found = False
        items_to_remove = []

        for action in actions:
            if separator_found:
                items_to_remove.append(action)
            elif action.isSeparator():
                separator_found = True

        for action in items_to_remove:
            self.window._window_menu.removeAction(action)

        # Add Main Window
        main_window_action = QAction("Main Window", self.window)
        main_window_action.setCheckable(True)
        main_window_action.setChecked(self.window.isActiveWindow())
        main_window_action.triggered.connect(lambda: self.activate_window(self.window))
        self.window._window_menu.addAction(main_window_action)

        # Add User Guide window if open
        if (
            self.window._user_guide_window is not None
            and self.window._user_guide_window.isVisible()
        ):
            guide_action = QAction(self.window._user_guide_window.windowTitle(), self.window)
            guide_action.setCheckable(True)
            guide_action.setChecked(self.window._user_guide_window.isActiveWindow())
            guide_action.triggered.connect(
                lambda: self.activate_window(self.window._user_guide_window)
            )
            self.window._window_menu.addAction(guide_action)

        # Add any open dialogs from the central widget
        for title, dialog in self.window._central_widget.get_open_dialogs():
            dialog_action = QAction(title, self.window)
            dialog_action.setCheckable(True)
            dialog_action.setChecked(dialog.isActiveWindow())
            dialog_action.triggered.connect(lambda checked, w=dialog: self.activate_window(w))
            self.window._window_menu.addAction(dialog_action)

    def activate_window(self, window: QtWidgets.QWidget) -> None:
        """Activate and bring a window to the front.

        Args:
            window: The window to activate
        """
        if window.isMinimized():
            window.showNormal()
        window.raise_()
        window.activateWindow()

    # ========== Other Handlers ==========

    def handle_select_all(self) -> None:
        """Handle Ctrl+A / Cmd+A keyboard shortcut."""
        self.window._central_widget.select_all()

    def on_bbox_overlay_support_changed(self, supported: bool) -> None:
        """Enable/disable the bounding box overlay menu item based on whether the current pose supports it.

        Args:
            supported: Whether bounding box overlay is supported
        """
        self.window._identity_overlay_bbox.setEnabled(supported)
        if (
            not supported
            and self.window._central_widget.id_overlay_mode
            == PlayerWidget.IdentityOverlayMode.BBOX
        ):
            # If the user had bbox overlay selected but the new video doesn't support it, switch to floating
            self.window._central_widget.id_overlay_mode = PlayerWidget.IdentityOverlayMode.FLOATING
            self.window._identity_overlay_floating.setChecked(True)

    # ========== Helper Methods ==========

    def move_files_to_recycle_bin_with_delete_fallback(self, files: set[Path]) -> None:
        """Move files to recycle bin, with fallback to permanent deletion if recycling fails.

        Args:
            files: Set of file paths to delete
        """
        for file in files:
            try:
                send_file_to_recycle_bin(file)
            except Exception as e:
                # If we can't send to recycle bin, ask user if they want to permanently delete
                reply = QtWidgets.QMessageBox.question(
                    self.window,
                    "Unable to Move to Recycle Bin",
                    f"Unable to move {file.name} to recycle bin.\n\n"
                    f"Error: {e}\n\n"
                    "Would you like to permanently delete this file instead?",
                    QtWidgets.QMessageBox.StandardButton.Yes
                    | QtWidgets.QMessageBox.StandardButton.No,
                )
                if reply == QtWidgets.QMessageBox.StandardButton.Yes:
                    file.unlink()
