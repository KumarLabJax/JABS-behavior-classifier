# noinspection PyProtectedMember
"""Menu builder for MainWindow.

This module handles the creation and configuration of all menus in the main window.

Note: This module intentionally accesses protected members of MainWindow
as it serves as a helper/companion class for menu handling.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui, QtWidgets

from jabs.feature_extraction.landmark_features import LandmarkFeatureGroup

from ..player_widget import PlayerWidget
from .constants import SESSION_TRACKING_ENABLED_KEY

if TYPE_CHECKING:
    from .main_window import MainWindow


@dataclass
class MenuReferences:
    """Container for menu and action references needed by MainWindow."""

    # Menus
    app_menu: QtWidgets.QMenu
    file_menu: QtWidgets.QMenu
    view_menu: QtWidgets.QMenu
    feature_menu: QtWidgets.QMenu
    window_menu: QtWidgets.QMenu
    open_recent_menu: QtWidgets.QMenu

    # File menu actions
    export_training: QtGui.QAction
    archive_behavior: QtGui.QAction
    prune_action: QtGui.QAction
    clear_cache: QtGui.QAction

    # View menu actions
    view_playlist: QtGui.QAction
    show_track: QtGui.QAction
    overlay_pose: QtGui.QAction
    overlay_landmark: QtGui.QAction
    overlay_segmentation: QtGui.QAction
    behavior_search: QtGui.QAction

    # Timeline actions
    timeline_labels_preds: QtGui.QAction
    timeline_labels: QtGui.QAction
    timeline_preds: QtGui.QAction
    timeline_all_animals: QtGui.QAction
    timeline_selected_animal: QtGui.QAction

    # Label overlay actions
    label_overlay_none: QtGui.QAction
    label_overlay_labels: QtGui.QAction
    label_overlay_preds: QtGui.QAction

    # Identity overlay actions
    identity_overlay_centroid: QtGui.QAction
    identity_overlay_floating: QtGui.QAction
    identity_overlay_minimal: QtGui.QAction
    identity_overlay_bbox: QtGui.QAction

    # Feature menu actions
    enable_cm_units: QtGui.QAction
    enable_window_features: QtGui.QAction
    enable_fft_features: QtGui.QAction
    enable_social_features: QtGui.QAction
    enable_landmark_features: dict[str, QtGui.QAction]
    enable_segmentation_features: QtGui.QAction


class MenuBuilder:
    """Builds and configures all menus for the MainWindow."""

    def __init__(self, main_window: "MainWindow", app_name: str, app_name_long: str):
        """Initialize the menu builder.

        Args:
            main_window: The MainWindow instance
            app_name: Short application name
            app_name_long: Full application name
        """
        self.main_window = main_window
        self.app_name = app_name
        self.app_name_long = app_name_long
        self._central_widget = main_window._central_widget
        self._settings = main_window._settings
        self.handlers = main_window.menu_handlers  # Reference to MenuHandlers instance

    def build_menus(self) -> MenuReferences:
        """Create all menus and return references to menus and actions.

        Returns:
            MenuReferences object containing all menu and action references
        """
        menu_bar = self.main_window.menuBar()

        app_menu = menu_bar.addMenu(self.app_name)
        file_menu = menu_bar.addMenu("File")
        view_menu = menu_bar.addMenu("View")
        feature_menu = menu_bar.addMenu("Features")
        window_menu = menu_bar.addMenu("Window")

        # Build each menu
        self._build_app_menu(app_menu)
        file_actions = self._build_file_menu(file_menu)
        view_actions = self._build_view_menu(view_menu)
        feature_actions = self._build_feature_menu(feature_menu)
        self._build_window_menu(window_menu)

        # Add global shortcuts
        self._add_global_shortcuts()

        return MenuReferences(
            app_menu=app_menu,
            file_menu=file_menu,
            view_menu=view_menu,
            feature_menu=feature_menu,
            window_menu=window_menu,
            clear_cache=self._clear_cache,
            **file_actions,
            **view_actions,
            **feature_actions,
        )

    def _build_app_menu(self, menu: QtWidgets.QMenu) -> None:
        """Build the application menu (About, User Guide, Quit, etc.)."""
        # About action
        about_action = QtGui.QAction(f" &About {self.app_name}", self.main_window)
        about_action.setStatusTip("About this application")
        about_action.triggered.connect(self.handlers.show_about_dialog)
        menu.addAction(about_action)

        # User guide action
        user_guide_action = QtGui.QAction(" &User Guide", self.main_window)
        user_guide_action.setStatusTip("Open User Guide")
        user_guide_action.setShortcut(QtGui.QKeySequence("Ctrl+U"))
        user_guide_action.triggered.connect(self.handlers.open_user_guide)
        menu.addAction(user_guide_action)

        # License action
        license_action = QtGui.QAction("View License Agreement", self.main_window)
        license_action.setStatusTip("View License Agreement")
        license_action.triggered.connect(self.handlers.view_license)
        menu.addAction(license_action)

        # Session tracking toggle
        session_tracking_action = QtGui.QAction("Enable Session Tracking", self.main_window)
        session_tracking_action.setStatusTip("Enable or disable session tracking")
        session_tracking_action.setCheckable(True)
        session_tracking_action.triggered.connect(self.handlers.on_session_tracking_triggered)
        session_tracking_action.setChecked(
            self._settings.value(SESSION_TRACKING_ENABLED_KEY, False, type=bool)
        )
        menu.addAction(session_tracking_action)

        # Clear cache action (store as instance variable for later reference)
        self._clear_cache = QtGui.QAction("Clear Project Cache", self.main_window)
        self._clear_cache.setStatusTip("Clear Project Cache")
        self._clear_cache.setEnabled(False)
        self._clear_cache.triggered.connect(self.handlers.clear_cache_action)
        menu.addAction(self._clear_cache)

        # Quit action
        exit_action = QtGui.QAction(f" &Quit {self.app_name}", self.main_window)
        exit_action.setShortcut(QtGui.QKeySequence("Ctrl+Q"))
        exit_action.setStatusTip("Exit application")
        exit_action.triggered.connect(QtCore.QCoreApplication.quit)
        menu.addAction(exit_action)

    def _build_file_menu(self, menu: QtWidgets.QMenu) -> dict:
        """Build the File menu.

        Returns:
            Dictionary of file menu action references
        """
        # Open project action
        open_action = QtGui.QAction("&Open Project", self.main_window)
        open_action.setShortcut(QtGui.QKeySequence("Ctrl+O"))
        open_action.setStatusTip("Open Project")
        open_action.triggered.connect(self.handlers.show_project_open_dialog)
        menu.addAction(open_action)

        # Open recent submenu
        open_recent_menu = QtWidgets.QMenu("Open Recent", self.main_window)
        menu.addMenu(open_recent_menu)

        # Export training data action
        export_training = QtGui.QAction("Export Training Data", self.main_window)
        export_training.setShortcut(QtGui.QKeySequence("Ctrl+T"))
        export_training.setStatusTip("Export training data for this classifier")
        export_training.setEnabled(False)
        export_training.triggered.connect(self.handlers.export_training_data)
        menu.addAction(export_training)

        # Archive behavior action
        archive_behavior = QtGui.QAction("Archive Behavior", self.main_window)
        archive_behavior.setStatusTip("Open Archive Behavior Dialog")
        archive_behavior.setEnabled(False)
        archive_behavior.triggered.connect(self.handlers.open_archive_behavior_dialog)
        menu.addAction(archive_behavior)

        # Prune project action
        prune_action = QtGui.QAction("Prune Project", self.main_window)
        prune_action.setStatusTip("Remove videos with no labels")
        prune_action.setEnabled(False)
        prune_action.triggered.connect(self.handlers.show_project_pruning_dialog)
        menu.addAction(prune_action)

        return {
            "open_recent_menu": open_recent_menu,
            "export_training": export_training,
            "archive_behavior": archive_behavior,
            "prune_action": prune_action,
        }

    def _build_view_menu(self, menu: QtWidgets.QMenu) -> dict:
        """Build the View menu.

        Returns:
            Dictionary of view menu action references
        """
        # View playlist action
        view_playlist = QtGui.QAction("View Playlist", self.main_window)
        view_playlist.setCheckable(True)
        view_playlist.triggered.connect(self.handlers.set_video_list_visibility)
        menu.addAction(view_playlist)

        # Timeline submenu
        timeline_actions = self._build_timeline_submenu(menu)

        # Label overlay submenu
        label_overlay_actions = self._build_label_overlay_submenu(menu)

        # Identity overlay submenu
        identity_overlay_actions = self._build_identity_overlay_submenu(menu)

        # Overlay annotations
        overlay_annotations = QtGui.QAction("Overlay Annotations", self.main_window)
        overlay_annotations.setCheckable(True)
        overlay_annotations.setChecked(self._central_widget.overlay_annotations_enabled)
        overlay_annotations.triggered.connect(
            lambda checked: setattr(self._central_widget, "overlay_annotations_enabled", checked)
        )
        menu.addAction(overlay_annotations)

        # Show track
        show_track = QtGui.QAction("Show Track", self.main_window)
        show_track.setCheckable(True)
        show_track.triggered.connect(self.handlers.set_animal_track_visibility)
        menu.addAction(show_track)

        # Overlay pose
        overlay_pose = QtGui.QAction("Overlay Pose", self.main_window)
        overlay_pose.setCheckable(True)
        overlay_pose.triggered.connect(self.handlers.set_pose_overlay_visibility)
        menu.addAction(overlay_pose)

        # Overlay landmarks
        overlay_landmark = QtGui.QAction("Overlay Landmarks", self.main_window)
        overlay_landmark.setCheckable(True)
        overlay_landmark.triggered.connect(self.handlers.set_landmark_overlay_visibility)
        menu.addAction(overlay_landmark)

        # Overlay segmentation
        overlay_segmentation = QtGui.QAction("Overlay Segmentation", self.main_window)
        overlay_segmentation.setCheckable(True)
        overlay_segmentation.triggered.connect(self.handlers.set_segmentation_overlay_visibility)
        menu.addAction(overlay_segmentation)

        # Behavior search
        behavior_search = QtGui.QAction("Search Behaviors", self.main_window)
        behavior_search.setShortcut(QtGui.QKeySequence.StandardKey.Find)
        behavior_search.setStatusTip("Search for behaviors")
        behavior_search.setEnabled(False)
        behavior_search.triggered.connect(self.handlers.search_behaviors)
        menu.addAction(behavior_search)

        return {
            "view_playlist": view_playlist,
            "show_track": show_track,
            "overlay_pose": overlay_pose,
            "overlay_landmark": overlay_landmark,
            "overlay_segmentation": overlay_segmentation,
            "behavior_search": behavior_search,
            **timeline_actions,
            **label_overlay_actions,
            **identity_overlay_actions,
        }

    def _build_timeline_submenu(self, parent_menu: QtWidgets.QMenu) -> dict:
        """Build the Timeline submenu.

        Returns:
            Dictionary of timeline action references
        """
        timeline_menu = QtWidgets.QMenu("Timeline", self.main_window)
        parent_menu.addMenu(timeline_menu)

        # First group: Labels & Predictions, Labels, Predictions
        timeline_group = QtGui.QActionGroup(self.main_window)
        timeline_group.setExclusive(True)

        timeline_labels_preds = QtGui.QAction(
            "Labels && Predictions", self.main_window, checkable=True
        )
        timeline_labels = QtGui.QAction("Labels", self.main_window, checkable=True)
        timeline_preds = QtGui.QAction("Predictions", self.main_window, checkable=True)

        timeline_group.addAction(timeline_labels_preds)
        timeline_group.addAction(timeline_labels)
        timeline_group.addAction(timeline_preds)

        timeline_menu.addAction(timeline_labels_preds)
        timeline_menu.addAction(timeline_labels)
        timeline_menu.addAction(timeline_preds)

        timeline_labels_preds.triggered.connect(self.handlers.on_timeline_view_mode_changed)
        timeline_labels.triggered.connect(self.handlers.on_timeline_view_mode_changed)
        timeline_preds.triggered.connect(self.handlers.on_timeline_view_mode_changed)

        timeline_menu.addSeparator()

        # Second group: All Animals, Selected Animals
        animal_group = QtGui.QActionGroup(self.main_window)
        animal_group.setExclusive(True)

        timeline_all_animals = QtGui.QAction("All Animals", self.main_window, checkable=True)
        timeline_selected_animal = QtGui.QAction(
            "Selected Animal", self.main_window, checkable=True
        )

        animal_group.addAction(timeline_all_animals)
        animal_group.addAction(timeline_selected_animal)

        timeline_menu.addAction(timeline_all_animals)
        timeline_menu.addAction(timeline_selected_animal)

        timeline_all_animals.triggered.connect(self.handlers.on_timeline_identity_mode_changed)
        timeline_selected_animal.triggered.connect(self.handlers.on_timeline_identity_mode_changed)

        # Set defaults
        timeline_labels_preds.setChecked(True)
        timeline_selected_animal.setChecked(True)

        return {
            "timeline_labels_preds": timeline_labels_preds,
            "timeline_labels": timeline_labels,
            "timeline_preds": timeline_preds,
            "timeline_all_animals": timeline_all_animals,
            "timeline_selected_animal": timeline_selected_animal,
        }

    def _build_label_overlay_submenu(self, parent_menu: QtWidgets.QMenu) -> dict:
        """Build the Label Overlay submenu.

        Returns:
            Dictionary of label overlay action references
        """
        label_overlay_menu = QtWidgets.QMenu("Label Overlay", self.main_window)
        parent_menu.addMenu(label_overlay_menu)

        label_overlay_group = QtGui.QActionGroup(self.main_window)
        label_overlay_group.setExclusive(True)

        label_overlay_none = QtGui.QAction(
            "No Overlay", self.main_window, checkable=True, checked=True
        )
        label_overlay_labels = QtGui.QAction("Labels", self.main_window, checkable=True)
        label_overlay_preds = QtGui.QAction("Predictions", self.main_window, checkable=True)

        label_overlay_group.addAction(label_overlay_none)
        label_overlay_group.addAction(label_overlay_labels)
        label_overlay_group.addAction(label_overlay_preds)

        label_overlay_menu.addAction(label_overlay_none)
        label_overlay_menu.addAction(label_overlay_labels)
        label_overlay_menu.addAction(label_overlay_preds)

        label_overlay_none.triggered.connect(self.handlers.on_label_overlay_mode_changed)
        label_overlay_labels.triggered.connect(self.handlers.on_label_overlay_mode_changed)
        label_overlay_preds.triggered.connect(self.handlers.on_label_overlay_mode_changed)

        return {
            "label_overlay_none": label_overlay_none,
            "label_overlay_labels": label_overlay_labels,
            "label_overlay_preds": label_overlay_preds,
        }

    def _build_identity_overlay_submenu(self, parent_menu: QtWidgets.QMenu) -> dict:
        """Build the Identity Overlay submenu.

        Returns:
            Dictionary of identity overlay action references
        """
        identity_overlay_menu = QtWidgets.QMenu("Identity Overlay", self.main_window)
        parent_menu.addMenu(identity_overlay_menu)

        identity_overlay_group = QtGui.QActionGroup(self.main_window)
        identity_overlay_group.setExclusive(True)

        identity_overlay_centroid = QtGui.QAction("Centroid", self.main_window, checkable=True)
        identity_overlay_floating = QtGui.QAction("Floating", self.main_window, checkable=True)
        identity_overlay_minimal = QtGui.QAction("Minimalist", self.main_window, checkable=True)
        identity_overlay_bbox = QtGui.QAction("Bounding Box", self.main_window, checkable=True)

        identity_overlay_group.addAction(identity_overlay_centroid)
        identity_overlay_group.addAction(identity_overlay_floating)
        identity_overlay_group.addAction(identity_overlay_minimal)
        identity_overlay_group.addAction(identity_overlay_bbox)

        identity_overlay_menu.addAction(identity_overlay_centroid)
        identity_overlay_menu.addAction(identity_overlay_floating)
        identity_overlay_menu.addAction(identity_overlay_minimal)
        identity_overlay_menu.addAction(identity_overlay_bbox)

        # Set checked state based on current mode
        match self._central_widget.id_overlay_mode:
            case PlayerWidget.IdentityOverlayMode.CENTROID:
                identity_overlay_centroid.setChecked(True)
            case PlayerWidget.IdentityOverlayMode.FLOATING:
                identity_overlay_floating.setChecked(True)
            case PlayerWidget.IdentityOverlayMode.BBOX:
                identity_overlay_bbox.setChecked(True)
            case _:
                identity_overlay_minimal.setChecked(True)

        # Connect actions
        identity_overlay_centroid.triggered.connect(
            lambda: setattr(
                self._central_widget, "id_overlay_mode", PlayerWidget.IdentityOverlayMode.CENTROID
            )
        )
        identity_overlay_floating.triggered.connect(
            lambda: setattr(
                self._central_widget, "id_overlay_mode", PlayerWidget.IdentityOverlayMode.FLOATING
            )
        )
        identity_overlay_minimal.triggered.connect(
            lambda: setattr(
                self._central_widget, "id_overlay_mode", PlayerWidget.IdentityOverlayMode.MINIMAL
            )
        )
        identity_overlay_bbox.triggered.connect(
            lambda: setattr(
                self._central_widget, "id_overlay_mode", PlayerWidget.IdentityOverlayMode.BBOX
            )
        )

        return {
            "identity_overlay_centroid": identity_overlay_centroid,
            "identity_overlay_floating": identity_overlay_floating,
            "identity_overlay_minimal": identity_overlay_minimal,
            "identity_overlay_bbox": identity_overlay_bbox,
        }

    def _build_feature_menu(self, menu: QtWidgets.QMenu) -> dict:
        """Build the Features menu.

        Returns:
            Dictionary of feature menu action references
        """
        # CM Units
        enable_cm_units = QtGui.QAction("CM Units", self.main_window)
        enable_cm_units.setCheckable(True)
        enable_cm_units.triggered.connect(self.handlers.set_use_cm_units)
        menu.addAction(enable_cm_units)

        # Window features
        enable_window_features = QtGui.QAction("Enable Window Features", self.main_window)
        enable_window_features.setCheckable(True)
        enable_window_features.triggered.connect(self.handlers.set_window_features_enabled)
        menu.addAction(enable_window_features)

        # Signal features
        enable_fft_features = QtGui.QAction("Enable Signal Features", self.main_window)
        enable_fft_features.setCheckable(True)
        enable_fft_features.triggered.connect(self.handlers.set_fft_features_enabled)
        menu.addAction(enable_fft_features)

        # Social features
        enable_social_features = QtGui.QAction("Enable Social Features", self.main_window)
        enable_social_features.setCheckable(True)
        enable_social_features.triggered.connect(self.handlers.set_social_features_enabled)
        menu.addAction(enable_social_features)

        # Landmark features (static objects)
        enable_landmark_features = {}
        for landmark_name in LandmarkFeatureGroup.feature_map:
            landmark_action = QtGui.QAction(
                f"Enable {landmark_name.capitalize()} Features", self.main_window
            )
            landmark_action.setCheckable(True)
            # Use lambda to pass landmark_name explicitly instead of using sender()
            landmark_action.triggered.connect(
                lambda checked,
                name=landmark_name: self.handlers.set_static_object_features_enabled(checked, name)
            )
            menu.addAction(landmark_action)
            enable_landmark_features[landmark_name] = landmark_action

        # Segmentation features
        enable_segmentation_features = QtGui.QAction(
            "Enable Segmentation Features", self.main_window
        )
        enable_segmentation_features.setCheckable(True)
        enable_segmentation_features.triggered.connect(
            self.handlers.set_segmentation_features_enabled
        )
        menu.addAction(enable_segmentation_features)

        return {
            "enable_cm_units": enable_cm_units,
            "enable_window_features": enable_window_features,
            "enable_fft_features": enable_fft_features,
            "enable_social_features": enable_social_features,
            "enable_landmark_features": enable_landmark_features,
            "enable_segmentation_features": enable_segmentation_features,
        }

    def _build_window_menu(self, menu: QtWidgets.QMenu) -> None:
        """Build the Window menu."""
        # Minimize action
        minimize_action = QtGui.QAction("Minimize", self.main_window)
        minimize_action.setShortcut(QtGui.QKeySequence("Ctrl+M"))
        minimize_action.triggered.connect(self.main_window.showMinimized)
        menu.addAction(minimize_action)

        # Zoom action
        zoom_action = QtGui.QAction("Zoom", self.main_window)
        zoom_action.triggered.connect(self.handlers.toggle_zoom)
        menu.addAction(zoom_action)

        # Bring All to Front action
        bring_all_to_front_action = QtGui.QAction("Bring All to Front", self.main_window)
        bring_all_to_front_action.triggered.connect(self.handlers.bring_all_windows_to_front)
        menu.addAction(bring_all_to_front_action)

        menu.addSeparator()

        # Dynamic window list updated when menu is shown
        menu.aboutToShow.connect(self.handlers.update_window_menu)

    def _add_global_shortcuts(self) -> None:
        """Add global shortcuts that aren't in menus."""
        # Select all action
        select_all_action = QtGui.QAction(self.main_window)
        select_all_action.setShortcut(QtGui.QKeySequence.StandardKey.SelectAll)
        select_all_action.triggered.connect(self.handlers.handle_select_all)
        self.main_window.addAction(select_all_action)
