from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QResizeEvent, QShowEvent
from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLayout,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from jabs.classifier import Classifier
from jabs.constants import DEFAULT_CALIBRATION_CV, DEFAULT_CALIBRATION_METHOD
from jabs.project.settings_manager import SettingsManager


class CollapsibleSection(QWidget):
    """A simple collapsible section with a header ToolButton and a content area."""

    sizeChanged = Signal()

    def __init__(self, title: str, content: QWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._content = content
        self._toggle_btn = QToolButton(self)
        self._toggle_btn.setStyleSheet("QToolButton { border: none; }")
        self._toggle_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle_btn.setArrowType(Qt.ArrowType.RightArrow)
        self._toggle_btn.setText(title)
        self._toggle_btn.setCheckable(True)
        self._toggle_btn.setChecked(False)
        self._toggle_btn.toggled.connect(self._on_toggled)

        line = QFrame(self)
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)

        self._content.setVisible(False)
        # Ensure the collapsible widget and its content expand to fit content
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self._content.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._toggle_btn)
        lay.addWidget(line)
        lay.addWidget(self._content)

    def _on_toggled(self, checked: bool) -> None:
        self._toggle_btn.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )
        self._content.setVisible(checked)
        self._content.updateGeometry()

        # Ask ancestors to recompute layout so the page grows inside the scroll area
        parent = self.parentWidget()
        if parent is not None and parent.layout() is not None:
            parent.layout().activate()

        if self.layout() is not None:
            self.layout().activate()

        # Let ancestors recompute size hints and notify listeners
        if parent is not None:
            parent.updateGeometry()
        self.updateGeometry()
        self.sizeChanged.emit()


class JabsSettingsDialog(QDialog):
    """
    Dialog for changing project settings.

    Args:
        settings_manager (SettingsManager): Project settings manager used to load and save settings.
        parent (QWidget | None, optional): Parent widget for this dialog. Defaults to None.
    """

    def __init__(self, settings_manager: SettingsManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Project Settings")
        self._settings_manager = settings_manager

        # Allow resizing and show scrollbars if content overflows
        self.setSizeGripEnabled(True)

        # Widgets
        self._calibrate_checkbox = QCheckBox(
            "Enable probability calibration (calibrate_probabilities)"
        )
        self._method_selection = QComboBox()
        self._method_selection.addItems(Classifier.CALIBRATION_METHODS)
        self._cv_selection = QSpinBox()
        self._cv_selection.setRange(2, 10)
        self._cv_selection.setAccelerated(True)
        self._cv_selection.setToolTip("Number of CV folds used inside the calibrator")
        self._save_reliability_checkbox = QCheckBox("Save reliability plots")
        self._save_reliability_checkbox.setToolTip(
            "If enabled, save reliability (calibration) plots after training/validation."
        )

        # Load current values from project settings
        current_settings = settings_manager.jabs_settings
        calibrate = current_settings.get("calibrate_probabilities", False)
        method = current_settings.get("calibration_method", DEFAULT_CALIBRATION_METHOD)
        cv = current_settings.get("calibration_cv", DEFAULT_CALIBRATION_CV)
        save_reliability = current_settings.get("save_reliability_plots", False)

        self._calibrate_checkbox.setChecked(calibrate)
        self._method_selection.setCurrentIndex(max(0, self._method_selection.findText(method)))
        self._cv_selection.setValue(cv)
        self._save_reliability_checkbox.setChecked(save_reliability)

        # Layout for form
        form = QWidget(self)
        grid = QGridLayout(form)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)
        grid.setColumnStretch(0, 0)  # labels column: natural size
        grid.setColumnStretch(1, 0)  # inputs column: natural size
        grid.setColumnStretch(
            2, 1
        )  # consume extra width on the right (keeps content left-aligned)

        # Keep inputs compact; whitespace grows in column 2
        self._method_selection.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._method_selection.setFixedWidth(self._method_selection.sizeHint().width() + 24)
        self._method_selection.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self._cv_selection.setFixedWidth(90)
        self._cv_selection.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self._calibrate_checkbox.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._save_reliability_checkbox.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
        )

        grid.addWidget(QLabel("Calibrate probabilities:"), 0, 0, Qt.AlignmentFlag.AlignRight)
        grid.addWidget(self._calibrate_checkbox, 0, 1)

        grid.addWidget(QLabel("Calibration method:"), 1, 0, Qt.AlignmentFlag.AlignRight)
        grid.addWidget(self._method_selection, 1, 1)

        grid.addWidget(QLabel("calibration cv (folds):"), 2, 0, Qt.AlignmentFlag.AlignRight)
        grid.addWidget(self._cv_selection, 2, 1)

        grid.addWidget(QLabel("Save reliability plots:"), 3, 0, Qt.AlignmentFlag.AlignRight)
        grid.addWidget(self._save_reliability_checkbox, 3, 1)
        grid.addItem(
            QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum), 0, 2, 4, 1
        )

        # Help / inline docs (rich text)
        help_label = QLabel(self)
        help_label.setTextFormat(Qt.TextFormat.RichText)
        help_label.setWordWrap(True)
        help_label.setText(
            """
            <h3>What do these parameters do?</h3>
            <p><b>Calibrate probabilities</b> remaps raw model scores to better probabilities using
            cross-validation inside training. This improves log-loss, Brier score, and makes thresholding
            (e.g., show if p ≥ 0.7) more reliable.</p>

            <ul>
              <li><b>calibration_method</b>:<br/>
                <b>auto (default)</b> — automatically selects between <i>isotonic</i> and <i>sigmoid</i> calibration
                using a simple heuristic based on data size. If each calibration fold has roughly
                <b>≥ 500 labeled samples per class</b>, isotonic is used; otherwise, sigmoid is chosen for stability.
                Larger number of folds (<code>calibration_cv</code> setting) increase the data required for selecting
                isotonic.<br/>
                <b>isotonic:</b> learns a flexible mapping and produces highly accurate probabilities when enough
                data is available, but can overfit if the calibration set is small.<br/>
                <b>sigmoid:</b> (Platt scaling) is smoother and more stable for smaller datasets.
              </li>
              <li><b>calibration_cv</b>: Number of folds used internally by the calibrator. Typical values are <b>3-5</b>.
                Each fold fits the base model and calibrates on held-out training data to avoid leakage.
                Larger values slow training and require more data per fold for isotonic calibration to be effective.
              </li>
            </ul>

            <p><b>Guidance:</b> If your dataset is large (thousands of labeled frames and roughly balanced),
            <i>auto</i> will select isotonic. If it selects sigmoid, you can collect more labels or reduce
            <code>calibration_cv</code> to allow isotonic to activate.</p>

            <p><b>Tip:</b> Most users should leave <code>calibration_method = auto</code>.</p>
            <p><b>Saving reliability plots:</b> If <i>Save reliability plots</i> is enabled, JABS will write reliability
            figures after training/validation to <code>&lt;project dir&gt;/plots/&lt;timestamp&gt;/</code>.
            Each run creates a new timestamped folder so results are easy to compare.</p>
            """
        )
        help_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

        calibration_help_panel = CollapsibleSection("What do these do?", help_label, self)
        calibration_help_panel.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )
        help_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        _help_toggle_btn = (
            calibration_help_panel._toggle_btn
        )  # used to wire scrolling after scroll area is created

        def _reflow_calibration_group():
            """Recompute sizes so the group grows to fit all help content; scrolling handled by the dialog"""
            help_label.adjustSize()
            calibration_help_panel.adjustSize()
            calibration_group.adjustSize()
            page.adjustSize()
            page_layout.activate()
            page.adjustSize()
            # Sync page width so group box fills viewport when scrollbar appears/disappears
            self._sync_page_width()
            # Defer one tick so QScrollArea can recompute its scroll range correctly
            QTimer.singleShot(0, lambda: scroll.ensureWidgetVisible(calibration_help_panel))

        calibration_help_panel.sizeChanged.connect(_reflow_calibration_group)

        # Group box for Model Calibration section
        calibration_group = QGroupBox("Model Calibration", self)
        calibration_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        group_vbox = QVBoxLayout(calibration_group)
        group_vbox.setContentsMargins(12, 12, 12, 12)
        group_vbox.setSpacing(8)
        group_vbox.addWidget(form)
        group_vbox.addWidget(calibration_help_panel)
        group_vbox.addStretch(0)

        # Scrollable page to host settings sections
        page = QWidget(self)
        page.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        page_layout = QVBoxLayout(page)
        page_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinAndMaxSize)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(10)
        page_layout.addWidget(calibration_group)
        page_layout.setAlignment(calibration_group, Qt.AlignmentFlag.AlignTop)
        page_layout.addStretch(1)

        scroll = QScrollArea(self)
        scroll.setWidget(page)
        page.adjustSize()
        scroll.setWidgetResizable(False)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Keep references for width syncing so the group box fills the viewport width
        self._scroll = scroll
        self._page = page

        # Keep content width constant: viewport gutter above matches scrollbar width
        scroll.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)

        def _on_help_toggled(checked: bool) -> None:
            if checked:
                # Defer one tick so QScrollArea can recompute its scroll range correctly, then scroll to the panel
                QTimer.singleShot(0, lambda: scroll.ensureWidgetVisible(calibration_help_panel))

        _help_toggle_btn.toggled.connect(_on_help_toggled)

        # Initial width sync
        self._sync_page_width()

        # Buttons
        btn_box = QDialogButtonBox(self)
        btn_save = btn_box.addButton("Save", QDialogButtonBox.ButtonRole.AcceptRole)
        btn_close = btn_box.addButton("Close", QDialogButtonBox.ButtonRole.RejectRole)
        btn_save.clicked.connect(self._on_save)
        btn_close.clicked.connect(self.reject)

        # Main layout
        main = QVBoxLayout(self)
        main.addWidget(scroll, 1)
        main.addWidget(btn_box)

        self.setLayout(main)

        # Size to content initially, then give a taller starting height; user-resize preserved later
        self.adjustSize()
        self.resize(max(self.width(), 700), max(self.height(), 600))

    def _sync_page_width(self) -> None:
        """Ensure the inner page expands to the scroll viewport width to avoid right-side whitespace."""
        try:
            vp = self._scroll.viewport()
            if vp is not None:
                w = vp.width()
                # Fill available width; when the vertical scrollbar appears, viewport width shrinks automatically
                self._page.setMinimumWidth(w)
                self._page.updateGeometry()
        except Exception:
            pass

    def showEvent(self, e: QShowEvent) -> None:
        """Handle the show event.

        Ensures the settings page width is synchronized with the viewport when the dialog is first shown.

        Args:
            e (QShowEvent): The Qt show event.
        """
        super().showEvent(e)
        self._sync_page_width()

    def resizeEvent(self, e: QResizeEvent) -> None:
        """Handle the resize event.

        Ensures the settings page width matches the viewport width when the dialog is resized.

        Args:
            e (QResizeEvent): The Qt resize event.
        """
        super().resizeEvent(e)
        self._sync_page_width()

    def _on_save(self) -> None:
        """Save settings to project and close dialog."""
        settings = {
            "settings": {
                "calibrate_probabilities": self._calibrate_checkbox.isChecked(),
                "calibration_method": self._method_selection.currentText(),
                "calibration_cv": self._cv_selection.value(),
                "save_reliability_plots": self._save_reliability_checkbox.isChecked(),
            }
        }
        self._settings_manager.save_project_file(settings)
        self.accept()
