from PySide6 import QtGui, QtWidgets

from jabs.behavior_search import (
    BehaviorSearchQuery,
    LabelBehaviorSearchQuery,
    PredictionBehaviorSearchQuery,
)
from jabs.project.project import Project


class BehaviorSearchDialog(QtWidgets.QDialog):
    """Dialog for selecting behavior search parameters.

    This dialog allows the user to choose between search methods:
    1. Label Search: Searches for specific behavior labels.
    2. Prediction Search: Searches based on prediction probabilities.
    """

    def __init__(self, project: Project, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Behavior Search")
        self.setModal(True)
        self.resize(500, 320)

        proj_settings = project.settings
        self._behavior_labels = sorted(proj_settings.get("behavior", {}).keys())

        # === Main Layout ===
        main_layout = QtWidgets.QVBoxLayout(self)

        # select search method
        method_layout = QtWidgets.QHBoxLayout()
        method_label = QtWidgets.QLabel("Search method:")
        self.method_combo = QtWidgets.QComboBox()
        self.method_combo.addItems(["Label Search", "Prediction Search"])
        method_layout.addWidget(method_label)
        method_layout.addWidget(self.method_combo)
        method_layout.addStretch()
        main_layout.addLayout(method_layout)

        # # checkboxes for limiting search scope
        # self.limit_to_video_checkbox = QtWidgets.QCheckBox(
        #     "Limit search to selected video"
        # )
        # self.limit_to_identity_checkbox = QtWidgets.QCheckBox(
        #     "Limit search to selected identity"
        # )
        # main_layout.addWidget(self.limit_to_video_checkbox)
        # main_layout.addWidget(self.limit_to_identity_checkbox)

        # === GroupBox container with stacked widget ===
        group_box = QtWidgets.QGroupBox()
        group_box_layout = QtWidgets.QVBoxLayout(group_box)
        group_box.setStyleSheet("QGroupBox { margin-top: 10px; }")

        self.stacked_widget = QtWidgets.QStackedWidget()
        group_box_layout.addWidget(self.stacked_widget)
        main_layout.addWidget(group_box)

        # --- Label Search Panel ---
        label_widget = QtWidgets.QWidget()
        label_layout = QtWidgets.QVBoxLayout(label_widget)

        # behavior dropdown
        behavior_row = QtWidgets.QHBoxLayout()
        behavior_label = QtWidgets.QLabel("Behavior Label:")
        self.behavior_combo = QtWidgets.QComboBox()
        self.behavior_combo.addItems(["All Behaviors", *self._behavior_labels])
        behavior_row.addWidget(behavior_label)
        behavior_row.addWidget(self.behavior_combo)
        behavior_row.addStretch()
        label_layout.addLayout(behavior_row)

        # Radio buttons for label selection
        self.label_radio_group = QtWidgets.QButtonGroup(label_widget)
        self.radio_both = QtWidgets.QRadioButton("Positive and negative behavior labels")
        self.radio_positive = QtWidgets.QRadioButton("Only positive behavior labels")
        self.radio_negative = QtWidgets.QRadioButton("Only negative behavior labels")
        self.radio_both.setChecked(True)

        self.label_radio_group.addButton(self.radio_both, 0)
        self.label_radio_group.addButton(self.radio_positive, 1)
        self.label_radio_group.addButton(self.radio_negative, 2)

        label_layout.addWidget(self.radio_both)
        label_layout.addWidget(self.radio_positive)
        label_layout.addWidget(self.radio_negative)
        label_layout.addStretch()
        self.stacked_widget.addWidget(label_widget)

        # --- Prediction Search Panel ---
        prediction_widget = QtWidgets.QWidget()
        prediction_layout = QtWidgets.QVBoxLayout(prediction_widget)

        # prediction behavior dropdown
        prediction_behavior_row = QtWidgets.QHBoxLayout()
        prediction_behavior_label = QtWidgets.QLabel("Behavior:")
        self.prediction_behavior_combo = QtWidgets.QComboBox()
        self.prediction_behavior_combo.addItems(["All Behaviors", *self._behavior_labels])
        prediction_behavior_row.addWidget(prediction_behavior_label)
        prediction_behavior_row.addWidget(self.prediction_behavior_combo)
        prediction_behavior_row.addStretch()
        prediction_layout.addLayout(prediction_behavior_row)

        # Probability range row
        prob_range_layout = QtWidgets.QHBoxLayout()
        self.prob_greater_value = QtWidgets.QLineEdit()
        self.prob_greater_value.setValidator(QtGui.QDoubleValidator(0.0, 1.0, 3))
        self.prob_greater_value.setText("0.0")
        prob_range_layout.addWidget(self.prob_greater_value)

        prob_range_layout.addWidget(QtWidgets.QLabel("≤ behavior probability ≤"))

        self.prob_less_value = QtWidgets.QLineEdit()
        self.prob_less_value.setValidator(QtGui.QDoubleValidator(0.0, 1.0, 3))
        self.prob_less_value.setText("1.0")
        prob_range_layout.addWidget(self.prob_less_value)

        prediction_layout.addLayout(prob_range_layout)

        min_frame_layout = QtWidgets.QHBoxLayout()
        frame_label = QtWidgets.QLabel("Min contiguous matching frames:")
        self.min_frame_count = QtWidgets.QLineEdit()
        self.min_frame_count.setValidator(QtGui.QIntValidator(1, 1000000))
        self.min_frame_count.setText("1")
        min_frame_layout.addWidget(frame_label)
        min_frame_layout.addWidget(self.min_frame_count)

        prediction_layout.addLayout(min_frame_layout)
        prediction_layout.addStretch()

        self.stacked_widget.addWidget(prediction_widget)

        # Dialog buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)

        # Update view based on combo box
        self.method_combo.currentIndexChanged.connect(self.stacked_widget.setCurrentIndex)

    @property
    def behavior_search_query(self) -> BehaviorSearchQuery:
        """Return a BehaviorSearchQuery based on the current dialog values."""
        if self.method_combo.currentIndex() == 0:  # Label Search
            radio_id = self.label_radio_group.checkedId()
            behavior_label = (
                self.behavior_combo.currentText()
                if self.behavior_combo.currentIndex() != 0
                else None
            )

            return LabelBehaviorSearchQuery(
                behavior_label=behavior_label,
                positive=radio_id in (0, 1),
                negative=radio_id in (0, 2),
            )

        else:  # Prediction Search
            behavior_label = (
                self.prediction_behavior_combo.currentText()
                if self.prediction_behavior_combo.currentIndex() != 0
                else None
            )
            prob_greater_value = float(self.prob_greater_value.text())
            prob_less_value = float(self.prob_less_value.text())
            min_frames = int(self.min_frame_count.text())

            return PredictionBehaviorSearchQuery(
                behavior_label=behavior_label,
                prob_greater_value=prob_greater_value,
                prob_less_value=prob_less_value,
                min_contiguous_frames=min_frames,
            )
