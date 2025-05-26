from PySide6 import QtGui, QtWidgets

from jabs.ui.behavior_search_query import (
    BehaviorSearchQuery,
    LabelBehaviorSearchQuery,
    PredictionLabelSearchQuery,
)


class BehaviorSearchDialog(QtWidgets.QDialog):
    """Dialog for selecting behavior search parameters.

    This dialog allows the user to choose between search methods:
    1. Label Search: Searches for specific behavior labels.
    2. Prediction Search: Searches based on prediction probabilities.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Behavior Search")
        self.setModal(True)
        self.resize(500, 320)

        # === Main Layout ===
        main_layout = QtWidgets.QVBoxLayout(self)

        # Dropdown selector
        method_layout = QtWidgets.QHBoxLayout()
        method_label = QtWidgets.QLabel("Search method:")
        self.method_combo = QtWidgets.QComboBox()
        self.method_combo.addItems(["Label Search", "Prediction Search"])
        method_layout.addWidget(method_label)
        method_layout.addWidget(self.method_combo)
        method_layout.addStretch()
        main_layout.addLayout(method_layout)

        # === GroupBox container with stacked widget ===
        group_box = QtWidgets.QGroupBox()  # No title for neutral framing
        group_box_layout = QtWidgets.QVBoxLayout(group_box)
        group_box.setStyleSheet("QGroupBox { margin-top: 10px; }")

        self.stacked_widget = QtWidgets.QStackedWidget()
        group_box_layout.addWidget(self.stacked_widget)
        main_layout.addWidget(group_box)

        # --- Label Search Panel ---
        label_widget = QtWidgets.QWidget()
        label_layout = QtWidgets.QVBoxLayout(label_widget)
        self.positive_checkbox = QtWidgets.QCheckBox("Positive behavior labels")
        self.negative_checkbox = QtWidgets.QCheckBox("Negative behavior labels")
        label_layout.addWidget(self.positive_checkbox)
        label_layout.addWidget(self.negative_checkbox)
        label_layout.addStretch()
        self.stacked_widget.addWidget(label_widget)

        # --- Prediction Search Panel ---
        prediction_widget = QtWidgets.QWidget()
        prediction_layout = QtWidgets.QVBoxLayout(prediction_widget)

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
        self.method_combo.currentIndexChanged.connect(
            self.stacked_widget.setCurrentIndex
        )

    @property
    def behavior_search_query(self):
        """Return a BehaviorSearchQuery based on the current dialog values."""
        if self.method_combo.currentIndex() == 0:  # Label Search
            label_query = LabelBehaviorSearchQuery(
                positive=self.positive_checkbox.isChecked(),
                negative=self.negative_checkbox.isChecked(),
            )

            return BehaviorSearchQuery(
                label_search_query=label_query,
                prediction_search_query=None,
            )
        else:  # Prediction Search
            prob_greater_value = float(self.prob_greater_value.text())
            prob_less_value = float(self.prob_less_value.text())
            min_frames = int(self.min_frame_count.text())
            prediction_query = PredictionLabelSearchQuery(
                prob_greater_value=prob_greater_value,
                prob_less_value=prob_less_value,
                min_contiguous_frames=min_frames,
            )

            return BehaviorSearchQuery(
                label_search_query=None,
                prediction_search_query=prediction_query,
            )
