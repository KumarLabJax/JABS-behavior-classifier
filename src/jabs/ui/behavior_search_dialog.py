from enum import IntEnum, auto

from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import QObject, QThread, Signal

from jabs.behavior_search import (
    BehaviorSearchQuery,
    LabelBehaviorSearchQuery,
    PredictionBehaviorSearchQuery,
    PredictionSearchKind,
)
from jabs.behavior_search.behavior_search_util import TimelineAnnotationSearchQuery
from jabs.project import Project


class _SearchMethod(IntEnum):
    """Enumeration for different search methods in the behavior search dialog."""

    LABEL = 0
    PREDICTION = auto()
    TIMELINE_ANNOTATION = auto()


class _GatherTimelineAnnotationTagsWorker(QObject):
    """Asynchronous worker to gather timeline annotation tags."""

    finished = Signal(list)

    def __init__(self, project: Project):
        super().__init__()
        self.project = project

    def run(self):
        all_tags = set()

        video_manager = self.project.video_manager
        for video in video_manager.videos:
            anno_dict = video_manager.load_annotations(video)
            if anno_dict is not None:
                for annotation in anno_dict.get("annotations", []):
                    if "tag" in annotation:
                        all_tags.add(annotation["tag"])

        self.finished.emit(sorted(all_tags))


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
        self.method_combo.addItems(
            [
                "Label Search",
                "Prediction Search",
                "Timeline Annotation Search",
            ]
        )
        method_layout.addWidget(method_label)
        method_layout.addWidget(self.method_combo)
        method_layout.addStretch()
        main_layout.addLayout(method_layout)

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

        # Prediction radio buttons
        self.pred_radio_group = QtWidgets.QButtonGroup(prediction_widget)
        self.radio_pred_positive = QtWidgets.QRadioButton("Positive behavior predictions")
        self.radio_pred_negative = QtWidgets.QRadioButton("Negative behavior predictions")
        self.radio_pred_range = QtWidgets.QRadioButton("Behavior probability range:")
        self.radio_pred_positive.setChecked(True)

        self.pred_radio_group.addButton(self.radio_pred_positive, 0)
        self.pred_radio_group.addButton(self.radio_pred_negative, 1)
        self.pred_radio_group.addButton(self.radio_pred_range, 2)

        prediction_layout.addWidget(self.radio_pred_positive)
        prediction_layout.addWidget(self.radio_pred_negative)
        prediction_layout.addWidget(self.radio_pred_range)

        # Probability range row
        prob_range_layout = QtWidgets.QHBoxLayout()
        self.prob_greater_value = QtWidgets.QLineEdit()
        self.prob_greater_value.setValidator(QtGui.QDoubleValidator(0.0, 1.0, 3))
        self.prob_greater_value.setText("0.0")
        prob_range_layout.addWidget(self.prob_greater_value)

        prob_range_layout.addWidget(QtWidgets.QLabel("≤ probability ≤"))

        self.prob_less_value = QtWidgets.QLineEdit()
        self.prob_less_value.setValidator(QtGui.QDoubleValidator(0.0, 1.0, 3))
        self.prob_less_value.setText("1.0")
        prob_range_layout.addWidget(self.prob_less_value)

        prediction_layout.addLayout(prob_range_layout)

        # Enable/disable probability range fields based on radio selection
        def update_prob_range_enabled():
            enabled = self.radio_pred_range.isChecked()
            self.prob_greater_value.setEnabled(enabled)
            self.prob_less_value.setEnabled(enabled)

        self.radio_pred_positive.toggled.connect(update_prob_range_enabled)
        self.radio_pred_negative.toggled.connect(update_prob_range_enabled)
        self.radio_pred_range.toggled.connect(update_prob_range_enabled)
        update_prob_range_enabled()

        prediction_layout.addSpacing(10)

        # Frame count range
        self.pred_min_frame_count, self.pred_max_frame_count = self._add_frame_range_ui_to_layout(
            prediction_layout
        )
        prediction_layout.addStretch()

        self.stacked_widget.addWidget(prediction_widget)

        # --- Timeline Annotation Search Panel ---
        timeline_anno_widget = QtWidgets.QWidget()
        timeline_anno_layout = QtWidgets.QVBoxLayout(timeline_anno_widget)

        # timeline annotation tag dropdown. Because computing tags can be relatively
        # expensive and requires I/O, we will defer adding them here and compute
        # them asynchronously.
        timeline_anno_tag_row = QtWidgets.QHBoxLayout()
        timeline_anno_tag_label = QtWidgets.QLabel("Annotation Tag:")
        self.timeline_anno_tag_combo = QtWidgets.QComboBox()
        self.timeline_anno_tag_combo.addItems(["Any Tag"])
        timeline_anno_tag_row.addWidget(timeline_anno_tag_label)
        timeline_anno_tag_row.addWidget(self.timeline_anno_tag_combo)
        timeline_anno_tag_row.addStretch()
        timeline_anno_layout.addLayout(timeline_anno_tag_row)

        self.timeline_anno_min_frame_count, self.timeline_anno_max_frame_count = (
            self._add_frame_range_ui_to_layout(timeline_anno_layout)
        )
        timeline_anno_layout.addStretch()

        self.stacked_widget.addWidget(timeline_anno_widget)

        # Dialog buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.validate_and_accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)

        # Update view based on combo box
        self.method_combo.currentIndexChanged.connect(self.stacked_widget.setCurrentIndex)

        # Compute timeline annotation tags asynchronously
        self._start_gather_timeline_annotation_tags_worker(project)

    def _add_frame_range_ui_to_layout(self, layout: QtWidgets.QVBoxLayout):
        """Add frame count UI elements to the given layout."""
        layout.addWidget(QtWidgets.QLabel("Limit results by frame count:"))
        frame_count_range_layout = QtWidgets.QHBoxLayout()
        min_frame_count_edit = QtWidgets.QLineEdit()
        min_frame_count_edit.setValidator(QtGui.QIntValidator(0, 100000))
        min_frame_count_edit.setPlaceholderText("1 (default)")
        frame_count_range_layout.addWidget(min_frame_count_edit)

        frame_count_range_layout.addWidget(QtWidgets.QLabel("≤ frame count ≤"))

        max_frame_count_edit = QtWidgets.QLineEdit()
        max_frame_count_edit.setValidator(QtGui.QIntValidator(0, 100000))
        max_frame_count_edit.setPlaceholderText("∞ (default)")
        frame_count_range_layout.addWidget(max_frame_count_edit)

        layout.addLayout(frame_count_range_layout)

        return min_frame_count_edit, max_frame_count_edit

    def _min_max_frames_valid(
        self,
        min_frames_edit: QtWidgets.QLineEdit,
        max_frames_edit: QtWidgets.QLineEdit,
    ):
        """Validate the minimum and maximum frame count inputs.

        Args:
            min_frames_edit (QtWidgets.QLineEdit): The line edit for minimum frames.
            max_frames_edit (QtWidgets.QLineEdit): The line edit for maximum frames.

        Returns:
            bool: True if valid, False otherwise.
        """
        try:
            min_frames = self._text_to_maybe_int(min_frames_edit.text())
            max_frames = self._text_to_maybe_int(max_frames_edit.text())

            min_frames = min_frames if min_frames is not None else 1
            max_frames = max_frames if max_frames is not None else float("inf")
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Input",
                "Please enter valid integers for frame count.",
            )
            return False

        if min_frames < 1 or max_frames < 1:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Frame Count",
                "Frame counts must be positive integers.",
            )
            return False

        if min_frames > max_frames:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Frame Range",
                "Minimum frame count cannot be greater than maximum frame count.",
            )
            return False

        return True

    def validate_and_accept(self):
        """Validate user input and accept the dialog if all checks pass."""
        match self.method_combo.currentIndex():
            case _SearchMethod.PREDICTION:
                if self.radio_pred_range.isChecked():
                    # if radio_pred_range is selected, check probability values
                    try:
                        min_prob = self._text_to_maybe_float(self.prob_greater_value.text())
                        max_prob = self._text_to_maybe_float(self.prob_less_value.text())
                        if min_prob is None and max_prob is None:
                            QtWidgets.QMessageBox.warning(
                                self,
                                "Invalid Input",
                                "Please enter at least one probability value for the range.",
                            )
                            return

                        min_prob = min_prob if min_prob is not None else 0.0
                        max_prob = max_prob if max_prob is not None else 1.0
                    except ValueError:
                        QtWidgets.QMessageBox.warning(
                            self,
                            "Invalid Input",
                            "Please enter valid numbers for probability range.",
                        )
                        return

                    if not (0.0 <= min_prob <= 1.0 and 0.0 <= max_prob <= 1.0):
                        QtWidgets.QMessageBox.warning(
                            self,
                            "Out of Range",
                            "Probability values must be between 0.0 and 1.0.",
                        )
                        return

                    if min_prob > max_prob:
                        QtWidgets.QMessageBox.warning(
                            self,
                            "Invalid Range",
                            "Minimum probability cannot be greater than maximum probability.",
                        )
                        return

                # check frame range values
                if not self._min_max_frames_valid(
                    self.pred_min_frame_count,
                    self.pred_max_frame_count,
                ):
                    return

            case _SearchMethod.TIMELINE_ANNOTATION:
                # check frame range values
                if not self._min_max_frames_valid(
                    self.timeline_anno_min_frame_count,
                    self.timeline_anno_max_frame_count,
                ):
                    return

        # all checks passed
        self.accept()

    def _start_gather_timeline_annotation_tags_worker(self, project: Project):
        # Create thread and worker instances
        thread = QThread()
        worker = _GatherTimelineAnnotationTagsWorker(project)
        worker.moveToThread(thread)

        def on_finished(tags):
            self.timeline_anno_tag_combo.addItems(tags)
            thread.quit()
            worker.deleteLater()
            thread.deleteLater()

        thread.started.connect(worker.run)
        worker.finished.connect(on_finished)

        # Start the thread
        thread.start()

    def _text_to_maybe_float(self, text: str) -> float | None:
        """Convert text to float, returning None if empty.

        Args:
            text (str): The text to convert.

        Returns:
            float | None: The converted float or None if the text is empty.

        Raises:
            ValueError: If the text cannot be converted to a float.
        """
        text = text.strip()
        return float(text) if text else None

    def _text_to_maybe_int(self, text: str) -> int | None:
        """Convert text to int, returning None if empty.

        Args:
            text (str): The text to convert.

        Returns:
            int | None: The converted int or None if the text is empty.

        Raises:
            ValueError: If the text cannot be converted to an int.
        """
        text = text.strip()
        return int(text) if text else None

    @property
    def behavior_search_query(self) -> BehaviorSearchQuery:
        """Return a BehaviorSearchQuery based on the current dialog values."""
        match self.method_combo.currentIndex():
            case _SearchMethod.LABEL:
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

            case _SearchMethod.PREDICTION:
                behavior_label = (
                    self.prediction_behavior_combo.currentText()
                    if self.prediction_behavior_combo.currentIndex() != 0
                    else None
                )
                min_frames = self._text_to_maybe_int(self.pred_min_frame_count.text())
                max_frames = self._text_to_maybe_int(self.pred_max_frame_count.text())

                if self.radio_pred_positive.isChecked():
                    return PredictionBehaviorSearchQuery(
                        search_kind=PredictionSearchKind.POSITIVE_PREDICTION,
                        behavior_label=behavior_label,
                        min_contiguous_frames=min_frames,
                        max_contiguous_frames=max_frames,
                    )
                elif self.radio_pred_negative.isChecked():
                    return PredictionBehaviorSearchQuery(
                        search_kind=PredictionSearchKind.NEGATIVE_PREDICTION,
                        behavior_label=behavior_label,
                        min_contiguous_frames=min_frames,
                        max_contiguous_frames=max_frames,
                    )
                else:
                    prob_greater_value = self._text_to_maybe_float(self.prob_greater_value.text())
                    prob_less_value = self._text_to_maybe_float(self.prob_less_value.text())

                    return PredictionBehaviorSearchQuery(
                        search_kind=PredictionSearchKind.PROBABILITY_RANGE,
                        behavior_label=behavior_label,
                        prob_greater_value=prob_greater_value,
                        prob_less_value=prob_less_value,
                        min_contiguous_frames=min_frames,
                        max_contiguous_frames=max_frames,
                    )

            case _SearchMethod.TIMELINE_ANNOTATION:
                tag = (
                    self.timeline_anno_tag_combo.currentText()
                    if self.timeline_anno_tag_combo.currentIndex() != 0
                    else None
                )

                min_frames = self._text_to_maybe_int(self.timeline_anno_min_frame_count.text())
                max_frames = self._text_to_maybe_int(self.timeline_anno_max_frame_count.text())

                return TimelineAnnotationSearchQuery(
                    tag=tag,
                    min_contiguous_frames=min_frames,
                    max_contiguous_frames=max_frames,
                )

            case _:
                raise ValueError("Unknown search method selected.")
