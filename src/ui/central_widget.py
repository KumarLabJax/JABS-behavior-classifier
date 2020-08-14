from PyQt5 import QtWidgets, QtCore
import numpy as np

from src.feature_extraction.features import IdentityFeatures
from src.labeler.track_labels import TrackLabels
from src.classifier.skl_classifier import SklClassifier
from src.ui import (
    PlayerWidget,
    ManualLabelWidget,
    TimelineLabelWidget,
    IdentityComboBox,
    FrameLabelsWidget,
)


class CentralWidget(QtWidgets.QWidget):
    """
    QT Widget implementing our main window contents
    """

    def __init__(self, *args, **kwargs):
        super(CentralWidget, self).__init__(*args, **kwargs)

        # initial behavior labels to list in the drop down selection
        self._behaviors = [
            'Walking', 'Sleeping', 'Freezing', 'Grooming', 'Following',
            'Rearing (supported)', 'Rearing (unsupported)'
        ]

        # video player
        self._player_widget = PlayerWidget()
        self._player_widget.updateIdentities.connect(self._set_identities)
        self._player_widget.updateFrameNumber.connect(self._frame_change)

        self._loaded_video = None

        self._project = None
        self._labels = None

        self._selection_start = 0

        # options
        self._frame_jump = 10

        # behavior selection form components
        self.behavior_selection = QtWidgets.QComboBox()
        self.behavior_selection.addItems(self._behaviors)
        self.behavior_selection.currentIndexChanged.connect(
            self._change_behavior)

        add_label_button = QtWidgets.QPushButton("New Behavior")
        add_label_button.clicked.connect(self._new_label)

        behavior_layout = QtWidgets.QVBoxLayout()
        behavior_layout.addWidget(self.behavior_selection)
        behavior_layout.addWidget(add_label_button)

        behavior_group = QtWidgets.QGroupBox("Behavior Selection")
        behavior_group.setLayout(behavior_layout)

        # identity selection form components
        self.identity_selection = IdentityComboBox()
        self.identity_selection.currentIndexChanged.connect(
            self._change_identity)
        self.identity_selection.pop_up_visible.connect(self._identity_popup_visibility_changed)
        self.identity_selection.setEditable(False)
        self.identity_selection.installEventFilter(self.identity_selection)
        identity_layout = QtWidgets.QVBoxLayout()
        identity_layout.addWidget(self.identity_selection)
        identity_group = QtWidgets.QGroupBox("Identity Selection")
        identity_group.setLayout(identity_layout)

        # classifier controls
        self.train_button = QtWidgets.QPushButton("Train")
        self.train_button.clicked.connect(self._train_button_clicked)
        self.classify_button = QtWidgets.QPushButton("Classify")
        classfier_layout = QtWidgets.QVBoxLayout()
        classfier_layout.addWidget(self.train_button)
        classfier_layout.addWidget(self.classify_button)
        classifier_group = QtWidgets.QGroupBox("Classifier")
        classifier_group.setLayout(classfier_layout)

        # label components
        label_layout = QtWidgets.QVBoxLayout()

        self.label_behavior_button = QtWidgets.QPushButton()
        self.label_behavior_button.setText(
            self.behavior_selection.currentText())
        self.label_behavior_button.clicked.connect(self._label_behavior)
        self.label_behavior_button.setStyleSheet("""
            QPushButton {
                background-color: rgb(128, 0, 0);
                border-radius: 4px;
                padding: 2px;
                color: white;
            }
            QPushButton:pressed {
                background-color: rgb(255, 0, 0);
            }
            QPushButton:disabled {
                background-color: rgb(64, 0, 0);
                color: grey;
            }
        """)

        self.label_not_behavior_button = QtWidgets.QPushButton(
            f"Not {self.behavior_selection.currentText()}")
        self.label_not_behavior_button.clicked.connect(self._label_not_behavior)
        self.label_not_behavior_button.setStyleSheet("""
            QPushButton {
                background-color: rgb(0, 0, 128);
                border-radius: 4px;
                padding: 2px;
                color: white;
            }
            QPushButton:pressed {
                background-color: rgb(0, 0, 255);
            }
            QPushButton:disabled {
                background-color: rgb(0, 0, 64);
                color: grey;
            }
        """)

        self.clear_label_button = QtWidgets.QPushButton("Clear Label")
        self.clear_label_button.clicked.connect(self._clear_behavior_label)

        self.select_button = QtWidgets.QPushButton("Select Frames")
        self.select_button.setCheckable(True)
        self.select_button.clicked.connect(self._start_selection)

        # label buttons are disabled unless user has a range of frames selected
        self._disable_label_buttons()

        label_layout.addWidget(self.label_behavior_button)
        label_layout.addWidget(self.label_not_behavior_button)
        label_layout.addWidget(self.clear_label_button)
        label_layout.addWidget(self.select_button)
        label_group = QtWidgets.QGroupBox("Label")
        label_group.setLayout(label_layout)

        # control layout
        control_layout = QtWidgets.QVBoxLayout()
        control_layout.setSpacing(25)
        control_layout.addWidget(behavior_group)
        control_layout.addWidget(identity_group)
        control_layout.addWidget(classifier_group)
        control_layout.addStretch()
        control_layout.addWidget(label_group)

        # label widgets
        self.manual_labels = ManualLabelWidget()
        self.frame_ticks = FrameLabelsWidget()

        # timeline widget
        self.timeline_widget = TimelineLabelWidget()

        # main layout
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self._player_widget, 0, 0)
        layout.addLayout(control_layout, 0, 1)
        layout.addWidget(self.timeline_widget, 1, 0, 1, 2)
        layout.addWidget(self.manual_labels, 2, 0, 1, 2)
        layout.addWidget(self.frame_ticks,3, 0, 1, 2)
        self.setLayout(layout)

        # classifier
        self._classifier = SklClassifier()


    def set_project(self, project):
        """ set the currently opened project """
        self._project = project

        # This will get set when the first video in the project is loaded, but
        # we need to set it to None so that we don't try to cache the current
        # labels when we do so (the current labels belong to the previous
        # project)
        self._labels = None

    def get_labels(self):
        """
        get VideoLabels for currently opened video file
        note: the @property decorator doesn't work with QWidgets so we have
        not implemented this as a property
        """
        return self._labels

    def load_video(self, path):
        """ load new avi file """

        # if we have labels loaded, cache them before opening labels for
        # new video
        if self._labels is not None:
            self._project.cache_annotations(self._labels)
            self._start_selection(False)
            self.select_button.setChecked(False)

        try:
            # open the video
            self._player_widget.load_video(path)

            # load labels for new video and set track for current identity
            self._labels = self._project.load_annotation_track(path)
            self._set_label_track()

            # update ui components with properties of new video
            self.manual_labels.set_num_frames(self._player_widget.num_frames())
            self.manual_labels.set_framerate(self._player_widget.stream_fps())
            self.frame_ticks.set_num_frames(self._player_widget.num_frames())
            self.timeline_widget.set_num_frames(
                self._player_widget.num_frames())

            self._loaded_video = path
        except OSError as e:
            # error loading
            self._labels = None
            self._loaded_video = None
            self._player_widget.reset()
            raise e

    def keyPressEvent(self, event):
        """ handle key press events """

        def begin_select_mode():
            if not self.select_button.isChecked():
                self.select_button.toggle()
                self._start_selection(True)

        key = event.key()
        if key == QtCore.Qt.Key_Left:
            self._player_widget.previous_frame()
        elif key == QtCore.Qt.Key_Right:
            self._player_widget.next_frame()
        elif key == QtCore.Qt.Key_Up:
            self._player_widget.previous_frame(self._frame_jump)
        elif key == QtCore.Qt.Key_Down:
            self._player_widget.next_frame(self._frame_jump)
        elif key == QtCore.Qt.Key_Space:
            self._player_widget.toggle_play()
        elif key == QtCore.Qt.Key_Z:
            if self.select_button.isChecked():
                self._label_behavior()
            else:
                begin_select_mode()
        elif key == QtCore.Qt.Key_X:
            if self.select_button.isChecked():
                self._clear_behavior_label()
            else:
                begin_select_mode()
        elif key == QtCore.Qt.Key_C:
            if self.select_button.isChecked():
                self._label_not_behavior()
            else:
                begin_select_mode()

    def _new_label(self):
        """
        callback for the "new behavior" button
        opens a modal dialog to allow the user to enter a new behavior label
        """
        text, ok = QtWidgets.QInputDialog.getText(None, 'New Label',
                                                  'New Label Name:')
        if ok and text not in self._behaviors:
            self._behaviors.append(text)
            self.behavior_selection.addItem(text)

    def _change_behavior(self):
        """
        make UI changes to reflect the currently selected behavior
        """
        self.label_behavior_button.setText(
            self.behavior_selection.currentText())
        self.label_not_behavior_button.setText(
            f"Not {self.behavior_selection.currentText()}")
        self._set_label_track()

    def _start_selection(self, pressed):
        """
        handle click on "select" button. If button was previously in "unchecked"
        state, then grab the current frame to begin selecting a range. If the
        button was in the checked state, clicking cancels the current selection.

        While selection is in progress, the labeling buttons become active.
        """
        if pressed:
            self.label_behavior_button.setEnabled(True)
            self.label_not_behavior_button.setEnabled(True)
            self.clear_label_button.setEnabled(True)
            self._selection_start = self._player_widget.current_frame()
            self.manual_labels.start_selection(self._selection_start)
        else:
            self.label_behavior_button.setEnabled(False)
            self.label_not_behavior_button.setEnabled(False)
            self.clear_label_button.setEnabled(False)
            self.manual_labels.clear_selection()
        self.manual_labels.update()

    def _label_behavior(self):
        """ Apply behavior label to currently selected range of frames """
        label_range = sorted([self._selection_start,
                              self._player_widget.current_frame()])
        self._get_label_track().label_behavior(*label_range)
        self._disable_label_buttons()
        self.manual_labels.clear_selection()
        self.manual_labels.update()
        self.timeline_widget.update_labels()

    def _label_not_behavior(self):
        """ apply _not_ behavior label to currently selected range of frames """
        label_range = sorted([self._selection_start,
                              self._player_widget.current_frame()])
        self._get_label_track().label_not_behavior(*label_range)
        self._disable_label_buttons()
        self.manual_labels.clear_selection()
        self.manual_labels.update()
        self.timeline_widget.update_labels()

    def _clear_behavior_label(self):
        """ clear all behavior/not behavior labels from current selection """
        label_range = sorted([self._selection_start,
                              self._player_widget.current_frame()])
        self._get_label_track().clear_labels(*label_range)
        self._disable_label_buttons()
        self.manual_labels.clear_selection()
        self.manual_labels.update()
        self.timeline_widget.update_labels()

    def _set_identities(self, identities):
        """ populate the identity_selection combobox """
        self.identity_selection.clear()
        self.identity_selection.addItems([str(i) for i in identities])
        self._player_widget.set_identity_labels(identities)

    def _change_identity(self):
        """ handle changing value of identity_selection """
        self._player_widget._set_active_identity(
            self.identity_selection.currentIndex())
        self._set_label_track()

    def _disable_label_buttons(self):
        """ disable labeling buttons that require a selected range of frames """
        self.label_behavior_button.setEnabled(False)
        self.label_not_behavior_button.setEnabled(False)
        self.clear_label_button.setEnabled(False)
        self.select_button.setChecked(False)

    def _frame_change(self, new_frame):
        """
        called when the video player widget emits its updateFrameNumber signal
        """
        self.manual_labels.set_current_frame(new_frame)
        self.timeline_widget.set_current_frame(new_frame)
        self.frame_ticks.set_current_frame(new_frame)

    def _set_label_track(self):
        """
        loads new set of labels in self.manual_labels when the selected
        behavior or identity is changed
        """
        behavior = self.behavior_selection.currentText()
        identity = self.identity_selection.currentText()

        if identity != '' and behavior != '' and self._labels is not None:
            labels = self._labels.get_track_labels(identity, behavior)
            self.manual_labels.set_labels(labels)
            self.timeline_widget.set_labels(labels)

    def _get_label_track(self):
        """
        get the current label track for the currently selected identity and
        behavior
        """
        return self._labels.get_track_labels(
            self.identity_selection.currentText(),
            self.behavior_selection.currentText()
        )

    @QtCore.pyqtSlot(bool)
    def _identity_popup_visibility_changed(self, visible):
        """
        connected to the IdentityComboBox.pop_up_visible signal. When
        visible == True, we tell the player widget to overlay all identity
        labels on the frame.
        When visible == False we revert to the normal behavior of only labeling
        the currently selected identity
        """
        self._player_widget.set_identity_label_mode(visible)

    def _get_labeled_features(self):
        """
        get all the labeled data for the current behavior
        :return:
        """

        behavior = self.behavior_selection.currentText()

        all_per_frame = []
        all_window = []
        all_labels = []
        all_group_labels = []

        group_id = 0
        for video in self._project.videos:

            pose_est = self._project.load_pose_est(
                self._project.video_path(video))

            for identity in pose_est.identities:
                features = IdentityFeatures(video, identity,
                                            self._project.feature_dir,
                                            pose_est)

                if self._project.video_path(video) == self._loaded_video:
                    labels = self._labels.get_track_labels(
                        str(identity), behavior).get_labels()
                else:
                    labels = self._project.load_annotation_track(
                        video).get_track_labels(str(identity), behavior).get_labels()

                per_frame_features = features.get_per_frame(labels)
                # TODO make window size configurable
                window_features = features.get_window_features(5, labels)

                all_per_frame.append(per_frame_features)
                all_window.append(window_features)
                all_labels.append(labels[labels != TrackLabels.Label.NONE])

                # should be a better way to do this, but I'm getting the number
                # of frames in this group by looking at the shape of one of
                # the arrays included in the window_features
                all_group_labels.append(np.full(window_features['percent_frames_present'].shape[0], group_id))
                group_id += 1

        return {
            'window': IdentityFeatures.merge_window_features(all_window),
            'per_frame': IdentityFeatures.merge_per_frame_features(all_per_frame),
            'labels': np.concatenate(all_labels),
            'groups': np.concatenate(all_group_labels)
        }

    def _train_button_clicked(self):
        features = self._get_labeled_features()
        data = self._classifier.train_test_split(features['per_frame'], features['window'], features['labels'])
        self._classifier.train(data)
        predictions = self._classifier.predict(np.concatenate(data['test_data'], axis=1))

        correct = 0
        for p, truth in zip(predictions, data['test_labels']):
            if p == truth:
                correct += 1
        print(f"accuracy: {correct / len(predictions) * 100:.2f}%")

        self._classifier.print_feature_importance(data['feature_list'])




