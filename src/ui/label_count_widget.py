from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt


class FrameLabelCountWidget(QtWidgets.QWidget):
    """
    widget to show the number of frames and bouts for behavior, not behavior
    label classes for the currently selected identity/video as well as
    project-wide totals
    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # dict of QLabels to display frame counts for behavior/not behavior
        self._frame_labels = {
            'behavior_current': QtWidgets.QLabel("0"),
            'behavior_project': QtWidgets.QLabel("0"),
            'not_behavior_current': QtWidgets.QLabel("0"),
            'not_behavior_project': QtWidgets.QLabel("0"),
        }

        # dict of QLabels to display bout counts for behavior/not behavior
        self._bout_labels = {
            'behavior_current': QtWidgets.QLabel("0"),
            'behavior_project': QtWidgets.QLabel("0"),
            'not_behavior_current': QtWidgets.QLabel("0"),
            'not_behavior_project': QtWidgets.QLabel("0"),
        }

        font = QtGui.QFont("Courier New", 12)

        for l in self._frame_labels.values():
            l.setFont(font)

        for l in self._bout_labels.values():
            l.setFont(font)

        frame_header = QtWidgets.QLabel("Frames")
        bout_header = QtWidgets.QLabel("Bouts")

        layout = QtWidgets.QGridLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)

        # add static labels to grid
        layout.addWidget(frame_header, 0, 0, 1, 3,
                         alignment=Qt.AlignCenter)
        layout.addWidget(QtWidgets.QLabel("Current Ident."), 1, 1)
        layout.addWidget(QtWidgets.QLabel("Proj. Total"), 1, 2)
        layout.addWidget(QtWidgets.QLabel("Behavior"), 2, 0)
        layout.addWidget(QtWidgets.QLabel("Not Behavior"), 3, 0)
        layout.addWidget(bout_header, 4, 0, 1, 3,
                         alignment=Qt.AlignCenter)
        layout.addWidget(QtWidgets.QLabel("Behavior"), 5, 0)
        layout.addWidget(QtWidgets.QLabel("Not Behavior"), 6, 0)

        # add labels containing counts to grid
        layout.addWidget(self._frame_labels['behavior_current'], 2, 1,
                         alignment=Qt.AlignRight)
        layout.addWidget(self._frame_labels['behavior_project'], 2, 2,
                         alignment=Qt.AlignRight)
        layout.addWidget(self._frame_labels['not_behavior_current'], 3, 1,
                         alignment=Qt.AlignRight)
        layout.addWidget(self._frame_labels['not_behavior_project'], 3, 2,
                         alignment=Qt.AlignRight)
        layout.addWidget(self._bout_labels['behavior_current'], 5, 1,
                         alignment=Qt.AlignRight)
        layout.addWidget(self._bout_labels['behavior_project'], 5, 2,
                         alignment=Qt.AlignRight)
        layout.addWidget(self._bout_labels['not_behavior_current'], 6, 1,
                         alignment=Qt.AlignRight)
        layout.addWidget(self._bout_labels['not_behavior_project'], 6, 2,
                         alignment=Qt.AlignRight)

        self.setLayout(layout)

    def set_counts(self, frame_behavior_current, frame_not_behavior_current,
                   frame_behavior_project, frame_not_behavior_project,
                   bout_behavior_current, bout_not_behavior_current,
                   bout_behavior_project, bout_not_behavior_project):
        """
        update counts and redraw widget

        :param frame_behavior_current: #frames labeled behavior for current
        identity (in current video)
        :param frame_not_behavior_current: #frames labeled not behavior for
        current identity (in current video)
        :param frame_behavior_project:  #frames labeled behavior for project
        :param frame_not_behavior_project: #frames labeled not behavior for
        project
        :param bout_behavior_current: #bouts of behavior for current identity
        (in current video)
        :param bout_not_behavior_current: #bouts not behavior for current
        identity (in current video)
        :param bout_behavior_project: #bouts behavior for project
        :param bout_not_behavior_project: #bouts not behavior for project
        :return:
        """
        self._frame_labels['behavior_current'].setText(
            f"{frame_behavior_current}")
        self._frame_labels['not_behavior_current'].setText(
            f"{frame_not_behavior_current}")
        self._frame_labels['behavior_project'].setText(
            f"{frame_behavior_project}")
        self._frame_labels['not_behavior_project'].setText(
            f"{frame_not_behavior_project}")

        self._bout_labels['behavior_current'].setText(
            f"{bout_behavior_current}")
        self._bout_labels['not_behavior_current'].setText(
            f"{bout_not_behavior_current}")
        self._bout_labels['behavior_project'].setText(
            f"{bout_behavior_project}")
        self._bout_labels['not_behavior_project'].setText(
            f"{bout_not_behavior_project}")

        self.update()
