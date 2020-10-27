from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt


class FrameLabelCountWidget(QtWidgets.QWidget):

    def __init__(self, *args, **kwargs):

        super(FrameLabelCountWidget, self).__init__(*args, **kwargs)

        # dict of QLabels to display frame counts for behavior/not behavior
        self._frame_labels = {
            'behavior_current': QtWidgets.QLabel("0"),
            'behavior_total': QtWidgets.QLabel("0"),
            'not_behavior_current': QtWidgets.QLabel("0"),
            'not_behavior_total': QtWidgets.QLabel("0"),
        }

        # dict of QLabels to display bout counts for behavior/not behavior
        self._bout_labels = {
            'behavior_current': QtWidgets.QLabel("0"),
            'behavior_total': QtWidgets.QLabel("0"),
            'not_behavior_current': QtWidgets.QLabel("0"),
            'not_behavior_total': QtWidgets.QLabel("0"),
        }

        font = QtGui.QFont("Courier New", 14)

        for l in self._frame_labels.values():
            l.setFont(font)

        for l in self._bout_labels.values():
            l.setFont(font)

        frame_header = QtWidgets.QLabel("Frames Labeled")
        bout_header = QtWidgets.QLabel("Bouts Labeled")
        font = frame_header.font()
        font.setPointSize(16)
        frame_header.setFont(font)
        bout_header.setFont(font)

        layout = QtWidgets.QGridLayout()

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
        layout.addWidget(self._frame_labels['behavior_total'], 2, 2,
                         alignment=Qt.AlignRight)
        layout.addWidget(self._frame_labels['not_behavior_current'], 3, 1,
                         alignment=Qt.AlignRight)
        layout.addWidget(self._frame_labels['not_behavior_total'], 3, 2,
                         alignment=Qt.AlignRight)
        layout.addWidget(self._bout_labels['behavior_current'], 5, 1,
                         alignment=Qt.AlignRight)
        layout.addWidget(self._bout_labels['behavior_total'], 5, 2,
                         alignment=Qt.AlignRight)
        layout.addWidget(self._bout_labels['not_behavior_current'], 6, 1,
                         alignment=Qt.AlignRight)
        layout.addWidget(self._bout_labels['not_behavior_total'], 6, 2,
                         alignment=Qt.AlignRight)

        self.setLayout(layout)
