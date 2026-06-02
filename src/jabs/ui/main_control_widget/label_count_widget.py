import sys

from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import Qt


class FrameLabelCountWidget(QtWidgets.QWidget):
    """Widget to display the number of frames and bouts for behavior and not-behavior label classes.

    Shows counts for the currently selected identity/video as well as project-wide totals.

    Args:
        *args: Additional positional arguments for QWidget.
        **kwargs: Additional keyword arguments for QWidget.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # dict of QLabels to display frame counts for behavior/not behavior
        self._frame_labels = {
            "behavior_current": QtWidgets.QLabel("0"),
            "behavior_project": QtWidgets.QLabel("0"),
            "not_behavior_current": QtWidgets.QLabel("0"),
            "not_behavior_project": QtWidgets.QLabel("0"),
        }

        # dict of QLabels to display bout counts for behavior/not behavior
        self._bout_labels = {
            "behavior_current": QtWidgets.QLabel("0"),
            "behavior_project": QtWidgets.QLabel("0"),
            "not_behavior_current": QtWidgets.QLabel("0"),
            "not_behavior_project": QtWidgets.QLabel("0"),
        }

        if sys.platform == "darwin":
            font = QtGui.QFont("Courier New", 12)
        else:
            font = QtGui.QFont("Courier New", 10)

        for label in self._frame_labels.values():
            label.setFont(font)

        for label in self._bout_labels.values():
            label.setFont(font)

        frame_header = QtWidgets.QLabel("Frames")
        bout_header = QtWidgets.QLabel("Bouts")

        # Row-header labels for the positive ("Behavior") and negative
        # ("Not Behavior") classes. Kept as instance attributes so the text can
        # be retitled per classifier mode (e.g. the selected behavior name and
        # "None" in multi-class mode); see set_class_labels().
        self._positive_row_labels = [QtWidgets.QLabel("Behavior"), QtWidgets.QLabel("Behavior")]
        self._negative_row_labels = [
            QtWidgets.QLabel("Not Behavior"),
            QtWidgets.QLabel("Not Behavior"),
        ]

        layout = QtWidgets.QGridLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)

        # add static labels to grid
        layout.addWidget(frame_header, 0, 0, 1, 3, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(QtWidgets.QLabel("Subject"), 1, 1, alignment=Qt.AlignmentFlag.AlignRight)
        layout.addWidget(QtWidgets.QLabel("Total"), 1, 2, alignment=Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self._positive_row_labels[0], 2, 0)
        layout.addWidget(self._negative_row_labels[0], 3, 0)
        layout.addWidget(bout_header, 4, 0, 1, 3, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(QtWidgets.QLabel("Subject"), 5, 1, alignment=Qt.AlignmentFlag.AlignRight)
        layout.addWidget(QtWidgets.QLabel("Total"), 5, 2, alignment=Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self._positive_row_labels[1], 6, 0)
        layout.addWidget(self._negative_row_labels[1], 7, 0)

        # add labels containing counts to grid
        layout.addWidget(
            self._frame_labels["behavior_current"],
            2,
            1,
            alignment=Qt.AlignmentFlag.AlignRight,
        )
        layout.addWidget(
            self._frame_labels["behavior_project"],
            2,
            2,
            alignment=Qt.AlignmentFlag.AlignRight,
        )
        layout.addWidget(
            self._frame_labels["not_behavior_current"],
            3,
            1,
            alignment=Qt.AlignmentFlag.AlignRight,
        )
        layout.addWidget(
            self._frame_labels["not_behavior_project"],
            3,
            2,
            alignment=Qt.AlignmentFlag.AlignRight,
        )
        layout.addWidget(
            self._bout_labels["behavior_current"],
            6,
            1,
            alignment=Qt.AlignmentFlag.AlignRight,
        )
        layout.addWidget(
            self._bout_labels["behavior_project"],
            6,
            2,
            alignment=Qt.AlignmentFlag.AlignRight,
        )
        layout.addWidget(
            self._bout_labels["not_behavior_current"],
            7,
            1,
            alignment=Qt.AlignmentFlag.AlignRight,
        )
        layout.addWidget(
            self._bout_labels["not_behavior_project"],
            7,
            2,
            alignment=Qt.AlignmentFlag.AlignRight,
        )

        self.setLayout(layout)

    def set_class_labels(self, positive_label: str, negative_label: str) -> None:
        """Retitle the positive/negative row headers.

        Binary mode uses "Behavior"/"Not Behavior"; multi-class mode uses the
        selected behavior name and the reserved background class name ("None").

        Args:
            positive_label: text for the behavior (positive) rows.
            negative_label: text for the not-behavior (negative) rows.
        """
        for label in self._positive_row_labels:
            label.setText(positive_label)
        for label in self._negative_row_labels:
            label.setText(negative_label)

    def set_counts(
        self,
        frame_behavior_current,
        frame_not_behavior_current,
        frame_behavior_project,
        frame_not_behavior_project,
        bout_behavior_current,
        bout_not_behavior_current,
        bout_behavior_project,
        bout_not_behavior_project,
    ):
        """update counts and redraw widget

        Args:
            frame_behavior_current:
              #frames labeled behavior for current identity (in current video)
            frame_not_behavior_current:
              #frames labeled not behavior for current identity (in current video)
            frame_behavior_project:
              #frames labeled behavior for project
            frame_not_behavior_project:
              #frames labeled not behavior for project
            bout_behavior_current:
              #bouts of behavior for current identity (in current video)
            bout_not_behavior_current:
              #bouts not behavior for current identity (in current video)
            bout_behavior_project:
              #bouts behavior for project
            bout_not_behavior_project:
              #bouts not behavior for project
        """
        self._frame_labels["behavior_current"].setText(f"{frame_behavior_current}")
        self._frame_labels["not_behavior_current"].setText(f"{frame_not_behavior_current}")
        self._frame_labels["behavior_project"].setText(f"{frame_behavior_project}")
        self._frame_labels["not_behavior_project"].setText(f"{frame_not_behavior_project}")

        self._bout_labels["behavior_current"].setText(f"{bout_behavior_current}")
        self._bout_labels["not_behavior_current"].setText(f"{bout_not_behavior_current}")
        self._bout_labels["behavior_project"].setText(f"{bout_behavior_project}")
        self._bout_labels["not_behavior_project"].setText(f"{bout_not_behavior_project}")

        self.update()
