from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal


class KFoldSliderWidget(QtWidgets.QWidget):
    """
    widget to show the number of frames and bouts for behavior, not behavior
    label classes for the currently selected identity/video as well as
    project-wide totals
    """

    valueChanged = pyqtSignal(int)

    def __init__(self, kmax=10, *args, **kwargs):
        super(KFoldSliderWidget, self).__init__(*args, **kwargs)

        self._slider = QtWidgets.QSlider(Qt.Horizontal)
        self._slider.setMinimum(1)
        self._slider.setMaximum(kmax)
        self._slider.setTickInterval(1)
        self._slider.setValue(1)
        self._slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self._slider.valueChanged.connect(self.valueChanged)

        label_min = QtWidgets.QLabel("1", alignment=Qt.AlignLeft)
        label_max = QtWidgets.QLabel(f"{kmax}", alignment=Qt.AlignRight)

        slider_vbox = QtWidgets.QVBoxLayout()
        slider_hbox = QtWidgets.QHBoxLayout()
        slider_hbox.setContentsMargins(0, 0, 0, 0)
        slider_vbox.setContentsMargins(0, 0, 0, 0)
        slider_vbox.setSpacing(0)
        slider_vbox.addWidget(QtWidgets.QLabel("Cross Validation K:"))
        slider_vbox.addWidget(self._slider)
        slider_vbox.addLayout(slider_hbox)
        slider_hbox.addWidget(label_min, Qt.AlignLeft)
        slider_hbox.addWidget(label_max, Qt.AlignRight)

        self.setLayout(slider_vbox)

    def value(self):
        return self._slider.value()
