from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal


class KFoldSliderWidget(QtWidgets.QWidget):
    """
    widget to allow user to select k parameter for k-fold cross validation

    basically consists of a QSlider and three QLabel widgets with
    no spacing/margins
    """

    valueChanged = pyqtSignal(int)

    def __init__(self, kmax=10, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._slider = QtWidgets.QSlider(Qt.Horizontal)
        self._slider.setMinimum(1)
        self._slider.setMaximum(kmax)
        self._slider.setTickInterval(1)
        self._slider.setValue(1)
        self._slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self._slider.valueChanged.connect(self.valueChanged)

        # slider range labels
        label_min = QtWidgets.QLabel("1", alignment=Qt.AlignLeft)
        label_max = QtWidgets.QLabel(f"{kmax}", alignment=Qt.AlignRight)

        slider_vbox = QtWidgets.QVBoxLayout()
        slider_hbox = QtWidgets.QHBoxLayout()
        slider_hbox.setContentsMargins(0, 0, 0, 0)
        slider_vbox.setContentsMargins(0, 0, 0, 0)
        slider_vbox.setSpacing(0)
        slider_vbox.addWidget(QtWidgets.QLabel("Cross Validation k:"))
        slider_vbox.addWidget(self._slider)
        slider_vbox.addLayout(slider_hbox)
        slider_hbox.addWidget(label_min, Qt.AlignLeft)
        slider_hbox.addWidget(label_max, Qt.AlignRight)

        self.setLayout(slider_vbox)

    def value(self):
        """ return the slider value """
        return self._slider.value()
