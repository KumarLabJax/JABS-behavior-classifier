from PyQt5 import QtWidgets, QtCore


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

    def keyPressEvent(self, event):
        """
        override keyPressEvent so we can pass some key press events on
        to the centralWidget
        """
        key = event.key()

        # pass along some of the key press events to the central widget
        if key in [
            QtCore.Qt.Key_Left,
            QtCore.Qt.Key_Right,
            QtCore.Qt.Key_Down,
            QtCore.Qt.Key_Up,
            QtCore.Qt.Key_Space,
            QtCore.Qt.Key_Z,
            QtCore.Qt.Key_X,
            QtCore.Qt.Key_C
        ]:
            self.centralWidget().keyPressEvent(event)

        else:
            # anything else pass on to the super keyPressEvent
            super(MainWindow, self).keyPressEvent(event)
