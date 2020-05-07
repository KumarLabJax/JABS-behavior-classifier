from PyQt5 import QtWidgets, QtCore


class IdentityComboBox(QtWidgets.QComboBox):
    """
    subclass the combo box to emit a signal that indicates if it has been
    opened or closed
    """

    pop_up_visible = QtCore.pyqtSignal(bool)

    def __init__(self, parent=None):
        super(IdentityComboBox, self).__init__(parent=parent)

        # these two properties are related to a work around for a bug described
        # in showPopup
        self._need_to_emit = False
        self._signal_handler_connected = False

    def showPopup(self):
        self.pop_up_visible.emit(True)
        self._need_to_emit = True

        # this is a work-around for a bug that causes hidePopup to not get
        # called if user clicks outside of the pop up to dismiss it without
        # making a selection (observed on Mac OS).
        # this sets up a connection to the QComboBoxPrivateContainer resetButton
        # signal
        if not self._signal_handler_connected:
            self.findChild(QtWidgets.QFrame).resetButton.connect(self.cancel_popup)
            self._signal_handler_connected = True

        super(IdentityComboBox, self).showPopup()

    # the following is commented out because it is unnecessary due to the work-
    # around described in showPopup. The code has been left in place because
    # if the bug is fixed that requires the work-around, then we can swithc
    # back
    #
    # def hidePopup(self):
    #     self.pop_up_visible.emit(False)
    #     super(IdentityComboBox, self).hidePopup()

    def cancel_popup(self):
        if self._need_to_emit:
            self._need_to_emit = False
            self.pop_up_visible.emit(False)
