from PyQt5 import QtWidgets, QtCore


class IdentityComboBox(QtWidgets.QComboBox):
    """
    Subclass the combo box to emit a signal that indicates if it has been
    opened or closed. This is used to tell the PlayerWidget to switch to the
    "label identities mode".
    """

    pop_up_visible = QtCore.pyqtSignal(bool)

    def __init__(self, parent=None):
        super(IdentityComboBox, self).__init__(parent=parent)

        # these two properties are related to a work around for a bug described
        # in showPopup
        self._need_to_emit = False
        self._signal_handler_connected = False

    def showPopup(self):
        """
        showPopup is overridden so that we can emit a signal every time
        it is shown
        """
        self.pop_up_visible.emit(True)
        super(IdentityComboBox, self).showPopup()

        # Everything else is a work-around for a bug that causes hidePopup to
        # not get called if user clicks outside of the pop up to dismiss it
        # without making a selection.
        # (see https://bugreports.qt.io/browse/QTBUG-50055)
        # This sets up a connection to the QComboBoxPrivateContainer resetButton
        # signal
        self._need_to_emit = True
        if not self._signal_handler_connected:
            self.findChild(QtWidgets.QFrame).resetButton.connect(
                self.cancel_popup)
            self._signal_handler_connected = True

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
