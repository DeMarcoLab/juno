
from PyQt5 import QtWidgets, QtCore

def display_error_message(message, title="Error Message"):
    """PyQt dialog box displaying an error message."""
    # logging.debug('display_error_message')
    # logging.exception(message)
    error_dialog = QtWidgets.QErrorMessage()
    error_dialog.setWindowTitle(title)
    error_dialog.showMessage(message)
    error_dialog.showNormal()
    error_dialog.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
    error_dialog.exec_()