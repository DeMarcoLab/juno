import sys
import threading
from turtle import home
from lens_simulation.UI import LensCreator
from PyQt5 import QtCore, QtGui, QtWidgets

def main():
    """Launch the `piescope_gui` main application window."""
    application = QtWidgets.QApplication([])
    window = LensCreator.GUILensCreator()
    application.aboutToQuit.connect(window.disconnect)
    sys.exit(application.exec_())


if __name__ == "__main__":
    main()

