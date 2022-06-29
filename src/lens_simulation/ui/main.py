import sys
from lens_simulation.ui.LensSimulation import GUILensSimulation

from PyQt5 import QtWidgets


def main():
    """Launch the main application window. """
    application = QtWidgets.QApplication([])
    window = GUILensSimulation()
    application.aboutToQuit.connect(window.disconnect)  # cleanup & teardown
    sys.exit(application.exec_())


if __name__ == "__main__":
    main()
