import sys
from juno.ui.LensSimulation import GUILensSimulation

import napari
from PyQt5 import QtWidgets


def main():
    """Launch the main application window. """
    application = QtWidgets.QApplication([])

    viewer = napari.Viewer(ndisplay=3)
    sim_launcher_ui = GUILensSimulation(viewer=viewer)
    viewer.window.add_dock_widget(sim_launcher_ui, area='right')                  
    application.aboutToQuit.connect(sim_launcher_ui.disconnect)  # cleanup & teardown
    sys.exit(application.exec_())


if __name__ == "__main__":
    main()
