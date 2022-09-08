import sys

import juno.ui.qtdesigner_files.JunoLauncher as JunoLauncher
from juno.ui.BeamCreation import GUIBeamCreation
from juno.ui.SimulationRun import GUISimulationRun
from juno.ui.SimulationSetup import GUISimulationSetup
from juno.ui.VisualiseResults import GUIVisualiseResults
from juno.ui.ElementCreation import GUIElementCreation
from PyQt5 import QtWidgets, QtGui
import napari

import os
LOGO_FILE = os.path.join(os.path.dirname(__file__), "logo.png")

from napari import Viewer

class GUIJunoLauncher(JunoLauncher.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self, viewer: Viewer = None, parent_gui=None):
        super().__init__(parent=parent_gui)
        self.setupUi(MainWindow=self)
        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)
        self.setWindowTitle("Juno Launcher")

        self.viewer = viewer
        self.dock_widgets = []

        self.setup_connections()

        self.showNormal()

    def setup_connections(self):      

        self.pushButton_create_element.clicked.connect(self.launch_element_creation)
        self.pushButton_create_beam.clicked.connect(self.launch_beam_creation)
        self.pushButton_setup_sim.clicked.connect(self.launch_setup_simulation)
        self.pushButton_run_simulation.clicked.connect(self.launch_run_simulation)
        self.pushButton_view_results.clicked.connect(self.launch_view_results)

        pixmap = QtGui.QPixmap(LOGO_FILE)
        self.label_logo.setPixmap(pixmap) # https://www.geeksforgeeks.org/pyqt5-lower-method-for-labels/ overlay TODO
       
    def launch_element_creation(self):

        print("launch element creation")
        self.remove_current_docked_widgets()
        self.element_creation_ui = GUIElementCreation(viewer=self.viewer)
        dock_widget = self.viewer.window.add_dock_widget(self.element_creation_ui, area='right')                  
        self.dock_widgets.append(dock_widget)

    def launch_beam_creation(self):

        print("launch beam creation")
        self.remove_current_docked_widgets()
        self.beam_creator_ui = GUIBeamCreation(viewer=self.viewer)
        dock_widget = self.viewer.window.add_dock_widget(self.beam_creator_ui, area='right')                  
        self.dock_widgets.append(dock_widget)

    def launch_setup_simulation(self):

        print("launch setup_simulation")

        self.remove_current_docked_widgets()

        self.simulation_setup_ui = GUISimulationSetup(viewer=self.viewer)
        dock_widget = self.viewer.window.add_dock_widget(self.simulation_setup_ui, area='right')
        self.dock_widgets.append(dock_widget)
        

    def launch_run_simulation(self):

        print("launch run_simulation")
        self.remove_current_docked_widgets()

        self.simulation_run_ui = GUISimulationRun(viewer=self.viewer)
        dock_widget = self.viewer.window.add_dock_widget(self.simulation_run_ui, area='right')
        self.dock_widgets.append(dock_widget)


    def launch_view_results(self):

        print("launch view results")
        self.remove_current_docked_widgets()
        self.view_results_ui = GUIVisualiseResults(viewer=self.viewer)
        dock_widget = self.viewer.window.add_dock_widget(self.view_results_ui, area='right')
        self.dock_widgets.append(dock_widget)

    def remove_current_docked_widgets(self):

        for dock_widget in self.dock_widgets:
            try:
                self.viewer.window.remove_dock_widget(dock_widget)
            except Exception as e:
                print(f"Unable to remove docked widget: {e}")

        self.dock_widgets = []
        self.viewer.layers.clear()

def main():
    """Launch the main application window. """
    application = QtWidgets.QApplication([])

    viewer = napari.Viewer(ndisplay=3)
    sim_launcher_ui = GUIJunoLauncher(viewer=viewer)
    viewer.window.add_dock_widget(sim_launcher_ui, area='right')                  
    application.aboutToQuit.connect(sim_launcher_ui.disconnect)  # cleanup & teardown
    sys.exit(application.exec_())


if __name__ == "__main__":
    main()
