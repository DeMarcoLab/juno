import sys

import juno.ui.qtdesigner_files.LensSimulation as LensSimulation
from juno.ui.BeamCreator import GUIBeamCreator
from juno.ui.SimulationRun import GUISimulationRun
from juno.ui.SimulationSetup import GUISimulationSetup
from juno.ui.VisualiseResults import GUIVisualiseResults
from juno.ui.ElementCreation import GUIElementCreation
from PyQt5 import QtWidgets
import napari

from napari import Viewer

class GUILensSimulation(LensSimulation.Ui_MainWindow, QtWidgets.QMainWindow):
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

        self.pushButton_create_lens.clicked.connect(self.launch_lens_creation)
        self.pushButton_create_beam.clicked.connect(self.launch_beam_creation)
        self.pushButton_setup_sim.clicked.connect(self.launch_setup_simulation)
        self.pushButton_run_simulation.clicked.connect(self.launch_run_simulation)
        self.pushButton_view_results.clicked.connect(self.launch_view_results)
       
    # TODO: properly remove from dock on close?

    def launch_lens_creation(self):

        print("launch lens creation")
        self.remove_current_docked_widgets()
        self.element_creation_ui = GUIElementCreation(viewer=self.viewer)
        dock_widget = self.viewer.window.add_dock_widget(self.element_creation_ui, area='right')                  
        self.dock_widgets.append(dock_widget)

    def launch_beam_creation(self):

        print("launch beam creation")
        self.beam_creator = GUIBeamCreator()

        # TODO:


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
        # TODO:
        print("launch view results")
        self.view_results = GUIVisualiseResults()

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
    sim_launcher_ui = GUILensSimulation(viewer=viewer)
    viewer.window.add_dock_widget(sim_launcher_ui, area='right')                  
    application.aboutToQuit.connect(sim_launcher_ui.disconnect)  # cleanup & teardown
    sys.exit(application.exec_())


if __name__ == "__main__":
    main()
