import sys

import star_glass.ui.qtdesigner_files.LensSimulation as LensSimulation
from star_glass.ui.BeamCreator import GUIBeamCreator
from star_glass.ui.LensCreator import GUILensCreator
from star_glass.ui.SimulationRun import GUISimulationRun
from star_glass.ui.SimulationSetup import GUISimulationSetup
from star_glass.ui.VisualiseResults import GUIVisualiseResults
from PyQt5 import QtWidgets


class GUILensSimulation(LensSimulation.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self, parent_gui=None):
        super().__init__(parent=parent_gui)
        self.setupUi(MainWindow=self)
        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)
        self.setWindowTitle("Lens Simulation Launcher")

        self.setup_connections()

        self.showNormal()

    def setup_connections(self):      

        print("setup connections")

        self.pushButton_create_lens.clicked.connect(self.launch_lens_creation)
        self.pushButton_create_beam.clicked.connect(self.launch_beam_creation)
        self.pushButton_setup_sim.clicked.connect(self.launch_setup_simulation)
        self.pushButton_run_simulation.clicked.connect(self.launch_run_simulation)
        self.pushButton_view_results.clicked.connect(self.launch_view_results)
       

    def launch_lens_creation(self):

        print("launch lens creation")

        self.lens_creator = GUILensCreator()

    def launch_beam_creation(self):

        print("launch beam creation")
        self.beam_creator = GUIBeamCreator()


    def launch_setup_simulation(self):

        print("launch setup_simulation")

        self.simulation_setup = GUISimulationSetup()

    def launch_run_simulation(self):

        print("launch run_simulation")

        self.simulation_run = GUISimulationRun()

    def launch_view_results(self):

        print("launch view results")
        self.view_results = GUIVisualiseResults()

def main():
    """Launch the main application window. """
    application = QtWidgets.QApplication([])
    window = GUILensSimulation()
    application.aboutToQuit.connect(window.disconnect)  # cleanup & teardown
    sys.exit(application.exec_())


if __name__ == "__main__":
    main()