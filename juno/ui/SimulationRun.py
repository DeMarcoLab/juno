import os
import sys

import juno
import juno.ui.qtdesigner_files.SimulationRun as SimulationRun
from PyQt5 import QtWidgets
from juno.Simulation import Simulation
from juno.SimulationRunner import SimulationRunner

from napari.utils import progress
import napari.utils.notifications


class GUISimulationRun(SimulationRun.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self, viewer=None, parent_gui=None):
        super().__init__(parent=parent_gui)
        self.setupUi(MainWindow=self)
        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)
        self.setWindowTitle("Simulation Runner")

        self.viewer = viewer
        self.setup_connections()

        self.showNormal()

    def setup_connections(self):      

        self.pushButton_load_config.clicked.connect(self.load_config)
        self.pushButton_run_simulation.clicked.connect(self.run_simulations)
        self.pushButton_run_simulation.setVisible(False)
        self.label_config_info.setVisible(False)
        self.label_running_info.setVisible(False)
        self.progressBar_running.setVisible(False)

    def load_config(self):
        print("loading simulation config")

        # open file dialog
        sim_config_filename, _ = QtWidgets.QFileDialog.getOpenFileName(self,
                    caption="Load Simulation Config",
                    directory=os.path.dirname(juno.__file__),
                    filter="Yaml files (*.yml *.yaml)"
                    )
        if sim_config_filename:

            self.sim_runner = SimulationRunner(sim_config_filename)
            self.sim_runner.setup_simulation()
            self.n_sims = len(self.sim_runner.simulation_configurations)

            napari.utils.notifications.show_info(f"""Simulation config loaded from {os.path.basename(sim_config_filename)}. Generated {self.n_sims} simulation configurations ({self.sim_runner.petname}). Ready to run.""")
            self.label_config_info.setText(f"Generated {self.n_sims} simulation configurations ({self.sim_runner.petname}). Ready to run.")
            self.pushButton_run_simulation.setVisible(True)
            self.label_config_info.setVisible(True)


    def run_simulations(self):
        print("run_simulations...")
        self.statusBar.clearMessage()               

        self.viewer.window._status_bar._toggle_activity_dock(True)

        progress_bar = progress(self.sim_runner.simulation_configurations)
        for i, sim_config in enumerate(progress_bar, 1):
            
            sim = Simulation(sim_config)
            progress_bar.set_description(f"Running Simulation: {sim.petname} ({i}/{self.n_sims})")
            sim.run_simulation() 
                    
        self.sim_runner.finish_simulations()
        self.viewer.window._status_bar._toggle_activity_dock(False)

        napari.utils.notifications.show_info(f"Finished running {self.n_sims} simulations. Results were saved to: {self.sim_runner.data_path}.")
        


def main():
    """Launch the main application window. """
    application = QtWidgets.QApplication([])
    import napari
    viewer = napari.Viewer(ndisplay=3)
    sim_run_ui = GUISimulationRun(viewer=viewer)
    viewer.window.add_dock_widget(sim_run_ui, area='right') 
    application.aboutToQuit.connect(sim_run_ui.disconnect)  # cleanup & teardown
    sys.exit(application.exec_())


if __name__ == "__main__":
    main()
