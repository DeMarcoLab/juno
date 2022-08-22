import os
import sys

import juno
import juno.ui.qtdesigner_files.SimulationRun as SimulationRun
from PyQt5 import QtWidgets
from juno.Simulation import Simulation
from juno.SimulationRunner import SimulationRunner


class GUISimulationRun(SimulationRun.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self, parent_gui=None):
        super().__init__(parent=parent_gui)
        self.setupUi(MainWindow=self)
        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)
        self.setWindowTitle("Simulation Runner")

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

            self.label_config_info.setText(f"Generated {self.n_sims} simulation configurations ({self.sim_runner.petname}). Ready to run.")
            self.pushButton_run_simulation.setVisible(True)
            self.label_config_info.setVisible(True)

            self.statusBar.showMessage(f"Simulation config loaded from {os.path.basename(sim_config_filename)}")

    def run_simulations(self):
        print("run_simulations...")
        self.statusBar.clearMessage()

        self.label_running_info.setVisible(True)
        self.progressBar_running.setVisible(True)
                
        for i, sim_config in enumerate(self.sim_runner.simulation_configurations, 1):
            
            sim = Simulation(sim_config)
            self.label_running_info.setText(f"Running Simulation: {sim.petname}  ({i}/{self.n_sims})")
            self.progressBar_running.setValue(int(i / self.n_sims) * 100)
            sim.run_simulation()
        
        self.label_running_info.setText(f"")
        self.label_final_info.setText(f"Results were saved to: {self.sim_runner.data_path}.")
        self.statusBar.showMessage(f"Finished running {self.n_sims} simulations.")
        self.progressBar_running.setValue(100)
        self.sim_runner.finish_simulations()


def main():
    """Launch the main application window. """
    application = QtWidgets.QApplication([])
    window = GUISimulationRun()
    application.aboutToQuit.connect(window.disconnect)  # cleanup & teardown
    sys.exit(application.exec_())


if __name__ == "__main__":
    main()
