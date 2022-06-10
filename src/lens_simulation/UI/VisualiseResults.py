import sys
import traceback

import glob
import lens_simulation
import os

from lens_simulation import utils

import lens_simulation.UI.qtdesigner_files.VisualiseResults as VisualiseResults
from matplotlib.pyplot import grid
import numpy as np
from lens_simulation.Lens import GratingSettings, Lens, LensType
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QGroupBox, QGridLayout, QLabel, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap, QMovie



class GUIVisualiseResults(VisualiseResults.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self, parent_gui=None):
        super().__init__(parent=parent_gui)
        self.setupUi(MainWindow=self)
        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)
        self.setWindowTitle("Simulation Results")

        self.setup_connections()

        self.showNormal()
    
    def setup_connections(self):

        self.pushButton_load_simulation.clicked.connect(self.load_simulation)
        self.comboBox_select_simulation.currentIndexChanged.connect(self.show_simulation)


    def load_simulation(self):

        # select directory
        log_dir = os.path.join(os.path.dirname(lens_simulation.__file__), "log")

        directory = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Load Simulation Run", log_dir))

        if directory == "":
            return
        
        sim_run_name = os.path.basename(directory)
        self.sim_directories = [os.path.join(directory, path) for path in os.listdir(directory) if os.path.isdir(os.path.join(directory, path))] 

        sim_names = [os.path.basename(sim) for sim in self.sim_directories]
        self.directory = directory

        # set ui
        self.label_sim_run_name.setText(sim_run_name)
        self.comboBox_select_simulation.addItems(sim_names)


    def show_simulation(self):

        sim_name = self.comboBox_select_simulation.currentText()
        self.sim_directory = os.path.join(self.directory, sim_name)
        
        metadata = utils.load_metadata(self.sim_directory)
        
        stage_dir = [path for path in os.listdir(self.sim_directory) if os.path.isdir(os.path.join(self.sim_directory, path))] 

        # show plots
        self.show_simulation_plots(stage_dir)
        
        
    def show_simulation_plots(self, stage_dir):
        

        def generate_stage_grid_layout(stage_no, fnames) -> QGridLayout:
            stage_grid_layout = QGridLayout()

            # stage title
            stage_label = QLabel()
            stage_label.setText(f"Stage {stage_no}")
            stage_grid_layout.addWidget(stage_label, 0, 0)

            def show_image_on_label(fname, label, shape=(300, 300)):
                
                if "gif" in fname:
                    movie = QMovie(fname)
                    movie.setScaledSize(QtCore.QSize(shape[0], shape[1])) # scaled contents is the problem...
                    label.setScaledContents(True)
                    label.setMovie(movie)
                    if movie is not None:
                        movie.start()

                else:
                    label.setPixmap(QPixmap(fname).scaled(*shape))

                label.setStyleSheet("border-radius: 5px")
                
                return label

            for i, fname in enumerate(fnames):
                label = QLabel()
                
                label = show_image_on_label(fname, label)

                stage_grid_layout.addWidget(label, 1, i)

            return stage_grid_layout
        

        runGridLayout = QVBoxLayout()

        for sim_path in self.sim_directories:

            simGridLayout = QVBoxLayout()
            label_sim_title = QLabel()
            label_sim_title.setText(os.path.basename(sim_path))
            simGridLayout.addWidget(label_sim_title)
            
            for stage_no in stage_dir:
                sim_directory = os.path.join(sim_path, stage_no)

                profile_fname = os.path.join(sim_directory, "lens_profile.png")
                slices_fname = os.path.join(sim_directory, "lens_slices.png")
                topdown_fname = os.path.join(sim_directory, "topdown.png")
                sideon_fname = os.path.join(sim_directory, "sideon.png")
                propagation_fname = os.path.join(sim_directory, "propagation.gif")
                fnames = [profile_fname, slices_fname, topdown_fname, sideon_fname, propagation_fname]

                # TODO: gifs: https://pythonpyqt.com/pyqt-gif/
                stage_grid_layout = generate_stage_grid_layout(stage_no, fnames)

                horizontalGroupBox = QGroupBox(f"")
                horizontalGroupBox.setLayout(stage_grid_layout)
                simGridLayout.addWidget(horizontalGroupBox)

            simBox = QGroupBox(f"")
            simBox.setLayout(simGridLayout)
            runGridLayout.addWidget(simBox)
                
        runBox = QGroupBox(f"")
        runBox.setLayout(runGridLayout)
        self.scroll_area.setWidget(runBox)
        horizontalGroupBox.update()
        self.scroll_area.update()

        # run
        # sim
        # stage 
        # each needs a layout?
        print("done loading images")






def main():
    """Launch the main application window. """
    application = QtWidgets.QApplication([])
    window = GUIVisualiseResults()
    application.aboutToQuit.connect(window.disconnect)  # cleanup & teardown
    sys.exit(application.exec_())


if __name__ == "__main__":
    main()
