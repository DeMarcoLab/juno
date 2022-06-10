from curses import meta
import sys
import traceback

import glob
from importlib_metadata import metadata
import lens_simulation
import os
from nbformat import from_dict

import pandas as pd

from lens_simulation import utils

import lens_simulation.UI.qtdesigner_files.VisualiseResults as VisualiseResults
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


    def load_simulation(self):

        # select directory
        log_dir = os.path.join(os.path.dirname(lens_simulation.__file__), "log")

        directory = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Load Simulation Run", log_dir))

        if directory == "":
            return
        
        sim_run_name = os.path.basename(directory)
        self.sim_directories = [os.path.join(directory, path) for path in os.listdir(directory) if os.path.isdir(os.path.join(directory, path))] 

        # set ui
        self.label_sim_run_name.setText(f"Run: {sim_run_name}")
        self.label_sim_no_loaded.setText(f"{len(self.sim_directories)} simulation results loaded.")
        self.show_simulation_plots()
     
        
    def show_simulation_plots(self):
        
        runGridLayout = draw_run_layout(self.sim_directories)
        runBox = QGroupBox(f"")
        runBox.setLayout(runGridLayout)

        self.scroll_area.setWidget(runBox)
        self.scroll_area.update()

        for path in self.sim_directories:

            utils.load_simulation_data(path)

        print("done loading images")









def draw_image_on_label(fname, shape=(300, 300)):
    label = QLabel()
    if "gif" in fname:
        movie = QMovie(fname)
        movie.setScaledSize(QtCore.QSize(shape[0], shape[1])) 
        label.setScaledContents(True)
        label.setMovie(movie)
        if movie is not None:
            movie.start()

    else:
        label.setPixmap(QPixmap(fname).scaled(*shape))

    label.setStyleSheet("border-radius: 5px")
    
    return label

def generate_stage_grid_layout(stage_no, fnames) -> QGridLayout:
    stage_grid_layout = QGridLayout()

    # stage title
    stage_label = QLabel()
    stage_label.setText(f"Stage {stage_no}")
    stage_grid_layout.addWidget(stage_label, 0, 0)

    for i, fname in enumerate(fnames):

        label = draw_image_on_label(fname)

        stage_grid_layout.addWidget(label, 1, i)

    return stage_grid_layout
        

def draw_sim_grid_layout(sim_path):
    simGridLayout = QVBoxLayout()
    label_sim_title = QLabel()
    label_sim_title.setStyleSheet("font-size: 14px; font-weight: bold")
    label_sim_title.setText(os.path.basename(sim_path))
    simGridLayout.addWidget(label_sim_title)
        
    stage_dir = [path for path in os.listdir(sim_path) if os.path.isdir(os.path.join(sim_path, path))] 

    for stage_no in stage_dir:
        sim_directory = os.path.join(sim_path, stage_no)

        profile_fname = os.path.join(sim_directory, "lens_profile.png")
        slices_fname = os.path.join(sim_directory, "lens_slices.png")
        topdown_fname = os.path.join(sim_directory, "topdown.png")
        sideon_fname = os.path.join(sim_directory, "sideon.png")
        propagation_fname = os.path.join(sim_directory, "propagation.gif")
        fnames = [profile_fname, slices_fname, topdown_fname, sideon_fname, propagation_fname]

        stage_grid_layout = generate_stage_grid_layout(stage_no, fnames)

        horizontalGroupBox = QGroupBox(f"")
        horizontalGroupBox.setLayout(stage_grid_layout)
        simGridLayout.addWidget(horizontalGroupBox)
    return simGridLayout


def draw_run_layout(sim_directories):
    runGridLayout = QVBoxLayout()

    for sim_path in sim_directories:

        simGridLayout = draw_sim_grid_layout(sim_path)
        
        simBox = QGroupBox(f"")
        simBox.setLayout(simGridLayout)
        runGridLayout.addWidget(simBox)
        
    return runGridLayout


def main():
    """Launch the main application window. """
    application = QtWidgets.QApplication([])
    window = GUIVisualiseResults()
    application.aboutToQuit.connect(window.disconnect)  # cleanup & teardown
    sys.exit(application.exec_())


if __name__ == "__main__":
    main()
