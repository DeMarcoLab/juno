
import sys
import traceback

from enum import Enum, auto
import glob
from importlib_metadata import metadata
import lens_simulation
import os
from nbformat import from_dict
from numpy import int64

import pandas as pd

from lens_simulation import utils

import lens_simulation.UI.qtdesigner_files.VisualiseResults as VisualiseResults
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QGroupBox, QGridLayout, QLabel, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap, QMovie
import numpy as np



class MODIFIER(Enum):
    EQUAL_TO = auto()
    LESS_THAN = auto()
    GREATER_THAN = auto()
    CONTAINS = auto()



class GUIVisualiseResults(VisualiseResults.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self, parent_gui=None):
        super().__init__(parent=parent_gui)
        self.setupUi(MainWindow=self)
        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)
        self.setWindowTitle("Simulation Results")

        self.df = None

        self.setup_connections()

        self.showNormal()
    
    def setup_connections(self):

        self.pushButton_load_simulation.clicked.connect(self.load_simulation)
        # self.doubleSpinBox.valueChanged.connect(self.filter_dataframe)

        self.comboBox_column.currentIndexChanged.connect(self.set_column_value)
        # self.comboBox_value.currentIndexChanged.connect(self.filter_dataframe)
        self.pushButton_filter_data.clicked.connect(self.filter_dataframe_line)
        self.pushButton_reset_data.clicked.connect(self.load_dataframe)

        self.comboBox_modifier.addItems([modifier.name for modifier in MODIFIER])

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
        self.directory = directory

        self.load_dataframe()
        
    def load_dataframe(self):
        
        # load simulation data
        self.df = utils.load_run_simulation_data(self.directory)

        sim_paths = [os.path.join(self.directory, path) for path in self.df["petname"].unique()]
        stages = list(self.df["stage"].unique())

        self.comboBox_column.addItems(list(self.df.columns))

        self.update_simulation_display(sim_paths, stages)

    def filter_dataframe_line(self):
        
        # TODO: more programmatic
        # TODO: more filters
        # TODO: restrict modifiers based on type?

        MODIFIER = self.comboBox_modifier.currentText()
        TYPE = self.df.dtypes[self.filter_col]

        try:
                
            if TYPE == np.int64:
                value = int(self.lineEdit_value.text())
            if TYPE == np.float64:
                value = float(self.lineEdit_value.text())
            if TYPE == object:
                value = str(self.lineEdit_value.text())

            df_filter = filter_dataframe_by_modifier(self.df, self.filter_col, value, MODIFIER)

        except Exception as e:
            value = "an error occured."
            df_filter = self.df
            # TODO: show error?
            print(e)

        print("filter dataframe")
        print(f"TYPE: {TYPE}, MODIFIER: {MODIFIER}, VALUE: {value}")

        sim_paths = [os.path.join(self.directory, path) for path in df_filter["petname"].unique()]
        stages = list(df_filter["stage"].unique())

        self.update_simulation_display(sim_paths, stages)

    def set_column_value(self):

        # need to cast to correct types...
        if self.directory is None:
            return
        
        self.load_dataframe()

        self.filter_col = self.comboBox_column.currentText()

        # TODO: change the modifiers available here? 

        TYPE = self.df.dtypes[self.filter_col]


        if TYPE in [np.float64, np.int64]:
            MODIFIERS = [mod.name for mod in [MODIFIER.EQUAL_TO, MODIFIER.GREATER_THAN, MODIFIER.LESS_THAN]]

        if TYPE in [object]:
            MODIFIERS = [mod.name for mod in [MODIFIER.EQUAL_TO, MODIFIER.CONTAINS]]
        
        print(TYPE, MODIFIERS)
        self.comboBox_modifier.clear()
        self.comboBox_modifier.addItems(MODIFIERS)



        min_val, max_val = self.df[self.filter_col].min(), self.df[self.filter_col].max()
        self.label_min_max.setText(f"min: {min_val}, max: {max_val}")


    def update_simulation_display(self, sim_paths: list, stages: list):
        # TODO: add option to show beams by adding stage 0 to stages... 
        runGridLayout = draw_run_layout(sim_paths, stages)
        runBox = QGroupBox(f"")
        runBox.setLayout(runGridLayout)

        self.scroll_area.setWidget(runBox)
        self.scroll_area.update()


def filter_dataframe_by_modifier(df, filter_col, value, modifier):

    if modifier == MODIFIER.EQUAL_TO.name:
        df_filter = df[df[filter_col] == value]
    if modifier == MODIFIER.LESS_THAN.name:
        df_filter = df[df[filter_col] < value]
    if modifier == MODIFIER.GREATER_THAN.name:
        df_filter = df[df[filter_col] > value]
    if modifier == MODIFIER.CONTAINS.name:
        df_filter = df[df[filter_col].isin([value])]
    
    return df_filter  

def draw_image_on_label(fname: str, shape: tuple = (300, 300)) -> QLabel:
    """Load a image / gif from file, and set it on a label"""
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
        

def draw_sim_grid_layout(path, stages):
    simGridLayout = QVBoxLayout()
    label_sim_title = QLabel()
    label_sim_title.setStyleSheet("font-size: 14px; font-weight: bold")
    label_sim_title.setText(os.path.basename(path))
    simGridLayout.addWidget(label_sim_title)

    # loop through each stage, load images
    for stage_no in stages:
        sim_directory = os.path.join(path, str(stage_no))

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


def draw_run_layout(sim_directories, stages):
    runGridLayout = QVBoxLayout()    
    
    for sim_path in sim_directories:

        simGridLayout = draw_sim_grid_layout(sim_path, stages)
        
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
