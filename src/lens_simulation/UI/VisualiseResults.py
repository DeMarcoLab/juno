
import sys
import traceback

from enum import Enum, auto
import glob
from typing import Union
import lens_simulation
import os

from lens_simulation import utils
from lens_simulation.UI.utils import display_error_message
import lens_simulation.UI.qtdesigner_files.VisualiseResults as VisualiseResults
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QGroupBox, QGridLayout, QLabel, QVBoxLayout, QPushButton, QLineEdit, QComboBox
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

        self.SIMULATION_LOADED = False
        self.df = None

        self.setup_connections()

        self.showNormal()

    def setup_connections(self):

        # load data
        self.pushButton_load_simulation.clicked.connect(self.load_simulation)

        # filter data
        self.label_filter_title.setVisible(False)
        self.frame_filter.setVisible(False)
        self.pushButton_filter_data.clicked.connect(self.filter_by_each_filter)
        self.pushButton_reset_data.clicked.connect(self.load_dataframe)
        self.spinBox_num_filters.valueChanged.connect(self.update_filter_display)


    def load_simulation(self):
        try:
            # select directory
            log_dir = os.path.join(os.path.dirname(lens_simulation.__file__), "log")

            directory = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Load Simulation Run", log_dir))

            if directory == "":
                return

            sim_run_name = os.path.basename(directory)

            self.sim_directories = [os.path.join(directory, path) for path in os.listdir(directory) if os.path.isdir(os.path.join(directory, path))]

            # set ui
            self.label_sim_run_name.setText(f"Run: {sim_run_name}")
            self.label_sim_no_loaded.setText(f"{len(self.sim_directories)} simulations loaded.")
            self.directory = directory

            self.load_dataframe()

            self.label_filter_title.setVisible(True)
            self.frame_filter.setVisible(True)

            self.SIMULATION_LOADED = True
        except Exception as e:
            display_error_message(f'Error loading simulation folder: {e}')

    def load_dataframe(self):

        # load simulation data
        self.df = utils.load_run_simulation_data(self.directory)

        sim_paths = [os.path.join(self.directory, path) for path in self.df["petname"].unique()]
        stages = list(self.df["stage"].unique())

        self.update_filter_display()

        self.update_simulation_display(sim_paths, stages)

    def update_column_data(self):

        print("updating column data")

        for widget in self.filter_widgets:

            combo_col, combo_mod, lineedit_value, label_value = widget
            try:
                # NOTE: need to check the order this updates in, sometimes col_name is none
                col_name = combo_col.currentText()
                col_type = self.df.dtypes[col_name]
                col_modifiers = get_valid_modifiers(col_type)

                combo_mod.clear()
                combo_mod.addItems(col_modifiers)

                if col_type in [np.float64, np.int64]:
                    min_val, max_val = self.df[col_name].min(), self.df[col_name].max()
                    label_value.setText(f"min: {min_val}, max: {max_val}")
                else:
                    unique_vals = list(self.df[col_name].unique())
                    label_value.setText(f"values: {unique_vals}")
            except Exception as e:
                print(e)

    def filter_by_each_filter(self):

        df_filter = self.df

        for widgets in self.filter_widgets:

            try:
                col_name = widgets[0].currentText()
                modifier = widgets[1].currentText()
                col_type = df_filter.dtypes[col_name]
                value = get_column_value_by_type(widgets[2], col_type)

                # TODO: check if filter val is valid?
                # check if column exists?
                if value == "":
                    break # skip null filters

                df_filter = filter_dataframe_by_modifier(df_filter, col_name, value, modifier)
            except:
                value = "an error occured getting the value"

            print(f"Filtered to {len(df_filter)} simulation stages.")
            print(col_name, col_type, modifier, value)

        sim_paths = [os.path.join(self.directory, path) for path in df_filter["petname"].unique()]
        stages = list(df_filter["stage"].unique())

        self.label_num_filtered_simulations.setText(f"Filtered to {len(df_filter)} simulation stages.")
        self.update_simulation_display(sim_paths, stages)

    def update_filter_display(self):

        print("updating filter display")
        n_filters = int(self.spinBox_num_filters.value())
        filterLayout, filter_widgets = draw_filter_layout(n_filters)
        self.filter_widgets = filter_widgets
        filterBox = QGroupBox(f"")
        filterBox.setLayout(filterLayout)
        self.scroll_area_filter.setWidget(filterBox)

        # # combobox_column, combobox_modifier, line_edit_value, label_min_max = widget
        # connect filter widgets for updates
        for widget in self.filter_widgets:
            widget[0].currentIndexChanged.connect(self.update_column_data)
            widget[0].addItems(list(self.df.columns))

        self.label_num_filtered_simulations.setText(f"Filtered to {len(self.df)} simulation stages.")
        self.scroll_area_filter.update()


    def update_simulation_display(self, sim_paths: list, stages: list):

        print("updating simulation display")
        # TODO: add option to show beams by adding stage 0 to stages...
        if self.checkBox_show_beam.isChecked():
            stages.insert(0, 0)

        runGridLayout = draw_run_layout(sim_paths, stages)
        runBox = QGroupBox(f"")
        runBox.setLayout(runGridLayout)

        self.scroll_area.setWidget(runBox)
        self.scroll_area.update()


def get_column_value_by_type(lineEdit: QLineEdit, type) -> Union[str, int, float]:

    if type == np.int64:
        value = int(lineEdit.text())
    if type == np.float64:
        value = float(lineEdit.text())
    if type == object:
        value = str(lineEdit.text())
    if type == bool:
        value = str(lineEdit.text())
        if value == "False":
            value = False
        if value == "True":
            value = True

    return value

def get_valid_modifiers(type):
    if type in [np.float64, np.int64]:
        modifiers = [mod.name for mod in [MODIFIER.EQUAL_TO, MODIFIER.GREATER_THAN, MODIFIER.LESS_THAN]]

    if type in [object, bool]:
        modifiers = [mod.name for mod in [MODIFIER.EQUAL_TO, MODIFIER.CONTAINS]]

    return modifiers


def filter_dataframe_by_modifier(df, filter_col, value, modifier):

    if modifier == MODIFIER.EQUAL_TO.name:
        df_filter = df[df[filter_col] == value]
    if modifier == MODIFIER.LESS_THAN.name:
        df_filter = df[df[filter_col] < value]
    if modifier == MODIFIER.GREATER_THAN.name:
        df_filter = df[df[filter_col] > value]
    if modifier == MODIFIER.CONTAINS.name:
        values = value.split(",")
        df_filter = df[df[filter_col].isin(values)]

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


def draw_run_layout(sim_directories, stages, nlim=None):
    runGridLayout = QVBoxLayout()

    # limit the number of simulations shown
    if nlim is None:
        nlim = len(sim_directories)

    for sim_path in sim_directories[:nlim]:

        simGridLayout = draw_sim_grid_layout(sim_path, stages)

        simBox = QGroupBox(f"")
        simBox.setLayout(simGridLayout)
        runGridLayout.addWidget(simBox)

    return runGridLayout


def draw_filter_layout(n_filters = 5):

    filterLayout = QVBoxLayout()
    filter_widgets = []

    for i in range(n_filters):


        filterGridLayout = QGridLayout()
        filterGridLayout.setColumnStretch(0, 1)

        combobox_filter_column = QComboBox()
        combobox_filter_modifier = QComboBox()
        line_edit_filter_value = QLineEdit()
        label_min_max = QLabel()

        filter_widgets.append([combobox_filter_column, combobox_filter_modifier, line_edit_filter_value, label_min_max])

        filterGridLayout.addWidget(combobox_filter_column, 0, 0, 1, 2)
        filterGridLayout.addWidget(combobox_filter_modifier, 1, 0)
        filterGridLayout.addWidget(line_edit_filter_value, 1, 1)
        filterGridLayout.addWidget(label_min_max, 2, 0, 1, 2)

        filterBox = QGroupBox(f"")
        filterBox.setLayout(filterGridLayout)
        filterLayout.addWidget(filterBox)



    return filterLayout, filter_widgets



def main():
    """Launch the main application window. """
    application = QtWidgets.QApplication([])
    window = GUIVisualiseResults()
    application.aboutToQuit.connect(window.disconnect)  # cleanup & teardown
    sys.exit(application.exec_())


if __name__ == "__main__":
    main()
