
import glob
import os
import sys
import traceback
from enum import Enum, auto
from typing import Union

import lens_simulation
import lens_simulation.ui.qtdesigner_files.VisualiseResults as VisualiseResults
import napari
import numpy as np
from lens_simulation import plotting, utils
from lens_simulation.ui.utils import display_error_message
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QMovie, QPixmap
from PyQt5.QtWidgets import (QComboBox, QGridLayout, QGroupBox, QLabel,
                             QLineEdit, QPushButton, QTableWidget,
                             QTableWidgetItem, QVBoxLayout, QHeaderView, QTableView)

import pandas as pd
from PyQt5.QtCore import QAbstractTableModel, Qt

import matplotlib.pyplot as plt

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


        self.pushButton_open_napari.clicked.connect(self.open_sim_in_napari)
        self.pushButton_open_napari.setVisible(False)
        self.comboBox_napari_sim.setVisible(False)
        self.lineEdit_show_columns.setVisible(False)
        self.lineEdit_show_columns.setText("stage, lens, height, exponent")

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

        self.update_filter_display()

        self.update_simulation_display(df=self.df)

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
                    if col_type == np.float64:
                        label_value.setText(f"min: {min_val:.2e}, max: {max_val:.2e}")
                    else:
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

        self.label_num_filtered_simulations.setText(f"Filtered to {len(df_filter)} simulation stages.")
        self.update_simulation_display(df_filter)

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


    def update_simulation_display(self, df: pd.DataFrame):
        
        # filter columns
        cols_text = self.lineEdit_show_columns.text().split(", ")

        df["path"] = df["log_dir"] + "/" + df["petname"]
        filter_cols = [col for col in df.columns if col in cols_text] + ["petname", "path"]
        df = df[filter_cols]       

        # update available sims for napari
        self.pushButton_open_napari.setVisible(True)
        self.comboBox_napari_sim.setVisible(True)
        self.lineEdit_show_columns.setVisible(True)
        self.comboBox_napari_sim.clear()
        self.comboBox_napari_sim.addItems([os.path.basename(path) for path in df["petname"].unique()])

        print("updating simulation display")
        print(df)

        LOGARITHMIC_PLOTS = False
        if self.checkBox_log_plots.isChecked():
            LOGARITHMIC_PLOTS = True

        runGridLayout = draw_run_layout(df=df, logarithmic=LOGARITHMIC_PLOTS)
        runBox = QGroupBox(f"")
        runBox.setLayout(runGridLayout)

        self.scroll_area.setWidget(runBox)
        self.scroll_area.update()

    def open_sim_in_napari(self):

        sim_name = self.comboBox_napari_sim.currentText()

        print(f"Opening sim: {sim_name} in napari")

        path = os.path.join(self.directory, sim_name)
        full_sim = plotting.load_full_sim_propagation_v2(path)
        # self.napari_viewer = napari.view_image(full_sim, colormap="turbo")


        import napari
        from magicgui import magicgui
        from napari.layers import Image
        from napari.types import ImageData

        # create a viewer and add some images
        self.viewer = napari.Viewer()
        self.viewer.add_image(full_sim, name="simulation", colormap="turbo")

        # turn the gaussian blur function into a magicgui
        # for details on why the `-> ImageData` return annotation works:
        # https://napari.org/guides/magicgui.html#return-annotations
        @magicgui(
            # tells magicgui to call the function whenever a parameter changes
            auto_call=True,
            # `widget_type` to override the default (spinbox) "float" widget
            prop={"widget_type": "FloatSlider", "max": 1.0},
            axis={"choices": [0, 1, 2]},
            layout="horizontal",
        )
        def slice_image(layer: Image, prop: float = 0.5, axis: int = 0) -> ImageData:
            """Slice the volume along the selected axis"""
            if layer:
                return plotting.slice_simulation_view(layer.data, axis=axis, prop=prop)

        # Add it to the napari viewer
        self.viewer.window.add_dock_widget(slice_image, area="bottom")

        napari.run()

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

def draw_image_on_label(fname: str, shape: tuple = (250, 250)) -> QLabel:
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
        label.setPixmap(QPixmap(fname))#.scaled(*shape))

    label.setStyleSheet("border-radius: 5px")

    return label


def draw_sim_grid_layout(path, logarithmic: bool = False, df: pd.DataFrame = None):
    simGridLayout = QGridLayout()
    label_sim_title = QLabel()
    label_sim_title.setStyleSheet("font-size: 14px; font-weight: bold")
    label_sim_title.setText(os.path.basename(path))
    simGridLayout.addWidget(label_sim_title, 0, 0)


    # TODO: 
    # add side on toggle, 
    # filter to rows of table clicked...
    # data table

    model = pandasModel(df)
    view = QTableView()
    view.setModel(model)

    view.horizontalHeader().setStretchLastSection(True)
    view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    simGridLayout.addWidget(view, 1, 0)


    # plotting figures
    # need to regenerate plot to get log correctly
    log_prefix = "log_" if logarithmic is True else ""
    view_fname = os.path.join(path, f"{log_prefix}view.png")
    gif_fname = os.path.join(path, "propagation.gif")
    
    if not os.path.exists(view_fname):
        view_fig = plotting.plot_sim_propagation_v2(path, axis=1, prop=0.5, log=logarithmic)
        plotting.save_figure(view_fig, view_fname)
        plt.close(view_fig)

    if not os.path.exists(gif_fname):
        plotting.save_propagation_gif_full(path)
        
    fnames = [view_fname, gif_fname]

    # draw figures
    for i, fname in enumerate(fnames, 1):
        label = draw_image_on_label(fname)
        simGridLayout.addWidget(label, 1, i)


    return simGridLayout


def draw_run_layout(df: pd.DataFrame, logarithmic: bool = False, nlim: int = None):
    runGridLayout = QVBoxLayout()
    
    sim_directories = df["path"].unique()
    df = df.drop(columns=["path"])

    # limit the number of simulations shown
    if nlim is None:
        nlim = len(sim_directories)

    for sim_path in sim_directories[:nlim]:

        df_sim = df[df["petname"] == os.path.basename(sim_path)]
        df_sim = df_sim.drop(columns=["petname"])
        simGridLayout = draw_sim_grid_layout(sim_path, logarithmic=logarithmic, df=df_sim)

        simBox = QGroupBox(f"")
        simBox.setLayout(simGridLayout)
        simBox.setMaximumHeight(400)
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



# ref https://learndataanalysis.org/display-pandas-dataframe-with-pyqt5-qtableview-widget/
class pandasModel(QAbstractTableModel):

    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parnet=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None




def main():
    """Launch the main application window. """
    application = QtWidgets.QApplication([])
    window = GUIVisualiseResults()
    application.aboutToQuit.connect(window.disconnect)  # cleanup & teardown
    sys.exit(application.exec_())


if __name__ == "__main__":
    main()
