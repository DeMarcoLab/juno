
import glob
import os
import sys
import traceback
from enum import Enum, auto
from typing import Union

import juno
import juno.ui.qtdesigner_files.VisualiseResults as VisualiseResults
import napari
import napari.utils.notifications
import numpy as np
import pandas as pd
from juno import plotting, utils
from PyQt5 import QtWidgets
from PyQt5.QtCore import QAbstractTableModel, Qt
from PyQt5.QtWidgets import (QComboBox, QGridLayout, QGroupBox, QHeaderView,
                             QLabel, QLineEdit, QTableView, QVBoxLayout)


class MODIFIER(Enum):
    EQUAL_TO = auto()
    LESS_THAN = auto()
    GREATER_THAN = auto()
    CONTAINS = auto()

class GUIVisualiseResults(VisualiseResults.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self, viewer: napari.Viewer = None, parent_gui=None):
        super().__init__(parent=parent_gui)
        self.setupUi(MainWindow=self)
        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)
        self.setWindowTitle("Simulation Results")

        self.viewer = viewer
        self.viewer.axes.visible = True
        self.viewer.axes.dashed = True
        self.viewer.axes.labels = True
        self.viewer.axes.colored = False
        self.viewer.scale_bar.visible = True

        self.table_view = None
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

        self.pushButton_open_napari.clicked.connect(lambda: self.view_simulation(view_all=True))
        self.pushButton_open_napari.setVisible(False)
        self.comboBox_napari_sim.setVisible(False)
        self.lineEdit_show_columns.setVisible(False)
        self.lineEdit_show_columns.setText("stage, lens, height, exponent")

        self.label_vis_header.setVisible(False)
        self.checkBox_log_plots.setVisible(False)
        self.label_vis_scale.setVisible(False)
        self.doubleSpinBox_scale.setVisible(False)


    def load_simulation(self):
        try:
            # select directory
            log_dir = os.path.join(os.path.dirname(juno.__file__), "log")

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
            napari.utils.notifications.show_error(f'Error loading simulation folder: {e}')

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

        print("updating simulation display")
        print(df)

        self.paths = [path for path in df["path"].unique()]
        self.STACKABLE = plotting.check_simulations_are_stackable(self.paths)
        if self.STACKABLE:
            self.pushButton_open_napari.setVisible(True)

        # visualisation
        # update available sims for napari
        self.comboBox_napari_sim.setVisible(True)
        self.lineEdit_show_columns.setVisible(True)
        self.checkBox_log_plots.setVisible(True)
        self.label_vis_scale.setVisible(True)
        self.doubleSpinBox_scale.setVisible(True)
        self.label_vis_header.setVisible(True)

        try:
            self.comboBox_napari_sim.currentTextChanged.disconnect()
        except:
            pass
        self.comboBox_napari_sim.clear()
        self.comboBox_napari_sim.addItems([os.path.basename(path) for path in df["petname"].unique()])
        self.comboBox_napari_sim.currentTextChanged.connect(lambda: self.view_simulation(view_all=False))


        # results table
        if self.table_view is None:
            self.table_view = QTableView()
            self.table_view.horizontalHeader().setStretchLastSection(True)
            self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self.tableLayout.addWidget(self.table_view)

        df = df.drop(columns=["path"])
        model = pandasModel(df)
        self.table_view.setModel(model)
        self.tableLayout.update()

        self.view_simulation(view_all=True)

        return 
    
    def view_simulation(self, view_all=False):
        # view sims
        
        if self.STACKABLE is False:
            napari.utils.notifications.show_warning("Simulations do not have the same dimensions, and therefore cannot be displayed together.")

        self.viewer.layers.clear()

        if view_all:
            self.viewer.dims.ndisplay = 2
            name = "simulations"
            simulation_names = [self.comboBox_napari_sim.itemText(i) for i in range(self.comboBox_napari_sim.count())]
            paths = [os.path.join(self.directory, sim_name) for sim_name in simulation_names]
            sim = plotting.load_multi_simulations(paths)
        else:
            simulation_names = [self.comboBox_napari_sim.currentText()]
            name = simulation_names[0]
            path = os.path.join(self.directory, name)
            sim = plotting.load_full_sim_propagation_v2(path)
            self.viewer.dims.ndisplay = 2

        SCALE_DIM = float(self.doubleSpinBox_scale.text())
        scale = [SCALE_DIM, 1, 1]
        # sim = np.moveaxis(sim, [0, 1, 2], [2, 0, 1])#.T # get the simulation into side on view

        # use logarithmic plots
        if bool(self.checkBox_log_plots.isChecked()):
            sim = np.log(sim + 1e-12) 
            print("LOGARITHMIC PLOTS")

        try:
            try:
                self.viewer.layers[name].data = sim 
            except KeyError as e:
                self.viewer.add_image(sim, name=name, colormap="magma", rendering="average", interpolation="bicubic", scale=scale)
               
        except Exception as e:
            napari.utils.notifications.show_error(f"Failure to load viewer: {traceback.format_exc()}")

        # add the points
        sim_height = int(sim.shape[1] // len(simulation_names))
        points = np.array([[0, int(x*sim_height*SCALE_DIM)] for x in range(len(simulation_names))])
        features = {"name": simulation_names}

        text = {
            'string': "{name}",
            'size': 4,
            'color': 'white',
            'translation': np.array([0,-5]),
        }

        self.viewer.add_points(
            points,
            name="sim names",
            features=features,
            text=text,
            size=0.1,
            edge_width=0.01,
            edge_width_is_relative=False,
        )

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
    viewer = napari.Viewer(ndisplay=3)
    view_results_ui = GUIVisualiseResults(viewer=viewer)
    viewer.window.add_dock_widget(view_results_ui, area='right')    
    application.aboutToQuit.connect(view_results_ui.disconnect)  # cleanup & teardown
    sys.exit(application.exec_())


if __name__ == "__main__":
    main()
