import sys
import traceback

from enum import Enum, auto
import glob
import lens_simulation
import os
import yaml


from lens_simulation import utils
import matplotlib.pyplot as plt

import lens_simulation.UI.qtdesigner_files.SimulationSetup as SimulationSetup
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QGroupBox,
    QGridLayout,
    QLabel,
    QVBoxLayout,
    QPushButton,
    QLineEdit,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QCheckBox,
    QFileDialog,
)
from PyQt5.QtGui import QImage, QPixmap, QMovie
import numpy as np

from pathlib import Path
from pprint import pprint

from lens_simulation import validation

from lens_simulation.constants import (
    MICRON_TO_METRE,
    METRE_TO_MICRON,
    NANO_TO_METRE,
    METRE_TO_NANO,
)


# TODO: validate loaded configs
# TODO: validate entire setup


class GUISimulationSetup(SimulationSetup.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self, parent_gui=None):
        super().__init__(parent=parent_gui)
        self.setupUi(MainWindow=self)
        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)
        self.setWindowTitle("Simulation Setup")

        self.simulation_config = {}

        self.setup_connections()

        self.update_all_displays()

        self.showNormal()

    def setup_connections(self):

        print("setting up connections")

        self.spinBox_sim_num_stages.valueChanged.connect(
            self.update_stage_input_display
        )
        self.pushButton_generate_simulation.clicked.connect(
            self.generate_simulation_config
        )
        self.pushButton_sim_beam.clicked.connect(self.load_beam_config)

        self.pushButton_save_sim_config.clicked.connect(self.save_simulation_config)

        self.actionLoad_Config.triggered.connect(self.load_simulation_config)

    def update_all_displays(self):

        self.update_stage_input_display()

    def update_simulation_config(self):
        print("updating simulation config")

        sim_pixel_size = float(self.doubleSpinBox_pixel_size.value()) * MICRON_TO_METRE
        sim_width = float(self.doubleSpinBox_sim_width.value()) * MICRON_TO_METRE
        sim_height = float(self.doubleSpinBox_sim_height.value()) * MICRON_TO_METRE
        sim_wavelength = (
            float(self.doubleSpinBox_sim_wavelength.value()) * NANO_TO_METRE
        )
        sim_amplitude = float(self.doubleSpinBox_sim_amplitude.value())

        self.simulation_config["sim_parameters"] = {
            "A": sim_amplitude,
            "pixel_size": sim_pixel_size,
            "sim_height": sim_height,
            "sim_width": sim_width,
            "sim_wavelength": sim_wavelength,
        }

        log_dir = str(self.lineEdit_log_dir.text())
        save_raw = bool(self.checkBox_save_raw.isChecked())
        save_plot = bool(self.checkBox_save_plot.isChecked())

        self.simulation_config["options"] = {
            "log_dir": log_dir,
            "save": save_raw,
            "save_plot": save_plot,
        }

    def input_widget_state_change(self):

        for stage_no, widgets in enumerate(self.input_widgets, 1):

            if widgets[14].isChecked():
                widgets[9].setText("Focal Distance Start Multiple")
                widgets[11].setText("Focal Distance Finish Multiple")
            else:
                widgets[9].setText("Start Distance")
                widgets[11].setText("Finish Distance")

    def save_simulation_config(self):

        # open file dialog
        sim_config_filename, _ = QFileDialog.getSaveFileName(self,
                    caption="Save Simulation Config", 
                    directory=os.path.dirname(lens_simulation.__file__),
                    filter="Yaml files (*.yml, *.yaml)")
        if sim_config_filename:          
            # set name

            # same as yaml file
            with open(sim_config_filename, "w") as f:
                yaml.safe_dump(self.simulation_config, f)

            self.statusBar.showMessage(f"Simulation config saved to {sim_config_filename}")

    def load_simulation_config(self):
        print("loading simulation config")
        # open file dialog
        sim_config_filename, _ = QFileDialog.getOpenFileName(self,
                    caption="Load Simulation Config", 
                    directory=os.path.dirname(lens_simulation.__file__),
                    filter="Yaml files (*.yml, *.yaml)"
                    )
        if sim_config_filename:          
            
            config = utils.load_config(sim_config_filename)
            
            self.statusBar.showMessage(f"Simulation config loaded from {sim_config_filename}")

            pprint(config)

            print("loaded config")
            # TODO: how to handle partial configs??? throw error?

            # load config values into UI....
            self.simulation_config = config

            # set sim parameters
            self.doubleSpinBox_pixel_size.setValue(float(config["sim_parameters"]["pixel_size"]) * METRE_TO_MICRON)
            self.doubleSpinBox_sim_width.setValue(float(config["sim_parameters"]["sim_width"])* METRE_TO_MICRON)
            self.doubleSpinBox_sim_height.setValue(float(config["sim_parameters"]["sim_height"]) * METRE_TO_MICRON)
            self.doubleSpinBox_sim_wavelength.setValue(float(config["sim_parameters"]["sim_wavelength"]) * METRE_TO_NANO)
            self.doubleSpinBox_sim_amplitude.setValue(float(config["sim_parameters"]["A"]))

            # set sim optiosn
            self.lineEdit_log_dir.setText(str(config["options"]["log_dir"]))
            self.checkBox_save_plot.setChecked(bool(config["options"]["save_plot"]))
            self.checkBox_save_raw.setChecked(bool(config["options"]["save"]))

            # set beam
            self.pushButton_sim_beam.setText("beam")

            # set n stages
            self.spinBox_sim_num_stages.setValue(int(len(config["stages"])))

            # update each stage info

            for i, stage_config in enumerate(config["stages"]):

                widgets = self.input_widgets[i]
                widgets[2].setText(str(stage_config["lens"]))
                widgets[4].setValue(float(stage_config["output"]))
                widgets[6].setValue(int(stage_config["n_slices"]))
                widgets[8].setValue(float(stage_config["step_size"]))
                widgets[10].setValue(float(stage_config["start_distance"]))
                widgets[12].setValue(float(stage_config["finish_distance"]))

                widgets[14].setChecked(bool(stage_config["use_equivalent_focal_distance"]))

                if widgets[14].isChecked():
                    widgets[10].setValue(float(stage_config["focal_distance_start_multiple"]))
                    widgets[12].setValue(float(stage_config["focal_distance_multiple"]))               
                

            self.SIMULATION_CONFIG_LOADED = True

    def generate_simulation_config(self):
        print("generating simulation config")

        # TODO: need to check if things are loaded...
        
        self.update_simulation_config()
        self.read_stage_input_values()

        print("-" * 50)
        pprint(self.simulation_config)
        print("-" * 50)

        try:
            validation._validate_simulation_config(self.simulation_config)
            print("Configuration is valid.")
            self.draw_simulation_stage_display()

            self.pushButton_save_sim_config.setEnabled(True)

        except Exception as e:
            display_error_message(f"Invalid simulation config. \n{e}")

        # checks
        # beam selected
        # lenses selected
        # required fields...?

    def draw_simulation_stage_display(self):

        # Think this can only be called once? why?

        stage_layout, widgets = create_stage_structure_display(self.simulation_config)       
        groupBox_stage_display = QGroupBox(f"")
        groupBox_stage_display.setLayout(stage_layout)
        self.scrollArea_stage_display.setWidget(groupBox_stage_display)
        self.scrollArea_stage_display.update()


    def read_stage_input_values(self):

        stage_configs = []

        for widgets in self.input_widgets:

            stage_config = {}

            lens_name = str(widgets[2].text())
            output = float(widgets[4].value())
            n_slices = int(widgets[6].value())
            step_size = float(widgets[8].value())
            
            use_focal_distance = bool(widgets[14].isChecked())
            if use_focal_distance:
                start_distance = 0.0
                finish_distance = 0.5e-3
                focal_distance_start_multiple = float(widgets[10].value())
                focal_distance_multiple = float(widgets[12].value())
            else:
                start_distance = float(widgets[10].value())
                finish_distance = float(widgets[12].value())
                focal_distance_start_multiple = 0.0
                focal_distance_multiple = 0.0

            stage_config = {
                "lens": lens_name,
                "output": output,
                "n_slices": n_slices,
                "step_size": step_size,
                "start_distance": start_distance,
                "finish_distance": finish_distance,
                "options": {
                    "use_equivalent_focal_distance": use_focal_distance,
                    "focal_distance_start_multiple": focal_distance_start_multiple,
                    "focal_distance_multiple": focal_distance_multiple,
                },
            }

            stage_configs.append(stage_config)

        self.simulation_config["stages"] = stage_configs

        # 0. label_title,
        # 1. lens_label,
        # 2. lens_button,
        # 3. output_label,
        # 4. output_spinbox,
        # 5. n_slices_label,
        # 6. n_slices_spinbox,
        # 7. step_size_label,
        # 8. step_size_spinbox,
        # 9. start_distance_label,
        # 10. start_distance_spinbox,
        # 11. finish_distance_label,
        # 12. finish_distance_spinbox,
        # 13. use_focal_label,
        # 14. use_focal_checkbox

    def load_lens_config(self):

        print("loading lens_config")

        lens_config_filename, _ = QFileDialog.getOpenFileName(
            self, "Select Lens Configuration", os.path.dirname(lens_simulation.__file__), "Yaml files (*.yml, *.yaml)"
        )
        
        try:
            lens_config = utils.load_yaml_config(lens_config_filename)

            if "lenses" not in self.simulation_config:
                self.simulation_config["lenses"] = []

            validation._validate_default_lens_config(lens_config)
            self.simulation_config["lenses"].append(lens_config)
            self.sender().setText(f"{lens_config['name']}")
        except Exception as e:
            display_error_message(f"Invalid config. \n{e}")

    def load_beam_config(self):

        print("loading beam config")

        beam_config_filename, _ = QFileDialog.getOpenFileName(
            self, "Select Beam Configuration", os.path.dirname(lens_simulation.__file__), "Yaml files (*.yml, *.yaml)"
        )

        try:
            beam_config = utils.load_yaml_config(beam_config_filename)

            validation._validate_default_beam_config(beam_config)

            self.simulation_config["beam"] = beam_config
            self.pushButton_sim_beam.setText(
                f"{Path(beam_config_filename).stem}"
            )
        except Exception as e:
            display_error_message(f"Invalid config. \n{e}")

    def update_stage_input_display(self):

        print("updating stage display")

        sim_num_stages = int(self.spinBox_sim_num_stages.value())

        input_layout = QVBoxLayout()

        self.input_widgets = []

        for stage_no, _ in enumerate(range(sim_num_stages), 1):

            layout, widgets = create_stage_input_display(stage_no)

            widgets[2].clicked.connect(self.load_lens_config)
            widgets[14].stateChanged.connect(self.input_widget_state_change)

            self.input_widgets.append(widgets)

            stageBox = QGroupBox(f"")
            stageBox.setLayout(layout)
            input_layout.addWidget(stageBox)

        inputBox = QGroupBox(f"")
        inputBox.setLayout(input_layout)
        self.scrollArea_stages.setWidget(inputBox)
        self.scrollArea_stages.update()


def create_stage_structure_display(config):

    # create / delete tmp directory
    tmp_directory = os.path.join(os.path.dirname(lens_simulation.__file__), "tmp")
    os.makedirs(tmp_directory, exist_ok=True)

    layout = QGridLayout()

    # TODO: need to account for focal distance multiple.... otherwise it gets too large to plot

    # simulation setup
    fig = utils.plot_simulation_setup(config)
    sim_setup_fname  = os.path.join(tmp_directory, "sim_setup.png")
    utils.save_figure(fig, sim_setup_fname)
    plt.close(fig)

    sim_label = QLabel()
    sim_label.setPixmap(QPixmap(sim_setup_fname))

    layout.addWidget(sim_label, 0, 0, 1, len(config["stages"]) + 1)

    # beam
    beam_label = QLabel()
    beam_label.setText("I'm a Beam (side-on).")

    beam_top_down_label = QLabel()
    beam_top_down_label.setText("I'm a Beam (top-down).")
    layout.addWidget(beam_label, 1, 0)
    layout.addWidget(beam_top_down_label, 2, 0)

    display_widgets = [[beam_label, beam_top_down_label]]

    # stages
    for i, stage_config in enumerate(config["stages"], 1):

        def array_to_qlabel(arr):
            """Convert a numpy array to a QLabel"""
            label = QLabel()
            if arr.ndim == 2:
                qimage = QImage(arr, arr.shape[1], arr.shape[0], QImage.Format_Grayscale8)
            if arr.ndim == 3:
                qimage = QImage(arr, arr.shape[1], arr.shape[0], arr.shape[1] * 3, QImage.Format_RGB888)
            label.setPixmap(QPixmap(qimage))
            return label

        lens_name = stage_config["lens"]

        for conf in config["lenses"]:
            if conf["name"] == lens_name:
                lens_config = conf
   
        from lens_simulation.Lens import generate_lens
        from lens_simulation.Medium import Medium

        lens = generate_lens(lens_config, 
                    Medium(lens_config["medium"], config["sim_parameters"]["sim_wavelength"]), 
                    config["sim_parameters"]["pixel_size"])

        output = stage_config["output"]

        fig = utils.plot_lens_profile_slices(lens, max_height=lens.height)
        side_on_fname = os.path.join(tmp_directory, "side_on_profile.png")
        utils.save_figure(fig, side_on_fname)
        plt.close(fig)

        fig = utils.plot_lens_profile_2D(lens)
        top_down_fname = os.path.join(tmp_directory, "top_down_profile.png")
        utils.save_figure(fig, top_down_fname)
        plt.close(fig)

        img = np.random.rand(400,400,3).astype(np.uint8) * 255
        output_img = np.ones_like(img) * output

        stage_label = QLabel()
        stage_label.setPixmap(QPixmap(side_on_fname).scaled(300, 300))

        stage_top_down_label = QLabel()
        stage_top_down_label.setPixmap(QPixmap(top_down_fname).scaled(300, 300)) #array_to_qlabel(lens.profile)
                
        layout.addWidget(stage_label, 1, i)
        layout.addWidget(stage_top_down_label, 2, i)

        display_widgets.append([stage_label, stage_top_down_label])


    # generate beam
    # generate lens

    # plot
    # show on label
    
    return layout, display_widgets


def create_stage_input_display(stage_no):

    layout = QGridLayout()

    label_title = QLabel()
    label_title.setText(f"Stage Number {stage_no}")

    # lens
    lens_label = QLabel()
    lens_label.setText(f"Lens")

    lens_button = QPushButton()
    lens_button.setText("...")  # use a different kind of widget?

    # output
    output_label = QLabel()
    output_label.setText(f"Output Medium")
    output_spinbox = QDoubleSpinBox()

    # n_slices
    n_slices_label = QLabel()
    n_slices_label.setText(f"Number of Slices")
    n_slices_spinbox = QSpinBox()

    # step size
    step_size_label = QLabel()
    step_size_label.setText(f"Step Size")
    step_size_spinbox = QDoubleSpinBox()
    step_size_spinbox.setDecimals(6)

    # start distance
    start_distance_label = QLabel()
    start_distance_label.setText(f"Start Distance")
    start_distance_spinbox = QDoubleSpinBox()
    start_distance_spinbox.setDecimals(6)

    # finish distance
    finish_distance_label = QLabel()
    finish_distance_label.setText(f"Finish Distance")
    finish_distance_spinbox = QDoubleSpinBox()
    finish_distance_spinbox.setDecimals(6)

    # options
    #   use focal distance
    use_focal_label = QLabel()
    use_focal_label.setText(f"Use Focal Distance")
    use_focal_checkbox = QCheckBox()

    # if use_focal_checkbox is checked, change start, finsish distance to these options...
    #   start focal distance
    #   end focal distance

    layout.addWidget(label_title, 0, 0, 1, 2)
    layout.addWidget(lens_label, 1, 0)
    layout.addWidget(lens_button, 1, 1)

    layout.addWidget(output_label, 2, 0)
    layout.addWidget(output_spinbox, 2, 1)

    layout.addWidget(n_slices_label, 3, 0)
    layout.addWidget(n_slices_spinbox, 3, 1)

    layout.addWidget(step_size_label, 4, 0)
    layout.addWidget(step_size_spinbox, 4, 1)

    layout.addWidget(start_distance_label, 5, 0)
    layout.addWidget(start_distance_spinbox, 5, 1)

    layout.addWidget(finish_distance_label, 6, 0)
    layout.addWidget(finish_distance_spinbox, 6, 1)

    layout.addWidget(use_focal_label, 7, 0)
    layout.addWidget(use_focal_checkbox, 7, 1)

    widgets = [
        label_title,
        lens_label,
        lens_button,
        output_label,
        output_spinbox,
        n_slices_label,
        n_slices_spinbox,
        step_size_label,
        step_size_spinbox,
        start_distance_label,
        start_distance_spinbox,
        finish_distance_label,
        finish_distance_spinbox,
        use_focal_label,
        use_focal_checkbox,
    ]

    return layout, widgets


def display_error_message(message, title="Error Message"):
    """PyQt dialog box displaying an error message."""
    # logging.debug('display_error_message')
    # logging.exception(message)
    error_dialog = QtWidgets.QErrorMessage()
    error_dialog.setWindowTitle(title)
    error_dialog.showMessage(message)
    error_dialog.showNormal()
    error_dialog.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
    error_dialog.exec_()


def main():
    """Launch the main application window. """
    application = QtWidgets.QApplication([])
    window = GUISimulationSetup()
    application.aboutToQuit.connect(window.disconnect)  # cleanup & teardown
    sys.exit(application.exec_())


if __name__ == "__main__":
    main()