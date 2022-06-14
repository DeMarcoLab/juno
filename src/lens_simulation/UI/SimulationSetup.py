import sys
import traceback

from enum import Enum, auto
import glob
from typing import Union
from venv import create
import lens_simulation
import os

from lens_simulation import utils

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
    QDoubleSpinBox, QSpinBox, QCheckBox
)
from PyQt5.QtGui import QImage, QPixmap, QMovie
import numpy as np

from pprint import pprint

from lens_simulation.constants import (
    MICRON_TO_METRE,
    METRE_TO_MICRON,
    NANO_TO_METRE,
    METRE_TO_NANO,
)


class GUISimulationSetup(SimulationSetup.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self, parent_gui=None):
        super().__init__(parent=parent_gui)
        self.setupUi(MainWindow=self)
        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)
        self.setWindowTitle("Simulation Setup")

        self.setup_connections()

        self.update_all_displays()

        self.showNormal()

    def setup_connections(self):

        print("setting up connections")

        # setup sim parameters
        self.doubleSpinBox_pixel_size.valueChanged.connect(
            self.update_simulation_config
        )
        self.doubleSpinBox_sim_width.valueChanged.connect(self.update_simulation_config)
        self.doubleSpinBox_sim_height.valueChanged.connect(
            self.update_simulation_config
        )
        self.doubleSpinBox_sim_wavelength.valueChanged.connect(
            self.update_simulation_config
        )
        self.doubleSpinBox_sim_amplitude.valueChanged.connect(
            self.update_simulation_config
        )
        self.spinBox_sim_num_stages.valueChanged.connect(
            self.update_stage_input_display
        )

        self.pushButton_generate_simulation.clicked.connect(self.read_stage_input_values)

    def update_all_displays(self):

        self.update_stage_input_display()

    def update_simulation_config(self):
        print("updating simulation config")

        sim_config = {}

        sim_pixel_size = float(self.doubleSpinBox_pixel_size.value()) * MICRON_TO_METRE
        sim_width = float(self.doubleSpinBox_sim_width.value()) * MICRON_TO_METRE
        sim_height = float(self.doubleSpinBox_sim_height.value()) * MICRON_TO_METRE
        sim_wavelength = (
            float(self.doubleSpinBox_sim_wavelength.value()) * NANO_TO_METRE
        )
        sim_amplitude = float(self.doubleSpinBox_sim_amplitude.value())

        sim_num_stages = int(self.spinBox_sim_num_stages.value())

        sim_config["sim_parameters"] = {
            "A": sim_amplitude,
            "pixel_size": sim_pixel_size,
            "sim_height": sim_height,
            "sim_width": sim_width,
            "sim_wavelength": sim_wavelength,
        }

        from pprint import pprint

        print("-" * 50)
        print("Sim Config:")
        pprint(sim_config)
        print(f"sim_num_stages: ", sim_num_stages)
        print("-" * 50)

    def test_state_change(self):

        for stage_no, widgets in enumerate(self.input_widgets, 1):

            print("STAGE_NO: ", stage_no, "CHECKED: ", widgets[14].isChecked())
            
            if widgets[14].isChecked():
                widgets[9].setText("Focal Distance Start Multiple")
                widgets[11].setText("Focal Distance Finish Multiple")
            else:
                widgets[9].setText("Start Distance")
                widgets[11].setText("Finish Distance")

    def read_stage_input_values(self):

        stage_configs = []

        for widgets in self.input_widgets:
            
            stage_config = {}

            # lens = TODO:
            output = float(widgets[4].value())
            n_slices = int(widgets[6].value())
            step_size = float(widgets[8].value())
            start_distance = float(widgets[10].value())
            finish_distance = float(widgets[12].value())
            use_focal_distance = bool(widgets[14].isChecked())

            stage_config = {
                "output": output,
                "n_slices": n_slices,
                "step_size": step_size,
                "start_distance": start_distance,
                "finish_distance": finish_distance,
                "options": {
                    "use_equivalent_focal_distance": use_focal_distance,
                    "focal_distance_start_multiple": start_distance,
                    "focal_distance_multiple": finish_distance
                }
            }

            stage_configs.append(stage_config)


        
        pprint(stage_configs)

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
     



    def update_stage_input_display(self):

        print("updating stage display")

        sim_num_stages = int(self.spinBox_sim_num_stages.value())

        input_layout = QVBoxLayout()

        self.input_widgets = []

        for stage_no, _ in enumerate(range(sim_num_stages), 1):

            layout, widgets = create_stage_input_display(stage_no)

            widgets[14].stateChanged.connect(self.test_state_change)

            self.input_widgets.append(widgets)

            stageBox = QGroupBox(f"")
            stageBox.setLayout(layout)
            input_layout.addWidget(stageBox)

        inputBox = QGroupBox(f"")
        inputBox.setLayout(input_layout)
        self.scrollArea_stages.setWidget(inputBox)
        self.scrollArea_stages.update()


def create_stage_input_display(stage_no):

    layout = QGridLayout()

    label_title = QLabel()
    label_title.setText(f"Stage Number {stage_no}")

    # lens
    lens_label = QLabel()
    lens_label.setText(f"Stage Lens")

    lens_button = QPushButton()
    lens_button.setText("Load Lens")  # use a different kind of widget?

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
    step_size_spinbox = QSpinBox()

    # start distance
    start_distance_label = QLabel()
    start_distance_label.setText(f"Start Distance")
    start_distance_spinbox = QSpinBox()

    # finish distance
    finish_distance_label = QLabel()
    finish_distance_label.setText(f"Finish Distance")
    finish_distance_spinbox = QSpinBox()

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
            lens_label, lens_button,
            output_label, output_spinbox,
            n_slices_label, n_slices_spinbox,
            step_size_label, step_size_spinbox,
            start_distance_label, start_distance_spinbox,
            finish_distance_label, finish_distance_spinbox,
            use_focal_label, use_focal_checkbox
    ]

    return layout, widgets


def main():
    """Launch the main application window. """
    application = QtWidgets.QApplication([])
    window = GUISimulationSetup()
    application.aboutToQuit.connect(window.disconnect)  # cleanup & teardown
    sys.exit(application.exec_())


if __name__ == "__main__":
    main()
