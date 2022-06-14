import os
import sys
import traceback

import lens_simulation.UI.qtdesigner_files.BeamCreator as BeamCreator
import numpy as np
import yaml
from lens_simulation import constants, utils
from lens_simulation.beam import Beam, generate_beam
from lens_simulation.Lens import GratingSettings, LensType, Medium
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt5 import QtCore, QtGui, QtWidgets

from lens_simulation.structures import SimulationParameters

lens_type_dict = {"Cylindrical": LensType.Cylindrical, "Spherical": LensType.Spherical}

# maps the index of comboboxes to a constant
units_dict = {
    0: constants.NANO_TO_METRE,
    1: constants.MICRON_TO_METRE,
    2: constants.MM_TO_METRE,
}

beam_spread_dict = {
    "Plane": 0,
    "Converging": 1,
    "Diverging": 2,
}

beam_shape_dict = {
    "Circular": 0,
    "Square": 1,
    "Rectangular": 2,
}


class GUIBeamCreator(BeamCreator.Ui_BeamCreator, QtWidgets.QMainWindow):
    def __init__(self, parent_gui=None):
        super().__init__(parent=parent_gui)
        self.setupUi(BeamCreator=self)
        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)

        # default to um
        self.comboBox_Units.setCurrentIndex(1)
        self.units = units_dict[1]

        # set up of image frames
        self.pc_Profile = None
        self.pc_Convergence = None

        self.create_new_beam_dict()
        self.update_UI()

        self.setup_connections()

        self.center_window()
        self.showNormal()

    ### Setup methods ###

    def setup_connections(self):
        # self.pushButton_LoadProfile.clicked.connect(self.load_profile)
        # self.pushButton_GenerateProfile.clicked.connect(self.create_lens)
        # self.pushButton_SaveProfile.clicked.connect(self.save_profile)

        # self.comboBox_Units.currentIndexChanged.connect(self.update_units)

        # connect each of the lens parameter selectors to update profile in live view
        [
            value.valueChanged.connect(self.live_update_profile)
            for value in self.__dict__.values()
            if value.__class__ is QtWidgets.QDoubleSpinBox
        ]
        [
            value.currentTextChanged.connect(self.live_update_profile)
            for value in self.__dict__.values()
            if value.__class__ is QtWidgets.QComboBox
        ]

    ### Generation methods ###

    def create_new_beam_dict(self):
        # dummy sim
        self.sim_dict = dict()
        self.sim_dict["pixel_size"] = 1.0e-6
        self.sim_dict["width"] = 500.0e-6
        self.sim_dict["height"] = 500.0e-6
        self.sim_dict["wavelength"] = 488.0e-9

        self.beam_dict = dict()
        self.beam_dict["name"] = "Beam"
        self.beam_dict["distance_mode"] = "direct"
        self.beam_dict["spread"] = "plane"
        self.beam_dict["shape"] = "rectangular"
        self.beam_dict["width"] = 300.0e-6
        self.beam_dict["height"] = 50.0e-6
        self.beam_dict["position_x"] = 0.0e-6
        self.beam_dict["position_y"] = 0.0e-6
        self.beam_dict["theta"] = None
        self.beam_dict["numerical_aperture"] = None
        self.beam_dict["tilt_x"] = 0.0
        self.beam_dict["tilt_y"] = 0.0
        self.beam_dict["source_distance"] = 2.0e-3
        self.beam_dict["final_width"] = None
        self.beam_dict["focal_multiple"] = None
        self.beam_dict["n_slices"] = 10
        self.beam_dict["lens_type"] = "Spherical"

        self.parameters = SimulationParameters(
            A=10000,
            pixel_size=self.sim_dict["pixel_size"],
            sim_width=self.sim_dict["width"],
            sim_height=self.sim_dict["height"],
            sim_wavelength=self.sim_dict["wavelength"]
        )

    def create_beam(self):
        self.beam = generate_beam(config=self.beam_dict, parameters=self.parameters)


    ### UI <-> Config methods ###

    def update_beam_dict(self):
        self.update_config_general()
        self.update_config_beam_spread()
        self.update_config_beam_shape()

    def update_UI(self):
        self.update_UI_general()
        self.update_UI_beam_spread()
        self.update_UI_beam_shape()

    def update_UI_general(self):
        # Config -> UI | General settings #
        self.lineEdit_LensName.setText(self.beam_dict["name"])
        self.spinBox_NSlices.setValue(self.beam_dict["n_slices"])
        self.doubleSpinBox_ShiftX.setValue(self.beam_dict["position_x"]/self.units)
        self.doubleSpinBox_ShiftY.setValue(self.beam_dict["position_y"]/self.units)
        self.doubleSpinBox_Width.setValue(self.beam_dict["width"]/self.units)
        self.doubleSpinBox_Height.setValue(self.beam_dict["height"]/self.units)

    def update_config_general(self):
        # UI -> config | General settings #
        self.beam_dict["name"] = self.lineEdit_LensName.text()
        self.beam_dict["n_slices"] = self.spinBox_NSlices.value()
        self.beam_dict["position_x"] = self.doubleSpinBox_ShiftX.value() * self.units
        self.beam_dict["position_y"] = self.doubleSpinBox_ShiftY.value() * self.units
        self.beam_dict["width"] = self.doubleSpinBox_Width.value() * self.units
        self.beam_dict["height"] = self.doubleSpinBox_Height.value() * self.units

    def update_UI_beam_spread(self):
        # Config -> UI | Beam Spread settings #
        self.comboBox_BeamSpread.setCurrentIndex(beam_spread_dict[self.beam_dict["spread"].title()])

    def update_config_beam_spread(self):
        # UI -> config | Beam Spread settings #
        if self.comboBox_BeamSpread.currentText() == "Planar":
            self.beam_dict["spread"] = "plane"
        if self.comboBox_BeamSpread.currentText() == "Converging":
            self.beam_dict["spread"] = "converging"
        if self.comboBox_BeamSpread.currentText() == "Diverging":
            self.beam_dict["spread"] = "diverging"

    def update_UI_beam_shape(self):
        # Config -> UI | Beam Shape settings #
        self.comboBox_BeamShape.clear()
        if self.beam_dict["spread"].title() == "Plane":
            self.comboBox_BeamShape.addItem("Circular")
            self.comboBox_BeamShape.addItem("Square")
            self.comboBox_BeamShape.addItem("Rectangular")
            self.comboBox_BeamShape.setCurrentIndex(beam_shape_dict[self.beam_dict["shape"].title()])
        else:
            self.comboBox_BeamShape.addItem("Circular")
            self.beam_dict["beam_shape"] = "Circular"
            self.comboBox_BeamShape.setCurrentIndex(0)


    def update_config_beam_shape(self):
        # UI -> config | Beam Shape settings #
        self.beam_dict["shape"] = self.comboBox_BeamShape.currentText()

    def live_update_profile(self):
        if self.checkBox_LiveUpdate.isChecked():
            try:
                self.checkBox_LiveUpdate.setChecked(False)
                self.update_beam_dict()
                # self.create_lens()
                # self.update_UI_limits()
                self.update_UI()
                self.checkBox_LiveUpdate.setChecked(True)
            except Exception as e:
                self.display_error_message(traceback.format_exc())

    def center_window(self):
        """Centers the window in the display"""
        # Get the desktop dimensions
        desktop = QtWidgets.QDesktopWidget()
        self.move(
            (desktop.width() - self.width()) / 2,
            (desktop.height() - self.height()) / 3.0,
        )

    def display_error_message(self, message):
        """PyQt dialog box displaying an error message."""
        # logging.debug('display_error_message')
        # logging.exception(message)
        self.error_dialog = QtWidgets.QErrorMessage()
        self.error_dialog.showMessage(message)
        self.error_dialog.showNormal()
        self.error_dialog.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        self.error_dialog.exec_()

def main():
    """Launch the `piescope_gui` main application window."""
    application = QtWidgets.QApplication([])
    window = GUIBeamCreator()
    application.aboutToQuit.connect(window.disconnect)  # cleanup & teardown
    sys.exit(application.exec_())


if __name__ == "__main__":
    main()
