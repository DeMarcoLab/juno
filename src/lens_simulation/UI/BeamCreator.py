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
    "Rectangular": 1,
}

# TODO: Add amplitude

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
        self.create_beam()
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
        self.beam_dict["theta"] = 1. # Degrees
        self.beam_dict["numerical_aperture"] = None
        self.beam_dict["tilt_x"] = 0.0
        self.beam_dict["tilt_y"] = 0.0
        self.beam_dict["source_distance"] = 200.e-6
        self.beam_dict["final_width"] = None
        self.beam_dict["focal_multiple"] = None
        self.beam_dict["n_slices"] = 10
        self.beam_dict["lens_type"] = "Spherical"

    def create_beam(self):
        self.parameters = SimulationParameters(
            A=10000,
            pixel_size=self.sim_dict["pixel_size"],
            sim_width=self.sim_dict["width"],
            sim_height=self.sim_dict["height"],
            sim_wavelength=self.sim_dict["wavelength"]
            # TODO: add wavelength
        )

        self.beam = generate_beam(config=self.beam_dict, parameters=self.parameters)
        self.update_image_frames()


    ### UI <-> Config methods ###

    def update_beam_dict(self):
        self.update_config_general()
        self.update_config_beam_spread()
        self.update_config_beam_shape()
        self.update_config_convergence_angle()
        self.update_config_distance()
        self.update_config_tilt()
        self.update_config_sim()

    def update_UI(self):
        self.update_UI_general()
        self.update_UI_beam_spread()
        self.update_UI_beam_shape()
        self.update_UI_convergence_angle()
        self.update_UI_distance()
        self.update_UI_tilt()
        self.update_UI_sim()

    def update_UI_general(self):
        # Config -> UI | General settings #
        self.lineEdit_LensName.setText(self.beam_dict["name"])
        self.spinBox_NSlices.setValue(self.beam_dict["n_slices"])
        self.doubleSpinBox_ShiftX.setValue(self.beam_dict["position_x"]/self.units)
        self.doubleSpinBox_ShiftY.setValue(self.beam_dict["position_y"]/self.units)
        self.doubleSpinBox_Width.setValue(self.beam_dict["width"]/self.units)
        self.doubleSpinBox_Height.setValue(self.beam_dict["height"]/self.units)
        self.comboBox_LensType.setCurrentText(self.beam_dict["lens_type"].title())

    def update_config_general(self):
        # UI -> config | General settings #
        self.beam_dict["name"] = self.lineEdit_LensName.text()
        self.beam_dict["n_slices"] = self.spinBox_NSlices.value()
        self.beam_dict["position_x"] = self.format_float(self.doubleSpinBox_ShiftX.value() * self.units)
        self.beam_dict["position_y"] = self.format_float(self.doubleSpinBox_ShiftY.value() * self.units)
        self.beam_dict["width"] = self.format_float(self.doubleSpinBox_Width.value() * self.units)
        self.beam_dict["height"] = self.format_float(self.doubleSpinBox_Height.value() * self.units)
        self.beam_dict["lens_type"] = self.comboBox_LensType.currentText().lower()

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
            self.comboBox_BeamShape.addItem("Rectangular")
            self.comboBox_BeamShape.setCurrentIndex(beam_shape_dict[self.beam_dict["shape"].title()])
        else:
            self.comboBox_BeamShape.addItem("Circular")
            self.beam_dict["beam_shape"] = "Circular"
            self.comboBox_BeamShape.setCurrentIndex(0)

    def update_config_beam_shape(self):
        # UI -> config | Beam Shape settings #
        if self.comboBox_BeamSpread.currentText() != "Planar":
            self.beam_dict["shape"] = "circular"
        else:
            self.beam_dict["shape"] = self.comboBox_BeamShape.currentText()

    def update_UI_convergence_angle(self):
        # Config -> UI | Angle settings #
        if self.beam_dict["spread"].title() == "Plane":
            self.frame_BeamAngle.setEnabled(False)
        else:
            self.frame_BeamAngle.setEnabled(True)

        if self.beam_dict["theta"] is not None and self.beam_dict["theta"] != 0.:
            self.comboBox_BeamAngle.setCurrentText("Theta")
            self.doubleSpinBox_BeamAngle.setValue(self.beam_dict["theta"]
            )
            return

        self.comboBox_BeamAngle.setCurrentText("Numerical Aperture")
        self.doubleSpinBox_BeamAngle.setValue(self.beam_dict["numerical_aperture"])

    def update_config_convergence_angle(self):
        # UI -> config | Angle settings #
        if self.comboBox_BeamAngle.currentText() == "Numerical Aperture":
            self.beam_dict["theta"] = 0.
            self.beam_dict["numerical_aperture"] = self.doubleSpinBox_BeamAngle.value()
            print(self.beam_dict)
            return

        self.beam_dict["theta"] = self.doubleSpinBox_BeamAngle.value()
        self.beam_dict["numerical_aperture"] = 0.

    def update_UI_distance(self):
        # Config -> UI | Distance settings #
        if self.beam_dict["distance_mode"] == "direct":
            self.comboBox_DistanceMode.setCurrentText("Absolute Distance")
            self.doubleSpinBox_Distance.setValue(self.beam_dict["source_distance"]/self.units)
        elif self.beam_dict["distance_mode"] == "width":
            self.comboBox_DistanceMode.setCurrentText("Final Beam Width")
            self.doubleSpinBox_Distance.setValue(self.beam_dict["final_width"]/self.units)
        elif self.beam_dict["distance_mode"] == "focal":
            self.comboBox_DistanceMode.setCurrentText("Focal Length Multiple")
            self.doubleSpinBox_Distance.setValue(self.beam_dict["focal_multiple"])

    def update_config_distance(self):
        # UI -> config | Distance settings #
        if self.comboBox_DistanceMode.currentText() == "Absolute Distance":
            self.beam_dict["distance_mode"] = "direct"
            self.beam_dict["source_distance"] = self.format_float(self.doubleSpinBox_Distance.value() * self.units)
        elif self.comboBox_DistanceMode.currentText() == "Final Beam Width":
            self.beam_dict["distance_mode"] = "width"
            self.beam_dict["final_width"] = self.format_float(self.doubleSpinBox_Distance.value() * self.units)
        elif self.comboBox_DistanceMode.currentText() == "Focal Length Multiple":
            self.beam_dict["distance_mode"] = "focal"
            self.beam_dict["focal_multiple"] = self.doubleSpinBox_Distance.value()

    def update_UI_tilt(self):
        # Config -> UI | Tilt settings #
        self.doubleSpinBox_BeamTiltX.setValue(self.beam_dict["tilt_x"])
        self.doubleSpinBox_BeamTiltY.setValue(self.beam_dict["tilt_y"])

    def update_config_tilt(self):
        # UI -> config | Tilt settings #
        self.beam_dict["tilt_x"] = self.doubleSpinBox_BeamTiltX.value()
        self.beam_dict["tilt_y"] = self.doubleSpinBox_BeamTiltY.value()

    def update_UI_sim(self):
        # Config -> UI | Simulation settings #
        self.doubleSpinBox_PixelSize.setValue(self.sim_dict["pixel_size"] / self.units)
        self.doubleSpinBox_SimWidth.setValue(self.sim_dict["width"] / self.units)
        self.doubleSpinBox_SimHeight.setValue(self.sim_dict["height"] / self.units)

    def update_config_sim(self):
        # UI -> config | Simulation settings #
        self.sim_dict["pixel_size"] = self.format_float(self.doubleSpinBox_PixelSize.value() * self.units)
        self.sim_dict["width"] = self.format_float(self.doubleSpinBox_SimWidth.value() * self.units)
        self.sim_dict["height"] = self.format_float(self.doubleSpinBox_SimHeight.value() * self.units)

    def format_float(self, num):
        # np format_float_scientific() might be the same?
        return float(f"{num:4e}")

    ### Update methods ###

    def update_image_frames(self):
        plt.close("all")

        self.pc_Profile = self.update_frame(
            label=self.label_Profile,
            pc=self.pc_Profile,
            image=self.beam.lens.profile,
            ndim=2,
            mask=False,
        )

        # self.pc_Convergence = self.update_frame(
        #     label=self.label_ProfileMask,
        #     pc=self.pc_ProfileMask,
        #     image=profile_mask_image,
        #     ndim=2,
        #     mask=True,
        # )

    def update_frame(self, label, pc, image, ndim, mask):
        """Helper function for update_image_frames"""
        if label.layout() is None:
            label.setLayout(QtWidgets.QVBoxLayout())
        if pc is not None:
            label.layout().removeWidget(pc)
            pc.deleteLater()

        pc = _ImageCanvas(
            parent=label, image=image, lens=self.beam.lens, ndim=ndim, mask=mask
        )

        label.layout().addWidget(pc)

        return pc

    def live_update_profile(self):
        if self.checkBox_LiveUpdate.isChecked():
            try:
                self.checkBox_LiveUpdate.setChecked(False)
                self.update_beam_dict()
                self.create_beam()
                self.update_UI()
                self.checkBox_LiveUpdate.setChecked(True)
            except Exception as e:
                self.display_error_message(traceback.format_exc())

    ### Window methods ###

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

class _ImageCanvas(FigureCanvasQTAgg, QtWidgets.QWidget):
    def __init__(self, parent=None, image=None, lens=None, ndim=2, mask=False):

        if ndim == 2:
            colorbar_ticks = None
            if mask:
                if image is None:
                    image = np.zeros(shape=lens.profile.shape)
                    colorbar_ticks = [0]
                func_ = utils.plot_array_2D
                thing_to_plot = image
            else:
                func_ = utils.plot_lens_profile_2D
                thing_to_plot = lens

            self.fig = func_(
                thing_to_plot,
                facecolor="#f0f0f0",
                extent=[
                    -lens.diameter / 2,
                    lens.diameter / 2,
                    -lens.length / 2,
                    lens.length / 2,
                ],
                colorbar_ticks=colorbar_ticks,
            )

        elif ndim == 1:
            if mask:
                if image is None:
                    image = np.zeros(shape=lens.profile.shape[0])

                self.fig = plt.figure()
                self.fig.set_facecolor("#f0f0f0")
                gridspec = self.fig.add_gridspec(1, 1)
                axes = self.fig.add_subplot(gridspec[0], title="")
                axes.ticklabel_format(
                    axis="both", style="sci", scilimits=(0, 0), useMathText=True
                )
                axes.locator_params(nbins=4)

                image = axes.plot(image)

            else:
                self.fig = utils.plot_lens_profile_slices(
                    lens=lens, max_height=lens.height, title="", facecolor="#f0f0f0"
                )

        FigureCanvasQTAgg.__init__(self, self.fig)
        self.setParent(parent)


def main():
    """Launch the `piescope_gui` main application window."""
    application = QtWidgets.QApplication([])
    window = GUIBeamCreator()
    application.aboutToQuit.connect(window.disconnect)  # cleanup & teardown
    sys.exit(application.exec_())


if __name__ == "__main__":
    main()
