import os
from pprint import pprint
import sys
import traceback

import lens_simulation.ui.qtdesigner_files.BeamCreator as BeamCreator
import numpy as np
import yaml
from lens_simulation import constants, plotting, utils
from lens_simulation.Lens import Medium
from lens_simulation.beam import generate_beam
from lens_simulation.Simulation import SimulationStage, SimulationParameters, SimulationOptions, propagate_wavefront
from lens_simulation.ui.utils import display_error_message
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt5 import QtCore, QtGui, QtWidgets

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
        self.pc_FinalProfile = None

        self.result = None

        # shift from tilt
        self.t_shift_x = 0
        self.t_shift_y = 0
        self.c_shift_l = 0
        self.c_shift_r = 0

        self.create_new_beam_dict()
        self.create_new_sim_dict()
        self.create_beam()
        self.calculate_final_profile()
        self.update_UI_limits()
        self.update_UI()

        self.setup_connections()

        # self.center_window()
        self.showNormal()

    ### Setup methods ###

    def setup_connections(self):
        self.pushButton_LoadProfile.clicked.connect(self.load_profile)
        self.pushButton_GenerateProfile.clicked.connect(self.create_beam)
        self.pushButton_SaveProfile.clicked.connect(self.save_profile)
        self.pushButton_CalculateFinalProfile.clicked.connect(self.calculate_final_profile)

        self.comboBox_Units.currentIndexChanged.connect(self.update_units)

        # connect each of the lens parameter selectors to update profile in live view
        [
            value.editingFinished.connect(self.live_update_profile)
            for value in self.__dict__.values()
            if value.__class__ is QtWidgets.QDoubleSpinBox
        ]
        [
            value.currentTextChanged.connect(self.live_update_profile)
            for value in self.__dict__.values()
            if value.__class__ is QtWidgets.QComboBox
        ]


    ### Generation methods ###

    def create_new_sim_dict(self):
        # dummy sim
        self.sim_dict = dict()
        self.sim_dict["pixel_size"] = .1 * self.units

        if self.beam_dict["spread"].title() != "Plane":
            self.beam_dict["shape"] = "Circular"

        # if self.beam_dict["shape"].title() == "Circular":
        #     sim_dimensions = max(self.beam_dict["width"] + 2*abs(self.beam_dict["position_x"]), self.beam_dict["width"] + 2*abs(self.beam_dict["position_y"]))
        # else:
        #     sim_dimensions = max(self.beam_dict["width"] + 2*abs(self.beam_dict["position_x"]), self.beam_dict["height"] + 2*abs(self.beam_dict["position_y"]))

        self.sim_dict["width"] = self.beam_dict["width"] + 2 * abs(
            self.beam_dict["position_x"]
        )
        if self.beam_dict["shape"].title() == "Circular":
            self.sim_dict["height"] = self.beam_dict["width"] + 2 * abs(
                self.beam_dict["position_y"]
            )
        else:
            self.sim_dict["height"] = self.beam_dict["height"] + 2 * abs(
                self.beam_dict["position_y"]
            )
        self.sim_dict["wavelength"] = 488.0e-9

    def create_new_beam_dict(self):
        self.beam_dict = dict()
        self.beam_dict["name"] = "Beam"
        self.beam_dict["distance_mode"] = "direct"
        self.beam_dict["spread"] = "plane"
        self.beam_dict["shape"] = "rectangular"
        self.beam_dict["width"] = 50.0e-6
        self.beam_dict["height"] = 50.0e-6
        self.beam_dict["position_x"] = 0.0e-6
        self.beam_dict["position_y"] = 0.0e-6
        self.beam_dict["theta"] = 1.0  # Degrees
        self.beam_dict["numerical_aperture"] = None
        self.beam_dict["tilt_x"] = 0.0
        self.beam_dict["tilt_y"] = 0.0
        self.beam_dict["source_distance"] = 25.0e-6
        self.beam_dict["final_diameter"] = None
        self.beam_dict["focal_multiple"] = None
        # self.beam_dict["n_steps"] = 10
        self.beam_dict["step_size"] = 3.3e-6
        self.beam_dict["output_medium"] = 1.0
        # This only exists because config yaml loading gives it the lens value
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
        try:
            self.beam = generate_beam(config=self.beam_dict, parameters=self.parameters)
            self.update_image_frames()
        except Exception as e:
            display_error_message(traceback.format_exc())

    ### ui <-> Config methods ###

    def update_config(self):
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
        self.update_image_frames()

    def update_UI_general(self):
        # Config -> ui | General settings #
        self.lineEdit_LensName.setText(self.beam_dict["name"])
        if self.beam_dict["n_steps"] != 0:
            self.doubleSpinBox_DistanceMethod.setDecimals(0)
            self.comboBox_DistanceMethod.setCurrentText("# Steps")
            self.doubleSpinBox_DistanceMethod.setValue(self.beam_dict["n_steps"])
        elif self.beam_dict["step_size"] != 0:
            self.comboBox_DistanceMethod.setCurrentText("Step Size")
            self.doubleSpinBox_DistanceMethod.setDecimals(2)
            self.doubleSpinBox_DistanceMethod.setValue(
                self.beam_dict["step_size"] / self.units
            )
        self.doubleSpinBox_ShiftX.setValue(self.beam_dict["position_x"] / self.units)
        self.doubleSpinBox_ShiftY.setValue(self.beam_dict["position_y"] / self.units)
        self.doubleSpinBox_Width.setValue(self.beam_dict["width"] / self.units)
        self.doubleSpinBox_Height.setValue(self.beam_dict["height"] / self.units)
        self.label_Height.setEnabled(self.beam_dict["shape"].title() != "Circular")
        self.doubleSpinBox_Height.setEnabled(
            self.beam_dict["shape"].title() != "Circular"
        )
        self.doubleSpinBox_OutputMedium.setValue(self.beam_dict["output_medium"])

    def update_config_general(self):
        # ui -> config | General settings #
        self.beam_dict["name"] = self.lineEdit_LensName.text()
        if self.comboBox_DistanceMethod.currentText() == "# Steps":
            self.beam_dict["n_steps"] = round(self.doubleSpinBox_DistanceMethod.value())
            self.beam_dict["step_size"] = 0
        else:
            self.beam_dict["n_steps"] = 0
            self.beam_dict["step_size"] = (
                self.doubleSpinBox_DistanceMethod.value() * self.units
            )

        self.beam_dict["position_x"] = self.format_float(
            self.doubleSpinBox_ShiftX.value() * self.units
        )
        self.beam_dict["position_y"] = self.format_float(
            self.doubleSpinBox_ShiftY.value() * self.units
        )
        self.beam_dict["width"] = self.format_float(
            self.doubleSpinBox_Width.value() * self.units
        )
        self.beam_dict["height"] = self.format_float(
            self.doubleSpinBox_Height.value() * self.units
        )
        self.beam_dict["output_medium"] = self.doubleSpinBox_OutputMedium.value()

    def update_UI_beam_spread(self):
        # Config -> ui | Beam Spread settings #
        self.comboBox_BeamSpread.setCurrentIndex(
            beam_spread_dict[self.beam_dict["spread"].title()]
        )

    def update_config_beam_spread(self):
        # ui -> config | Beam Spread settings #
        if self.comboBox_BeamSpread.currentText() == "Planar":
            self.beam_dict["spread"] = "plane"
        if self.comboBox_BeamSpread.currentText() == "Converging":
            self.beam_dict["spread"] = "converging"
        if self.comboBox_BeamSpread.currentText() == "Diverging":
            self.beam_dict["spread"] = "diverging"

    def update_UI_beam_shape(self):
        # Config -> ui | Beam Shape settings #
        self.comboBox_BeamShape.clear()
        if self.beam_dict["spread"].title() == "Plane":
            self.comboBox_BeamShape.addItem("Circular")
            self.comboBox_BeamShape.addItem("Rectangular")
            self.comboBox_BeamShape.setCurrentIndex(
                beam_shape_dict[self.beam_dict["shape"].title()]
            )
        else:
            self.comboBox_BeamShape.addItem("Circular")
            self.beam_dict["beam_shape"] = "Circular"
            self.comboBox_BeamShape.setCurrentIndex(0)

    def update_config_beam_shape(self):
        # ui -> config | Beam Shape settings #
        if self.comboBox_BeamSpread.currentText() != "Planar":
            self.beam_dict["shape"] = "circular"
        else:
            self.beam_dict["shape"] = self.comboBox_BeamShape.currentText()

    def update_UI_convergence_angle(self):
        # Config -> ui | Angle settings #
        if self.beam_dict["spread"].title() == "Plane":
            self.frame_BeamAngle.setEnabled(False)
        else:
            self.frame_BeamAngle.setEnabled(True)

        if self.beam_dict["theta"] is not None and self.beam_dict["theta"] != 0.0:
            self.comboBox_BeamAngle.setCurrentText("Theta")
            self.doubleSpinBox_BeamAngle.setValue(self.beam_dict["theta"])
            return

        self.comboBox_BeamAngle.setCurrentText("Numerical Aperture")
        self.doubleSpinBox_BeamAngle.setValue(self.beam_dict["numerical_aperture"])

    def update_config_convergence_angle(self):
        # ui -> config | Angle settings #
        if self.comboBox_BeamAngle.currentText() == "Numerical Aperture":
            self.beam_dict["theta"] = 0.0
            self.beam_dict["numerical_aperture"] = self.doubleSpinBox_BeamAngle.value()
            return

        self.beam_dict["theta"] = self.doubleSpinBox_BeamAngle.value()
        self.beam_dict["numerical_aperture"] = 0.0

    def update_UI_distance(self):
        # Config -> ui | Distance settings #

        self.comboBox_DistanceMode.clear()
        if self.beam_dict["spread"].title() == "Plane":
            self.comboBox_DistanceMode.addItem("Absolute Distance")
        else:
            self.comboBox_DistanceMode.addItem("Absolute Distance")
            self.comboBox_DistanceMode.addItem("Final Beam Diameter")
            self.comboBox_DistanceMode.addItem("Focal Length Multiple")

        if self.beam_dict["distance_mode"].title() == "Direct":
            self.comboBox_DistanceMode.setCurrentText("Absolute Distance")
            self.doubleSpinBox_Distance.setValue(
                self.beam_dict["source_distance"] / self.units
            )
        elif self.beam_dict["distance_mode"].title() == "Diameter":
            self.comboBox_DistanceMode.setCurrentText("Final Beam Diameter")
            self.doubleSpinBox_Distance.setValue(
                self.beam_dict["final_diameter"] / self.units
            )
        elif self.beam_dict["distance_mode"].title() == "Focal":
            self.comboBox_DistanceMode.setCurrentText("Focal Length Multiple")
            self.doubleSpinBox_Distance.setValue(self.beam_dict["focal_multiple"])

    def update_config_distance(self):
        # ui -> config | Distance settings #
        if self.comboBox_DistanceMode.currentText() == "Absolute Distance":
            self.beam_dict["distance_mode"] = "direct"
            self.beam_dict["source_distance"] = self.format_float(
                self.doubleSpinBox_Distance.value() * self.units
            )
        elif self.comboBox_DistanceMode.currentText() == "Final Beam Diameter":
            self.beam_dict["distance_mode"] = "Diameter"
            self.beam_dict["final_diameter"] = self.format_float(
                self.doubleSpinBox_Distance.value() * self.units
            )
        elif self.comboBox_DistanceMode.currentText() == "Focal Length Multiple":
            self.beam_dict["distance_mode"] = "focal"
            self.beam_dict["focal_multiple"] = self.doubleSpinBox_Distance.value()

    def update_UI_tilt(self):
        # Config -> ui | Tilt settings #
        self.doubleSpinBox_BeamTiltX.setValue(self.beam_dict["tilt_x"])
        self.doubleSpinBox_BeamTiltY.setValue(self.beam_dict["tilt_y"])

    def update_config_tilt(self):
        # ui -> config | Tilt settings #
        self.beam_dict["tilt_x"] = self.doubleSpinBox_BeamTiltX.value()
        self.beam_dict["tilt_y"] = self.doubleSpinBox_BeamTiltY.value()

    def update_UI_sim(self):
        # Config -> ui | Simulation settings #
        self.doubleSpinBox_PixelSize.setValue(self.sim_dict["pixel_size"] / self.units)
        self.doubleSpinBox_SimWidth.setValue(self.sim_dict["width"] / self.units)
        self.doubleSpinBox_SimHeight.setValue(self.sim_dict["height"] / self.units)

    def update_config_sim(self):
        # ui -> config | Simulation settings #
        self.sim_dict["pixel_size"] = self.format_float(
            self.doubleSpinBox_PixelSize.value() * self.units
        )
        self.sim_dict["width"] = self.format_float(
            self.doubleSpinBox_SimWidth.value() * self.units
        )
        self.sim_dict["height"] = self.format_float(
            self.doubleSpinBox_SimHeight.value() * self.units
        )

    def update_UI_limits(self):

        pixel_size = self.sim_dict["pixel_size"]
        pixel_size_units = pixel_size / self.units

        self.doubleSpinBox_Width.setMinimum(1 * pixel_size_units)
        self.doubleSpinBox_Height.setMinimum(1 * pixel_size_units)
        self.doubleSpinBox_Width.setMaximum(
            (self.sim_dict["width"] - 2 * abs(self.beam_dict["position_x"]))
            / self.units
        )
        self.doubleSpinBox_Height.setMaximum(
            (self.sim_dict["height"] - 2 * abs(self.beam_dict["position_y"]))
            / self.units
        )

        self.doubleSpinBox_SimWidth.setMinimum(
            (self.beam_dict["width"] + 2 * abs(self.beam_dict["position_x"]))
            / self.units
        )
        # TODO: check how to make rectangular loading smart too
        if self.beam_dict["shape"].title() == "Circular":
            self.doubleSpinBox_SimHeight.setMinimum(
                (self.beam_dict["width"] + 2 * abs(self.beam_dict["position_y"]))
                / self.units
            )
        else:
            self.doubleSpinBox_SimHeight.setMinimum(
                (self.beam_dict["height"] + 2 * abs(self.beam_dict["position_y"]))
                / self.units
            )

        self.doubleSpinBox_ShiftX.setMaximum(
            (self.sim_dict["width"] - self.beam_dict["width"]) / 2 / self.units
        )
        self.doubleSpinBox_ShiftX.setMinimum(
            -(self.sim_dict["width"] - self.beam_dict["width"]) / 2 / self.units
        )

        if self.beam_dict["shape"].title() == "Circular":
            self.doubleSpinBox_ShiftY.setMaximum(
                (self.sim_dict["height"] - self.beam_dict["width"]) / 2 / self.units
            )
            self.doubleSpinBox_ShiftY.setMinimum(
                -(self.sim_dict["height"] - self.beam_dict["width"]) / 2 / self.units
            )
        else:
            self.doubleSpinBox_ShiftY.setMaximum(
                (self.sim_dict["height"] - self.beam_dict["height"]) / 2 / self.units
            )
            self.doubleSpinBox_ShiftY.setMinimum(
                -(self.sim_dict["height"] - self.beam_dict["height"]) / 2 / self.units
            )

        if self.beam_dict["distance_mode"].title() == "Diameter":
            if self.beam_dict["spread"].title() == "Diverging":
                self.doubleSpinBox_Distance.setMinimum(
                    self.beam_dict["width"] / self.units
                )
                self.doubleSpinBox_Distance.setMaximum(99999)
            elif self.beam_dict["spread"].title() == "Converging":
                self.doubleSpinBox_Distance.setMinimum(0)
                self.doubleSpinBox_Distance.setMaximum(
                    self.beam_dict["width"] / self.units
                )
        else:
            self.doubleSpinBox_Distance.setMinimum(0)
            self.doubleSpinBox_Distance.setMaximum(99999)

        if self.beam_dict["spread"].title() == "Converging":
            pass

    def format_float(self, num):
        # np format_float_scientific() might be the same?
        return float(f"{num:4e}")

    ### I/O methods ###

    def load_profile(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Profile", filter="Yaml config (*.yml *.yaml)",
        )

        if filename is "":
            return

        # get the status of live update to restore it post loading
        was_live = self.checkBox_LiveUpdate.isChecked()
        try:
            # turn off live update to avoid memory issues
            self.checkBox_LiveUpdate.setChecked(False)
            # TODO: check how to validate for beam
            self.beam_dict = utils.load_yaml_config(filename)
            self.create_new_sim_dict()
            self.create_beam()
            self.update_UI_limits()
            self.update_UI()

            self.checkBox_LiveUpdate.setChecked(was_live)
        except Exception as e:
            display_error_message(traceback.format_exc())

    def save_profile(self):
        filename, ext = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Profile",
            self.beam_dict["name"],
            filter="Yaml config (*.yml *.yaml)",
        )

        if filename == "":
            return

        self.beam_dict["name"] = os.path.basename(filename).split(".")[0]
        self.lineEdit_LensName.setText(self.beam_dict["name"])

        with open(filename, "w") as f:
            yaml.safe_dump(self.beam_dict, f, sort_keys=False)

    ### Update methods ###

    def update_image_frames(self):
        plt.close("all")

        self.calculate_tilt_effect()
        self.calculate_convergence_effect()

        pos = [self.beam_dict["position_x"], self.beam_dict["position_y"]]

        self.pc_Profile = self.update_frame(
            label=self.label_Profile,
            pc=self.pc_Profile,
            profile="main",
            points=[self.t_shift_x, self.t_shift_y],
            position=pos,
            units=self.units
        )

        axis = 0

        if self.comboBox_ConvergenceDisplay.currentText() == "X":
            convergence_array = np.zeros(shape=(100, self.beam.lens.profile.shape[1]))
            axis = 0
        else:
            convergence_array = np.zeros(shape=(100, self.beam.lens.profile.shape[0]))
            axis = 1

        self.pc_Convergence = self.update_frame(
            label=self.label_Convergence,
            pc=self.pc_Convergence,
            profile="convergence",
            points=[self.c_shift_l],
            position=pos,
            array=convergence_array,
            axis=axis
        )

        if self.result is not None:
            self.pc_FinalProfile = self.update_frame(
                label=self.label_FinalProfile,
                pc=self.pc_FinalProfile,
                profile="final",
                points=[self.c_shift_l],
                position=pos,
                array=self.result.sim[-1],
                axis=axis
            )



    def update_frame(self, label, pc, profile, points, position, array=None, axis=0, units=1e-6):
        """Helper function for update_image_frames"""
        if label.layout() is None:
            label.setLayout(QtWidgets.QVBoxLayout())
        if pc is not None:
            label.layout().removeWidget(pc)
            pc.deleteLater()

        pc = _ImageCanvas(
            parent=label,
            lens=self.beam.lens,
            profile=profile,
            points=points,
            position=position,
            array=array,
            axis=axis,
            units=units
        )

        label.layout().addWidget(pc)

        return pc

    def update_units(self):
        old_units = self.units

        self.units = units_dict[self.comboBox_Units.currentIndex()]

        unit_conversion = self.units / old_units

        self.sim_dict["pixel_size"] *= unit_conversion
        self.sim_dict["width"] *= unit_conversion
        self.sim_dict["height"] *= unit_conversion

        self.beam_dict["width"] *= unit_conversion
        self.beam_dict["height"] *= unit_conversion
        self.beam_dict["position_x"] *= unit_conversion
        self.beam_dict["position_y"] *= unit_conversion
        self.beam_dict["source_distance"] *= unit_conversion

    def live_update_profile(self):
        if self.checkBox_LiveUpdate.isChecked():
            try:
                self.checkBox_LiveUpdate.setChecked(False)
                self.update_config()
                self.create_beam()
                self.update_UI_limits()
                self.update_config()
                self.create_beam()
                self.update_UI()
                self.update_image_frames()
                self.checkBox_LiveUpdate.setChecked(True)
            except Exception as e:
                display_error_message(traceback.format_exc())

    ### Calculation methods ###
    def calculate_tilt_effect(self):
        finish_distance = self.beam.calculate_propagation_distance()[1]

        self.t_shift_x = 0
        self.t_shift_y = 0

        if self.beam_dict["tilt_x"] != 0:
            self.t_shift_x = finish_distance * np.tan(
                np.deg2rad(self.beam_dict["tilt_x"])
            )
        if self.beam_dict["tilt_y"] != 0:
            self.t_shift_y = finish_distance * np.tan(
                np.deg2rad(self.beam_dict["tilt_y"])
            )

    def calculate_convergence_effect(self):
        finish_distance = self.beam.calculate_propagation_distance()[1]

        self.c_shift_l = 0
        self.c_shift_yr = 0

        if self.beam_dict["spread"].title() == "Plane":
            return

        if self.beam.theta != 0:
            self.c_shift_l = finish_distance * np.tan(self.beam.theta)
            self.c_shift_r = -self.c_shift_l

        if self.beam_dict["spread"].title() == "Diverging":
            self.c_shift_l *= -1
            self.c_shift_r *= -1

    def calculate_final_profile(self):

        from lens_simulation.Simulation import calculate_propagation_distances
        distances = calculate_propagation_distances(
            self.beam.calculate_propagation_distance()[0],
            self.beam.calculate_propagation_distance()[1],
            n_steps=2
        )

        stage = SimulationStage(lens=self.beam.lens,
                                output=Medium(self.beam_dict["output_medium"]),
                                distances=distances,
                                tilt={"x":self.beam_dict["tilt_x"], "y":self.beam_dict["tilt_y"]},
                                )

        parameters = SimulationParameters(A=10000,
                                          pixel_size=self.sim_dict["pixel_size"],
                                          sim_width=self.sim_dict["width"],
                                          sim_height=self.sim_dict["height"],
                                          sim_wavelength=self.sim_dict["wavelength"])

        options = SimulationOptions(log_dir='', save=False, save_plot=False)

        self.result = propagate_wavefront(stage=stage, parameters=parameters, options=options)

        self.update_image_frames()

    ### Window methods ###

    def center_window(self):
        """Centers the window in the display"""
        # Get the desktop dimensions
        desktop = QtWidgets.QDesktopWidget()
        self.move(
            (desktop.width() - self.width()) / 2,
            (desktop.height() - self.height()) / 3.0,
        )


class _ImageCanvas(FigureCanvasQTAgg, QtWidgets.QWidget):
    def __init__(
        self, parent=None, profile=None, lens=None, points=None, position=[0, 0], array=None, axis=0, units=1e-6
    ):

        if profile == "main":
            self.fig = plotting.plot_lens_profile_2D(
                lens,
                facecolor="#f0f0f0",
                extent=[
                    -lens.profile.shape[1] / 2 * lens.pixel_size,
                    lens.profile.shape[1] / 2 * lens.pixel_size,
                    -lens.profile.shape[0] / 2 * lens.pixel_size,
                    lens.profile.shape[0] / 2 * lens.pixel_size,
                ],
                colorbar_ticks=None,
            )
            axes = self.fig.axes[0]
            # axes.plot(
            #     [position[0], position[0] + points[0]],
            #     (-position[1], -position[1] + points[1]),
            #     "r",
            # )
            arrow = axes.arrow(
                x=position[0],
                y=-position[1],
                dx=points[0],
                dy=-points[1],
                width=1*units,
                length_includes_head=True,
            )
            arrow.set_label("Tilt Direction and Shift")
            axes.legend()

        elif profile == "convergence":
            self.fig = plotting.plot_array_2D(
                array=array,
                facecolor="#f0f0f0",
                extent=[
                    -lens.profile.shape[1-axis] / 2 * lens.pixel_size,
                    lens.profile.shape[1-axis] / 2 * lens.pixel_size,
                    100,
                    0,
                ],
                title="",
                colorbar_ticks=None
            )
            axes = self.fig.axes[0]
            axes.plot(
                [position[axis]-lens.diameter/2, position[axis]-lens.diameter/2 + points[0]],
                (0, 100),
                "k",
                linewidth=2
            )
            axes.plot(
                [position[axis]+lens.diameter/2, position[axis]+lens.diameter/2 - points[0]],
                (0, 100),
                "k",
                linewidth=2
            )
            # DO

        else:
            self.fig = plotting.plot_array_2D(
                array=array,
                facecolor="#f0f0f0",
                extent=[
                    -lens.profile.shape[1] / 2 * lens.pixel_size,
                    lens.profile.shape[1] / 2 * lens.pixel_size,
                    -lens.profile.shape[0] / 2 * lens.pixel_size,
                    lens.profile.shape[0] / 2 * lens.pixel_size,
                ],
                title="",
                colorbar_ticks=None
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
