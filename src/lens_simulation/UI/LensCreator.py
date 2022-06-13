import sys
import traceback

import lens_simulation.UI.qtdesigner_files.LensCreator as LensCreator
import numpy as np
from lens_simulation.Lens import (
    GratingSettings,
    Lens,
    LensType,
    Medium,
    generate_lens,
    apply_modifications,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib import pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets
from lens_simulation import constants, utils, validation

lens_type_dict = {"Cylindrical": LensType.Cylindrical, "Spherical": LensType.Spherical}

# maps the index of comboboxes to a constant
units_dict = {
    0: constants.NANO_TO_METRE,
    1: constants.MICRON_TO_METRE,
    2: constants.MM_TO_METRE,
}


class GUILensCreator(LensCreator.Ui_LensCreator, QtWidgets.QMainWindow):
    def __init__(self, parent_gui=None):
        super().__init__(parent=parent_gui)
        self.setupUi(LensCreator=self)
        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)
        self.DEBUG = False

        # default to um
        self.comboBox_Units.setCurrentIndex(1)

        # set up of image frames
        self.pc_CrossSection = None
        self.pc_CrossSectionMask = None
        self.pc_Profile = None
        self.pc_ProfileMask = None

        self.units = units_dict[1]

        # creat initial lens config
        self.create_initial_lens_dict()
        self.create_base_lens()
        self.update_UI()
        self.update_UI_limits()
        self.generate_profile()

        self.setup_connections()

        self.center_window()
        self.showNormal()

    ### Setup methods ###

    def setup_connections(self):
        self.pushButton_LoadProfile.clicked.connect(self.load_profile)
        self.pushButton_GenerateProfile.clicked.connect(self.generate_profile)
        self.pushButton_SaveProfile.clicked.connect(self.save_profile)

        self.comboBox_Units.currentIndexChanged.connect(self.update_units)

        # connect each of the lens parameter selectors to update profile in live view
        [
            value.valueChanged.connect(self.live_update_profile)
            for value in self.__dict__.values()
            if value.__class__ is QtWidgets.QDoubleSpinBox
        ]
        [
            value.toggled.connect(self.live_update_profile)
            for value in self.__dict__.values()
            if value.__class__ in [QtWidgets.QCheckBox, QtWidgets.QGroupBox]
        ]
        [
            value.currentTextChanged.connect(self.live_update_profile)
            for value in self.__dict__.values()
            if value.__class__ is QtWidgets.QComboBox
        ]
        [
            value.currentChanged.connect(self.live_update_profile)
            for value in self.__dict__.values()
            if value.__class__ is QtWidgets.QTabWidget
        ]

        # this removes the live update checkbox triggering live_update_profile
        self.checkBox_LiveUpdate.toggled.disconnect()

    ### Generation methods ###

    def generate_profile(self):
        """Generates a profile based on the inputs to the GUI"""
        # generate the lens based off the parameters selected in GUI
        # TODO: move this?

        # self.update_units()

        self.frame_TruncationAperture.setEnabled(
            self.lens_dict["truncation"] is not None
            and self.lens_dict["aperture"] is not None
            # self.groupBox_Truncation.isChecked() and self.groupBox_Aperture.isChecked()
        )

        try:
            # self.update_profile_parameters()
            self.create_base_lens()

            self.update_masks()

            # TODO: apply modifications requires sim
            # self.lens = apply_modifications(self.lens, self.lens_dict, parameters=None)
            self.lens.apply_masks(
                grating=self.lens_dict["grating"] is not None,
                truncation=self.lens_dict["truncation"] is not None,
                aperture=True#self.lens_dict["aperture"] is not None,
            )

            if self.lens_dict["inverted"]:  # checkBox_InvertedProfile.isChecked():
                self.lens.invert_profile()

            self.update_image_frames()

        except Exception as e:
            self.display_error_message(traceback.format_exc())

    # def generate_base_lens(self):
    #     self.lens = Lens(
    #         diameter=self.doubleSpinBox_LensDiameter.value() * self.units,
    #         height=self.doubleSpinBox_LensHeight.value() * self.units,
    #         exponent=self.doubleSpinBox_LensExponent.value(),
    #         medium=self.doubleSpinBox_LensMedium.value(),
    #         lens_type=self.lens_type,
    #     )

    def create_base_lens(self):
        # generate Lens object from self.lens_dict
        # TODO: replace with generate_lens function
        self.lens = Lens(
            diameter=self.lens_dict["diameter"],
            height=self.lens_dict["height"],
            exponent=self.lens_dict["exponent"],
            medium=Medium(self.lens_dict["medium"]),
            lens_type=self.lens_dict["lens_type"],
        )
        self.lens.generate_profile(
            pixel_size=self.lens_dict["pixel_size"], length=self.lens_dict["length"],
        )

    def create_initial_lens_dict(self, filename=None):
        self.lens_dict = dict()
        self.lens_dict["name"] = "Lens"
        self.lens_dict["medium"] = 2.348
        self.lens_dict["exponent"] = 2.0
        self.lens_dict["diameter"] = 100.0e-6
        self.lens_dict["lens_type"] = LensType.Spherical
        self.lens_dict["length"] = self.lens_dict["diameter"]
        self.lens_dict["height"] = 10.0e-6
        self.lens_dict["pixel_size"] = 0.1e-6
        self.lens_dict["custom"] = filename
        self.lens_dict["inverted"] = False
        self.lens_dict["escape_path"] = None
        self.lens_dict["grating"] = None
        self.lens_dict["truncation"] = None
        self.lens_dict["aperture"] = None

        if self.DEBUG:
            self.lens_dict["grating"] = dict()
            self.lens_dict["grating"]["x"] = True
            self.lens_dict["grating"]["y"] = False
            self.lens_dict["grating"]["width"] = 1.0e-6
            self.lens_dict["grating"]["distance"] = 2.0e-6
            self.lens_dict["grating"]["depth"] = 3.0e-6
            self.lens_dict["grating"]["centred"] = True

            self.lens_dict["truncation"] = dict()
            self.lens_dict["truncation"]["height"] = 9.0e-6
            self.lens_dict["truncation"]["radius"] = 19.0e-6
            self.lens_dict["truncation"]["type"] = "height"
            self.lens_dict["truncation"]["aperture"] = False

            self.lens_dict["aperture"] = dict()
            self.lens_dict["aperture"]["inner"] = 20.0e-6
            self.lens_dict["aperture"]["outer"] = 40.0e-6
            self.lens_dict["aperture"]["type"] = "radial"
            self.lens_dict["aperture"]["invert"] = False

    ### UI <-> Config methods ###

    def update_lens_dict(self):
        """Helper function to update full config"""
        self.update_config_general()
        self.update_config_grating()
        self.update_config_truncation()
        self.update_config_aperture()

    def update_UI(self):
        """Helper function to update full UI"""
        self.update_UI_general()
        self.update_UI_grating()
        self.update_UI_truncation()
        self.update_UI_aperture()

    def update_UI_general(self):
        # Config -> UI | General settings #
        self.lineEdit_LensName.setText(self.lens_dict["name"])
        self.doubleSpinBox_LensMedium.setValue(self.lens_dict["medium"])
        self.doubleSpinBox_LensExponent.setValue(self.lens_dict["exponent"])
        self.comboBox_LensType.setCurrentText(self.lens_dict["lens_type"].name)
        self.doubleSpinBox_PixelSize.setValue(self.lens_dict["pixel_size"] / self.units)
        self.doubleSpinBox_LensDiameter.setValue(
            self.lens_dict["diameter"] / self.units
        )
        # set length to diameter for spherical lenses
        if self.lens_dict["lens_type"] is LensType.Spherical:
            self.doubleSpinBox_LensLength.setValue(
                self.lens_dict["diameter"] / self.units
            )
            self.frame_LensLength.setEnabled(False)
        else:
            self.doubleSpinBox_LensLength.setValue(
                self.lens_dict["length"] / self.units
            )
            self.frame_LensLength.setEnabled(True)
            self.doubleSpinBox_LensLength.setEnabled(True)
            self.doubleSpinBox_LensLength.setValue(12)

        self.doubleSpinBox_LensHeight.setValue(self.lens_dict["height"] / self.units)
        if self.lens_dict["escape_path"] is not None:
            self.doubleSpinBox_LensEscapePath.setValue(self.lens_dict["escape_path"])
        else:
            self.doubleSpinBox_LensEscapePath.setValue(0.0)
        if self.lens_dict["inverted"] is not None:
            self.checkBox_InvertedProfile.setChecked(self.lens_dict["inverted"])
        else:
            self.checkBox_InvertedProfile.setChecked(False)

    def update_config_general(self):
        # UI -> Config | General settings #
        self.lens_dict["name"] = self.lineEdit_LensName.text()
        self.lens_dict["medium"] = self.doubleSpinBox_LensMedium.value()
        self.lens_dict["exponent"] = self.doubleSpinBox_LensExponent.value()
        self.lens_dict["lens_type"] = LensType[self.comboBox_LensType.currentText()]
        self.lens_dict["escape_path"] = self.doubleSpinBox_LensEscapePath.value()
        self.lens_dict["inverted"] = self.checkBox_InvertedProfile.isChecked()

        self.lens_dict["diameter"] = self.format_float(
            self.doubleSpinBox_LensDiameter.value() * self.units
        )
        self.lens_dict["length"] = self.format_float(
            self.doubleSpinBox_LensLength.value() * self.units
        )
        self.lens_dict["height"] = self.format_float(
            self.doubleSpinBox_LensHeight.value() * self.units
        )
        self.lens_dict["pixel_size"] = self.format_float(
            self.doubleSpinBox_PixelSize.value() * self.units
        )

    def update_UI_grating(self):
        # Config -> UI | Grating settings #
        if self.lens_dict["grating"] is None:
            self.groupBox_Gratings.setChecked(False)
            return

        self.groupBox_Gratings.setChecked(True)
        self.doubleSpinBox_GratingWidth.setValue(
            self.lens_dict["grating"]["width"] / self.units
        )
        self.doubleSpinBox_GratingDistance.setValue(
            self.lens_dict["grating"]["distance"] / self.units
        )
        self.doubleSpinBox_GratingDepth.setValue(
            self.lens_dict["grating"]["depth"] / self.units
        )

        # TODO: change to 2 checkboxes
        self.checkBox_GratingDirectionX.setChecked(self.lens_dict["grating"]["x"])
        self.checkBox_GratingDirectionY.setChecked(self.lens_dict["grating"]["y"])
        self.checkBox_GratingCentred.setChecked(self.lens_dict["grating"]["centred"])

    def update_config_grating(self):
        # UI -> Config | Grating Settings #
        if not self.groupBox_Gratings.isChecked():
            self.lens_dict["grating"] = None
            return
        self.lens_dict["grating"] = dict()
        self.lens_dict["grating"]["x"] = self.checkBox_GratingDirectionX.isChecked()
        self.lens_dict["grating"]["y"] = self.checkBox_GratingDirectionY.isChecked()
        self.lens_dict["grating"]["width"] = self.format_float(
            self.doubleSpinBox_GratingWidth.value() * self.units
        )
        self.lens_dict["grating"]["distance"] = self.format_float(
            self.doubleSpinBox_GratingDistance.value() * self.units
        )
        self.lens_dict["grating"]["depth"] = self.format_float(
            self.doubleSpinBox_GratingDepth.value() * self.units
        )
        self.lens_dict["grating"]["centred"] = self.checkBox_GratingCentred.isChecked()

    def update_UI_truncation(self):
        # Config -> UI | Truncation Settings #
        if self.lens_dict["truncation"] is None:
            self.groupBox_Truncation.setChecked(False)
            return

        self.groupBox_Truncation.setChecked(True)
        self.doubleSpinBox_TruncationRadius.setValue(
            self.lens_dict["truncation"]["radius"] / self.units
        )
        self.doubleSpinBox_TruncationValue.setValue(
            self.lens_dict["truncation"]["height"] / self.units
        )

        if self.lens_dict["truncation"]["type"] == "radial":
            self.comboBox_TruncationMode.setCurrentText("Radius")
        else:
            self.comboBox_TruncationMode.setCurrentText("Height")

        self.checkBox_TruncationAperture.setChecked(
            self.lens_dict["truncation"]["aperture"]
        )

    def update_config_truncation(self):
        # UI -> Config | Truncation Settings #
        if not self.groupBox_Truncation.isChecked():
            self.lens_dict["truncation"] = None
            return
        self.lens_dict["truncation"] = dict()
        self.lens_dict["truncation"]["height"] = self.format_float(
            self.doubleSpinBox_TruncationValue.value() * self.units
        )
        self.lens_dict["truncation"]["radius"] = self.format_float(
            self.doubleSpinBox_TruncationRadius.value() * self.units
        )
        self.lens_dict["truncation"][
            "aperture"
        ] = self.checkBox_TruncationAperture.isChecked()
        if self.comboBox_TruncationMode.currentText() == "Radius":
            self.lens_dict["truncation"]["type"] = "radial"
        else:
            self.lens_dict["truncation"]["type"] = "value"

    def update_UI_aperture(self):
        # Config -> UI | Aperture settings #
        if self.lens_dict["aperture"] is None:
            self.groupBox_Aperture.setChecked(False)
            return

        self.groupBox_Aperture.setChecked(True)
        self.checkBox_ApertureInverted.setChecked(self.lens_dict["aperture"]["invert"])

        if self.lens_dict["truncation"]["type"] == "radial":
            self.comboBox_ApertureMode.setCurrentText("Circle")
        else:
            self.comboBox_ApertureMode.setCurrentText("Square")

        self.doubleSpinBox_ApertureInner.setValue(self.lens_dict["aperture"]["inner"])
        self.doubleSpinBox_ApertureOuter.setValue(self.lens_dict["aperture"]["outer"])

    def update_config_aperture(self):
        # UI -> Config | Aperture Settings #
        if not self.groupBox_Aperture.isChecked():
            self.lens_dict["aperture"] = None
            return
        self.lens_dict["aperture"] = dict()
        self.lens_dict["aperture"]["inner"] = self.format_float(
            self.doubleSpinBox_ApertureInner.value() * self.units
        )
        self.lens_dict["aperture"]["outer"] = self.format_float(
            self.doubleSpinBox_ApertureOuter.value() * self.units
        )
        self.lens_dict["aperture"][
            "invert"
        ] = self.checkBox_ApertureInverted.isChecked()
        if self.comboBox_ApertureMode.currentText() == "Circle":
            self.lens_dict["aperture"]["type"] = "radial"
        else:
            self.lens_dict["aperture"]["type"] = "square"

    def update_UI_limits(self):
        """Method to update limits all at once from dict"""
        self.doubleSpinBox_LensDiameter.setMinimum(
            2 * self.lens_dict["pixel_size"] / self.units
        )
        self.doubleSpinBox_LensLength.setMinimum(
            1 * self.lens_dict["pixel_size"] / self.units
        )

        self.doubleSpinBox_GratingDistance.setMinimum(
            2 * self.lens_dict["pixel_size"] / self.units
        )
        self.doubleSpinBox_GratingDistance.setMaximum(
            (self.lens_dict["diameter"] - self.lens_dict["pixel_size"]) / self.units
        )

        self.doubleSpinBox_GratingWidth.setMinimum(
            1 * self.lens_dict["pixel_size"] / self.units
        )
        # use other doubleSpinbox value to set mins as lens_dict property will not exist
        self.doubleSpinBox_GratingWidth.setMaximum(
            (
                self.doubleSpinBox_GratingDistance.value() * self.units
                - self.lens_dict["pixel_size"]
            )
            / self.units
        )

        self.doubleSpinBox_TruncationValue.setMinimum(
            self.lens_dict["pixel_size"] / self.units
        )
        self.doubleSpinBox_TruncationValue.setMaximum(
            self.lens_dict["height"] / self.units
        )

        self.doubleSpinBox_TruncationRadius.setMinimum(
            self.lens_dict["pixel_size"] / self.units
        )
        self.doubleSpinBox_TruncationRadius.setMaximum(
            ((self.lens_dict["diameter"] / 2) - self.lens_dict["pixel_size"])
            / self.units
        )

        self.doubleSpinBox_ApertureOuter.setMinimum(
            self.lens_dict["pixel_size"] * 2 / self.units
        )

        self.doubleSpinBox_ApertureOuter.setMaximum(
            ((self.lens_dict["diameter"] / 2) - self.lens_dict["pixel_size"])
            / self.units
        )

        self.doubleSpinBox_ApertureInner.setMinimum(
            self.lens_dict["pixel_size"] / self.units
        )

        # use other doubleSpinbox value to set mins as lens_dict property will not exist
        self.doubleSpinBox_ApertureInner.setMaximum(
            (
                self.doubleSpinBox_ApertureOuter.value() / self.units
                - self.lens_dict["pixel_size"]
            )
            / self.units
        )

    def format_float(self, num):
        # np format_float_scientific() might be the same?
        return float(f"{num:4e}")

    ### I/O methods ###

    def load_profile2(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Profile",
            filter="All types (*.yml *.yaml *.npy) ;;Yaml config (*.yml *.yaml) ;;Numpy array (*.npy)",
        )

        if filename is "":
            return

        if filename.endswith(".npy"):
            self.create_lens_dict(filename)

    def load_profile(self):
        """Loads a custom lens profile (numpy.ndarray) through Qt's file opening system"""
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Profile",
            filter="All types (*.yml *.yaml *.npy) ;;Yaml config (*.yml *.yaml) ;;Numpy array (*.npy)",
        )

        if filename is "":
            return

        if filename.endswith(".npy"):
            self.create_base_lens()
            self.lens.load_profile(
                fname=filename,
                pixel_size=self.doubleSpinBox_PixelSize.value() * self.units,
            )

            self.update_image_frames()
            self.checkBox_LiveUpdate.setChecked(False)

        elif filename.endswith((".yml", ".yaml")):
            try:
                lens_config = utils.load_yaml_config(filename)
                self.lens = generate_lens(
                    lens_config=lens_config, medium=Medium(lens_config["medium"])
                )
            except Exception as e:
                self.display_error_message(traceback.format_exc())

            print(lens_config)

            # get the status of live update to restore it post loading
            was_live = self.checkBox_LiveUpdate.isChecked()

            # turn off live update to avoid memory issues
            self.checkBox_LiveUpdate.setChecked(False)

            self.comboBox_LensType.setCurrentText(self.lens.lens_type.name)

            self.doubleSpinBox_LensDiameter.setValue(self.lens.diameter / self.units)

            # default loaded generating profiles to be 1000x1000 pixels
            self.doubleSpinBox_PixelSize.setValue(self.lens.diameter / self.units / 50)
            self.doubleSpinBox_LensHeight.setValue(self.lens.height / self.units)

            if self.lens.lens_type is LensType.Spherical:
                # if the lens is spherical, default length to diameter
                self.doubleSpinBox_LensLength.setValue(self.lens.diameter / self.units)
            else:
                # if no length is given for a cylindrical lens, default length to 1 pixel
                self.doubleSpinBox_LensLength.setValue(self.lens.length / self.units)

            self.doubleSpinBox_LensMedium.setValue(lens_config["medium"])
            self.doubleSpinBox_LensExponent.setValue(self.lens.exponent)

            # generate the profile
            self.generate_profile()
            self.checkBox_LiveUpdate.setChecked(was_live)

        else:
            return

    def save_profile(self):
        self.load_aperture_UI()
        self.load_truncation_UI()
        self.load_general_UI()
        self.load_grating_UI()

    ### Update methods ###

    # def update_profile_parameters(self):
    #     """Helper function for generate_profile"""
    #     # read lens type so that we can generate the profile
    #     # self.lens_type = lens_type_dict[self.comboBox_LensType.currentText()]
    #     # self.pixel_size = self.doubleSpinBox_PixelSize.value() * self.units

    #     # set minimums to avoid erroring
    #     # self.doubleSpinBox_LensDiameter.setMinimum(2 * self.pixel_size / self.units)
    #     # self.doubleSpinBox_LensLength.setMinimum(1 * self.pixel_size / self.units)

    #     if self.lens_type is LensType.Spherical:
    #         self.lens_length = self.doubleSpinBox_LensDiameter.value() * self.units
    #     else:
    #         self.lens_length = self.doubleSpinBox_LensLength.value() * self.units

    def update_masks(self):
        """Helper function for generate_profile"""
        if self.lens_dict["grating"] is not None:
            self.update_grating_mask()

        if self.lens_dict["truncation"] is not None:
            self.update_truncation_mask()

        if self.lens_dict["aperture"] is not None:
            self.update_custom_aperture_mask()

    def update_grating_mask(self):
        # # Update minimums/maximums to not give user ability to error out
        # self.doubleSpinBox_GratingDistance.setMinimum(2 * self.pixel_size / self.units)

        # self.doubleSpinBox_GratingDistance.setMaximum(
        #     (self.lens.diameter - self.pixel_size) / self.units
        # )

        # self.doubleSpinBox_GratingWidth.setMinimum(1 * self.pixel_size / self.units)

        # self.doubleSpinBox_GratingWidth.setMaximum(
        #     (self.doubleSpinBox_GratingDistance.value() * self.units - self.pixel_size)
        #     / self.units
        # )

        try:
            grating_settings = GratingSettings(
                width=self.lens_dict["grating"]["width"],
                distance=self.lens_dict["grating"]["distance"],
                depth=self.lens_dict["grating"]["depth"],
                # TODO: check if axis does anything, don't think it does
                axis=5,
                centred=self.lens_dict["grating"]["centred"],
            )

            self.lens.create_grating_mask(
                settings=grating_settings,
                x_axis=self.lens_dict["grating"]["x"],
                y_axis=self.lens_dict["grating"]["y"],
            )

        except Exception as e:
            self.display_error_message(traceback.format_exc())

    def update_truncation_mask(self):
        # truncation_mode = self.lens_dict["truncation"][
        #     "mode"
        # ]  # self.comboBox_TruncationMode.currentText()

        # self.doubleSpinBox_TruncationValue.setMinimum(self.pixel_size / self.units)
        # self.doubleSpinBox_TruncationValue.setMaximum(self.lens.height / self.units)

        # self.doubleSpinBox_TruncationRadius.setMinimum(self.pixel_size / self.units)
        # self.doubleSpinBox_TruncationRadius.setMaximum(
        #     ((self.lens.diameter / 2) - self.pixel_size) / self.units
        # )

        # truncation_value = self.doubleSpinBox_TruncationValue.value() * self.units
        # truncation_radius = self.doubleSpinBox_TruncationRadius.value() * self.units

        # if truncation_mode == "Height":
        #     truncation_mode = "value"
        # if truncation_mode == "Radius":
        #     truncation_mode = "radial"

        try:
            self.lens.create_truncation_mask(
                truncation_height=self.lens_dict["truncation"]["height"],
                radius=self.lens_dict["truncation"]["radius"],
                type=self.lens_dict["truncation"]["type"],
                aperture=self.checkBox_TruncationAperture.isChecked(),
            )
        except Exception as e:
            self.display_error_message(traceback.format_exc())

    def update_custom_aperture_mask(self):

        # self.doubleSpinBox_ApertureOuter.setMinimum(self.pixel_size * 2 / self.units)

        # self.doubleSpinBox_ApertureOuter.setMaximum(
        #     ((self.lens.diameter / 2) - self.pixel_size) / self.units
        # )
        # self.doubleSpinBox_ApertureInner.setMinimum(self.pixel_size * 2 / self.units)

        # self.doubleSpinBox_ApertureInner.setMaximum(
        #     (self.doubleSpinBox_ApertureOuter.value() * self.units - self.pixel_size)
        #     / self.units
        # )

        # aperture_mode = self.comboBox_ApertureMode.currentText()
        # aperture_inner = self.doubleSpinBox_ApertureInner.value() * self.units
        # aperture_outer = self.doubleSpinBox_ApertureOuter.value() * self.units
        # aperture_inverted = self.checkBox_ApertureInverted.isChecked()

        # if aperture_mode == "Square":
        #     aperture_mode = "square"
        # if aperture_mode == "Circle":
        #     aperture_mode = "radial"

        try:
            self.lens.create_custom_aperture(
                inner_m=self.lens_dict["aperture"]["inner"],
                outer_m=self.lens_dict["aperture"]["outer"],
                type=self.lens_dict["aperture"]["type"],
                # TODO: make inverted standard
                inverted=self.lens_dict["aperture"]["invert"],
            )
        except Exception as e:
            self.display_error_message(traceback.format_exc())

    def update_image_frames(self):

        plt.close("all")

        mask_dict = {
            "tab_General": [None, "Mask"],
            "tab_Gratings": [self.lens.grating_mask, "Grating Mask"],
            "tab_Truncation": [self.lens.truncation_mask, "Truncation Mask"],
            "tab_Aperture": [self.lens.custom_aperture_mask, "Aperture Mask"],
        }

        cs_mask_image = None
        profile_mask_image = None
        current_tab = self.tabWidget.currentWidget().objectName()
        self.label_TitleMask.setText(mask_dict[current_tab][1])

        if mask_dict[current_tab][0] is not None:
            cs_mask_image = mask_dict[current_tab][0][
                self.lens.profile.shape[0] // 2, :
            ]
            profile_mask_image = mask_dict[current_tab][0]

        self.pc_Profile = self.update_frame(
            label=self.label_Profile,
            pc=self.pc_Profile,
            image=self.lens.profile,
            ndim=2,
            mask=False,
        )

        self.pc_ProfileMask = self.update_frame(
            label=self.label_ProfileMask,
            pc=self.pc_ProfileMask,
            image=profile_mask_image,
            ndim=2,
            mask=True,
        )

        self.pc_CrossSection = self.update_frame(
            label=self.label_CrossSection,
            pc=self.pc_CrossSection,
            image=self.lens.profile[self.lens.profile.shape[0] // 2, :],
            ndim=1,
            mask=False,
        )

        self.pc_CrossSectionMask = self.update_frame(
            label=self.label_CrossSectionMask,
            pc=self.pc_CrossSectionMask,
            image=cs_mask_image,
            ndim=1,
            mask=True,
        )

    def update_frame(self, label, pc, image, ndim, mask):
        """Helper function for update_image_frames"""
        if label.layout() is None:
            label.setLayout(QtWidgets.QVBoxLayout())
        if pc is not None:
            label.layout().removeWidget(pc)
            pc.deleteLater()

        pc = _ImageCanvas(
            parent=label, image=image, lens=self.lens, ndim=ndim, mask=mask
        )

        label.layout().addWidget(pc)

        return pc

    def update_units(self):
        old_units = self.units

        self.units = units_dict[self.comboBox_Units.currentIndex()]

        unit_conversion = self.units/old_units

        print(unit_conversion)

        self.lens_dict["pixel_size"] *= unit_conversion
        self.lens_dict["diameter"] *= unit_conversion
        self.lens_dict["height"] *= unit_conversion
        self.lens_dict["length"] *= unit_conversion

        # self.doubleSpinBox_GratingWidth.setValue(self.doubleSpinBox_GratingWidth.value()*unit_conversion)
        # self.doubleSpinBox_GratingDistance.setValue(self.doubleSpinBox_GratingDistance.value()*unit_conversion)
        # self.doubleSpinBox_GratingDepth.setValue(self.doubleSpinBox_GratingDepth.value()*unit_conversion)

        if self.lens_dict["grating"] is not None:
            self.lens_dict["grating"]["width"] *= unit_conversion
            self.lens_dict["grating"]["distance"] *= unit_conversion
            self.lens_dict["grating"]["depth"] *= unit_conversion

        # self.doubleSpinBox_TruncationRadius.setValue(self.doubleSpinBox_TruncationRadius.value()*unit_conversion)
        # self.doubleSpinBox_TruncationValue.setValue(self.doubleSpinBox_TruncationValue.value()*unit_conversion)

        if self.lens_dict["truncation"] is not None:
            self.lens_dict["truncation"]["radius"] *= unit_conversion
            self.lens_dict["truncation"]["height"] *= unit_conversion

        # self.doubleSpinBox_ApertureInner.setValue(self.doubleSpinBox_ApertureInner.value()*unit_conversion)
        # self.doubleSpinBox_ApertureOuter.setValue(self.doubleSpinBox_ApertureOuter.value()*unit_conversion)

        if self.lens_dict["aperture"] is not None:
            self.lens_dict["aperture"]["inner"] *= unit_conversion
            self.lens_dict["aperture"]["outer"] *= unit_conversion

        # modify values that rely on units to new unit system




    def live_update_profile(self):
        if self.checkBox_LiveUpdate.isChecked():
            try:
                self.checkBox_LiveUpdate.setChecked(False)
                self.update_UI_limits()
                self.update_lens_dict()
                self.generate_profile()
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
                max_height = np.amax(lens.profile)
                self.fig = utils.plot_lens_profile_slices(
                    lens=lens, max_height=max_height, title="", facecolor="#f0f0f0"
                )

        FigureCanvasQTAgg.__init__(self, self.fig)
        self.setParent(parent)


def main():
    """Launch the `piescope_gui` main application window."""
    application = QtWidgets.QApplication([])
    window = GUILensCreator()
    application.aboutToQuit.connect(window.disconnect)  # cleanup & teardown
    sys.exit(application.exec_())


if __name__ == "__main__":
    main()
