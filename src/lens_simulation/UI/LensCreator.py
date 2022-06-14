import os
import sys
import traceback

from sympy import DiagMatrix

import lens_simulation.UI.qtdesigner_files.LensCreator as LensCreator
import numpy as np
import yaml
from lens_simulation import constants, utils
from lens_simulation.Lens import GratingSettings, LensType, Medium, generate_lens
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt5 import QtCore, QtGui, QtWidgets

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

        # default to um
        self.comboBox_Units.setCurrentIndex(1)

        # set up of image frames
        self.pc_CrossSection = None
        self.pc_CrossSectionMask = None
        self.pc_Profile = None
        self.pc_ProfileMask = None

        self.units = units_dict[1]

        # creat initial lens config
        self.create_new_lens_dict()
        self.update_UI()
        self.update_UI_limits()
        self.create_lens()

        self.setup_connections()

        self.center_window()
        self.showNormal()

    ### Setup methods ###

    def setup_connections(self):
        self.pushButton_LoadProfile.clicked.connect(self.load_profile)
        self.pushButton_GenerateProfile.clicked.connect(self.create_lens)
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

    def create_new_lens_dict(self, filename=None):
        self.lens_dict = dict()
        self.lens_dict["name"] = "Lens"
        self.lens_dict["medium"] = 2.348
        self.lens_dict["exponent"] = 2.0
        self.lens_dict["diameter"] = 100.0e-6
        self.lens_dict["lens_type"] = "Spherical"
        self.lens_dict["length"] = self.lens_dict["diameter"]
        self.lens_dict["height"] = 10.0e-6
        self.lens_dict["pixel_size"] = 0.1e-6
        self.lens_dict["custom"] = filename
        self.lens_dict["inverted"] = False
        self.lens_dict["escape_path"] = None
        self.lens_dict["grating"] = None
        self.lens_dict["truncation"] = None
        self.lens_dict["aperture"] = None

    def create_lens(self):
        self.lens = generate_lens(
            lens_config=self.lens_dict,
            medium=Medium(self.lens_dict["medium"]),
            pixel_size=self.lens_dict["pixel_size"],
        )

        self.lens_dict['diameter'] = self.lens.diameter
        self.lens_dict['pixel_size'] = self.lens.pixel_size
        self.lens_dict['height'] = self.lens.height

        if self.lens_dict["inverted"]:
            self.lens.invert_profile()

        self.update_image_frames()

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

        self.frame_TruncationAperture.setEnabled(
            self.lens_dict["truncation"] is not None
            and self.lens_dict["aperture"] is not None
        )

    def update_UI_general(self):
        # Config -> UI | General settings #
        self.lineEdit_LensName.setText(self.lens_dict["name"])
        self.doubleSpinBox_LensMedium.setValue(self.lens_dict["medium"])
        self.doubleSpinBox_LensExponent.setValue(self.lens_dict["exponent"])
        self.comboBox_LensType.setCurrentText(self.lens_dict["lens_type"])
        self.doubleSpinBox_PixelSize.setValue(self.lens_dict["pixel_size"] / self.units)
        self.doubleSpinBox_LensDiameter.setValue(
            self.lens_dict["diameter"] / self.units
        )
        # set length to diameter for spherical lenses
        if self.lens_dict["lens_type"] == "Spherical":
            self.doubleSpinBox_LensLength.setValue(
                self.lens_dict["diameter"] / self.units
            )
        else:
            self.doubleSpinBox_LensLength.setValue(
                self.lens_dict["length"] / self.units
            )
        self.frame_LensLength.setEnabled(not self.lens_dict["lens_type"] == "Spherical")
        self.doubleSpinBox_LensLength.setEnabled(not self.lens_dict["lens_type"] == "Spherical")

        self.doubleSpinBox_LensHeight.setValue(self.lens_dict["height"] / self.units)
        if self.lens_dict["escape_path"] is not None:
            self.doubleSpinBox_LensEscapePath.setValue(self.lens_dict["escape_path"])
        else:
            self.doubleSpinBox_LensEscapePath.setValue(0.0)
        if self.lens_dict["inverted"] is not None:
            self.checkBox_InvertedProfile.setChecked(self.lens_dict["inverted"])
        else:
            self.checkBox_InvertedProfile.setChecked(False)

        # If custom lens, take some options away
        self.frame_LensDiameter.setEnabled(self.lens_dict["custom"] is None)
        self.frame_LensHeight.setEnabled(self.lens_dict["custom"] is None)
        self.frame_LensLength.setEnabled(self.lens_dict["custom"] is None)
        self.frame_LensExponent.setEnabled(self.lens_dict["custom"] is None)
        self.frame_LensType.setEnabled(self.lens_dict["custom"] is None)
        self.frame_InvertedProfile.setEnabled(self.lens_dict["custom"] is None)
        self.frame_LensEscapePath.setEnabled(self.lens_dict["custom"] is None)

    def update_config_general(self):
        # UI -> Config | General settings #
        self.lens_dict["name"] = self.lineEdit_LensName.text()
        self.lens_dict["medium"] = self.doubleSpinBox_LensMedium.value()
        self.lens_dict["exponent"] = self.doubleSpinBox_LensExponent.value()
        self.lens_dict["lens_type"] = self.comboBox_LensType.currentText()

        if self.doubleSpinBox_LensEscapePath.value() == 0:
            self.lens_dict["escape_path"] = None
        else:
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

        if self.lens_dict["aperture"]["type"] == "radial":
            self.comboBox_ApertureMode.setCurrentText("Circle")
        else:
            self.comboBox_ApertureMode.setCurrentText("Square")

        self.doubleSpinBox_ApertureOuter.setValue(
            self.lens_dict["aperture"]["outer"] / self.units
        )
        self.doubleSpinBox_ApertureInner.setValue(
            self.lens_dict["aperture"]["inner"] / self.units
        )

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
                self.doubleSpinBox_ApertureOuter.value() * self.units
                - self.lens_dict["pixel_size"]
            )
            / self.units
        )

    def format_float(self, num):
        # np format_float_scientific() might be the same?
        return float(f"{num:4e}")

    ### I/O methods ###

    def load_profile(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Profile",
            filter="All types (*.yml *.yaml *.npy) ;;Yaml config (*.yml *.yaml) ;;Numpy array (*.npy)",
        )

        if filename is "":
            return

        # get the status of live update to restore it post loading
        was_live = self.checkBox_LiveUpdate.isChecked()

        if filename.endswith(".npy"):
            try:
                self.checkBox_LiveUpdate.setChecked(False)
                self.create_new_lens_dict(filename)
                self.update_UI_limits()
                self.create_lens()
                self.update_UI()
                self.checkBox_LiveUpdate.setChecked(was_live)
            except Exception as e:
                self.display_error_message(traceback.format_exc())

        else:
            try:
                # turn off live update to avoid memory issues
                self.checkBox_LiveUpdate.setChecked(False)
                self.lens_dict = utils.load_yaml_config(filename)
                self.update_UI_limits()
                self.update_UI()
                self.update_UI_limits()
                self.update_UI()
                self.create_lens()
                self.checkBox_LiveUpdate.setChecked(was_live)
            except Exception as e:
                self.display_error_message(traceback.format_exc())

    def save_profile(self):
        filename, ext = QtWidgets.QFileDialog.getSaveFileName(self, "Save Profile", self.lens_dict["name"], filter="Yaml config (*.yml *.yaml)")

        if filename == "":
            return

        print(filename)

        with open(filename, "w") as f:
            yaml.safe_dump(self.lens_dict, f, sort_keys=False)

    ### Update methods ###

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

        unit_conversion = self.units / old_units

        self.lens_dict["pixel_size"] *= unit_conversion
        self.lens_dict["diameter"] *= unit_conversion
        self.lens_dict["height"] *= unit_conversion
        self.lens_dict["length"] *= unit_conversion

        if self.lens_dict["grating"] is not None:
            self.lens_dict["grating"]["width"] *= unit_conversion
            self.lens_dict["grating"]["distance"] *= unit_conversion
            self.lens_dict["grating"]["depth"] *= unit_conversion

        if self.lens_dict["truncation"] is not None:
            self.lens_dict["truncation"]["radius"] *= unit_conversion
            self.lens_dict["truncation"]["height"] *= unit_conversion

        if self.lens_dict["aperture"] is not None:
            self.lens_dict["aperture"]["inner"] *= unit_conversion
            self.lens_dict["aperture"]["outer"] *= unit_conversion

    def live_update_profile(self):
        if self.checkBox_LiveUpdate.isChecked():
            try:
                self.checkBox_LiveUpdate.setChecked(False)
                self.update_lens_dict()
                self.create_lens()
                self.update_UI_limits()
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
    window = GUILensCreator()
    application.aboutToQuit.connect(window.disconnect)  # cleanup & teardown
    sys.exit(application.exec_())


if __name__ == "__main__":
    main()
