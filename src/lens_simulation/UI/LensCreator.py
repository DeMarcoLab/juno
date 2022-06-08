import sys
import traceback

from matplotlib import units

import lens_simulation.UI.qtdesigner_files.LensCreator as LensCreator
import numpy as np
from lens_simulation.Lens import GratingSettings, Lens, LensType
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtGui, QtWidgets
from lens_simulation import constants, utils

lens_type_dict = {
    "Cylindrical": LensType.Cylindrical,
    "Spherical": LensType.Spherical,
}

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
        self.setup_connections()

        # set up of image frames
        self.pc_CrossSection = None
        self.pc_CrossSectionMask = None
        self.pc_Profile = None
        self.pc_ProfileMask = None

        self.generate_profile()

        self.center_window()
        self.showNormal()

    ### Setup methods ###

    def setup_connections(self):
        self.pushButton_LoadProfile.clicked.connect(self.load_profile)
        self.pushButton_GenerateProfile.clicked.connect(self.generate_profile)

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

    ### Generation methods ###

    def generate_profile(self):
        """Generates a profile based on the inputs to the GUI"""
        # generate the lens based off the parameters selected in GUI
        self.masks_applied = False
        self.units = units_dict[self.comboBox_Units.currentIndex()]
        self.frame_TruncationAperture.setEnabled(self.groupBox_Truncation.isChecked())


        try:
            self.update_profile_parameters()
            self.generate_base_lens()

            self.lens.generate_profile(
                pixel_size=self.pixel_size,
                lens_type=self.lens_type,
                length=self.lens_length,
            )

            self.update_masks()

            # this loop is here to avoid double applying masks in Live Mode
            if self.masks_applied is False:
                self.lens.apply_masks(
                    grating=self.groupBox_Gratings.isChecked(),
                    truncation=self.groupBox_Truncation.isChecked(),
                    aperture=self.groupBox_Aperture.isChecked(),
                )
                self.masks_applied = True

            if self.checkBox_InvertedProfile.isChecked():
                self.lens.invert_profile()

            self.update_image_frames()

        except Exception as e:
            self.display_error_message(traceback.format_exc())

    def generate_base_lens(self):
        self.lens = Lens(
            diameter=self.doubleSpinBox_LensDiameter.value() * self.units,
            height=self.doubleSpinBox_LensHeight.value() * self.units,
            exponent=self.doubleSpinBox_LensExponent.value(),
            medium=self.doubleSpinBox_LensMedium.value(),
        )

    ### I/O methods ###

    def load_profile(self):
        """Loads a custom lens profile (numpy.ndarray) through Qt's file opening system"""
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Profile", filter="All types (*.yml *.yaml *.npy) ;;Yaml config (*.yml *.yaml) ;;Numpy array (*.npy)"
        )

        if filename is "":
            return


        if filename.endswith('.npy'):
            self.generate_base_lens()
            self.lens.load_profile(
                fname=filename,
                pixel_size=self.doubleSpinBox_PixelSize.value() * self.units,
            )

            self.update_image_frames()
            self.checkBox_LiveUpdate.setChecked(False)

        elif filename.endswith(('.yml', '.yaml')):
            config = utils.load_config(filename)

            self.doubleSpinBox_PixelSize.setValue(config["diameter"])
            self.doubleSpinBox_LensDiameter.setValue(config["diameter"])
            self.doubleSpinBox_LensHeight.setValue(config["height"])
            self.doubleSpinBox_LensLength.setValue(config["length"])
            self.doubleSpinBox_LensMedium.setValue(config["medium"])
            self.doubleSpinBox_LensExponent.setValue(config["exponent"])

        else:
            return



    ### Update methods ###

    def live_update_profile(self):
        if self.checkBox_LiveUpdate.isChecked():
            try:
                self.generate_profile()
            except Exception as e:
                self.display_error_message(traceback.format_exc())

    def update_profile_parameters(self):
        # read lens type so that we can generate the profile
        self.lens_type = lens_type_dict[self.comboBox_LensType.currentText()]
        self.pixel_size = self.doubleSpinBox_PixelSize.value() * self.units

        # set minimums to avoid erroring
        self.doubleSpinBox_LensDiameter.setMinimum(2 * self.pixel_size / self.units)
        self.doubleSpinBox_LensLength.setMinimum(1 * self.pixel_size / self.units)

        self.lens_length = self.doubleSpinBox_LensLength.value() * self.units

    def update_masks(self):
        if self.groupBox_Gratings.isChecked():
            self.update_grating_mask()

        if self.groupBox_Truncation.isChecked():
            self.update_truncation_mask()

        if self.groupBox_Aperture.isChecked():
            self.update_custom_aperture_mask()

    def update_grating_mask(self):
        # Update minimums/maximums to not give user ability to error out
        self.doubleSpinBox_GratingDistance.setMinimum(2 * self.pixel_size / self.units)

        self.doubleSpinBox_GratingDistance.setMaximum(
            (self.lens.diameter - self.pixel_size) / self.units
        )

        self.doubleSpinBox_GratingWidth.setMinimum(1 * self.pixel_size / self.units)

        self.doubleSpinBox_GratingWidth.setMaximum(
            (self.doubleSpinBox_GratingDistance.value() * self.units - self.pixel_size)
            / self.units
        )

        # only applied if adaptive steps are turned off
        self.doubleSpinBox_GratingDistance.setSingleStep(
            1 * self.pixel_size / self.units
        )
        self.doubleSpinBox_GratingWidth.setSingleStep(1 * self.pixel_size / self.units)

        grating_width = self.doubleSpinBox_GratingWidth.value() * self.units
        grating_distance = self.doubleSpinBox_GratingDistance.value() * self.units
        grating_depth = self.doubleSpinBox_GratingDepth.value() * self.units
        grating_axis = self.comboBox_GratingAxis.currentText()
        grating_centered = self.checkBox_GratingCentered.isChecked()

        grating_settings = GratingSettings(
            width=grating_width,
            distance=grating_distance,
            depth=grating_depth,
            axis=grating_axis,
            centred=grating_centered,
        )

        x_axis = False
        y_axis = False

        if grating_axis in ["Vertical", "Both"]:
            x_axis = True

        if grating_axis in ["Horizontal", "Both"]:
            y_axis = True

        try:
            self.lens.calculate_grating_mask(
                settings=grating_settings, x_axis=x_axis, y_axis=y_axis
            )

        except Exception as e:
            self.display_error_message(traceback.format_exc())

    def update_truncation_mask(self):
        truncation_mode = self.comboBox_TruncationMode.currentText()

        self.doubleSpinBox_TruncationValue.setMinimum(self.pixel_size / self.units)
        self.doubleSpinBox_TruncationValue.setMaximum(self.lens.height / self.units)

        self.doubleSpinBox_TruncationRadius.setMinimum(self.pixel_size / self.units)
        self.doubleSpinBox_TruncationRadius.setMaximum(
            ((self.lens.diameter / 2) - self.pixel_size) / self.units
        )

        truncation_value = self.doubleSpinBox_TruncationValue.value() * self.units
        truncation_radius = self.doubleSpinBox_TruncationRadius.value() * self.units

        if truncation_mode == "Height":
            truncation_mode = "value"
        if truncation_mode == "Radius":
            truncation_mode = "radial"

        try:
            self.lens.create_truncation_mask(
                truncation_height=truncation_value,
                radius=truncation_radius,
                type=truncation_mode,
                aperture=self.checkBox_TruncationAperture.isChecked(),
            )
        except Exception as e:
            self.display_error_message(traceback.format_exc())

    def update_custom_aperture_mask(self):

        self.doubleSpinBox_ApertureOuter.setMinimum(self.pixel_size * 2 / self.units)

        self.doubleSpinBox_ApertureOuter.setMaximum(
            ((self.lens.diameter / 2) - self.pixel_size) / self.units
        )
        self.doubleSpinBox_ApertureInner.setMinimum(self.pixel_size * 2 / self.units)

        self.doubleSpinBox_ApertureInner.setMaximum(
            (self.doubleSpinBox_ApertureOuter.value() * self.units - self.pixel_size)
            / self.units
        )

        aperture_mode = self.comboBox_ApertureMode.currentText()
        aperture_inner = self.doubleSpinBox_ApertureInner.value() * self.units
        aperture_outer = self.doubleSpinBox_ApertureOuter.value() * self.units
        aperture_inverted = self.checkBox_ApertureInverted.isChecked()

        if aperture_mode == "Square":
            aperture_mode = "square"
        if aperture_mode == "Circle":
            aperture_mode = "radial"

        try:
            self.lens.calculate_aperture(
                inner_m=aperture_inner,
                outer_m=aperture_outer,
                type=aperture_mode,
                inverted=aperture_inverted,
            )
        except Exception as e:
            self.display_error_message(traceback.format_exc())

    def update_image_frames(self):

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

        # Cross section initialisation
        if self.label_CrossSection.layout() is None:
            self.label_CrossSection.setLayout(QtWidgets.QVBoxLayout())

        if self.pc_CrossSection is not None:
            self.label_CrossSection.layout().removeWidget(self.pc_CrossSection)
            self.pc_CrossSection.deleteLater()

        self.pc_CrossSection = _ImageCanvas(
            parent=self.label_CrossSection,
            image=self.lens.profile[self.lens.profile.shape[0] // 2, :],
            lens=self.lens,
        )

        self.label_CrossSection.layout().addWidget(self.pc_CrossSection)

        # Mask cross section initialisation
        if self.label_CrossSectionMask.layout() is None:
            self.label_CrossSectionMask.setLayout(QtWidgets.QVBoxLayout())

        if self.pc_CrossSectionMask is not None:
            self.label_CrossSectionMask.layout().removeWidget(self.pc_CrossSectionMask)
            self.pc_CrossSectionMask.deleteLater()

        self.pc_CrossSectionMask = _ImageCanvas(
            parent=self.label_CrossSectionMask, image=cs_mask_image, lens=self.lens,
        )

        self.label_CrossSectionMask.layout().addWidget(self.pc_CrossSectionMask)

        # Profile initialisation
        if self.label_Profile.layout() is None:
            self.label_Profile.setLayout(QtWidgets.QVBoxLayout())

        if self.pc_Profile is not None:
            self.label_Profile.layout().removeWidget(self.pc_Profile)
            self.pc_Profile.deleteLater()

        self.pc_Profile = _ImageCanvas(
            parent=self.label_Profile, image=self.lens.profile, lens=self.lens
        )

        self.label_Profile.layout().addWidget(self.pc_Profile)

        # Mask profile initialisation
        if self.label_ProfileMask.layout() is None:
            self.label_ProfileMask.setLayout(QtWidgets.QVBoxLayout())

        if self.pc_ProfileMask is not None:
            self.label_ProfileMask.layout().removeWidget(self.pc_ProfileMask)
            self.pc_ProfileMask.deleteLater()

        self.pc_ProfileMask = _ImageCanvas(
            parent=self.label_ProfileMask, image=profile_mask_image, lens=self.lens
        )

        self.label_ProfileMask.layout().addWidget(self.pc_ProfileMask)

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


class _ImageCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, image=None, lens=None):
        self.fig = Figure(layout="constrained")
        self.fig.set_facecolor("#f0f0f0")

        FigureCanvasQTAgg.__init__(self, self.fig)
        self.setParent(parent)

        gridspec = self.fig.add_gridspec(1, 1)
        self.axes = self.fig.add_subplot(gridspec[0], xticks=[], yticks=[], title="")
        self.axes.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
        self.axes.clear()

        if image is not None:
            # Display image
            if image.ndim == 2:
                self.axes.imshow(
                    image,
                    aspect="auto",
                    # TODO: add length here
                    extent=[-lens.diameter / 2, lens.diameter / 2, 0, 1],
                )
            else:
                self.axes.plot(image)


def main():
    """Launch the `piescope_gui` main application window."""
    application = QtWidgets.QApplication([])
    window = GUILensCreator()
    application.aboutToQuit.connect(window.disconnect)  # cleanup & teardown
    sys.exit(application.exec_())


if __name__ == "__main__":
    main()
