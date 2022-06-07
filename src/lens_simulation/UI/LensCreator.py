import sys
import traceback

import lens_simulation.UI.qtdesigner_files.LensCreator as LensCreator
import numpy as np
from lens_simulation.Lens import GratingSettings, Lens, LensType
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtGui, QtWidgets


class GUILensCreator(LensCreator.Ui_LensCreator, QtWidgets.QMainWindow):
    def __init__(self, parent_gui=None):
        super().__init__(parent=parent_gui)
        self.setupUi(LensCreator=self)
        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)

        self.setup_connections()
        # set up of image frames
        self.pc_CrossSection = None
        self.pc_Profile = None

        self.generate_profile()

        self.center_window()
        self.showNormal()

    def setup_connections(self):
        self.pushButton_LoadProfile.clicked.connect(self.load_profile)
        self.pushButton_GenerateProfile.clicked.connect(self.generate_profile)

        # connect each of the lens parameter selectors to update profile in live view
        [value.valueChanged.connect(self.live_update_profile) for value in self.__dict__.values() if value.__class__ is QtWidgets.QDoubleSpinBox]
        [value.toggled.connect(self.live_update_profile) for value in self.__dict__.values() if value.__class__ is QtWidgets.QCheckBox]
        [value.currentTextChanged.connect(self.live_update_profile) for value in self.__dict__.values() if value.__class__ is QtWidgets.QComboBox]

    def live_update_profile(self):
        if self.checkBox_LiveUpdate.isChecked():
            try:
                self.generate_profile()
            except Exception as e:
                self.display_error_message(traceback.format_exc())

    def generate_profile(self):
        """Generates a profile based on the inputs to the GUI"""
        # generate the lens based off the parameters selected in GUI
        self.generate_base_lens()

        # read lens type so that we can generate the profile
        lens_type_dict = {
            "Cylindrical": LensType.Cylindrical,
            "Spherical": LensType.Spherical,
        }

        lens_type = lens_type_dict[self.comboBox_LensType.currentText()]
        pixel_size = self.doubleSpinBox_PixelSize.value()
        cylindrical_extrusion = self.doubleSpinBox_CylindricalExtrusion.value()

        try:
            self.lens.generate_profile(pixel_size=pixel_size, lens_type=lens_type)
        except Exception as e:
            self.display_error_message(traceback.format_exc())

        if self.lens.lens_type is LensType.Cylindrical:
            if cylindrical_extrusion < pixel_size:
                cylindrical_extrusion = pixel_size
            try:
                self.lens.extrude_profile(length=cylindrical_extrusion)
            except Exception as e:
                self.display_error_message(traceback.format_exc())

        if self.groupBox_Gratings.isChecked():
            grating_width = self.doubleSpinBox_GratingWidth.value()
            grating_distance = self.doubleSpinBox_GratingDistance.value()

            if grating_distance >= self.lens.diameter:
                grating_distance = self.lens.diameter - pixel_size

            if grating_width >= grating_distance:
                self.statusBar.showMessage(
                    "Grating width adjusted to be less than grating distance"
                )
                grating_width = grating_distance - pixel_size

            grating_depth = self.doubleSpinBox_GratingDepth.value()
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

            # TODO: Tell Patrick to fix this in generation
            if self.lens.profile.shape[0] == 1:
                y_axis = False

            try:
                self.lens.calculate_grating_mask(
                    settings=grating_settings, x_axis=x_axis, y_axis=y_axis
                )
                self.lens.apply_masks(grating=True)
            except Exception as e:
                self.display_error_message(traceback.format_exc())

        if self.groupBox_Truncation.isChecked():
            truncation_mode = self.comboBox_TruncationMode.currentText()
            truncation_value = self.doubleSpinBox_TruncationValue.value()
            truncation_radius = self.doubleSpinBox_TruncationRadius.value()

            if truncation_mode == "Height":
                truncation_mode = "value"
            if truncation_mode == "Radius":
                truncation_mode = "radial"

            try:
                self.lens.calculate_truncation_mask(
                    truncation=truncation_value,
                    radius=truncation_radius,
                    type=truncation_mode,
                )
                self.lens.apply_masks(truncation=True)
            except Exception as e:
                self.display_error_message(traceback.format_exc())

        self.update_image_frames()

    def load_profile(self):
        """Loads a custom lens profile (numpy.ndarray) through Qt's file opening system"""
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Profile", filter="Numpy array (*.npy)"
        )

        if filename is "":
            return

        self.generate_base_lens()
        self.lens.load_profile(
            fname=filename, pixel_size=self.doubleSpinBox_PixelSize.value()
        )

        self.update_image_frames()

    def generate_base_lens(self):
        self.lens = Lens(
            diameter=self.doubleSpinBox_LensDiameter.value(),
            height=self.doubleSpinBox_LensHeight.value(),
            exponent=self.doubleSpinBox_LensExponent.value(),
            medium=self.doubleSpinBox_LensMedium.value(),
        )

    def update_image_frames(self):
        # Cross section initialisation
        if self.pc_CrossSection is not None:
            self.label_CrossSection.layout().removeWidget(self.pc_CrossSection)
            self.pc_CrossSection.deleteLater()
        self.pc_CrossSection = _ImageCanvas(
            parent=self.label_CrossSection,
            image=self.lens.profile[self.lens.profile.shape[0] // 2, :],
        )
        if self.label_CrossSection.layout() is None:
            self.label_CrossSection.setLayout(QtWidgets.QVBoxLayout())
        self.label_CrossSection.layout().addWidget(self.pc_CrossSection)

        # Cross section initialisation
        if self.pc_Profile is not None:
            self.label_Profile.layout().removeWidget(self.pc_Profile)
            self.pc_Profile.deleteLater()
        self.pc_Profile = _ImageCanvas(
            parent=self.label_Profile, image=self.lens.profile
        )
        if self.label_Profile.layout() is None:
            self.label_Profile.setLayout(QtWidgets.QVBoxLayout())
        self.label_Profile.layout().addWidget(self.pc_Profile)

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
    def __init__(self, parent=None, image=None):
        # self.fig = Figure()
        self.fig = Figure(layout="constrained")
        FigureCanvasQTAgg.__init__(self, self.fig)
        self.setParent(parent)

        gridspec = self.fig.add_gridspec(1, 1)
        self.axes = self.fig.add_subplot(gridspec[0], xticks=[], yticks=[], title="")

        # Push the image to edges of border as much as we can
        # self.axes.axis('off')
        self.axes.spines["top"].set_visible(False)
        self.axes.spines["right"].set_visible(False)
        self.axes.spines["bottom"].set_visible(False)
        self.axes.spines["left"].set_visible(False)
        self.fig.set_facecolor("#f0f0f0")

        # Display image
        if image.ndim == 2:
            self.axes.imshow(
                image, aspect="auto"
            )  # , extent=[0, self.lens.diameter, ])
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
