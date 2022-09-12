
# DONE
# create 
# load config
# save config
# load profile

## features
# custom profile (load)
# QOL: disable useless options

import os
import sys
import traceback
from pprint import pprint

import juno.ui.qtdesigner_files.ElementCreation as ElementCreation
import napari
import numpy as np
import yaml
from juno import plotting, utils, validation
from juno.Lens import LensType, Medium, generate_lens
from juno.ui.utils import display_error_message
from PyQt5 import QtWidgets
import napari.utils.notifications

default_lens_config = {
    "name": "Lens", 
    "diameter": 200e-6,
    "height": 50e-6,
    "exponent": 2.0,
    "lens_type": "Spherical",
    "medium": 2.348, 
    "length": 10e-6,
    "escape_path": 0.0,
}

class GUIElementCreation(ElementCreation.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self, parent=None, viewer: napari.Viewer = None):
        super().__init__(parent=parent)
        self.setupUi(self)
        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)
        self.setWindowTitle("Element Creator")

        self.CONFIG_UPDATE = False

        self.viewer = viewer
        self.setup_connections()
        
        # set initial config
        validated_default_config = validation._validate_default_lens_config(default_lens_config)
        
        # update ui parameters and viz
        self.update_ui_from_config(validated_default_config)
        self.update_layer()
        self.show()

    def setup_connections(self):

        self.pushButton_generate_profile.clicked.connect(self.update_layer)

        # comboboxes
        self.comboBox_type.addItems([type.name for type in LensType][::-1]) # lens types
        self.comboBox_truncation_mode.addItems(["Height", "Radial"])    # truncation modes
        self.comboBox_aperture_mode.addItems(["Radial", "Square"])      # aperture modes

        # general
        self.lineEdit_pixelsize.textChanged.connect(self.update_layer)
        self.lineEdit_diameter.textChanged.connect(self.update_layer)
        self.lineEdit_length.textChanged.connect(self.update_layer)
        self.lineEdit_height.textChanged.connect(self.update_layer)
        self.lineEdit_medium.textChanged.connect(self.update_layer)
        self.lineEdit_exponent.textChanged.connect(self.update_layer)
        self.lineEdit_escape_path.textChanged.connect(self.update_layer)
        self.comboBox_type.currentTextChanged.connect(self.update_layer)
        self.checkBox_invert_profile.stateChanged.connect(self.update_layer)
        self.lineEdit_name.textChanged.connect(self.update_layer)

        # grating
        self.checkBox_use_grating.toggled.connect(self.update_layer)
        self.lineEdit_grating_width.textChanged.connect(self.update_layer)
        self.lineEdit_grating_distance.textChanged.connect(self.update_layer)
        self.lineEdit_grating_depth.textChanged.connect(self.update_layer)
        self.checkBox_grating_x_axis.toggled.connect(self.update_layer)
        self.checkBox_grating_y_axis.toggled.connect(self.update_layer)
        self.checkBox_grating_centred.toggled.connect(self.update_layer)

        # truncation
        self.checkBox_use_truncation.toggled.connect(self.update_layer)
        self.comboBox_truncation_mode.currentTextChanged.connect(self.update_layer)
        self.lineEdit_truncation_value.textChanged.connect(self.update_layer)
        self.checkBox_truncation_aperture.toggled.connect(self.update_layer) 

        # aperture
        self.checkBox_use_aperture.toggled.connect(self.update_layer)
        self.comboBox_aperture_mode.currentTextChanged.connect(self.update_layer)
        self.lineEdit_aperture_inner.textChanged.connect(self.update_layer)
        self.lineEdit_aperture_outer.textChanged.connect(self.update_layer)
        self.checkBox_aperture_invert.toggled.connect(self.update_layer)

        # buttons
        self.pushButton_generate_profile.clicked.connect(self.update_layer)

        # actions
        self.actionSave_Configuration.triggered.connect(self.save_configuration)
        self.actionLoad_Configuration.triggered.connect(self.load_configuration)

        self.actionSave_Profile.triggered.connect(self.save_profile)
        self.actionLoad_Profile.triggered.connect(self.load_profile)

    def testing_function(self):

        print("testing function!!")

    def update_ui_from_config(self, config: dict):

        self.CONFIG_UPDATE = True

        # general
        self.lineEdit_pixelsize.setText(str(1e-6))
        self.lineEdit_diameter.setText(str(config["diameter"]))
        self.lineEdit_length.setText(str(config["length"]))
        self.lineEdit_height.setText(str(config["height"]))
        self.lineEdit_medium.setText(str(config["medium"]))
        self.lineEdit_exponent.setText(str(config["exponent"]))
        self.lineEdit_escape_path.setText(str(config["escape_path"]))
        self.comboBox_type.setCurrentText(str(config["lens_type"]).capitalize())
        self.checkBox_invert_profile.setChecked(bool(config["inverted"]))
        self.lineEdit_name.setText(str(config["name"]))

        # grating
        use_grating =  bool(config["grating"])
        self.checkBox_use_grating.setChecked(use_grating)
        if use_grating:
            self.lineEdit_grating_width.setText(str(config["grating"]["width"]))
            self.lineEdit_grating_distance.setText(str(config["grating"]["distance"]))
            self.lineEdit_grating_depth.setText(str(config["grating"]["depth"]))
            self.checkBox_grating_x_axis.setChecked(bool(config["grating"]["x"]))
            self.checkBox_grating_y_axis.setChecked(bool(config["grating"]["y"]))
            self.checkBox_grating_centred.setChecked(bool(config["grating"]["centred"]))

        # # truncation
        use_truncation = bool(config["truncation"])
        self.checkBox_use_truncation.setChecked(use_truncation)
        if use_truncation:
            truncation_mode = str(config["truncation"]["type"])
            self.comboBox_truncation_mode.setCurrentText(truncation_mode.capitalize())
            if truncation_mode == "value":
                self.lineEdit_truncation_value.setText(str(config["truncation"]["height"]))
            if truncation_mode == "radial":
                self.lineEdit_truncation_value.setText(str(config["truncation"]["radius"]))
            self.checkBox_truncation_aperture.setChecked(bool(config["truncation"]["aperture"]))


        # # aperture
        use_aperture = bool(config["aperture"])
        self.checkBox_use_aperture.setChecked(use_aperture)
        if use_aperture:
            aperture_mode = str(config["aperture"]["type"])
            self.comboBox_aperture_mode.setCurrentText(aperture_mode.capitalize())
            self.lineEdit_aperture_inner.setText(str(config["aperture"]["inner"]))
            self.lineEdit_aperture_outer.setText(str(config["aperture"]["outer"]))
            self.checkBox_aperture_invert.setChecked(bool(config["aperture"]["invert"]))

        # self.update_ui_components()

        self.CONFIG_UPDATE = False

        return


    def update_ui_components(self):

        print("updating ui components...")

        # enable / disable general components

        # lenght based on lens type


        # custom lens disable certain settings
        # TODO


        # enable / disable grating components
        use_grating = self.checkBox_use_grating.isChecked()
        self.lineEdit_grating_width.setEnabled(use_grating)
        self.lineEdit_grating_distance.setEnabled(use_grating)
        self.lineEdit_grating_depth.setEnabled(use_grating)
        self.checkBox_grating_x_axis.setEnabled(use_grating)
        self.checkBox_grating_y_axis.setEnabled(use_grating)
        self.checkBox_grating_centred.setEnabled(use_grating)

        # enable / disable truncation components
        use_truncation = self.checkBox_use_truncation.isChecked()
        self.comboBox_truncation_mode.setEnabled(use_truncation)
        self.lineEdit_truncation_value.setEnabled(use_truncation)
        self.lineEdit_truncation_value.setEnabled(use_truncation)
        self.checkBox_truncation_aperture.setEnabled(use_truncation)

        # enable / disable aperture components   
        use_aperture = self.checkBox_use_aperture.isChecked()
        self.comboBox_aperture_mode.setEnabled(use_aperture)
        self.lineEdit_aperture_inner.setEnabled(use_aperture)
        self.lineEdit_aperture_outer.setEnabled(use_aperture)
        self.checkBox_aperture_invert.setEnabled(use_aperture)

    def load_configuration(self):

        print("load profile...")

        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Profile",
            filter="All types (*.yml *.yaml *.npy) ;;Yaml config (*.yml *.yaml) ;;Numpy array (*.npy)",
        )

        if filename == "":
            return

        if filename.endswith(".npy"):
            print("numpy file... TODO: custom profile load")

        else:
            print("lens configuration")
            self.lens_config = utils.load_yaml_config(filename)
            
            # validate config...
            self.lens_config = validation._validate_default_lens_config(self.lens_config)

            # validate config?
            try:
                self.update_ui_from_config(self.lens_config)
            except:
                napari.utils.notifications.show_error(traceback.format_exc())
            
            self.update_layer() 

    def save_configuration(self):

        try:
            self.update_config()
        except Exception as e:
            print(f"Unable to update config... {e}")

        filename, ext = QtWidgets.QFileDialog.getSaveFileName(self, "Save Profile", self.lens_config["name"], filter="Yaml config (*.yml *.yaml)")

        if filename == "":
            return

        self.lens_config["name"] = os.path.basename(filename).split('.')[0]
        self.lineEdit_name.setText(self.lens_config["name"])

        with open(filename, "w") as f:
            yaml.safe_dump(self.lens_config, f, sort_keys=False)

    def save_profile(self):
        print("save profile (.npy)")

        # profile must be valid?
        
        # generate profile

        # save npy file

    def load_profile(self):
        print("load profile (.npy)")

        # select npy file

        # load profile

        # set config

        # disable some options

    def update_config(self):

        # try except block

        lens_config = {}
    
        # core
        lens_config["height"] = float(self.lineEdit_height.text()) 
        lens_config["diameter"] = float(self.lineEdit_diameter.text()) 
        lens_config["exponent"] = float(self.lineEdit_exponent.text())
        lens_config["inverted"] = bool(self.checkBox_invert_profile.isChecked())
        lens_config["escape_path"] = float(self.lineEdit_escape_path.text())
        lens_config["length"] = float(self.lineEdit_length.text())
        lens_config["lens_type"] = self.comboBox_type.currentText()
        lens_config["name"] = self.lineEdit_name.text()
        lens_config["medium"] = float(self.lineEdit_medium.text())

        # grating
        if self.checkBox_use_grating.isChecked():
            lens_config["grating"] = {}
            lens_config["grating"]["width"] =  float(self.lineEdit_grating_width.text())
            lens_config["grating"]["distance"] = float(self.lineEdit_grating_distance.text())
            lens_config["grating"]["depth"] = float(self.lineEdit_grating_depth.text())
            lens_config["grating"]["x"] = bool(self.checkBox_grating_x_axis.isChecked())
            lens_config["grating"]["y"] = bool(self.checkBox_grating_y_axis.isChecked())
            lens_config["grating"]["centred"] = bool(self.checkBox_grating_centred.isChecked())
        else:
            lens_config["grating"] = None
        
        # truncation
        if self.checkBox_use_truncation.isChecked():
            lens_config["truncation"] = {}
            lens_config["truncation"]["height"] = float(self.lineEdit_truncation_value.text())
            lens_config["truncation"]["radius"] = float(self.lineEdit_truncation_value.text())
            lens_config["truncation"]["type"] = self.comboBox_truncation_mode.currentText() 
            lens_config["truncation"]["aperture"] = bool(self.checkBox_truncation_aperture.isChecked())
        else:
            lens_config["truncation"] = None

        # aperture
        if self.checkBox_use_aperture.isChecked():
            lens_config["aperture"] = {}
            lens_config["aperture"]["inner"] = float(self.lineEdit_aperture_inner.text())            
            lens_config["aperture"]["outer"] = float(self.lineEdit_aperture_outer.text())
            lens_config["aperture"]["type"] = str(self.comboBox_aperture_mode.currentText())
            lens_config["aperture"]["invert"] = bool(self.checkBox_aperture_invert.isChecked())
        else:
            lens_config["aperture"] = None

        self.lens_config = lens_config


    def update_layer(self):

        # TODO: enable this properly.        
        # self.update_ui_components()

        # dont update the layers when the config is updating the ui...
        if self.CONFIG_UPDATE:
            return

        # TODO: double update bug??

        # live update
        self.LIVE_UPDATE = self.checkBox_live_update.isChecked()

        # if not, check if button was the sender?
        if self.LIVE_UPDATE is False:
            if self.sender() is not self.pushButton_generate_profile:
                return

        # get updated config
        try:
            self.update_config()
        except Exception as e:
            napari.utils.notifications.show_error(f"ERROR: {traceback.format_exc()}")
            return

        lens = None
        arr3d = None

        try:
            # params
            medium = float(self.lineEdit_medium.text())
            pixelsize = float(self.lineEdit_pixelsize.text())


            
            if not valid_element_to_display(self.lens_config, pixelsize): 
                napari.utils.notifications.show_error(f"Element dimensions are too large to display.")
                return 

            # generate the lens profile
            lens = generate_lens(self.lens_config, Medium(medium), pixelsize)
            lens.apply_aperture_masks()
            arr3d = plotting.create_3d_lens(lens)

            if lens.grating_mask is None:
                lens.grating_mask = np.zeros_like(lens.profile)
            if lens.truncation_mask is None:
                lens.truncation_mask = np.zeros_like(lens.profile)

        except Exception as e:
            napari.utils.notifications.show_error(f"ERROR: {traceback.format_exc()}")
            return


        if lens is None or arr3d is None:
            return        

        self.viewer.dims.ndisplay = 3
        self.viewer.axes.visible = True
        self.viewer.axes.colored = False
        self.viewer.axes.dashed = True
        self.viewer.axes.labels = True
        self.viewer.scale_bar.visible = True


        # https://forum.image.sc/t/update-refresh-a-layer-in-napari-julia-napari-jl/59728
        print("update_layer")

        # TODO: set lens layer as active
        # TODO: find a better way to set the initial view iamges... probably better to separate
        # TODO: load profile

        # update layer in place 
        try:
            try:
                self.viewer.layers["Aperture Mask"].data = lens.aperture
                self.viewer.layers["Truncation Mask"].data = lens.truncation_mask
                self.viewer.layers["Grating Mask"].data = lens.grating_mask
                self.viewer.layers["Element"].data = arr3d
            except KeyError as e:
                self.viewer.add_image(lens.aperture, name="Aperture Mask", opacity=0.4, colormap="yellow", rendering="translucent")
                self.viewer.add_image(lens.truncation_mask, name="Truncation Mask", opacity=0.4, colormap="cyan", rendering="translucent")
                self.viewer.add_image(lens.grating_mask, name="Grating Mask", opacity=0.4, colormap="green", rendering="translucent")
                self.viewer.add_image(arr3d, name="Element", colormap="gray", rendering="iso", depiction="volume")
        except Exception as e:
            napari.utils.notifications.show_error(f"Failure to load viewer: {traceback.format_exc()}")

def valid_element_to_display(config: dict, pixelsize: int, max_dim: int = 10000) -> bool:
    """Check if the element array will be greater than the maximum allows array size for display"""
    # TODO: add more validation stuff here
    
    valid_element: bool = True

    for k in ["length", "diameter", "height"]:
        if config[k] / pixelsize > max_dim:
            valid_element = False
            print(f"invalid dimension: {k}, {config[k]}: {config[k] / pixelsize}")

    return valid_element


def main():
    """Launch the `piescope_gui` main application window."""
    application = QtWidgets.QApplication([])

    viewer = napari.Viewer(ndisplay=3)
    element_creation_ui = GUIElementCreation(viewer=viewer)                                          
    viewer.window.add_dock_widget(element_creation_ui, area='right')                  

    sys.exit(application.exec_())


if __name__ == "__main__":
    main()


