
## features
# create
# load config
# load profile
# save config


import os
import sys
import traceback

import juno.ui.qtdesigner_files.ElementCreation as ElementCreation
import numpy as np
import yaml
from juno import constants, plotting, utils
from juno.Lens import GratingSettings, LensType, Medium, generate_lens
from juno.ui.utils import display_error_message
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt5 import QtCore, QtGui, QtWidgets

import napari
import numpy as np
from pprint import pprint

class GUIElementCreation(ElementCreation.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self, parent=None, viewer: napari.Viewer = None):
        super().__init__(parent=parent)
        self.setupUi(self)
        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)

        self.viewer = viewer
        self.setup_connections()
        
        self.update_layer()
        self.show()

    def setup_connections(self):


        self.pushButton_generate_profile.clicked.connect(self.update_layer)

        # comboboxes
        self.comboBox_type.addItems([type.name for type in LensType][::-1]) # lens types
        self.comboBox_truncation_mode.addItems(["Height", "Radial"])  # truncation modes
        self.comboBox_aperture_mode.addItems(["Radial", "Square"]) # aperture modes

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
        self.checkBox_use_grating.stateChanged.connect(self.update_layer)
        self.lineEdit_grating_width.textChanged.connect(self.update_layer)
        self.lineEdit_grating_distance.textChanged.connect(self.update_layer)
        self.lineEdit_grating_depth.textChanged.connect(self.update_layer)
        self.checkBox_grating_x_axis.stateChanged.connect(self.update_layer)
        self.checkBox_grating_y_axis.stateChanged.connect(self.update_layer)
        self.checkBox_grating_centred.stateChanged.connect(self.update_layer)

        # truncation
        self.checkBox_use_truncation.stateChanged.connect(self.update_layer)
        self.comboBox_truncation_mode.currentTextChanged.connect(self.update_layer)
        self.lineEdit_truncation_value.textChanged.connect(self.update_layer)

        # aperture
        self.checkBox_use_aperture.stateChanged.connect(self.update_layer)
        self.comboBox_aperture_mode.currentTextChanged.connect(self.update_layer)
        self.lineEdit_aperture_inner.textChanged.connect(self.update_layer)
        self.lineEdit_aperture_outer.textChanged.connect(self.update_layer)
        self.checkBox_aperture_invert.stateChanged.connect(self.update_layer)
        self.checkBox_aperture_truncation.stateChanged.connect(self.update_layer)



    def update_config(self):

        # TODO: default values
        # try except block

        lens_config = {}
    
        #   core
        lens_config["height"] = float(self.lineEdit_height.text()) 
        lens_config["diameter"] = float(self.lineEdit_diameter.text()) 
        lens_config["exponent"] = float(self.lineEdit_exponent.text())
        lens_config["inverted"] = bool(self.checkBox_invert_profile.isChecked())
        lens_config["escape_path"] = float(self.lineEdit_escape_path.text())
        lens_config["length"] = float(self.lineEdit_length.text())
        lens_config["lens_type"] = self.comboBox_type.currentText()
        lens_config["name"] = self.lineEdit_name.text()
        lens_config["medium"] = float(self.lineEdit_medium.text())

        # print(lens_config["lens_type"])
        
        #   # grating
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
            lens_config["truncation"]["aperture"] = bool(self.checkBox_aperture_truncation.isChecked())
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

        # pprint(self.lens_config)


    def update_layer(self):

        # get updated config
        try:
            self.update_config()
        except Exception as e:
            print(f"Failure to read config values: {e}")


        lens = None
        arr3d = None

        try:
            # params
            medium = float(self.lineEdit_medium.text())
            pixelsize = float(self.lineEdit_pixelsize.text())

            lens = generate_lens(self.lens_config, Medium(medium), pixelsize)
            lens.apply_aperture_masks()
            arr3d = plotting.create_3d_lens(lens)

            if lens.grating_mask is None:
                lens.grating_mask = np.zeros_like(lens.profile)
            if lens.truncation_mask is None:
                lens.truncation_mask = np.zeros_like(lens.profile)

        except Exception as e:
            print(f"Failure to load 3d lens: {e}")
            arr3d = np.random.random(size=(1000, 1000))

            return


        if lens is None or arr3d is None:
            return        

        self.viewer.axes.visible = True
        self.viewer.axes.colored = False
        self.viewer.axes.dashed = True
        self.viewer.axes.labels = True
        self.viewer.scale_bar.visible = True


        # https://forum.image.sc/t/update-refresh-a-layer-in-napari-julia-napari-jl/59728
        print("update_layer")

        # TODO: check the lens and arr3d are valid here... otherwise dont update
        # TODO: set lens layer as active
        # TODO: find a better way to set the initial view iamges... probably better to separate
        # TODO: load initial values
        # TODO: load from config
        # TODO: load profile
        # TODO: do better validation, so that the viewer doesnt crash

        # update layer in place 
        try:
            try:
                self.viewer.layers["Lens"].data = arr3d
                self.viewer.layers["Aperture Mask"].data = lens.aperture
                self.viewer.layers["Grating Mask"].data = lens.grating_mask
                self.viewer.layers["Truncation Mask"].data = lens.truncation_mask

            except KeyError as e:
                # TODO: why doesnt this exist on the first pass?
                self.viewer.add_image(arr3d, name="Lens", colormap="gray", rendering="iso", depiction="volume")
                self.viewer.add_image(lens.aperture, name="Aperture Mask", opacity=0.4, colormap="yellow", rendering="translucent")
                self.viewer.add_image(lens.truncation_mask, name="Truncation Mask", opacity=0.4, colormap="cyan", rendering="translucent")
                self.viewer.add_image(lens.grating_mask, name="Grating Mask", opacity=0.4, colormap="green", rendering="translucent")
        except Exception as e:
            print(f"Failure to load viewer: {e}")

    


def main():
    """Launch the `piescope_gui` main application window."""
    application = QtWidgets.QApplication([])

    viewer = napari.Viewer(ndisplay=3)


    element_creation_ui = GUIElementCreation(viewer=viewer)                                          
    viewer.window.add_dock_widget(element_creation_ui, area='right')                  

    sys.exit(application.exec_())


if __name__ == "__main__":
    main()


