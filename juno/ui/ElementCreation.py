
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

        # lens types
        self.comboBox_type.addItems([type.name for type in LensType])


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
        #   if grating:
        #     lens_config["grating"] = {}
        #     lens_config["grating"]["width"] = grating_width * MICRON_TO_METRE
        #     lens_config["grating"]["distance"] = grating_distance * MICRON_TO_METRE
        #     lens_config["grating"]["depth"] = grating_depth * MICRON_TO_METRE
        #     lens_config["grating"]["x"] = grating_x 
        #     lens_config["grating"]["y"] = grating_y 
        #     lens_config["grating"]["centred"] = grating_centred 
        #   else:
        lens_config["grating"] = None
        
        # truncation
        # if truncation:
        # lens_config["truncation"] = {}
        # lens_config["truncation"]["height"] = truncation_height * MICRON_TO_METRE
        # lens_config["truncation"]["radius"] = truncation_radius * MICRON_TO_METRE
        # lens_config["truncation"]["type"] = truncation_type 
        # lens_config["truncation"]["aperture"] = truncation_aperture
        # else:
        lens_config["truncation"] = None

        # aperture
        # if aperture:
        #     lens_config["aperture"] = {}
        #     lens_config["aperture"]["inner"] = aperture_inner * MICRON_TO_METRE
        #     lens_config["aperture"]["outer"] = aperture_outer * MICRON_TO_METRE
        #     lens_config["aperture"]["type"] = aperture_type 
        #     lens_config["aperture"]["invert"] = aperture_invert
        # else:
        lens_config["aperture"] = None

        self.lens_config = lens_config


        pprint(self.lens_config)


    def update_layer(self):

        # get updated config
        try:
            self.update_config()
        except Exception as e:
            print(f"Failure to read config values: {e}")



        try:
            # params
            medium = float(self.lineEdit_medium.text())
            pixelsize = float(self.lineEdit_pixelsize.text())

            lens = generate_lens(self.lens_config, Medium(medium), pixelsize)
            arr3d = plotting.create_3d_lens(lens)

        except Exception as e:
            print(f"Failure to load 3d lens: {e}")
            arr3d = np.random.random(size=(1000, 1000))



    # return [(arr3d, {"name": "Lens", "colormap": "gray", "rendering": "iso", "depiction": "volume"}),
    #         (lens.aperture, {"name": "Lens Aperture", "opacity": 0.4, "colormap": "yellow", "rendering": "translucent"}, "image"),
    #         (lens.grating_mask, {"name": "Lens Grating Mask", "opacity": 0.4, "colormap": "green", "rendering": "translucent"}, "image"),
    #         (lens.truncation_mask, {"name": "Lens Truncation Mask", "opacity": 0.4, "colormap": "cyan", "rendering": "translucent"}, "image")]

        self.viewer.axes.visible = True
        self.viewer.axes.colored = False
        self.viewer.axes.dashed = True
        self.viewer.axes.labels = True
        self.viewer.scale_bar.visible = True


        # https://forum.image.sc/t/update-refresh-a-layer-in-napari-julia-napari-jl/59728
        print("update_layer")

        # update layer in place 
        try:
            self.viewer.layers["Lens"].data = arr3d
        except KeyError:
            self.viewer.add_image(arr3d, name="Lens", colormap="gray", rendering="iso", depiction="volume")
    


def main():
    """Launch the `piescope_gui` main application window."""
    application = QtWidgets.QApplication([])

    viewer = napari.Viewer(ndisplay=3)


    element_creation_ui = GUIElementCreation(viewer=viewer)                                          
    viewer.window.add_dock_widget(element_creation_ui, area='right')                  

    sys.exit(application.exec_())


if __name__ == "__main__":
    main()


def generate_3d_lens(config: dict, medium: float, pixel_size: float):


    return arr3d