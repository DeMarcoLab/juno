
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
from juno import beam

import juno.ui.qtdesigner_files.BeamCreation as BeamCreation
import napari
import numpy as np
import yaml
from juno import plotting, utils, validation
from PyQt5 import QtWidgets
import napari.utils.notifications

from juno.Simulation import (generate_beam_simulation_stage, calculate_stage_phase, 
        calculate_wavefront_v2, propagate_wavefront_v2)
from juno.structures import SimulationOptions
from juno.Simulation import generate_simulation_parameters


from juno.beam import Beam, BeamShape, BeamSpread, DistanceMode, generate_beam 

class GUIBeamCreation(BeamCreation.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self, parent=None, viewer: napari.Viewer = None):
        super().__init__(parent=parent)
        self.setupUi(self)
        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)
        self.setWindowTitle("Beam Creation")

        self.CONFIG_UPDATE = False

        self.viewer = viewer
        self.setup_connections()

        # # update ui parameters and viz
        # self.update_ui_from_config(validated_default_config)
        self.update_layer()
        self.show()

    def setup_connections(self):

        # comboboxes (spread, shape, convergence, distance mode, propagation)
        self.comboBox_spread.addItems([spread.name for spread in BeamSpread])       # beam spread
        self.comboBox_shape.addItems([shape.name for shape in BeamShape])           # beam shape
        self.comboBox_distance_mode.addItems([mode.name for mode in DistanceMode])  # distance mode
        self.comboBox_convergence.addItems(["Theta", "Numerical Aperture"])         # convergence
        self.comboBox_propagation_type.addItems(["Num Steps", "Step Size"])

        # general
        self.lineEdit_name.textChanged.connect(self.update_layer)
        self.lineEdit_beam_width.textChanged.connect(self.update_layer)
        self.lineEdit_beam_height.textChanged.connect(self.update_layer)
        self.lineEdit_shift_x.textChanged.connect(self.update_layer)
        self.lineEdit_shift_y.textChanged.connect(self.update_layer)
        
        # shaping
        self.comboBox_spread.currentTextChanged.connect(self.update_layer)
        self.comboBox_shape.currentTextChanged.connect(self.update_layer)
        self.comboBox_convergence.currentTextChanged.connect(self.update_layer)
        self.lineEdit_convergence_value.textChanged.connect(self.update_layer)
        self.comboBox_distance_mode.currentTextChanged.connect(self.update_layer)
        self.lineEdit_distance_value.textChanged.connect(self.update_layer)

        # tilt
        self.lineEdit_tilt_x.textChanged.connect(self.update_layer)
        self.lineEdit_tilt_y.textChanged.connect(self.update_layer)

        # simulation
        self.lineEdit_pixelsize.textChanged.connect(self.update_layer)
        self.lineEdit_sim_width.textChanged.connect(self.update_layer)
        self.lineEdit_sim_height.textChanged.connect(self.update_layer)
        self.comboBox_propagation_type.currentTextChanged.connect(self.update_layer)
        self.lineEdit_propagation_step.textChanged.connect(self.update_layer)
        self.lineEdit_medium.textChanged.connect(self.update_layer)

        # buttons
        self.pushButton_generate_beam.clicked.connect(self.update_layer)
        self.pushButton_load_config.clicked.connect(self.load_config)
        self.pushButton_save_config.clicked.connect(self.save_config)


    def load_config(self):

        print("load config")

    def save_config(self):

        print("save config")

    # def testing_function(self):

    #     print("testing function!!")

    # def update_ui_from_config(self, config: dict):

    #     self.CONFIG_UPDATE = True

    #     # general
    #     self.lineEdit_pixelsize.setText(str(1e-6))
    #     self.lineEdit_diameter.setText(str(config["diameter"]))
    #     self.lineEdit_length.setText(str(config["length"]))
    #     self.lineEdit_height.setText(str(config["height"]))
    #     self.lineEdit_medium.setText(str(config["medium"]))
    #     self.lineEdit_exponent.setText(str(config["exponent"]))
    #     self.lineEdit_escape_path.setText(str(config["escape_path"]))
    #     self.comboBox_type.setCurrentText(str(config["lens_type"]).capitalize())
    #     self.checkBox_invert_profile.setChecked(bool(config["inverted"]))
    #     self.lineEdit_name.setText(str(config["name"]))

    #     # grating
    #     use_grating =  bool(config["grating"])
    #     self.checkBox_use_grating.setChecked(use_grating)
    #     if use_grating:
    #         self.lineEdit_grating_width.setText(str(config["grating"]["width"]))
    #         self.lineEdit_grating_distance.setText(str(config["grating"]["distance"]))
    #         self.lineEdit_grating_depth.setText(str(config["grating"]["depth"]))
    #         self.checkBox_grating_x_axis.setChecked(bool(config["grating"]["x"]))
    #         self.checkBox_grating_y_axis.setChecked(bool(config["grating"]["y"]))
    #         self.checkBox_grating_centred.setChecked(bool(config["grating"]["centred"]))

    #     # # truncation
    #     use_truncation = bool(config["truncation"])
    #     self.checkBox_use_truncation.setChecked(use_truncation)
    #     if use_truncation:
    #         truncation_mode = str(config["truncation"]["type"])
    #         self.comboBox_truncation_mode.setCurrentText(truncation_mode.capitalize())
    #         if truncation_mode == "value":
    #             self.lineEdit_truncation_value.setText(str(config["truncation"]["height"]))
    #         if truncation_mode == "radial":
    #             self.lineEdit_truncation_value.setText(str(config["truncation"]["radius"]))
    #         self.checkBox_truncation_aperture.setChecked(bool(config["truncation"]["aperture"]))


    #     # # aperture
    #     use_aperture = bool(config["aperture"])
    #     self.checkBox_use_aperture.setChecked(use_aperture)
    #     if use_aperture:
    #         aperture_mode = str(config["aperture"]["type"])
    #         self.comboBox_aperture_mode.setCurrentText(aperture_mode.capitalize())
    #         self.lineEdit_aperture_inner.setText(str(config["aperture"]["inner"]))
    #         self.lineEdit_aperture_outer.setText(str(config["aperture"]["outer"]))
    #         self.checkBox_aperture_invert.setChecked(bool(config["aperture"]["invert"]))

    #     # self.update_ui_components()

    #     self.CONFIG_UPDATE = False

    #     return


    def update_ui_components(self):

        # enable / disable general components
        print("updating ui components...")


         # spread
         # TODO: also toggle the labels....
        if self.comboBox_spread.currentText() == "Plane":
            self.comboBox_convergence.setEnabled(False)
            self.lineEdit_convergence_value.setEnabled(False)
            self.comboBox_distance_mode.setEnabled(False)
            self.comboBox_distance_mode.setCurrentText(DistanceMode.Direct.name)
        else:
            self.comboBox_convergence.setEnabled(True)
            self.lineEdit_convergence_value.setEnabled(True)
            self.comboBox_distance_mode.setEnabled(True)
            self.comboBox_shape.setCurrentText(BeamShape.Circular.name)

        # convergence
        if self.comboBox_convergence.currentText() == "Theta":
            self.label_convergence_value.setText("Theta (deg)")
        else:
            self.label_convergence_value.setText("NA")


        # distance
        if self.comboBox_distance_mode.currentText() == "Direct":
            self.label_distance_value.setText("Distance (m)")
        if self.comboBox_distance_mode.currentText() == "Diameter":
            self.label_distance_value.setText("Final Diameter (m)")
        if self.comboBox_distance_mode.currentText() == "Focal":
            self.label_distance_value.setText("Focal Multiple")
            

    #     # enable / disable grating components
    #     use_grating = self.checkBox_use_grating.isChecked()
    #     self.lineEdit_grating_width.setEnabled(use_grating)
    #     self.lineEdit_grating_distance.setEnabled(use_grating)
    #     self.lineEdit_grating_depth.setEnabled(use_grating)
    #     self.checkBox_grating_x_axis.setEnabled(use_grating)
    #     self.checkBox_grating_y_axis.setEnabled(use_grating)
    #     self.checkBox_grating_centred.setEnabled(use_grating)

    #     # enable / disable truncation components
    #     use_truncation = self.checkBox_use_truncation.isChecked()
    #     self.comboBox_truncation_mode.setEnabled(use_truncation)
    #     self.lineEdit_truncation_value.setEnabled(use_truncation)
    #     self.lineEdit_truncation_value.setEnabled(use_truncation)
    #     self.checkBox_truncation_aperture.setEnabled(use_truncation)

    #     # enable / disable aperture components   
    #     use_aperture = self.checkBox_use_aperture.isChecked()
    #     self.comboBox_aperture_mode.setEnabled(use_aperture)
    #     self.lineEdit_aperture_inner.setEnabled(use_aperture)
    #     self.lineEdit_aperture_outer.setEnabled(use_aperture)
    #     self.checkBox_aperture_invert.setEnabled(use_aperture)

    # def load_profile(self):

    #     print("load profile...")

    #     filename, _ = QtWidgets.QFileDialog.getOpenFileName(
    #         self,
    #         "Load Profile",
    #         filter="All types (*.yml *.yaml *.npy) ;;Yaml config (*.yml *.yaml) ;;Numpy array (*.npy)",
    #     )

    #     if filename == "":
    #         return

    #     if filename.endswith(".npy"):
    #         print("numpy file... TODO: custom profile load")

    #     else:
    #         print("lens configuration")
    #         self.lens_config = utils.load_yaml_config(filename)
            
    #         # validate config...
    #         self.lens_config = validation._validate_default_lens_config(self.lens_config)

    #         # validate config?
    #         try:
    #             self.update_ui_from_config(self.lens_config)
    #         except:
    #             napari.utils.notifications.show_error(traceback.format_exc())
            
    #         self.update_layer() 

    def save_config(self):

        try:
            self.update_config()
        except Exception as e:
            print(f"Unable to update config... {e}")

        filename, ext = QtWidgets.QFileDialog.getSaveFileName(self, "Save Profile", self.lineEdit_name.text(), filter="Yaml config (*.yml *.yaml)")

        if filename == "":
            return

        self.lineEdit_name.setText(os.path.basename(filename).split('.')[0])

        with open(filename, "w") as f:
            yaml.safe_dump(self.config["beam"], f, sort_keys=False)

    def update_config(self):

    #     # try except block

        beam_config = {}
        parameters_config =  {}
    
        # core
        beam_config["width"] = float(self.lineEdit_beam_width.text())
        beam_config["height"] = float(self.lineEdit_beam_height.text())
        beam_config["position_x"] = float(self.lineEdit_shift_x.text())
        beam_config["position_y"] = float(self.lineEdit_shift_y.text())

        # shaping
        beam_config["spread"] = str(self.comboBox_spread.currentText())
        beam_config["shape"] = str(self.comboBox_shape.currentText())
        
        # distance mode
        beam_config["distance_mode"] = str(self.comboBox_distance_mode.currentText())
        if beam_config["distance_mode"] == "Direct":
            beam_config["source_distance"] = float(self.lineEdit_distance_value.text())
        if beam_config["distance_mode"] == "Diameter":
            beam_config["final_diameter"] = float(self.lineEdit_distance_value.text())
        if beam_config["distance_mode"] == "Focal":
            beam_config["focal_multiple"] = float(self.lineEdit_distance_value.text())

        # convergence
        if self.comboBox_convergence.currentText() == "Theta":
            beam_config["theta"]  = float(self.lineEdit_convergence_value.text())
        else:
            beam_config["numerical_aperture"]  = float(self.lineEdit_convergence_value.text())

        # tilt
        beam_config["tilt_x"] = float(self.lineEdit_tilt_x.text())
        beam_config["tilt_y"] = float(self.lineEdit_tilt_y.text())

        # propagation steps
        if self.comboBox_propagation_type.currentText() == "Num Steps":
            beam_config["n_steps"] = int(self.lineEdit_propagation_step.text())
        if self.comboBox_propagation_type.currentText() == "Step Size":
            beam_config["step_size"] = float(self.lineEdit_propagation_step.text())

        # sim parameters
        parameters_config["A"] = 10000
        parameters_config["pixel_size"] = float(self.lineEdit_pixelsize.text())
        parameters_config["sim_width"] = float(self.lineEdit_sim_width.text())
        parameters_config["sim_height"] = float(self.lineEdit_sim_height.text())
        parameters_config["sim_wavelength"] = 488.e-9

        self.config = {
            "sim_parameters": parameters_config,
            "beam": beam_config,
        }


    def update_layer(self):

        print("updating layer")
        self.update_ui_components()

        # dont update the layers when the config is updating the ui...
        if self.CONFIG_UPDATE:
            return

        # get updated config
        try:
            self.update_config()
        except Exception as e:
            napari.utils.notifications.show_error(f"ERROR: {traceback.format_exc()}")
            return

        pprint(self.config)

        try:

            options = SimulationOptions(log_dir=os.path.join(os.path.dirname(__file__), "tmp"), 
            save_plot=False)

            parameters = generate_simulation_parameters(self.config)
            stage = generate_beam_simulation_stage(self.config, parameters)

            if stage.wavefront is not None:
                propagation = stage.wavefront

                previous_wavefront = propagation

                # calculate stage phase profile
                phase = calculate_stage_phase(stage, parameters)

                # electric field (wavefront)
                amplitude: float = parameters.A if stage._id == 0 else 1.0
                wavefront = calculate_wavefront_v2(
                    phase=phase,
                    previous_wavefront=previous_wavefront,
                    A=amplitude,
                    aperture=stage.lens.aperture,
                ) 

                ## propagate wavefront #TODO: replace with v3 (vectorised)
                result = propagate_wavefront_v2(wavefront=wavefront, 
                                    stage=stage, 
                                    parameters=parameters, 
                                    options=options)
                
                # pass the wavefront to the next stage
                propagation = result.propagation

        except:
            napari.utils.notifications.show_error(f"Failure to propagate wavefron: {traceback.format_exc()}")
            return

    
        self.viewer.dims.ndisplay = 3
        self.viewer.axes.visible = True
        self.viewer.axes.colored = False
        self.viewer.axes.dashed = True
        self.viewer.axes.labels = True
        self.viewer.scale_bar.visible = True

        # load beam propgation
    
        try:
            path = os.path.join(options.log_dir, str(stage._id), "sim.zarr")
            print("path:", path)
            sim = utils.load_simulation(path)

        except:
            napari.utils.notifications.show_error(f"Failure to load simulation: {traceback.format_exc()}")
            return

        SCALE_DIM = 100

        # update layer in place 
        try:
            try:
                self.viewer.layers["Propagation"].data = sim
            except KeyError as e:
                self.viewer.add_image(sim, name="Propagation", colormap="turbo", rendering="average", depiction="volume", scale=[SCALE_DIM, 1, 1])
        except Exception as e:
            napari.utils.notifications.show_error(f"Failure to load viewer: {traceback.format_exc()}")
            return

def main():
    """Launch the `piescope_gui` main application window."""
    application = QtWidgets.QApplication([])

    viewer = napari.Viewer(ndisplay=3)
    beam_creation_ui = GUIBeamCreation(viewer=viewer)                                          
    viewer.window.add_dock_widget(beam_creation_ui, area='right')                  

    sys.exit(application.exec_())


if __name__ == "__main__":
    main()




# TODO: look into
# https://github.com/napari/napari/blob/3c7cb3af367edac361baf4a7e25d929a02f6e99f/napari/_qt/qt_viewer.py