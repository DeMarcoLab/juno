
import os
import sys
import traceback
from pprint import pprint

import juno.ui.qtdesigner_files.BeamCreation as BeamCreation
import napari
import napari.utils.notifications
import numpy as np
import yaml
from juno import beam, plotting, utils, validation
from juno.beam import BeamShape, BeamSpread, DistanceMode
from juno.Simulation import (generate_beam_simulation_stage,
                             generate_simulation_parameters,
                             propagate_stage)
from juno.structures import SimulationOptions, SimulationParameters
from PyQt5 import QtWidgets
import dask.array as da

PROPAGATION_DISTANCE_DISPLAY_LIMIT_PX = 100_000
BEAM_SHAPE_DISPLAY_LIMIT_PX = 10_000
BEAM_N_STEPS_DISPLAY_LIMIT = 1000
BEAM_STEP_SIZE_DISPLAY_LIMIT = 1e-9

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
        self.update_visualisation()
        self.show()

    def setup_connections(self):

        # comboboxes (spread, shape, convergence, distance mode, propagation)
        self.comboBox_spread.addItems([spread.name for spread in BeamSpread])       # beam spread
        self.comboBox_shape.addItems([shape.name for shape in BeamShape])           # beam shape
        self.comboBox_distance_mode.addItems([mode.name for mode in DistanceMode])  # distance mode
        self.comboBox_convergence.addItems(["Theta", "Numerical Aperture"])         # convergence
        self.comboBox_propagation_type.addItems(["Num Steps", "Step Size"])

        # general
        self.lineEdit_name.textChanged.connect(self.update_visualisation)
        self.lineEdit_beam_width.textChanged.connect(self.update_visualisation)
        self.lineEdit_beam_height.textChanged.connect(self.update_visualisation)
        self.lineEdit_shift_x.textChanged.connect(self.update_visualisation)
        self.lineEdit_shift_y.textChanged.connect(self.update_visualisation)
        
        # shaping
        self.comboBox_spread.currentTextChanged.connect(self.update_visualisation)
        self.comboBox_shape.currentTextChanged.connect(self.update_visualisation)
        self.comboBox_convergence.currentTextChanged.connect(self.update_visualisation)
        self.lineEdit_convergence_value.textChanged.connect(self.update_visualisation)
        self.comboBox_distance_mode.currentTextChanged.connect(self.update_visualisation)
        self.lineEdit_distance_value.textChanged.connect(self.update_visualisation)

        # tilt
        self.lineEdit_tilt_x.textChanged.connect(self.update_visualisation)
        self.lineEdit_tilt_y.textChanged.connect(self.update_visualisation)

        # gaussian
        self.checkBox_gaussian_enabled.toggled.connect(self.update_visualisation)
        self.lineEdit_gaussian_waist_x.textChanged.connect(self.update_visualisation)
        self.lineEdit_gaussian_waist_y.textChanged.connect(self.update_visualisation)
        self.lineEdit_gaussian_axial_z0.textChanged.connect(self.update_visualisation)
        # self.lineEdit_gaussian_axial_z_total.textChanged.connect(self.update_visualisation)
        
        # simulation
        self.lineEdit_pixelsize.textChanged.connect(self.update_visualisation)
        self.lineEdit_sim_width.textChanged.connect(self.update_visualisation)
        self.lineEdit_sim_height.textChanged.connect(self.update_visualisation)
        self.comboBox_propagation_type.currentTextChanged.connect(self.update_visualisation)
        self.lineEdit_propagation_step.textChanged.connect(self.update_visualisation)
        self.lineEdit_medium.textChanged.connect(self.update_visualisation)

        # buttons
        self.pushButton_generate_beam.clicked.connect(self.update_visualisation)
        
        # actions
        self.actionLoad_Configuration.triggered.connect(self.load_configuration)
        self.actionSave_Configuration.triggered.connect(self.save_configuration)

    def update_ui_from_config(self, config: dict):
        # read the config, update ui elements...

        self.CONFIG_UPDATE = True

        # general
        self.lineEdit_beam_width.setText(str(config["width"]))
        self.lineEdit_beam_height.setText(str(config["height"]))
        self.lineEdit_shift_x.setText(str(config["position_x"]))
        self.lineEdit_shift_y.setText(str(config["position_y"]))

        self.comboBox_shape.setCurrentText(str(config["shape"]))
        self.comboBox_spread.setCurrentText(str(config["spread"]))

        if config["numerical_aperture"] is None:
            self.comboBox_convergence.setCurrentText("Theta")
            self.lineEdit_convergence_value.setText(str(config["theta"]))       
        else:
            self.comboBox_convergence.setCurrentText("Numerical Aperture")
            self.lineEdit_convergence_value.setText(str(config["numerical_aperture"]))

        self.lineEdit_tilt_x.setText(str(config["tilt_x"]))
        self.lineEdit_tilt_y.setText(str(config["tilt_y"]))


        self.comboBox_distance_mode.setCurrentText(str(config["distance_mode"]))    
        if config["distance_mode"] == "Direct":
            distance_value = config["source_distance"]
        if config["distance_mode"] == "Diameter":
            distance_value = config["final_diameter"]
        if config["distance_mode"] == "Focal":
            distance_value = config["focal_multiple"]
        self.lineEdit_distance_value.setText(str(distance_value))
        

        # gaussian
        use_gaussian = config["operator"] == "Gaussian"
        self.checkBox_gaussian_enabled.setChecked(use_gaussian)
        self.lineEdit_gaussian_waist_x.setText(str(config["gaussian_wx"]))
        self.lineEdit_gaussian_waist_y.setText(str(config["gaussian_wy"]))
        self.lineEdit_gaussian_axial_z0.setText(str(config["gaussian_z0"]))
        # self.lineEdit_gaussian_axial_z_total.setText(str(config["gaussian_z"]))

        # simulation
        if config["step_size"] is None:
            self.comboBox_propagation_type.setCurrentText("Step Size")
            self.lineEdit_propagation_step.setText(str(config["step_size"]))
        else:
            self.comboBox_propagation_type.setCurrentText("Num Steps")
            self.lineEdit_propagation_step.setText(str(config["n_steps"]))


        self.update_ui_components()

        self.CONFIG_UPDATE = False

        return


    def update_ui_components(self):
        """enable / disable general components"""

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
            

    def load_configuration(self):
        """Load the beam configuration from file"""
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Configuration",
            filter="All types (*.yml *.yaml *.npy) ;;Yaml config (*.yml *.yaml);;",
        )

        if filename == "":
            return

        try:
            # load beam config and validate
            beam_config = utils.load_yaml_config(filename)
            beam_config = validation._validate_default_beam_config(beam_config)

            # update ui
            self.update_ui_from_config(beam_config)
            self.update_visualisation() 
        except:
            napari.utils.notifications.show_error(traceback.format_exc())
            

    def save_configuration(self):

        try:
            self.update_config()
        except Exception as e:
            napari.utils.notifications.show_error(f"Unable to update config... {e}")

        filename, ext = QtWidgets.QFileDialog.getSaveFileName(self, "Save Configuration", self.lineEdit_name.text(), filter="Yaml config (*.yml *.yaml)")

        if filename == "":
            return

        self.lineEdit_name.setText(os.path.basename(filename).split('.')[0])

        with open(filename, "w") as f:
            yaml.safe_dump(self.config["beam"], f, sort_keys=False)

    def update_config(self):

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

        # gaussian
        if self.checkBox_gaussian_enabled.isChecked():
            beam_config["operator"] = "Gaussian"
            beam_config["gaussian_wx"] = float(self.lineEdit_gaussian_waist_x.text())
            beam_config["gaussian_wy"] = float(self.lineEdit_gaussian_waist_y.text())
            beam_config["gaussian_z0"] = float(self.lineEdit_gaussian_axial_z0.text())
            # beam_config["gaussian_z"] = float(self.lineEdit_gaussian_axial_z_total.text())

        # sim parameters
        parameters_config["A"] = float(self.lineEdit_sim_amplitude.text())
        parameters_config["pixel_size"] = float(self.lineEdit_pixelsize.text())
        parameters_config["sim_width"] = float(self.lineEdit_sim_width.text())
        parameters_config["sim_height"] = float(self.lineEdit_sim_height.text())
        parameters_config["sim_wavelength"] = float(self.lineEdit_sim_wavelength.text())

        self.config = {
            "sim_parameters": parameters_config,
            "beam": beam_config,
        }


    def update_visualisation(self):
        
        # update ui components
        self.update_ui_components()

        # dont update the layers when the config is updating the ui...
        if self.CONFIG_UPDATE:
            return

        # only update when button pressed, if not live-updating
        if self.checkBox_live_update.isChecked() is False:
            if self.sender() is not self.pushButton_generate_beam:
                return

        # get updated config
        try:
            self.update_config()
        except Exception as e:
            napari.utils.notifications.show_error(f"Error reading configuration ui: {traceback.format_exc()}")
            return

        try:

            options = SimulationOptions(log_dir=os.path.join(os.path.dirname(__file__), "tmp"), 
            save_plot=False)

            parameters = generate_simulation_parameters(self.config)


            # validate beam shape for display
            if not validate_beam_for_display(self.config, parameters):
                napari.utils.notifications.show_error(f"Beam / Simulation size is too large to display.")
                return 

            # generate hte beams
            stage = generate_beam_simulation_stage(self.config, parameters)

            # validate sim size for display
            if np.max(stage.distances.shape) > PROPAGATION_DISTANCE_DISPLAY_LIMIT_PX:
                napari.utils.notifications.show_error(f"Beam propagation distance to display. {stage.distances.shape} elements")
                return 

            # beam propagation
            result = propagate_stage(stage, parameters, options, None)
           
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
            sim = da.from_zarr(utils.load_simulation(path))

        except:
            napari.utils.notifications.show_error(f"Failure to load simulation: {traceback.format_exc()}")
            return

        SCALE_DIM = 10
        # update layer in place 
        try:
            try:
                self.viewer.layers["Propagation"].data = sim
            except KeyError as e:
                self.viewer.add_image(sim, name="Propagation", colormap="turbo", rendering="average", depiction="volume", scale=[SCALE_DIM, 1, 1])
        except Exception as e:
            napari.utils.notifications.show_error(f"Failure to load viewer: {traceback.format_exc()}")
            return

def validate_beam_for_display(config: dict, parameters: SimulationParameters) -> bool:
    
    valid_beam: bool = True

    if parameters.sim_height / parameters.pixel_size > BEAM_SHAPE_DISPLAY_LIMIT_PX:
        valid_beam = False
    if parameters.sim_width / parameters.pixel_size > BEAM_SHAPE_DISPLAY_LIMIT_PX:
        valid_beam = False

    for k in ["width", "height"]:
        if config["beam"][k] / parameters.pixel_size > BEAM_SHAPE_DISPLAY_LIMIT_PX:
            valid_beam = False

    if config["beam"]["n_steps"] > BEAM_N_STEPS_DISPLAY_LIMIT:
        valid_beam = False
    if "step_size" in config["beam"]:
        if config["beam"]["step_size"] < BEAM_STEP_SIZE_DISPLAY_LIMIT:
            valid_beam = False    

    return valid_beam


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
