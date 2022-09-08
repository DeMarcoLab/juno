
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
from juno.structures import SimulationOptions
from PyQt5 import QtWidgets


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

        # gaussian
        self.checkBox_gaussian_enabled.toggled.connect(self.update_layer)
        self.lineEdit_gaussian_waist_x.textChanged.connect(self.update_layer)
        self.lineEdit_gaussian_waist_y.textChanged.connect(self.update_layer)
        self.lineEdit_gaussian_axial_z0.textChanged.connect(self.update_layer)
        self.lineEdit_gaussian_axial_z_total.textChanged.connect(self.update_layer)
        
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

    # def testing_function(self):

    #     print("testing function!!")

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
        self.lineEdit_gaussian_axial_z_total.setText(str(config["gaussian_z"]))

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
            

    def load_config(self):

        print("load config...")

        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Configuration",
            filter="All types (*.yml *.yaml *.npy) ;;Yaml config (*.yml *.yaml) ;;Numpy array (*.npy)",
        )

        if filename == "":
            return

        print("beam configuration")
        beam_config = utils.load_yaml_config(filename)
        
        # validate config...
        beam_config = validation._validate_default_beam_config(beam_config)

        # validate config?
        try:
            self.update_ui_from_config(beam_config)
        except:
            napari.utils.notifications.show_error(traceback.format_exc())
            
        self.update_layer() 

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
            beam_config["gaussian_z"] = float(self.lineEdit_gaussian_axial_z_total.text())

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

            result = propagate_stage(stage, parameters, options, None)


            # TODO: use the actual propgation from sim, not this mess
            # if stage.wavefront is not None:
            #     previous_wavefront = stage.wavefront

            # # calculate stage phase profile
            # phase = calculate_stage_phase(stage, parameters)

            # # electric field (wavefront)
            # amplitude: float = parameters.A if stage._id == 0 else 1.0
            # wavefront = calculate_wavefront_v2(
            #     phase=phase,
            #     previous_wavefront=previous_wavefront,
            #     A=amplitude,
            #     aperture=stage.lens.aperture,
            # ) 

            # ## propagate wavefront #TODO: replace with v3 (vectorised)
            # result = propagate_wavefront_v2(wavefront=wavefront, 
            #                     stage=stage, 
            #                     parameters=parameters, 
            #                     options=options)
            
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
            import dask.array as da
            sim = da.from_zarr(utils.load_simulation(path))

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
