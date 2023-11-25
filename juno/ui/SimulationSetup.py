import os
import sys
import traceback
from pathlib import Path
from pprint import pprint

import juno
import juno.ui.qtdesigner_files.SimulationSetup as SimulationSetup
import yaml
from juno import plotting, utils, validation
from juno.ui.ParameterSweep import GUIParameterSweep
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (QCheckBox, QFileDialog, QGridLayout, QGroupBox,
                             QLabel, QLineEdit, QPushButton, QVBoxLayout)
import napari
import napari.utils.notifications
import  numpy as np

SIM_ELEMENTS_SIZE_DISPLAY_LIMIT = 10_000_000_000
BASE_PATH = os.path.dirname(juno.__file__)

class GUISimulationSetup(SimulationSetup.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self, viewer: napari.Viewer = None, parent_gui=None):
        super().__init__(parent=parent_gui)
        self.setupUi(MainWindow=self)
        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)
        self.setWindowTitle("Simulation Setup")

        self.viewer: napari.Viewer = viewer

        self.simulation_config = {}
        self.SAVE_FROM_PARAMETER_SWEEP = False
        self.input_widgets = []
        self._validated_simulation_config = False

        self.setup_connections()

        self.update_all_displays()

        self._update_ui_components()

    def setup_connections(self):

        self.spinBox_sim_num_stages.valueChanged.connect(
            self.update_stage_input_display
        )   

        # buttons
        self.pushButton_generate_simulation.clicked.connect(
            self.generate_simulation_config
        )
        self.pushButton_sim_beam.clicked.connect(self.load_beam_config)
        self.pushButton_setup_parameter_sweep.clicked.connect(self.setup_parameter_sweep)
        self.pushButton_visualise_simulation.clicked.connect(self.visualise_simulation)

        # actions
        self.actionLoad_Configuration.triggered.connect(self.load_simulation_config)
        self.actionSave_Configuration.triggered.connect(self.save_simulation_config)

    def _update_ui_components(self):

        # clear viewer
        self.viewer.layers.clear()

        # enable parameter sweep
        self.pushButton_setup_parameter_sweep.setVisible(self._validated_simulation_config)
        self.pushButton_setup_parameter_sweep.setEnabled(self._validated_simulation_config)
        
        # enable saving
        self.actionSave_Configuration.setEnabled(self._validated_simulation_config)
        
        # enable visualisation
        self.label_visualisation_scale.setVisible(self._validated_simulation_config)
        self.spinBox_visualisation_scale.setVisible(self._validated_simulation_config)
        self.pushButton_visualise_simulation.setVisible(self._validated_simulation_config)
        self.pushButton_visualise_simulation.setEnabled(self._validated_simulation_config)


    def update_all_displays(self):

        self.update_stage_input_display()

    def update_simulation_config(self):

        sim_pixel_size = float(self.lineEdit_pixel_size.text())
        sim_width = float(self.lineEdit_sim_width.text())
        sim_height = float(self.lineEdit_sim_height.text())
        sim_wavelength = (float(self.lineEdit_sim_wavelength.text()))
        sim_amplitude = float(self.lineEdit_sim_amplitude.text())

        self.simulation_config["sim_parameters"] = {
            "A": sim_amplitude,
            "pixel_size": sim_pixel_size,
            "sim_height": sim_height,
            "sim_width": sim_width,
            "sim_wavelength": sim_wavelength,
        }

        sim_name = str(self.lineEdit_sim_name.text())
        sim_name = None if sim_name == "" else sim_name
        log_dir = str(self.lineEdit_log_dir.text())
        save_plot = bool(self.checkBox_save_plot.isChecked())

        self.simulation_config["options"] = {
            "name": sim_name,
            "log_dir": log_dir,
            "save_plot": save_plot,
        }

    def input_widget_state_change(self):

        for stage_no, widgets in enumerate(self.input_widgets, 1):

            if widgets[10].isChecked():
                widgets[11].setText("Start Multiple")
                widgets[13].setText("Finish Multiple")
            else:
                widgets[11].setText("Start Distance")
                widgets[13].setText("Finish Distance")
    
    def setup_parameter_sweep(self):
        # param sweep ui
        self.param_sweep_ui = GUIParameterSweep(self.simulation_config, parent_gui=self)
        
    def save_simulation_config(self):
        
        # open file dialog
        sim_config_filename, _ = QFileDialog.getSaveFileName(self,
                    caption="Save Simulation Configuration",
                    directory=os.path.dirname(BASE_PATH),
                    filter="Yaml files (*.yml *.yaml)")

        if sim_config_filename:
            # set name
            # same as yaml file
            with open(sim_config_filename, "w") as f:
                yaml.safe_dump(self.simulation_config, f)

            napari.utils.notifications.show_info(f"Simulation config saved to {sim_config_filename}")

    def load_simulation_config(self):

        # open file dialog
        sim_config_filename, _ = QFileDialog.getOpenFileName(self,
                    caption="Load Simulation Config",
                    directory=os.path.dirname(BASE_PATH),
                    filter="Yaml files (*.yml *.yaml)"
                    )
        if sim_config_filename:

            config = utils.load_config(sim_config_filename)

            self.update_status(f"Simulation config loaded from {sim_config_filename}")

            # TODO: how to handle partial configs??? throw error? this will fail if sim isnt valid...

            # load config values into ui....
            self.simulation_config = config

            # set sim parameters
            self.lineEdit_pixel_size.setText(str(config["sim_parameters"]["pixel_size"]))
            self.lineEdit_sim_width.setText(str(config["sim_parameters"]["sim_width"]))
            self.lineEdit_sim_height.setText(str(config["sim_parameters"]["sim_height"]) )
            self.lineEdit_sim_wavelength.setText(str(config["sim_parameters"]["sim_wavelength"]) )
            self.lineEdit_sim_amplitude.setText(str(config["sim_parameters"]["A"]))

            # set sim optiosn
            self.lineEdit_log_dir.setText(str(config["options"]["log_dir"]))
            self.checkBox_save_plot.setChecked(bool(config["options"]["save_plot"]))

            # set beam
            self.pushButton_sim_beam.setText("beam")

            # set n stages
            self.spinBox_sim_num_stages.setValue(int(len(config["stages"])))

            # update each stage info
            load_stage_config_widgets(config, self.input_widgets)
            self.SIMULATION_CONFIG_LOADED = True

            # update ui
            self._validated_simulation_config = False
            self._update_ui_components()

    def update_status(self, msg= "Generating Simulation Configuration..."):
        """update status within button press..."""
        napari.utils.notifications.show_info(msg)


    def generate_simulation_config(self):
        # TODO: need to check if things are loaded...
        
        napari.utils.notifications.show_info("Generating Simulation Configuration...")
        try:

            self.update_simulation_config()
            self.read_stage_input_values()

            validation._validate_simulation_config(self.simulation_config)
            napari.utils.notifications.show_info(f"Valid Simulation Configuration.")

            # update ui            
            self._validated_simulation_config = True
            self._update_ui_components()

            napari.utils.notifications.show_info(f"Generate Simulation Configuration Finished.")

        except Exception as e:
            napari.utils.notifications.show_error(f"Invalid simulation config. \n{traceback.format_exc()}")
            self.pushButton_setup_parameter_sweep.setEnabled(False)
            self.actionSave_Configuration.setEnabled(False)



    def visualise_simulation(self):

        self.pushButton_visualise_simulation.setEnabled(False)
        self.pushButton_visualise_simulation.setText("Visualising...")
        self.pushButton_visualise_simulation.setStyleSheet("background-color: orange")

        # TODO: find a way to make the viewing of this much more performant
        pixel_size = self.simulation_config["sim_parameters"]["pixel_size"] 
        self.simulation_config["sim_parameters"]["pixel_size"] *= self.spinBox_visualisation_scale.value() # scale visualisation

        arr_elements = plotting.plot_simulation_setup_v2(self.simulation_config)
        arr_medium = plotting.plot_simulation_setup_v2(self.simulation_config, medium_only=True)

        if (arr_elements.size + arr_medium.size) > SIM_ELEMENTS_SIZE_DISPLAY_LIMIT:
            self.viewer.dims.ndisplay = 2
        else:
            self.viewer.dims.ndisplay = 3

        # 3D mode warning
        GL_MAX_TEXTURE_SIZE = 2048 # gpu dependant?
        if np.max(arr_elements.shape) > GL_MAX_TEXTURE_SIZE:
            napari.utils.notifications.show_warning(f"Data shape {arr_elements.shape} exceeds GL_MAX_TEXTURE_SIZE {GL_MAX_TEXTURE_SIZE}. It is recommended to downscale the visualisation if viewing in 3D mode.")
        try:
            self.viewer.layers.clear()
            self.viewer.add_image(arr_medium, name="Simulation Medium", colormap="turbo", opacity=0.5, rendering="translucent")
            self.viewer.add_image(arr_elements, name="Simulation Elements", colormap="gray", rendering="iso")
               
        except Exception as e:
            napari.utils.notifications.show_error(f"Failure to load viewer: {traceback.format_exc()}")

        # reset pixel_size
        self.simulation_config["sim_parameters"]["pixel_size"] = pixel_size

        self.pushButton_visualise_simulation.setEnabled(False)
        self.pushButton_visualise_simulation.setText("Visualise Simulation")
        self.pushButton_visualise_simulation.setStyleSheet("background-color: gray")

    def read_stage_input_values(self):

        stage_configs = []

        for widgets in self.input_widgets:

            stage_config = {}

            lens_name = str(widgets[2].text())
            output = float(widgets[4].text())

            # only need one of these
            n_steps_text = widgets[6].text()
            step_size_text = widgets[8].text()
            
            n_steps = 0 if n_steps_text == "" else int(n_steps_text)
            step_size = 0 if step_size_text == "" else float(step_size_text)

            if n_steps == 0 and step_size == 0:
                raise ValueError(f"Both n_steps and step_size cannot be zero. Please set at least one.")

            use_focal_distance = bool(widgets[10].isChecked())
            if use_focal_distance:
                start_distance = 0.0
                finish_distance = 0.5e-3
                focal_distance_start_multiple = float(widgets[12].text())
                focal_distance_multiple = float(widgets[14].text())
            else:
                start_distance = float(widgets[12].text())
                finish_distance = float(widgets[14].text())
                focal_distance_start_multiple = 0.0
                focal_distance_multiple = 0.0

            stage_config = {
                "lens": lens_name,
                "output": output,
                "n_steps": n_steps,
                "step_size": step_size,
                "start_distance": start_distance,
                "finish_distance": finish_distance,
                "use_equivalent_focal_distance": use_focal_distance,
                "focal_distance_start_multiple": focal_distance_start_multiple,
                "focal_distance_multiple": focal_distance_multiple,
            }

            stage_configs.append(stage_config)

        self.simulation_config["stages"] = stage_configs

        # 0. label_title,
        # 1. lens_label,
        # 2. lens_button,
        # 3. output_label,
        # 4. output_lineEdit,
        # 5. n_steps_label,
        # 6. n_steps_lineEdit,
        # 7. step_size_label,
        # 8. step_size_lineEdit,
        # 9. use_focal_label,
        # 10. use_focal_checkbox
        # 11. start_distance_label,
        # 12. start_distance_lineEdit,
        # 13. finish_distance_label,
        # 14. finish_distance_lineEdit,

    def load_lens_config(self):
        """Load the element configuration from file."""
        lens_config_filename, _ = QFileDialog.getOpenFileName(
            self, "Select Lens Configuration", os.path.dirname(juno.__file__), "Yaml files (*.yml *.yaml)"
        )

        if lens_config_filename == "":
            return

        try:
            lens_config = utils.load_yaml_config(lens_config_filename)

            if "lenses" not in self.simulation_config:
                self.simulation_config["lenses"] = []

            validation._validate_default_lens_config(lens_config)
            self.simulation_config["lenses"].append(lens_config)
            self.sender().setText(f"{lens_config['name']}")
        except Exception as e:
            napari.utils.notifications.show_error(f"Invalid config. \n{e}")

    def load_beam_config(self):

        print("loading beam config")

        beam_config_filename, _ = QFileDialog.getOpenFileName(
            self, "Select Beam Configuration", os.path.dirname(juno.__file__), "Yaml files (*.yml *.yaml)"
        )

        if beam_config_filename == "":
            return

        try:
            beam_config = utils.load_yaml_config(beam_config_filename)

            validation._validate_default_beam_config(beam_config)

            self.simulation_config["beam"] = beam_config
            self.pushButton_sim_beam.setText(
                f"{Path(beam_config_filename).stem}"
            )
        except Exception as e:
            napari.utils.notifications.show_error(f"Invalid config. \n{e}")

    def update_stage_input_display(self):

        sim_num_stages = int(self.spinBox_sim_num_stages.value())

        input_layout = QVBoxLayout()

        stored_values = []
        if self.input_widgets:

            for wid in self.input_widgets:
                  
                sv = [w.text() for w in wid]

                stored_values.append(sv)
        
        # pprint(stored_values)

        self.input_widgets = []

        # TODO: change it to store the current stage info before updating
        # TODO: fix it to store always not only for valid stuff 

        for stage_no, _ in enumerate(range(sim_num_stages), 1):

            layout, widgets = create_stage_input_display(stage_no)

            widgets[2].clicked.connect(self.load_lens_config)
            widgets[10].stateChanged.connect(self.input_widget_state_change)

            self.input_widgets.append(widgets)

            stageBox = QGroupBox(f"")
            stageBox.setLayout(layout)
            input_layout.addWidget(stageBox)

        if stored_values:
            for i, wid in enumerate(self.input_widgets):
                if i == len(stored_values):
                    break
                sv = stored_values[i]
                for j, w in enumerate(wid):
                    w.setText(sv[j])

        inputBox = QGroupBox(f"")
        inputBox.setLayout(input_layout)
        self.scrollArea_stages.setWidget(inputBox)
        self.scrollArea_stages.update()


def create_stage_input_display(stage_no):

    layout = QGridLayout()

    label_title = QLabel()
    label_title.setText(f"Stage Number {stage_no}")
    label_title.setStyleSheet("font-weight: bold")

    # lens
    lens_label = QLabel()
    lens_label.setText(f"Lens")

    lens_button = QPushButton()
    lens_button.setText("...")  # use a different kind of widget?

    # output
    output_label = QLabel()
    output_label.setText(f"Output Medium")
    output_lineEdit = QLineEdit()

    # n_steps
    n_steps_label = QLabel()
    n_steps_label.setText(f"Number of steps")
    n_steps_lineEdit = QLineEdit()

    # step size
    step_size_label = QLabel()
    step_size_label.setText(f"Step Size")
    step_size_lineEdit = QLineEdit()

    # start distance
    start_distance_label = QLabel()
    start_distance_label.setText(f"Start Distance")
    start_distance_lineEdit = QLineEdit()


    # finish distance
    finish_distance_label = QLabel()
    finish_distance_label.setText(f"Finish Distance")
    finish_distance_lineEdit = QLineEdit()

    # options
    #   use focal distance
    use_focal_label = QLabel()
    use_focal_label.setText(f"Use Focal Distance")
    use_focal_checkbox = QCheckBox()

    # if use_focal_checkbox is checked, change start, finsish distance to these options...
    #   start focal distance
    #   end focal distance

    layout.addWidget(label_title, 0, 0, 1, 2)
    layout.addWidget(lens_label, 1, 0)
    layout.addWidget(lens_button, 1, 1)

    layout.addWidget(output_label, 2, 0)
    layout.addWidget(output_lineEdit, 2, 1)

    layout.addWidget(n_steps_label, 3, 0)
    layout.addWidget(n_steps_lineEdit, 3, 1)

    layout.addWidget(step_size_label, 4, 0)
    layout.addWidget(step_size_lineEdit, 4, 1)

    layout.addWidget(use_focal_label, 5, 0)
    layout.addWidget(use_focal_checkbox, 5, 1)

    layout.addWidget(start_distance_label, 6, 0)
    layout.addWidget(start_distance_lineEdit, 6, 1)

    layout.addWidget(finish_distance_label, 7, 0)
    layout.addWidget(finish_distance_lineEdit, 7, 1)


    widgets = [
        label_title,
        lens_label,
        lens_button,
        output_label,
        output_lineEdit,
        n_steps_label,
        n_steps_lineEdit,
        step_size_label,
        step_size_lineEdit,
        use_focal_label,
        use_focal_checkbox,
        start_distance_label,
        start_distance_lineEdit,
        finish_distance_label,
        finish_distance_lineEdit,
    ]

    return layout, widgets

def load_stage_config_widgets(config, all_widgets):
    for i, stage_config in enumerate(config["stages"]):

        widgets = all_widgets[i]
        widgets[2].setText(str(stage_config["lens"]))
        widgets[4].setText(str(stage_config["output"]))
        widgets[6].setText(str(stage_config["n_steps"]))
        widgets[8].setText(str(stage_config["step_size"]))
        widgets[12].setText(str(stage_config["start_distance"]))
        widgets[14].setText(str(stage_config["finish_distance"]))

        widgets[10].setChecked(bool(stage_config["use_equivalent_focal_distance"]))

        if widgets[10].isChecked():
            widgets[12].setText(str(stage_config["focal_distance_start_multiple"]))
            widgets[14].setText(str(stage_config["focal_distance_multiple"]))

def main():
    import napari
    viewer = napari.Viewer(ndisplay=3)
    simulation_setup_ui = GUISimulationSetup(viewer=viewer)                                          
    viewer.window.add_dock_widget(simulation_setup_ui, area='right')                  
    napari.run()

if __name__ == "__main__":
    main()
