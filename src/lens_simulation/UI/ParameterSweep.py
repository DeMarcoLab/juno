import sys
import traceback

import glob

import lens_simulation
import os
import yaml


from lens_simulation import utils
import matplotlib.pyplot as plt

import lens_simulation.UI.qtdesigner_files.ParameterSweep as ParameterSweep
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QGroupBox,
    QGridLayout,
    QLabel,
    QVBoxLayout,
    QPushButton,
    QLineEdit,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QCheckBox,
    QFileDialog,
)
import numpy as np

from pprint import pprint

from lens_simulation import validation, constants
from lens_simulation.SimulationRunner import generate_parameter_sweep




class GUIParameterSweep(ParameterSweep.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self, config: dict, parent_gui=None ):
        super().__init__(parent=parent_gui)
        self.setupUi(MainWindow=self)
        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)
        self.setWindowTitle("Parameter Sweep")
        
        self.config = config 

        self.setup_connections()

        self.showNormal()


    def setup_connections(self):

        print("setup_connections")

        self.pushButton_save_config.clicked.connect(self.save_sweepable_config)
      
        paramBox = QGroupBox()
        paramGridLayout = QGridLayout()

        current_idx = 0

        # titles
        # param, start, stop, step, n_combo
        label_start_title = QLabel()
        label_start_title.setText("Start")
        label_start_title.setStyleSheet("font-weight: bold;")
        label_start_title.setAlignment(QtCore.Qt.AlignCenter)

        label_stop_title = QLabel()
        label_stop_title.setText("Stop")
        label_stop_title.setStyleSheet("font-weight: bold;")
        label_stop_title.setAlignment(QtCore.Qt.AlignCenter)

        label_step_size_title = QLabel()
        label_step_size_title.setText("Step Size")
        label_step_size_title.setStyleSheet("font-weight: bold")
        label_step_size_title.setAlignment(QtCore.Qt.AlignCenter)
    
        label_n_combo_title = QLabel()
        label_n_combo_title.setText("Combination")
        label_n_combo_title.setStyleSheet("font-weight: bold;")
        label_n_combo_title.setAlignment(QtCore.Qt.AlignCenter)

        paramGridLayout.addWidget(label_start_title, current_idx, 1)
        paramGridLayout.addWidget(label_stop_title, current_idx, 2)
        paramGridLayout.addWidget(label_step_size_title, current_idx, 3)
        paramGridLayout.addWidget(label_n_combo_title, current_idx, 4)

        current_idx +=1

        # beam
        label_beam_title = QLabel()
        label_beam_title.setText("Beam")
        label_beam_title.setStyleSheet("font-weight: bold")

        paramGridLayout.addWidget(label_beam_title, current_idx, 0)
        current_idx +=1

        all_beam_widgets, current_idx = create_config_elements(self.config["beam"], 
                        constants.BEAM_SWEEPABLE_KEYS, paramGridLayout, 
                        current_idx)

        # lenses
        label_lens_title = QLabel()
        label_lens_title.setText("Lens")
        label_lens_title.setStyleSheet("font-weight: bold; text-align: center")

        paramGridLayout.addWidget(label_lens_title, current_idx, 0)
        current_idx +=1 

        all_lens_widgets = [] 

        for lens_config in self.config["lenses"]:
            
            lens_widgets = {}
            lens_name_label = QLabel()
            lens_name_label.setText(f"Lens: {lens_config['name']}")
            
            paramGridLayout.addWidget(lens_name_label, current_idx, 0)
            current_idx +=1
                
            for key, val in lens_config.items():

                if key in constants.LENS_SWEEPABLE_KEYS:
                    widgets = create_param_widgets(key, val, paramGridLayout, current_idx,
                                            stop_val=lens_config[f"{key}_stop"],
                                            step_val=lens_config[f"{key}_step"])
                    lens_widgets[key] = widgets
                    current_idx += 1

                if isinstance(val, dict):
                    
                    lens_widgets[key] = {}
                    for k, v in val.items():
                        if k in constants.MODIFICATION_SWEEPABLE_KEYS:
                            widgets = create_param_widgets(f"{key}_{k}", v, 
                                            paramGridLayout, current_idx, 
                                            stop_val=val[f"{k}_stop"],
                                            step_val=val[f"{k}_step"])
                            lens_widgets[key][k]= widgets
                            current_idx += 1

            all_lens_widgets.append(lens_widgets)

        # stages
        label_stage_title = QLabel()
        label_stage_title.setText("Stages")
        label_stage_title.setStyleSheet("font-weight: bold; text-align: center")

        paramGridLayout.addWidget(label_stage_title,current_idx, 0)
        current_idx+=1

        all_stage_widgets = []

        for i, stage_config in enumerate(self.config["stages"], 1):
            
            stage_widgets = {}

            stage_name_label = QLabel()
            stage_name_label.setText(f"Stage: {i}")
            
            paramGridLayout.addWidget(stage_name_label, current_idx, 0)
            current_idx +=1

            stage_widgets, current_idx = create_config_elements(stage_config, 
                                            constants.STAGE_SWEEPABLE_KEYS, 
                                            paramGridLayout, 
                                            current_idx)                

            all_stage_widgets.append(stage_widgets) 

        paramBox.setLayout(paramGridLayout)
        self.scrollArea.setWidget(paramBox)
        self.scrollArea.update()

        self.beam_widgets = all_beam_widgets
        self.lens_widgets = all_lens_widgets
        self.stage_widgets = all_stage_widgets

        # connect all widgets for combo updates
        self.connect_all_widgets()

# TODO: consolidate and rationalise all these different data structures.... beam, lens and stage are all different?

    def connect_all_widgets(self):

        for k, v in self.beam_widgets.items():

            self.connect_widgets(v)

        for lw in self.lens_widgets:
            for k, v in lw.items():
                if isinstance(v, list):
                    self.connect_widgets(v)
                
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        self.connect_widgets(v2)
        
        for sw in self.stage_widgets:
            for k, v in sw.items():
                if isinstance(v, list):
                    self.connect_widgets(v)



    def connect_widgets(self, widgets):

        if isinstance(widgets, list):
            widgets[1].textChanged.connect(self.on_update)
            widgets[2].textChanged.connect(self.on_update)
            widgets[3].textChanged.connect(self.on_update)

        
    def on_update(self):
        def get_param_values(widgets):
            start, stop, step = float(widgets[1].text()), float(widgets[2].text()), float(widgets[3].text())
            
            return start, stop, step
        
        def update_combination_widget(widgets):
            try:

                start, stop, step = get_param_values(widgets)
                sweep = generate_parameter_sweep(start, stop, step)
                widgets[4].setText(f"{len(sweep)}")
                self.statusBar.clearMessage()

            except Exception as e:
                widgets[4].setText(f"INVALID")
                self.statusBar.showMessage(f"Invalid parameter values: {e}")


        def check_matching_widget(widgets: list, sender, k):
            for w in widgets:
                if w == self.sender():
                    
                    print(f"IT WAS ME: {k}: {sender} ")
                    update_combination_widget(widgets)
                    return True
            
            return False

    
        for k, widgets in self.beam_widgets.items():
            if check_matching_widget(widgets, self.sender(), k):
                break


        for lw in self.lens_widgets:
            for k, v in lw.items():
                if isinstance(v, list):
                    if check_matching_widget(v, self.sender(), k):
                        break
                
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        if check_matching_widget(v2, self.sender(), f"{k}_{k2}"):
                            break
        
        for sw in self.stage_widgets:
            for k, v in sw.items():
                if isinstance(v, list):
                    if check_matching_widget(v, self.sender(), k):
                        break



    def read_sweep_values_into_config(self):
        """Read parameter sweep values into configuration."""
        self.sweep_config = {
            "beam": {},
            "lenses": [],
            "stages": []

        }

        # label, start, stop, step, combo
        for k, v in self.beam_widgets.items():

            self.sweep_config["beam"][k] = float(v[1].text())
            self.sweep_config["beam"][f"{k}_stop"] = None if v[2].text() == "" else float(v[2].text())
            self.sweep_config["beam"][f"{k}_step"] = None if v[3].text() == "" else float(v[3].text())

        # lenses
        lens_configs = []
        for val, conf in zip(self.lens_widgets, self.config["lenses"]):

            if isinstance(val, dict):
                for k, v in val.items():

                    if isinstance(v, dict): 
                        for k2, v2 in v.items():
                            conf[k][k2] = float(v2[1].text())
                            conf[k][f"{k2}_stop"] = None if v2[2].text() == "" else float(v2[2].text())
                            conf[k][f"{k2}_step"] = None if v2[3].text() == "" else float(v2[3].text())
                    else:            
                                                
                        conf[k] = float(v[1].text())
                        conf[f"{k}_stop"] = None if v[2].text() == "" else float(v[2].text())
                        conf[f"{k}_step"] = None if v[3].text() == "" else float(v[3].text())
                        
            lens_configs.append(conf)

        self.sweep_config["lenses"] = lens_configs

        # stages
        stage_configs = []
        for val, conf in zip(self.stage_widgets, self.config["stages"]):
            if isinstance(val, dict):
                for k, v in val.items():    

                    conf[k] = float(v[1].text())
                    conf[f"{k}_stop"] = None if v[2].text() == "" else float(v[2].text())
                    conf[f"{k}_step"] = None if v[3].text() == "" else float(v[3].text())

            stage_configs.append(conf)

        self.sweep_config["stages"] = stage_configs

        # update config
        self.config.update(self.sweep_config)


    def save_sweepable_config(self):

        self.read_sweep_values_into_config()

        # open file dialog
        sim_config_filename, _ = QFileDialog.getSaveFileName(self,
                    caption="Save Simulation Config", 
                    directory=os.path.dirname(lens_simulation.__file__),
                    filter="Yaml files (*.yml, *.yaml)")
        if sim_config_filename:          

            # same as yaml file
            with open(sim_config_filename, "w") as f:
                yaml.safe_dump(self.config, f)

            self.statusBar.showMessage(f"Simulation config saved to {sim_config_filename}")


# TODO: update combination counts?

# TODO: validate sweep values here?


def create_config_elements(config: dict, sweep_keys, layout, current_idx: int) -> list:

    all_widgets = {}

    for key, val in config.items():

        if key in sweep_keys:
            if val is None:
                continue # skip none values?

            widgets = create_param_widgets(key, val, layout, current_idx, 
                                stop_val=config[f"{key}_stop"], 
                                step_val=config[f"{key}_step"])
            
            all_widgets[key] = widgets

            current_idx += 1

    return all_widgets, current_idx


def create_param_widgets(key, val, layout, idx, stop_val = None, step_val = None ) -> list :
    label_param, lineedit_start, lineedit_stop, lineedit_step_size, label_n_combo = QLabel(), QLineEdit(), QLineEdit(), QLineEdit(), QLabel()
    label_n_combo.setAlignment(QtCore.Qt.AlignCenter)

    label_param.setText(str(key))
    lineedit_start.setText(str(val))
    label_n_combo.setText("1")

    # stop
    if stop_val is not None:
        lineedit_stop.setText(str(stop_val))
    # step
    if step_val is not None:
        lineedit_step_size.setText(str(step_val))

    layout.addWidget(label_param, idx, 0)
    layout.addWidget(lineedit_start, idx, 1)
    layout.addWidget(lineedit_stop, idx, 2)
    layout.addWidget(lineedit_step_size, idx, 3)
    layout.addWidget(label_n_combo, idx, 4)

    return [label_param, lineedit_start, lineedit_stop, lineedit_step_size, label_n_combo]
                        



def main():
    config = utils.load_config(os.path.join(os.path.dirname(lens_simulation.__file__), "sweep.yaml"))

    """Launch the main application window. """
    application = QtWidgets.QApplication([])
    window = GUIParameterSweep(config)
    application.aboutToQuit.connect(window.disconnect)  # cleanup & teardown
    sys.exit(application.exec_())


if __name__ == "__main__":
    main()
