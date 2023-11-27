import napari
import numpy as np
import yaml
from napari.qt.threading import thread_worker
from PyQt5 import QtCore, QtGui, QtWidgets

import juno_custom.tools.element_tools as element_tools
from juno_custom.ui import main as main_ui


class GuiMainWindow(main_ui.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self, viewer: napari.Viewer = None):
        super(GuiMainWindow, self).__init__()
        self.setupUi(self)
        self.viewer = viewer
        self.params = dict()

        self.setup_viewer()
        self.generate_element_list()
        self.generate_options()
        self.setup_connections()

    def setup_viewer(self):
        self.viewer.window._qt_viewer.dockLayerList.setVisible(False)
        self.viewer.window._qt_viewer.dockLayerControls.setVisible(False)
        self.viewer.window._qt_viewer.dockConsole.setVisible(False)

    def setup_connections(self):
        self.comboBox_Elements.currentTextChanged.connect(self.generate_options)
        self.pushButton_GenerateProfile.clicked.connect(self.run_generate_profile)
        self.pushButton_SaveProfile.clicked.connect(self.save_profile)
        self.pushButton_SaveParameters.clicked.connect(self.save_parameters)

    def generate_element_list(self):
        # import custom elements from elements folder
        element_list = element_tools.get_custom_elements()

        self.comboBox_Elements.clear()

        self.elements = dict()
        for element in element_list:
            self.comboBox_Elements.addItem(element.__name__)
            self.elements[element.__name__] = {
                "class": element,
                "keys": element.__keys__(),
            }

    def generate_options(self):
        element_name = self.comboBox_Elements.currentText()
        print(f"Generating options for {element_name}...")

        # clear options frame
        for widget in self.frame_Options.children():
            if isinstance(widget, QtWidgets.QWidget):
                self.frame_Options.layout().removeWidget(widget)
                widget.deleteLater()

        # re/set layout
        if self.frame_Options.layout() is None:
            self.frame_Options.setLayout(QtWidgets.QVBoxLayout())

        self.params = dict()
        self.lineedits = dict()

        # load in each key
        for key, value in self.elements[element_name]["keys"].items():
            self.params[key] = None

            key_label = QtWidgets.QLabel(key)
            key_label.setToolTip(value[4])

            key_line_edit = QtWidgets.QLineEdit()
            key_line_edit.setText(str(value[1]))
            if value[3]:
                key_line_edit.setEnabled(False)
            self.lineedits[key] = key_line_edit

            self.frame_Options.layout().addWidget(key_label)
            self.frame_Options.layout().addWidget(key_line_edit)

    def run_generate_profile(self):
        self.pushButton_GenerateProfile.setEnabled(False)
        element_name = self.comboBox_Elements.currentText()
        params = self.get_parameters(element_name=element_name)
        self.element = self.load_element(element_name)

        worker = self.generate_profile(params=params)
        worker.signals.finished.connect(self.plot_profile)
        worker.signals.finished.connect(
            lambda: self.pushButton_GenerateProfile.setEnabled(True)
        )
        worker.start()

    def get_parameters(self, element_name: str):
        params = dict()
        for key in self.params.keys():
            type_ = self.elements[element_name]["keys"][key][0]
            params[key] = type_(self.lineedits[key].text())
        return params

    @thread_worker
    def generate_profile(self, params):
        self.element.generate_profile(params)

    def plot_profile(self):
        self.viewer.layers.clear()
        self.viewer.add_image(self.element.profile, colormap="gray", name="element")

    def load_element(self, element_name: str):
        element = self.elements[element_name]["class"]()
        return element

    def save_profile(self):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Profile", "", "Numpy Arrays (*.npy)"
        )
        if filename:
            np.save(filename, self.element.profile)

    def save_parameters(self):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Parameters", "", "YAML (*.yml *.yaml)"
        )
        if filename:
            parameters = self.get_parameters(
                element_name=self.comboBox_Elements.currentText()
            )
            with open(filename, "w") as f:
                yaml.dump(parameters, f, sort_keys=False)


if __name__ == "__main__":
    viewer = napari.Viewer(ndisplay=2)
    user_interface = GuiMainWindow(viewer=viewer)
    viewer.window.add_dock_widget(user_interface, area="right")
    napari.run()
