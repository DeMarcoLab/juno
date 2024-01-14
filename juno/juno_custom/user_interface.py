import logging

import napari
import numpy as np
import yaml
from napari.qt.threading import thread_worker
from PyQt5 import QtWidgets

import juno.juno_custom.tools.element_tools as element_tools
from juno.juno_custom.ui.qtdesignerfiles import \
    CustomElementGenerator as main_ui


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
        self.viewer.window._qt_viewer.dockLayerList.setVisible(True)
        self.viewer.window._qt_viewer.dockLayerControls.setVisible(True)
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
        if self.comboBox_Elements.currentText() == "":
            return
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
        # get viewer camera and mode
        center = self.viewer.camera.center
        zoom = self.viewer.camera.zoom
        angles = self.viewer.camera.angles
        ndim = self.viewer.dims.ndim
        ndisplay = self.viewer.dims.ndisplay
        range = self.viewer.dims.range
        order = self.viewer.dims.order
        # order=(1, 2, 0),

        # TODO: add persistent mode?
        self.viewer.layers.clear()
        # TODO: Multi-return

        # turn into a dict of each profile
        if not isinstance(self.element.profile, dict):
            self.element.profile = {"element_profile": self.element.profile}

        if self.element.display_profile is None:
            raw_visible = True
        else:
            raw_visible = False
            self.viewer.add_image(
                self.element.display_profile,
                colormap="gray",
                name="display_profile",
                visible=True,
            )
            self.viewer.layers["display_profile"].scale = [1, 1000, 1]
            ndim = 3
            # TODO: not resetting?
            order = [1, 0, 2]
            angles = [0, 0, -90]

        for profile in self.element.profile.keys():
            # if profile is complex
            if np.iscomplexobj(self.element.profile[profile]):
                logging.warning(
                    f"Profile {profile} is complex. Displaying absolute value."
                )
                plot_ = np.abs(self.element.profile[profile])
                name_ = f"{profile}_(abs)"
            else:
                plot_ = self.element.profile[profile]
                name_ = profile

            self.viewer.add_image(
                plot_, colormap="gray", name=name_, visible=raw_visible
            )

        self.viewer.dims.ndim = ndim
        self.viewer.dims.ndisplay = ndisplay
        self.viewer.dims.range = range
        self.viewer.dims.order = order
        self.viewer.camera.center = center
        self.viewer.camera.zoom = zoom
        self.viewer.camera.angles = angles

        # self.viewer.dims.set = mode

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
    viewer = napari.Viewer(ndisplay=3)
    user_interface = GuiMainWindow(viewer=viewer)
    viewer.window.add_dock_widget(user_interface, area="right")
    napari.run()
