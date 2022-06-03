from cmath import pi
import sys

import lens_simulation.UI.qtdesigner_files.LensCreator as LensCreator
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtGui, QtWidgets
from lens_simulation.Lens import Lens, LensType


class GUILensCreator(LensCreator.Ui_LensCreator, QtWidgets.QMainWindow):

    def __init__(self, parent_gui=None):
        super().__init__(parent=parent_gui)
        self.setupUi(LensCreator=self)

        self.setup_connections()

        self.wp_CrossSection = None
        self.pc_Profile = None

        self.showNormal()
        self.center_window()

    def setup_connections(self):
        self.pushButton_LoadProfile.clicked.connect(self.load_profile)
        self.pushButton_GenerateProfile.clicked.connect(self.generate_profile)

    def generate_profile(self):
        """Generates a profile based on the inputs to the GUI"""
        self.lens = Lens(diameter=1,
                              height=1,
                              exponent=1,
                              medium=1
                              )

        pixel_size = 5e-2

        self.lens.generate_profile(pixel_size=pixel_size, lens_type=LensType.Cylindrical)

        if self.lens.lens_type is LensType.Cylindrical:
            self.lens.extrude_profile(pixel_size)

        self.array_Profile = self.lens.profile
        self.array_CrossSection = self.lens.profile[self.lens.profile.shape[0]//2, :]
        self.update_image_frames()



    def load_profile(self):
        """Loads a custom lens profile (numpy.ndarray) through Qt's file opening system"""
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Load Profile", filter="Numpy array (*.npy)"
            )

        if filename is '':
            return

        array = np.load(filename)
        self.array_Profile = array
        self.array_CrossSection = array[:, array.shape[-1]//2]
        self.update_image_frames()

    def update_image_frames(self):
        # Cross section initialisation
        if self.wp_CrossSection is not None:
            self.label_CrossSection.layout().removeWidget(self.wp_CrossSection)
            self.wp_CrossSection.deleteLater()
        self.wp_CrossSection = _ImageCanvas(parent=self.label_CrossSection, image=self.array_CrossSection)
        self.label_CrossSection.setLayout(QtWidgets.QVBoxLayout())
        self.label_CrossSection.layout().addWidget(self.wp_CrossSection)

        # Cross section initialisation
        if self.pc_Profile is not None:
            self.label_Profile.layout().removeWidget(self.pc_Profile)
            self.pc_Profile.deleteLater()
        self.pc_Profile = _ImageCanvas(parent=self.label_Profile, image=self.array_Profile)
        if self.label_Profile.layout() is None:
            self.label_Profile.setLayout(QtWidgets.QVBoxLayout())
        self.label_Profile.layout().addWidget(self.pc_Profile)


    def center_window(self):
        """Centers the window in the display"""
        # Get the desktop dimensions
        desktop = QtWidgets.QDesktopWidget()
        self.move((desktop.width()-self.width())/2, (desktop.height()-self.height())/3.)


class _ImageCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, image=None):
        # self.fig = Figure()
        self.fig = Figure(layout='constrained')
        FigureCanvasQTAgg.__init__(self, self.fig)
        self.setParent(parent)

        gridspec = self.fig.add_gridspec(1, 1)
        self.axes = self.fig.add_subplot(gridspec[0], xticks=[], yticks=[], title="")

        # Push the image to edges of border as much as we can
        # self.axes.axis('off')
        self.axes.spines['top'].set_visible(False)
        self.axes.spines['right'].set_visible(False)
        self.axes.spines['bottom'].set_visible(False)
        self.axes.spines['left'].set_visible(False)
        self.fig.set_facecolor('#f0f0f0')

        # Display image
        if image.ndim == 2:
            self.axes.imshow(image, aspect='auto')
        else:
            self.axes.plot(image)


def main():
    """Launch the `piescope_gui` main application window."""
    application = QtWidgets.QApplication([])
    window = GUILensCreator()
    application.aboutToQuit.connect(window.disconnect)  # cleanup & teardown
    window.showNormal()
    window.center_window()

    sys.exit(application.exec_())


if __name__ == "__main__":
    main()
