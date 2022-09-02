# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'SimulationSetup.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(408, 905)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.frame)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setSpacing(0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.frame_2 = QtWidgets.QFrame(self.frame)
        self.frame_2.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.frame_2)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.scrollArea = QtWidgets.QScrollArea(self.frame_2)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents_2 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_2.setGeometry(QtCore.QRect(0, 0, 366, 704))
        self.scrollAreaWidgetContents_2.setObjectName("scrollAreaWidgetContents_2")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents_2)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.frame_4 = QtWidgets.QFrame(self.scrollAreaWidgetContents_2)
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.frame_4)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label_sim_wavelength = QtWidgets.QLabel(self.frame_4)
        self.label_sim_wavelength.setObjectName("label_sim_wavelength")
        self.gridLayout_4.addWidget(self.label_sim_wavelength, 7, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 60, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        self.gridLayout_4.addItem(spacerItem, 21, 0, 1, 2)
        self.label_sim_amplitude = QtWidgets.QLabel(self.frame_4)
        self.label_sim_amplitude.setObjectName("label_sim_amplitude")
        self.gridLayout_4.addWidget(self.label_sim_amplitude, 8, 0, 1, 1)
        self.label_stage_parameters_title = QtWidgets.QLabel(self.frame_4)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_stage_parameters_title.setFont(font)
        self.label_stage_parameters_title.setObjectName("label_stage_parameters_title")
        self.gridLayout_4.addWidget(self.label_stage_parameters_title, 14, 0, 1, 1)
        self.label_sim_width = QtWidgets.QLabel(self.frame_4)
        self.label_sim_width.setObjectName("label_sim_width")
        self.gridLayout_4.addWidget(self.label_sim_width, 5, 0, 1, 1)
        self.spinBox_sim_num_stages = QtWidgets.QSpinBox(self.frame_4)
        self.spinBox_sim_num_stages.setMinimum(1)
        self.spinBox_sim_num_stages.setMaximum(10)
        self.spinBox_sim_num_stages.setSingleStep(1)
        self.spinBox_sim_num_stages.setObjectName("spinBox_sim_num_stages")
        self.gridLayout_4.addWidget(self.spinBox_sim_num_stages, 15, 1, 1, 1)
        self.scrollArea_stages = QtWidgets.QScrollArea(self.frame_4)
        self.scrollArea_stages.setMinimumSize(QtCore.QSize(0, 0))
        self.scrollArea_stages.setWidgetResizable(True)
        self.scrollArea_stages.setObjectName("scrollArea_stages")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 326, 182))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.scrollArea_stages.setWidget(self.scrollAreaWidgetContents)
        self.gridLayout_4.addWidget(self.scrollArea_stages, 17, 0, 4, 2)
        self.label_sim_param_title = QtWidgets.QLabel(self.frame_4)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_sim_param_title.setFont(font)
        self.label_sim_param_title.setObjectName("label_sim_param_title")
        self.gridLayout_4.addWidget(self.label_sim_param_title, 0, 0, 1, 2)
        self.pushButton_sim_beam = QtWidgets.QPushButton(self.frame_4)
        self.pushButton_sim_beam.setObjectName("pushButton_sim_beam")
        self.gridLayout_4.addWidget(self.pushButton_sim_beam, 9, 1, 1, 1)
        self.label_sim_beam = QtWidgets.QLabel(self.frame_4)
        self.label_sim_beam.setObjectName("label_sim_beam")
        self.gridLayout_4.addWidget(self.label_sim_beam, 9, 0, 1, 1)
        self.label_options_title = QtWidgets.QLabel(self.frame_4)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_options_title.setFont(font)
        self.label_options_title.setObjectName("label_options_title")
        self.gridLayout_4.addWidget(self.label_options_title, 10, 0, 1, 1)
        self.lineEdit_pixel_size = QtWidgets.QLineEdit(self.frame_4)
        self.lineEdit_pixel_size.setObjectName("lineEdit_pixel_size")
        self.gridLayout_4.addWidget(self.lineEdit_pixel_size, 3, 1, 1, 1)
        self.checkBox_save_plot = QtWidgets.QCheckBox(self.frame_4)
        self.checkBox_save_plot.setChecked(True)
        self.checkBox_save_plot.setObjectName("checkBox_save_plot")
        self.gridLayout_4.addWidget(self.checkBox_save_plot, 13, 1, 1, 1)
        self.lineEdit_sim_amplitude = QtWidgets.QLineEdit(self.frame_4)
        self.lineEdit_sim_amplitude.setObjectName("lineEdit_sim_amplitude")
        self.gridLayout_4.addWidget(self.lineEdit_sim_amplitude, 8, 1, 1, 1)
        self.lineEdit_sim_name = QtWidgets.QLineEdit(self.frame_4)
        self.lineEdit_sim_name.setObjectName("lineEdit_sim_name")
        self.gridLayout_4.addWidget(self.lineEdit_sim_name, 12, 1, 1, 1)
        self.label_sim_height = QtWidgets.QLabel(self.frame_4)
        self.label_sim_height.setObjectName("label_sim_height")
        self.gridLayout_4.addWidget(self.label_sim_height, 6, 0, 1, 1)
        self.label_pixel_size = QtWidgets.QLabel(self.frame_4)
        self.label_pixel_size.setObjectName("label_pixel_size")
        self.gridLayout_4.addWidget(self.label_pixel_size, 3, 0, 1, 1)
        self.lineEdit_sim_height = QtWidgets.QLineEdit(self.frame_4)
        self.lineEdit_sim_height.setObjectName("lineEdit_sim_height")
        self.gridLayout_4.addWidget(self.lineEdit_sim_height, 6, 1, 1, 1)
        self.label_sim_num_stages = QtWidgets.QLabel(self.frame_4)
        self.label_sim_num_stages.setObjectName("label_sim_num_stages")
        self.gridLayout_4.addWidget(self.label_sim_num_stages, 15, 0, 1, 1)
        self.lineEdit_sim_wavelength = QtWidgets.QLineEdit(self.frame_4)
        self.lineEdit_sim_wavelength.setObjectName("lineEdit_sim_wavelength")
        self.gridLayout_4.addWidget(self.lineEdit_sim_wavelength, 7, 1, 1, 1)
        self.lineEdit_log_dir = QtWidgets.QLineEdit(self.frame_4)
        self.lineEdit_log_dir.setObjectName("lineEdit_log_dir")
        self.gridLayout_4.addWidget(self.lineEdit_log_dir, 11, 1, 1, 1)
        self.lineEdit_sim_width = QtWidgets.QLineEdit(self.frame_4)
        self.lineEdit_sim_width.setObjectName("lineEdit_sim_width")
        self.gridLayout_4.addWidget(self.lineEdit_sim_width, 5, 1, 1, 1)
        self.label_sim_name = QtWidgets.QLabel(self.frame_4)
        self.label_sim_name.setObjectName("label_sim_name")
        self.gridLayout_4.addWidget(self.label_sim_name, 12, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.frame_4)
        self.label_2.setObjectName("label_2")
        self.gridLayout_4.addWidget(self.label_2, 11, 0, 1, 1)
        self.gridLayout_5.addWidget(self.frame_4, 0, 0, 1, 1)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents_2)
        self.gridLayout_3.addWidget(self.scrollArea, 1, 0, 1, 1)
        self.pushButton_save_sim_config = QtWidgets.QPushButton(self.frame_2)
        self.pushButton_save_sim_config.setEnabled(False)
        self.pushButton_save_sim_config.setObjectName("pushButton_save_sim_config")
        self.gridLayout_3.addWidget(self.pushButton_save_sim_config, 5, 0, 1, 1)
        self.pushButton_setup_parameter_sweep = QtWidgets.QPushButton(self.frame_2)
        self.pushButton_setup_parameter_sweep.setEnabled(False)
        self.pushButton_setup_parameter_sweep.setObjectName("pushButton_setup_parameter_sweep")
        self.gridLayout_3.addWidget(self.pushButton_setup_parameter_sweep, 3, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.frame_2)
        self.label.setMaximumSize(QtCore.QSize(16777215, 200))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.gridLayout_3.addWidget(self.label, 0, 0, 1, 1)
        self.pushButton_generate_simulation = QtWidgets.QPushButton(self.frame_2)
        self.pushButton_generate_simulation.setObjectName("pushButton_generate_simulation")
        self.gridLayout_3.addWidget(self.pushButton_generate_simulation, 2, 0, 1, 1)
        self.gridLayout_2.addWidget(self.frame_2, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.frame, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 408, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionLoad_Config = QtWidgets.QAction(MainWindow)
        self.actionLoad_Config.setObjectName("actionLoad_Config")
        self.menuFile.addAction(self.actionLoad_Config)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        MainWindow.setTabOrder(self.lineEdit_pixel_size, self.lineEdit_sim_width)
        MainWindow.setTabOrder(self.lineEdit_sim_width, self.lineEdit_sim_height)
        MainWindow.setTabOrder(self.lineEdit_sim_height, self.lineEdit_sim_wavelength)
        MainWindow.setTabOrder(self.lineEdit_sim_wavelength, self.lineEdit_sim_amplitude)
        MainWindow.setTabOrder(self.lineEdit_sim_amplitude, self.pushButton_sim_beam)
        MainWindow.setTabOrder(self.pushButton_sim_beam, self.lineEdit_log_dir)
        MainWindow.setTabOrder(self.lineEdit_log_dir, self.checkBox_save_plot)
        MainWindow.setTabOrder(self.checkBox_save_plot, self.spinBox_sim_num_stages)
        MainWindow.setTabOrder(self.spinBox_sim_num_stages, self.scrollArea_stages)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_sim_wavelength.setText(_translate("MainWindow", "Simulation Wavelength (m)"))
        self.label_sim_amplitude.setText(_translate("MainWindow", "Simulation Amplitude"))
        self.label_stage_parameters_title.setText(_translate("MainWindow", "Stage Parameters"))
        self.label_sim_width.setText(_translate("MainWindow", "Simulation Width (m)"))
        self.label_sim_param_title.setText(_translate("MainWindow", "Simulation Parameters"))
        self.pushButton_sim_beam.setText(_translate("MainWindow", "..."))
        self.label_sim_beam.setText(_translate("MainWindow", "Simulation Beam"))
        self.label_options_title.setText(_translate("MainWindow", "Simulation Options"))
        self.lineEdit_pixel_size.setText(_translate("MainWindow", "1e-6"))
        self.checkBox_save_plot.setText(_translate("MainWindow", "Save Plots"))
        self.lineEdit_sim_amplitude.setText(_translate("MainWindow", "1"))
        self.label_sim_height.setText(_translate("MainWindow", "Simulation Height (m)"))
        self.label_pixel_size.setText(_translate("MainWindow", "Pixel Size (m)"))
        self.lineEdit_sim_height.setText(_translate("MainWindow", "1000e-6"))
        self.label_sim_num_stages.setText(_translate("MainWindow", "Number of Stages"))
        self.lineEdit_sim_wavelength.setText(_translate("MainWindow", "488.e-9"))
        self.lineEdit_sim_width.setText(_translate("MainWindow", "1000e-6"))
        self.label_sim_name.setText(_translate("MainWindow", "Simulation Name"))
        self.label_2.setText(_translate("MainWindow", "Logging Directory"))
        self.pushButton_save_sim_config.setText(_translate("MainWindow", "Save Simulation"))
        self.pushButton_setup_parameter_sweep.setText(_translate("MainWindow", "Setup Parameter Sweep"))
        self.label.setText(_translate("MainWindow", "Simulation Setup"))
        self.pushButton_generate_simulation.setText(_translate("MainWindow", "Generate Simulation"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionLoad_Config.setText(_translate("MainWindow", "Load Config"))
