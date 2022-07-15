# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\SimulationRun.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(500, 389)
        MainWindow.setMaximumSize(QtCore.QSize(500, 16777215))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.frame)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_config_info = QtWidgets.QLabel(self.frame)
        self.label_config_info.setWordWrap(True)
        self.label_config_info.setObjectName("label_config_info")
        self.gridLayout_2.addWidget(self.label_config_info, 2, 0, 1, 1)
        self.label_running_info = QtWidgets.QLabel(self.frame)
        self.label_running_info.setWordWrap(True)
        self.label_running_info.setObjectName("label_running_info")
        self.gridLayout_2.addWidget(self.label_running_info, 4, 0, 1, 1)
        self.pushButton_load_config = QtWidgets.QPushButton(self.frame)
        self.pushButton_load_config.setObjectName("pushButton_load_config")
        self.gridLayout_2.addWidget(self.pushButton_load_config, 1, 0, 1, 1)
        self.pushButton_run_simulation = QtWidgets.QPushButton(self.frame)
        self.pushButton_run_simulation.setObjectName("pushButton_run_simulation")
        self.gridLayout_2.addWidget(self.pushButton_run_simulation, 3, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_2.addItem(spacerItem, 7, 0, 1, 1)
        self.progressBar_running = QtWidgets.QProgressBar(self.frame)
        self.progressBar_running.setProperty("value", 24)
        self.progressBar_running.setObjectName("progressBar_running")
        self.gridLayout_2.addWidget(self.progressBar_running, 5, 0, 1, 1)
        self.label_title = QtWidgets.QLabel(self.frame)
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_title.setFont(font)
        self.label_title.setObjectName("label_title")
        self.gridLayout_2.addWidget(self.label_title, 0, 0, 1, 1)
        self.label_final_info = QtWidgets.QLabel(self.frame)
        self.label_final_info.setText("")
        self.label_final_info.setObjectName("label_final_info")
        self.gridLayout_2.addWidget(self.label_final_info, 6, 0, 1, 1)
        self.gridLayout.addWidget(self.frame, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 500, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_config_info.setText(_translate("MainWindow", "Simulation Info"))
        self.label_running_info.setText(_translate("MainWindow", "Running Info"))
        self.pushButton_load_config.setText(_translate("MainWindow", "Load Simulation Config"))
        self.pushButton_run_simulation.setText(_translate("MainWindow", "Run Simulation"))
        self.label_title.setText(_translate("MainWindow", "Simulation Runner"))