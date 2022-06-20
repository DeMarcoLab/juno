# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'VisualiseResults.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1952, 1040)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.frame_main = QtWidgets.QFrame(self.centralwidget)
        self.frame_main.setGeometry(QtCore.QRect(9, 9, 1951, 1001))
        self.frame_main.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_main.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_main.setObjectName("frame_main")
        self.gridLayout = QtWidgets.QGridLayout(self.frame_main)
        self.gridLayout.setObjectName("gridLayout")
        self.frame_display = QtWidgets.QFrame(self.frame_main)
        self.frame_display.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_display.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_display.setObjectName("frame_display")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.frame_display)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.scroll_area = QtWidgets.QScrollArea(self.frame_display)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setObjectName("scroll_area")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 1630, 959))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.scroll_area.setWidget(self.scrollAreaWidgetContents)
        self.gridLayout_3.addWidget(self.scroll_area, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.frame_display, 0, 1, 1, 1)
        self.frame_selection = QtWidgets.QFrame(self.frame_main)
        self.frame_selection.setMaximumSize(QtCore.QSize(300, 16777215))
        self.frame_selection.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_selection.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_selection.setObjectName("frame_selection")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.frame_selection)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_sim_no_loaded = QtWidgets.QLabel(self.frame_selection)
        self.label_sim_no_loaded.setText("")
        self.label_sim_no_loaded.setObjectName("label_sim_no_loaded")
        self.gridLayout_2.addWidget(self.label_sim_no_loaded, 3, 0, 1, 1)
        self.label_title = QtWidgets.QLabel(self.frame_selection)
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.label_title.setFont(font)
        self.label_title.setObjectName("label_title")
        self.gridLayout_2.addWidget(self.label_title, 0, 0, 1, 1)
        self.pushButton_load_simulation = QtWidgets.QPushButton(self.frame_selection)
        self.pushButton_load_simulation.setObjectName("pushButton_load_simulation")
        self.gridLayout_2.addWidget(self.pushButton_load_simulation, 2, 0, 1, 1)
        self.label_filter_title = QtWidgets.QLabel(self.frame_selection)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_filter_title.setFont(font)
        self.label_filter_title.setObjectName("label_filter_title")
        self.gridLayout_2.addWidget(self.label_filter_title, 4, 0, 1, 1)
        self.frame_filter = QtWidgets.QFrame(self.frame_selection)
        self.frame_filter.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_filter.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_filter.setObjectName("frame_filter")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.frame_filter)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.scroll_area_filter = QtWidgets.QScrollArea(self.frame_filter)
        self.scroll_area_filter.setMaximumSize(QtCore.QSize(400, 400))
        self.scroll_area_filter.setWidgetResizable(True)
        self.scroll_area_filter.setObjectName("scroll_area_filter")
        self.scrollAreaWidgetContents_2 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_2.setGeometry(QtCore.QRect(0, 0, 231, 287))
        self.scrollAreaWidgetContents_2.setObjectName("scrollAreaWidgetContents_2")
        self.scroll_area_filter.setWidget(self.scrollAreaWidgetContents_2)
        self.gridLayout_4.addWidget(self.scroll_area_filter, 2, 0, 1, 3)
        self.pushButton_filter_data = QtWidgets.QPushButton(self.frame_filter)
        self.pushButton_filter_data.setObjectName("pushButton_filter_data")
        self.gridLayout_4.addWidget(self.pushButton_filter_data, 3, 0, 1, 3)
        self.pushButton_reset_data = QtWidgets.QPushButton(self.frame_filter)
        self.pushButton_reset_data.setObjectName("pushButton_reset_data")
        self.gridLayout_4.addWidget(self.pushButton_reset_data, 4, 0, 1, 3)
        self.label = QtWidgets.QLabel(self.frame_filter)
        self.label.setObjectName("label")
        self.gridLayout_4.addWidget(self.label, 0, 0, 1, 1)
        self.spinBox_num_filters = QtWidgets.QSpinBox(self.frame_filter)
        self.spinBox_num_filters.setMinimum(1)
        self.spinBox_num_filters.setMaximum(10)
        self.spinBox_num_filters.setObjectName("spinBox_num_filters")
        self.gridLayout_4.addWidget(self.spinBox_num_filters, 0, 1, 1, 1)
        self.label_num_filtered_simulations = QtWidgets.QLabel(self.frame_filter)
        self.label_num_filtered_simulations.setObjectName("label_num_filtered_simulations")
        self.gridLayout_4.addWidget(self.label_num_filtered_simulations, 5, 0, 1, 3)
        self.checkBox_show_beam = QtWidgets.QCheckBox(self.frame_filter)
        self.checkBox_show_beam.setObjectName("checkBox_show_beam")
        self.gridLayout_4.addWidget(self.checkBox_show_beam, 0, 2, 1, 1)
        self.gridLayout_2.addWidget(self.frame_filter, 5, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_2.addItem(spacerItem, 7, 0, 1, 1)
        self.label_sim_run_name = QtWidgets.QLabel(self.frame_selection)
        self.label_sim_run_name.setObjectName("label_sim_run_name")
        self.gridLayout_2.addWidget(self.label_sim_run_name, 1, 0, 1, 1)
        self.gridLayout.addWidget(self.frame_selection, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1952, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        MainWindow.setTabOrder(self.pushButton_load_simulation, self.spinBox_num_filters)
        MainWindow.setTabOrder(self.spinBox_num_filters, self.checkBox_show_beam)
        MainWindow.setTabOrder(self.checkBox_show_beam, self.scroll_area_filter)
        MainWindow.setTabOrder(self.scroll_area_filter, self.pushButton_filter_data)
        MainWindow.setTabOrder(self.pushButton_filter_data, self.pushButton_reset_data)
        MainWindow.setTabOrder(self.pushButton_reset_data, self.scroll_area)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_title.setText(_translate("MainWindow", "Simulation Results"))
        self.pushButton_load_simulation.setText(_translate("MainWindow", "Load Simulation"))
        self.label_filter_title.setText(_translate("MainWindow", "Filter Results"))
        self.pushButton_filter_data.setText(_translate("MainWindow", "Filter Data"))
        self.pushButton_reset_data.setText(_translate("MainWindow", "Reset Data"))
        self.label.setText(_translate("MainWindow", "Number of filters"))
        self.label_num_filtered_simulations.setText(_translate("MainWindow", "TextLabel"))
        self.checkBox_show_beam.setText(_translate("MainWindow", "Show Beam"))
        self.label_sim_run_name.setText(_translate("MainWindow", "No simulation loaded."))
