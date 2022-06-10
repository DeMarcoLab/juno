# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\VisualiseResults.ui'
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
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 1603, 959))
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
        self.label_filter_title = QtWidgets.QLabel(self.frame_selection)
        self.label_filter_title.setObjectName("label_filter_title")
        self.gridLayout_2.addWidget(self.label_filter_title, 4, 0, 1, 1)
        self.label_sim_run_name = QtWidgets.QLabel(self.frame_selection)
        self.label_sim_run_name.setObjectName("label_sim_run_name")
        self.gridLayout_2.addWidget(self.label_sim_run_name, 1, 0, 1, 1)
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
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_2.addItem(spacerItem, 6, 0, 1, 1)
        self.tabWidget = QtWidgets.QTabWidget(self.frame_selection)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.frame = QtWidgets.QFrame(self.tab)
        self.frame.setGeometry(QtCore.QRect(0, 0, 271, 221))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.frame)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.doubleSpinBox = QtWidgets.QDoubleSpinBox(self.frame)
        self.doubleSpinBox.setObjectName("doubleSpinBox")
        self.gridLayout_4.addWidget(self.doubleSpinBox, 0, 1, 1, 1)
        self.label = QtWidgets.QLabel(self.frame)
        self.label.setObjectName("label")
        self.gridLayout_4.addWidget(self.label, 0, 0, 1, 1)
        self.doubleSpinBox_2 = QtWidgets.QDoubleSpinBox(self.frame)
        self.doubleSpinBox_2.setObjectName("doubleSpinBox_2")
        self.gridLayout_4.addWidget(self.doubleSpinBox_2, 1, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.frame)
        self.label_2.setObjectName("label_2")
        self.gridLayout_4.addWidget(self.label_2, 1, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.frame)
        self.label_3.setObjectName("label_3")
        self.gridLayout_4.addWidget(self.label_3, 2, 0, 1, 1)
        self.doubleSpinBox_3 = QtWidgets.QDoubleSpinBox(self.frame)
        self.doubleSpinBox_3.setObjectName("doubleSpinBox_3")
        self.gridLayout_4.addWidget(self.doubleSpinBox_3, 2, 1, 1, 1)
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.tabWidget.addTab(self.tab_2, "")
        self.gridLayout_2.addWidget(self.tabWidget, 5, 0, 1, 1)
        self.label_sim_no_loaded = QtWidgets.QLabel(self.frame_selection)
        self.label_sim_no_loaded.setText("")
        self.label_sim_no_loaded.setObjectName("label_sim_no_loaded")
        self.gridLayout_2.addWidget(self.label_sim_no_loaded, 3, 0, 1, 1)
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
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_filter_title.setText(_translate("MainWindow", "Filter Results"))
        self.label_sim_run_name.setText(_translate("MainWindow", "No simulation loaded."))
        self.label_title.setText(_translate("MainWindow", "Simulation Results"))
        self.pushButton_load_simulation.setText(_translate("MainWindow", "Load Simulation"))
        self.label.setText(_translate("MainWindow", "Stage"))
        self.label_2.setText(_translate("MainWindow", "TextLabel"))
        self.label_3.setText(_translate("MainWindow", "TextLabel"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Tab 1"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Tab 2"))
