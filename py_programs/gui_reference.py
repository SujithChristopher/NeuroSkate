from PySide6 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1242, 609)
        
        self.left_offset = 1400
        self.top_offset = 500
        self.font_size = 30
        
        
        self.button_width = 165
        self.button_height = 50
        
        button_ss = """QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    border-radius: 5px;
                    border: none;
                    font-size: 30px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
                QPushButton:pressed {
                    background-color: #398038;
                }"""
        
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.video_disp = QtWidgets.QLabel(self.centralwidget)
        self.video_disp.setGeometry(QtCore.QRect(20, 150, 1280, 900))
        self.video_disp.setText("")
        self.video_disp.setObjectName("video_disp")


        self.disp_pose = QtWidgets.QCheckBox(self.centralwidget)
        self.disp_pose.setGeometry(QtCore.QRect(self.left_offset, 260 + self.top_offset, 121, 31))
        self.disp_pose.setObjectName("disp_pose")
        
        """Buttons"""

        
        self.start_button = QtWidgets.QPushButton(self.centralwidget)
        self.start_button.setGeometry(QtCore.QRect(self.left_offset, 320 + self.top_offset, self.button_width, self.button_height))
        self.start_button.setObjectName("start_button")
        
        self.start_program = QtWidgets.QPushButton(self.centralwidget)
        self.start_program.setGeometry(QtCore.QRect(self.left_offset, 380 + self.top_offset, self.button_width, self.button_height))
        self.start_program.setObjectName("start_program")
        
        self.set_orgin = QtWidgets.QPushButton(self.centralwidget)
        self.set_orgin.setGeometry(QtCore.QRect(self.left_offset, 440 + self.top_offset, self.button_width, self.button_height))
        self.set_orgin.setObjectName("set_orgin")
        
        self.label_x = QtWidgets.QLabel(self.centralwidget)
        self.label_x.setGeometry(QtCore.QRect(self.left_offset, 100 + self.top_offset, 221, 41))
        font = QtGui.QFont()
        font.setFamily("Segoe Script")
        font.setPointSize(self.font_size)
        self.label_x.setFont(font)
        self.label_x.setObjectName("label_x")

        self.label_y = QtWidgets.QLabel(self.centralwidget)
        self.label_y.setGeometry(QtCore.QRect(self.left_offset, 140 + self.top_offset, 221, 41))
        font = QtGui.QFont()
        font.setFamily("Segoe Script")
        font.setPointSize(self.font_size)
        self.label_y.setFont(font)
        self.label_y.setObjectName("label_y")
        self.label_z = QtWidgets.QLabel(self.centralwidget)
        self.label_z.setGeometry(QtCore.QRect(self.left_offset, 180 + self.top_offset, 221, 41))
        font = QtGui.QFont()
        font.setFamily("Segoe Script")
        font.setPointSize(self.font_size)
        self.label_z.setFont(font)
        self.label_z.setObjectName("label_z")

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1242, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        
        
        """Buttons"""
        self.start_button.setStyleSheet(button_ss)
        self.start_program.setStyleSheet(button_ss)
        self.set_orgin.setStyleSheet(button_ss)
        
        
        """Vtk widget"""

        
        self.vtk_image = QtWidgets.QLabel(self.centralwidget)
        self.vtk_image.setGeometry(QtCore.QRect(self.left_offset - 50 , 240, 300, 300))
        self.vtk_image.setText("")
        self.vtk_image.setObjectName("vtk_image")

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.start_button.setText(_translate("MainWindow", "Initialize"))
        self.disp_pose.setText(_translate("MainWindow", "Display POSE"))
        self.label_x.setText(_translate("MainWindow", "0"))
        self.start_program.setText(_translate("MainWindow", "Start"))
        self.label_y.setText(_translate("MainWindow", "0"))
        self.label_z.setText(_translate("MainWindow", "0"))
        self.set_orgin.setText(_translate("MainWindow", "Reset Orgin"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec())
