import sys
from PySide6 import QtCore, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1096, 850)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(-70, 20, 1331, 811))
        self.widget.setObjectName("widget")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.widget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(70, 40, 1101, 771))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")

        self.spinBox = QtWidgets.QSpinBox(self.widget)
        self.spinBox.setGeometry(QtCore.QRect(90, 0, 151, 31))
        self.spinBox.setObjectName("spinBox")
        self.spinBox.setMaximum(4)

        self.start_button = QtWidgets.QPushButton(self.widget)
        self.start_button.setGeometry(QtCore.QRect(500, 0, 93, 28))
        self.start_button.setStyleSheet("background-color: rgb(155, 170, 200);")
        self.start_button.setObjectName("start_button")
        self.start_button.setText("Start")

        self.stop_button = QtWidgets.QPushButton(self.widget)
        self.stop_button.setGeometry(QtCore.QRect(650, 0, 93, 28))
        self.stop_button.setStyleSheet("background-color: rgb(155, 170, 200);")
        self.stop_button.setObjectName("stop_button")
        self.stop_button.setText("Stop")

        self.clear_button = QtWidgets.QPushButton(self.widget)
        self.clear_button.setGeometry(QtCore.QRect(790, 0, 93, 28))
        self.clear_button.setStyleSheet("background-color: rgb(155, 170, 200);")
        self.clear_button.setObjectName("clear_button")
        self.clear_button.setText("Clear")

        self.close_button = QtWidgets.QPushButton(self.widget)
        self.close_button.setGeometry(QtCore.QRect(920, 0, 93, 28))
        self.close_button.setStyleSheet("background-color: rgb(155, 170, 200);")
        self.close_button.setObjectName("close_button")
        self.close_button.setText("Close")

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 0, 55, 16))
        self.label.setObjectName("label")
        self.label.setText("Windows")

        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(200, 0, 55, 16))
        self.label_2.setObjectName("label_2")
        self.label_2.setText("Time")

        MainWindow.setCentralWidget(self.centralwidget)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec())
