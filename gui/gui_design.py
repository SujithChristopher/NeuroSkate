import sys
from PySide6.QtWidgets import *
from PySide6.QtGui import *
from PySide6.QtCore import *

try:
    from gui.gui_support import *
    from gui.toggle_button import *
except ModuleNotFoundError:
    from gui_support import *
    from toggle_button import *


class ViewTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
       

        self.frame_label = QLabel()
        self.frame_label.setGeometry(QRect(20, 150, 1280, 900))
        self.frame_label.setFixedSize( 1280, 900)
        self.frame_label.setStyleSheet("background-color: black; color: white;")
        
        #Select camera drop down
        self.camera_label = QLabel("Select camera")
        self.camera_label.setStyleSheet(get_label_ss())
        
        self.camera_dropdown = QComboBox()
        self.camera_dropdown.setStyleSheet(get_dropdown_ss())

        self.start_button = QPushButton("Start")
        self.start_button.setStyleSheet(get_button_ss())

        self.zero_button = QPushButton("Set-Zero")
        self.zero_button.setStyleSheet(get_button_ss())
        
        self.update_calib_button = QPushButton("Update")
        self.update_calib_button.setStyleSheet(get_button_ss())

        """Coordinate display label"""
        self.x_coord_label = QLabel()
        self.y_coord_label = QLabel()
        self.z_coord_label = QLabel()

        self.x_coord_label.setStyleSheet(get_label_ss())
        self.y_coord_label.setStyleSheet(get_label_ss())
        self.z_coord_label.setStyleSheet(get_label_ss())

        self.x_coord_label.setAlignment(Qt.AlignCenter)
        self.y_coord_label.setAlignment(Qt.AlignCenter)
        self.z_coord_label.setAlignment(Qt.AlignCenter)

        self.ar_toggle_button = AnimatedToggle()
        self.ar_toggle_button.setFixedSize(100, 60)
        self.ar_toggle_button.setChecked(True)

        self.ar_toggle_text = QLabel()
        self.ar_toggle_text.setText("AR/YOLO")
        self.ar_toggle_text.setStyleSheet(get_label_ss())

        self.mp_toggle_button = AnimatedToggle()
        self.mp_toggle_button.setFixedSize(100, 60)
        self.mp_toggle_button.setChecked(False)
        
        self.mp_toggle_text = QLabel()
        self.mp_toggle_text.setText("MediaPipe")
        self.mp_toggle_text.setStyleSheet(get_label_ss())

        self.sub_hbox = QHBoxLayout()
        self.sub_hbox.addWidget(self.ar_toggle_button)
        self.sub_hbox.addWidget(self.ar_toggle_text)
        self.sub_hbox_widget = QWidget()
        self.sub_hbox_widget.setLayout(self.sub_hbox)

        self.sub_hbox_mp = QHBoxLayout()
        self.sub_hbox_mp.addWidget(self.mp_toggle_button)
        self.sub_hbox_mp.addWidget(self.mp_toggle_text)
        self.sub_hbox_widget_mp = QWidget()
        self.sub_hbox_widget_mp.setLayout(self.sub_hbox_mp)

        self.x_label = QLabel()
        self.x_label.setText("X")
        self.x_label.setStyleSheet(get_label_ss())
        self.x_label.setAlignment(Qt.AlignRight)
        self.x_label.setMaximumSize(50, 50)

        self.y_label = QLabel()
        self.y_label.setText("Y")
        self.y_label.setStyleSheet(get_label_ss())
        self.y_label.setAlignment(Qt.AlignRight)
        self.y_label.setMaximumSize(50, 50)

        self.z_label = QLabel()
        self.z_label.setText("Z")
        self.z_label.setStyleSheet(get_label_ss())
        self.z_label.setAlignment(Qt.AlignRight)
        self.z_label.setMaximumSize(50, 50)

        self.sub_hbox_x = QHBoxLayout()
        self.sub_hbox_x.addWidget(self.x_label)
        self.sub_hbox_x.addWidget(self.x_coord_label)
        self.sub_hbox_x_widget = QWidget()
        self.sub_hbox_x_widget.setLayout(self.sub_hbox_x)

        self.sub_hbox_y = QHBoxLayout()
        self.sub_hbox_y.addWidget(self.y_label)
        self.sub_hbox_y.addWidget(self.y_coord_label)
        self.sub_hbox_y_widget = QWidget()
        self.sub_hbox_y_widget.setLayout(self.sub_hbox_y)

        self.sub_hbox_z = QHBoxLayout()
        self.sub_hbox_z.addWidget(self.z_label)
        self.sub_hbox_z.addWidget(self.z_coord_label)
        self.sub_hbox_z_widget = QWidget()
        self.sub_hbox_z_widget.setLayout(self.sub_hbox_z)

        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.camera_label)
        self.vbox.addWidget(self.camera_dropdown)
        self.vbox.addWidget(self.sub_hbox_x_widget)
        self.vbox.addWidget(self.sub_hbox_y_widget)
        self.vbox.addWidget(self.sub_hbox_z_widget)
        self.vbox.addWidget(self.sub_hbox_widget)
        self.vbox.addWidget(self.sub_hbox_widget_mp)
        self.vbox.addWidget(self.start_button)
        self.vbox.addWidget(self.zero_button)
        self.vbox.addWidget(self.update_calib_button)
        self.vbox.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        self.vbox_widget = QWidget()
        self.vbox_widget.setLayout(self.vbox)

        self.hbox = QHBoxLayout()
        self.hbox.addWidget(self.frame_label)
        self.hbox.addWidget(self.vbox_widget)
        self.setLayout(self.hbox)

class CalibrationTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.frame_calib = QLabel()
        self.frame_calib.setGeometry(QRect(20, 150, 1280, 900))
        self.frame_calib.setFixedSize(1280, 900)
        self.frame_calib.setStyleSheet("background-color: black; color: white;")
        
        self.corner_label = QLabel("No of Corners")
        self.corner_label.setStyleSheet(get_label_ss())
        self.corner_disp_label = QLabel("TBD")
        self.corner_disp_label.setStyleSheet(get_label_ss())
        
        self.total_frames_label = QLabel("Total Frames")
        self.total_frames_label.setStyleSheet(get_label_ss())
        self.total_frames_disp_label = QLabel()
  
        self.start_button = QPushButton("Start")
        self.start_button.setStyleSheet(get_button_ss())

        self.stop_button = QPushButton("Stop")
        self.stop_button.setStyleSheet(get_button_ss())
        self.calibrate_button = QPushButton("Calculate")
        self.calibrate_button.setStyleSheet(get_button_ss())
        
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        self.progress_bar.setRange(0, 100)
        
        self.progress_text = QLabel()
        self.progress_text.setText("")
        self.progress_text.setStyleSheet(get_label_ss())
        
        
        vbox = QVBoxLayout()
        vbox.addWidget(self.corner_label)
        vbox.addWidget(self.corner_disp_label)
        vbox.addWidget(self.total_frames_label)
        vbox.addWidget(self.total_frames_disp_label)
        vbox.addWidget(self.start_button)
        vbox.addWidget(self.stop_button)
        vbox.addWidget(self.calibrate_button)  
        vbox.addWidget(self.progress_text)
        vbox.addWidget(self.progress_bar)
        
        vbox_widget = QWidget()
        vbox_widget.setLayout(vbox)   
        
        hbox = QHBoxLayout()
        hbox.addWidget(self.frame_calib)
        hbox.addWidget(vbox_widget)
        
        self.setLayout(hbox)
        
class SettingsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.locate_label = QLabel()
        self.locate_label.setText("Locate settings file")
        self.locate_label.setStyleSheet(get_label_ss())
        
        self.locate_button = QPushButton("Browse")
        
        self.camera_parameters = QLabel()
        self.camera_parameters.setText("Camera parameters")
        self.camera_parameters.setStyleSheet(get_label_ss())
        
        self.camera_resolution = QLabel()
        self.camera_resolution.setText("Camera resolution")
        self.camera_resolution.setStyleSheet(get_label_ss())
        
        self.camera_resolution_dropdown = QComboBox()
        self.camera_resolution_dropdown.addItems(["640x480", "1280x720", "1920x1080"])
        
        
        hbox = QHBoxLayout()
        
        vbox1 = QVBoxLayout()
        vbox1.addWidget(self.locate_label)
        vbox1.addWidget(self.locate_button)
        
        vbox2 = QVBoxLayout()
        
        
        vbox_widget = QWidget()
        vbox_widget.setLayout(vbox1)
        
        hbox.addWidget(vbox_widget)
        hbox.addWidget(self.camera_parameters)
        
        self.setLayout(hbox)
        

        
        
        
        
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Tab Example")
        
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        self.viewTab = ViewTab()
        self.calibrationTab = CalibrationTab()
        
        self.tabs.addTab(self.viewTab, "Realtime view")
        self.tabs.addTab(self.calibrationTab, "Calibration tab")
        self.tabs.addTab(SettingsTab(), "Settings")
        
        
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
