from PySide6.QtWidgets import *
from PySide6.QtGui import *
from PySide6.QtCore import *
from py_programs.gui_support import *
from py_programs.toggle_button import *
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


def create_layout(self):
    self.setWindowTitle("VTK Widget with Button")
    self.setGeometry(100, 100, 1200, 800)
    # Create VTK widget
    self.renderer = vtk.vtkRenderer()
    self.renderer.SetBackgroundAlpha(1)
    self.renderer.SetBackground(0.96, 0.96, 0.96)
    self.vtk_widget = QVTKRenderWindowInteractor(self)
    self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
    self.vtk_widget.setFixedSize(300, 300)

    self.frame_label = QLabel()
    self.frame_label.setGeometry(QRect(20, 150, 1280, 900))
    self.frame_label.setObjectName("frame_label")

    self.start_button = QPushButton("Start")
    self.start_button.setStyleSheet(get_button_ss())

    self.zero_button = QPushButton("Set-Zero")
    self.zero_button.setStyleSheet(get_button_ss())
    self.zero_button.clicked.connect(self.set_zero)

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

    self.vbox = QVBoxLayout()
    self.vbox.addWidget(self.vtk_widget)
    self.vbox.addWidget(self.start_button)

    self.ar_toggle_button = AnimatedToggle()
    self.ar_toggle_button.setFixedSize(100, 60)
    self.ar_toggle_button.setChecked(True)
    self.ar_toggle_button.clicked.connect(self.toggle_fun)

    self.ar_toggle_text = QLabel()
    self.ar_toggle_text.setText("AR/YOLO")
    self.ar_toggle_text.setStyleSheet(get_label_ss())

    self.mp_toggle_button = AnimatedToggle()
    self.mp_toggle_button.setFixedSize(100, 60)
    self.mp_toggle_button.setChecked(False)
    self.mp_toggle_button.clicked.connect(self.mp_toggle_fun)

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

    self.vtk_text_label = QLabel()
    self.vtk_text_label.setText("Mirror 3D Orientation")
    self.vtk_text_label.setStyleSheet(get_label_ss())
    self.vtk_text_label.setAlignment(Qt.AlignCenter)

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
    self.vbox.addWidget(self.vtk_text_label)
    self.vbox.addWidget(self.vtk_widget)
    self.vbox.addWidget(self.sub_hbox_x_widget)
    self.vbox.addWidget(self.sub_hbox_y_widget)
    self.vbox.addWidget(self.sub_hbox_z_widget)
    self.vbox.addWidget(self.sub_hbox_widget)
    self.vbox.addWidget(self.sub_hbox_widget_mp)
    self.vbox.addWidget(self.start_button)
    self.vbox.addWidget(self.zero_button)
    self.vbox.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
    self.vbox_widget = QWidget()
    self.vbox_widget.setLayout(self.vbox)

    self.hbox = QHBoxLayout()
    self.hbox.addWidget(self.frame_label)
    self.hbox.addWidget(self.vbox_widget)

    self.tabs = QTabWidget()
    self.tabs.setLayout(self.hbox)
    self.setCentralWidget(self.tabs)
    self.create_vtk_scene()

    create_calibration_window(self)

def create_calibration_window(self):
    self.tabs.addTab(QWidget(), 'something')
    self.tabs.addTab(QWidget(), 'second')