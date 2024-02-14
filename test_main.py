import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PySide6.QtWidgets import *
from PySide6.QtGui import *
from PySide6.QtCore import *
import numpy as np
import cv2
from cv2 import aruco
from ultralytics import YOLO
import os
import msgpack as mp
import msgpack_numpy as mpn
from support import *
from py_programs.py_toggle import AnimatedToggle
import mediapipe as mdpipe

model = YOLO("./models/mip_ar_200e_noise.pt")
mp_pose = mdpipe.solutions.pose
mp_drawing = mdpipe.solutions.drawing_utils 
pose = mp_pose.Pose()

"""open file dialog and path selection"""
calib_pth = ".//calibration//webcam_calibration.msgpack"
print(str(calib_pth))
# Check for camera calibration data
if not os.path.exists(calib_pth):
    print("You need to calibrate the camera you'll be using. See calibration project directory for details.")
    exit()
else:    
    with open(calib_pth, "rb") as f:
        webcam_calib = mp.Unpacker(f, object_hook=mpn.decode)
        _temp = next(webcam_calib)
        cameraMatrix = _temp[0]
        distCoeffs = _temp[1]
    if cameraMatrix is None or distCoeffs is None:
        print(
            "Calibration issue. Remove ./calibration/CameraCalibration.pckl and recalibrate your camera with calibration_ChAruco.py.")
        exit()
        

ARUCO_PARAMETERS = aruco.DetectorParameters()
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_MIP_36H12)
detector = aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMETERS)
marker_size = 0.05
markerSeperation = 0.01

board = aruco.GridBoard(
        size= [1,1],
        markerLength=marker_size,
        markerSeparation=markerSeperation,
        dictionary=ARUCO_DICT)

# Create vectors we'll be using for rotations and translations for postures
rotation_vectors, translation_vectors = None, None
axis = np.float32([[-.5, -.5, 0], [-.5, .5, 0], [.5, .5, 0], [.5, -.5, 0],
                   [-.5, -.5, 1], [-.5, .5, 1], [.5, .5, 1], [.5, -.5, 1]])

marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                            [marker_size / 2, marker_size / 2, 0],
                            [marker_size / 2, -marker_size / 2, 0],
                            [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)



def my_estimatePoseSingleMarkers(corners, marker_points, mtx, distortion):
    trash = []
    rvecs = []
    tvecs = []
    for c in corners:
        nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, flags= cv2.SOLVEPNP_ITERATIVE)
        R = R.reshape((1, 3))
        t = t.reshape((1, 3))
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return np.array(rvecs), np.array(tvecs), trash

class MainWindow(QMainWindow):
    progress_callback = Signal(QPixmap)

    def __init__(self):
        super().__init__()
        self.threadpool = QThreadPool()
        
        self.inference = "YOLO"

        self.setWindowTitle("VTK Widget with Button")
        self.setGeometry(100, 100, 1200, 800)

        # Create VTK widget
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackgroundAlpha(1)
        self.renderer.SetBackground(0.8, 0.8, 0.8)
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
        
        self.capture = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1000 / 60)  # 30 fps
        
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
        self.vtk_text_label.setText("Mirror View")
        self.vtk_text_label.setStyleSheet(get_label_ss())
        self.vtk_text_label.setAlignment(Qt.AlignCenter)
    
        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.vtk_text_label)
        self.vbox.addWidget(self.vtk_widget)
        self.vbox.addWidget(self.x_coord_label)
        self.vbox.addWidget(self.y_coord_label)
        self.vbox.addWidget(self.z_coord_label)
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

        central_widget = QWidget()
        central_widget.setLayout(self.hbox)
        self.setCentralWidget(central_widget)
        self.create_vtk_scene()
        self.setCentralWidget(central_widget)
        
        self.mpose = True
        self.start_p = True
        
        self.vtk_matrix = vtk.vtkMatrix4x4()
        self.vtk_transform = vtk.vtkTransform()
        
        self.default_ids = np.array([12, 88, 89])
        
    def mp_toggle_fun(self):
        if self.mp_toggle_button.isChecked():
            self.mpose = True
        else:
            self.mpose = False
        
    def toggle_fun(self):
        if self.ar_toggle_button.isChecked():
            self.inference = "YOLO"
        else:
            self.inference = "ARUCO"

        
    def update_frame(self):
        
        def set_pose():
            self.x_coord_label.setText(str(round(self.tvec_dist[0]*100, 2)) + " cm")
            self.y_coord_label.setText(str(round(self.tvec_dist[1]*100, 2)) + " cm")
            self.z_coord_label.setText(str(round(self.tvec_dist[2]*100, 2)) + " cm")
            
            M = np.empty((4, 4))
            M[:3, :3] = self.rmat
            M[3, :] = [0, 0, 0, 1]
            for i in range(4):
                for j in range(4):
                    self.vtk_matrix.SetElement(i, j, M[i, j])
                    
        ret, frame = self.capture.read()
        if ret:
            self.colorImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.colorImage.flags.writeable = False
            try:
                if self.inference == "ARUCO":
                    gray = cv2.cvtColor(self.colorImage, cv2.COLOR_BGR2GRAY)
                    corners, ids, rejected_image_points = detector.detectMarkers(gray)
                    corners, ids, rejectedpoints,_ = detector.refineDetectedMarkers(image=gray,board=board ,detectedCorners=corners, detectedIds=ids, rejectedCorners=rejected_image_points, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs)
                    
                    if (ids is not None and len(ids) > 0) and all(item in self.default_ids for item in np.array(ids)):
                        rotation_vectors, translation_vectors, _objPoints = my_estimatePoseSingleMarkers(corners, marker_points, cameraMatrix, distCoeffs)
                        self.rmat = cv2.Rodrigues(rotation_vectors[0][0])[0]
                        self.tvec_dist = translation_vectors[0][0]
                        for rvec, tvec in zip(rotation_vectors, translation_vectors):
                            self.colorImage = aruco.drawDetectedMarkers(self.colorImage, corners=corners, ids=ids)
                            self.colorImage = cv2.drawFrameAxes(self.colorImage, cameraMatrix, distCoeffs, rvec, tvec, 0.05)
                        set_pose()
                    else:
                        pass
                                            
                if self.inference == "YOLO":
                    yolo_results = model.predict(self.colorImage, verbose=False)[0]
                    self.colorImage = yolo_results.plot()
                    modelcorners = []
                    for _keys in yolo_results.keypoints.data:
                        modelcorners.append(_keys[0:4].cpu().numpy())
                    modelcorners = np.array(modelcorners)
                    corners = modelcorners                            
                    if len(yolo_results.boxes.cls.cpu().numpy()) != 0: # if there are any detections else None
                        _idx = yolo_results.boxes.cls.cpu().numpy()
                        ids = []
                        for i in _idx:
                            match i:
                                case 0:
                                    ids.append([12])
                                case 1:
                                    ids.append([88])
                                case 2:
                                    ids.append([89])
                        ids = np.array(ids, dtype=np.int32)
                    else:
                        ids = None
                        
                    if ids is not None and len(ids) > 0:
                        rotation_vectors, translation_vectors, _objPoints = my_estimatePoseSingleMarkers(corners, marker_points, cameraMatrix, distCoeffs)
                        self.tvec_dist = translation_vectors[0][0]
                        self.rmat = cv2.Rodrigues(rotation_vectors[0][0])[0]
                        for rvec, tvec in zip(rotation_vectors, translation_vectors):
                            self.colorImage = cv2.drawFrameAxes(self.colorImage, cameraMatrix, distCoeffs, rvec, tvec, 0.05)
                        
                        set_pose()
                    else:
                        pass                    

            except:
                pass
            
            if self.mpose:
                mp_result = pose.process(self.colorImage)
                
                if mp_result.pose_landmarks:
                    mp_drawing.draw_landmarks(self.colorImage, mp_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                            
            
            h1, w1, ch = self.colorImage.shape
            bytesPerLine = ch * w1
            convertToQtFormat = QImage(self.colorImage.data.tobytes(), w1, h1, bytesPerLine, QImage.Format_RGB888)
            p = convertToQtFormat.scaled(1280, 900, Qt.KeepAspectRatio)
            pixmap = QPixmap.fromImage(p)
            self.frame_label.setPixmap(pixmap)
            
            self.vtk_transform.SetMatrix(self.vtk_matrix)
            self.actor.SetUserTransform(self.vtk_transform)
            self.vtk_widget.GetRenderWindow().Render()


    def create_vtk_scene(self):
        # Create a cone
        reader = vtk.vtkSTLReader()
        reader.SetFileName("models/3D_model.stl")
        # Create a mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(reader.GetOutputPort())

        self.actor = vtk.vtkActor()
        self.actor.AddOrientation(-90, 180, 0)
        self.actor.AddPosition(0, 30, 0)
        self.actor.SetMapper(mapper)
        
        # Add actor to the renderer
        self.renderer.AddActor(self.actor)
        self.renderer.ResetCamera()

        # Render the scene
        self.vtk_widget.GetRenderWindow().Render()


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
