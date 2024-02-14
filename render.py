import os
import sys
import time

import cv2
# import mediapipe as mp
import numpy as np
import pandas as pd
from PySide6 import QtWidgets
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6 import QtCore
from PySide6.QtWidgets import * 
from PySide6 import QtCore

from cv2 import aruco

from scipy.spatial.transform import Rotation as R
import msgpack as mp
import msgpack_numpy as mpn


from ultralytics import YOLO
from py_programs.gui_reference import Ui_MainWindow


import vtk
from vtk.util.numpy_support import vtk_to_numpy
model = YOLO("./models/mip_ar_200e_noise.pt")

import matplotlib.pyplot as plt


# mp_pose = mp.solutions.pose

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
        
inference = "YOLO"

ARUCO_PARAMETERS = aruco.DetectorParameters()
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_MIP_36H12)
detector = aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMETERS)
markerLength = 0.05
markerSeperation = 0.01

board = aruco.GridBoard(
        size= [1,1],
        markerLength=markerLength,
        markerSeparation=markerSeperation,
        dictionary=ARUCO_DICT)

# Create vectors we'll be using for rotations and translations for postures
rotation_vectors, translation_vectors = None, None
axis = np.float32([[-.5, -.5, 0], [-.5, .5, 0], [.5, .5, 0], [.5, -.5, 0],
                   [-.5, -.5, 1], [-.5, .5, 1], [.5, .5, 1], [.5, -.5, 1]])

marker_size = markerLength

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


def find_peaks(data, threshold):
    count = 0
    trigger = True
    for i in data:
        if i > threshold and trigger:
            count += 1
            trigger = False
        if i < threshold and not trigger:
            trigger = True
    return count


class WorkerSignals(QObject):
    finished = Signal()
    error = Signal(tuple)
    result = Signal(QImage)
    progress = Signal(QImage)
    changePixmap = Signal(QImage)


class Worker(QRunnable):

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress

    @Slot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            pass
        else:
            pass
        finally:
            self.signals.finished.emit()  # Done


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    xyRectPos = Signal(int)
    imageStatus = Signal(str)
    progress_callback = Signal(QImage)
    newPixmap = Signal(QImage)

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        
        # QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
        self.setupUi(self)
        self.threadpool = QThreadPool()
        
        print
        
        self.reader = vtk.vtkSTLReader()
        self.reader.SetFileName("models/3D_model.stl")
        self.reader.Update()
        
        # Create a mapper
        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputConnection(self.reader.GetOutputPort())

        # Create an actor
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
        

        self.t_vec = []
        self.r_vec = []

        self.pose = False
        self.data_points = np.nan
        self.data_points = np.array(self.data_points)

        self.disp_pose.stateChanged.connect(self.checkedcpp)
        self.start_button.clicked.connect(self.readImage)
        self.start_program.clicked.connect(self.start_process)
        self.set_orgin.clicked.connect(self.orign_set)

        self.start_p = False
        self.cur_time = 0
        self.time_count = 0
        self.res = []
        self.cl_names = ["time", "X", "Y", "Z"]

        # selecting the webcam
        self.capture_device = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.capture_device.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        
        self.capture_device.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.capture_device.set(cv2.CAP_PROP_FPS, 30)
        
        finish = QAction("Quit", self)
        finish.triggered.connect(self.closeEventExit)
        self.close_button_pressed = False

        self.cam_space = pd.DataFrame(columns=self.cl_names)
        self.cam_space = self.cam_space.astype(np.float32)
        self.tvec_dist = None
        self.initial_value = True
        self.offset = np.array([0, 0, 0])
        self.angle = np.array([0, 0, 0])
        self.init_rmat = np.eye(3)
        self.rmat = np.eye(3)
        self.RMAT_TRIGGER = True
        
        self.FIRST_ITER = False
        self.previous_angle = np.array([0, 0, 0])
        self.current_angle = np.array([0, 0, 0])
        
        # self.vtk_thread()
        
    def vtk_thread(self):
        
        angle = self.angle
        
        if not self.FIRST_ITER:
            self.previous_angle = angle
            self.FIRST_ITER = True
            
        else:
            self.current_angle = angle
            
            angle_diff = self.current_angle - self.previous_angle
            
            transform = vtk.vtkTransform()
           
            M = np.empty((4, 4))
            M[:3, :3] = self.rmat
            M[3, :] = [0, 0, 0, 1]
            
            vtk_matrix = vtk.vtkMatrix4x4()
            for i in range(4):
                for j in range(4):
                    vtk_matrix.SetElement(i, j, M[i, j])
                    
            transform = vtk.vtkTransform()
            transform.SetMatrix(vtk_matrix)
            self.actor.SetUserTransform(transform)
            
            # Create a renderer
            self.renderer = vtk.vtkRenderer()
            self.renderer.AddActor(self.actor)
            self.renderer.SetBackground(0.5, 0.5, 0.5)  # Set background to white

            # Create a render window
            self.renderWindow = vtk.vtkRenderWindow()
            self.renderWindow.AddRenderer(self.renderer)
            self.renderWindow.SetOffScreenRendering(1)  # Enable off-screen rendering
            self.renderWindow.Render()
            
            # Capture the image
            self.windowToImageFilter = vtk.vtkWindowToImageFilter()
            self.windowToImageFilter.SetInput(self.renderWindow)
            self.windowToImageFilter.Update()
            
            vtk_image = self.windowToImageFilter.GetOutput()

            width, height, _ = vtk_image.GetDimensions()
            vtk_array = vtk_image.GetPointData().GetScalars()
            components = vtk_array.GetNumberOfComponents()
            self.rendered_image = vtk_to_numpy(vtk_array).reshape((height, width, components))
            
            # # Convert the NumPy array to a QImage
            height, width, channel = self.rendered_image.shape
            bytes_per_line = channel * width
            qimage = QImage(self.rendered_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            qpixmap = QPixmap.fromImage(qimage)
        
            self.vtk_image.setPixmap(qpixmap)                
                
    # Slot(QPixmap)
    # def vtkSlot(self, image):
    #     self.vtk_image.setPixmap(image)
        
        
    def closeEventExit(self):
        time.sleep(0.5)
        self.close_button_pressed = True
        self.capture_device.release()
        self.close()
        sys.exit()

    def orign_set(self):
        self.initial_value = True
        pass

    def checkedcpp(self, checked):
        if checked:
            self.pose = True
        else:
            self.pose = False

    def readImage(self):
        
        worker = Worker(self.runCam)
        worker.signals.progress.connect(self.set_image)
        worker.signals.finished.connect(self.thread_complete)
        worker.signals.result.connect(self.thread_complete)
        self.threadpool.start(worker)
        
        # worker1 = Worker(self.vtk_thread)
        # worker1.signals.progress.connect(self.vtkSlot)
        # worker1.signals.finished.connect(self.thread_complete)
        # worker1.signals.result.connect(self.thread_complete)
        # self.threadpool.start(worker1)


    def start_process(self):
        self.start_p = True

    def do_nothing(self):
        pass

    def print_progress(self):
        pass

    def thread_complete(self):
        print("thread completed")
        self.capture_device.release()

    @Slot(QImage)
    def set_image(self, image):
        self.video_disp.setPixmap(QPixmap.fromImage(image))
        # self.vtk_thread()
        
    def runCam(self, progress_callback):
        tr = True
        self.peaks = 0

        count = 0
        ln = len(self.cam_space.columns)
        empt_list = []
        for i in range(ln):
            empt_list.append(np.nan)

        time_tr = True

        while True:

            # Capture frame-by-frame
            if self.close_button_pressed:
                break
            
            ret, frame = self.capture_device.read()
            if ret:
                # img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # image = img[yPos * 2:yPos * 2 + yRes, xPos * 2:xPos * 2 + xRes].copy()
                image = img.copy()

                self.colorImage = image

                if self.start_p and time_tr:
                    self.st_time = time.time()

                    time_tr = False

                if self.start_p:
                    self.cur_time = time.time()
                    self.time_count = int(self.cur_time - self.st_time)
                    self.time_float = self.cur_time - self.st_time

                if self.pose:
                    try:
                        
                        if inference == "ARUCO":
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            corners, ids, rejected_image_points = detector.detectMarkers(gray)
                            corners, ids, rejectedpoints,_ = detector.refineDetectedMarkers(image=gray,board=board ,detectedCorners=corners, detectedIds=ids, rejectedCorners=rejected_image_points, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs)
                            
                            if ids is not None and len(ids) > 0:
                                # Estimate the posture per each Aruco marker
                                rotation_vectors, translation_vectors, _objPoints = my_estimatePoseSingleMarkers(corners, marker_points, cameraMatrix, distCoeffs)
                                
                                
                                if self.RMAT_TRIGGER:
                                    self.init_rmat = cv2.Rodrigues(rotation_vectors[0][0])[0]
                                    self.RMAT_TRIGGER = False
                                else:
                                    
                                    self.rmat = cv2.Rodrigues(rotation_vectors[0][0])[0]
                                    _delr = self.rmat
                                    
                                    ang_x = np.rad2deg(np.arctan2(_delr[2][1], _delr[2][2]))
                                    ang_y = np.rad2deg(np.arctan2(-_delr[2][0], np.sqrt(_delr[2][1]**2 + _delr[2][2]**2)))
                                    ang_z = np.rad2deg(np.arctan2(_delr[1][0], _delr[0][0]))
                                    
                                    self.angle = np.array([ang_x, ang_y, ang_z])  
                                
                                self.tvec_dist = translation_vectors[0]
                                for rvec, tvec in zip(rotation_vectors, translation_vectors):
                                    self.colorImage = aruco.drawDetectedMarkers(img, corners=corners, ids=ids)
                                    self.colorImage = cv2.drawFrameAxes(img, cameraMatrix, distCoeffs, rvec, tvec, 0.05)
                                    self.tvec_dist = tvec

                            else:
                                pass
                                                    
                        
                        if inference == "YOLO":
                            yolo_results = model.predict(image, verbose=False)[0]
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
                                
                            # print(corners, ids)
                                
                            if ids is not None and len(ids) > 0:
                                # Estimate the posture per each Aruco marker
                                rotation_vectors, translation_vectors, _objPoints = my_estimatePoseSingleMarkers(corners, marker_points, cameraMatrix, distCoeffs)
                                self.tvec_dist = translation_vectors[0]
                                
                                self.rmat = cv2.Rodrigues(rotation_vectors[0][0])[0]
                                _delr = self.rmat
                                
                                ang_x = np.rad2deg(np.arctan2(_delr[2][1], _delr[2][2]))
                                ang_y = np.rad2deg(np.arctan2(-_delr[2][0], np.sqrt(_delr[2][1]**2 + _delr[2][2]**2)))
                                ang_z = np.rad2deg(np.arctan2(_delr[1][0], _delr[0][0]))
                                
                                self.angle = np.array([ang_x, ang_y, ang_z])  
                            
                                for rvec, tvec in zip(rotation_vectors, translation_vectors):
                                    self.colorImage = cv2.drawFrameAxes(self.colorImage, cameraMatrix, distCoeffs, rvec, tvec, 0.05)
                                    self.tvec_dist = tvec
                            else:
                                pass
                            
                    except:
                        pass
                    
                # print(self.tvec_dist)
                flpimg = self.colorImage
                # flpimg = cv2.resize(flpimg, (1920, 1080))
                h1, w1, ch = flpimg.shape
                bytesPerLine = ch * w1
                convertToQtFormat = QImage(flpimg.data.tobytes(), w1, h1, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(1280, 900, Qt.KeepAspectRatio)

                if self.start_p and tr:
                    img12 = cv2.cvtColor(self.colorImage, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(".//src//color.jpeg", img12)
                    tr = False

                if self.time_count >= 500 and self.start_p:
                    self.run_analysis()
                    self.start_p = False

                if self.start_p:
                    try:
                        # self.label_x.setText(str(f"X m"))
                        if (self.initial_value == True) and (self.tvec_dist is not None):
                            self.offset = self.tvec_dist[0]
                            self.initial_value = False
                        # cordinate
                        self.label_x.setText(str(f"X {round((self.tvec_dist[0][0] - self.offset[0])*100, 2)} cm"))
                        self.label_y.setText(str(f"Y {round((self.tvec_dist[0][1] - self.offset[1])*100, 2)} cm"))
                        self.label_z.setText(str(f"Z {round((self.tvec_dist[0][2] - self.offset[2])*100, 2)} cm"))
                        
                    except:
                        pass

                progress_callback.emit(p)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    # ImageUpdate()
    w.show()
    sys.exit(app.exec())
