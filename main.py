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
from py_programs.gui_support import *
from py_programs.toggle_button import AnimatedToggle
from scipy.spatial.transform import Rotation as R

from py_programs.gui import create_layout
import mediapipe as m_pipe

model = YOLO("./models/mip_ar_200e_noise.pt")
mp_pose = m_pipe.solutions.pose
mp_drawing = m_pipe.solutions.drawing_utils
pose = mp_pose.Pose()

calib_pth = ".//calibration//webcam_calibration.msgpack"
if os.path.exists(calib_pth):
    cameraMatrix, distCoeffs = mp.load(open(calib_pth, 'rb'), object_hook=mpn.decode)
else:
    print("Calibration issue, defaulting to identity")
    cameraMatrix, distCoeffs = np.eye(3), np.zeros((1,5))


ARUCO_PARAMETERS = aruco.DetectorParameters()
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_MIP_36H12)
detector = aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMETERS)
marker_size = 0.05
markerSeperation = 0.01

board = aruco.GridBoard(
    size=[1, 1],
    markerLength=marker_size,
    markerSeparation=markerSeperation,
    dictionary=ARUCO_DICT)

# Create vectors we'll be using for rotations and translations for postures
rotation_vectors, translation_vectors = None, None

marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                          [marker_size / 2, marker_size / 2, 0],
                          [marker_size / 2, -marker_size / 2, 0],
                          [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)


def estimate_marker_pose(corners, m_points, mtx, distortion):
    trash = []
    rvecs = []
    tvecs = []
    for c in corners:
        nada, R, t = cv2.solvePnP(m_points, c, mtx, distortion, False, flags=cv2.SOLVEPNP_ITERATIVE)
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

        create_layout(self)

        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.capture.set(cv2.CAP_PROP_FPS, 60)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1000 / 30)  # 30 fps

        self.timer2 = QTimer(self)
        self.timer2.timeout.connect(self.vtk_set)
        self.timer2.start(1000 / 30)  # 30 fps

        self.mpose = False
        self.start_p = True

        self.RMAT_INIT = False
        self.initial_rmat = np.eye(3)

        self.vtk_matrix = vtk.vtkMatrix4x4()
        self.vtk_transform = vtk.vtkTransform()

        self.tvec_12 = np.nan
        self.tvec_88 = np.nan
        self.tvec_89 = np.nan

        self.rvec_12 = np.nan
        self.rvec_88 = np.nan
        self.rvec_89 = np.nan

        self.tv_origin = np.zeros((3, 1))

        self.default_ids = np.array([12, 88, 89])
        self.does_not_exist = []

        self.p_90 = (R.from_euler('zyx', [0, 90, 0], degrees=True)).as_matrix()
        self.n_90 = (R.from_euler('zyx', [0, -90, 0], degrees=True)).as_matrix()
        self.zero_vec = np.zeros(3)
        self.tvec_dist = np.zeros(3)
        self.temp_tvec = np.zeros((3, 1))
        self.rmat = np.eye(3)

    def set_zero(self):
        self.zero_vec = self.tvec_dist
        self.tv_origin = np.array(self.temp_tvec).reshape(3, 1)
        self.renderer.ResetCamera()
        self.RMAT_INIT = False

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

    def set_pose(self):

        self.tvec_dist = self.tvec_dist - self.zero_vec
        self.x_coord_label.setText(str(round(self.tvec_dist[0] * 100, 2)) + " cm")
        self.y_coord_label.setText(str(round(self.tvec_dist[1] * 100, 2)) + " cm")
        self.z_coord_label.setText(str(round(self.tvec_dist[2] * 100, 2)) + " cm")

        M = np.empty((4, 4))
        M[:3, :3] = self.rmat
        M[3, :] = [0, 0, 0, 1]
        for i in range(4):
            for j in range(4):
                self.vtk_matrix.SetElement(i, j, M[i, j])

    def preprocess_ids(self, ids, rotation_vectors, translation_vectors):
        self.does_not_exist = []
        for idx, id in enumerate(ids):
            match np.array(id):
                case 12:
                    self.tvec_12 = translation_vectors[idx][0]
                    self.rvec_12 = rotation_vectors[idx][0]
                case 88:
                    self.tvec_88 = translation_vectors[idx][0]
                    self.rvec_88 = rotation_vectors[idx][0]
                case 89:
                    self.tvec_89 = translation_vectors[idx][0]
                    self.rvec_89 = rotation_vectors[idx][0]
        _ids = np.array(ids)
        if self.default_ids[0] not in _ids:
            self.does_not_exist.append(self.default_ids[0])
        if self.default_ids[1] not in _ids:
            self.does_not_exist.append(self.default_ids[1])
        if self.default_ids[2] not in _ids:
            self.does_not_exist.append(self.default_ids[2])

        for _d in self.does_not_exist:
            match np.array(_d):
                case 12:
                    self.tvec_12 = np.nan
                    self.rvec_12 = np.nan
                case 88:
                    self.tvec_88 = np.nan
                    self.rvec_88 = np.nan
                case 89:
                    self.tvec_89 = np.nan
                    self.rvec_89 = np.nan

    def coordinate_transform(self):
        tv_12 = np.nan
        tv_88 = np.nan
        tv_89 = np.nan

        rm_12 = np.eye(3)
        rm_88 = np.eye(3)
        rm_89 = np.eye(3)

        id_12_offset = np.array([-0.05, 0.03, -0.055]).reshape(3, 1)
        id_88_offset = np.array([0.00, 0.03, -0.11]).reshape(3, 1)
        id_89_offset = np.array([0.05, 0.03, -0.055]).reshape(3, 1)

        if self.tvec_12 is not np.nan:
            tv_12 = np.array(self.tvec_12).reshape(3, 1)
            rm_12 = cv2.Rodrigues(self.rvec_12)[0]
        if self.tvec_88 is not np.nan:
            tv_88 = np.array(self.tvec_88).reshape(3, 1)
            rm_88 = cv2.Rodrigues(self.rvec_88)[0]
        if self.tvec_89 is not np.nan:
            tv_89 = np.array(self.tvec_89).reshape(3, 1)
            rm_89 = cv2.Rodrigues(self.rvec_89)[0]

        pa_c_12 = rm_12 @ id_12_offset + tv_12
        pa_b_c_12 = self.initial_rmat.T @ (pa_c_12 - self.tv_origin)

        pa_c_88 = rm_88 @ id_88_offset + tv_88
        pa_b_c_88 = self.initial_rmat.T @ (pa_c_88 - self.tv_origin)

        pa_c_89 = rm_89 @ id_89_offset + tv_89
        pa_b_c_89 = self.initial_rmat.T @ (pa_c_89 - self.tv_origin)

        self.tvec_dist = np.nanmedian(np.array([pa_b_c_12, pa_b_c_88, pa_b_c_89]), axis=0)
        self.tvec_dist = self.tvec_dist.T[0]

    def update_frame(self):
        ret, frame = self.capture.read()
        if ret:
            self.colorImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # self.inference = "None"
            try:
                if self.inference == "ARUCO":
                    gray = cv2.cvtColor(self.colorImage, cv2.COLOR_BGR2GRAY)
                    corners, ids, rejected_image_points = detector.detectMarkers(gray)
                    corners, ids, _, _ = detector.refineDetectedMarkers(image=gray, board=board,
                                                                        detectedCorners=corners, detectedIds=ids,
                                                                        rejectedCorners=rejected_image_points,
                                                                        cameraMatrix=cameraMatrix,
                                                                        distCoeffs=distCoeffs)
                    if (ids is not None and len(ids) > 0) and all(item in self.default_ids for item in np.array(ids)):
                        rotation_vectors, translation_vectors, _ = estimate_marker_pose(corners, marker_points,
                                                                                        cameraMatrix,
                                                                                        distCoeffs)
                        self.preprocess_ids(ids, rotation_vectors, translation_vectors)

                        for rvec, tvec in zip(rotation_vectors, translation_vectors):
                            self.colorImage = aruco.drawDetectedMarkers(self.colorImage, corners=corners, ids=ids)
                            self.colorImage = cv2.drawFrameAxes(self.colorImage, cameraMatrix, distCoeffs, rvec, tvec,
                                                                0.05)
                        self.coordinate_transform()
                        if self.rvec_12 is not np.nan:
                            _r = cv2.Rodrigues(self.rvec_12)[0]
                            self.rmat = _r @ self.n_90
                        if self.rvec_88 is not np.nan:
                            self.rmat = cv2.Rodrigues(self.rvec_88)[0]
                        if self.rvec_89 is not np.nan:
                            _r = cv2.Rodrigues(self.rvec_89)[0]
                            self.rmat = _r @ self.p_90

                        self.temp_tvec = translation_vectors[0][0]

                        if not self.RMAT_INIT:
                            self.initial_rmat = self.rmat
                            self.RMAT_INIT = True

                    else:
                        self.zero_vec = np.zeros(3)

                if self.inference == "YOLO":

                    yolo_results = model.predict(self.colorImage, verbose=False, imgsz=640)[0]
                    self.colorImage = yolo_results.plot()
                    modelcorners = []
                    for _keys in yolo_results.keypoints.data:
                        modelcorners.append(_keys[0:4].cpu().numpy())
                    modelcorners = np.array(modelcorners)
                    corners = modelcorners
                    if len(yolo_results.boxes.cls.cpu().numpy()) != 0:  # if there are any detections else None
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
                        rotation_vectors, translation_vectors, _ = estimate_marker_pose(corners, marker_points,
                                                                                        cameraMatrix,
                                                                                        distCoeffs)
                        self.preprocess_ids(ids, rotation_vectors, translation_vectors)
                        for rvec, tvec in zip(rotation_vectors, translation_vectors):
                            self.colorImage = cv2.drawFrameAxes(self.colorImage, cameraMatrix, distCoeffs, rvec, tvec,
                                                                0.05)
                        self.coordinate_transform()

                        if self.rvec_12 is not np.nan:
                            _r = cv2.Rodrigues(self.rvec_12)[0]
                            self.rmat = _r @ self.n_90
                        if self.rvec_88 is not np.nan:
                            self.rmat = cv2.Rodrigues(self.rvec_88)[0]
                        if self.rvec_89 is not np.nan:
                            _r = cv2.Rodrigues(self.rvec_89)[0]
                            self.rmat = _r @ self.p_90
                        if not self.RMAT_INIT:
                            self.initial_rmat = self.rmat
                            self.RMAT_INIT = True
                    else:
                        self.zero_vec = np.zeros(3)

            except:
                pass

            if self.mpose:
                mp_result = pose.process(self.colorImage)

                if mp_result.pose_landmarks:
                    mp_drawing.draw_landmarks(self.colorImage, mp_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            h1, w1, ch = self.colorImage.shape
            bytesPerLine = ch * w1
            convertToQtFormat = QImage(self.colorImage.data.tobytes(), w1, h1, bytesPerLine, QImage.Format_RGB888)
            # p = convertToQtFormat.scaled(1280, 900, Qt.KeepAspectRatio)
            p = convertToQtFormat
            pixmap = QPixmap.fromImage(p)
            self.frame_label.setPixmap(pixmap)

    def vtk_set(self):
        self.set_pose()
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
        self.actor.GetProperty().SetColor(0.529, 0.808, 0.922)

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
