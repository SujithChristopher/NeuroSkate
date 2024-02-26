import os
import toml
from calibration.aruco_parameters import *
from gui.gui_design import *
from PySide6.QtMultimedia import *


_toml_pth = "settings.toml"
if os.path.exists(_toml_pth):
    settings = toml.load(_toml_pth)
    cameraMatrix, dist_coeffs = settings['calibration']["camera_matrix"], settings['calibration']["dist_coeffs"]
    cameraMatrix = np.array(cameraMatrix).reshape(3, 3)
    dist_coeffs = np.array(dist_coeffs)
else:
    print("Calibration file missing/corrupted, defaulting to identity")
    cameraMatrix, dist_coeffs = np.eye(3), np.zeros((1, 5))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.viewTab = ViewTab()
        self.calibrationTab = CalibrationTab()

        self.tabs.addTab(self.viewTab, "View")
        self.tabs.addTab(self.calibrationTab, "Calibration")
        self.tabs.currentChanged.connect(self.tab_changed)
        self.active_tab = 0  # 0 ViewTab, 1 CalibrationTab
        
        for cam in QMediaDevices().videoInputs():
            self.viewTab.camera_dropdown.addItem(cam.description())
            
        self.viewTab.camera_dropdown.setCurrentIndex(0)
        self.viewTab.camera_dropdown.currentIndexChanged.connect(self.select_camera)

        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.start_calibration = False

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start()  # 30 fps 

        self.init_parameters()
        self.buttons_connect()
        
    def select_camera(self):
        self.camera.release()
        self.camera = cv2.VideoCapture(self.viewTab.camera_dropdown.currentIndex())
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
    def init_parameters(self):
        self.ar_detector = get_aruco_detector()
        self.ar_parameters = get_aruco_parameters()
        self.ar_dict = get_aruco_dictionary(aruco.DICT_ARUCO_ORIGINAL)
        self.board = get_board()
        self.marker_size = settings['aruco']['marker_length']
        self.marker_spacing = settings['aruco']['marker_spacing']

    def buttons_connect(self):
        self.calibrationTab.start_button.clicked.connect(self.init_calib_params)
        self.calibrationTab.stop_button.clicked.connect(
            lambda: setattr(self, 'start_calibration', not self.start_calibration))
        self.calibrationTab.calibrate_button.clicked.connect(self.calibrate_camera)

    def init_calib_params(self):
        [self.corners_list, self.ids_list, self.counter] = [], [], []
        self.start_calibration = True
        self.first_calib_frame = True

    def collect_corners(self):
        if self.first_calib_frame:
            self.corners_list = self.corners
            self.ids_list = self.ids
            self.first_calib_frame = False
        else:
            self.corners_list = np.vstack((self.corners_list, self.corners))
            self.ids_list = np.vstack((self.ids_list, self.ids))
        self.counter.append(len(self.ids))
        self.calibrationTab.corner_label.setText(str(len(self.corners_list)))
        self.calibrationTab.total_frames_label.setText(str(sum(self.counter)))

    def calibrate_camera(self):
        mtx2 = np.zeros((3, 3))
        dist2 = np.zeros((1, 8))
        calibration_flags = cv2.CALIB_USE_LU
        term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        rnd = np.random.choice(len(self.counter), 150, replace=False)
        self.counter = np.array(self.counter)
        mask = np.ones(len(self.corners_list), dtype=bool)
        current_index = 0
        previous_index = 0
        counter_len = len(self.counter)

        self.calibrationTab.progress_text.setText("Calibrating...")

        for idx, vals in enumerate(self.counter):
            previous_index = current_index
            current_index += vals
            progress_value = np.uint8(idx * 100 / counter_len - 2)
            self.calibrationTab.progress_bar.setValue(progress_value)
            if idx not in rnd:
                mask[previous_index:current_index] = False

        selected_corners = self.corners_list[mask, ...]
        selected_ids = self.ids_list[mask, ...]
        selected_counter = self.counter[rnd]

        _, mtx1, dist1, _, _ = aruco.calibrateCameraAruco(selected_corners, selected_ids, selected_counter, self.board,
                                                          self.colorImage.shape[:2], mtx2, mtx2, dist2,
                                                          flags=calibration_flags, criteria=term_criteria)

        self.calibrationTab.progress_bar.setValue(100)

        data = toml.load(_toml_pth)
        data['calibration']['camera_matrix'] = mtx1.tolist()
        data['calibration']['dist_coeffs'] = dist1.tolist()
        with open(_toml_pth, 'w') as f:
            toml.dump(data, f)
        self.calibrationTab.progress_text.setText("Calibration done")

    def detect_aruco(self, image):
        corners, ids, rejectedImgPoints = self.ar_detector.detectMarkers(image)
        corners, ids, _, _ = self.ar_detector.refineDetectedMarkers(image, board=self.board,
                                                                    detectedCorners=corners, detectedIds=ids,
                                                                    rejectedCorners=rejectedImgPoints,
                                                                    cameraMatrix=cameraMatrix,
                                                                    distCoeffs=dist_coeffs)
        return corners, ids

    def change_format(self, image):
        h1, w1, ch = image.shape
        bytes_per_line = w1 * ch
        convert_to_qt_format = QImage(image.data.tobytes(), w1, h1, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(convert_to_qt_format)
        return pixmap

    def get_active_tab(self):
        return self.tabs.currentWidget()

    def tab_changed(self, index):
        self.active_tab = index

    @Slot()
    def update_frame(self):
        ret, frame = self.camera.read()
        if ret:
            self.colorImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            corners, ids = self.detect_aruco(self.colorImage)
            self.corners, self.ids = corners, ids
            if ids is not None:
                self.colorImage = aruco.drawDetectedMarkers(self.colorImage, corners, ids)

                if self.start_calibration:
                    self.collect_corners()

            match self.active_tab:
                case 0:
                    self.viewTab.frame_label.setPixmap(self.change_format(self.colorImage))
                case 1:
                    self.calibrationTab.frame_calib.setPixmap(self.change_format(self.colorImage))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
