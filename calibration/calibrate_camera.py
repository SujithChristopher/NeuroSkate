import numpy as np
from threading import Thread
from tqdm import tqdm
from cv2 import aruco

import msgpack as mp
import msgpack_numpy as mpn
import os

from aruco_parameters import *

class calibrate_camera:
    def __init__(self, video_source = './/calibration//webcam_color.msgpack', markerLength = 0.05, markerSeperation = 0.01, ARUCO_DICT = get_aruco_dictionary()):
        self.camera_matrix = np.eye(3)
        self.dist_coeffs = np.zeros((1,5))
        self._pth = video_source
        
        self.ARUCO_PARAMETERS = get_aruco_parameters()
        self.ARUCO_DICT = ARUCO_DICT
        self.detector = get_aruco_detector()
        self.board = get_board()
        self.markerLength = markerLength
        self.markerSeperation = markerSeperation
        self.rnd = np.random.choice(self.get_frame_length(), 150, replace=False)
        self.first_frame = True
        
    
    def get_frame_length(self):
        if self.check_pth():
            with open(self._pth, 'rb') as video_file:
                unpacker = mp.Unpacker(video_file, object_hook=mpn.decode)
                return sum(1 for _ in unpacker)
        
    def check_pth(self):
        if not os.path.exists(self._pth):
            print('File does not exist')
            return False
        return True
        
    def detect_markers(self):
        if self.check_pth():
            marker_corners, id_list, counter = [], [], []
            with open(self._pth, 'rb') as video_file:
                unpacker = mp.Unpacker(video_file, object_hook=mpn.decode)
                for idx, _frame in enumerate(unpacker):
                    gray = cv2.cvtColor(_frame, cv2.COLOR_BGR2GRAY)
                    corners, ids, _ = self.detector.detectMarkers(gray)
                    if ids is not None and (idx in self.rnd):
                        if self.first_frame:
                            marker_corners = corners
                            id_list = ids
                            self.first_frame = False
                        else:
                            marker_corners = np.vstack((marker_corners, corners))
                            id_list = np.vstack((id_list, ids))
                        
                        # marker_corners.append(corners)
                        # id_list.append(ids)
                        # counter.append(len(ids))
                return np.array(marker_corners), np.array(id_list), np.array(counter), _frame.shape[:2]
    def calibrate(self):
        markers, id_list, counter, size = self.detect_markers()
        print(markers.shape, id_list.shape, counter.shape, size)
        
        
        # print(markers.size, id_list.size, counter.size, size)
        # mtx2 = np.zeros((3, 3))
        # dist2 = np.zeros((1, 8))
        # rvec2 = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(counter))]
        # tvec2 = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(counter))]
        # calibration_flags = cv2.CALIB_RATIONAL_MODEL
        # term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        
        # ret1, mtx1, dist1, rvecs1, tvecs1 = aruco.calibrateCameraAruco(markers, id_list, counter, self.board, size, mtx2, dist2, flags=calibration_flags, criteria = term_criteria)

        # print(mtx1, dist1)
                
if __name__ == "__main__":
    cc = calibrate_camera()
    cc.calibrate()