from cv2 import aruco
import numpy as np
import cv2
def get_aruco_parameters():
    return aruco.DetectorParameters()

def get_aruco_dictionary(DICT = aruco.DICT_ARUCO_ORIGINAL):
    return aruco.getPredefinedDictionary(DICT)

def get_aruco_detector():
    return aruco.ArucoDetector(get_aruco_dictionary(), get_aruco_parameters())

def get_board(markerLength=0.05, markerSeperation=0.01, ARUCO_DICT=get_aruco_dictionary()):
    return aruco.GridBoard(
        size= [1,1],
        markerLength=markerLength,
        markerSeparation=markerSeperation,
        dictionary=ARUCO_DICT
    )
    
def estimate_pose(corners, m_points, mtx, distortion):
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
    