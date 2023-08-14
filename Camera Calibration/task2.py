import cv2
import numpy as np
from typing import List, Tuple

from cv2 import cvtColor, COLOR_BGR2GRAY, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, \
    findChessboardCorners, cornerSubPix, drawChessboardCorners

def find_corner_img_coord(image: np.ndarray) -> np.ndarray:
    '''
    Returns the 32 checkerboard corners' pixel coordinates.
    '''
    gray = cvtColor(image, COLOR_BGR2GRAY)
    _, corners = findChessboardCorners(gray, (9, 4), flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    refined_corners = cornerSubPix(gray, corners, (5, 5), (-1, -1), (TERM_CRITERIA_MAX_ITER + TERM_CRITERIA_EPS, 100, 0.01))
    
    return np.delete(refined_corners.reshape(36, 2), [4, 13, 22, 31], axis=0)

def find_corner_world_coord() -> np.ndarray:
    '''
    Returns the 32 checkerboard corners' world coordinates.
    '''
    return np.array([[i, j, k] for k in [10, 20, 30, 40] for i in [40, 30, 20, 10, 0] for j in [0, 10, 20, 30, 40]][::5][:32])

def compute_projection_matrix(img_coord: np.ndarray, world_coord: np.ndarray) -> np.ndarray:
    A = []
    for i in range(len(world_coord)):
        w_pt, img_pt = world_coord[i], img_coord[i]
        x, y, z = w_pt
        u, v = img_pt
        A.append([x, y, z, 1, 0, 0, 0, 0, -u*x, -u*y, -u*z, -u])
        A.append([0, 0, 0, 0, x, y, z, 1, -v*x, -v*y, -v*z, -v])

    _, _, Vt = np.linalg.svd(A)
    return Vt[-1].reshape(3, 4)

def find_intrinsic(img_coord: np.ndarray, world_coord: np.ndarray) -> Tuple[float, float, float, float]:
    P = compute_projection_matrix(img_coord, world_coord)
    
    m1, m2, m3 = P[0, :-1], P[1, :-1], P[2, :-1]
    cx, cy = np.dot(m1, m3), np.dot(m2, m3)
    fx, fy = np.sqrt(np.dot(m1, m1) - cx**2), np.sqrt(np.dot(m2, m2) - cy**2)
    
    return fx, fy, cx, cy

def find_extrinsic(img_coord: np.ndarray, world_coord: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    P = compute_projection_matrix(img_coord, world_coord)
    fx, fy, cx, cy = find_intrinsic(img_coord, world_coord)
    M_in = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    M_ext = np.linalg.inv(M_in) @ P
    R, T = M_ext[:, :3], M_ext[:, 3]
    
    return R, T
