# load libraries
import cv2
assert cv2.__version__[0] >= '3' # fisheye module requires version >= 3
import numpy as np
from pathlib import Path
import glob # regex for file paths

# config parameters for checkerboard calibration
checkerboard = (6, 9)
subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW

# create array to store raw object points
objp = np.zeros((1, checkerboard[0] * checkerboard[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)

# initialize input and flat 2d point arrays
_img_shape = None
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane

# load checkerboard images from different angles
images = glob.glob('/Users/matthewallen/robots/PiCar/car/data/images/*.jpg')

# iterate through images
for fname in images:
    # load image and store dimensions for consistency check
    img = cv2.imread(fname)
    if _img_shape == None:
        _img_shape = img.shape[:2]
    else:
        assert _img_shape == img.shape[:2], "All images must share the same dimensions."

    # convert to grayscale and identify checkerboard corners
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, checkerboard,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    # if ok, store raw and flag 2d points for a map
    if ret == True:
        objpoints.append(objp)
        cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
        imgpoints.append(corners)

# generate k and d arrays, these are crucial inputs for the fisheye conversion
n_ok = len(objpoints)
k = np.zeros((3, 3))
d = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(n_ok)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(n_ok)]
rms, _, _, _, _ = cv2.fisheye.calibrate(objpoints,
                                        imgpoints,
                                        gray.shape[::-1],
                                        k, d, rvecs, tvecs,
                                        calibration_flags,
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
                                    
print('Found ' + str(n_ok) + ' valid images for calibration.')
print("DIM=" + str(_img_shape[::-1]))
print("K=np.array(" + str(k.tolist()) + ")")
print("D=np.array(" + str(d.tolist()) + ")")
