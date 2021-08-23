# load libraries
import cv2
import numpy as np

'''

Adapted from:
https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0

This method takes a mapping of the distortion a fisheye lens creates
and converts the input image from a fisheye perspective to a rectified,
normal perspective.

With default settings, the image will be rectified but we will lose pixels
from the image due to the dimension conversion between raw and transformed.
By adjusting the values for balance, dim2 and dim3 you can reduce the loss of
pixels. See below post for further details:
https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-part-2-13990f1b157f

'''

# initialize recitification parameters
# these inputs are produced by the 'calibrate_fisheye_lens.py' script
# and are specific to the fisheye camera used in my hardware setup
dim = 1
k = np.array()
d = np.array()

# convert a fisheye lens image to a rectified, normal image
def rectify_fisheye(img, balance=0.0, dim2=None, dim3=None):
    # extract dimensions of raw input
    dim1 = img.shape[:2][::-1]

    # raw image must have same dimensions as the image used for calibration
    assert dim1[0] / dim1[1] == dim[0] / dim[1]

    # dim2 and dim3 are the same dimensionas as the raw image if not otherwise specified
    # specifying alternate dimensions here allows you to adjust the output aspect ratio,
    # potentially helping avoid pixel loss in the final image due to distortion and mapping
    if not dim2:
        dim2 = dim1
    if not dim3:
        dim3 = dim1

    # calculate scale adjustment between raw input and desired output dimensions
    scaled_k = k * dim1[0] / dim[0]
    scaled_k[2][2] = 1.0 # except that k[2][2] is always 1.0

    # calculate new matrix based on desired output dimensions
    new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortedRectify(scaled_k, d, dim2, np.eye(3), balance=balance)

    # create x and y map of raw fisheye camera to rectified, desired output image
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_k, d, np.eye(3), new_k, dim3, cv2.CV_16SC2) # np.eye(3) = 3x3 grid with diagonal 1s

    # transform image from fisheye perspective to rectified, normal view
    img_t = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # return rectified, transformed image
    #cv2.imshow('Transformed', img_t)
    return img_t
