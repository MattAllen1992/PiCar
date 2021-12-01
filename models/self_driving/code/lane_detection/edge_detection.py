import cv2
import numpy as np
from images import show_image

# canny edge detection
def detect_edges_canny(frame):
    # convert image to HSV for ease of colour extraction
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    show_image('HSV', hsv, False)

    # blur image for better edge detection
    # 3, 3 strength of blur in x and y directions
    hsv_blur = cv2.GaussianBlur(hsv, (3, 3), 0) 
    
    # extract colour regions in the blue range (colour of lane lines)
    lower_blue = np.array([84, 55, 120]) # upper and lower bounds calibrated using colour_picker.py
    upper_blue = np.array([150, 215, 255])
    mask = cv2.inRange(hsv_blur, lower_blue, upper_blue)
    show_image('Blue Mask', mask)

    # convert to gray scale for simplicity of edge detection
    # can't go straight from hsv to gray, need rgb middle man
    # mask_rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2RGB)
    # mask_gray = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2GRAY)
    
    # perform canny edge detection
    # gaussian blur to reduce noise, retain/clarify edges
    # calculate gradients (Sobel edge detector kernel) to determine direction and likelihood of edge
    # non-max suppression to clarify and produce thin, crisp edges (suppress non maximum values along suspected edge)
    # hysteresis thresholding to determine final edges (looking at connections, pixel value vs. min and max thresholds)
    lower_threshold = 85 # calibrated using canny_edge_calibration.py
    upper_threshold = lower_threshold * 3 # recommended ratio in canny docs
    edges = cv2.Canny(mask, lower_threshold, upper_threshold)
    
    # return final edges
    return edges