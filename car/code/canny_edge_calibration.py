# import libs
import cv2
import numpy as np

# config parameters
threshold_ratio = 3 # ratio of lower to upper canny threshold, 3 is recommended in canny docs
kernel_size = 3 # sobel kernel size (i.e. 3x3 filter for convolution)

# get images from camera
#cap = cv2.VideoCapture("car/data/videos/study_test_video.avi") # load local video
cap = cv2.VideoCapture("car/data/videos/study_test_video2.avi")

# empty callback for trackbar
def empty(x):
    pass

# blur image, perform canny edge detection, show result
def Canny(lower_threshold, img):
    # set thresholds for canny edge detection
    upper_threshold = lower_threshold * threshold_ratio

    # convert image to grayscale (better for sobel edge detection)
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('Grayscale Image', img_gray)

    # blur image (improves edge detection and reduces noise/image info)
    #img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)
    cv2.imshow('Blurred Image', img_blur)

    # extract colour regions in the blue range (colour of lane lines)
    lower_blue = np.array([84, 55, 120]) # upper and lower bounds calibrated using colour_picker.py
    upper_blue = np.array([150, 215, 255])
    img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img_hsv, lower_blue, upper_blue)
    cv2.imshow('Blue Mask', mask)

    # perform canny edge detection using specified sobel kernel size and threshold values
    edges = cv2.Canny(mask, lower_threshold, upper_threshold, kernel_size)
    cv2.imshow('Canny Edge Detection', edges) # show result

# create trackbar to adjust lower threshold for canny edge detection (max threshold = lower * ratio)
cv2.namedWindow('Canny')
cv2.createTrackbar('Canny Lower Threshold Value', 'Canny', 0, 200, empty)

# continuously capture images until user terminates session
while True:
    # get image and perform canny edge detection
    ret, img = cap.read()   
    if ret == True:
        # get user selections for canny lower threshold
        lower_threshold = cv2.getTrackbarPos('Canny Lower Threshold Value', 'Canny')

        # perform canny edge detection using user provided lower threshold
        Canny(lower_threshold, img)
        print(lower_threshold) # show user selected lower threshold

        # wait and close when prompted
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # loop video when it finishes
    else:
        print("Resetting video to start.")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# close session neatly
cap.release()
cv2.destroyAllWindows()