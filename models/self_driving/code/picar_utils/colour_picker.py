# import libs
import cv2
import numpy as np

# get images from camera
#cap = cv2.VideoCapture("car/data/videos/study_test_video.avi") # load local video
cap = cv2.VideoCapture("car/data/videos/study_test_video2.avi")

# callback for trackbar
def empty(a):
    pass

# create trackbar to set HSV values for image colour extraction
# each trackbar requires a callback method to call once it's value is set/changes
cv2.namedWindow("HSV")
cv2.resizeWindow("HSV", 640, 240)
cv2.createTrackbar("HUE Min", "HSV", 0, 179, empty) # hue values range between 0 and 180
cv2.createTrackbar("HUE Max", "HSV", 179, 179, empty)
cv2.createTrackbar("SAT Min", "HSV", 0, 255, empty) # saturation and value are uint8 hence range between 0 and 255
cv2.createTrackbar("SAT Max", "HSV", 255, 255, empty)
cv2.createTrackbar("VAL Min", "HSV", 0, 255, empty)
cv2.createTrackbar("VAL Max", "HSV", 255, 255, empty)

# continuously capture images until user terminates session
while True:
    # get image and create HSV copy of raw image (easier to extract colours for lanes)
    ret, img = cap.read()   
    if ret == True:
        imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # get user selections for HSV values
        h_min = cv2.getTrackbarPos("HUE Min", "HSV")
        h_max = cv2.getTrackbarPos("HUE Max", "HSV")
        s_min = cv2.getTrackbarPos("SAT Min", "HSV")
        s_max = cv2.getTrackbarPos("SAT Max", "HSV")
        v_min = cv2.getTrackbarPos("VAL Min", "HSV")
        v_max = cv2.getTrackbarPos("VAL Max", "HSV")

        # output HSV min/max values for calibration
        print("HSV Min/Max Values:", h_min, h_max, s_min, s_max, v_min, v_max)

        # extract lanes based on their colour using user defined HSV values
        # define lower and upper HSV ranges to extract from image
        # create a mask to extract colour range of interest (e.g. blue lanes)
        # apply mask to raw image to produce extracted lanes/colour range
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(imgHsv, lower, upper)
        result = cv2.bitwise_and(img, img, mask=mask)

        # convert mask to RGB for realistic portrayal
        # stack all 3 images in one horizontal display
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        hStack = np.hstack([img, mask, result])
        cv2.imshow('Horizontal Stacking', hStack)

        # show images separately
        # cv2.imshow('Raw Image', img)
        # cv2.imshow('HSV Image', imgHsv)
        # cv2.imshow('Mask', mask)
        # cv2.imshow('Result', result)

        # get fps of video
        # calculate delay required to play video in realtime
        # delay = 1 / fps (e.g. delay = 1 / 20 fps = 0.05s = 50ms)
        # http://www.learningaboutelectronics.com/Articles/How-to-find-the-frames-per-second-of-a-video-Python-OpenCV.php
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        delay = int(1/fps*1000) # waitkey is in ms, hence *1000 here
        cv2.waitKey(delay)

        # break when user terminates
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # handle camera read errors
    else:
        print("Resetting video to start.")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# close session neatly
cap.release()
cv2.destroyAllWindows