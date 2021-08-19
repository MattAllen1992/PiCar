# import libs
import cv2
import numpy as np

# get images from camera and set height/width
cap = cv2.VideoCapture("car/data/videos/study_test_video.avi") # load local video
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# cap.set(3, width)
# cap.set(4, height)

# continuously capture images until user terminates
while True:
    # get image
    ret, img = cap.read()
    if ret == False:
        print("Cannot read image from camera.")
        break

    # show image
    cv2.imshow('Raw Image', img)

    # break when user terminates
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# close session neatly
cap.release()
cv2.destroyAllWindows