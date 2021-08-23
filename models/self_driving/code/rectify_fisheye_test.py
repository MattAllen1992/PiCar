# load libraries
import cv2
from rectify_fisheye import rectify_fisheye

while True:
    # load image and show
    img = cv2.imread('/Users/matthewallen/robots/PiCar/car/data/images/')
    cv2.imshow('Raw Image', img)

    # rectify image and show
    img_t = rectify_fisheye(img)
    cv2.imshow('Rectified Image', img_t)

    # wait until user terminates session
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# close and release all
cv2.destroyAllWindows()
