# load libraries
import cv2
from rectify_fisheye import rectify_fisheye

while True:
    # load image and show
    img = cv2.imread('/Users/matthewallen/robots/PiCar/car/data/checkerboard_calibration_images/checkerboard_08252021_190031.jpg')
    cv2.imshow('Raw Image', img)

    # rectify image and show
    # adjust this between 0 and 1 to reduce pixel loss
    # 0 = most loss, 1 = least but includes black spots
    img_t = rectify_fisheye(img, 0.4)
    cv2.imshow('Rectified Image', img_t)

    # wait until user terminates session
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# close and release all
cv2.destroyAllWindows()
