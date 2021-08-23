import cv2
from datetime import datetime

# assign camera
cap = cv2.VideoCapture(0)

# show RGB image and capture when user presses 'q'
while(True):
    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    cv2.imshow('RGB Image', rgb)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        now = datetime.now()
        now = now.strftime('%m%d%Y_%H%M%S')
        out = cv2.imwrite('/home/pi/work/PiCar/car/data/images/checkerboard_' + now + '.jpg', frame)
        break

# close and release all
cap.release()
cv2.destroyAllWindows()