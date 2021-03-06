import logging
import picar
import cv2
from time import sleep
import datetime
from picar_utils.rectify_fisheye import rectify_fisheye
from lane_follower_manual import LaneFollowerManual
from road_objects_processor import RoadObjectsProcessor

"""
Car is an instance of the PiCar itself, equipped with
video capture, driving controls, lane detection,
object detection and storage of the recorded videos
"""
class Car(object):
    def __init__(self, video_source=0):
        # initialize car API
        logging.info('Initializing Car...')
        picar.setup()
        
        # select camera and set image dimensions
        logging.debug('Initializing Camera...')
        self.camera = cv2.VideoCapture(video_source)
        self.width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.camera.set(3, self.width)
        self.camera.set(4, self.height)
        
        # disabled for non-motorized camera
        # calibrate servo motors to centre (90)
        # DO NOT exceed -30:30 for offset calibration (offset = rotation)
        #self.pan_servo = picar.Servo.Servo(1)
        #self.pan_servo.offset = 30
        #self.pan_servo.write(90) # point straight ahead
        #self.tilt_servo = picar.Servo.Servo(2)
        #self.tilt_servo.offset = 90
        #self.tilt_servo.write(20) # aim slightly towards the ground for lane following
        
        # initialize API for back wheels (allows forward, backward, stop methods etc.)
        logging.debug('Initializing back wheels...')
        self.back_wheels = picar.back_wheels.Back_Wheels()
        self.back_wheels.speed = 0 # ranges from 0 - 100 (stop - fastest)
        
        # initialize API for front wheels (allows left, right, straight and custom turn etc.)
        logging.debug('Initializing front wheels...')
        self.front_wheels = picar.front_wheels.Front_Wheels()
        #self.front_wheels.turning_offset = -25
        #sleep(0.01) # sleep to ensure adjustments are made
        self.front_wheels.turn(90) # ranges from 45 to 90 to 135 (left, center, right
        
        # initialize lane following and object detection algorithms
        self.lane_follower_manual = LaneFollowerManual(self)
        self.process_road_objects = RoadObjectsProcessor(self, width=self.width, height=self.height)
        
        # setup writer to store recorded video
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID') # write video to XVID format
        time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        self.video_orig = self.create_video_recorder('../data/tmp/car_video_orig%s.avi' % time) # raw
        self.video_lane = self.create_video_recorder('../data/tmp/car_video_lane%s.avi' % time) # lane detection overlay
        self.video_objs = self.create_video_recorder('../data/tmp/car_video_objs%s.avi' % time) # object detection overlay
        
        logging.info('Car Ready!')
        
    # create video writer at 20 fps in XVID format
    def create_video_recorder(self, path):
        return cv2.VideoWriter(path, self.fourcc, 20.0, (self.width, self.height)) # 20 fps
    
    # with statements - https://www.geeksforgeeks.org/with-statement-in-python/
    # entry point of video writer with statement
    # return the class itself, allows class instantiation
    def __enter__(self):
        return self
    
    # exit point of video writer with statement
    def __exit__(self, _type, value, traceback):
        # on exit, show error if occurred
        if traceback is not None:
            logging.error('Exiting with statement with exception %s' % traceback)
        
        # reset hardware
        self.cleanup()
        
    # reset hardware to original state
    def cleanup(self):
        logging.info('Shutting down Car, resetting hardware...')
        self.back_wheels.speed = 0 # reset wheels to centre/stopped
        self.front_wheels.turn(90)
        sleep(0.01) # delay to allow front and back wheel adjustments to register
        self.camera.release() # close camera and writer sessions
        self.video_orig.release()
        self.video_lane.release()
        self.video_objs.release()
        cv2.destroyAllWindows() # close opencv sessions/windows
        
    # drive car at specified speed, perform lane and object detection and record/write resulting images
    def drive(self, speed=0):
        # accelerate back wheels to requested speed (range 0-100)
        logging.info('Car driving at speed %s...' % speed)
        self.back_wheels.speed = speed
        sleep(0.01) # allow wheels to register change
        i = 0
        
        # while camera is capturing frames
        while self.camera.isOpened():            
            # read frame and write to raw video
            ret, img_lane = self.camera.read()
            if not ret:
                logging.error('Camera couldn\'t get image.')
                continue
            
            # # skip first 5 frames while car is initializing
            # i += 1
            # if i < 5:
            #     continue

            # rectify image from fisheye to normal view
            # pick between 0 and 1 (most and least) pixel loss in final image
            # 0 results in full image but lose some pixels
            # 1 results in more pixels but black gaps in image due to warping
            img_lane = rectify_fisheye(img_lane, 0.4)
            show_image('Rectified Image', img_lane)
            
            # create copy of frame for separate lane and object detection
            img_objs = img_lane.copy()
            self.video_orig.write(img_lane)
        
            # run object detection, save frame with objects overlay to video and show image
            img_objs = self.process_road_objects.process_road_objects(img_objs)
            self.video_objs.write(img_objs)
            show_image('Detected Objects', img_objs)
        
            # run lane detection, save frame with lanes overlay to video and show image
            img_lane = self.lane_follower_manual.follow_lane(img_lane)
            self.video_lane.write(img_lane)
            show_image('Lane Lines', img_lane)
            
            # shutdown cleanly and reset hardware
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.cleanup()
                break
    
"""
Helper Functions
    show_image creates a window showing the live video
    main method instantiates the car, sets its speed at 40% and runs above code
    call main method initializes logging
"""

def show_image(title, frame, show=True):
    if show:
        cv2.imshow(title, frame)

# instantiate Car object and enter defined with statement
def main():    
    with Car() as car:
        car.drive(40)
        
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)-5s:%(asctime)s: %(message)s')
    main()