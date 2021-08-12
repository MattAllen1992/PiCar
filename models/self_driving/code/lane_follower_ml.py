import cv2
import numpy as np
import logging
import math
from keras.models import load_model
from lane_follower_manual import LaneFollowerManual

'''
Deep Learning Lane Detection and Navigation
    This module uses a trained model to detect lanes and perform
    navigation to keep the car in the middle of the lanes. All model
    parameters for preprocessing, edge detection, image adjustment,
    lane detection and navigation have been determined during training
    and optimization so that only the model inference is required here.
    
    This model should cope well with the introuction of new road objects,
    adjustments to lane line colours and other changes to the conditions
    which would otherwise have required significant tuning.
'''

# this class uses a lane navigation algorithm to determine the path to follow to remain in the centre
# of the detected lanes and adjust the steering angle of the wheels to achieve that path
# it also draws the detected lanes and determined heading line to follow on top of the raw image
class LaneFollowerML(object):
    # initialize and calibrate car, steering angle and model
    # h5 files are "Hierarchical Data Format" files used to store large amounts of scientific data (e.g. an ML model)
    def __init__(self, car=None, model='/home/pi/work/PiCar/models/lane_navigation/data/sample_model_results/lane_navigation.h5'):
        logging.info('Initializing Deep Learning Lane Follower...')
        self.car = car
        self.curr_steering_angle = 90
        self.model = load_model(model)
    
    # calculate the steering angle required to follow the determined heading path
    def follow_lane(self, frame):
        # display raw image and calculate required steering angle to follow heading path
        show_image('Raw Image',  frame)
        self.curr_steering_angle = self.compute_steering_angle(frame)
        logging.debug('Current steering angle: %d' % self.curr_steering_angle)
        
        # if car exists, turn wheels to follow required steering angle
        # overlay lanes and heading path onto image
        if self.car is not None:
            self.car.front_wheels.turn(self.curr_steering_angle)
        final_frame = display_heading_line(frame, self.curr_steering_angle)
        
        # return image with lanes and heading path overlayed
        return final_frame
    
    # preprocess image and pass to model to determine
    # required steering angle to follow heading path
    def compute_steering_angle(self, frame):
        # preprocess image and extract resulting transformation
        preprocessed = img_preprocess(frame)
        X = np.asarray([preprocessed])
        
        # make inference using model and return result
        new_steering_angle = self.model.predict(X)[0]        
        logging.debug('New steering angle: %d' % new_steering_angle)
        return int(math.ceil(new_steering_angle)) # round to next whole int (avoids 0 angle adjustment)

########## PROCESSING METHODS ##########

# crop, adjust colour space, blur, resize and normalize
# prepares the image for the lane follower model input
def img_preprocess(image):
    # remove top half of image (only need bottom for lane detection)
    height, _, _ = image.shape
    image = image[int(height/2):,:,:]
    
    # Nvidea suggested YUV as the best colour space for lane detection
    # https://docs.nvidia.com/cuda/archive/10.0/npp/group__bgrtoyuv.html
    # https://en.wikipedia.org/wiki/YUV
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV) # Y' (luma/brightness with gamma correction), U (blue chroma), V (red chroma)
    image = cv2.GaussianBlue(image, (3, 3), 0)     # reduce noise/data but retain clear edges
    image = cv2.resize(image, (200, 66))           # recommended dimensions (by Nvidea model)
    image = image / 255                            # normalize pixel values to between 0 and 1
    return image

# calculate heading line to keep the car in the middle of the detected lanes
# overlay the heading line on top of the input image and return result
def display_heading_line(frame, steering_angle, line_color=(0, 0, 255), line_width=5):
    # create mask/template to add heading path to
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape
    
    # convert to radians and calculate start and end point of heading path
    # r = d / 180 * pi (https://www.cuemath.com/geometry/radians-to-degrees)
    steering_angle_radian = steering_angle / 180.0 * math.pi
    
    # start point is image bottom centre
    x1 = int(width / 2)
    y1 = height # shouldn't this be 0? i.e. start from bottom of image?
    
    # end point is calculated with trigonometry
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)
    
    # draw heading path and overlay on top of input image
    cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)
    
    # return image with heading path overlayed
    return heading_image

# display frame if enabled/allowed
def show_image(title, frame, show=False):
    if show:
        cv2.imshow(title, frame)

########## TEST METHODS ##########

# load a test image, run lane detection, display lanes and calculated heading path on raw image
def test_photo(file):
    # read image and perform lane detection
    lane_follower_ml = LaneFollowerML()
    frame = cv2.imread(file)
    img_lanes = lane_follower_ml.follow_lane(frame)
    
    # overlay lanes and heading line on image and disokay
    show_image('Deep Learning Lane Detection', img_lanes, True)
    logging.info('Image=%s, Suggested Steering Angle=%3d' % (file, lane_follower.curr_steering_angle))
    
    # shutdown and close sessions
    cv2.waitKey(0)
    cs2.destroyAllWindows()

# run lane detection and calculation of steering angle/heading line on test video
# compute lane detection using the deep learning model and the manually coded version
def test_video(video):
    # initialize models and load video
    lane_follower_manual = LaneFollowerManual()
    lane_follower_ml = LaneFollowerML()
    camera = cv2.VideoCapture(video + '.avi')
    
    # skip first 3 frames (camera initializing)
    for i in range(3):
        _, frame = camera.read()
    
    # create writer to output avi at 20 fps
    video_format = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter('%s_lane_follower.avi' % video, video_format, 20.0, (320, 240)) # 20 fps
    
    try:
        # counter for storing/logging each frame
        i = 0
        while camera.isOpened():
            # read frame and perform both versions of lane detection on separate copies
            _, frame = camera.read()
            frame_copy = frame.copy()
            logging.info('Frame %s' % i)
            img_lanes_manual = lane_follower_manual.follow_lane(frame)
            img_lanes_ml = lane_follower_ml.follow_lane(frame_copy)
            
            # calculate difference between deep learning and manual models
            # write deep learning result to output video and show both model's lane detection results
            diff = lane_follower_manual.curr_steering_angle - lane_follower_ml.curr_steering_angle
            logging.info('[Desired=%3d] | [Model=%3d] | [Difference=%3d]' % lane_follower_manual.curr_steering_angle, lane_follower_ml.curr_steering_angle, diff)
            writer.write(img_lanes_ml)
            cv2.imshow('Manual Lane Detection', img_lanes_manual)
            cv2.imshow('Deep Learning Lane Detection', img_lanes_ml)
            
            # increment counter and quit when signal received
            i += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # shutdown and close all sessions
    finally:
        camera.release()
        writer.release()
        cv2.destroyAllWindows()

# run tests when script is run
if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    test_video('/home/pi/work/PiCar/models/self_driving/data/lanes/test_lanes_video.avi')
    test_photo('/home/pi/work/PiCar/models/self_driving/data/lanes/test_lanes_image.jpg')